import math
import torch
import torch.nn as nn
import numpy as np
from scipy import special as sp

from ..utils import log_sum_exp

use_gpu = True
if torch.cuda.is_available() and use_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def GVar(x):
    return x.to(device)  # pytorch 0.4.1

class vMF(torch.nn.Module):
    def __init__(self, hid_dim, lat_dim, kappa=80):
        """
        von Mises-Fisher distribution class with batch support and manual tuning kappa value.
        Implementation follows description of my paper and Guu's.
        """

        super().__init__()
        self.hid_dim = hid_dim
        self.lat_dim = lat_dim
        self.kappa = kappa
        # self.func_kappa = torch.nn.Linear(hid_dim, lat_dim)
        #self.func_mu = torch.nn.Linear(hid_dim, lat_dim)

        self.kld = GVar(torch.from_numpy(vMF._vmf_kld(kappa, lat_dim)).float())
        print('KLD: {}'.format(self.kld.data[0]))

    def estimate_param(self, latent_code):
        ret_dict = {}
        ret_dict['kappa'] = self.kappa

        # Only compute mu, use mu/mu_norm as mu,
        #  use 1 as norm, use diff(mu_norm, 1) as redundant_norm
        mu = latent_code

        norm = torch.norm(mu, 2, 1, keepdim=True)
        mu_norm_sq_diff_from_one = torch.pow(torch.add(norm, -1), 2)
        redundant_norm = torch.sum(mu_norm_sq_diff_from_one, dim=1, keepdim=True)
        ret_dict['norm'] = torch.ones_like(mu)
        ret_dict['redundant_norm'] = redundant_norm

        mu = mu / torch.norm(mu, p=2, dim=1, keepdim=True)
        ret_dict['mu'] = mu

        return ret_dict

    def compute_KLD(self, tup, batch_sz):
        return self.kld.expand(batch_sz)

    @staticmethod
    def _vmf_kld(k, d):
        tmp = (k * ((sp.iv(d / 2.0 + 1.0, k) + sp.iv(d / 2.0, k) * d / (2.0 * k)) / sp.iv(d / 2.0, k) - d / (2.0 * k)) \
               + d * np.log(k) / 2.0 - np.log(sp.iv(d / 2.0, k)) \
               - sp.loggamma(d / 2 + 1) - d * np.log(2) / 2).real
        if tmp != tmp:
            exit()
        return np.array([tmp])

    @staticmethod
    def _vmf_kld_davidson(k, d):
        """
        This should be the correct KLD.
        Empirically we find that _vmf_kld (as in the Guu paper) only deviates a little (<2%) in most cases we use.
        """
        tmp = k * sp.iv(d / 2, k) / sp.iv(d / 2 - 1, k) + (d / 2 - 1) * torch.log(k) - torch.log(
            sp.iv(d / 2 - 1, k)) + np.log(np.pi) * d / 2 + np.log(2) - sp.loggamma(d / 2).real - (d / 2) * np.log(
            2 * np.pi)
        if tmp != tmp:
            exit()
        return np.array([tmp])

    def build_bow_rep(self, lat_code, n_sample):
        batch_sz = lat_code.size()[0]
        tup = self.estimate_param(latent_code=lat_code)
        mu = tup['mu']
        norm = tup['norm']
        kappa = tup['kappa']

        kld = self.compute_KLD(tup, batch_sz)
        vecs = []
        if n_sample == 1:
            return tup, kld, self.sample_cell(mu, norm, kappa)
        for n in range(n_sample):
            sample = self.sample_cell(mu, norm, kappa)
            vecs.append(sample)
        vecs = torch.cat(vecs, dim=0)
        return tup, kld, vecs

    def sample_cell(self, mu, norm, kappa):
        batch_sz, lat_dim = mu.size()
        # mu = GVar(mu)
        mu = mu / torch.norm(mu, p=2, dim=1, keepdim=True)
        w = self._sample_weight_batch(kappa, lat_dim, batch_sz)
        w = w.unsqueeze(1)

        # batch version
        w_var = GVar(w * torch.ones(batch_sz, lat_dim))
        v = self._sample_ortho_batch(mu, lat_dim)
        scale_factr = torch.sqrt(
            GVar(torch.ones(batch_sz, lat_dim)) - torch.pow(w_var, 2))
        orth_term = v * scale_factr
        muscale = mu * w_var
        sampled_vec = orth_term + muscale

        return sampled_vec

    def _sample_weight_batch(self, kappa, dim, batch_sz=1):
        result = torch.FloatTensor((batch_sz))
        for b in range(batch_sz):
            result[b] = self._sample_weight(kappa, dim)
        return result

    def _sample_weight(self, kappa, dim):
        """Rejection sampling scheme for sampling distance from center on
        surface of the sphere.
        """
        dim = dim - 1  # since S^{n-1}
        b = dim / (np.sqrt(4. * kappa ** 2 + dim ** 2) + 2 * kappa)  # b= 1/(sqrt(4.* kdiv**2 + 1) + 2 * kdiv)
        x = (1. - b) / (1. + b)
        c = kappa * x + dim * np.log(1 - x ** 2)  # dim * (kdiv *x + np.log(1-x**2))

        while True:
            z = np.random.beta(dim / 2., dim / 2.)  # concentrates towards 0.5 as d-> inf
            w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
            u = np.random.uniform(low=0, high=1)
            if kappa * w + dim * np.log(1. - x * w) - c >= np.log(
                    u):  # thresh is dim *(kdiv * (w-x) + log(1-x*w) -log(1-x**2))
                return w

    def _sample_ortho_batch(self, mu, dim):
        """
        :param mu: Variable, [batch size, latent dim]
        :param dim: scala. =latent dim
        :return:
        """
        _batch_sz, _lat_dim = mu.size()
        assert _lat_dim == dim
        squeezed_mu = mu.unsqueeze(1)

        v = GVar(torch.randn(_batch_sz, dim, 1))  # TODO random

        # v = GVar(torch.linspace(-1, 1, steps=dim))
        # v = v.expand(_batch_sz, dim).unsqueeze(2)

        rescale_val = torch.bmm(squeezed_mu, v).squeeze(2)
        proj_mu_v = mu * rescale_val
        ortho = v.squeeze() - proj_mu_v
        ortho_norm = torch.norm(ortho, p=2, dim=1, keepdim=True)
        y = ortho / ortho_norm
        return y

    def _sample_orthonormal_to(self, mu, dim):
        """Sample point on sphere orthogonal to mu.
        """
        v = GVar(torch.randn(dim))  # TODO random

        # v = GVar(torch.linspace(-1,1,steps=dim))

        rescale_value = mu.dot(v) / mu.norm()
        proj_mu_v = mu * rescale_value.expand(dim)
        ortho = v - proj_mu_v
        ortho_norm = torch.norm(ortho)
        return ortho / ortho_norm.expand_as(ortho)




class VMFEncoderBase(nn.Module):
    """docstring for EncoderBase"""
    def __init__(self):
        super(VMFEncoderBase, self).__init__()
        self.dist = vMF(1024, 32, kappa=13)

    def forward(self, x):
        """
        Args:
            x: (batch_size, *)

        Returns: Tensor1, Tensor2
            Tensor1: the mean tensor, shape (batch, nz)
            Tensor2: the logvar tensor, shape (batch, nz)
        """

        raise NotImplementedError

    def encode_stats(self, x):

        return self.forward(x)

    def sample(self, input, nsamples):
        """sampling from the encoder
        Returns: Tensor1, Tuple
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tuple: contains the tensor mu [batch, nz] and
                logvar[batch, nz]
        """

        # (batch_size, nz)
        mu, logvar = self.forward(input)

        # (batch, nsamples, nz)
        #z = self.reparameterize(mu, logvar, nsamples)
        tup, kld, z = self.dist.build_bow_rep(mu, 1)
        z = z.unsqueeze(1)

        return z, (mu, mu)

    def encode(self, input, nsamples):
        """perform the encoding and compute the KL term

        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]

        """

        # (batch_size, nz)
        mu, logvar = self.forward(input)

        # (batch, nsamples, nz)
        tup, kld, z = self.dist.build_bow_rep(mu, nsamples)
        z=z.unsqueeze(1)
        return z, kld

    def reparameterize(self, mu, logvar, nsamples=1):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)

            logvar: Tensor
                logvar of gaussian distibution with shape (batch, nz)

        Returns: Tensor
            Sampled z with shape (batch, nsamples, nz)
        """
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)

        eps = torch.zeros_like(std_expd).normal_()

        return mu_expd + torch.mul(eps, std_expd)

    def sample_from_inference(self, x, nsamples=1):
        """this function samples from q(Z | X), for the Gaussian family we use
        mode locations as samples

        Returns: Tensor
            Tensor: the mode locations, shape [batch_size, nsamples, nz]

        """
        # (batch_size, nz)
        mu, logvar,_,_ = self.forward(x)
        # std = logvar.mul(0.5).exp()

        # batch_size = mu.size(0)
        # zrange = zrange.unsqueeze(1).expand(zrange.size(0), batch_size, self.nz)

        # infer_dist = torch.distributions.normal.Normal(mu, std)

        # # (batch_size, k^2)
        # log_prob = infer_dist.log_prob(zrange).sum(dim=-1).permute(1, 0)


        # # (K^2)
        # log_prob = log_prob.sum(dim=0)
        batch_size, nz = mu.size()

        return mu.unsqueeze(1).expand(batch_size, nsamples, nz)

    def eval_inference_dist(self, x, z, param=None):
        """this function computes log q(z | x)
        Args:
            z: tensor
                different z points that will be evaluated, with
                shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log q(z|x) with shape [batch, nsamples]
        """

        nz = z.size(2)

        if not param:
            mu, logvar,_,_ = self.forward(x)
        else:
            mu, logvar = param

        # (batch_size, 1, nz)
        mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
        var = logvar.exp()

        # (batch_size, nsamples, nz)
        dev = z - mu

        # (batch_size, nsamples)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        return log_density

    def calc_mi(self, x):
        """Approximate the mutual information between x and z
        I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))

        Returns: Float

        """

        # [x_batch, nz]
        mu, logvar = self.forward(x)

        x_batch, nz = mu.size()

        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
        neg_entropy = (-0.5 * nz * math.log(2 * math.pi)- 0.5 * (1 + logvar).sum(-1)).mean()

        # [z_batch, 1, nz]
        z_samples = self.reparameterize(mu, logvar, 1)

        # [1, x_batch, nz]
        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar.exp()

        # (z_batch, x_batch, nz)
        dev = z_samples - mu

        # (z_batch, x_batch)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz = log_sum_exp(log_density, dim=1) - math.log(x_batch)

        return (neg_entropy - log_qz.mean(-1)).item()
