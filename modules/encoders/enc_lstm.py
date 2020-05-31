from itertools import chain
import math
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .gaussian_encoder import GaussianEncoderBase
from .encoder_vmf import VMFEncoderBase
from ..utils import log_sum_exp

from sympy import *

class VMFLSTMEncoder(VMFEncoderBase):
    """Gaussian LSTM Encoder with constant-length input"""
    def __init__(self, args, vocab_size, model_init, emb_init):
        super(VMFLSTMEncoder, self).__init__()
        self.ni = args.ni
        self.nh = args.enc_nh
        self.nz = args.nz
        self.args = args

        self.embed = nn.Embedding(vocab_size, args.ni)

        self.lstm = nn.LSTM(input_size=args.ni,
                            hidden_size=args.enc_nh,
                            num_layers=1,
                            batch_first=True,
                            dropout=0)
        
        self.linear = nn.Linear(args.enc_nh, 2 * args.nz, bias=False)

        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init, reset=False):
        # for name, param in self.lstm.named_parameters():
        #     # self.initializer(param)
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)
        #         # model_init(param)
        #     elif 'weight' in name:
        #         model_init(param)

        # model_init(self.linear.weight)
        # emb_init(self.embed.weight)
            #if self.args.gamma > -1:
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self, input):
        """
        Args:
            x: (batch_size, seq_len)

        Returns: Tensor1, Tensor2
            Tensor1: the mean tensor, shape (batch, nz)
            Tensor2: the logvar tensor, shape (batch, nz)
        """

        # (batch_size, seq_len-1, args.ni)
        word_embed = self.embed(input)

        _, (last_state, last_cell) = self.lstm(word_embed)

        mean, logvar = self.linear(last_state).chunk(2, -1)
        if self.args.fix_var > 0:
            logvar = mean.new_tensor([[[math.log(self.args.fix_var)]]]).expand_as(mean)
            
        return mean.squeeze(0), logvar.squeeze(0)


class DeltaGaussianLSTMEncoder(GaussianEncoderBase):
    """Gaussian LSTM Encoder with constant-length input"""
    def __init__(self, args, vocab_size, model_init, emb_init):
        super(DeltaGaussianLSTMEncoder, self).__init__()
        self.ni = args.ni
        self.nh = args.enc_nh
        self.nz = args.nz
        self.args = args

        self.delta = args.delta
        print(self.delta)
        # EQ 4 in the delta-vae paper
        x = Symbol('x')
        l_var, u_var = solve([ln(x)-x + 2*self.delta + 1],[x])
        l_std, u_std = sqrt(l_var[0]), sqrt(u_var[0])
        
        self.l = torch.tensor(float(l_std), device=args.device)
        self.u = torch.tensor(float(u_std), device=args.device)

        self.embed = nn.Embedding(vocab_size, args.ni)

        self.lstm = nn.LSTM(input_size=args.ni,
                            hidden_size=args.enc_nh,
                            num_layers=1,
                            batch_first=True,
                            dropout=0)
        
        self.linear = nn.Linear(args.enc_nh, 2 * args.nz, bias=False)

        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        # for name, param in self.lstm.named_parameters():
        #     # self.initializer(param)
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)
        #         # model_init(param)
        #     elif 'weight' in name:
        #         model_init(param)

        # model_init(self.linear.weight)
        # emb_init(self.embed.weight)
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self, input):
        """
        Args:
            x: (batch_size, seq_len)

        Returns: Tensor1, Tensor2
            Tensor1: the mean tensor, shape (batch, nz)
            Tensor2: the logvar tensor, shape (batch, nz)
        """

        # (batch_size, seq_len-1, args.ni)
        word_embed = self.embed(input)

        _, (last_state, last_cell) = self.lstm(word_embed)

        mean, logvar = self.linear(last_state).chunk(2, -1)
        mean = mean.squeeze(0)
        logvar = logvar.squeeze(0)

        std = self.l + (self.u - self.l) * (1. / torch.clamp((1. + torch.exp(-logvar)),1., 50. ))
        logvar = torch.log(std**2)
        mean = torch.sqrt(2 * self.delta + 1 + logvar - torch.exp(logvar) + torch.max(torch.tensor(0.0, device=self.args.device), mean) + 1e-6 )
        #mean = torch.sqrt(torch.max(2 * self.delta + 1 + logvar - torch.exp(logvar), torch.tensor(0.0, device=self.args.device)) + torch.max(torch.tensor(0.0, device=self.args.device), mean))

        assert(not torch.isnan(mean).sum())
        assert(not torch.isnan(logvar).sum())
        return mean, logvar

class GaussianLSTMEncoder(GaussianEncoderBase):
    """Gaussian LSTM Encoder with constant-length input"""
    def __init__(self, args, vocab_size, model_init, emb_init):
        super(GaussianLSTMEncoder, self).__init__()
        self.ni = args.ni
        self.nh = args.enc_nh
        self.nz = args.nz
        self.args = args

        self.embed = nn.Embedding(vocab_size, args.ni)

        self.lstm = nn.LSTM(input_size=args.ni,
                            hidden_size=args.enc_nh,
                            num_layers=1,
                            batch_first=True,
                            dropout=0)
        
        self.linear = nn.Linear(args.enc_nh, 2 * args.nz, bias=False)
        self.mu_bn = nn.BatchNorm1d(args.nz)
        self.mu_bn.weight.requires_grad = False

        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init, reset=False):
        # for name, param in self.lstm.named_parameters():
        #     # self.initializer(param)
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)
        #         # model_init(param)
        #     elif 'weight' in name:
        #         model_init(param)

        # model_init(self.linear.weight)
        # emb_init(self.embed.weight)
        if not reset:
            #if self.args.gamma > -1:
                #for param in self.parameters():
                #    model_init(param)
                #emb_init(self.embed.weight)
            self.mu_bn.weight.fill_(self.args.gamma)
        else:
            print('reset bn!')
            self.mu_bn.weight.fill_(self.args.gamma)
            nn.init.constant_(self.mu_bn.bias, 0.0)

    def forward(self, input):
        """
        Args:
            x: (batch_size, seq_len)

        Returns: Tensor1, Tensor2
            Tensor1: the mean tensor, shape (batch, nz)
            Tensor2: the logvar tensor, shape (batch, nz)
        """

        # (batch_size, seq_len-1, args.ni)
        word_embed = self.embed(input)

        _, (last_state, last_cell) = self.lstm(word_embed)

        mean, logvar = self.linear(last_state).chunk(2, -1)
        if self.args.gamma > 0:
            mean = self.mu_bn(mean.squeeze(0))
        else:
            mean = mean.squeeze(0)
        # fix variance as a pre-defined value
        if self.args.fix_var > 0:
            logvar = mean.new_tensor([[[math.log(self.args.fix_var)]]]).expand_as(mean)
            
        return mean, logvar.squeeze(0)



class VarLSTMEncoder(GaussianLSTMEncoder):
    """Gaussian LSTM Encoder with variable-length input"""
    def __init__(self, args, vocab_size, model_init, emb_init):
        super(VarLSTMEncoder, self).__init__(args, vocab_size, model_init, emb_init)


    def forward(self, input):
        """
        Args:
            input: tuple which contains x and sents_len
                    x: (batch_size, seq_len)
                    sents_len: long tensor of sentence lengths

        Returns: Tensor1, Tensor2
            Tensor1: the mean tensor, shape (batch, nz)
            Tensor2: the logvar tensor, shape (batch, nz)
        """

        input, sents_len = input
        # (batch_size, seq_len, args.ni)
        word_embed = self.embed(input)

        packed_embed = pack_padded_sequence(word_embed, sents_len.tolist(), batch_first=True)

        _, (last_state, last_cell) = self.lstm(packed_embed)

        mean, logvar = self.linear(last_state).chunk(2, -1)

        return mean.squeeze(0), logvar.squeeze(0)

    def encode(self, input, nsamples):
        """perform the encoding and compute the KL term
        Args:
            input: tuple which contains x and sents_len

        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]

        """

        # (batch_size, nz)
        mu, logvar = self.forward(input)

        # (batch, nsamples, nz)
        z = self.reparameterize(mu, logvar, nsamples)

        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        return z, KL



