# BN-VAE
This is the pytorch implementation of the [paper](https://arxiv.org/abs/2004.12585):

A Batch Normalized Inference Network Keeps the KL Vanishing Away


## Acknowledgements

Thanks for sharing the code to public! A large portion of this repo is borrowed from https://github.com/jxhe/vae-lagging-encoder and 
https://github.com/bohanli/vae-pretraining-encoder

## Requirements

* Python >= 3.6
* PyTorch >= 1.0
* pip install editdistance

## Data

Datasets used in the paper can be downloaded with:

```
python prepare_data.py
python prepare_data_yelp_yahoo.py
```

## Usage

Train a AE if you want

```
python text_beta.py \
    --dataset yahoo \
    --beta 0 \
    --lr 0.5
```

Train VAE with different algorithms, check parameters "--fb", if you want to use bn-vae, remember set args.gamma.
An example of training:
```
python text_anneal_fb.py \
    --dataset yahoo \
    --kl_start 0 \
    --warm_up 10 \
    --target_kl 8 \
    --fb 0 \
    --lr 0.5
    --gamma 0.5
```
