#!/usr/bin/env bash
set -e

# Resume or load GAN-AE training from an existing checkpoint.

CKPT_PATH="runs/gan_ae_cfpd/ckpt_epoch050.pt"

python Gan_autoencoder_Cfp.py \
  --ckpt "$CKPT_PATH"
