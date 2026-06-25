#!/usr/bin/env bash
set -e

# Decode images from a saved HDF5 latent file.
# Update CKPT_PATH to the checkpoint used for the original model.

H5_PATH="runs/flowers_latent.h5"
CKPT_PATH="runs/gan_ae_flowers/ckpt_epoch050.pt"

python Gan_autoencoder_Cfp.py \
  --decode_h5 "$H5_PATH" \
  --ckpt "$CKPT_PATH" \
  --decode_n 5
