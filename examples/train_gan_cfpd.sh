#!/usr/bin/env bash
set -e

# Train the GAN-guided Autoencoder on a CFPD-style face dataset.
# Update DATA_DIR to match your local CFPD dataset location.

DATA_DIR="data/cfpd/Data/Images"
OUTPUT_DIR="runs/gan_ae_cfpd"

python Gan_autoencoder_Cfp.py \
  --dataset_type cfpd \
  --data "$DATA_DIR" \
  --out "$OUTPUT_DIR" \
  --epochs 50 \
  --batch_size 128 \
  --image_size 64 \
  --latent_ch 16
