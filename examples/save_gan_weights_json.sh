#!/usr/bin/env bash
set -e

# Train or run GAN-AE while saving model configuration and model weights as JSON.

DATA_DIR="data/cfpd/Data/Images"
OUTPUT_DIR="runs/gan_ae_cfpd_json"

python Gan_autoencoder_Cfp.py \
  --dataset_type cfpd \
  --data "$DATA_DIR" \
  --out "$OUTPUT_DIR" \
  --epochs 50 \
  --batch_size 128 \
  --latent_ch 16 \
  --save_weights_json
