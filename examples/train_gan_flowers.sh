#!/usr/bin/env bash
set -e

# Train the GAN-guided Autoencoder on Oxford Flowers or any flower image folder.

DATA_DIR="data/flowers"
OUTPUT_DIR="runs/gan_ae_flowers"

python Gan_autoencoder_Cfp.py \
  --dataset_type image_folder \
  --data "$DATA_DIR" \
  --out "$OUTPUT_DIR" \
  --epochs 50 \
  --batch_size 128 \
  --image_size 64 \
  --latent_ch 16
