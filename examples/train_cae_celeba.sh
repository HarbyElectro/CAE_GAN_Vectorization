#!/usr/bin/env bash
set -e

# Train the Convolutional Autoencoder on CelebA.
# Update DATA_DIR if your CelebA folder is stored somewhere else.

DATA_DIR="data/celeba/img_align_celeba"
OUTPUT_DIR="runs/celeba_cae"

python CAE_CelebA.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --image_size 64 \
  --epochs 100 \
  --batch_size 64 \
  --latent_dim 64
