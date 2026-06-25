#!/usr/bin/env bash
set -e

# Quick CAE test run using a smaller number of images.
# Useful for checking that the environment and dataset path work correctly.

DATA_DIR="data/celeba/img_align_celeba"
OUTPUT_DIR="runs/celeba_cae_quick_test"

python CAE_CelebA.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --image_size 64 \
  --max_images 1000 \
  --epochs 10 \
  --batch_size 32 \
  --latent_dim 64
