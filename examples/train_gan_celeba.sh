#!/usr/bin/env bash
set -e

# Train the GAN-guided Autoencoder on CelebA using generic image-folder mode.
# The folder should follow torchvision ImageFolder format.

DATA_DIR="data/celeba"
OUTPUT_DIR="runs/gan_ae_celeba"

python Gan_autoencoder_Cfp.py \
  --dataset_type image_folder \
  --data "$DATA_DIR" \
  --out "$OUTPUT_DIR" \
  --epochs 50 \
  --batch_size 128 \
  --image_size 64 \
  --latent_ch 16
