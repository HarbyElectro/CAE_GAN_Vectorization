#!/usr/bin/env bash
set -e

# Store compressed latent vectors and optionally images in HDF5 format.

DATA_DIR="data/flowers"
OUTPUT_DIR="runs/gan_ae_flowers"
LATENTS_PATH="runs/flowers_latent.h5"

python Gan_autoencoder_Cfp.py \
  --dataset_type image_folder \
  --data "$DATA_DIR" \
  --out "$OUTPUT_DIR" \
  --latents_path "$LATENTS_PATH" \
  --store_images_in_h5 \
  --h5_gzip 6 \
  --latents_n 5000
