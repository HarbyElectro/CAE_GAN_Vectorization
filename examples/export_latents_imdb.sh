#!/usr/bin/env bash
set -e

# Export latent vectors from a generic image-folder dataset.
# This example uses IMDb Faces.

DATA_DIR="data/imdb_faces"
OUTPUT_DIR="runs/gan_ae_imdb"
LATENTS_PATH="runs/imdb_latents.h5"

python Gan_autoencoder_Cfp.py \
  --dataset_type image_folder \
  --data "$DATA_DIR" \
  --out "$OUTPUT_DIR" \
  --latents_path "$LATENTS_PATH" \
  --latents_n 5000 \
  --latents_dtype float16
