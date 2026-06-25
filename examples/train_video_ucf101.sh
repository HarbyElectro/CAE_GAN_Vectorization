#!/usr/bin/env bash
set -e

# Train the Conv + BiLSTM Video Autoencoder.
# Update DATA_DIR if your Video_CAE.py script supports a dataset path argument.

OUTPUT_DIR="runs/video_ae"

python Video_CAE.py \
  --latent_sizes 8,16,32,64,128 \
  --num_frames 30 \
  --img_size 224 \
  --batch_size 16 \
  --epochs 100 \
  --num_train 100 \
  --num_test 20 \
  --output_dir "$OUTPUT_DIR"
