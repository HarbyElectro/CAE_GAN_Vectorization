#!/usr/bin/env bash
set -e

# Run quick smoke tests.
# Make sure the dataset paths in each script exist before running this file.

bash examples/train_cae_quick_test.sh

# Uncomment the lines below after confirming your dataset paths.
# bash examples/train_gan_cfpd.sh
# bash examples/train_video_ucf101.sh
