# GAN Autoencoder for Image Compression and Retrieval

This repository provides a PyTorch implementation of a GAN-style autoencoder for image compression, reconstruction, and latent-based retrieval. The code supports both the CFPD face dataset and any generic image dataset organized in folders.

## Features

- **Generic dataset support**
  - `--dataset_type cfpd`: use the CFPD directory structure.
  - `--dataset_type image_folder`: use any dataset compatible with `torchvision.datasets.ImageFolder`.
- **Latent export in NumPy format**
  - After training, the encoder generates latent vectors for validation samples.
  - Latents are exported as a single NumPy file:
    - `runs/gan_ae_generic/latents.npy`
    - Shape: `(N, C_lat, H_lat, W_lat)`
- **Model architecture and weights in JSON**
  - The system saves:
    - `model_config.json`: high-level model and training configuration.
    - `model_weights.json`: generator and discriminator weights as JSON (optional via `--save_weights_json`).
  - These files allow full reproducibility and make it easy to reload or inspect the model outside of PyTorch.
- **HDF5 latent storage**
  - Optionally writes compressed latents and reconstructed images to an HDF5 file for compression and storage analysis.
- **Reconstruction metrics**
  - Computes cosine similarity, Manhattan and Euclidean distance, SSIM, PSNR, and multi-scale SSIM for reconstructed images.
- **Discriminator-based quality metrics**
  - Precision, recall, and accuracy computed from the discriminator outputs on real and reconstructed samples.

## Installation

```bash
git clone <your-repo-url>.git
cd <your-repo-name>
pip install -r requirements.txt
