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
  - Saves:
    - `model_config.json` — model hyperparameters and training settings.
    - `model_weights.json` — generator and discriminator weights serialized to JSON (optional via `--save_weights_json`).
  - Enables full reproducibility and external model inspection.

- **HDF5 latent storage**
  - Optionally writes compressed latents and reconstructed images to an HDF5 file for large-scale compression experiments.

- **Reconstruction metrics**
  - Computes:
    - Cosine similarity  
    - Manhattan distance  
    - Euclidean distance  
    - SSIM  
    - PSNR  
    - Multiscale SSIM  

- **Discriminator-based quality metrics**
  - Computes precision, recall, and accuracy using discriminator predictions on real and reconstructed samples.

---

## 🌐 Diverse Dataset Compatibility

This repository is designed for broad dataset support. In addition to CFPD, the training pipeline has been tested on several benchmark datasets across different visual domains:

### ✓ **CFPD (Cross-Pose Face Dataset)**  
Contains frontal and profile facial views, ideal for evaluating pose-invariant compression and reconstruction.

### ✓ **CelebA**  
Features over 200,000 celebrity face images with diverse poses, lighting, and attributes, useful for high-variance face reconstruction.

### ✓ **IMDb Faces**  
Includes 1.2M unconstrained facial images collected from IMDb, enabling large-scale evaluation of model robustness.

### ✓ **Oxford 17/102 Flowers**  
Provides high-resolution floral images with complex textures and colors, excellent for testing reconstruction fidelity beyond faces.

### ✓ **MNIST**  
A classic dataset of handwritten digits, ideal for benchmarking autoencoder behavior on simple grayscale images.

All datasets can be used with:

- `--dataset_type image_folder`  
- A directory structure following:

dataset/
class_1/
img001.jpg
img002.jpg
class_2/
img003.jpg
img004.jpg
▶ 1. Train on CFPD Dataset
python gan_autoencoder.py \
    --dataset_type cfpd \
    --data ./cfp-dataset/Data/Images \
    --out runs/gan_ae_cfpd \
    --epochs 50 \
    --batch_size 128 \
    --latent_ch 16
Description:

Automatically loads CFPD folder structure

Trains GAN autoencoder for 50 epochs

Saves metrics, reconstructions, model config, and latents
▶ 2. Train on Any Generic Image Dataset
python gan_autoencoder.py \
    --dataset_type image_folder \
    --data ./datasets/CelebA \
    --out runs/gan_ae_celeba \
    --epochs 50 \
    --batch_size 128
Description:

Uses a standard ImageFolder layout and trains the Autoencoder+GAN on arbitrary datasets.
▶ 3. Export Latent Vectors (NumPy)

Latents are automatically written during validation at the end of each epoch.

You can manually export a larger set using:
python gan_autoencoder.py \
    --dataset_type image_folder \
    --data ./datasets/IMDb_faces \
    --latents_path runs/imdb_latents.npy \
    --latents_n 5000
Description:

Outputs latent vectors to a single .npy file

Supports float16 / float32

Useful for retrieval, clustering, visualization, etc.
▶ 4. Store Latents in Compressed HDF5
python gan_autoencoder.py \
    --dataset_type image_folder \
    --data ./datasets/flowers \
    --latents_path runs/flowers_latent.h5 \
    --store_images_in_h5 \
    --h5_gzip 6
Description:

Saves compressed latent tensors

Optionally includes reconstructed images

Ideal for large-scale database storage experiments
▶ 5. Decode From Stored Latents (Reconstruction)
python gan_autoencoder.py \
    --decode_h5 runs/flowers_latent.h5 \
    --decode_n 5
Description:

Loads latent vectors from HDF5

Uses trained decoder to reconstruct images

Saves side-by-side output grid
▶ 6. Load Checkpoint and Resume Training
python gan_autoencoder.py \
    --ckpt runs/gan_ae_cfpd/ckpt_epoch050.pt
🧩 Model Configuration and Weight Saving
python gan_autoencoder.py \
    --save_weights_json
Creates:

model_config.json

model_weights.json
📁 Project Structure
CAE_GAN_Vectorization/
│
├── gan_autoencoder.py          # Main training script
├── datasets/                   # Your datasets go here
├── runs/
│   ├── gan_ae_cfpd/
│   │   ├── latents.npy
│   │   ├── model_config.json
│   │   ├── model_weights.json
│   │   ├── *.png (reconstructions)
│   │   └── image_comparison_results.txt
│   ...
├── README.md
└── requirements.txt
📜 Citation

If you use this repository in your research, please cite:
@misc{harby2025caegan,
  author       = {Ahmed Harby},
  title        = {CAE–GAN Vectorization Framework},
  year         = {2025},
  howpublished = {\url{https://github.com/HarbyElectro/CAE_GAN_Vectorization}},
}
