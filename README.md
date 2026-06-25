# AI-guided Vectorization for Efficient Storage and Semantic Retrieval of Visual Data

<p align="center">
  <b>Convolutional Autoencoder • GAN Autoencoder • Video Autoencoder • Latent Vector Extraction • Visual Data Reconstruction</b>
</p>

<p align="center">
  <a href="https://github.com/HarbyElectro/CAE_GAN_Vectorization">
    <img src="https://img.shields.io/badge/GitHub-CAE__GAN__Vectorization-black?logo=github" alt="GitHub Repository">
  </a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-Keras-orange" alt="TensorFlow Keras">
  <img src="https://img.shields.io/badge/PyTorch-GAN--AE-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/Research-Visual%20Data%20Vectorization-green" alt="Research">
</p>

---

# Overview

`CAE_GAN_Vectorization` is a research-oriented repository for AI-guided visual data vectorization using deep autoencoder models.

The repository provides implementations for:

1. **Convolutional Autoencoder (CAE)** for image compression, reconstruction, and latent-vector extraction.
2. **GAN-guided Autoencoder (GAN-AE)** for image compression, reconstruction, adversarial learning, and latent-space export.
3. **Conv + BiLSTM Video Autoencoder** for video sequence compression and frame reconstruction.

The project is designed to support research on efficient multimedia storage, learned latent representations, visual data reconstruction, and semantic retrieval using compressed feature vectors.

---

# Repository Link

```bash
https://github.com/HarbyElectro/CAE_GAN_Vectorization
```

Clone the repository:

```bash
git clone https://github.com/HarbyElectro/CAE_GAN_Vectorization.git
cd CAE_GAN_Vectorization
```

---

# Table of Contents

- [Overview](#overview)
- [Research Motivation](#research-motivation)
- [Main Contributions](#main-contributions)
- [Repository Structure](#repository-structure)
- [Core Scripts](#core-scripts)
- [Supported Models](#supported-models)
- [Supported Datasets](#supported-datasets)
- [Installation](#installation)
- [Dataset Organization](#dataset-organization)
- [Training and Usage](#training-and-usage)
- [Latent Vector Export and Storage](#latent-vector-export-and-storage)
- [Evaluation Metrics](#evaluation-metrics)
- [Experiment Outputs](#experiment-outputs)
- [Research Workflow](#research-workflow)
- [Citation](#citation)
- [Contact](#contact)

---

# Research Motivation

Modern multimedia datasets contain large volumes of images and videos. Storing, searching, retrieving, and reconstructing these files efficiently is challenging because raw visual data is large and computationally expensive to process.

This repository investigates the use of deep learning models to encode images and videos into compact latent vectors that can be used for:

- Compression
- Reconstruction
- Latent-space analysis
- Similarity search
- Semantic retrieval
- Efficient storage
- Multimedia indexing
- Research on AI-guided data vectorization

The project is aligned with research on transforming raw visual data into compact learned representations that preserve important visual and semantic information.

---

# Main Contributions

This repository provides:

- A TensorFlow/Keras **Convolutional Autoencoder** for image compression and reconstruction.
- A PyTorch **GAN-guided Autoencoder** for image reconstruction and latent-vector extraction.
- A TensorFlow/Keras **Conv + BiLSTM Video Autoencoder** for video sequence compression.
- Support for generic image-folder datasets.
- Support for CFPD-style face datasets.
- Latent-vector export in NumPy format.
- Optional HDF5 storage for latent vectors and reconstructed images.
- Model configuration export in JSON format.
- Optional model-weight export in JSON format.
- Reconstruction evaluation using SSIM, MS-SSIM, and PSNR.
- Similarity evaluation using cosine, Euclidean, and Manhattan distances.
- GAN discriminator evaluation using precision, recall, and accuracy.
- A foundation for reproducible research on visual data vectorization.

---

# Repository Structure

```bash
CAE_GAN_Vectorization/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── CAE_CelebA.py
├── Gan_autoencoder_Cfp.py
├── Video_CAE.py
│
├── data/
│   ├── README.md
│   ├── celeba/
│   │   └── img_align_celeba/
│   ├── cfpd/
│   │   └── Data/
│   │       └── Images/
│   ├── cfpw/
│   ├── imdb_faces/
│   ├── flowers/
│   ├── mnist/
│   └── UCF101/
│
├── datasets/
│   ├── README.md
│   ├── CelebA/
│   ├── CFPD/
│   ├── IMDb_faces/
│   ├── Oxford_Flowers/
│   └── MNIST/
│
├── runs/
│   ├── README.md
│   ├── celeba_cae/
│   │   ├── autoencoder.png
│   │   ├── encoder.png
│   │   ├── decoder.png
│   │   ├── autoencoder_weights.h5
│   │   └── celeba_encoded_images.npy
│   │
│   ├── gan_ae_cfpd/
│   │   ├── ckpt_epoch001.pt
│   │   ├── ckpt_epoch050.pt
│   │   ├── latents.npy
│   │   ├── latents.h5
│   │   ├── model_config.json
│   │   ├── model_weights.json
│   │   ├── decoded_row5.png
│   │   ├── train_recon_row5_step0000000.png
│   │   ├── val_recon_row5_epoch001.png
│   │   └── image_comparison_results.txt
│   │
│   ├── gan_ae_celeba/
│   │   ├── latents.npy
│   │   ├── latents.h5
│   │   ├── model_config.json
│   │   └── image_comparison_results.txt
│   │
│   ├── gan_ae_flowers/
│   │   ├── flowers_latent.h5
│   │   ├── decoded_row5.png
│   │   └── image_comparison_results.txt
│   │
│   └── video_ae/
│       ├── recon_latent8_sample0.png
│       ├── recon_latent16_sample0.png
│       ├── recon_latent32_sample0.png
│       ├── recon_latent64_sample0.png
│       ├── recon_latent128_sample0.png
│       ├── reconstruction_loss_comparison.png
│       └── psnr_ssim_latent_comparison.png
│
├── docs/
│   ├── README.md
│   ├── datasets.md
│   ├── experiments.md
│   ├── model_architectures.md
│   ├── latent_vector_format.md
│   ├── metrics.md
│   └── reproducibility.md
│
├── examples/
│   ├── train_cae_celeba.sh
│   ├── train_gan_cfpd.sh
│   ├── train_gan_celeba.sh
│   ├── train_gan_flowers.sh
│   ├── export_latents.sh
│   ├── decode_latents.sh
│   └── train_video_ucf101.sh
│
├── configs/
│   ├── cae_celeba.yaml
│   ├── gan_cfpd.yaml
│   ├── gan_image_folder.yaml
│   ├── gan_flowers.yaml
│   └── video_ucf101.yaml
│
├── notebooks/
│   ├── latent_space_visualization.ipynb
│   ├── reconstruction_comparison.ipynb
│   ├── retrieval_demo.ipynb
│   └── compression_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── image_folder_loader.py
│   │   ├── cfpd_loader.py
│   │   ├── celeba_loader.py
│   │   └── video_loader.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cae.py
│   │   ├── gan_autoencoder.py
│   │   └── video_bilstm_autoencoder.py
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── reconstruction_metrics.py
│   │   ├── similarity_metrics.py
│   │   └── discriminator_metrics.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── checkpoints.py
│       ├── visualization.py
│       ├── latent_io.py
│       ├── hdf5_utils.py
│       └── config.py
│
└── tests/
    ├── test_datasets.py
    ├── test_models.py
    ├── test_metrics.py
    └── test_latent_io.py
```

---

# Core Scripts

| Script | Framework | Purpose |
|---|---|---|
| `CAE_CelebA.py` | TensorFlow/Keras | Trains a convolutional autoencoder for image compression, reconstruction, and latent-vector extraction. |
| `Gan_autoencoder_Cfp.py` | PyTorch | Trains a GAN-style autoencoder for image reconstruction, adversarial learning, latent export, HDF5 storage, and evaluation. |
| `Video_CAE.py` | TensorFlow/Keras | Trains and evaluates a Conv + BiLSTM video autoencoder across multiple latent sizes. |

---

# Supported Models

# 1. Convolutional Autoencoder

Main script:

```bash
CAE_CelebA.py
```

The Convolutional Autoencoder is designed for image compression and reconstruction.

It supports:

- Image loading
- Image resizing
- Encoder-decoder training
- Dense latent bottleneck extraction
- Reconstruction learning
- Autoencoder architecture visualization
- Encoder architecture visualization
- Decoder architecture visualization
- Latent-vector export

Important output files:

```bash
autoencoder.png
encoder.png
decoder.png
autoencoder_weights.h5
celeba_encoded_images.npy
```

---

# 2. GAN-Guided Autoencoder

Main script:

```bash
Gan_autoencoder_Cfp.py
```

The GAN-guided autoencoder is designed for adversarial reconstruction and latent-space extraction.

It supports:

- CFPD dataset layout
- Generic `ImageFolder` dataset layout
- Generator-based reconstruction
- Discriminator-based quality evaluation
- Latent export to `.npy`
- Latent and image storage in `.h5`
- Checkpoint saving
- Decode-only mode from HDF5 latent files
- Model configuration export
- Optional model weight export to JSON

Important output files:

```bash
ckpt_epoch001.pt
ckpt_epoch050.pt
latents.npy
latents.h5
model_config.json
model_weights.json
decoded_row5.png
image_comparison_results.txt
```

---

# 3. Conv + BiLSTM Video Autoencoder

Main script:

```bash
Video_CAE.py
```

The video autoencoder is designed for sequence compression and frame reconstruction.

It supports:

- Video frame-sequence modeling
- Convolutional feature extraction
- Bidirectional LSTM latent encoding
- Latent-size ablation
- Reconstruction visualization
- PSNR evaluation
- SSIM evaluation

Important output files:

```bash
recon_latent<size>_sample<index>.png
reconstruction_loss_comparison.png
psnr_ssim_latent_comparison.png
```

---

# Datasets

# CelebA

CelebA contains large-scale celebrity face images with diverse poses, lighting conditions, and facial attributes.

Use cases:

- Face reconstruction
- Image compression
- Latent-space representation learning

---

# CFPD / CFPW-Style Face Datasets

CFPD-style datasets include frontal and profile facial views.

Use cases:

- Pose-invariant face reconstruction
- Face compression
- Latent robustness testing

Expected CFPD-style structure:

```bash
cfp-dataset/
└── Data/
    └── Images/
        ├── person_001/
        │   ├── frontal/
        │   │   ├── image_001.jpg
        │   │   └── image_002.jpg
        │   └── profile_001.jpg
        └── person_002/
            ├── frontal/
            └── profile_001.jpg
```

---

# IMDb Faces

IMDb Faces contains large-scale unconstrained face images.

Use cases:

- Robust reconstruction testing
- Large-scale face image compression
- Generalization evaluation

---

# Oxford 17/102 Flowers

Oxford Flowers datasets contain high-resolution flower images with complex colors and textures.

Use cases:

- Non-face reconstruction testing
- Texture reconstruction analysis
- Color-preserving compression

---

# MNIST

MNIST contains handwritten digit images.

Use cases:

- Simple autoencoder benchmarking
- Low-complexity reconstruction
- Grayscale image compression

---

# UCF101

UCF101 is a video action-recognition dataset.

Use cases:

- Video sequence compression
- Frame reconstruction
- Temporal latent-space evaluation

---

# Generic Folder-Based Image Datasets

The GAN-AE script supports generic folder-based datasets using `torchvision.datasets.ImageFolder`.

Expected structure:

```bash
dataset/
    class_1/
        img001.jpg
        img002.jpg
    class_2/
        img003.jpg
        img004.jpg
```

Use:

```bash
--dataset_type image_folder
```

---

# Installation

# 1. Clone the Repository

```bash
git clone https://github.com/HarbyElectro/CAE_GAN_Vectorization.git
cd CAE_GAN_Vectorization
```

# 2. Create a Python Environment

```bash
conda create -n cae-gan-vectorization python=3.10
conda activate cae-gan-vectorization
```

# 3. Install Dependencies

If `requirements.txt` is available:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available yet, install the main dependencies manually:

```bash
pip install numpy pandas matplotlib scikit-learn scikit-image scipy h5py opencv-python pillow tqdm
pip install tensorflow
pip install torch torchvision
```

---

# Dataset Organization

Large datasets should not be committed to GitHub.

Local structure:

```bash
data/
│
├── celeba/
│   └── img_align_celeba/
│
├── cfpd/
│   └── Data/
│       └── Images/
│
├── imdb_faces/
├── flowers/
├── mnist/
└── UCF101/
```

```markdown
# Data Directory

This folder is reserved for local datasets.

Datasets are not tracked by Git because of size and licensing restrictions.

Expected folders:

- `data/celeba/`
- `data/cfpd/`
- `data/imdb_faces/`
- `data/flowers/`
- `data/mnist/`
- `data/UCF101/`
```

---

# Training and Usage

# Train CAE on CelebA

```bash
python CAE_CelebA.py \
    --data_dir data/celeba/img_align_celeba \
    --output_dir runs/celeba_cae \
    --image_size 64 \
    --epochs 100 \
    --batch_size 64 \
    --latent_dim 64
```

Optional quick experiment:

```bash
python CAE_CelebA.py \
    --data_dir data/celeba/img_align_celeba \
    --output_dir runs/celeba_cae_test \
    --max_images 1000 \
    --epochs 10 \
    --batch_size 32
```

---

# Train GAN-AE on CFPD Dataset

```bash
python Gan_autoencoder_Cfp.py \
    --dataset_type cfpd \
    --data data/cfpd/Data/Images \
    --out runs/gan_ae_cfpd \
    --epochs 50 \
    --batch_size 128 \
    --image_size 64 \
    --latent_ch 16
```

---

# Train GAN-AE on a Generic Image Dataset

```bash
python Gan_autoencoder_Cfp.py \
    --dataset_type image_folder \
    --data data/celeba \
    --out runs/gan_ae_celeba \
    --epochs 50 \
    --batch_size 128 \
    --image_size 64
```

---

# Export Latent Vectors

The GAN-AE script exports validation latent vectors after training.

```bash
python Gan_autoencoder_Cfp.py \
    --dataset_type image_folder \
    --data ./datasets/IMDb_faces \
    --out runs/gan_ae_imdb \
    --latents_path runs/imdb_latents.h5 \
    --latents_n 5000
```

The script also saves:

```bash
runs/gan_ae_imdb/latents.npy
```

Expected NumPy latent shape:

```bash
(N, C_lat, H_lat, W_lat)
```

---

# Store Latents in Compressed HDF5 Format

```bash
python Gan_autoencoder_Cfp.py \
    --dataset_type image_folder \
    --data ./datasets/flowers \
    --out runs/gan_ae_flowers \
    --latents_path runs/flowers_latent.h5 \
    --store_images_in_h5 \
    --h5_gzip 6
```

---

# Decode From Stored Latents

```bash
python Gan_autoencoder_Cfp.py \
    --decode_h5 runs/flowers_latent.h5 \
    --ckpt runs/gan_ae_flowers/ckpt_epoch050.pt \
    --decode_n 5
```

---

# Resume or Load From Checkpoint

```bash
python Gan_autoencoder_Cfp.py \
    --ckpt runs/gan_ae_cfpd/ckpt_epoch050.pt
```

---

# Save Model Configuration and Weights as JSON

```bash
python Gan_autoencoder_Cfp.py \
    --save_weights_json
```

This can save:

```bash
model_config.json
model_weights.json
```

---

# Train Video Autoencoder

```bash
python Video_CAE.py \
    --latent_sizes 8,16,32,64,128 \
    --num_frames 30 \
    --img_size 224 \
    --batch_size 16 \
    --epochs 100 \
    --output_dir runs/video_ae
```

---

# Command-Line Arguments

# CAE_CelebA.py

| Argument | Default | Description |
|---|---:|---|
| `--data_dir` | `archive/img_align_celeba` | Path to dataset root directory. |
| `--output_dir` | `runs/celeba_cae` | Directory for weights, plots, and latents. |
| `--image_size` | `64` | Input image size. |
| `--max_images` | `None` | Optional maximum number of images to load. |
| `--test_split` | `0.2` | Fraction of data used for testing. |
| `--epochs` | `100` | Number of training epochs. |
| `--batch_size` | `64` | Training batch size. |
| `--latent_dim` | `64` | Dense bottleneck latent dimension. |

# Gan_autoencoder_Cfp.py

| Argument | Default | Description |
|---|---:|---|
| `--dataset_type` | `cfpd` | Dataset type: `cfpd` or `image_folder`. |
| `--data` | `./cfp-dataset/Data/Images` | Dataset root directory. |
| `--out` | `./runs/gan_ae_generic` | Output directory. |
| `--epochs` | `50` | Number of training epochs. |
| `--batch_size` | `128` | Training batch size. |
| `--lr` | `2e-4` | Learning rate. |
| `--lambda_rec` | `0.6` | Reconstruction loss weight. |
| `--base_ch` | `64` | Base channel count. |
| `--latent_ch` | `16` | Latent channel count. |
| `--image_size` | `64` | Input image size. |
| `--seed` | `0` | Random seed. |
| `--cpu` | `False` | Force CPU execution. |
| `--log_every` | `100` | Logging interval. |
| `--sample_every` | `500` | Sample output interval. |
| `--eval_batch` | `16` | Evaluation batch size. |
| `--eval_n` | `5` | Number of samples for evaluation. |
| `--pr_threshold` | `0.5` | Discriminator threshold. |
| `--latents_path` | `./runs/latents.h5` | HDF5 latent output path. |
| `--latents_n` | `64` | Number of latent vectors to export. |
| `--latents_dtype` | `float16` | Latent dtype: `float16` or `float32`. |
| `--disable_h5` | `False` | Disable HDF5 export. |
| `--store_images_in_h5` | `False` | Store images in HDF5 file. |
| `--h5_gzip` | `4` | HDF5 gzip compression level. |
| `--decode_h5` | empty | HDF5 file for decode-only mode. |
| `--decode_n` | `5` | Number of decoded samples. |
| `--ckpt` | empty | Checkpoint path. |
| `--save_weights_json` | `False` | Save model weights as JSON. |
| `--ckpt_every` | `1` | Checkpoint interval in epochs. |

# Video_CAE.py

| Argument | Default | Description |
|---|---:|---|
| `--img_size` | `224` | Frame height and width. |
| `--num_frames` | `30` | Number of frames per video sequence. |
| `--channels` | `3` | Number of image channels. |
| `--batch_size` | `16` | Training batch size. |
| `--epochs` | `100` | Maximum number of epochs. |
| `--patience` | `10` | Early stopping patience. |
| `--latent_sizes` | `8,16,32,64,128` | Latent sizes to evaluate. |
| `--num_train` | `100` | Number of training sequences. |
| `--num_test` | `20` | Number of test sequences. |
| `--output_dir` | `runs/video_ae` | Directory for plots and results. |

---

# Latent Vector Export and Storage

Latent vectors are compact learned representations generated by the encoder.

They can be used for:

- Compression
- Reconstruction
- Similarity search
- Semantic retrieval
- Multimedia indexing
- Storage optimization

Supported storage formats include:

```bash
.npy
.h5
.json
```

---

# Evaluation Metrics

The repository supports reconstruction, similarity, and discriminator-based metrics.

# Reconstruction Metrics

- Structural Similarity Index Measure (SSIM)
- Multiscale SSIM (MS-SSIM)
- Peak Signal-to-Noise Ratio (PSNR)
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

# Latent Similarity Metrics

- Cosine similarity
- Euclidean distance
- Manhattan distance

# GAN Discriminator Metrics

- Precision
- Recall
- Accuracy
- Real-vs-reconstructed discriminator score

---

# Experiment Outputs

A typical CAE experiment may produce:

```bash
runs/celeba_cae/
│
├── autoencoder.png
├── encoder.png
├── decoder.png
├── autoencoder_weights.h5
└── celeba_encoded_images.npy
```

A typical GAN-AE experiment may produce:

```bash
runs/gan_ae_cfpd/
│
├── ckpt_epoch001.pt
├── ckpt_epoch050.pt
├── latents.npy
├── latents.h5
├── model_config.json
├── model_weights.json
├── decoded_row5.png
├── val_recon_row5_epoch001.png
└── image_comparison_results.txt
```

A typical video autoencoder experiment may produce:

```bash
runs/video_ae/
│
├── recon_latent8_sample0.png
├── recon_latent16_sample0.png
├── recon_latent32_sample0.png
├── recon_latent64_sample0.png
├── recon_latent128_sample0.png
├── reconstruction_loss_comparison.png
└── psnr_ssim_latent_comparison.png
```

---

# Research Workflow

1. Select a dataset.
2. Choose a model architecture:
   - CAE for image compression.
   - GAN-AE for adversarial image reconstruction.
   - Video-AE for sequence compression.
3. Train the selected model.
4. Export latent vectors.
5. Store compressed representations in `.npy` or `.h5` format.
6. Reconstruct images or video frames from latent vectors.
7. Evaluate reconstruction quality using PSNR, SSIM, and similarity metrics.
8. Compare compression ratio, retrieval performance, and reconstruction fidelity.
9. Save model configuration and experiment metadata.
10. Report results for reproducible research.

---

# Citation

If you use this repository in your research, please cite:

```bibtex
@article{harby2025aivectorization,
  author       = {Harby, Ahmed A. and Zulkernine, F. and Abdulsalam, H. M.},
  title        = {AI-guided Vectorization for Efficient Storage and Semantic Retrieval of Visual Data},
  journal      = {Discover Artificial Intelligence},
  year         = {2025},
  publisher    = {Springer Nature}
}
```

---

# Contact

Repository maintainer:

```text
Ahmed A. Harby
Email: ahmed.harby@queensu.ca
Repository: https://github.com/HarbyElectro/CAE_GAN_Vectorization
```

---

# Acknowledgment

## Acknowledgment

This repository was developed as part of a PhD research work focused on AI-guided visual data vectorization, efficient multimedia storage, and semantic retrieval using learned latent representations.

This work is associated with **BAMLab Research** and was conducted under the supervision of **Prof. Farhana Zulkernine**.
