# AI-Guided Vectorization for Visual Data Compression and Retrieval

This repository provides research-oriented implementations of autoencoder-based visual data vectorization models for image and video compression, reconstruction, latent-space extraction, and semantic retrieval.

The repository includes implementations of:

1. **Convolutional Autoencoder (CAE)** for image compression, reconstruction, and latent-vector extraction.
2. **GAN-guided Autoencoder (GAN-AE)** for image compression, reconstruction, and latent-based retrieval.
3. **Conv + BiLSTM Video Autoencoder** for sequence compression and video-frame reconstruction.

The project is designed as a research foundation for experiments involving latent representation learning, multimedia compression, reconstruction quality evaluation, and efficient storage of visual data.

---

## Repository Goals

This repository aims to support research on:

- AI-guided visual data vectorization
- Image and video compression
- Autoencoder-based latent representation learning
- Latent-space extraction and storage
- Semantic retrieval using compressed visual features
- Reconstruction quality evaluation
- Large-scale multimedia storage optimization
- Comparative analysis of CAE, GAN-AE, and sequence-based autoencoders

---

## Main Features

- Convolutional Autoencoder for image compression and reconstruction
- GAN-guided Autoencoder with discriminator-based quality evaluation
- Conv + BiLSTM video autoencoder for frame-sequence compression
- Support for multiple image datasets
- Support for generic folder-based datasets
- Latent-vector export in NumPy format
- Optional HDF5 storage for compressed latent vectors and reconstructions
- Reconstruction quality evaluation using PSNR and SSIM
- Similarity evaluation using cosine, Manhattan, and Euclidean distances
- Checkpoint saving and training continuation
- Experiment output organization for reproducible research
- Citation-ready format for academic use

---

## Supported Models

### 1. Convolutional Autoencoder (CAE)

The CAE model is implemented using TensorFlow/Keras and is designed for image compression, reconstruction, and latent-space extraction.

It supports:

- CelebA
- CFPD / CFPW-style face datasets
- IMDb Faces
- Oxford 17/102 Flowers
- MNIST
- Any folder-based image dataset

The CAE uses an encoder-decoder architecture with a dense latent bottleneck for compact visual representation.

---

### 2. GAN-Guided Autoencoder (GAN-AE)

The GAN-AE model is implemented using PyTorch and combines autoencoder reconstruction with adversarial learning.

It supports:

- CFPD face dataset structure
- Generic image folders compatible with `torchvision.datasets.ImageFolder`
- Latent export in NumPy format
- Optional HDF5 latent storage
- Checkpoint-based training continuation
- Model configuration export
- Optional JSON-based model weight export
- Reconstruction and discriminator-based evaluation metrics

The model is useful for studying whether adversarial learning improves reconstruction quality and latent-space representation for image compression and retrieval.

---

### 3. Conv + BiLSTM Video Autoencoder

The video autoencoder combines convolutional feature extraction with bidirectional LSTM sequence modeling.

It is designed for:

- Video sequence compression
- Frame reconstruction
- Latent-space ablation studies
- Reconstruction quality visualization
- PSNR and SSIM evaluation
- UCF101-based video experiments

---

## Dataset Compatibility

This repository is designed for broad dataset compatibility across different visual domains.

### CFPD / CFPW Face Dataset

Contains frontal and profile facial images and is suitable for evaluating pose-invariant compression and reconstruction.

### CelebA

Contains more than 200,000 celebrity face images with diverse poses, lighting conditions, and facial attributes. It is suitable for high-variance face reconstruction experiments.

### IMDb Faces

Contains large-scale unconstrained facial images collected from IMDb. It is useful for testing model robustness on diverse facial images.

### Oxford 17/102 Flowers

Contains high-resolution flower images with complex colors, textures, and object structures. It is useful for evaluating reconstruction fidelity beyond facial datasets.

### MNIST

A classic handwritten digit dataset useful for benchmarking autoencoder behavior on simple grayscale images.

### UCF101

A widely used video action-recognition dataset suitable for evaluating video sequence compression and reconstruction.

### Generic Folder-Based Image Datasets

Any image dataset organized using a standard folder structure can be used:

```bash
dataset/
    class_1/
        img001.jpg
        img002.jpg
    class_2/
        img003.jpg
        img004.jpg
Use:
```bash
--dataset_type image_folder
```
---
Repository Structure
```bash
AI-Vectorization-Autoencoders/
│
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── configs/
│   ├── cae_celeba.yaml
│   ├── gan_cfpd.yaml
│   ├── gan_image_folder.yaml
│   └── video_ucf101.yaml
│
├── scripts/
│   ├── train_cae.py
│   ├── train_gan_autoencoder.py
│   ├── train_video_autoencoder.py
│   ├── export_latents.py
│   ├── decode_latents.py
│   └── evaluate_reconstruction.py
│
├── src/
│   ├── datasets/
│   │   ├── image_folder_loader.py
│   │   ├── cfpd_loader.py
│   │   └── video_loader.py
│   │
│   ├── models/
│   │   ├── cae.py
│   │   ├── gan_autoencoder.py
│   │   └── video_bilstm_autoencoder.py
│   │
│   ├── metrics/
│   │   ├── reconstruction_metrics.py
│   │   ├── similarity_metrics.py
│   │   └── discriminator_metrics.py
│   │
│   ├── utils/
│   │   ├── checkpoints.py
│   │   ├── visualization.py
│   │   ├── latent_io.py
│   │   └── config.py
│   │
│   └── __init__.py
│
├── docs/
│   ├── datasets.md
│   ├── experiments.md
│   ├── latent_vector_format.md
│   ├── metrics.md
│   └── reproducibility.md
│
├── examples/
│   ├── train_cae_celeba.sh
│   ├── train_gan_cfpd.sh
│   ├── train_gan_flowers.sh
│   ├── export_latents.sh
│   └── train_video_ucf101.sh
│
├── notebooks/
│   ├── latent_space_visualization.ipynb
│   ├── reconstruction_comparison.ipynb
│   └── retrieval_demo.ipynb
│
├── data/
│   └── README.md
│
├── runs/
│   └── README.md
│
└── tests/
    ├── test_datasets.py
    ├── test_models.py
    └── test_metrics.py
```
---
Installation
Clone the repository:
```bash
git clone https://github.com/your-username/AI-Vectorization-Autoencoders.git
cd AI-Vectorization-Autoencoders
```
Create a Python environment:
```bash
conda create -n visual-vectorization python=3.10
conda activate visual-vectorization
```
Install the required packages:
```bash
pip install -r requirements.txt
```
Alternatively, if an `environment.yml` file is provided:
```bash
conda env create -f environment.yml
conda activate visual-vectorization
```
---
# Basic Usage
This repository supports both TensorFlow/Keras and PyTorch-based models.
- Use the CAE implementation for TensorFlow/Keras image autoencoder experiments.
- Use the GAN-AE implementation for PyTorch-based adversarial autoencoder experiments.
- Use the video autoencoder implementation for frame-sequence compression experiments.
---
Training Examples
Train CAE on CelebA
```bash
python CAE_CelebA.py \
    --data_dir archive/img_align_celeba \
    --output_dir runs/celeba_cae \
    --epochs 100 \
    --batch_size 64
```
---
Train GAN-AE on CFPD Dataset
```bash
python Gan_autoencoder_Cfp.py \
    --dataset_type cfpd \
    --data ./cfp-dataset/Data/Images \
    --out runs/gan_ae_cfpd \
    --epochs 50 \
    --batch_size 128 \
    --latent_ch 16
```
---
Train GAN-AE on Any Generic Image Dataset
```bash
python Gan_autoencoder_Cfp.py \
    --dataset_type image_folder \
    --data ./datasets/CelebA \
    --out runs/gan_ae_celeba \
    --epochs 50 \
    --batch_size 128
```
---
Export Latent Vectors
```bash
python Gan_autoencoder_Cfp.py \
    --dataset_type image_folder \
    --data ./datasets/IMDb_faces \
    --latents_path runs/imdb_latents.npy \
    --latents_n 5000
```
The exported latent vectors are saved as:
```bash
runs/imdb_latents.npy
```
Expected latent shape:
```bash
(N, C_lat, H_lat, W_lat)
```
---
Store Latents in Compressed HDF5 Format
```bash
python Gan_autoencoder_Cfp.py \
    --dataset_type image_folder \
    --data ./datasets/flowers \
    --latents_path runs/flowers_latent.h5 \
    --store_images_in_h5 \
    --h5_gzip 6
```
---
Decode From Stored Latents
```bash
python Gan_autoencoder_Cfp.py \
    --decode_h5 runs/flowers_latent.h5 \
    --decode_n 5
```
---
Resume Training From Checkpoint
```bash
python Gan_autoencoder_Cfp.py \
    --ckpt runs/gan_ae_cfpd/ckpt_epoch050.pt
```
---
Save Model Configuration and Weights as JSON
```bash
python Gan_autoencoder_Cfp.py \
    --save_weights_json
```
This saves:
```bash
model_config.json
model_weights.json
```
---
Train Video Autoencoder on UCF101
```bash
python Video_CAE.py \
    --latent_sizes 16,32,64 \
    --num_frames 20 \
    --num_train 200 \
    --num_test 50
```
---
Model Outputs
A typical experiment directory may contain:
```bash
runs/gan_ae_cfpd/
│
├── ckpt_epoch050.pt
├── latents.npy
├── model_config.json
├── model_weights.json
├── reconstruction_samples.png
├── image_comparison_results.txt
├── metrics.json
└── logs/
```
---
Latent Vector Storage
This repository supports latent-vector storage in different formats.
NumPy Format
```bash
latents.npy
```
Recommended for large-scale compression experiments and structured storage.
JSON Format
```bash
model_config.json
model_weights.json
```
#External inspection and reproducibility metadata.
---
Evaluation Metrics
The repository supports reconstruction, similarity, and discriminator-based metrics.
Reconstruction Metrics
- Structural Similarity Index Measure (SSIM)
- Multiscale SSIM
- Peak Signal-to-Noise Ratio (PSNR)
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Latent Similarity Metrics
- Cosine similarity
- Euclidean distance
- Manhattan distance
- GAN Discriminator Metrics
- Precision
- Recall
- Accuracy
Real-vs-reconstructed discrimination score
---
Reproducibility Guidelines
For each experiment, the following information should be saved:
- Model type
- Dataset name
- Dataset path
- Image resolution
- Number of frames for video experiments
- Latent dimension
- Batch size
- Number of epochs
- Optimizer settings
- Learning rate
- Random seed
- Checkpoint path
- Reconstruction metrics
- Latent-vector output path
Software versions

---
Research Workflow
Select a dataset.
Choose a model architecture:
CAE for image compression
GAN-AE for adversarial image reconstruction
Video-AE for sequence compression
Train the model.
Export latent vectors.
Store compressed representations in `.npy` or `.h5` format.
Reconstruct images or video frames from latent vectors.
Evaluate reconstruction quality using PSNR, SSIM, and similarity metrics.
Compare compression ratio, retrieval performance, and reconstruction fidelity.
Save model configuration and experiment metadata.
Report results for reproducible research.
---
Example Research Questions
This repository can support experiments such as:
How does latent-space size affect reconstruction quality?
How do CAE and GAN-AE models compare for face reconstruction?
Can compressed latent vectors support semantic retrieval?
How does reconstruction quality vary across faces, flowers, digits, and videos?
What is the trade-off between compression ratio and visual fidelity?
How well do models generalize across visual domains?
Can latent-vector storage improve multimedia lakehouse efficiency?
---
Notes on Dataset Storage
Large datasets should not be committed to GitHub.
Recommended local structure:
```bash
data/
│
├── celeba/
├── cfpd/
├── imdb_faces/
├── flowers/
├── mnist/
└── UCF101/
```
Add a `data/README.md` file explaining where datasets should be placed.
Example:
```markdown
# Data Directory

This folder is reserved for local datasets.

Datasets are not tracked by Git because of size and licensing restrictions.

Expected structure:

- `data/celeba/`
- `data/cfpd/`
- `data/imdb_faces/`
- `data/flowers/`
- `data/mnist/`
- `data/UCF101/`
```
---
Notes on Experiment Outputs
Experiment outputs should be saved under:
```bash
runs/
```
The `runs/` directory may contain checkpoints, latent vectors, reconstructions, logs, and evaluation reports.
These files are usually not tracked by Git.
Add a `runs/README.md` file explaining the purpose of this folder.
Example:
```markdown
# Runs Directory

This folder stores experiment outputs such as:

- Model checkpoints
- Latent vectors
- Reconstruction images
- Metric reports
- Logs

Generated experiment files are not tracked by Git.
```
---
Citation
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
Contact
For questions, research collaboration, or citation-related inquiries, please contact the repository maintainer.
---
Acknowledgment
This repository was developed as part of research on AI-guided visual data vectorization, efficient multimedia storage, and semantic retrieval using learned latent representations.
