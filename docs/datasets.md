# Datasets

This document describes the datasets supported by `CAE_GAN_Vectorization` and the recommended local folder structure.

Large datasets are not included in this repository because of file size limits and licensing restrictions. Users should download datasets from their official sources and place them under the local `data/` directory.

---

## Recommended Data Structure

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

---

## CelebA

CelebA is a large-scale celebrity face dataset containing diverse facial images with variations in pose, lighting, expression, and attributes.

Used by:

```bash
CAE_CelebA.py
```

Recommended path:

```bash
data/celeba/img_align_celeba/
```

Example command:

```bash
python CAE_CelebA.py \
  --data_dir data/celeba/img_align_celeba \
  --output_dir runs/celeba_cae \
  --epochs 100 \
  --batch_size 64
```

---

## CFPD / CFPW-Style Face Dataset

CFPD-style datasets contain frontal and profile facial images. They are useful for evaluating pose-invariant compression and reconstruction.

Used by:

```bash
Gan_autoencoder_Cfp.py
```

Recommended path:

```bash
data/cfpd/Data/Images/
```

Example command:

```bash
python Gan_autoencoder_Cfp.py \
  --dataset_type cfpd \
  --data data/cfpd/Data/Images \
  --out runs/gan_ae_cfpd \
  --epochs 50 \
  --batch_size 128
```

---

## Generic Folder-Based Image Datasets

The GAN-AE script supports generic image folders compatible with `torchvision.datasets.ImageFolder`.

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

Example command:

```bash
python Gan_autoencoder_Cfp.py \
  --dataset_type image_folder \
  --data data/flowers \
  --out runs/gan_ae_flowers \
  --epochs 50 \
  --batch_size 128
```

---

## IMDb Faces

IMDb Faces is a large-scale unconstrained face dataset. It can be used to evaluate robustness and generalization on real-world facial images.

Recommended path:

```bash
data/imdb_faces/
```

Recommended usage:

```bash
python Gan_autoencoder_Cfp.py \
  --dataset_type image_folder \
  --data data/imdb_faces \
  --out runs/gan_ae_imdb
```

---

## Oxford Flowers

Oxford Flowers datasets contain high-resolution flower images with complex colors and textures.

Recommended path:

```bash
data/flowers/
```

Recommended usage:

```bash
python Gan_autoencoder_Cfp.py \
  --dataset_type image_folder \
  --data data/flowers \
  --out runs/gan_ae_flowers
```

---

## MNIST

MNIST is useful for simple autoencoder benchmarking and grayscale image reconstruction experiments.

Recommended path:

```bash
data/mnist/
```

---

## UCF101

UCF101 is a video action-recognition dataset suitable for video sequence compression and reconstruction.

Used by:

```bash
Video_CAE.py
```

Recommended path:

```bash
data/UCF101/
```

Example command:

```bash
python Video_CAE.py \
  --latent_sizes 8,16,32,64,128 \
  --num_frames 30 \
  --num_train 100 \
  --num_test 20 \
  --output_dir runs/video_ae
```

---

## GitHub Dataset Policy

Do not upload large datasets to GitHub.

Use this `.gitignore` rule:

```gitignore
data/*
!data/README.md
```
