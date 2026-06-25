# Experiments

This document describes recommended experiment workflows for the `CAE_GAN_Vectorization` repository.

---

## Experiment Goals

The repository supports experiments on:

- Image compression
- Image reconstruction
- Video sequence reconstruction
- Latent-vector extraction
- Latent-space analysis
- Similarity search
- Semantic retrieval
- Compression-ratio analysis
- Reconstruction quality evaluation

---

## Experiment 1: CAE on CelebA

Purpose:

Evaluate image reconstruction quality and latent-vector extraction using a Convolutional Autoencoder.

Script:

```bash
CAE_CelebA.py
```

Example command:

```bash
python CAE_CelebA.py \
  --data_dir data/celeba/img_align_celeba \
  --output_dir runs/celeba_cae \
  --image_size 64 \
  --epochs 100 \
  --batch_size 64 \
  --latent_dim 64
```

Expected outputs:

```bash
runs/celeba_cae/
├── autoencoder_weights.h5
├── autoencoder.png
├── encoder.png
├── decoder.png
└── celeba_encoded_images.npy
```

---

## Experiment 2: GAN-AE on CFPD

Purpose:

Evaluate adversarial image reconstruction on pose-varied face images.

Script:

```bash
Gan_autoencoder_Cfp.py
```

Example command:

```bash
python Gan_autoencoder_Cfp.py \
  --dataset_type cfpd \
  --data data/cfpd/Data/Images \
  --out runs/gan_ae_cfpd \
  --epochs 50 \
  --batch_size 128 \
  --latent_ch 16
```

Expected outputs:

```bash
runs/gan_ae_cfpd/
├── ckpt_epoch050.pt
├── latents.npy
├── model_config.json
├── model_weights.json
├── decoded_row5.png
└── image_comparison_results.txt
```

---

## Experiment 3: GAN-AE on Generic Image Folder

Purpose:

Evaluate reconstruction and latent-vector extraction on non-face datasets such as flowers or custom image folders.

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

## Experiment 4: HDF5 Latent Storage

Purpose:

Store latent vectors and reconstructed images in compressed HDF5 format for large-scale compression studies.

Example command:

```bash
python Gan_autoencoder_Cfp.py \
  --dataset_type image_folder \
  --data data/flowers \
  --out runs/gan_ae_flowers \
  --latents_path runs/flowers_latent.h5 \
  --store_images_in_h5 \
  --h5_gzip 6
```

---

## Experiment 5: Decode From Stored Latents

Purpose:

Evaluate whether stored latent vectors can reconstruct images after compression.

Example command:

```bash
python Gan_autoencoder_Cfp.py \
  --decode_h5 runs/flowers_latent.h5 \
  --ckpt runs/gan_ae_flowers/ckpt_epoch050.pt \
  --decode_n 5
```

---

## Experiment 6: Video Autoencoder Latent-Size Ablation

Purpose:

Compare video reconstruction performance across several latent sizes.

Script:

```bash
Video_CAE.py
```

Example command:

```bash
python Video_CAE.py \
  --latent_sizes 8,16,32,64,128 \
  --num_frames 30 \
  --img_size 224 \
  --batch_size 16 \
  --epochs 100 \
  --num_train 100 \
  --num_test 20 \
  --output_dir runs/video_ae
```

Expected outputs:

```bash
runs/video_ae/
├── recon_latent8_sample0.png
├── recon_latent16_sample0.png
├── recon_latent32_sample0.png
├── reconstruction_loss_comparison.png
└── psnr_ssim_latent_comparison.png
```

---

## Recommended Experiment Naming

Use clear output folder names:

```bash
runs/celeba_cae/
runs/gan_ae_cfpd/
runs/gan_ae_flowers/
runs/video_ae/
```

---

## Recommended Reporting Table

| Model | Dataset | Latent Size | PSNR | SSIM | Compression Ratio |
|---|---|---:|---:|---:|---:|
| CAE | CelebA | 64 | TBD | TBD | TBD |
| GAN-AE | CFPD | 16 channels | TBD | TBD | TBD |
| GAN-AE | Flowers | 16 channels | TBD | TBD | TBD |
| Video-AE | UCF101 | 64 | TBD | TBD | TBD |
