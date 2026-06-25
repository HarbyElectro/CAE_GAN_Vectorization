# Reproducibility

This document provides guidelines for making experiments in `CAE_GAN_Vectorization` reproducible.

Reproducibility is important for research because it allows other researchers to verify, compare, and extend experimental results.

---

## Recommended Reproducibility Checklist

For every experiment, record:

- Model name
- Script name
- Dataset name
- Dataset path
- Dataset split
- Image resolution
- Number of video frames
- Latent dimension or latent channels
- Batch size
- Number of epochs
- Optimizer
- Learning rate
- Random seed
- Checkpoint path
- Output directory
- Hardware information
- Software versions
- Evaluation metrics

---

## Recommended Output Structure

```bash
runs/
└── experiment_name/
    ├── model_config.json
    ├── metrics.json
    ├── command.txt
    ├── checkpoint.pt
    ├── latents.npy
    ├── latents.h5
    └── reconstruction_samples.png
```

---

## Save the Command Used

For each experiment, save the command that produced the results.

Example:

```bash
python Gan_autoencoder_Cfp.py \
  --dataset_type cfpd \
  --data data/cfpd/Data/Images \
  --out runs/gan_ae_cfpd \
  --epochs 50 \
  --batch_size 128 \
  --latent_ch 16
```

Recommended file:

```bash
runs/gan_ae_cfpd/command.txt
```

---

## Use Random Seeds

Use fixed random seeds when possible.

Example:

```bash
--seed 42
```

This improves consistency across repeated runs.

---

## Save Configuration Files

Recommended configuration files are stored in:

```bash
configs/
```

Example:

```bash
configs/gan_cfpd.yaml
```

Configuration files should include:

- Dataset settings
- Model settings
- Training settings
- Evaluation settings
- Output paths

---

## Save Model Metadata

Recommended metadata file:

```bash
model_config.json
```

Example content:

```json
{
  "model": "gan_autoencoder",
  "dataset": "CFPD",
  "image_size": 64,
  "latent_channels": 16,
  "epochs": 50,
  "batch_size": 128,
  "learning_rate": 0.0002
}
```

---

## Save Evaluation Results

Recommended file:

```bash
metrics.json
```

Example content:

```json
{
  "psnr": 28.5,
  "ssim": 0.91,
  "cosine_similarity": 0.87,
  "euclidean_distance": 1.42
}
```

---

## Track Software Versions

Record the versions of key packages:

```bash
python --version
pip freeze > requirements_experiment.txt
```

Recommended file:

```bash
runs/experiment_name/requirements_experiment.txt
```

---

## Do Not Commit Large Outputs

Do not commit:

- Large datasets
- Model checkpoints
- Raw video files
- Large latent arrays
- Generated output folders

Use `.gitignore`:

```gitignore
data/*
!data/README.md
runs/*
!runs/README.md
*.pt
*.pth
*.h5
*.npy
*.npz
```

---

## Recommended Research Report Format

| Field | Value |
|---|---|
| Model | GAN-AE |
| Dataset | CFPD |
| Image size | 64 x 64 |
| Latent channels | 16 |
| Epochs | 50 |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 0.0002 |
| PSNR | TBD |
| SSIM | TBD |
| Compression ratio | TBD |
