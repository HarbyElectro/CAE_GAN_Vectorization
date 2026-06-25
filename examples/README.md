# Example Shell Scripts

This folder contains ready-to-run shell scripts for common experiments in `CAE_GAN_Vectorization`.

Before running any script, update the dataset paths if needed.

## Available Scripts

- `train_cae_celeba.sh` — train the CAE model on CelebA.
- `train_cae_quick_test.sh` — quick CAE smoke test.
- `train_gan_cfpd.sh` — train GAN-AE on a CFPD-style dataset.
- `train_gan_celeba.sh` — train GAN-AE on CelebA using image-folder mode.
- `train_gan_flowers.sh` — train GAN-AE on Oxford Flowers or a similar image folder.
- `export_latents_imdb.sh` — export latent vectors for IMDb Faces.
- `store_latents_h5_flowers.sh` — store latent vectors in HDF5 format.
- `decode_latents_h5.sh` — reconstruct images from stored HDF5 latents.
- `resume_gan_training.sh` — resume GAN-AE from a checkpoint.
- `save_gan_weights_json.sh` — save GAN-AE configuration and weights as JSON.
- `train_video_ucf101.sh` — train the video autoencoder.
- `run_all_quick_tests.sh` — run selected smoke tests.

## Usage

From the repository root:

```bash
bash examples/train_cae_celeba.sh
```

or make executable and run:

```bash
chmod +x examples/train_cae_celeba.sh
./examples/train_cae_celeba.sh
```
