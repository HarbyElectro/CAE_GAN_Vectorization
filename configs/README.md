# Configurations

This folder contains experiment configuration files for the models in this repository.

Each configuration file stores dataset paths, model settings, training parameters, evaluation settings, and output locations.

These files are useful for reproducible research because they document the exact settings used for each experiment.

## Files

- `cae_celeba.yaml` — configuration for training the Convolutional Autoencoder on CelebA.
- `gan_cfpd.yaml` — configuration for training the GAN Autoencoder on CFPD-style face data.
- `gan_image_folder.yaml` — configuration for training the GAN Autoencoder on any folder-based image dataset.
- `gan_flowers.yaml` — configuration for training the GAN Autoencoder on Oxford Flowers or similar datasets.
- `video_ucf101.yaml` — configuration for training the Conv + BiLSTM Video Autoencoder.

## Note

The current scripts mainly use command-line arguments. These YAML files are provided to document recommended experiment settings and can be used later if the scripts are extended to support a `--config` argument.
