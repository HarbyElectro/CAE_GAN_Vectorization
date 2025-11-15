#!/usr/bin/env python3
"""
Convolutional Autoencoder for CelebA (or similar folder-based image datasets).

- Loads images from a directory with subfolders (one per class or split).
- Trains a convolutional autoencoder on 64x64 RGB images.
- Saves:
  - Trained autoencoder weights
  - Encoder and decoder model plots
  - Encoded latent representations as a NumPy file

Usage example:
    python train_celeba_cae.py \
        --data_dir archive/img_align_celeba \
        --output_dir runs/celeba_cae \
        --epochs 100 \
        --batch_size 64
"""

import os
import argparse

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Dense,
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def read_celeba_images(dataset_dir, image_size=(64, 64), max_images=None):
    """
    Read images from a CelebA-style directory structure.

    Parameters
    ----------
    dataset_dir : str
        Root directory containing image files (optionally in subfolders).
    image_size : tuple(int, int)
        Target (width, height) for resizing images.
    max_images : int or None
        Optional limit on the number of images to load.

    Returns
    -------
    X : np.ndarray
        Array of images of shape (N, H, W, 3) in uint8.
    labels : np.ndarray
        Array of labels (subfolder names) of length N.
    """
    images = []
    labels = []

    for subdir in os.listdir(dataset_dir):
        subdir_path = os.path.join(dataset_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        for filename in os.listdir(subdir_path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(subdir_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_size)

            images.append(img)
            labels.append(subdir)

            if max_images is not None and len(images) >= max_images:
                break

        if max_images is not None and len(images) >= max_images:
            break

    X = np.array(images)
    labels = np.array(labels)
    return X, labels


# ---------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------
def build_autoencoder(input_shape=(64, 64, 3), latent_dim=64):
    """
    Build a simple convolutional autoencoder.

    Encoder:
      Conv2D(256) -> MaxPool
      Conv2D(128) -> MaxPool
      Conv2D(64)  -> MaxPool
      Dense(latent_dim)

    Decoder:
      Conv2D(64)  -> UpSampling
      Conv2D(128) -> UpSampling
      Conv2D(256) -> UpSampling
      Conv2D(3)   (sigmoid)

    Parameters
    ----------
    input_shape : tuple
        Shape of the input image.
    latent_dim : int
        Size of the dense latent bottleneck.

    Returns
    -------
    autoencoder : Model
    encoder : Model
    decoder : Model
    """
    # Encoder
    input_img = Input(shape=input_shape, name="input_image")
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(input_img)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    # Dense bottleneck on the 8x8x64 feature map
    encoded = Dense(latent_dim, activation="relu", name="latent_dense")(x)

    # Decoder
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation="sigmoid", padding="same", name="reconstruction")(x)

    autoencoder = Model(input_img, decoded, name="conv_autoencoder")

    # Encoder model (from input to latent)
    encoder = Model(inputs=input_img, outputs=encoded, name="encoder")

    # Decoder model (from latent to reconstruction)
    latent_input = Input(shape=encoded.shape[1:], name="latent_input")
    dec_x = autoencoder.layers[-4](latent_input)  # Conv2D(64)
    dec_x = autoencoder.layers[-3](dec_x)        # UpSampling2D
    dec_x = autoencoder.layers[-2](dec_x)        # Conv2D(128) or 256 depending on layer order
    dec_x = autoencoder.layers[-1](dec_x)        # final Conv2D(3)
    decoder = Model(inputs=latent_input, outputs=dec_x, name="decoder")

    return autoencoder, encoder, decoder


# ---------------------------------------------------------------------
# Training script
# ---------------------------------------------------------------------
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Loading images from: {args.data_dir}")
    X, labels = read_celeba_images(
        dataset_dir=args.data_dir,
        image_size=(args.image_size, args.image_size),
        max_images=args.max_images,
    )

    print(f"[INFO] Loaded {len(X)} images.")
    print(f"[INFO] Unique labels: {np.unique(labels)}")

    # Normalize to [0, 1]
    X = X.astype("float32") / 255.0

    # Train/test split
    x_train, x_test = train_test_split(
        X,
        test_size=args.test_split,
        train_size=1.0 - args.test_split,
        random_state=42,
        shuffle=True,
    )

    print(f"[INFO] Train set: {x_train.shape}, Test set: {x_test.shape}")

    # Build model
    autoencoder, encoder, decoder = build_autoencoder(
        input_shape=(args.image_size, args.image_size, 3),
        latent_dim=args.latent_dim,
    )

    autoencoder.compile(optimizer="adam", loss="mse")
    print(autoencoder.summary())

    # Callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=5,
        min_lr=1e-5,
        verbose=1,
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=1,
        restore_best_weights=True,
    )

    # Plot model graphs
    plot_model(
        autoencoder,
        to_file=os.path.join(args.output_dir, "autoencoder.png"),
        show_shapes=True,
        show_layer_names=True,
    )
    plot_model(
        encoder,
        to_file=os.path.join(args.output_dir, "encoder.png"),
        show_shapes=True,
        show_layer_names=True,
    )
    plot_model(
        decoder,
        to_file=os.path.join(args.output_dir, "decoder.png"),
        show_shapes=True,
        show_layer_names=True,
    )

    # Train
    history = autoencoder.fit(
        x=x_train,
        y=x_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test, x_test),
        callbacks=[reduce_lr, early_stop],
        verbose=1,
    )

    # Save weights
    weights_path = os.path.join(args.output_dir, "autoencoder_weights.h5")
    autoencoder.save_weights(weights_path)
    print(f"[INFO] Saved autoencoder weights to: {weights_path}")

    # Encode training images
    print("[INFO] Encoding training set with encoder...")
    encoded_train = encoder.predict(x_train, batch_size=args.batch_size)
    latents_path = os.path.join(args.output_dir, "celeba_encoded_images.npy")
    np.save(latents_path, encoded_train)
    print(f"[INFO] Saved encoded representations to: {latents_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a convolutional autoencoder on CelebA-style image folders."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="archive/img_align_celeba",
        help="Path to dataset root directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/celeba_cae",
        help="Directory to store outputs (weights, plots, latents).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        help="Image size (images are resized to image_size x image_size).",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional maximum number of images to load (for quick experiments).",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Fraction of data used for testing.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size.",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=64,
        help="Dimensionality of the latent dense bottleneck.",
    )

    args = parser.parse_args()
    main(args)
