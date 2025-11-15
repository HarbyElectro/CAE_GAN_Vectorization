#!/usr/bin/env python3
"""
BiLSTM Video Autoencoder for Sequence Compression and Reconstruction

This script trains a Conv + BiLSTM-based autoencoder on video frame sequences.
It compares reconstruction performance across multiple latent space sizes and
computes PSNR and SSIM for each configuration.

By default, it uses dummy random data. To use real videos, replace the
`generate_dummy_data` function with a call to `load_video` or a custom loader.
"""

import os
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Reshape,
    RepeatVector,
    Bidirectional,
    LSTM,
    TimeDistributed,
    Input,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# -------------------------------------------------------------------
# Model builder
# -------------------------------------------------------------------
def build_autoencoder(img_size, num_frames, channels, latent_space_size):
    """
    Build a Conv + BiLSTM video autoencoder with a given latent size.

    Input shape: (num_frames, img_size, img_size, channels)
    """
    input_layer = Input(shape=(num_frames, img_size, img_size, channels), name="video_input")

    # Encoder
    x = TimeDistributed(Conv2D(32, (3, 3), activation="relu", padding="same"))(input_layer)
    x = TimeDistributed(MaxPooling2D((2, 2), padding="same"))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(0.2))(x)

    x = TimeDistributed(Conv2D(64, (3, 3), activation="relu", padding="same"))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), padding="same"))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(0.2))(x)

    x = TimeDistributed(Conv2D(128, (3, 3), activation="relu", padding="same"))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), padding="same"))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(0.2))(x)

    # Reshape for LSTM
    shape = tf.keras.backend.int_shape(x)  # (None, T, H', W', C')
    _, T, H, W, C = shape
    x = Reshape((T, H * W * C))(x)

    # Temporal encoding
    x = Bidirectional(LSTM(128, return_sequences=False), name="bilstm_encoder")(x)
    x = Dropout(0.2)(x)

    # Latent space
    latent = Dense(latent_space_size, activation="relu", name="latent")(x)

    # Decoder
    y = RepeatVector(T)(latent)
    y = Bidirectional(LSTM(128, return_sequences=True), name="bilstm_decoder")(y)
    y = Dropout(0.5)(y)

    y = TimeDistributed(Dense(H * W * C, activation="relu"))(y)
    y = Reshape((T, H, W, C))(y)

    y = TimeDistributed(UpSampling2D((2, 2)))(y)
    y = TimeDistributed(Conv2D(128, (3, 3), activation="relu", padding="same"))(y)
    y = TimeDistributed(BatchNormalization())(y)

    y = TimeDistributed(UpSampling2D((2, 2)))(y)
    y = TimeDistributed(Conv2D(64, (3, 3), activation="relu", padding="same"))(y)
    y = TimeDistributed(BatchNormalization())(y)

    y = TimeDistributed(UpSampling2D((2, 2)))(y)
    y = TimeDistributed(Conv2D(32, (3, 3), activation="relu", padding="same"))(y)
    output = TimeDistributed(
        Conv2D(channels, (3, 3), activation="sigmoid", padding="same"), name="reconstruction"
    )(y)

    autoencoder = Model(input_layer, output, name=f"video_ae_latent_{latent_space_size}")
    autoencoder.compile(optimizer="adam", loss="mse")

    return autoencoder


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def resize_frames(frames, min_size=7):
    """Ensure each frame is at least min_size x min_size."""
    resized = []
    for frame in frames:
        h, w = frame.shape[:2]
        if h < min_size or w < min_size:
            frame = cv2.resize(frame, (min_size, min_size), interpolation=cv2.INTER_AREA)
        resized.append(frame)
    return np.array(resized)


def calculate_psnr(original, reconstructed):
    """Compute PSNR between two sequences of frames in [0,1]."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 1.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def calculate_ssim_sequence(original, reconstructed):
    """
    Compute mean SSIM over all frames in a video sequence.
    Expects shape (T, H, W, C) with pixel values in [0, 1].
    """
    original = resize_frames(original)
    reconstructed = resize_frames(reconstructed)
    ssim_values = []

    num_frames = min(original.shape[0], reconstructed.shape[0])
    for t in range(num_frames):
        # skimage.ssim returns a scalar score
        score = ssim(
            original[t],
            reconstructed[t],
            channel_axis=-1,
            data_range=1.0,
            win_size=3,
        )
        ssim_values.append(score)

    return float(np.mean(ssim_values))


def crop_center_square(frame):
    """Crop the central square from a frame."""
    y, x = frame.shape[:2]
    min_dim = min(y, x)
    start_x = (x - min_dim) // 2
    start_y = (y - min_dim) // 2
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(224, 224)):
    """
    Load a single video file into a sequence of frames in [0,1], RGB.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]  # BGR -> RGB
            frame = frame.astype("float32") / 255.0
            frames.append(frame)
            if max_frames > 0 and len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)


def generate_dummy_data(num_train, num_test, num_frames, img_size, channels):
    """
    Generate random sequences for quick testing.
    Replace this with real video loading for actual experiments.
    """
    x_train = np.random.rand(num_train, num_frames, img_size, img_size, channels).astype("float32")
    x_test = np.random.rand(num_test, num_frames, img_size, img_size, channels).astype("float32")
    return x_train, x_test


# -------------------------------------------------------------------
# Main training and evaluation loop
# -------------------------------------------------------------------
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Data: replace this with real video loading if desired
    print("[INFO] Generating dummy video data...")
    x_train, x_test = generate_dummy_data(
        num_train=args.num_train,
        num_test=args.num_test,
        num_frames=args.num_frames,
        img_size=args.img_size,
        channels=args.channels,
    )

    latent_space_sizes = [int(s) for s in args.latent_sizes.split(",")]

    losses = {}
    psnr_scores = {}
    ssim_scores = {}

    for latent_space_size in latent_space_sizes:
        print(f"\n[INFO] Training autoencoder with latent space size = {latent_space_size}...")
        autoencoder = build_autoencoder(
            img_size=args.img_size,
            num_frames=args.num_frames,
            channels=args.channels,
            latent_space_size=latent_space_size,
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            verbose=1,
            restore_best_weights=True,
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
        )

        history = autoencoder.fit(
            x_train,
            x_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            shuffle=True,
            validation_data=(x_test, x_test),
            callbacks=[early_stop, reduce_lr],
            verbose=1,
        )

        val_loss_history = history.history["val_loss"]
        losses[latent_space_size] = val_loss_history

        # Evaluate PSNR and SSIM
        psnr_values = []
        ssim_values = []

        for i in range(len(x_test)):
            original = x_test[i]
            reconstructed = autoencoder.predict(x_test[i : i + 1], verbose=0)[0]
            psnr_values.append(calculate_psnr(original, reconstructed))
            ssim_values.append(calculate_ssim_sequence(original, reconstructed))

        psnr_scores[latent_space_size] = float(np.mean(psnr_values))
        ssim_scores[latent_space_size] = float(np.mean(ssim_values))

        print(
            f"[RESULT] Latent {latent_space_size}: "
            f"PSNR={psnr_scores[latent_space_size]:.3f}, "
            f"SSIM={ssim_scores[latent_space_size]:.3f}"
        )

        # Visual comparison for a few samples
        for sample_idx in range(min(3, len(x_test))):
            original = x_test[sample_idx]
            reconstructed = autoencoder.predict(x_test[sample_idx : sample_idx + 1], verbose=0)[0]

            plt.figure(figsize=(12, 4))
            max_frames_to_show = min(args.num_frames, 5)
            for t in range(max_frames_to_show):
                # Original
                plt.subplot(2, max_frames_to_show, t + 1)
                plt.imshow(original[t])
                plt.title("Original")
                plt.axis("off")

                # Reconstructed
                plt.subplot(2, max_frames_to_show, t + 1 + max_frames_to_show)
                plt.imshow(reconstructed[t])
                plt.title("Reconstructed")
                plt.axis("off")

            plt.suptitle(f"Latent {latent_space_size} – Sample {sample_idx}")
            fig_path = os.path.join(
                args.output_dir,
                f"recon_latent{latent_space_size}_sample{sample_idx}.png",
            )
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f"[INFO] Saved reconstruction comparison to {fig_path}")

    # Plot reconstruction losses
    plt.figure(figsize=(10, 6))
    for latent_space_size, val_loss in losses.items():
        plt.plot(val_loss, label=f"Latent {latent_space_size}")
    plt.title("Validation Reconstruction Loss vs. Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss (MSE)")
    plt.legend()
    plt.grid(True)
    loss_fig_path = os.path.join(args.output_dir, "reconstruction_loss_comparison.png")
    plt.savefig(loss_fig_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved loss comparison plot to {loss_fig_path}")

    # Plot PSNR + SSIM comparison
    plt.figure(figsize=(10, 6))
    latent_list = sorted(latent_space_sizes)
    psnr_vals = [psnr_scores[l] for l in latent_list]
    ssim_vals = [ssim_scores[l] for l in latent_list]

    bar_width = 0.35
    indices = np.arange(len(latent_list))

    plt.bar(indices - bar_width / 2, psnr_vals, width=bar_width, label="PSNR")
    plt.bar(indices + bar_width / 2, ssim_vals, width=bar_width, label="SSIM")

    plt.xticks(indices, latent_list)
    plt.title("PSNR and SSIM for Different Latent Space Sizes")
    plt.xlabel("Latent Space Size")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True, axis="y")
    metric_fig_path = os.path.join(args.output_dir, "psnr_ssim_latent_comparison.png")
    plt.savefig(metric_fig_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved PSNR/SSIM comparison plot to {metric_fig_path}")

    # Print average losses
    print("\n[SUMMARY] Average validation loss per latent size:")
    for latent_space_size, val_loss in losses.items():
        avg_loss = float(np.mean(val_loss))
        print(f"  Latent {latent_space_size}: avg val_loss = {avg_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate a BiLSTM video autoencoder with multiple latent sizes."
    )
    parser.add_argument("--img_size", type=int, default=224, help="Frame height/width in pixels.")
    parser.add_argument("--num_frames", type=int, default=30, help="Number of frames per video.")
    parser.add_argument("--channels", type=int, default=3, help="Number of channels (e.g., 3 for RGB).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of training epochs.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on val_loss.")
    parser.add_argument(
        "--latent_sizes",
        type=str,
        default="8,16,32,64,128",
        help="Comma-separated list of latent space sizes to evaluate.",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=100,
        help="Number of training sequences (for dummy data).",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=20,
        help="Number of test sequences (for dummy data).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/video_ae",
        help="Directory to save plots and results.",
    )

    args = parser.parse_args()
    main(args)
