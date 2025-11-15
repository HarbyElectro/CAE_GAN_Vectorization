# -*- coding: utf-8 -*-
"""
GAN-style autoencoder for generic image datasets (CFPD or ImageFolder).

Features:
- Generic dataset support via --dataset_type (cfpd or image_folder)
- Latent vectors exported as NumPy .npy
- Model architecture and learned weights saved to JSON
- Optional HDF5 latent export for compression analysis

Original author: Ahmed Harby
Updated for generic usage and GitHub release
"""

import os, argparse, random, math, json
from pathlib import Path
import h5py
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import utils as vutils, datasets, transforms

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cityblock, euclidean
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ============================================================
#  CFPD DATASET LOADER
# ============================================================
def load_cfpd_dataset(dataset_dir, target_size=(64, 64)):
    """
    Load CFPD dataset from the given directory.

    Expected structure:
        dataset_dir/
            person_1/
                frontal/*.jpg|.png
                other_view_files.jpg|.png
            person_2/
                frontal/*.jpg|.png
                ...

    frontal    -> label 1
    all other  -> label 0
    """
    images = []
    labels = []

    for subdir in os.listdir(dataset_dir):
        sub_dir_path = os.path.join(dataset_dir, subdir)
        if not os.path.isdir(sub_dir_path):
            continue

        frontal_dir_path = os.path.join(sub_dir_path, "frontal")
        if os.path.isdir(frontal_dir_path):
            label = 1
            for file in os.listdir(frontal_dir_path):
                file_path = os.path.join(frontal_dir_path, file)
                if os.path.isfile(file_path) and file.lower().endswith((".jpg", ".png")):
                    image = cv2.imread(file_path)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, target_size)
                        images.append(image)
                        labels.append(label)
        else:
            label = 0
            for file in os.listdir(sub_dir_path):
                file_path = os.path.join(sub_dir_path, file)
                if os.path.isfile(file_path) and file.lower().endswith((".jpg", ".png")):
                    image = cv2.imread(file_path)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, target_size)
                        images.append(image)
                        labels.append(label)

    return np.array(images), np.array(labels)


class CFPDDataset(Dataset):
    """Simple torch Dataset wrapper for CFPD numpy arrays."""
    def __init__(self, images, labels):
        self.images = images.astype(np.float32) / 255.0  # [0,1]
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]  # HWC, [0,1]
        x = torch.tensor(x).permute(2, 0, 1) * 2 - 1     # CHW, [-1,1]
        y = int(self.labels[idx])
        return x, y


# ============================================================
# UTILS
# ============================================================
def set_seed(s=0):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def denorm_01(x: torch.Tensor) -> torch.Tensor:
    """Map [-1,1] tensor to [0,1]."""
    return x.add(1).div(2).clamp(0, 1)


@torch.no_grad()
def save_row5(tensor_imgs, out_path):
    """Save a single row of up to 5 images in a grid."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    n = min(5, tensor_imgs.size(0))
    grid = vutils.make_grid(
        tensor_imgs[:n], nrow=n, normalize=True, value_range=(-1, 1)
    )
    vutils.save_image(grid, out_path)


# ============================================================
# METRICS
# ============================================================
@torch.no_grad()
def compute_metrics_batch(x, xhat, n=5):
    n = min(n, x.size(0))
    x01 = denorm_01(x[:n]).permute(0, 2, 3, 1).cpu().numpy()
    xh01 = denorm_01(xhat[:n]).permute(0, 2, 3, 1).cpu().numpy()

    cos_total = man_total = euc_total = ssi_total = psnr_total = msssim_total = 0.0
    for i in range(n):
        a, b = x01[i], xh01[i]
        af = a.reshape(-1)
        bf = b.reshape(-1)
        cos_total += float(cosine_similarity(af.reshape(1, -1), bf.reshape(1, -1))[0, 0])
        man_total += float(cityblock(af, bf))
        euc_total += float(euclidean(af, bf))
        ssi_total += float(ssim(a, b, data_range=1.0, channel_axis=-1, win_size=3))
        psnr_total += float(psnr(a, b, data_range=1.0))
        msssim_total += float(ssim(a, b, data_range=1.0, channel_axis=-1, win_size=3, multiscale=True))

    return {
        "count": n,
        "cosine_similarity_avg": cos_total / n,
        "manhattan_distance_avg": man_total / n,
        "euclidean_distance_avg": euc_total / n,
        "ssim_avg": ssi_total / n,
        "psnr_avg": psnr_total / n,
        "msssim_avg": msssim_total / n,
    }


def append_metrics_to_file(path, metrics, epoch, step):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a") as f:
        f.write(f"[epoch {epoch:03d} | step {step:07d}]\n")
        for k, v in metrics.items():
            if k != "count":
                f.write(f"{k}: {v}\n")
        f.write("=" * 80 + "\n")


# ============================================================
# Discriminator PR metrics
# ============================================================
@torch.no_grad()
def pr_from_discriminator(D, real_batch, fake_batch, threshold=0.5):
    real_logits = D(real_batch)
    fake_logits = D(fake_batch)

    real_pred = (real_logits >= threshold)
    fake_pred = (fake_logits >= threshold)

    TP = real_pred.sum().item()
    FN = (~real_pred).sum().item()
    FP = fake_pred.sum().item()
    TN = (~fake_pred).sum().item()

    precision_real = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall_real = TP / (TP + FN) if TP + FN > 0 else 0.0
    acc_D = (TP + TN) / (TP + TN + FP + FN)

    real_pred_f = (real_logits < threshold)
    fake_pred_f = (fake_logits < threshold)
    TP_f = fake_pred_f.sum().item()
    FN_f = (~fake_pred_f).sum().item()
    FP_f = real_pred_f.sum().item()

    precision_fake = TP_f / (TP_f + FP_f) if TP_f + FP_f > 0 else 0.0
    recall_fake = TP_f / (TP_f + FN_f) if TP_f + FN_f > 0 else 0.0

    return {
        "acc_D": acc_D,
        "precision_real": precision_real,
        "recall_real": recall_real,
        "precision_fake": precision_fake,
        "recall_fake": recall_fake,
    }


# ============================================================
# MODELS
# ============================================================
class Generator(nn.Module):
    """
    GAN-style autoencoder generator (encoder + decoder).
    """

    def __init__(self, in_ch=3, base_ch=64, latent_ch=32):
        super().__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.latent_ch = latent_ch

        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(base_ch),

            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(base_ch * 2),

            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(base_ch * 4),

            nn.Conv2d(base_ch * 4, latent_ch, 4, 2, 1, bias=False),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, base_ch * 4, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_ch * 4),

            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_ch * 2),

            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_ch),

            nn.ConvTranspose2d(base_ch, in_ch, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        return self.decode(self.encode(x))


class Discriminator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).view(-1)


# ============================================================
# HDF5 Latent Storage
# ============================================================
@torch.no_grad()
def write_latents_h5(encoder, data_loader, out_path, n_items, device,
                     dtype="float16", store_images=False, gzip_level=4):
    """
    Encodes first n_items from data_loader and writes them to HDF5.
    """
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    x0, _ = next(iter(data_loader))
    C, H, W = x0.shape[1:]
    z0 = encoder(x0.to(device)).cpu()
    C_lat, H_lat, W_lat = z0.shape[1:]
    raw_image_bytes_per_item = C * H * W

    np_dtype = np.float16 if dtype == "float16" else np.float32
    n_written = 0

    with h5py.File(out_path, "w") as f:
        d_lat = f.create_dataset(
            "latents",
            shape=(n_items, C_lat, H_lat, W_lat),
            dtype=np_dtype,
            compression="gzip",
            compression_opts=gzip_level,
            chunks=True,
        )
        if store_images:
            d_img = f.create_dataset(
                "images_uint8",
                shape=(n_items, H, W, C),
                dtype=np.uint8,
                compression="gzip",
                compression_opts=gzip_level,
                chunks=True,
            )

        for xb, _ in data_loader:
            xb = xb.to(device)
            zb = encoder(xb).cpu().numpy().astype(np_dtype)
            bs = zb.shape[0]
            take = min(bs, n_items - n_written)
            d_lat[n_written:n_written + take] = zb[:take]
            if store_images:
                imgs01 = denorm_01(xb).permute(0, 2, 3, 1).cpu().numpy()
                imgs_u8 = (np.clip(imgs01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
                d_img[n_written:n_written + take] = imgs_u8[:take]
            n_written += take
            if n_written >= n_items:
                break

    file_size_bytes = out_path.stat().st_size
    raw_image_bytes = raw_image_bytes_per_item * n_written
    return n_written, (C_lat, H_lat, W_lat), file_size_bytes, raw_image_bytes


@torch.no_grad()
def decode_from_h5(decoder, h5_path, out_dir, n=5, device="cuda"):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    with h5py.File(h5_path, "r") as f:
        lat = torch.from_numpy(f["latents"][:n]).to(device)
    recon = decoder(lat)
    save_row5(recon, out_dir / "decoded_row5.png")


# ============================================================
# NumPy latent export
# ============================================================
@torch.no_grad()
def export_latents_npy(encoder, data_loader, out_path, n_items, device):
    """
    Export first n_items latents to a NumPy .npy file.
    Shape: (N, C_lat, H_lat, W_lat)
    """
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    encoder.eval()
    latents = []
    count = 0
    for xb, _ in data_loader:
        xb = xb.to(device)
        zb = encoder(xb).cpu().numpy()
        bs = zb.shape[0]
        take = min(bs, n_items - count)
        latents.append(zb[:take])
        count += take
        if count >= n_items:
            break

    if count == 0:
        return 0, None

    latents_arr = np.concatenate(latents, axis=0)
    np.save(out_path, latents_arr)
    return count, latents_arr.shape


# ============================================================
# Save model config and weights as JSON
# ============================================================
def save_model_config_and_weights_json(G, D, args, out_dir):
    """
    Save:
    - model_config.json: hyperparameters and architecture config
    - model_weights.json: state_dict weights as nested lists (can be large)
    """

    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    config = {
        "generator": {
            "in_ch": G.in_ch,
            "base_ch": G.base_ch,
            "latent_ch": G.latent_ch,
        },
        "discriminator": {
            "in_ch": D.in_ch,
            "base_ch": D.base_ch,
        },
        "training_args": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lambda_rec": args.lambda_rec,
            "image_size": args.image_size,
            "dataset_type": args.dataset_type,
            "data_root": args.data,
        },
    }

    with open(out_dir / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    if args.save_weights_json:
        gen_state = {k: v.cpu().tolist() for k, v in G.state_dict().items()}
        disc_state = {k: v.cpu().tolist() for k, v in D.state_dict().items()}
        weights = {
            "generator": gen_state,
            "discriminator": disc_state,
        }
        with open(out_dir / "model_weights.json", "w") as f:
            json.dump(weights, f)


# ============================================================
# DATASET BUILDER
# ============================================================
def build_dataloaders(args):
    """
    Build train and validation DataLoaders for either:
    - CFPD dataset (custom loader), or
    - generic ImageFolder dataset.
    """
    if args.dataset_type == "cfpd":
        print(f"Loading CFPD dataset from {args.data} ...")
        imgs, labels = load_cfpd_dataset(
            args.data, target_size=(args.image_size, args.image_size)
        )
        split = int(0.8 * len(imgs))
        train_imgs, train_labels = imgs[:split], labels[:split]
        val_imgs, val_labels = imgs[split:], labels[split:]

        train_set = CFPDDataset(train_imgs, train_labels)
        valid_set = CFPDDataset(val_imgs, val_labels)

    elif args.dataset_type == "image_folder":
        print(f"Loading ImageFolder dataset from {args.data} ...")
        tfm = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),                       # [0,1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),  # [-1,1]
        ])
        full_ds = datasets.ImageFolder(root=args.data, transform=tfm)
        split = int(0.8 * len(full_ds))
        train_set, valid_set = random_split(
            full_ds, [split, len(full_ds) - split],
            generator=torch.Generator().manual_seed(args.seed)
        )
    else:
        raise ValueError(f"Unknown dataset_type: {args.dataset_type}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_set, batch_size=args.eval_batch,
        shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, valid_loader


# ============================================================
# TRAINING
# ============================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    # Decode-only mode from HDF5
    if args.decode_h5:
        G = Generator(in_ch=3, base_ch=args.base_ch, latent_ch=args.latent_ch).to(device)
        if not args.ckpt:
            raise ValueError("--decode_h5 requires --ckpt to load model weights.")
        ckpt = torch.load(args.ckpt, map_location=device)
        G.load_state_dict(ckpt["G"])
        G.eval()
        decode_from_h5(G.decode, args.decode_h5, args.out, n=args.decode_n, device=device)
        print(f"Decoded a single row of {args.decode_n} images into {args.out}")
        return

    # Data
    train_loader, valid_loader = build_dataloaders(args)

    # Models
    G = Generator(in_ch=3, base_ch=args.base_ch, latent_ch=args.latent_ch).to(device)
    D = Discriminator(in_ch=3, base_ch=args.base_ch).to(device)

    # Optimizers and loss
    g_opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCELoss()

    out_dir = Path(args.out)
    ensure_dir(out_dir)
    metrics_txt = out_dir / "image_comparison_results.txt"

    step = 0
    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()
        for x, _ in train_loader:
            x = x.to(device)

            # Train D
            d_opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                xhat_det = G(x).detach()
            real_logits = D(x)
            fake_logits = D(xhat_det)
            d_loss = bce(real_logits, torch.ones_like(real_logits)) + \
                     bce(fake_logits, torch.zeros_like(fake_logits))
            d_loss.backward()
            d_opt.step()

            # Train G
            g_opt.zero_grad(set_to_none=True)
            xhat = G(x)
            fake_logits_g = D(xhat)
            rec_loss = F.l1_loss(xhat, x)
            adv_loss = bce(fake_logits_g, torch.ones_like(fake_logits_g))
            g_loss = args.lambda_rec * rec_loss + (1.0 - args.lambda_rec) * adv_loss
            g_loss.backward()
            g_opt.step()

            if step % args.sample_every == 0:
                save_row5(xhat, out_dir / f"train_recon_row5_step{step:07d}.png")

            if step % args.log_every == 0:
                print(
                    f"[epoch {epoch:03d} | step {step:07d}] "
                    f"D: {d_loss.item():.4f} | G: {g_loss.item():.4f} "
                    f"(rec {rec_loss.item():.4f}, adv {adv_loss.item():.4f})"
                )

            step += 1

        # Validation
        G.eval()
        with torch.no_grad():
            val_x, _ = next(iter(valid_loader))
            val_x = val_x.to(device)
            val_xhat = G(val_x)

            save_row5(val_xhat, out_dir / f"val_recon_row5_epoch{epoch:03d}.png")

            metrics = compute_metrics_batch(val_x, val_xhat, n=args.eval_n)
            metrics.update(pr_from_discriminator(D, val_x, val_xhat, args.pr_threshold))

            if not args.disable_h5:
                lat_written, latent_shape, file_bytes, raw_bytes = write_latents_h5(
                    G.encode, valid_loader, args.latents_path, args.latents_n, device,
                    dtype=args.latents_dtype, store_images=args.store_images_in_h5,
                    gzip_level=args.h5_gzip
                )
                if raw_bytes > 0 and file_bytes > 0:
                    reduction_percent = 100.0 * (1.0 - file_bytes / float(raw_bytes))
                    compression_ratio = raw_bytes / float(file_bytes)
                else:
                    reduction_percent = float("nan")
                    compression_ratio = float("inf")

                metrics.update({
                    "latent_h5_items": lat_written,
                    "latent_file_bytes": file_bytes,
                    "raw_image_bytes": raw_bytes,
                    "file_size_reduction_percent": reduction_percent,
                    "file_size_compression_ratio": compression_ratio,
                    "latent_shape_saved": latent_shape,
                    "latents_dtype": args.latents_dtype,
                })

            append_metrics_to_file(metrics_txt, metrics, epoch, step)
            print(f"[epoch {epoch:03d}] metrics:", metrics)

        # Checkpoint
        if args.ckpt_every and epoch % args.ckpt_every == 0:
            torch.save(
                {
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "g_opt": g_opt.state_dict(),
                    "d_opt": d_opt.state_dict(),
                    "epoch": epoch,
                    "step": step,
                },
                out_dir / f"ckpt_epoch{epoch:03d}.pt",
            )

    # ========================================================
    # After training: export latents (.npy) and JSON model
    # ========================================================
    print("Exporting latents to NumPy .npy ...")
    n_lat, lat_shape = export_latents_npy(
        G.encode, valid_loader, out_dir / "latents.npy", args.latents_n, device
    )
    print(f"Saved {n_lat} latents to latents.npy with shape {lat_shape}")

    print("Saving model config and weights to JSON ...")
    save_model_config_and_weights_json(G, D, args, out_dir)
    print("Training complete. Outputs saved in:", out_dir.resolve())


# ============================================================
# ARGUMENT PARSER
# ============================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GAN-style autoencoder trainer")

    # Dataset
    p.add_argument(
        "--dataset_type",
        type=str,
        default="cfpd",
        choices=["cfpd", "image_folder"],
        help="Dataset type: 'cfpd' for CFPD layout or 'image_folder' for generic ImageFolder.",
    )
    p.add_argument(
        "--data",
        type=str,
        default="./cfp-dataset/Data/Images",
        help="Root directory for the dataset.",
    )

    # Output
    p.add_argument("--out", type=str, default="./runs/gan_ae_generic",
                   help="Output directory for logs, images, and checkpoints.")

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda_rec", type=float, default=0.6,
                   help="Weight for reconstruction loss in generator objective.")
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--latent_ch", type=int, default=16)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")

    # Logging and sampling
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--sample_every", type=int, default=500)

    # Evaluation
    p.add_argument("--eval_batch", type=int, default=16)
    p.add_argument("--eval_n", type=int, default=5)
    p.add_argument("--pr_threshold", type=float, default=0.5)

    # HDF5 latent export
    p.add_argument("--latents_path", type=str, default="./runs/latents.h5")
    p.add_argument("--latents_n", type=int, default=64)
    p.add_argument("--latents_dtype", type=str, default="float16",
                   choices=["float16", "float32"])
    p.add_argument("--disable_h5", action="store_true")
    p.add_argument("--store_images_in_h5", action="store_true")
    p.add_argument("--h5_gzip", type=int, default=4)

    # Decode-from-H5 mode
    p.add_argument("--decode_h5", type=str, default="",
                   help="Path to HDF5 file with /latents dataset for decode-only mode.")
    p.add_argument("--decode_n", type=int, default=5)
    p.add_argument("--ckpt", type=str, default="",
                   help="Path to checkpoint .pt when using --decode_h5.")

    # JSON export
    p.add_argument(
        "--save_weights_json",
        action="store_true",
        help="Also save full model weights as JSON (can be large).",
    )

    # Checkpoints
    p.add_argument("--ckpt_every", type=int, default=1)

    args = p.parse_args()
    main(args)
