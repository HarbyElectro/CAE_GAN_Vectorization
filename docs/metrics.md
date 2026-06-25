# Metrics

This document describes the evaluation metrics used in `CAE_GAN_Vectorization`.

The repository supports three main metric categories:

1. Reconstruction metrics
2. Latent similarity metrics
3. GAN discriminator metrics

---

## Reconstruction Metrics

Reconstruction metrics compare the original input image or video frame with the reconstructed output.

---

## PSNR

Peak Signal-to-Noise Ratio measures reconstruction fidelity based on pixel-level error.

Higher PSNR usually indicates better reconstruction quality.

Used for:

- Image reconstruction evaluation
- Video frame reconstruction evaluation
- Latent-size ablation studies

---

## SSIM

Structural Similarity Index Measure evaluates perceived structural similarity between two images.

Higher SSIM means the reconstructed image is more structurally similar to the original.

Used for:

- Image reconstruction evaluation
- Video reconstruction evaluation
- Visual fidelity analysis

---

## MS-SSIM

Multiscale SSIM extends SSIM by evaluating similarity at multiple image scales.

It is useful for evaluating reconstruction quality across different spatial resolutions.

---

## MSE

Mean Squared Error measures the average squared pixel difference between original and reconstructed images.

Lower MSE means smaller reconstruction error.

---

## MAE

Mean Absolute Error measures the average absolute pixel difference between original and reconstructed images.

Lower MAE means better reconstruction.

---

# Latent Similarity Metrics

Latent similarity metrics compare encoded representations in latent space.

They are useful for semantic retrieval and nearest-neighbor search.

---

## Cosine Similarity

Cosine similarity measures the angle between two latent vectors.

Higher cosine similarity means the vectors are more similar in direction.

Used for:

- Image retrieval
- Similarity search
- Latent-space clustering

---

## Euclidean Distance

Euclidean distance measures straight-line distance between two latent vectors.

Lower distance means higher similarity.

---

## Manhattan Distance

Manhattan distance measures the sum of absolute differences across latent dimensions.

Lower distance means higher similarity.

---

# GAN Discriminator Metrics

The GAN-AE model can evaluate real and reconstructed samples using discriminator predictions.

---

## Precision

Precision measures how many samples predicted as real are actually real.

---

## Recall

Recall measures how many real samples are correctly detected as real.

---

## Accuracy

Accuracy measures the overall classification correctness of real and reconstructed samples.

---

# Recommended Metric Table

| Metric | Category | Better Value | Used For |
|---|---|---|---|
| PSNR | Reconstruction | Higher | Image/video fidelity |
| SSIM | Reconstruction | Higher | Structural similarity |
| MS-SSIM | Reconstruction | Higher | Multiscale structural similarity |
| MSE | Reconstruction | Lower | Pixel-level error |
| MAE | Reconstruction | Lower | Pixel-level error |
| Cosine similarity | Latent similarity | Higher | Retrieval |
| Euclidean distance | Latent similarity | Lower | Retrieval |
| Manhattan distance | Latent similarity | Lower | Retrieval |
| Precision | GAN discriminator | Higher | Real/reconstructed classification |
| Recall | GAN discriminator | Higher | Real/reconstructed classification |
| Accuracy | GAN discriminator | Higher | Real/reconstructed classification |

---

# Reporting Recommendations

For research papers or experiment reports, include:

- Dataset name
- Model type
- Latent dimension
- Compression ratio
- PSNR
- SSIM
- Retrieval metric, if applicable
- Training epochs
- Image size
- Batch size
