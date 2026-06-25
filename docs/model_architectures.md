# Model Architectures

This document summarizes the main model architectures used in `CAE_GAN_Vectorization`.

---

## 1. Convolutional Autoencoder

Script:

```bash
CAE_CelebA.py
```

Framework:

```bash
TensorFlow / Keras
```

Purpose:

- Image compression
- Image reconstruction
- Latent-vector extraction

General architecture:

```bash
Input Image
   ↓
Convolutional Encoder
   ↓
Dense Latent Bottleneck
   ↓
Convolutional Decoder
   ↓
Reconstructed Image
```

Main components:

- Convolutional layers for spatial feature extraction
- Downsampling to reduce spatial dimensions
- Dense latent bottleneck for compact representation
- Decoder layers for image reconstruction

Research role:

The CAE model provides a baseline for learning compact image representations without adversarial training.

---

## 2. GAN-Guided Autoencoder

Script:

```bash
Gan_autoencoder_Cfp.py
```

Framework:

```bash
PyTorch
```

Purpose:

- Image reconstruction
- Adversarial visual quality improvement
- Latent-vector extraction
- Latent-based retrieval

General architecture:

```bash
Input Image
   ↓
Encoder
   ↓
Latent Tensor
   ↓
Decoder / Generator
   ↓
Reconstructed Image
   ↓
Discriminator Evaluation
```

Main components:

- Encoder for latent feature extraction
- Decoder/generator for reconstruction
- Discriminator for real-vs-reconstructed evaluation
- Reconstruction loss for pixel-level similarity
- Adversarial loss for perceptual realism

Research role:

The GAN-AE model investigates whether adversarial learning can improve reconstruction quality and latent-space usefulness.

---

## 3. Conv + BiLSTM Video Autoencoder

Script:

```bash
Video_CAE.py
```

Framework:

```bash
TensorFlow / Keras
```

Purpose:

- Video sequence compression
- Frame reconstruction
- Temporal representation learning
- Latent-size ablation

General architecture:

```bash
Video Frames
   ↓
Convolutional Feature Extraction
   ↓
Bidirectional LSTM Encoder
   ↓
Latent Sequence Representation
   ↓
Bidirectional LSTM Decoder
   ↓
Reconstructed Video Frames
```

Main components:

- Convolutional feature extractor
- Bidirectional LSTM sequence encoder
- Latent representation layer
- Sequence decoder
- Frame reconstruction output

Research role:

The video autoencoder evaluates how latent-space size affects temporal reconstruction quality.

---

## Architecture Comparison

| Model | Data Type | Framework | Latent Type | Main Use |
|---|---|---|---|---|
| CAE | Images | TensorFlow/Keras | Dense vector | Image compression baseline |
| GAN-AE | Images | PyTorch | Latent tensor | Adversarial reconstruction and retrieval |
| Video-AE | Videos | TensorFlow/Keras | Sequence latent | Video compression and reconstruction |

---

## Suggested Future Architectures

Future versions of this repository can add:

- Variational Autoencoders
- Transformer Autoencoders
- Vision Transformer encoders
- Vector-quantized Autoencoders
- Diffusion-based reconstruction models
- Hybrid CNN-Transformer video autoencoders
