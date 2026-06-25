# Notebooks

This folder contains research-oriented Jupyter notebooks for analyzing outputs from `CAE_GAN_Vectorization`.

## Notebook Files

- `latent_space_visualization.ipynb`  
  Visualizes exported latent vectors using PCA and t-SNE.

- `reconstruction_comparison.ipynb`  
  Compares original and reconstructed images using PSNR and SSIM.

- `retrieval_demo.ipynb`  
  Demonstrates simple nearest-neighbor retrieval using latent vectors.

- `compression_analysis.ipynb`  
  Estimates compression ratio by comparing raw image size with latent-vector size.

## Usage

From the repository root, start Jupyter:

```bash
jupyter notebook
```

Then open any notebook from the `notebooks/` folder.

Before running, update file paths such as:

```python
LATENTS_PATH = Path("../runs/gan_ae_cfpd/latents.npy")
```

to match your actual experiment output.
