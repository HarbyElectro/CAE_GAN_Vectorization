import numpy as np


def test_cae_latent_shape():
    latents = np.random.rand(100, 64)

    assert latents.ndim == 2
    assert latents.shape[0] == 100
    assert latents.shape[1] == 64


def test_gan_latent_shape():
    latents = np.random.rand(100, 16, 8, 8)

    assert latents.ndim == 4
    assert latents.shape[0] == 100
    assert latents.shape[1] == 16
    assert latents.shape[2] == 8
    assert latents.shape[3] == 8


def test_flatten_gan_latents():
    latents = np.random.rand(100, 16, 8, 8)
    flattened = latents.reshape(latents.shape[0], -1)

    assert flattened.shape == (100, 16 * 8 * 8)


def test_latent_dtype_float16():
    latents = np.random.rand(10, 16, 8, 8).astype(np.float16)

    assert latents.dtype == np.float16
