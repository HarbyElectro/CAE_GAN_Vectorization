# Tests

This folder contains lightweight tests for the `CAE_GAN_Vectorization` repository.

The tests are designed to check:

- Whether the main scripts exist
- Whether important command-line arguments are present
- Whether dataset folder assumptions are documented
- Whether latent-vector shapes are valid
- Whether basic reconstruction and similarity metrics work correctly

These tests do not require full datasets or long model training.

---

## Test Files

- `test_imports.py` — checks that the main project scripts exist.
- `test_dataset_paths.py` — checks expected dataset path names.
- `test_cae_arguments.py` — checks important CAE script arguments.
- `test_gan_arguments.py` — checks important GAN-AE script arguments.
- `test_video_arguments.py` — checks important Video-AE script arguments.
- `test_latent_shapes.py` — checks expected latent-vector shapes using fake arrays.
- `test_metrics.py` — checks reconstruction and similarity metric calculations.

---

## Running Tests

Install pytest:

```bash
pip install pytest
```

Run all tests from the repository root:

```bash
pytest tests/
```

Run one test file:

```bash
pytest tests/test_metrics.py
```

---

## Notes

These tests are intentionally lightweight.

They are meant to verify the repository structure and basic research utilities before running large training experiments.
