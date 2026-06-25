from pathlib import Path


def test_gan_script_contains_expected_arguments():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "Gan_autoencoder_Cfp.py"

    assert script_path.exists(), "Gan_autoencoder_Cfp.py is missing"

    content = script_path.read_text(encoding="utf-8", errors="ignore")

    expected_terms = [
        "dataset_type",
        "image_folder",
        "cfpd",
        "data",
        "out",
        "epochs",
        "batch_size",
        "latent_ch",
        "latents_path",
        "decode_h5",
        "save_weights_json",
    ]

    missing = [term for term in expected_terms if term not in content]
    assert not missing, f"Missing expected GAN-AE terms: {missing}"
