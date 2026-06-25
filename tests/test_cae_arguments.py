from pathlib import Path


def test_cae_script_contains_expected_arguments():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "CAE_CelebA.py"

    assert script_path.exists(), "CAE_CelebA.py is missing"

    content = script_path.read_text(encoding="utf-8", errors="ignore")

    expected_terms = [
        "data_dir",
        "output_dir",
        "epochs",
        "batch_size",
        "latent_dim",
    ]

    missing = [term for term in expected_terms if term not in content]
    assert not missing, f"Missing expected CAE terms: {missing}"
