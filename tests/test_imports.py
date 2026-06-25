from pathlib import Path


def test_main_scripts_exist():
    repo_root = Path(__file__).resolve().parents[1]

    assert (repo_root / "CAE_CelebA.py").exists(), "CAE_CelebA.py is missing"
    assert (repo_root / "Gan_autoencoder_Cfp.py").exists(), "Gan_autoencoder_Cfp.py is missing"
    assert (repo_root / "Video_CAE.py").exists(), "Video_CAE.py is missing"


def test_readme_exists():
    repo_root = Path(__file__).resolve().parents[1]

    assert (repo_root / "README.md").exists(), "README.md is missing"
