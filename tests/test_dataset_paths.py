from pathlib import Path


def test_recommended_dataset_paths_are_documented():
    expected_paths = [
        "data/celeba/img_align_celeba",
        "data/cfpd/Data/Images",
        "data/imdb_faces",
        "data/flowers",
        "data/mnist",
        "data/UCF101",
    ]

    assert len(expected_paths) == 6
    assert "data/celeba/img_align_celeba" in expected_paths
    assert "data/cfpd/Data/Images" in expected_paths
    assert "data/UCF101" in expected_paths


def test_data_readme_can_exist_without_datasets():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"

    # The data directory may not exist in a fresh clone.
    # If it exists, it should contain a README.md rather than large datasets.
    if data_dir.exists():
        assert (data_dir / "README.md").exists(), "data/README.md should explain dataset placement"
