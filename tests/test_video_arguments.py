from pathlib import Path


def test_video_script_contains_expected_arguments():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "Video_CAE.py"

    assert script_path.exists(), "Video_CAE.py is missing"

    content = script_path.read_text(encoding="utf-8", errors="ignore")

    expected_terms = [
        "latent_sizes",
        "num_frames",
        "num_train",
        "num_test",
        "epochs",
        "batch_size",
    ]

    missing = [term for term in expected_terms if term not in content]
    assert not missing, f"Missing expected Video-AE terms: {missing}"
