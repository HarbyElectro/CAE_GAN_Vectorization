import numpy as np
from scipy.spatial.distance import cityblock, euclidean
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def test_cosine_similarity_identical_vectors():
    a = np.array([[1, 0, 0]])
    b = np.array([[1, 0, 0]])

    score = cosine_similarity(a, b)[0][0]

    assert score == 1.0


def test_euclidean_distance():
    a = np.array([0, 0])
    b = np.array([3, 4])

    distance = euclidean(a, b)

    assert distance == 5.0


def test_manhattan_distance():
    a = np.array([1, 2, 3])
    b = np.array([4, 6, 8])

    distance = cityblock(a, b)

    assert distance == 12


def test_psnr_identical_images():
    image = np.ones((64, 64, 3), dtype=np.float32)

    score = peak_signal_noise_ratio(image, image, data_range=1.0)

    assert score == float("inf")


def test_ssim_identical_images():
    image = np.ones((64, 64, 3), dtype=np.float32)

    score = structural_similarity(image, image, channel_axis=-1, data_range=1.0)

    assert score == 1.0
