import cv2
import numpy as np

from src.preprocessing.grayscale import convert_to_grayscale
from src.preprocessing.filtering import (
    apply_gaussian_blur,
    apply_median_filter,
)
from src.preprocessing.enhancement import (
    apply_histogram_equalization,
)
from src.preprocessing.thresholding import (
    apply_binary_threshold,
    apply_adaptive_threshold,
)


def create_test_image():
    """
    Create a small synthetic test image.
    """

    return np.random.randint(
        0,
        256,
        (100, 100, 3),
        dtype=np.uint8
    )


def test_grayscale_conversion():
    image = create_test_image()

    result = convert_to_grayscale(image)

    assert len(result.shape) == 2
    assert result.shape == (100, 100)


def test_gaussian_blur():
    image = create_test_image()

    result = apply_gaussian_blur(image)

    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_median_filter():
    image = create_test_image()

    result = apply_median_filter(image)

    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_histogram_equalization():
    image = create_test_image()

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    result = apply_histogram_equalization(
        grayscale
    )

    assert result.shape == grayscale.shape
    assert result.dtype == grayscale.dtype


def test_binary_threshold():
    image = create_test_image()

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    result = apply_binary_threshold(
        grayscale
    )

    unique_values = np.unique(result)

    assert set(unique_values).issubset(
        {0, 255}
    )


def test_adaptive_threshold():
    image = create_test_image()

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    result = apply_adaptive_threshold(
        grayscale
    )

    unique_values = np.unique(result)

    assert set(unique_values).issubset(
        {0, 255}
    )