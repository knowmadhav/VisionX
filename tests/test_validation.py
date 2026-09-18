import numpy as np
import pytest

from src.utils.validators import (
    validate_image_dimensions,
    validate_image_path,
    validate_minimum_area,
)


def test_valid_image_path(tmp_path):
    image_path = tmp_path / "sample.jpg"

    image_path.write_bytes(
        b"test image file"
    )

    validate_image_path(
        str(image_path)
    )


def test_missing_image_path():
    with pytest.raises(FileNotFoundError):
        validate_image_path(
            "images/does_not_exist.jpg"
        )


def test_unsupported_image_format(tmp_path):
    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        "not an image"
    )

    with pytest.raises(ValueError):
        validate_image_path(
            str(file_path)
        )


def test_directory_is_rejected(tmp_path):
    directory = tmp_path / "images"

    directory.mkdir()

    with pytest.raises(ValueError):
        validate_image_path(
            str(directory)
        )


def test_valid_image_dimensions():
    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8
    )

    validate_image_dimensions(
        image
    )


def test_none_image_is_rejected():
    with pytest.raises(ValueError):
        validate_image_dimensions(
            None
        )


def test_empty_image_is_rejected():
    image = np.array(
        [],
        dtype=np.uint8
    )

    with pytest.raises(ValueError):
        validate_image_dimensions(
            image
        )


def test_invalid_image_dimensions_are_rejected():
    image = np.array(
        [1, 2, 3],
        dtype=np.uint8
    )

    with pytest.raises(ValueError):
        validate_image_dimensions(
            image
        )


def test_valid_minimum_area():
    validate_minimum_area(500)


def test_zero_minimum_area_is_valid():
    validate_minimum_area(0)


def test_negative_minimum_area_is_rejected():
    with pytest.raises(ValueError):
        validate_minimum_area(-10)