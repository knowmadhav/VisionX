from pathlib import Path

import cv2

from src.utils.validators import (
    validate_image_dimensions,
    validate_image_path,
)


def load_image(image_path: str):
    """
    Validate and load an image using OpenCV.

    Args:
        image_path: Path to the input image.

    Returns:
        Loaded OpenCV image.

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If the image format or content is invalid.
    """

    # Validate the image path and file format
    validate_image_path(image_path)

    path = Path(image_path)

    # Load the image
    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    # Validate loaded image dimensions
    validate_image_dimensions(image)

    return image