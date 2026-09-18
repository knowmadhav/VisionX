from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


def validate_image_path(image_path: str):
    """
    Validate the input image path.

    Args:
        image_path: Path to the input image.

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If the path is not a file or
                    the file format is unsupported.
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Input path is not a file: {image_path}"
        )

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported image format. "
            "Supported formats: JPG, JPEG, PNG"
        )


def validate_image_dimensions(image):
    """
    Validate the loaded image dimensions.

    Args:
        image: OpenCV image.

    Raises:
        ValueError: If the image is empty or invalid.
    """

    if image is None:
        raise ValueError(
            "Image data is empty."
        )

    if image.size == 0:
        raise ValueError(
            "Image contains no pixel data."
        )

    if len(image.shape) < 2:
        raise ValueError(
            "Invalid image dimensions."
        )

def validate_minimum_area(minimum_area):
    """
    Validate the minimum contour area.

    Args:
        minimum_area: Minimum contour area.

    Raises:
        ValueError: If the area is negative.
    """

    if minimum_area < 0:
        raise ValueError(
            "Minimum contour area cannot be negative."
        )