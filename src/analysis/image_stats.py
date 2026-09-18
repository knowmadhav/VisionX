import cv2
import numpy as np


def calculate_image_statistics(image):
    """
    Calculate basic quantitative statistics of an image.

    Args:
        image: Input OpenCV image.

    Returns:
        Dictionary containing image statistics.
    """

    height, width = image.shape[:2]

    if len(image.shape) == 3:
        channels = image.shape[2]
    else:
        channels = 1

    total_pixels = height * width

    grayscale_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    mean_intensity = float(
        np.mean(grayscale_image)
    )

    standard_deviation = float(
        np.std(grayscale_image)
    )

    minimum_intensity = int(
        np.min(grayscale_image)
    )

    maximum_intensity = int(
        np.max(grayscale_image)
    )

    return {
        "width": width,
        "height": height,
        "channels": channels,
        "total_pixels": total_pixels,
        "mean_intensity": round(
            mean_intensity,
            2
        ),
        "standard_deviation": round(
            standard_deviation,
            2
        ),
        "minimum_intensity": minimum_intensity,
        "maximum_intensity": maximum_intensity,
    }