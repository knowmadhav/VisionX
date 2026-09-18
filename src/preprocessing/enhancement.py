import cv2


def apply_histogram_equalization(image):
    """
    Apply histogram equalization to a grayscale image.

    Args:
        image: Input grayscale image.

    Returns:
        Histogram-equalized image.
    """

    return cv2.equalizeHist(image)