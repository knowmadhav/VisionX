import cv2


def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """
    Apply Gaussian blur to reduce image noise.

    Args:
        image: Input OpenCV image.
        kernel_size: Size of the Gaussian kernel.

    Returns:
        Gaussian blurred image.
    """

    return cv2.GaussianBlur(image, kernel_size, 0)


def apply_median_filter(image, kernel_size=5):
    """
    Apply median filtering to reduce noise.

    Args:
        image: Input OpenCV image.
        kernel_size: Size of the median filter.

    Returns:
        Median filtered image.
    """

    return cv2.medianBlur(image, kernel_size)