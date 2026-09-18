import cv2


def apply_binary_threshold(image, threshold_value=127):
    """
    Apply binary thresholding to a grayscale image.

    Args:
        image: Input grayscale image.
        threshold_value: Pixel intensity threshold.

    Returns:
        Binary thresholded image.
    """

    _, thresholded_image = cv2.threshold(
        image,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    return thresholded_image


def apply_adaptive_threshold(image):
    """
    Apply adaptive thresholding to a grayscale image.

    Args:
        image: Input grayscale image.

    Returns:
        Adaptively thresholded image.
    """

    thresholded_image = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return thresholded_image