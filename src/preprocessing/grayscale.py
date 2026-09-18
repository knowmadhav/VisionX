import cv2


def convert_to_grayscale(image):
    """
    Convert a BGR image into grayscale.

    Args:
        image: OpenCV image in BGR format.

    Returns:
        Grayscale image.
    """

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return grayscale_image