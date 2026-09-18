import cv2


def detect_edges(image, lower_threshold=100, upper_threshold=200):
    """
    Detect edges in an image using the Canny edge detection algorithm.

    Args:
        image: Input OpenCV image.
        lower_threshold: Lower threshold for Canny hysteresis.
        upper_threshold: Upper threshold for Canny hysteresis.

    Returns:
        Binary image containing detected edges.
    """

    # Convert image to grayscale
    grayscale_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Reduce noise before edge detection
    blurred_image = cv2.GaussianBlur(
        grayscale_image,
        (5, 5),
        0
    )

    # Apply Canny edge detection
    edges = cv2.Canny(
        blurred_image,
        lower_threshold,
        upper_threshold
    )

    return edges