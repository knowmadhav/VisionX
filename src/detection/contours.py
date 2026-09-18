import cv2


def classify_shape(contour):
    """
    Classify a contour based on its approximated polygon.

    Args:
        contour: OpenCV contour.

    Returns:
        Estimated shape name.
    """

    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return "Unknown"

    approximation = cv2.approxPolyDP(
        contour,
        0.04 * perimeter,
        True
    )

    vertices = len(approximation)

    if vertices == 3:
        return "Triangle"

    if vertices == 4:
        x, y, width, height = cv2.boundingRect(
            approximation
        )

        aspect_ratio = width / float(height)

        if 0.95 <= aspect_ratio <= 1.05:
            return "Square"

        return "Rectangle"

    if vertices == 5:
        return "Pentagon"

    if vertices == 6:
        return "Hexagon"

    return "Circle/Other"


def detect_contours(
    image,
    minimum_area=500
):
    """
    Detect and analyze significant contours.

    Args:
        image: Input OpenCV image.
        minimum_area: Minimum contour area to retain.

    Returns:
        Tuple containing:
        - Annotated image
        - List of contour information
    """

    # Convert to grayscale
    grayscale_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Reduce noise
    blurred_image = cv2.GaussianBlur(
        grayscale_image,
        (5, 5),
        0
    )

    # Detect edges
    edges = cv2.Canny(
        blurred_image,
        100,
        200
    )

    # Find external contours
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    annotated_image = image.copy()

    contour_data = []

    contour_id = 1

    for contour in contours:

        # Calculate contour area
        area = cv2.contourArea(contour)

        # Ignore very small contours
        if area < minimum_area:
            continue

        # Calculate perimeter
        perimeter = cv2.arcLength(
            contour,
            True
        )

        # Bounding rectangle
        x, y, width, height = cv2.boundingRect(
            contour
        )

        # Calculate centroid
        moments = cv2.moments(contour)

        if moments["m00"] != 0:

            centroid_x = int(
                moments["m10"] /
                moments["m00"]
            )

            centroid_y = int(
                moments["m01"] /
                moments["m00"]
            )

        else:

            centroid_x = 0
            centroid_y = 0

        # Calculate shape
        shape = classify_shape(contour)

        # Draw contour
        cv2.drawContours(
            annotated_image,
            [contour],
            -1,
            (0, 255, 0),
            2
        )

        # Draw bounding box
        cv2.rectangle(
            annotated_image,
            (x, y),
            (x + width, y + height),
            (255, 0, 0),
            2
        )

        # Draw centroid
        cv2.circle(
            annotated_image,
            (centroid_x, centroid_y),
            5,
            (0, 0, 255),
            -1
        )

        # Add shape label
        cv2.putText(
            annotated_image,
            f"{shape} #{contour_id}",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        contour_data.append(
            {
                "id": contour_id,
                "shape": shape,
                "area": round(area, 2),
                "perimeter": round(
                    perimeter,
                    2
                ),
                "bounding_box": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                },
                "centroid": {
                    "x": centroid_x,
                    "y": centroid_y,
                },
            }
        )

        contour_id += 1

    return annotated_image, contour_data