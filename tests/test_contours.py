import cv2
import numpy as np

from src.detection.contours import (
    classify_shape,
    detect_contours,
)


def create_test_image():
    """
    Create a synthetic image containing a rectangle.
    """

    image = np.zeros(
        (300, 300, 3),
        dtype=np.uint8
    )

    cv2.rectangle(
        image,
        (75, 75),
        (225, 225),
        (255, 255, 255),
        -1
    )

    return image


def test_contour_detection_returns_data():
    image = create_test_image()

    annotated_image, contour_data = (
        detect_contours(
            image,
            minimum_area=500
        )
    )

    assert annotated_image is not None
    assert isinstance(
        contour_data,
        list
    )


def test_contour_detection_finds_rectangle():
    image = create_test_image()

    _, contour_data = detect_contours(
        image,
        minimum_area=500
    )

    assert len(contour_data) > 0

    shapes = [
        contour["shape"]
        for contour in contour_data
    ]

    assert (
        "Rectangle" in shapes
        or "Square" in shapes
    )


def test_contour_contains_required_information():
    image = create_test_image()

    _, contour_data = detect_contours(
        image,
        minimum_area=500
    )

    assert len(contour_data) > 0

    contour = contour_data[0]

    assert "id" in contour
    assert "shape" in contour
    assert "area" in contour
    assert "perimeter" in contour
    assert "bounding_box" in contour
    assert "centroid" in contour


def test_contour_bounding_box():
    image = create_test_image()

    _, contour_data = detect_contours(
        image,
        minimum_area=500
    )

    assert len(contour_data) > 0

    bounding_box = contour_data[0][
        "bounding_box"
    ]

    assert bounding_box["width"] > 0
    assert bounding_box["height"] > 0


def test_contour_centroid():
    image = create_test_image()

    _, contour_data = detect_contours(
        image,
        minimum_area=500
    )

    assert len(contour_data) > 0

    centroid = contour_data[0]["centroid"]

    assert centroid["x"] > 0
    assert centroid["y"] > 0


def test_triangle_classification():
    triangle = np.array(
        [
            [[100, 50]],
            [[50, 150]],
            [[150, 150]],
        ],
        dtype=np.int32
    )

    shape = classify_shape(
        triangle
    )

    assert shape == "Triangle"