import numpy as np

from src.detection.edges import detect_edges


def create_test_image():
    """
    Create a synthetic image containing a simple shape.
    """

    image = np.zeros(
        (200, 200, 3),
        dtype=np.uint8
    )

    image[50:150, 50:150] = 255

    return image


def test_edge_detection_output_shape():
    image = create_test_image()

    result = detect_edges(image)

    assert result.shape == (
        200,
        200
    )


def test_edge_detection_output_type():
    image = create_test_image()

    result = detect_edges(image)

    assert result.dtype == np.uint8


def test_edge_detection_produces_edges():
    image = create_test_image()

    result = detect_edges(image)

    # The synthetic square should produce
    # at least some edge pixels.
    assert np.count_nonzero(result) > 0


def test_edge_detection_is_binary():
    image = create_test_image()

    result = detect_edges(image)

    unique_values = np.unique(result)

    assert set(unique_values).issubset(
        {0, 255}
    )