import json

import numpy as np

from src.analysis.image_stats import (
    calculate_image_statistics,
)
from src.analysis.report import (
    generate_json_report,
)


def create_test_image():
    """
    Create a simple synthetic test image.
    """

    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8
    )

    image[:, :] = (
        100,
        150,
        200
    )

    return image


def test_image_statistics():
    image = create_test_image()

    statistics = (
        calculate_image_statistics(
            image
        )
    )

    assert statistics["width"] == 100
    assert statistics["height"] == 100
    assert statistics["channels"] == 3
    assert statistics["total_pixels"] == 10000


def test_image_statistics_contains_required_fields():
    image = create_test_image()

    statistics = (
        calculate_image_statistics(
            image
        )
    )

    required_fields = [
        "width",
        "height",
        "channels",
        "total_pixels",
        "mean_intensity",
        "standard_deviation",
        "minimum_intensity",
        "maximum_intensity",
    ]

    for field in required_fields:
        assert field in statistics


def test_image_statistics_intensity_range():
    image = create_test_image()

    statistics = (
        calculate_image_statistics(
            image
        )
    )

    assert (
        0 <= statistics["minimum_intensity"] <= 255
    )

    assert (
        0 <= statistics["maximum_intensity"] <= 255
    )


def test_json_report_generation(tmp_path):
    image_statistics = {
        "width": 100,
        "height": 100,
        "channels": 3,
        "total_pixels": 10000,
        "mean_intensity": 150.0,
        "standard_deviation": 20.0,
        "minimum_intensity": 100,
        "maximum_intensity": 200,
    }

    contour_data = [
        {
            "id": 1,
            "shape": "Square",
            "area": 5000.0,
            "perimeter": 280.0,
            "bounding_box": {
                "x": 20,
                "y": 20,
                "width": 80,
                "height": 80,
            },
            "centroid": {
                "x": 60,
                "y": 60,
            },
        }
    ]

    report_path = (
        tmp_path /
        "test_report.json"
    )

    result = generate_json_report(
        image_path="test.jpg",
        image_statistics=image_statistics,
        contour_data=contour_data,
        output_path=report_path,
    )

    assert result == report_path
    assert report_path.exists()


def test_json_report_content(tmp_path):
    image_statistics = {
        "width": 100,
        "height": 100,
        "channels": 3,
        "total_pixels": 10000,
        "mean_intensity": 150.0,
        "standard_deviation": 20.0,
        "minimum_intensity": 100,
        "maximum_intensity": 200,
    }

    contour_data = []

    report_path = (
        tmp_path /
        "test_report.json"
    )

    generate_json_report(
        image_path="test.jpg",
        image_statistics=image_statistics,
        contour_data=contour_data,
        output_path=report_path,
    )

    with open(
        report_path,
        "r",
        encoding="utf-8"
    ) as file:

        report = json.load(file)

    assert report["project"] == "VisionX"
    assert (
        report["input_image"] == "test.jpg"
    )

    assert (
        report["image_statistics"]["width"]
        == 100
    )

    assert (
        report["contour_analysis"][
            "total_significant_contours"
        ]
        == 0
    )

def test_json_report_contains_object_detection(
    tmp_path
):
    image_statistics = {
        "width": 100,
        "height": 100,
        "channels": 3,
        "total_pixels": 10000,
        "mean_intensity": 150.0,
        "standard_deviation": 20.0,
        "minimum_intensity": 100,
        "maximum_intensity": 200,
    }

    contour_data = []

    detected_objects = [
        {
            "id": 1,
            "object": "person",
            "confidence": 0.91,
            "bounding_box": {
                "x": 10,
                "y": 20,
                "width": 50,
                "height": 70,
            },
        }
    ]

    report_path = (
        tmp_path /
        "object_report.json"
    )

    generate_json_report(
        image_path="test.jpg",
        image_statistics=image_statistics,
        contour_data=contour_data,
        output_path=report_path,
        detected_objects=detected_objects,
    )

    with open(
        report_path,
        "r",
        encoding="utf-8"
    ) as file:

        report = json.load(file)

    assert "object_detection" in report

    assert (
        report["object_detection"][
            "total_objects"
        ]
        == 1
    )

    assert (
        report["object_detection"][
            "objects"
        ][0]["object"]
        == "person"
    )


def test_json_report_object_confidence(
    tmp_path
):
    image_statistics = {
        "width": 100,
        "height": 100,
        "channels": 3,
        "total_pixels": 10000,
        "mean_intensity": 150.0,
        "standard_deviation": 20.0,
        "minimum_intensity": 100,
        "maximum_intensity": 200,
    }

    detected_objects = [
        {
            "id": 1,
            "object": "car",
            "confidence": 0.87,
            "bounding_box": {
                "x": 20,
                "y": 30,
                "width": 60,
                "height": 40,
            },
        }
    ]

    report_path = (
        tmp_path /
        "confidence_report.json"
    )

    generate_json_report(
        image_path="test.jpg",
        image_statistics=image_statistics,
        contour_data=[],
        output_path=report_path,
        detected_objects=detected_objects,
    )

    with open(
        report_path,
        "r",
        encoding="utf-8"
    ) as file:

        report = json.load(file)

    confidence = (
        report["object_detection"][
            "objects"
        ][0]["confidence"]
    )

    assert confidence == 0.87