import json
from pathlib import Path


def generate_json_report(
    image_path,
    image_statistics,
    contour_data,
    output_path,
    detected_objects=None
):
    """
    Generate a JSON analysis report.

    Args:
        image_path: Path of the input image.
        image_statistics: Calculated image statistics.
        contour_data: List of detected contour information.
        output_path: Path where the JSON report will be saved.
        detected_objects: List of YOLO object detections.

    Returns:
        Path of the generated report.
    """

    if detected_objects is None:
        detected_objects = []

    report = {
        "project": "VisionX",
        "input_image": str(image_path),

        "image_statistics": image_statistics,

        "contour_analysis": {
            "total_significant_contours": len(
                contour_data
            ),
            "contours": contour_data,
        },

        "object_detection": {
            "total_objects": len(
                detected_objects
            ),
            "objects": detected_objects,
        },
    }

    output_path = Path(output_path)

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    return output_path