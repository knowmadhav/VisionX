from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_NAME = "yolo11n.pt"


def detect_objects(
    image,
    confidence_threshold=0.5
):
    """
    Detect objects in an image using YOLO.

    Args:
        image: Input OpenCV image.
        confidence_threshold: Minimum confidence required
                              for a detection.

    Returns:
        Tuple containing:
        - Annotated image
        - List of detected object information
    """

    # Load the pretrained YOLO model
    model = YOLO(MODEL_NAME)

    # Run object detection
    results = model(
        image,
        conf=confidence_threshold,
        verbose=False
    )

    annotated_image = image.copy()

    detected_objects = []

    object_id = 1

    # Process detection results
    for result in results:

        boxes = result.boxes

        for box in boxes:

            # Bounding box coordinates
            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            width = x2 - x1
            height = y2 - y1

            # Confidence score
            confidence = float(
                box.conf[0]
            )

            # Class ID
            class_id = int(
                box.cls[0]
            )

            # Class name
            class_name = result.names[
                class_id
            ]

            # Draw bounding box
            cv2.rectangle(
                annotated_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw label
            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                annotated_image,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            # Store detection information
            detected_objects.append(
                {
                    "id": object_id,
                    "object": class_name,
                    "confidence": round(
                        confidence,
                        4
                    ),
                    "bounding_box": {
                        "x": x1,
                        "y": y1,
                        "width": width,
                        "height": height,
                    },
                }
            )

            object_id += 1

    return (
        annotated_image,
        detected_objects
    )