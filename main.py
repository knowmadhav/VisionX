import argparse
from pathlib import Path

import cv2

from src.preprocessing.grayscale import (
    convert_to_grayscale,
)
from src.preprocessing.filtering import (
    apply_gaussian_blur,
    apply_median_filter,
)
from src.preprocessing.enhancement import (
    apply_histogram_equalization,
)
from src.preprocessing.thresholding import (
    apply_binary_threshold,
    apply_adaptive_threshold,
)

from src.detection.edges import detect_edges
from src.detection.contours import detect_contours
from src.detection.objects import detect_objects

from src.analysis.image_stats import (
    calculate_image_statistics,
)
from src.analysis.report import (
    generate_json_report,
)

from src.utils.image_loader import load_image
from src.utils.validators import (
    validate_minimum_area,
)
from src.utils.logger import setup_logger


def main():
    """
    Main command-line entry point for VisionX.
    """

    logger = setup_logger()

    logger.info(
        "VisionX application started"
    )

    # ==================================================
    # ARGUMENT PARSER
    # ==================================================

    parser = argparse.ArgumentParser(
        description=(
            "VisionX - Command-Line Computer "
            "Vision Analysis System"
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    # ==================================================
    # PREPROCESS
    # ==================================================

    preprocess_parser = (
        subparsers.add_parser(
            "preprocess",
            help=(
                "Perform image preprocessing "
                "operations"
            ),
        )
    )

    preprocess_parser.add_argument(
        "--input",
        required=True,
        help="Path to the input image",
    )

    preprocess_parser.add_argument(
        "--operation",
        required=True,
        choices=[
            "grayscale",
            "gaussian",
            "median",
            "equalize",
            "binary",
            "adaptive",
        ],
        help="Preprocessing operation",
    )

    # ==================================================
    # EDGES
    # ==================================================

    edge_parser = subparsers.add_parser(
        "edges",
        help="Detect edges using Canny",
    )

    edge_parser.add_argument(
        "--input",
        required=True,
        help="Path to the input image",
    )

    # ==================================================
    # CONTOURS
    # ==================================================

    contour_parser = subparsers.add_parser(
        "contours",
        help="Detect and analyze contours",
    )

    contour_parser.add_argument(
        "--input",
        required=True,
        help="Path to the input image",
    )

    contour_parser.add_argument(
        "--min-area",
        type=float,
        default=500,
        help=(
            "Minimum contour area "
            "(default: 500)"
        ),
    )

    # ==================================================
    # OBJECT DETECTION
    # ==================================================

    detect_parser = subparsers.add_parser(
        "detect",
        help="Detect objects using YOLO",
    )

    detect_parser.add_argument(
        "--input",
        required=True,
        help="Path to the input image",
    )

    detect_parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help=(
            "Minimum detection confidence "
            "(default: 0.5)"
        ),
    )

    # ==================================================
    # ANALYZE
    # ==================================================

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Perform complete image analysis",
    )

    analyze_parser.add_argument(
        "--input",
        required=True,
        help="Path to the input image",
    )

    analyze_parser.add_argument(
        "--min-area",
        type=float,
        default=500,
        help=(
            "Minimum contour area "
            "(default: 500)"
        ),
    )

    analyze_parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help=(
            "Minimum detection confidence "
            "(default: 0.5)"
        ),
    )

    # ==================================================
    # PIPELINE
    # ==================================================

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help=(
            "Run the complete VisionX "
            "computer vision pipeline"
        ),
    )

    pipeline_parser.add_argument(
        "--input",
        required=True,
        help="Path to the input image",
    )

    pipeline_parser.add_argument(
        "--min-area",
        type=float,
        default=500,
        help=(
            "Minimum contour area "
            "(default: 500)"
        ),
    )

    pipeline_parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help=(
            "Minimum detection confidence "
            "(default: 0.5)"
        ),
    )

    # ==================================================
    # PARSE ARGUMENTS
    # ==================================================

    args = parser.parse_args()

    try:

        # ==================================================
        # VALIDATION
        # ==================================================

        if args.command in [
            "contours",
            "analyze",
            "pipeline",
        ]:

            validate_minimum_area(
                args.min_area
            )

        if args.command in [
            "detect",
            "analyze",
            "pipeline",
        ]:

            if not (
                0.0
                <= args.confidence
                <= 1.0
            ):

                raise ValueError(
                    "Confidence must be between "
                    "0.0 and 1.0."
                )

        # ==================================================
        # LOAD IMAGE
        # ==================================================

        image = load_image(
            args.input
        )

        logger.info(
            f"Input image loaded: {args.input}"
        )

        # ==================================================
        # OUTPUT DIRECTORY
        # ==================================================

        output_directory = Path(
            "outputs"
        )

        output_directory.mkdir(
            exist_ok=True
        )

        # ==================================================
        # PREPROCESS
        # ==================================================

        if args.command == "preprocess":

            if args.operation == "grayscale":

                result = convert_to_grayscale(
                    image
                )

                output_path = (
                    output_directory
                    / "grayscale.jpg"
                )

                cv2.imwrite(
                    str(output_path),
                    result
                )

                print(
                    "Grayscale conversion completed."
                )

            elif args.operation == "gaussian":

                result = apply_gaussian_blur(
                    image
                )

                output_path = (
                    output_directory
                    / "gaussian_blur.jpg"
                )

                cv2.imwrite(
                    str(output_path),
                    result
                )

                print(
                    "Gaussian blur completed."
                )

            elif args.operation == "median":

                result = apply_median_filter(
                    image
                )

                output_path = (
                    output_directory
                    / "median_filter.jpg"
                )

                cv2.imwrite(
                    str(output_path),
                    result
                )

                print(
                    "Median filtering completed."
                )

            elif args.operation == "equalize":

                grayscale = (
                    convert_to_grayscale(
                        image
                    )
                )

                result = (
                    apply_histogram_equalization(
                        grayscale
                    )
                )

                output_path = (
                    output_directory
                    / "histogram_equalized.jpg"
                )

                cv2.imwrite(
                    str(output_path),
                    result
                )

                print(
                    "Histogram equalization completed."
                )

            elif args.operation == "binary":

                grayscale = (
                    convert_to_grayscale(
                        image
                    )
                )

                result = (
                    apply_binary_threshold(
                        grayscale
                    )
                )

                output_path = (
                    output_directory
                    / "binary_threshold.jpg"
                )

                cv2.imwrite(
                    str(output_path),
                    result
                )

                print(
                    "Binary thresholding completed."
                )

            elif args.operation == "adaptive":

                grayscale = (
                    convert_to_grayscale(
                        image
                    )
                )

                result = (
                    apply_adaptive_threshold(
                        grayscale
                    )
                )

                output_path = (
                    output_directory
                    / "adaptive_threshold.jpg"
                )

                cv2.imwrite(
                    str(output_path),
                    result
                )

                print(
                    "Adaptive thresholding completed."
                )

        # ==================================================
        # EDGES
        # ==================================================

        elif args.command == "edges":

            result = detect_edges(
                image
            )

            output_path = (
                output_directory
                / "edges.jpg"
            )

            cv2.imwrite(
                str(output_path),
                result
            )

            logger.info(
                "Canny edge detection completed"
            )

            print(
                "Canny edge detection completed."
            )

            print(
                f"Output saved to: {output_path}"
            )

        # ==================================================
        # CONTOURS
        # ==================================================

        elif args.command == "contours":

            annotated_image, contour_data = (
                detect_contours(
                    image,
                    minimum_area=args.min_area,
                )
            )

            output_path = (
                output_directory
                / "contours.jpg"
            )

            cv2.imwrite(
                str(output_path),
                annotated_image
            )

            logger.info(
                "Contour analysis completed"
            )

            print(
                "Contour detection and analysis "
                "completed."
            )

            print(
                "Significant contours detected: "
                f"{len(contour_data)}"
            )

            print(
                f"Output saved to: {output_path}"
            )

        # ==================================================
        # OBJECT DETECTION
        # ==================================================

        elif args.command == "detect":

            annotated_image, detected_objects = (
                detect_objects(
                    image,
                    confidence_threshold=(
                        args.confidence
                    ),
                )
            )

            output_path = (
                output_directory
                / "detected_objects.jpg"
            )

            success = cv2.imwrite(
                str(output_path),
                annotated_image
            )

            if not success:

                raise ValueError(
                    "Failed to save object "
                    "detection output."
                )

            logger.info(
                "Object detection completed"
            )

            print(
                "Object detection completed."
            )

            print(
                "Objects detected: "
                f"{len(detected_objects)}"
            )

            print(
                f"Output saved to: {output_path}"
            )

        # ==================================================
        # COMPLETE ANALYSIS
        # ==================================================

        elif args.command == "analyze":

            # Image statistics.
            image_statistics = (
                calculate_image_statistics(
                    image
                )
            )

            # Contour analysis.
            contour_image, contour_data = (
                detect_contours(
                    image,
                    minimum_area=args.min_area,
                )
            )

            contour_output = (
                output_directory
                / "analysis_contours.jpg"
            )

            cv2.imwrite(
                str(contour_output),
                contour_image
            )

            # Object detection.
            detection_image, detected_objects = (
                detect_objects(
                    image,
                    confidence_threshold=(
                        args.confidence
                    ),
                )
            )

            detection_output = (
                output_directory
                / "analysis_objects.jpg"
            )

            cv2.imwrite(
                str(detection_output),
                detection_image
            )

            # Generate JSON report.
            report_path = (
                output_directory
                / "analysis_report.json"
            )

            generate_json_report(
                image_path=args.input,
                image_statistics=(
                    image_statistics
                ),
                contour_data=contour_data,
                detected_objects=(
                    detected_objects
                ),
                output_path=report_path,
            )

            logger.info(
                "Complete analysis finished"
            )

            print(
                "\nImage analysis completed."
            )

            print(
                "\nImage Statistics:"
            )

            print(
                f"Width: "
                f"{image_statistics['width']} px"
            )

            print(
                f"Height: "
                f"{image_statistics['height']} px"
            )

            print(
                f"Channels: "
                f"{image_statistics['channels']}"
            )

            print(
                f"Total Pixels: "
                f"{image_statistics['total_pixels']}"
            )

            print(
                "\nContour Analysis:"
            )

            print(
                "Significant contours: "
                f"{len(contour_data)}"
            )

            print(
                "\nObject Detection:"
            )

            print(
                "Objects detected: "
                f"{len(detected_objects)}"
            )

            print(
                f"\nJSON report: {report_path}"
            )

        # ==================================================
        # COMPLETE PIPELINE
        # ==================================================

        elif args.command == "pipeline":

            print(
                "\n================================"
            )

            print(
                "       VisionX Pipeline"
            )

            print(
                "================================"
            )

            # ----------------------------------------------
            # STEP 1 — GRAYSCALE
            # ----------------------------------------------

            print(
                "\n[1/7] Converting to grayscale..."
            )

            grayscale = (
                convert_to_grayscale(
                    image
                )
            )

            grayscale_path = (
                output_directory
                / "pipeline_grayscale.jpg"
            )

            cv2.imwrite(
                str(grayscale_path),
                grayscale
            )

            # ----------------------------------------------
            # STEP 2 — GAUSSIAN BLUR
            # ----------------------------------------------

            print(
                "[2/7] Applying Gaussian blur..."
            )

            blurred = (
                apply_gaussian_blur(
                    grayscale
                )
            )

            blur_path = (
                output_directory
                / "pipeline_gaussian_blur.jpg"
            )

            cv2.imwrite(
                str(blur_path),
                blurred
            )

            # ----------------------------------------------
            # STEP 3 — ADAPTIVE THRESHOLD
            # ----------------------------------------------

            print(
                "[3/7] Applying adaptive threshold..."
            )

            thresholded = (
                apply_adaptive_threshold(
                    blurred
                )
            )

            threshold_path = (
                output_directory
                / "pipeline_threshold.jpg"
            )

            cv2.imwrite(
                str(threshold_path),
                thresholded
            )

            # ----------------------------------------------
            # STEP 4 — CANNY EDGES
            # ----------------------------------------------

            print(
                "[4/7] Detecting edges..."
            )

            edges = detect_edges(
                image
            )

            edges_path = (
                output_directory
                / "pipeline_edges.jpg"
            )

            cv2.imwrite(
                str(edges_path),
                edges
            )

            # ----------------------------------------------
            # STEP 5 — CONTOURS
            # ----------------------------------------------

            print(
                "[5/7] Detecting contours..."
            )

            contour_image, contour_data = (
                detect_contours(
                    image,
                    minimum_area=args.min_area,
                )
            )

            contour_path = (
                output_directory
                / "pipeline_contours.jpg"
            )

            cv2.imwrite(
                str(contour_path),
                contour_image
            )

            # ----------------------------------------------
            # STEP 6 — OBJECT DETECTION
            # ----------------------------------------------

            print(
                "[6/7] Detecting objects..."
            )

            detection_image, detected_objects = (
                detect_objects(
                    image,
                    confidence_threshold=(
                        args.confidence
                    ),
                )
            )

            detection_path = (
                output_directory
                / "pipeline_objects.jpg"
            )

            cv2.imwrite(
                str(detection_path),
                detection_image
            )

            # ----------------------------------------------
            # STEP 7 — ANALYSIS & REPORT
            # ----------------------------------------------

            print(
                "[7/7] Generating analysis report..."
            )

            image_statistics = (
                calculate_image_statistics(
                    image
                )
            )

            report_path = (
                output_directory
                / "pipeline_report.json"
            )

            generate_json_report(
                image_path=args.input,
                image_statistics=(
                    image_statistics
                ),
                contour_data=contour_data,
                detected_objects=(
                    detected_objects
                ),
                output_path=report_path,
            )

            logger.info(
                "Complete VisionX pipeline "
                "executed successfully"
            )

            # ----------------------------------------------
            # FINAL SUMMARY
            # ----------------------------------------------

            print(
                "\n================================"
            )

            print(
                "       Pipeline Complete"
            )

            print(
                "================================"
            )

            print(
                "\nResults:"
            )

            print(
                f"Contours detected: "
                f"{len(contour_data)}"
            )

            print(
                f"Objects detected: "
                f"{len(detected_objects)}"
            )

            print(
                "\nGenerated outputs:"
            )

            print(
                f"Grayscale: {grayscale_path}"
            )

            print(
                f"Gaussian blur: {blur_path}"
            )

            print(
                f"Threshold: {threshold_path}"
            )

            print(
                f"Edges: {edges_path}"
            )

            print(
                f"Contours: {contour_path}"
            )

            print(
                f"Objects: {detection_path}"
            )

            print(
                f"Report: {report_path}"
            )

    # ==================================================
    # ERROR HANDLING
    # ==================================================

    except (
        FileNotFoundError,
        ValueError,
    ) as error:

        logger.error(
            f"Operation failed: {error}"
        )

        print(
            f"Error: {error}"
        )


if __name__ == "__main__":
    main()