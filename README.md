# VisionX

## A Modular Command-Line Computer Vision Analysis System

VisionX is a Python-based command-line computer-vision system that combines fundamental image-processing techniques, classical computer-vision algorithms, quantitative image analysis, and YOLO-based object detection into a modular workflow.

The system can process an input image, perform individual computer-vision operations, or execute a complete analysis pipeline from the terminal.

---

# 1. Features

VisionX currently supports:

* Image validation and loading
* Grayscale conversion
* Gaussian filtering
* Median filtering
* Histogram equalization
* Binary thresholding
* Adaptive thresholding
* Canny edge detection
* Contour detection
* Shape classification
* Contour area and perimeter calculation
* Bounding-box detection
* Centroid calculation
* YOLO11 object detection
* Image statistics
* JSON report generation
* Complete computer-vision pipeline
* Command-line interface
* Error handling
* Application logging
* Automated testing

---

# 2. Technologies Used

| Technology       | Purpose                              |
| ---------------- | ------------------------------------ |
| Python           | Main programming language            |
| OpenCV           | Image processing and computer vision |
| NumPy            | Numerical image analysis             |
| Ultralytics YOLO | Object detection                     |
| Pytest           | Automated testing                    |
| argparse         | Command-line interface               |
| JSON             | Structured analysis reports          |
| Git              | Version control                      |

---

# 3. Project Architecture

VisionX follows a modular architecture.

```text
                    ┌─────────────────────┐
                    │     User / CLI      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      main.py        │
                    │  Command Dispatcher │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
     ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
     │ Preprocessing│  │   Detection  │  │   Analysis   │
     └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
            │                 │                 │
            ▼                 ▼                 ▼
       OpenCV Ops       Canny / Contours     Statistics
                        YOLO Detection        Reporting
             │                 │                 │
             └─────────────────┼─────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │      Outputs        │
                    │ Images / JSON / Log │
                    └─────────────────────┘
```

---

# 4. Requirements

Before running VisionX, make sure the following are installed:

* Python 3
* pip
* Git
* Terminal

The project uses a Python virtual environment.

---

# 5. Installation

## Step 1 — Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project directory:

```bash
cd VisionX
```

---

## Step 2 — Create a Virtual Environment

```bash
python3 -m venv venv
```

---

## Step 3 — Activate the Virtual Environment

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```powershell
venv\Scripts\activate
```

---

## Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

The main dependencies include:

* OpenCV
* NumPy
* Ultralytics
* Pytest

---

# 6. Verify Installation

Run:

```bash
python main.py --help
```

You should see the available VisionX commands.

You can also run the test suite:

```bash
python -m pytest
```

---

# 7. Input Image

A sample image is included in:

```text
images/sample.jpg
```

You can replace it with your own JPG, JPEG, or PNG image.

Supported formats:

```text
.jpg
.jpeg
.png
```

---

# 8. Command-Line Usage

VisionX provides several commands.

---

## 8.1 Preprocessing

### Grayscale Conversion

```bash
python main.py preprocess --input images/sample.jpg --operation grayscale
```

Output:

```text
outputs/grayscale.jpg
```

---

### Gaussian Blur

```bash
python main.py preprocess --input images/sample.jpg --operation gaussian
```

Output:

```text
outputs/gaussian_blur.jpg
```

---

### Median Filtering

```bash
python main.py preprocess --input images/sample.jpg --operation median
```

Output:

```text
outputs/median_filter.jpg
```

---

### Histogram Equalization

```bash
python main.py preprocess --input images/sample.jpg --operation equalize
```

Output:

```text
outputs/histogram_equalized.jpg
```

---

### Binary Thresholding

```bash
python main.py preprocess --input images/sample.jpg --operation binary
```

Output:

```text
outputs/binary_threshold.jpg
```

---

### Adaptive Thresholding

```bash
python main.py preprocess --input images/sample.jpg --operation adaptive
```

Output:

```text
outputs/adaptive_threshold.jpg
```

---

# 9. Edge Detection

VisionX uses the Canny edge-detection algorithm.

Run:

```bash
python main.py edges --input images/sample.jpg
```

Output:

```text
outputs/edges.jpg
```

---

# 10. Contour Analysis

Run:

```bash
python main.py contours --input images/sample.jpg
```

The default minimum contour area is `500`.

To specify your own minimum area:

```bash
python main.py contours --input images/sample.jpg --min-area 1000
```

The output image contains:

* Detected contours
* Bounding boxes
* Centroids
* Shape labels

Output:

```text
outputs/contours.jpg
```

---

# 11. Object Detection

VisionX uses a pretrained YOLO11 model for object detection.

Run:

```bash
python main.py detect --input images/sample.jpg
```

The default confidence threshold is `0.5`.

To specify another threshold:

```bash
python main.py detect --input images/sample.jpg --confidence 0.7
```

Output:

```text
outputs/detected_objects.jpg
```

The first execution may download the required YOLO model automatically.

---

# 12. Complete Image Analysis

The `analyze` command combines:

* Image statistics
* Contour analysis
* Object detection
* JSON report generation

Run:

```bash
python main.py analyze --input images/sample.jpg
```

Optional parameters:

```bash
python main.py analyze \
    --input images/sample.jpg \
    --min-area 500 \
    --confidence 0.5
```

Generated outputs include:

```text
outputs/analysis_contours.jpg
outputs/analysis_objects.jpg
outputs/analysis_report.json
```

---

# 13. Complete VisionX Pipeline

The `pipeline` command executes the complete workflow.

Run:

```bash
python main.py pipeline --input images/sample.jpg
```

The pipeline performs:

```text
1. Grayscale Conversion
          ↓
2. Gaussian Blur
          ↓
3. Adaptive Thresholding
          ↓
4. Canny Edge Detection
          ↓
5. Contour Detection
          ↓
6. YOLO Object Detection
          ↓
7. Analysis Report Generation
```

The pipeline generates:

```text
outputs/pipeline_grayscale.jpg
outputs/pipeline_gaussian_blur.jpg
outputs/pipeline_threshold.jpg
outputs/pipeline_edges.jpg
outputs/pipeline_contours.jpg
outputs/pipeline_objects.jpg
outputs/pipeline_report.json
```

---

# 14. Output Structure

After running the application, the `outputs/` directory may contain:

```text
outputs/
├── grayscale.jpg
├── gaussian_blur.jpg
├── median_filter.jpg
├── histogram_equalized.jpg
├── binary_threshold.jpg
├── adaptive_threshold.jpg
├── edges.jpg
├── contours.jpg
├── detected_objects.jpg
├── analysis_contours.jpg
├── analysis_objects.jpg
├── analysis_report.json
├── pipeline_grayscale.jpg
├── pipeline_gaussian_blur.jpg
├── pipeline_threshold.jpg
├── pipeline_edges.jpg
├── pipeline_contours.jpg
├── pipeline_objects.jpg
├── pipeline_report.json
└── visionx.log
```

---

# 15. JSON Report

The generated JSON report contains structured information such as:

```json
{
    "project": "VisionX",
    "input_image": "images/sample.jpg",
    "image_statistics": {},
    "contour_analysis": {
        "total_significant_contours": 0,
        "contours": []
    },
    "object_detection": {
        "total_objects": 0,
        "objects": []
    }
}
```

Object detection records contain information such as:

```text
Object class
Confidence score
Bounding-box coordinates
Bounding-box dimensions
```

Contour records contain:

```text
Shape
Area
Perimeter
Bounding box
Centroid
```

---

# 16. Testing

VisionX uses Pytest for automated testing.

Run all tests:

```bash
python -m pytest
```

The test suite covers:

* Preprocessing
* Edge detection
* Contour detection
* Image analysis
* Input validation

The project currently contains **32 automated tests**.

---

# 17. Error Handling

VisionX validates:

* Whether the input image exists
* Whether the path points to a file
* Whether the image format is supported
* Whether the image contains valid pixel data
* Whether contour minimum area is valid
* Whether YOLO confidence is between `0.0` and `1.0`

Example:

```bash
python main.py edges --input images/missing.jpg
```

The application reports a controlled error instead of continuing with invalid input.

---

# 18. Logging

VisionX records important application events in:

```text
outputs/visionx.log
```

The log records events such as:

* Application startup
* Input image loading
* Successful processing
* Processing failures

---

# 19. Repository Structure

```text
VisionX/
│
├── README.md
├── statement.md
├── requirements.txt
├── .gitignore
├── main.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── grayscale.py
│   │   ├── filtering.py
│   │   ├── enhancement.py
│   │   └── thresholding.py
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── edges.py
│   │   ├── contours.py
│   │   └── objects.py
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── image_stats.py
│   │   └── report.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── image_loader.py
│       ├── validators.py
│       └── logger.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_edges.py
│   ├── test_contours.py
│   ├── test_analysis.py
│   └── test_validation.py
│
├── images/
│   └── sample.jpg
│
├── outputs/
│
└── docs/
```

---

# 20. Design Principles

VisionX follows several software-engineering principles:

### Modularity

Different computer-vision operations are separated into dedicated modules.

### Separation of Concerns

Image loading, preprocessing, detection, analysis, reporting, validation, and logging are handled independently.

### Reusability

Computer-vision functions can be imported and reused independently of the CLI.

### Testability

Core functionality is covered by automated tests.

### Extensibility

New algorithms can be added as additional modules without rewriting the complete application.

---

# 21. Computer Vision Concepts Demonstrated

VisionX provides practical implementation of the following concepts:

* Digital image representation
* Grayscale conversion
* Image filtering
* Gaussian smoothing
* Median filtering
* Histogram equalization
* Thresholding
* Adaptive thresholding
* Canny edge detection
* Contour extraction
* Polygon approximation
* Shape classification
* Image moments
* Bounding boxes
* Centroid calculation
* Object detection
* Confidence thresholds
* Quantitative image analysis

---

# 22. Limitations

The current implementation has some limitations:

* YOLO uses a pretrained model.
* Object-detection performance depends on the pretrained model.
* The application processes static images.
* Shape classification uses contour geometry.
* There is currently no GUI.
* Object detection requires additional computational resources compared with basic OpenCV operations.

---

# 23. Future Enhancements

Possible future improvements include:

* Webcam support
* Video processing
* Real-time object detection
* Image segmentation
* Custom-trained detection models
* GUI interface
* Performance optimization
* Parallel processing
* Database storage
* CSV/PDF report generation
* Interactive result visualization

---

# 24. Project Documentation

Additional project documentation will include:

```text
docs/
├── architecture.png
├── workflow.png
├── use-case.png
├── sequence.png
├── class-diagram.png
└── pipeline.png
```

These diagrams document the architecture, workflow, system interactions, class/module relationships, and computer-vision pipeline.

---

# 25. Academic Purpose

VisionX was developed as an academic computer-vision project to demonstrate the practical implementation of image-processing and computer-vision concepts together with modular software-engineering practices.

The project connects theoretical computer-vision concepts with executable Python implementations and measurable outputs.

---

# 26. License

This project is intended for academic and educational use.
