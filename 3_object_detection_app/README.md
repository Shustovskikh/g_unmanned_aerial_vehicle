# Object Detection App

An application for detecting objects (such as faces, circular objects, and other custom objects) in images and video streams, utilizing OpenCV and pretrained models like Haar Cascades and YOLO.

## Description

This app provides:
- Face detection using Haar Cascades.
- Face detection using the YOLO model.
- Real-time highlighting of detected objects with bounding boxes on images or video.
- Support for various sources: webcam, video files, and static images.

## Features

1. **Choose Detection Method**:
   - Haar Cascades for face detection.
   - YOLO for more precise and flexible detection.

2. **Select Data Source**:
   - Webcam.
   - Video file.
   - Static image.

3. **Display and Save Results**:
   - Detected faces are highlighted with bounding boxes.
   - Results are saved in the `output/` folder.

## Installation and Setup

### Dependencies

Make sure the following libraries are installed:
- Python 3.7+
- OpenCV
- Numpy
- for example, add all yolov3-wider_16000.weights to the folder 3_object_detection_app\models

You can install dependencies with:
```bash
pip install -r requirements.txt
