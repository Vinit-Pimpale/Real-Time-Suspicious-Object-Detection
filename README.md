# Real-Time Suspicious Object Detection

This repository contains an implementation of **Real-Time Suspicious Object Detection** using multiple object detection models for live video feeds. The project uses both classical machine learning and deep learning methods for detecting suspicious objects in various environments.

The goal of this project is to identify and track objects in real-time, providing useful information for surveillance systems or security-related applications.

---

## 1. Project Motivation and Objectives

With increasing reliance on video surveillance for security, real-time detection of suspicious or anomalous objects in live video feeds can significantly improve response times and effectiveness. This project integrates various detection models to identify objects that deviate from normal behavior or patterns in video streams, such as abandoned items or unusual movement.

The objectives of the project are:

- Implement object detection using different models: **Caffe**, **TensorFlow**, and **YOLO**.
- Provide a real-time detection pipeline with video feed support.
- Allow easy switching between classical object detection and deep learning-based methods.
- Extend the solution for integration into real-world surveillance applications.

---

## 2. Supported Detection Models

This project includes several methods for object detection and classification:

### 2.1. **Caffe Object Detection**
- Uses pre-trained Caffe models for detecting and classifying objects in images and video frames.
- Caffe offers efficient inference and is widely used in industry applications.

### 2.2. **TensorFlow Object Detection**
- Implements detection using TensorFlow models and provides a framework for training custom object detection models.
- Uses the TensorFlow Object Detection API, which provides pre-trained models for fast deployment.

### 2.3. **YOLO (You Only Look Once)**
- Implements YOLOv3 for real-time object detection with high accuracy and speed.
- YOLO is a state-of-the-art model known for its high performance in real-time detection.

---

## 3. System Overview

The system captures live video input (via webcam or external camera), processes each frame to detect objects, and identifies suspicious objects based on predefined categories (e.g., bags, weapons). Suspicious activity is flagged by the model, and results are visualized in real time with bounding boxes and class labels.

Key system features:
- **Real-Time Video Processing**: Live video input processed frame by frame.
- **Multi-Model Support**: Switching between different detection frameworks.
- **Suspicious Activity Detection**: Alerts for specific object types or unusual behavior.

---

## 4. Hardware Requirements

This project requires:

| Component               | Description |
|-------------------------|-------------|
| **Webcam or Camera**     | Captures real-time video feed |
| **GPU (optional)**       | Recommended for deep learning models (TensorFlow / YOLO) |
| **Computer with Python** | System running the object detection models |

---

## 5. Software Requirements

To run the project, the following software tools are required:

- **Python 3.7+**
- **OpenCV** for video capture and display
- **Caffe** (for Caffe-based object detection models)
- **TensorFlow** (for TensorFlow-based models)
- **YOLOv3** model files and weights (for YOLO-based detection)

For installation, use:

```bash
pip install opencv-python
pip install tensorflow
```

If using YOLO, you also need to download YOLOv3 weights and configuration files from the official repository.

---

## 6. Repository Structure

```text
Real-Time-Suspicious-Object-Detection/
├── CaffeObjectDetection/         # Caffe model-based detection scripts
├── TensorFlowObjectDetection/    # TensorFlow model-based detection scripts
├── YOLOObjectDetection/          # YOLO-based detection scripts
├── VideoObjectDetection/         # Scripts to process live video feeds
├── README.md                     # This file
```

Each subfolder corresponds to a different detection model, with scripts for using pre-trained models on input video.

---

## 7. Setup and Usage Instructions

### Step 1: Clone the repository

```bash
git clone https://github.com/Vinit-Pimpale/Real-Time-Suspicious-Object-Detection.git
cd Real-Time-Suspicious-Object-Detection
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Running Object Detection on Live Video

You can run the detection system on a live webcam feed by navigating to the respective model directory and executing the following:

```bash
python detect_live_video.py
```

This script will use the camera as input and apply object detection in real time. You can switch between models by modifying the script to use the desired detection model (Caffe, TensorFlow, or YOLO).

---

## 8. Video Processing and Model Switching

To process a recorded video instead of using a live camera feed:

```bash
python detect_video.py --input video_file.mp4 --model tensorflow
```

This will process the video using the specified model and output results with detected objects overlaid with bounding boxes.

---

## 9. Evaluation and Testing

To evaluate the performance of different models:

1. Test each detection method using a sample video with multiple objects.
2. Compare the processing time for real-time detection.
3. Evaluate the accuracy and precision of object detection for suspicious objects.

The performance of each model can be improved by training custom datasets or using higher-resolution models.

---

## 10. Limitations and Future Work

- The system is currently limited to a fixed number of suspicious objects (defined by the models).
- Performance may be constrained on lower-end hardware without a GPU.
- Future extensions could include:
  - Real-time alert notifications based on suspicious activity.
  - Enhanced object tracking capabilities.
  - Integration with cloud platforms for data storage and analysis.

---

## 11. Author

**Vinit Pimpale**

This project was developed as part of an exploration into real-time object detection and surveillance systems using deep learning and classical computer vision techniques.
