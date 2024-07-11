# Real-time Object Detection and Interaction
This repository contains scripts for real-time object detection and interaction using both MATLAB and Python, demonstrating different approaches and tools for computer vision tasks.

## MATLAB Script (ObjectDetection_MATLAB.m)
The MATLAB script utilizes MATLAB's Computer Vision Toolbox to perform the following tasks:

- Foreground Detection: Utilizes Gaussian mixture models (vision.ForegroundDetector) to detect foreground objects in a video stream.
- Blob Analysis: Uses vision.BlobAnalysis to identify connected components (blobs) based on the detected foreground.
- Visualization: Displays the original video frames, the extracted foreground, and the cleaned foreground (after morphological operations).
- Object Detection: Draws bounding boxes around detected objects and counts them, displaying the results in real-time using imshow and vision.VideoPlayer.

## Python Script (ObjectDetection_Python.py)
The Python script employs OpenCV for video processing and interaction via Tkinter and email notifications:

- Object Detection with Caffe Model: Uses cv2.dnn module to perform object detection using a pre-trained MobileNet SSD model (MobileNetSSD_deploy.caffemodel).
- User Interaction: Utilizes Tkinter for displaying message boxes to prompt user actions based on detected objects (e.g., sending email notifications with or without object images).
- Email Notifications: Sends email alerts with optional image attachments using smtplib and email modules upon detecting specific objects (e.g., 'bottle').
- Efficient Video Processing: Implements video capture and processing efficiently using VideoStream from imutils.video.

## Features Highlight
- Both scripts demonstrate real-time object detection capabilities, allowing users to interact based on detected objects.
- MATLAB script emphasizes ease of use with built-in toolbox functions for foreground detection and blob analysis.
- Python script showcases integration of OpenCV for complex computer vision tasks and external interactions such as email notifications.
