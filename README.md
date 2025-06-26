Drowsiness Detection System

This project is a real-time drowsiness detection system using Python, OpenCV, MediaPipe, and Pygame. It leverages MediaPipe’s Face Mesh model to identify key facial landmarks, especially around the eyes, and calculates the Eye Aspect Ratio (EAR), a well-known metric to determine eye closure.

When the EAR drops below a certain threshold (default: 0.25) for a sustained duration (e.g., 3 seconds), the system detects drowsiness. In response, it triggers a continuous beep sound using Pygame and sends an email alert to a specified recipient. To avoid spamming, the system enforces a cooldown period between email alerts.

The main window displays the webcam feed along with real-time landmark tracking and alert messages. In parallel, a secondary analysis window shows zoomed-in facial features (eyes, nose, mouth) on a black canvas and includes a live EAR graph for better monitoring. The landmarks and graph disappear in real-time when no face is detected to ensure clarity.

This solution is well-suited for driver monitoring systems, student engagement tracking, or workplace fatigue detection. It provides a compact and efficient setup that integrates computer vision with real-time alerts and analysis.

Features

Real-time webcam-based drowsiness detection

EAR-based eye closure tracking

Continuous beep alert during drowsy state

Email alert system with cooldown period

Visual landmark overlay and EAR graph in a secondary analysis window

Auto-clears visuals when no face is detected

Requirements

Python 3.10+

OpenCV

MediaPipe

Pygame

NumPy

smtplib (built-in)

Usage

Ensure all dependencies are installed.

Replace the email credentials and recipient in the send_email() function.

Run the script using a webcam-connected device.

Press q to quit the application.

For best results, use in well-lit environments with a clear frontal view of the face.
