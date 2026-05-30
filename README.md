## DIZZY 🚗💤

### Real-Time Driver Drowsiness & Dizziness Detection System

DIZZY is a real-time computer vision-based driver monitoring system that detects signs of **drowsiness, fatigue, yawning, and dizziness** using facial landmark analysis. The system continuously monitors the driver's face through a webcam and generates alerts when dangerous conditions are detected.

The project aims to reduce road accidents caused by driver fatigue by providing an affordable software-based safety solution.

---

# Features

* Real-time face detection
* Eye Aspect Ratio (EAR) based drowsiness detection
* Mouth Aspect Ratio (MAR) based yawn detection
* Head pose estimation for dizziness monitoring
* Vehicle vibration filtering to reduce false alarms
* Audio alert system
* Automatic screenshot capture during alert events
* User-specific calibration
* Signal smoothing using EMA and median filters
* Multi-threaded alert handling

---

# Problem Statement

Driver fatigue is one of the leading causes of road accidents worldwide. Traditional monitoring systems often require expensive hardware and specialized sensors.

DIZZY provides a low-cost software solution that:

* Detects prolonged eye closure
* Detects excessive yawning
* Detects abnormal head movements
* Reduces false detections caused by road vibrations
* Generates immediate alerts to regain driver attention

---

# System Architecture

```text
Webcam Feed
      │
      ▼
Face Detection
      │
      ▼
Facial Landmark Detection
      │
      ├────────────► Eye Analysis (EAR)
      │
      ├────────────► Mouth Analysis (MAR)
      │
      └────────────► Head Pose Estimation
                              │
                              ▼
                  Vibration Filtering
                              │
                              ▼
                    Decision Engine
                              │
                              ▼
                      Alert System
                              │
                              ▼
           Audio Alarm + Screenshot Logging
```

---

# Technologies Used

| Technology | Purpose                   |
| ---------- | ------------------------- |
| Python     | Core Programming Language |
| OpenCV     | Computer Vision           |
| Dlib       | Facial Landmark Detection |
| NumPy      | Numerical Computation     |
| SciPy      | Distance Calculations     |
| Pygame     | Audio Alerts              |
| Threading  | Background Alert Handling |

---

# Detection Techniques

## 1. Eye Aspect Ratio (EAR)

The system monitors eye closure using facial landmarks.

A continuously low EAR value indicates that the driver's eyes remain closed for an abnormal duration, which may suggest drowsiness or microsleep.

### Applications

* Drowsiness detection
* Fatigue monitoring
* Microsleep detection

---

## 2. Mouth Aspect Ratio (MAR)

The system measures mouth opening to detect yawning.

Frequent or prolonged yawning is treated as a fatigue indicator and contributes to the drowsiness score.

### Applications

* Fatigue estimation
* Driver alertness monitoring

---

## 3. Head Pose Estimation

Head orientation is estimated using facial landmarks and OpenCV's `solvePnP()` algorithm.

The system tracks:

* Pitch
* Roll
* Yaw

Abnormal head movements can indicate:

* Dizziness
* Driver distraction
* Loss of attention

---

## 4. Vibration Filtering

Road conditions often introduce noise into camera measurements.

To reduce false positives, the system uses:

### Median Filtering

Removes sudden spikes caused by vibration.

### Exponential Moving Average (EMA)

Smooths landmark movement over time.

### Velocity Variance Analysis

Differentiates between genuine driver movement and vehicle vibration.

---

# Project Structure

```text
DIZZY/
│
├── new.py
├── alert.py
├── requirements.txt
├── alert.wav
├── shape_predictor_68_face_landmarks.dat
├── screenshots/
└── README.md
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/rohitlodhi-hub/dizzy.git
cd dizzy
```

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Download Facial Landmark Model

This project requires Dlib's 68-point facial landmark predictor.

Download:

https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Extract the file and place:

```text
shape_predictor_68_face_landmarks.dat
```

inside the project root directory.

---

# Requirements

Create a `requirements.txt` file containing:

```text
opencv-python>=4.5.0
numpy>=1.19.0
scipy>=1.5.0
pygame>=2.0.0
dlib>=19.24.0
```

Install using:

```bash
pip install -r requirements.txt
```

---

# Running the Project

```bash
python new.py
```

Ensure:

* Webcam is connected
* Audio is enabled
* Landmark model file is present
* Adequate lighting conditions exist

---

# Alert Mechanism

The system triggers alerts when:

### Eye Alert

Eyes remain closed beyond a predefined duration.

### Yawn Alert

Yawning exceeds the configured threshold.

### Head Movement Alert

Abnormal head pose is detected for an extended period.

When an alert occurs:

1. Alarm sound is played.
2. Screenshot is captured.
3. Timestamp is logged.
4. Driver receives immediate feedback.

---

# Performance

Test Environment:

* Windows 11
* Python 3.12
* OpenCV 4.x
* Dlib 19.24+

Typical Performance:

| Metric             | Value              |
| ------------------ | ------------------ |
| Frame Rate         | 20–30 FPS          |
| Detection Speed    | Real-Time          |
| Alert Latency      | Less than 1 second |
| Camera Requirement | Standard Webcam    |

---

# Limitations

* Single-face tracking
* Performance depends on lighting conditions
* Glasses may reduce detection accuracy
* Dlib is slower than modern landmark detectors
* Not intended for medical diagnosis
* Accuracy decreases under extreme camera angles

---

# Future Improvements

## Planned Enhancements

* MediaPipe Face Mesh Integration
* Deep Learning-Based Fatigue Detection
* Night Vision Support
* Mobile Application
* Driver Identity Recognition
* Cloud-Based Event Storage
* Fleet Monitoring Dashboard
* Multi-Face Detection
* AI-Based Driver Risk Scoring

---

# Resume Description

Developed a real-time Driver Drowsiness and Dizziness Detection System using Python, OpenCV, and Dlib. Implemented Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and Head Pose Estimation techniques with vibration filtering to detect fatigue-related driving risks. Built a multi-threaded alert system with audio notifications and screenshot logging for real-time driver monitoring.

---

# Research Concepts Used

* Computer Vision
* Facial Landmark Detection
* Signal Processing
* Real-Time Systems
* Head Pose Estimation
* Human Behavior Analysis
* Driver Monitoring Systems
* Noise Reduction Techniques

---

# Author

**Rohit Lodhi**

B.Tech (CSE - AI & ML)

Computer Vision | Artificial Intelligence | Machine Learning

GitHub: https://github.com/rohitlodhi-hub

---

# License

This project is developed for educational and research purposes.

Feel free to use, modify, and improve the project with proper attribution.
