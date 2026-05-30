# DIZZY 🚗💤

## AI-Powered Driver Drowsiness & Fatigue Detection System

DIZZY is a real-time Driver Monitoring System (DMS) that uses Computer Vision and Facial Landmark Analysis to detect signs of:

* Drowsiness
* Fatigue
* Excessive Yawning
* Head Tilt / Attention Loss
* Possible Dizziness

The system analyzes facial landmarks using MediaPipe Face Mesh and generates instant alerts when fatigue-related behavior is detected.

---

## Key Features

### Real-Time Monitoring

* Live webcam-based face tracking
* Single-face monitoring
* Real-time processing (20–30 FPS)

### Drowsiness Detection

* Eye Aspect Ratio (EAR) based eye closure detection
* Adaptive threshold calibration
* Consecutive-frame verification

### Yawn Detection

* Mouth Aspect Ratio (MAR) based yawn detection
* Dynamic user-specific calibration

### Head Pose Analysis

* Pitch estimation
* Yaw estimation
* Roll estimation
* Sustained head tilt detection

### False Alarm Reduction

* Median filtering
* Exponential Moving Average (EMA)
* Velocity variance analysis
* Road vibration suppression

### Alert System

* Audio alarm
* Screenshot capture
* Timestamp logging
* Alert cooldown mechanism

### Performance Enhancements

* Multi-threaded alert handling
* Adaptive threshold learning
* Landmark smoothing
* Noise filtering

---

## Technologies Used

| Technology          | Purpose                   |
| ------------------- | ------------------------- |
| Python              | Core Programming Language |
| OpenCV              | Image Processing          |
| MediaPipe Face Mesh | Facial Landmark Detection |
| NumPy               | Numerical Computation     |
| SciPy               | Distance Calculations     |
| Pygame              | Audio Alert System        |
| Threading           | Concurrent Alert Handling |

---

## System Architecture

```text
Webcam Feed
      │
      ▼
MediaPipe Face Mesh
      │
      ▼
Facial Landmark Extraction
      │
      ├────────► Eye Aspect Ratio (EAR)
      │
      ├────────► Mouth Aspect Ratio (MAR)
      │
      └────────► Head Pose Estimation
                         │
                         ▼
             Smoothing & Filtering
                         │
                         ▼
                 Decision Engine
                         │
                         ▼
                  Alert Manager
                         │
      ┌──────────────────┴──────────────────┐
      ▼                                     ▼
Audio Alarm                     Screenshot Logging
```

---

## Detection Methods

### 1. Eye Aspect Ratio (EAR)

The system measures eye openness using six facial landmarks around each eye.

Low EAR values sustained across multiple frames indicate:

* Drowsiness
* Microsleep
* Fatigue

---

### 2. Mouth Aspect Ratio (MAR)

The system measures mouth opening using multiple lip landmarks.

High MAR values indicate:

* Yawning
* Fatigue
* Reduced alertness

---

### 3. Head Pose Estimation

Using OpenCV solvePnP and MediaPipe landmarks, the system estimates:

* Pitch
* Yaw
* Roll

Abnormal sustained head movements may indicate:

* Driver distraction
* Fatigue
* Dizziness

---

### 4. Vibration Filtering

Vehicle vibrations often create noisy measurements.

The system combines:

* Median Filter
* EMA Filter
* Velocity Variance Analysis

to suppress false alerts.

---

## Adaptive Calibration

Unlike fixed-threshold systems, DIZZY automatically calibrates itself during startup.

Calibration Period:

* First 120 frames

The system learns:

* User baseline EAR
* User baseline MAR

Then computes personalized thresholds.

Benefits:

* Higher accuracy
* Better user adaptation
* Reduced false positives

---

## Project Structure

```text
DIZZY/
│
├── drowsiness_detector.py
├── alert.wav
├── alerts/
│   ├── alert_eye_xxx.jpg
│   ├── alert_yawn_xxx.jpg
│   └── alert_head_xxx.jpg
│
├── requirements.txt
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/rohitlodhi-hub/dizzy.git
cd dizzy
```

### Create Virtual Environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

```txt
opencv-python
mediapipe
numpy
scipy
pygame
```

Install manually:

```bash
pip install opencv-python mediapipe numpy scipy pygame
```

---

## Running the Project

```bash
python drowsiness_detector.py
```

Press:

```text
q
```

to exit.

---

## Alert Conditions

### Eye Alert

Triggered when:

* EAR remains below threshold
* For multiple consecutive frames

### Yawn Alert

Triggered when:

* MAR remains above threshold
* For multiple consecutive frames

### Head Alert

Triggered when:

* Sustained abnormal pitch or yaw is detected

---

## Performance

Test Environment:

* Windows 11
* Python 3.12
* OpenCV 4.x
* MediaPipe 0.10+

Typical Performance:

| Metric        | Value               |
| ------------- | ------------------- |
| FPS           | 20–30               |
| Alert Latency | < 1 Second          |
| Webcam        | Standard USB Camera |
| Processing    | Real Time           |

---

## Limitations

* Single-face tracking only
* Performance depends on lighting conditions
* Extreme head angles may reduce accuracy
* Webcam quality affects detection reliability
* Not intended for medical diagnosis

---

## Future Enhancements

* Deep Learning Fatigue Detection
* Mobile Application
* Night Vision Support
* Driver Identification
* Cloud Event Storage
* Fleet Monitoring Dashboard
* Multi-Face Support
* Driver Risk Scoring
* Edge AI Deployment

---

## Resume Description

Developed an AI-powered Driver Drowsiness and Fatigue Detection System using Python, OpenCV, and MediaPipe Face Mesh. Implemented Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), head-pose estimation, adaptive calibration, and vibration filtering to detect fatigue-related driving risks in real time. Built a multi-threaded alert framework with audio alarms and automated screenshot logging.

---

## Author

Rohit Lodhi

B.Tech – Computer Science Engineering (AI & ML)

Interests:

* Artificial Intelligence
* Computer Vision
* Machine Learning
* Real-Time Monitoring Systems

GitHub:
https://github.com/rohitlodhi-hub/dizzy

---

## License

This project is intended for educational, research, and portfolio purposes.

Use responsibly and provide attribution when redistributing or modifying the project.
