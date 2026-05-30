import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from threading import Thread, Lock
from collections import deque
import time
import os
import datetime
import logging
import pygame

# --- Logger ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuration ---
class Config:
    # Fallback thresholds (before calibration)
    EYE_AR_THRESH = 0.25
    MOUTH_AR_THRESH = 0.55

    # Calibration multipliers
    EYE_AR_MULTIPLIER = 0.6        # alert when EAR < 60% of baseline
    MOUTH_AR_MULTIPLIER = 1.6      # alert when MAR > 160% of baseline

    # Consecutive frames to confirm a state
    EYE_AR_CONSEC_FRAMES = 15
    YAWN_CONSEC_FRAMES = 10
    HEAD_TILT_CONSEC_FRAMES = 20

    # Head tilt thresholds (degrees)
    HEAD_TILT_THRESH = 25.0
    HEAD_TILT_SUSTAINED_THRESH = 20.0

    # Smoothing & vibration detection
    HEAD_POSITION_BUFFER_SIZE = 10
    VELOCITY_BUFFER_SIZE = 8
    VELOCITY_VARIANCE_THRESHOLD = 4.0

    # Alerting
    ALERT_COOLDOWN = 3.0
    ALERT_SAVE_DIR = "alerts"

    # Camera
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    FPS_LIMIT = 30

    # Calibration
    CALIBRATION_FRAMES = 120


# --- MediaPipe landmark indices ---
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# 12-point mouth ring for reliable MAR:
# 0,1  = left/right corners
# 2,3  = upper-outer / lower-outer lip
# 4,5  = upper-mid / lower-mid lip  (second vertical pair)
# 6,7  = upper-inner / lower-inner  (third vertical pair)
# 8,9,10,11 = side fillers (not used in MAR formula but kept for drawing)
MOUTH_POINTS = [61, 291, 0, 17, 39, 269, 82, 312, 78, 308, 13, 14]
#               L   R   OU  OL  M1  M2   I1  I2  -- -- --  --
# Indices used in MAR:
# corners  : 0 (idx 61), 1 (idx 291)
# vertical1: 2 (idx 0),  3 (idx 17)
# vertical2: 4 (idx 39), 5 (idx 269)
# vertical3: 6 (idx 82), 7 (idx 312)


# --- Audio ---
pygame.mixer.init()
try:
    ALERT_SOUND = pygame.mixer.Sound("alert.wav")
except Exception as e:
    logging.warning(f"Could not load alert.wav: {e}. Alerts will be silent.")
    ALERT_SOUND = None


# ──────────────────────────────────────────────
# Metric functions
# ──────────────────────────────────────────────

def eye_aspect_ratio(eye):
    """6-point EAR (dlib/MediaPipe compatible)."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C > 1e-6 else 0.0


def mouth_aspect_ratio(mouth):
    """
    12-point MAR using three vertical pairs and one horizontal pair.
    mouth is shaped (12, 2) using MOUTH_POINTS indices.

    Layout (indices into the local 12-element array):
        0  = left corner  (landmark 61)
        1  = right corner (landmark 291)
        2  = upper-outer  (landmark 0)
        3  = lower-outer  (landmark 17)
        4  = upper-mid    (landmark 39)
        5  = lower-mid    (landmark 269)
        6  = upper-inner  (landmark 82)
        7  = lower-inner  (landmark 312)
    """
    A = dist.euclidean(mouth[2], mouth[3])   # outer vertical
    B = dist.euclidean(mouth[4], mouth[5])   # mid vertical
    C = dist.euclidean(mouth[6], mouth[7])   # inner vertical
    D = dist.euclidean(mouth[0], mouth[1])   # horizontal width
    return (A + B + C) / (3.0 * D) if D > 1e-6 else 0.0


def head_pose_estimation(shape, focal_length, center):
    """
    Returns (pitch, yaw, roll) in degrees using solvePnP + RQDecomp3x3.
    Uses 6 stable MediaPipe landmarks mapped to a generic 3-D face model.
    """
    # 2-D image points (MediaPipe landmark indices)
    image_points = np.array([
        shape[1],    # nose tip
        shape[152],  # chin
        shape[33],   # left eye left corner
        shape[263],  # right eye right corner
        shape[61],   # left mouth corner
        shape[291],  # right mouth corner
    ], dtype=np.float64)

    # Generic 3-D face model (mm scale)
    model_points = np.array([
        (0.0,    0.0,    0.0),    # nose tip
        (0.0,  -330.0, -65.0),   # chin
        (-225.0, 170.0, -135.0), # left eye corner
        (225.0,  170.0, -135.0), # right eye corner
        (-150.0,-150.0, -125.0), # left mouth corner
        (150.0, -150.0, -125.0), # right mouth corner
    ], dtype=np.float64)

    camera_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1        ]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rvec, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = angles[0], angles[1], angles[2]
    return pitch, yaw, roll


# ──────────────────────────────────────────────
# Filters
# ──────────────────────────────────────────────

class EMAFilter:
    """Exponential moving average over a tuple/array."""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.value is None:
            self.value = x.copy()
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return tuple(self.value.tolist())


class MedianThenEMAFilter:
    """Median over a short buffer → EMA.  Removes spike outliers."""
    def __init__(self, buffer_size=7, ema_alpha=0.35):
        self.buf = deque(maxlen=buffer_size)
        self.ema = EMAFilter(alpha=ema_alpha)

    def update(self, x_tuple):
        self.buf.append(x_tuple)
        med = np.median(np.array(self.buf), axis=0)
        return self.ema.update(med)


# ──────────────────────────────────────────────
# Head position tracker + vibration detector
# ──────────────────────────────────────────────

class HeadPositionFilter:
    def __init__(self,
                 buffer_size=Config.HEAD_POSITION_BUFFER_SIZE,
                 vel_buf_size=Config.VELOCITY_BUFFER_SIZE):
        self.filter  = MedianThenEMAFilter(buffer_size=buffer_size, ema_alpha=0.35)
        self.prev    = None
        self.vel_buf = deque(maxlen=vel_buf_size)

    def update(self, pitch, yaw, roll):
        smoothed = self.filter.update((pitch, yaw, roll))
        vel = (0.0, 0.0, 0.0) if self.prev is None else (
            smoothed[0] - self.prev[0],
            smoothed[1] - self.prev[1],
            smoothed[2] - self.prev[2],
        )
        self.prev = smoothed
        self.vel_buf.append(vel)
        return smoothed

    def get_velocity_variance(self):
        if len(self.vel_buf) < 3:
            return 0.0, 0.0, 0.0
        arr = np.array(self.vel_buf)
        return (float(np.var(arr[:, 0])),
                float(np.var(arr[:, 1])),
                float(np.var(arr[:, 2])))


# ──────────────────────────────────────────────
# Alert manager
# ──────────────────────────────────────────────

class AlertManager:
    MESSAGES = {
        'eye':  "⚠️  DROWSINESS — Eyes closed too long",
        'yawn': "⚠️  DROWSINESS — Yawning detected",
        'head': "⚠️  FATIGUE    — Abnormal head movement",
    }

    def __init__(self):
        self.last_alert_time = 0.0
        self.lock = Lock()

    def trigger_alert(self, alert_type, frame=None):
        now = time.time()
        with self.lock:
            if now - self.last_alert_time < Config.ALERT_COOLDOWN:
                return
            self.last_alert_time = now

        logging.warning(self.MESSAGES.get(alert_type, "⚠️  Alert"))

        if ALERT_SOUND:
            Thread(target=self._play_sound, daemon=True).start()

        if frame is not None:
            Thread(target=self._save_frame,
                   args=(frame.copy(), alert_type), daemon=True).start()

    def _play_sound(self):
        try:
            if ALERT_SOUND is not None:
                ALERT_SOUND.play()
        except Exception as e:
            logging.warning(f"Sound error: {e}")

    def _save_frame(self, frame, alert_type):
        try:
            os.makedirs(Config.ALERT_SAVE_DIR, exist_ok=True)
            ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fname = os.path.join(Config.ALERT_SAVE_DIR,
                                 f"alert_{alert_type}_{ts}.jpg")
            cv2.imwrite(fname, frame)
            logging.info(f"Saved alert frame → {fname}")
        except Exception as e:
            logging.warning(f"Could not save alert frame: {e}")


# ──────────────────────────────────────────────
# Main detector
# ──────────────────────────────────────────────

class DrowsinessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.alert_manager = AlertManager()
        self.head_filter   = HeadPositionFilter()

        # Frame counters
        self.counter_eye  = 0
        self.counter_yawn = 0
        self.counter_head = 0
        self.frame_idx    = 0

        # Camera intrinsics (set on first frame)
        self.focal_length = None
        self.center       = None

        # Calibration
        self.calib_ear  = []
        self.calib_mar  = []
        self.calibrated = False

        # Adaptive thresholds (initialised to fallbacks)
        self.eye_thresh   = Config.EYE_AR_THRESH
        self.mouth_thresh = Config.MOUTH_AR_THRESH

    # ------------------------------------------------------------------
    def _init_camera_params(self, frame_shape):
        if self.focal_length is None:
            h, w = frame_shape[:2]
            self.focal_length = float(w)
            self.center = (w / 2.0, h / 2.0)

    # ------------------------------------------------------------------
    def process_frame(self, frame):
        self._init_camera_params(frame.shape)
        self.frame_idx += 1

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self._draw_no_face(frame)
            return

        face = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]

        # Full landmark array  (shape: [468+, 2])
        shape = np.array(
            [[int(lm.x * w), int(lm.y * h)] for lm in face.landmark]
        )

        # Slice feature points
        left_eye  = shape[LEFT_EYE]
        right_eye = shape[RIGHT_EYE]
        mouth     = shape[MOUTH_POINTS]   # (12, 2)

        # Compute metrics
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # ── Calibration ──────────────────────────────────────────────
        if not self.calibrated:
            self.calib_ear.append(ear)
            self.calib_mar.append(mar)
            if len(self.calib_ear) >= Config.CALIBRATION_FRAMES:
                b_ear = max(float(np.mean(self.calib_ear)), 0.01)
                b_mar = max(float(np.mean(self.calib_mar)), 0.01)
                self.eye_thresh   = b_ear * Config.EYE_AR_MULTIPLIER
                self.mouth_thresh = b_mar * Config.MOUTH_AR_MULTIPLIER
                self.calibrated   = True
                logging.info(
                    f"Calibration done | "
                    f"baseline EAR={b_ear:.3f}  MAR={b_mar:.3f} | "
                    f"thresholds → EAR<{self.eye_thresh:.3f}  MAR>{self.mouth_thresh:.3f}"
                )

        # ── Head pose ────────────────────────────────────────────────
        pitch_raw, yaw_raw, roll_raw = head_pose_estimation(
            shape, self.focal_length, self.center
        )
        pitch, yaw, roll = self.head_filter.update(pitch_raw, yaw_raw, roll_raw)
        pvar, yvar, _    = self.head_filter.get_velocity_variance()

        # ── State update & alerts ────────────────────────────────────
        self._update_counters(ear, mar, pitch, yaw, pvar, yvar)
        self._check_alerts(frame)

        # ── Draw HUD ─────────────────────────────────────────────────
        self._draw_hud(frame, left_eye, right_eye, mouth,
                       ear, mar, pitch, yaw, roll, pvar, yvar)

    # ------------------------------------------------------------------
    def _update_counters(self, ear, mar, pitch, yaw, pvar, yvar):
        eth = self.eye_thresh   if self.calibrated else Config.EYE_AR_THRESH
        mth = self.mouth_thresh if self.calibrated else Config.MOUTH_AR_THRESH

        # Eye counter
        self.counter_eye  = self.counter_eye  + 1 if ear < eth  else 0
        # Yawn counter
        self.counter_yawn = self.counter_yawn + 1 if mar > mth  else 0

        # Head tilt counter (suppress during vibration / road-bump)
        high_velocity = (pvar > Config.VELOCITY_VARIANCE_THRESHOLD or
                         yvar > Config.VELOCITY_VARIANCE_THRESHOLD)
        if high_velocity:
            self.counter_head = 0
        else:
            tilted = (abs(pitch) > Config.HEAD_TILT_SUSTAINED_THRESH or
                      abs(yaw)   > Config.HEAD_TILT_SUSTAINED_THRESH)
            self.counter_head = self.counter_head + 1 if tilted else 0

    # ------------------------------------------------------------------
    def _check_alerts(self, frame):
        if self.counter_eye >= Config.EYE_AR_CONSEC_FRAMES:
            self.alert_manager.trigger_alert('eye', frame)
            self.counter_eye = 0
        if self.counter_yawn >= Config.YAWN_CONSEC_FRAMES:
            self.alert_manager.trigger_alert('yawn', frame)
            self.counter_yawn = 0
        if self.counter_head >= Config.HEAD_TILT_CONSEC_FRAMES:
            self.alert_manager.trigger_alert('head', frame)
            self.counter_head = 0

    # ------------------------------------------------------------------
    def _draw_hud(self, frame, left_eye, right_eye, mouth,
                  ear, mar, pitch, yaw, roll, pvar, yvar):
        # Landmarks
        for pt in np.vstack([left_eye, right_eye, mouth]):
            cv2.circle(frame, tuple(pt.astype(int)), 2, (0, 255, 0), -1)

        # Status colour: red if any counter is elevated
        danger = (self.counter_eye  > Config.EYE_AR_CONSEC_FRAMES  // 2 or
                  self.counter_yawn > Config.YAWN_CONSEC_FRAMES     // 2 or
                  self.counter_head > Config.HEAD_TILT_CONSEC_FRAMES // 2)
        status_color = (0, 0, 255) if danger else (0, 220, 0)

        # Build HUD lines
        if self.calibrated:
            calib_line = (f"EyeThr:{self.eye_thresh:.3f}  "
                          f"MouthThr:{self.mouth_thresh:.3f}", status_color)
        else:
            pct = int(100 * len(self.calib_ear) / Config.CALIBRATION_FRAMES)
            calib_line = (f"Calibrating... {pct}%", (0, 200, 255))

        lines = [
            (f"EAR: {ear:.3f}   (eye closed frames: {self.counter_eye})", (255, 255, 255)),
            (f"MAR: {mar:.3f}   (yawn frames: {self.counter_yawn})",       (255, 255, 255)),
            (f"Head  P={pitch:.1f}  Y={yaw:.1f}  R={roll:.1f}  "
             f"(tilt frames: {self.counter_head})",                         (255, 255, 255)),
            (f"VelVar  P={pvar:.2f}  Y={yvar:.2f}",                        (200, 200, 200)),
            calib_line,
        ]

        y0, dy = 20, 25
        for i, (text, color) in enumerate(lines):
            y = y0 + i * dy
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(frame, (7, y - th - 3), (7 + tw + 4, y + 3),
                          (0, 0, 0), -1)
            cv2.putText(frame, text, (9, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    def _draw_no_face(self, frame):
        msg = "No face detected"
        cv2.putText(frame, msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    detector = DrowsinessDetector()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          Config.FPS_LIMIT)

    if not cap.isOpened():
        logging.error("Cannot open camera — check device index or permissions.")
        return

    logging.info("Drowsiness Detection System started.  Press 'q' to quit.")
    frame_time = 1.0 / Config.FPS_LIMIT

    try:
        while True:
            t0 = time.time()

            ret, frame = cap.read()
            if not ret:
                logging.error("Frame capture failed — camera disconnected?")
                break

            detector.process_frame(frame)
            cv2.imshow("Drowsiness & Fatigue Detection", frame)

            # Soft FPS cap
            elapsed = time.time() - t0
            wait = max(1, int((frame_time - elapsed) * 1000))
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Shutdown complete.")


if __name__ == "__main__":
    main()