"""
doremi.py - Curwen solfege hand sign detector that plays musical notes.

Usage:
    python doremi.py             # normal mode (uses trained model if available)
    python doremi.py --collect   # record training samples
    python doremi.py --train     # fit model from recorded samples
    python doremi.py --debug     # print live feature values
"""

from __future__ import annotations

import sys
import time
from collections import deque

import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

try:
    import sounddevice as sd
    AUDIO_BACKEND = "sounddevice"
except Exception:
    try:
        import pygame
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
        AUDIO_BACKEND = "pygame"
    except Exception:
        AUDIO_BACKEND = None
        print("Warning: no audio backend found. Install sounddevice or pygame.")

# ── Constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 44100
DECAY_RATE = 3.0
NOTE_DURATION = 1.5  # seconds

HOLD_FRAMES = 6         # frames gesture must be stable before triggering
CONFIDENCE_THRESHOLD = 0.55
MIN_PLAY_INTERVAL = 0.3  # seconds between plays

NOTE_FREQS = {
    # Natural notes — right hand
    "Do_lo":  261.63,  # C4
    "Re_lo":  293.66,  # D4
    "Mi_lo":  329.63,  # E4
    "Fa_lo":  349.23,  # F4
    "Sol_lo": 392.00,  # G4
    "La_lo":  440.00,  # A4
    "Ti_lo":  493.88,  # B4
    "Do_hi":  523.25,  # C5
    "Re_hi":  587.33,  # D5
    "Mi_hi":  659.25,  # E5
    "Fa_hi":  698.46,  # F5
    "Sol_hi": 783.99,  # G5
    "La_hi":  880.00,  # A5
    "Ti_hi":  987.77,  # B5
    # Semitones — left hand
    "Do#_lo": 277.18,  # C#4
    "Re#_lo": 311.13,  # D#4
    "Fa#_lo": 369.99,  # F#4
    "Sol#_lo":415.30,  # G#4
    "La#_lo": 466.16,  # A#4
    "Do#_hi": 554.37,  # C#5
    "Re#_hi": 622.25,  # D#5
    "Fa#_hi": 739.99,  # F#5
    "Sol#_hi":830.61,  # G#5
    "La#_hi": 932.33,  # A#5
}

# y threshold (normalized 0-1): wrist above this = high octave
DO_SPLIT_Y = 0.5

# Colors per gesture (BGR) — hi variants brighter, semitones warm orange/yellow
NOTE_COLORS = {
    "Do_lo":  (60,  60,  200), "Do_hi":  (100, 100, 255),
    "Re_lo":  (60,  120, 200), "Re_hi":  (100, 180, 255),
    "Mi_lo":  (60,  200, 160), "Mi_hi":  (100, 255, 210),
    "Fa_lo":  (60,  200,  60), "Fa_hi":  (100, 255, 100),
    "Sol_lo": (60,  160, 200), "Sol_hi": (100, 220, 255),
    "La_lo":  (160,  60, 200), "La_hi":  (210, 100, 255),
    "Ti_lo":   (200,  60, 120), "Ti_hi":   (255, 100, 180),
    # Semitones — warm orange/amber
    "Do#_lo":  (0,  140, 255), "Do#_hi":  (0,  180, 255),
    "Re#_lo":  (0,  165, 255), "Re#_hi":  (0,  210, 255),
    "Fa#_lo":  (0,  190, 220), "Fa#_hi":  (0,  230, 255),
    "Sol#_lo": (0,  160, 200), "Sol#_hi": (0,  200, 240),
    "La#_lo":  (0,  130, 180), "La#_hi":  (0,  170, 220),
}

# Right-hand base gesture → sharpened version (Mi and Ti have no semitone)
SEMITONE_MAP = {
    "Do":  "Do#",
    "Re":  "Re#",
    "Fa":  "Fa#",
    "Sol": "Sol#",
    "La":  "La#",
}

# MediaPipe landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

DEBUG = "--debug" in sys.argv

def _arg(flag: str, default: int) -> int:
    try:
        return int(sys.argv[sys.argv.index(flag) + 1])
    except (ValueError, IndexError):
        return default

CAMERA_INDEX = _arg("--camera", 0)

# ── Audio ─────────────────────────────────────────────────────────────────────

def synthesize_note(freq: float) -> np.ndarray:
    """Synthesize a piano-like pluck tone at the given frequency."""
    t = np.linspace(0, NOTE_DURATION, int(SAMPLE_RATE * NOTE_DURATION), endpoint=False)
    wave = (
        0.6 * np.sin(2 * np.pi * freq * t)
        + 0.3 * np.sin(4 * np.pi * freq * t)
        + 0.1 * np.sin(6 * np.pi * freq * t)
    )
    envelope = np.exp(-DECAY_RATE * t)
    samples = (wave * envelope * 0.8).astype(np.float32)
    return samples


def play_note(gesture: str) -> None:
    freq = NOTE_FREQS[gesture]
    samples = synthesize_note(freq)
    if AUDIO_BACKEND == "sounddevice":
        sd.play(samples, samplerate=SAMPLE_RATE)
    elif AUDIO_BACKEND == "pygame":
        import pygame
        pcm = (samples * 32767).astype(np.int16)
        sound = pygame.sndarray.make_sound(pcm)
        sound.play()


# ── Feature Extraction ────────────────────────────────────────────────────────

MODEL_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
SAMPLES_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples.npz")
GESTURE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture_model.pkl")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Hand skeleton connections for drawing
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


def ensure_model() -> None:
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~6 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def get_landmarks(result) -> np.ndarray | None:
    """Return (21, 3) float32 array of normalized landmarks, or None."""
    if not result.hand_landmarks:
        return None
    hand = result.hand_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)


def get_both_hands(result) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (right_lm, left_lm) from the user's perspective.
    After horizontal flip: MediaPipe 'Left' label = user's right hand."""
    right_lm = left_lm = None
    if not result.hand_landmarks:
        return None, None
    for i, hand in enumerate(result.hand_landmarks):
        lm = np.array([[p.x, p.y, p.z] for p in hand], dtype=np.float32)
        label = result.handedness[i][0].category_name
        if label == "Left":   # after flip = user's right
            right_lm = lm
        else:                 # after flip = user's left
            left_lm = lm
    return right_lm, left_lm


def left_hand_active(lm: np.ndarray) -> bool:
    """True if any finger is raised on the left hand (semitone modifier)."""
    return any([
        finger_extended(lm, INDEX_TIP,  INDEX_MCP),
        finger_extended(lm, MIDDLE_TIP, MIDDLE_MCP),
        finger_extended(lm, RING_TIP,   RING_MCP),
        finger_extended(lm, PINKY_TIP,  PINKY_MCP),
    ])


def draw_landmarks(frame: np.ndarray, lm: np.ndarray) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm[i][0] * w), int(lm[i][1] * h)) for i in range(21)]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (80, 80, 80), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 200, 255), -1)


def finger_extended(lm: np.ndarray, tip: int, mcp: int, threshold: float = 1.6) -> bool:
    """True if the finger tip is farther from the wrist than its MCP (scaled)."""
    return (
        np.linalg.norm(lm[tip] - lm[WRIST])
        > np.linalg.norm(lm[mcp] - lm[WRIST]) * threshold
    )


def thumb_extended(lm: np.ndarray) -> bool:
    hand_scale = np.linalg.norm(lm[WRIST] - lm[MIDDLE_MCP])
    return np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_MCP]) > hand_scale * 0.6


def palm_normal(lm: np.ndarray) -> np.ndarray:
    v1 = lm[INDEX_MCP] - lm[WRIST]
    v2 = lm[PINKY_MCP] - lm[WRIST]
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-6 else np.zeros(3)


def hand_axis(lm: np.ndarray) -> np.ndarray:
    """Unit vector from wrist toward middle finger tip."""
    ax = lm[MIDDLE_TIP] - lm[WRIST]
    norm = np.linalg.norm(ax)
    return ax / norm if norm > 1e-6 else np.zeros(3)


def index_direction(lm: np.ndarray) -> np.ndarray:
    d = lm[INDEX_TIP] - lm[INDEX_MCP]
    norm = np.linalg.norm(d)
    return d / norm if norm > 1e-6 else np.zeros(3)


def thumb_direction(lm: np.ndarray) -> np.ndarray:
    d = lm[THUMB_TIP] - lm[THUMB_MCP]
    norm = np.linalg.norm(d)
    return d / norm if norm > 1e-6 else np.zeros(3)


def finger_avg_ratio(lm: np.ndarray) -> float:
    """Average ratio of fingertip distance to MCP distance from wrist.
    ~1.0 = fully curled, ~1.3 = softly bent, ~1.7+ = fully extended."""
    tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    mcps = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    ratios = []
    for tip, mcp in zip(tips, mcps):
        mcp_d = np.linalg.norm(lm[mcp] - lm[WRIST])
        if mcp_d > 1e-6:
            ratios.append(np.linalg.norm(lm[tip] - lm[WRIST]) / mcp_d)
    return float(np.mean(ratios)) if ratios else 1.0


# ── Gesture Classifiers ───────────────────────────────────────────────────────

def classify_gesture(lm: np.ndarray) -> tuple[str | None, float]:
    """Return (gesture_name, confidence) or (None, 0.0).

    Priority order: Ti → La → Fa → Do → Sol → Mi → Re
    (La and Fa before Do because all three have curled fingers;
    La/Fa are more specific and must be checked first.)
    """
    idx = finger_extended(lm, INDEX_TIP, INDEX_MCP)
    mid = finger_extended(lm, MIDDLE_TIP, MIDDLE_MCP)
    rng = finger_extended(lm, RING_TIP, RING_MCP)
    pnk = finger_extended(lm, PINKY_TIP, PINKY_MCP)
    thm = thumb_extended(lm)

    pn   = palm_normal(lm)
    ax   = hand_axis(lm)
    idr  = index_direction(lm)
    tdr  = thumb_direction(lm)
    ratio = finger_avg_ratio(lm)

    if DEBUG:
        print(
            f"ext=[{int(idx)},{int(mid)},{int(rng)},{int(pnk)}] "
            f"thm={int(thm)} ratio={ratio:.2f} "
            f"pn=[{pn[0]:.2f},{pn[1]:.2f},{pn[2]:.2f}] "
            f"ax=[{ax[0]:.2f},{ax[1]:.2f},{ax[2]:.2f}] "
            f"idr.y={idr[1]:.2f} tdr.y={tdr[1]:.2f}"
        )

    # ── Ti: only index finger pointing straight up, rest curled ──────────────
    if idx and not mid and not rng and not pnk:
        score = sum([
            idx, not mid, not rng, not pnk,
            idr[1] < -0.6,      # index tip above index MCP (up in image = negative y)
        ]) / 5
        if score >= CONFIDENCE_THRESHOLD:
            return "Ti", score

    # ── La: curved relaxed hand (like holding a small ball), softly bent ─────
    # Fingers are neither fully curled nor fully extended — ratio in middle range.
    # Must come before Do (fully curled fist) to catch the open-curve shape.
    if not idx and not mid and not rng and not pnk:
        score = sum([
            not idx, not mid, not rng, not pnk,
            1.05 < ratio < 1.5,   # softly bent, not a tight fist
        ]) / 5
        if score >= CONFIDENCE_THRESHOLD:
            return "La", score

    # ── Fa: loosely closed fingers, thumb pointing down, slight downward angle ─
    # Must come before Do (also curled) because thumb-down is the key signal.
    if not idx and not mid and not rng and not pnk:
        score = sum([
            not idx, not mid, not rng, not pnk,
            tdr[1] > 0.45,      # thumb tip below thumb MCP = thumb pointing down
            ax[1] > -0.1,       # hand axis NOT pointing upward (neutral or slightly down)
        ]) / 6
        if score >= CONFIDENCE_THRESHOLD:
            return "Fa", score

    # ── Do: tight closed fist — octave resolved later by wrist height ────────
    if not idx and not mid and not rng and not pnk and not thm:
        score = sum([
            not idx, not mid, not rng, not pnk, not thm,
            ratio < 1.15,       # fingers tightly curled (small tip-to-MCP ratio)
        ]) / 6
        if score >= CONFIDENCE_THRESHOLD:
            return "Do", score

    # ── Sol: flat horizontal hand, DORSUM (back of hand) facing camera ────────
    # All fingers extended, horizontal, palm faces AWAY from camera (pn.z < 0).
    if idx and mid and rng and pnk:
        score = sum([
            idx, mid, rng, pnk,
            abs(ax[1]) < 0.45,  # hand is roughly horizontal
            pn[2] < -0.25,      # palm normal points away from camera = dorsum toward camera
        ]) / 6
        if score >= CONFIDENCE_THRESHOLD:
            return "Sol", score

    # ── Mi: flat horizontal hand, PALM facing downward (toward the floor) ─────
    # All fingers extended, horizontal, palm faces down (pn.y > 0 in image space
    # because y increases downward, so positive pn.y = palm normal points down).
    if idx and mid and rng and pnk:
        score = sum([
            idx, mid, rng, pnk,
            abs(ax[1]) < 0.45,  # hand is roughly horizontal
            pn[1] > 0.3,        # palm normal points downward = palm faces floor
        ]) / 6
        if score >= CONFIDENCE_THRESHOLD:
            return "Mi", score

    # ── Re: four fingers extended, diagonal upward (like a ramp/slope) ────────
    if idx and mid and rng and pnk:
        score = sum([
            idx, mid, rng, pnk,
            ax[1] < -0.3,       # hand axis pointing upward (negative y = up in image)
            pn[2] > 0.2,        # palm has some camera-facing component
        ]) / 6
        if score >= CONFIDENCE_THRESHOLD:
            return "Re", score

    return None, 0.0


def resolve_octave(gesture: str | None, lm: np.ndarray | None) -> str | None:
    """Append _hi or _lo to any gesture based on wrist height on screen."""
    if gesture is None or lm is None:
        return gesture
    suffix = "_hi" if lm[WRIST][1] < DO_SPLIT_Y else "_lo"
    return gesture + suffix


# ── ML helpers ───────────────────────────────────────────────────────────────

# Key → gesture name for collection
COLLECT_KEYS = {
    ord('d'): 'Do',
    ord('r'): 'Re',
    ord('m'): 'Mi',
    ord('f'): 'Fa',
    ord('s'): 'Sol',
    ord('l'): 'La',
    ord('t'): 'Ti',
}


def extract_features(lm: np.ndarray) -> np.ndarray:
    """63-dim feature vector: landmarks translated to wrist origin, scale-normalised."""
    scale = np.linalg.norm(lm[MIDDLE_MCP] - lm[WRIST])
    if scale < 1e-6:
        scale = 1.0
    return ((lm - lm[WRIST]) / scale).flatten()


def make_landmarker():
    ensure_model()
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    options = HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def collect_samples() -> None:
    """Interactive sample collection. Hold a gesture and press its key to record."""
    landmarker = make_landmarker()
    cap = cv2.VideoCapture(CAMERA_INDEX)

    # Load existing samples so we can append
    if os.path.exists(SAMPLES_PATH):
        data = np.load(SAMPLES_PATH, allow_pickle=True)
        X = list(data['X'])
        y = list(data['y'])
        print(f"Loaded {len(X)} existing samples.")
    else:
        X, y = [], []

    counts: dict[str, int] = {}
    for name in COLLECT_KEYS.values():
        counts[name] = sum(1 for label in y if label == name)

    print("\nCollection mode — hold a gesture and press its key to record a sample.")
    print("  D=Do  R=Re  M=Mi  F=Fa  S=Sol  L=La  T=Ti   Q=save & quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))
        lm = get_landmarks(result)

        if lm is not None:
            draw_landmarks(frame, lm)

        # Instructions overlay
        y_pos = 30
        cv2.putText(frame, "COLLECT MODE", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        y_pos += 30
        for key_char, name in [('D','Do'),('R','Re'),('M','Mi'),('F','Fa'),
                                ('S','Sol'),('L','La'),('T','Ti')]:
            color = NOTE_COLORS.get(name, (200, 200, 200))
            cv2.putText(frame, f"{key_char}: {name} ({counts.get(name,0)})",
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            y_pos += 22
        cv2.putText(frame, "Q: save & quit", (10, y_pos + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in COLLECT_KEYS and lm is not None:
            name = COLLECT_KEYS[key]
            X.append(extract_features(lm))
            y.append(name)
            counts[name] = counts.get(name, 0) + 1
            # Flash feedback
            color = NOTE_COLORS.get(name, (255, 255, 255))
            cv2.putText(frame, f"+ {name}", (frame.shape[1]//2 - 40, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        cv2.imshow("Doremi — Collect", frame)

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()

    if X:
        np.savez(SAMPLES_PATH, X=np.array(X), y=np.array(y))
        print(f"\nSaved {len(X)} total samples to {SAMPLES_PATH}")
        for name, count in sorted(counts.items()):
            print(f"  {name}: {count}")
        print("\nRun with --train to fit the model.")
    else:
        print("No samples recorded.")


def train_model() -> None:
    """Train a KNN classifier from collected samples and save it."""
    if not os.path.exists(SAMPLES_PATH):
        print("No samples found. Run with --collect first.")
        return

    import joblib
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    data = np.load(SAMPLES_PATH, allow_pickle=True)
    X, y = data['X'], data['y']
    print(f"Training on {len(X)} samples across {len(set(y))} gestures...")

    clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    scores = cross_val_score(clf, X, y, cv=min(5, len(X)//len(set(y))))
    print(f"Cross-val accuracy: {scores.mean():.1%} ± {scores.std():.1%}")

    clf.fit(X, y)
    joblib.dump(clf, GESTURE_MODEL)
    print(f"Model saved to {GESTURE_MODEL}")
    print("Run normally to use it.")


def ml_classify(clf, lm: np.ndarray) -> tuple[str | None, float]:
    """Classify using trained KNN model."""
    features = extract_features(lm).reshape(1, -1)
    gesture = clf.predict(features)[0]
    confidence = clf.predict_proba(features)[0].max()
    if confidence < CONFIDENCE_THRESHOLD:
        return None, confidence
    return gesture, float(confidence)


# ── Main Loop ─────────────────────────────────────────────────────────────────

def main():
    # Load trained model if available
    clf = None
    if os.path.exists(GESTURE_MODEL):
        import joblib
        clf = joblib.load(GESTURE_MODEL)
        print(f"Using trained model: {GESTURE_MODEL}")
    else:
        print("No trained model found — using rule-based classifier.")
        print("Run with --collect then --train to improve accuracy.")

    landmarker = make_landmarker()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        sys.exit(1)

    gesture_history: deque[str | None] = deque(maxlen=HOLD_FRAMES)
    last_played: str | None = None
    last_play_time: float = 0.0
    fps_time = time.time()
    fps = 0.0

    print("Doremi running — show a Curwen hand sign to play a note. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        right_lm, left_lm = get_both_hands(result)

        if right_lm is None:
            gesture, confidence = None, 0.0
        elif clf is not None:
            gesture, confidence = ml_classify(clf, right_lm)
        else:
            gesture, confidence = classify_gesture(right_lm)

        # Apply semitone modifier if left hand is raised
        if gesture is not None and left_lm is not None and left_hand_active(left_lm):
            gesture = SEMITONE_MAP.get(gesture, gesture)

        gesture = resolve_octave(gesture, right_lm)

        gesture_history.append(gesture)

        if right_lm is not None:
            draw_landmarks(frame, right_lm)
        if left_lm is not None:
            draw_landmarks(frame, left_lm)

        # Trigger note if gesture has been stable for HOLD_FRAMES
        if (
            len(gesture_history) == HOLD_FRAMES
            and len(set(gesture_history)) == 1
            and gesture is not None
        ):
            if gesture != last_played or time.time() - last_play_time > 1.0:
                if time.time() - last_play_time > MIN_PLAY_INTERVAL:
                    play_note(gesture)
                    last_played = gesture
                    last_play_time = time.time()
        elif gesture is None:
            last_played = None

        # ── Overlay ──────────────────────────────────────────────────────────
        _, w = frame.shape[:2]

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - fps_time, 1e-6))
        fps_time = now
        cv2.putText(frame, f"FPS {fps:.0f}", (w - 90, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        DISPLAY_NAMES = {k: k.replace("_lo", " ↓").replace("_hi", " ↑")
                         for k in NOTE_FREQS}
        if gesture:
            color = NOTE_COLORS.get(gesture, (255, 255, 255))
            label = DISPLAY_NAMES.get(gesture, gesture)
            cv2.putText(frame, label, (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)
            cv2.putText(frame, f"conf {confidence:.2f}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        else:
            cv2.putText(frame, "—", (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (100, 100, 100), 2)

        if gesture:
            n_matching = sum(1 for g in gesture_history if g == gesture)
            bar_w = int((n_matching / HOLD_FRAMES) * 160)
            cv2.rectangle(frame, (20, 105), (20 + 160, 118), (60, 60, 60), -1)
            color = NOTE_COLORS.get(gesture, (255, 255, 255))
            cv2.rectangle(frame, (20, 105), (20 + bar_w, 118), color, -1)

        cv2.imshow("Doremi", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--collect" in sys.argv:
        collect_samples()
    elif "--train" in sys.argv:
        train_model()
    else:
        main()
