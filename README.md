# Doremi Hand Sign Music System

A real-time computer vision system that recognizes Curwen solfege hand signs using webcam input and plays corresponding musical notes through gesture recognition.

The project combines MediaPipe hand tracking, gesture classification, machine learning, and audio synthesis to create an interactive gesture-controlled musical interface.

## Overview

This system detects Curwen solfege hand signs from a webcam feed and maps them to musical notes in real time.

The implementation supports:

* Real-time hand landmark tracking
* Rule-based gesture classification
* Optional machine learning gesture recognition
* Audio synthesis and playback
* Octave detection
* Semitone modifiers
* Training-data collection pipeline
* Live visual feedback

## Curwen Solfege Hand Signs

The system recognizes Curwen solfege hand signs in real time using MediaPipe hand tracking and gesture classification.

<img width="554" height="242" alt="image" src="https://github.com/user-attachments/assets/8ddfa128-a7fa-4bd3-bda0-fc560a47a6f3" />

## Technologies Used

* Python
* MediaPipe
* OpenCV
* NumPy
* scikit-learn
* sounddevice / pygame

## How It Works

### Hand Tracking

The system uses MediaPipe to detect and track 21 hand landmarks in real time.

### Gesture Recognition

Gestures are classified using:

* finger extension states
* palm orientation
* thumb direction
* hand-axis direction
* relative landmark geometry

The project supports both:

* rule-based classification
* machine learning classification using KNN

### Audio Synthesis

Detected gestures trigger synthesized musical notes using additive waveform generation and exponential decay envelopes.

### Musical Logic

* Right hand controls natural notes
* Left hand acts as a semitone modifier
* Hand height determines octave

## Gesture Examples

### Natural Note Detection

A closed fist with the right hand is recognized as **Do**.

<img width="1406" height="778" alt="Screenshot 2026-05-28 at 10 36 56" src="https://github.com/user-attachments/assets/629600b7-14df-4666-9235-6e7ec47fcbc2" />

---

### Semitone Modifier

When the left hand raises one finger, the system applies a semitone modifier, turning **Do** into **Do#**.

<img width="1405" height="774" alt="Screenshot 2026-05-28 at 10 36 43" src="https://github.com/user-attachments/assets/bcc14698-5e16-44db-9431-385e3dae0ac9" />

---

### Octave Detection

The vertical position of the main hand controls octave shifts.  
When the hand moves to the upper half of the screen, the note is played one octave higher.

<img width="821" height="450" alt="Screenshot 2026-05-28 at 10 37 56" src="https://github.com/user-attachments/assets/b94139e3-1ea7-482e-9ff0-1fc309ba88ee" />

## Supported Notes

The system supports:

* Do
* Re
* Mi
* Fa
* Sol
* La
* Ti

including:

* high/low octaves
* semitone variants

## Running the Project

Install dependencies:

```bash id="ql5vms"
pip install -r requirements.txt
```

Run normally:

```bash id="t9hsp9"
python doremi_hand_sign_detector.py
```

Collect training samples:

```bash id="9yojvb"
python doremi_hand_sign_detector.py --collect
```

Train gesture model:

```bash id="1nqg6j"
python doremi_hand_sign_detector.py --train
```

Debug mode:

```bash id="hjlwmn"
python doremi_hand_sign_detector.py --debug
```
