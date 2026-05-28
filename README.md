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
