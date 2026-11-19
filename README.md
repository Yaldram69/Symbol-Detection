# Symbol Detection

Computer vision experiments for logo recognition and hand-sign recognition. The logo part trains a MobileNetV2-based classifier on the Flickr Logos 27 dataset and runs it on a webcam feed. The hand-sign part uses MediaPipe Holistic hand landmarks as input to two Keras models that classify left and right hand gestures.

## Features

- **Logo classification training** (`train.py`): MobileNetV2 with ImageNet weights and a frozen base, 128x128 input, a small dense head, data augmentation, up to 30 epochs with early stopping and learning-rate reduction.
- **Live logo recognition** (`test.py`): reads the webcam, classifies each frame, keeps a 10-frame buffer and draws a label when a class is the majority above 0.7 confidence.
- **Live hand-sign recognition** (`hand-sign-detection-main/run.py`): MediaPipe hand landmarks are fed to separate left and right gesture classifiers.

## Project structure

- `train.py` - trains the logo classifier on `dataset/archive (16)`; saves `mobilenet_logo_detection_model.h5` and `label_encoder.npy`.
- `test.py` - webcam demo that loads `mobilenet_logo_detection_model.h5` and `label_encoder.npy`.
- `mobilenet_logo_detection_model.h5` - trained model used by `test.py`.
- `label_encoder.npy` - class names saved by `train.py`.
- `logo_detection_model.h5`, `simple_logo_detection_model.h5`, `enhanced_logo_detection_model.h5` - other logo models; the included scripts do not load them.
- `hand-sign-detection-main/run.py` - MediaPipe + gesture-model webcam demo.
- `hand-sign-detection-main/left_hand_gesturev3.h5`, `hand-sign-detection-main/handgesture_rightv2.h5` - gesture models loaded by `run.py`.
- `hand-sign-detection-main/Hand Gesture.ipynb` - notebook for the gesture models (not covered here).
- `hand-sign-detection-main/README.md` - README carried over from the hand-sign sub-project.
- `dataset/archive (16)` - Flickr Logos 27 dataset (image folder plus training/query annotation text files). Used by `train.py`.
- `dataset/archive (14)` - `LogoDatabase.csv` and a `Logos` folder of sample logo images. Not read by the included scripts.

## Requirements

Imports seen in the code: `cv2`, `numpy`, `tensorflow`, `sklearn`, `mediapipe`.

```bash
pip install tensorflow opencv-python numpy scikit-learn mediapipe
```

## Installation

```bash
git clone <this-repo-url>
cd Symbol-Detection
pip install tensorflow opencv-python numpy scikit-learn mediapipe
```

No other setup step is required. A webcam is needed for `test.py` and `run.py`.

## Usage

Run from the project root, because `train.py` and `test.py` use relative paths.

```bash
# Train the logo classifier (expects ./dataset/archive (16)/flickr_logos_27_dataset/...)
python train.py

# Live logo recognition from webcam (expects mobilenet_logo_detection_model.h5 and label_encoder.npy)
python test.py

# Live hand-sign recognition
# First edit hand-sign-detection-main/run.py: the model paths are hard-coded to
# D:\PythonProjects\Symbol Detection\hand-sign-detection-main\...
python hand-sign-detection-main/run.py
```

Press `q` in the video window to quit `test.py` and `run.py`.

## Notes

- `train.py` overwrites `mobilenet_logo_detection_model.h5` and `label_encoder.npy` when run.
- `test.py` classifies the whole frame. It draws a border around the frame and a label; it does not localize the logo.
- `run.py` has known problems: the `LabelEncoder` it creates is never fitted, so the `le.inverse_transform` call that builds `label_names` is expected to raise an error. The `class_name` list in the same file is not used. Model paths are absolute Windows paths and must be changed.
- The `Hand Gesture.ipynb` notebook was not reviewed in detail, and the gesture models' training data is not included.
- The `dataset` folder is part of this copy. If it is too large for your repository, remove it and download Flickr Logos 27 separately into `dataset/archive (16)/flickr_logos_27_dataset/`.
