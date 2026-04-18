# AI-Based Gesture Control System for Games

This project uses a webcam to detect simple facial gestures and turn them into keyboard inputs for an endless runner game.

If your installed MediaPipe version uses the newer Face Landmarker task API, the app expects a model file at `models/face_landmarker.task`.

## Features

- Head tilt left -> `LEFT`
- Head tilt right -> `RIGHT`
- Mouth open -> `SPACE` for jump
- Eye blink -> `DOWN` for slide
- Neutral-face calibration at startup
- Smoothed gesture detection to reduce noise
- Live webcam preview with face mesh and gesture labels

## Project Files

- `face_detection.py`: webcam capture and MediaPipe Face Mesh tracking
- `gesture_recognition.py`: calibration, smoothing, and gesture detection
- `input_controller.py`: keyboard output using `pyautogui`
- `main.py`: application loop and on-screen overlay

## Install

1. Use Python `3.9` to `3.12`.

   MediaPipe's current PyPI classifiers list support for Python `3.9`, `3.10`, `3.11`, and `3.12`, so Python `3.14` may fail to install the dependency.

2. Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Download the Face Landmarker model file into the project:

```powershell
mkdir models
curl https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task -o models/face_landmarker.task
```

## Run

1. Open the endless runner game you want to control.
2. Make sure the game window can receive keyboard input.
3. Run the controller:

```powershell
python main.py
```

4. Sit in front of the webcam and keep a neutral face for about 3 seconds while calibration runs.
5. Click back into the game window after calibration if the OpenCV preview takes focus.
6. After calibration:
   - Tilt your head left to send `LEFT`
   - Tilt your head right to send `RIGHT`
   - Open your mouth to send `SPACE`
   - Blink to send `DOWN`

## Tips

- Press `q` to quit.
- Press `r` to recalibrate.
- If controls feel too sensitive, adjust the thresholds in `gesture_recognition.py`.
- For best results, use even lighting and keep your face centered in frame.

## Notes

- `pyautogui` sends keys to the currently focused window, so click into the game after calibration if needed.
- Browser-based endless runners and keyboard-controlled Unity prototypes should both work as long as they listen for standard keyboard input.
