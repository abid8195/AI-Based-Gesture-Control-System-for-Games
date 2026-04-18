"""Main entry point for the webcam gesture-to-keyboard controller."""

from __future__ import annotations

import time

import cv2

from face_detection import FaceTracker
from gesture_recognition import GestureRecognizer
from input_controller import InputController


CALIBRATION_SECONDS = 3


def draw_overlay(frame, gesture_state, calibration_mode, countdown_text):
    """Render status text for calibration and active gesture labels."""
    y = 30
    if calibration_mode:
        cv2.putText(
            frame,
            countdown_text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        y += 35
    else:
        cv2.putText(
            frame,
            "Calibration complete - press Q to quit",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        y += 35

    labels = gesture_state.labels if gesture_state.labels else ["Neutral"]
    for label in labels:
        cv2.putText(
            frame,
            f"Gesture: {label}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        y += 30

    cv2.putText(
        frame,
        f"Tilt: {gesture_state.roll_angle:+.1f} deg",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )
    y += 25
    cv2.putText(
        frame,
        f"Mouth delta: {gesture_state.mouth_ratio:.3f}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )
    y += 25
    cv2.putText(
        frame,
        f"Blink drop: {gesture_state.blink_ratio:.3f}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )


def main() -> None:
    try:
        tracker = FaceTracker()
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc)
        return

    recognizer = GestureRecognizer()
    controller = InputController()

    calibration_start = time.time()
    calibration_mode = True

    try:
        while True:
            result = tracker.read()
            if result is None:
                print("Unable to read from webcam.")
                break

            frame = result.frame.copy()
            frame = tracker.draw_landmarks(frame, result.landmarks)

            if calibration_mode:
                elapsed = time.time() - calibration_start
                remaining = max(0.0, CALIBRATION_SECONDS - elapsed)

                if result.landmarks is not None:
                    recognizer.collect_calibration_sample(result.landmarks)

                countdown_text = f"Hold a neutral face... {remaining:.1f}s"
                gesture_state = recognizer.process(result.landmarks)

                if elapsed >= CALIBRATION_SECONDS:
                    calibration_mode = not recognizer.finalize_calibration()
                    if calibration_mode:
                        calibration_start = time.time()
                        countdown_text = "Calibration failed - keep face visible"
                    else:
                        gesture_state = recognizer.process(result.landmarks)
                        countdown_text = "Calibration complete"
                draw_overlay(frame, gesture_state, calibration_mode, countdown_text)
            else:
                gesture_state = recognizer.process(result.landmarks)
                controller.update(gesture_state)
                draw_overlay(frame, gesture_state, calibration_mode, "")

            cv2.imshow("AI-Based Gesture Control System for Games", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                recognizer = GestureRecognizer()
                calibration_start = time.time()
                calibration_mode = True

    finally:
        tracker.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
