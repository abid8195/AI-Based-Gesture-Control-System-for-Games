"""Main entry point for the webcam gesture-to-keyboard controller."""

from __future__ import annotations

import time

import cv2

from face_detection import FaceTracker
from gesture_recognition import GestureRecognizer
from input_controller import InputController


CALIBRATION_SECONDS = 3


def draw_overlay(frame, gesture_state, calibration_mode, countdown_text, fps, tracking_ok):
    """Render grouped status, gesture, metric, and warning HUD panels."""
    panel_bg = (24, 24, 24)
    panel_border = (70, 70, 70)
    text = (245, 245, 245)
    muted = (170, 170, 170)
    active_green = (80, 220, 120)
    calibration_yellow = (0, 220, 255)
    danger_red = (40, 40, 230)
    neutral_gray = (150, 150, 150)

    gesture_colors = {
        "Tilt Left": (255, 160, 80),
        "Tilt Right": (220, 220, 80),
        "Mouth Open": (0, 165, 255),
        "Blink": (220, 90, 220),
        "Neutral": neutral_gray,
    }
    gesture_symbols = {
        "Tilt Left": "<",
        "Tilt Right": ">",
        "Mouth Open": "O",
        "Blink": "B",
        "Neutral": "-",
    }

    def draw_panel(x, y, w, h, accent):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), panel_bg, -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), panel_border, 1)
        cv2.rectangle(frame, (x, y), (x + 5, y + h), accent, -1)

    def draw_text(value, x, y, scale=0.55, color=text, thickness=1):
        cv2.putText(
            frame,
            value,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def draw_metric_bar(label, value_text, value, max_value, x, y, color):
        draw_text(label, x, y, 0.5, muted, 1)
        draw_text(value_text, x + 78, y, 0.5, text, 1)

        bar_x = x + 165
        bar_y = y - 12
        bar_w = 92
        bar_h = 8
        fill_w = int(bar_w * min(1.0, max(0.0, value / max_value)))

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (65, 65, 65), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (110, 110, 110), 1)

    _, width = frame.shape[:2]
    if not tracking_ok:
        mode = "TRACKING LOST"
        accent = danger_red
    elif calibration_mode:
        mode = "CALIBRATING"
        accent = calibration_yellow
    else:
        mode = "ACTIVE"
        accent = active_green

    margin = 14
    top_h = 54
    draw_panel(margin, margin, width - (margin * 2), top_h, accent)
    draw_text(mode, margin + 18, margin + 24, 0.65, accent, 2)
    draw_text("Q Quit | R Recalibrate", margin + 18, margin + 45, 0.45, muted, 1)

    fps_text = f"FPS {fps:04.1f}"
    (fps_w, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    draw_text(fps_text, width - margin - fps_w - 16, margin + 34, 0.6, text, 1)

    panel_y = margin + top_h + 12
    left_w = 300

    if calibration_mode or not tracking_ok:
        panel_h = 128
        draw_panel(margin, panel_y, left_w, panel_h, accent)

        if not tracking_ok:
            draw_text("TRACKING", margin + 18, panel_y + 28, 0.55, accent, 2)
            draw_text("No face detected", margin + 18, panel_y + 62, 0.6, text, 2)
            draw_text("Check lighting / camera angle", margin + 18, panel_y + 92, 0.5, muted, 1)
        else:
            draw_text("CALIBRATION", margin + 18, panel_y + 28, 0.55, accent, 2)
            draw_text(countdown_text, margin + 18, panel_y + 62, 0.58, text, 2)
            draw_text("Face camera | Relax expression", margin + 18, panel_y + 92, 0.5, muted, 1)
    else:
        labels = gesture_state.labels if gesture_state.labels else ["Neutral"]
        panel_h = 58 + max(1, len(labels)) * 32
        draw_panel(margin, panel_y, left_w, panel_h, accent)
        draw_text("GESTURES", margin + 18, panel_y + 28, 0.55, accent, 2)

        row_y = panel_y + 62
        for label in labels:
            color = gesture_colors.get(label, text)
            symbol = gesture_symbols.get(label, "*")
            cv2.circle(frame, (margin + 28, row_y - 7), 9, color, -1)
            draw_text(symbol, margin + 23, row_y - 3, 0.38, (20, 20, 20), 2)
            draw_text(label, margin + 48, row_y, 0.58, text, 1)
            row_y += 32

    metrics_w = 306
    metrics_h = 148
    metrics_x = max(margin, width - margin - metrics_w)
    draw_panel(metrics_x, panel_y, metrics_w, metrics_h, active_green if tracking_ok else danger_red)
    draw_text("METRICS", metrics_x + 18, panel_y + 28, 0.55, text if tracking_ok else danger_red, 2)

    if tracking_ok:
        draw_metric_bar(
            "Tilt",
            f"{gesture_state.roll_angle:+.1f} deg",
            abs(gesture_state.roll_angle),
            30.0,
            metrics_x + 18,
            panel_y + 62,
            gesture_colors["Tilt Right"] if gesture_state.roll_angle >= 0 else gesture_colors["Tilt Left"],
        )
        draw_metric_bar(
            "Mouth",
            f"{gesture_state.mouth_ratio:.3f}",
            gesture_state.mouth_ratio,
            0.35,
            metrics_x + 18,
            panel_y + 92,
            gesture_colors["Mouth Open"],
        )
        draw_metric_bar(
            "Blink",
            f"{gesture_state.blink_ratio:.3f}",
            gesture_state.blink_ratio,
            0.35,
            metrics_x + 18,
            panel_y + 122,
            gesture_colors["Blink"],
        )
    else:
        draw_text("Tilt", metrics_x + 18, panel_y + 62, 0.5, muted, 1)
        draw_text("--", metrics_x + 96, panel_y + 62, 0.5, text, 1)
        draw_text("Mouth", metrics_x + 18, panel_y + 92, 0.5, muted, 1)
        draw_text("--", metrics_x + 96, panel_y + 92, 0.5, text, 1)
        draw_text("Blink", metrics_x + 18, panel_y + 122, 0.5, muted, 1)
        draw_text("--", metrics_x + 96, panel_y + 122, 0.5, text, 1)


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
    previous_frame_time = time.time()
    fps = 0.0

    try:
        while True:
            result = tracker.read()
            if result is None:
                print("Unable to read from webcam.")
                break

            now = time.time()
            frame_delta = now - previous_frame_time
            previous_frame_time = now
            if frame_delta > 0:
                fps = 1.0 / frame_delta

            frame = result.frame.copy()
            frame = tracker.draw_landmarks(frame, result.landmarks)
            tracking_ok = result.landmarks is not None

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
                draw_overlay(frame, gesture_state, calibration_mode, countdown_text, fps, tracking_ok)
            else:
                gesture_state = recognizer.process(result.landmarks)
                controller.update(gesture_state)
                draw_overlay(frame, gesture_state, calibration_mode, "", fps, tracking_ok)

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
