"""Gesture calibration, smoothing, and classification logic."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import numpy as np


@dataclass
class GestureState:
    """Current smoothed gesture measurements and labels."""

    roll_angle: float = 0.0
    mouth_ratio: float = 0.0
    blink_ratio: float = 0.0
    tilt_direction: Optional[str] = None
    mouth_open: bool = False
    blink: bool = False
    labels: list[str] = field(default_factory=list)


class GestureRecognizer:
    """Turns face landmarks into stable gesture labels."""

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14
    MOUTH_LEFT = 78
    MOUTH_RIGHT = 308
    FOREHEAD = 10
    CHIN = 152

    def __init__(
        self,
        smoothing_window: int = 5,
        tilt_threshold_degrees: float = 8.0,
        mouth_open_threshold: float = 0.18,
        blink_threshold: float = 0.23,
    ) -> None:
        self.smoothing_window = smoothing_window
        self.tilt_threshold_degrees = tilt_threshold_degrees
        self.mouth_open_threshold = mouth_open_threshold
        self.blink_threshold = blink_threshold

        self.roll_history: Deque[float] = deque(maxlen=smoothing_window)
        self.mouth_history: Deque[float] = deque(maxlen=smoothing_window)
        self.blink_history: Deque[float] = deque(maxlen=smoothing_window)

        self.calibration_samples: list[Dict[str, float]] = []
        self.baseline = {
            "roll": 0.0,
            "mouth_ratio": 0.0,
            "left_ear": 0.0,
            "right_ear": 0.0,
        }

    def collect_calibration_sample(self, landmarks: list) -> None:
        """Store one neutral-face sample for later averaging."""
        metrics = self._compute_metrics(landmarks)
        if metrics is not None:
            self.calibration_samples.append(metrics)

    def finalize_calibration(self) -> bool:
        """Average the collected neutral samples into a stable baseline."""
        if not self.calibration_samples:
            return False

        self.baseline = {
            key: float(np.mean([sample[key] for sample in self.calibration_samples]))
            for key in self.baseline
        }
        self.calibration_samples.clear()
        self.roll_history.clear()
        self.mouth_history.clear()
        self.blink_history.clear()
        return True

    def process(self, landmarks: Optional[list]) -> GestureState:
        """Return smoothed gesture state for the current frame."""
        state = GestureState()
        if landmarks is None:
            return state

        metrics = self._compute_metrics(landmarks)
        if metrics is None:
            return state

        roll_delta = metrics["roll"] - self.baseline["roll"]
        mouth_delta = max(0.0, metrics["mouth_ratio"] - self.baseline["mouth_ratio"])
        baseline_blink = (self.baseline["left_ear"] + self.baseline["right_ear"]) / 2.0
        current_blink = (metrics["left_ear"] + metrics["right_ear"]) / 2.0
        blink_drop = max(0.0, baseline_blink - current_blink)

        self.roll_history.append(roll_delta)
        self.mouth_history.append(mouth_delta)
        self.blink_history.append(blink_drop)

        state.roll_angle = float(np.mean(self.roll_history))
        state.mouth_ratio = float(np.mean(self.mouth_history))
        state.blink_ratio = float(np.mean(self.blink_history))

        if state.roll_angle <= -self.tilt_threshold_degrees:
            state.tilt_direction = "LEFT"
            state.labels.append("Tilt Left")
        elif state.roll_angle >= self.tilt_threshold_degrees:
            state.tilt_direction = "RIGHT"
            state.labels.append("Tilt Right")

        state.mouth_open = state.mouth_ratio >= self.mouth_open_threshold
        if state.mouth_open:
            state.labels.append("Mouth Open")

        state.blink = state.blink_ratio >= self.blink_threshold
        if state.blink:
            state.labels.append("Blink")

        return state

    def _compute_metrics(self, landmarks: list) -> Optional[Dict[str, float]]:
        """Compute raw geometric metrics from MediaPipe landmarks."""
        if hasattr(landmarks, "landmark"):
            landmarks = landmarks.landmark

        try:
            left_eye_outer = landmarks[33]
            right_eye_outer = landmarks[263]
            dx = right_eye_outer.x - left_eye_outer.x
            dy = right_eye_outer.y - left_eye_outer.y
            roll = float(np.degrees(np.arctan2(dy, dx)))

            mouth_vertical = self._distance(landmarks[self.MOUTH_TOP], landmarks[self.MOUTH_BOTTOM])
            mouth_horizontal = self._distance(landmarks[self.MOUTH_LEFT], landmarks[self.MOUTH_RIGHT])
            mouth_ratio = mouth_vertical / max(mouth_horizontal, 1e-6)

            left_ear = self._eye_aspect_ratio(landmarks, self.LEFT_EYE)
            right_ear = self._eye_aspect_ratio(landmarks, self.RIGHT_EYE)
        except (IndexError, TypeError):
            return None

        return {
            "roll": roll,
            "mouth_ratio": mouth_ratio,
            "left_ear": left_ear,
            "right_ear": right_ear,
        }

    @staticmethod
    def _distance(point_a, point_b) -> float:
        return float(np.hypot(point_a.x - point_b.x, point_a.y - point_b.y))

    def _eye_aspect_ratio(self, landmarks: list, indices: list[int]) -> float:
        left_corner = landmarks[indices[0]]
        top_inner = landmarks[indices[1]]
        top_outer = landmarks[indices[2]]
        right_corner = landmarks[indices[3]]
        bottom_outer = landmarks[indices[4]]
        bottom_inner = landmarks[indices[5]]

        vertical_1 = self._distance(top_inner, bottom_inner)
        vertical_2 = self._distance(top_outer, bottom_outer)
        horizontal = self._distance(left_corner, right_corner)
        return (vertical_1 + vertical_2) / (2.0 * max(horizontal, 1e-6))
