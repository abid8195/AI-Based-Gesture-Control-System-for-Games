"""Keyboard output layer for browser or Unity endless runner controls."""

from __future__ import annotations

import time
from typing import Optional

import pyautogui

from gesture_recognition import GestureState


class InputController:
    """Maps detected gestures to keyboard taps with simple cooldowns."""

    def __init__(
        self,
        key_interval: float = 0.2,
        hold_to_repeat_interval: float = 0.35,
    ) -> None:
        pyautogui.FAILSAFE = False
        self.key_interval = key_interval
        self.hold_to_repeat_interval = hold_to_repeat_interval
        self.last_press_time = {
            "left": 0.0,
            "right": 0.0,
            "space": 0.0,
            "down": 0.0,
        }
        self.last_tilt_direction: Optional[str] = None
        self.last_mouth_state = False
        self.last_blink_state = False

    def update(self, gesture_state: GestureState) -> None:
        """Press keys when a fresh gesture or safe repeat is detected."""
        now = time.time()

        if gesture_state.tilt_direction == "LEFT":
            if self._can_repeat("left", now, gesture_state.tilt_direction):
                pyautogui.press("left")
                self.last_press_time["left"] = now
        elif gesture_state.tilt_direction == "RIGHT":
            if self._can_repeat("right", now, gesture_state.tilt_direction):
                pyautogui.press("right")
                self.last_press_time["right"] = now

        if gesture_state.mouth_open and not self.last_mouth_state:
            if now - self.last_press_time["space"] >= self.key_interval:
                pyautogui.press("space")
                self.last_press_time["space"] = now

        if gesture_state.blink and not self.last_blink_state:
            if now - self.last_press_time["down"] >= self.key_interval:
                pyautogui.press("down")
                self.last_press_time["down"] = now

        self.last_tilt_direction = gesture_state.tilt_direction
        self.last_mouth_state = gesture_state.mouth_open
        self.last_blink_state = gesture_state.blink

    def _can_repeat(self, key: str, now: float, current_direction: str) -> bool:
        """Allow an immediate press on direction change, then throttle repeats."""
        if self.last_tilt_direction != current_direction:
            return True
        return now - self.last_press_time[key] >= self.hold_to_repeat_interval
