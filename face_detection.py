"""Webcam face tracking utilities built on OpenCV and MediaPipe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"


@dataclass
class FaceResult:
    """Container for the latest frame and detected face landmarks."""

    frame: np.ndarray
    rgb_frame: np.ndarray
    landmarks: Optional[list]
    image_width: int
    image_height: int


class FaceTracker:
    """Handles webcam capture and MediaPipe face landmark inference."""

    FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
        379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
        234, 127, 162, 21, 54, 103, 67, 109, 10,
    ]
    LEFT_EYE = [33, 160, 158, 133, 153, 144, 33]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380, 362]
    MOUTH = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 78]

    def __init__(
        self,
        camera_index: int = 0,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
    ) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open the webcam.")

        self.use_solutions_api = hasattr(mp, "solutions")
        if self.use_solutions_api:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.drawing_utils = mp.solutions.drawing_utils
            self.drawing_styles = mp.solutions.drawing_styles
            self.face_mesh_connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
        else:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    "MediaPipe Face Landmarker model not found. "
                    f"Download it to: {MODEL_PATH}"
                )

            base_options = mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=max_num_faces,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_face_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            self.frame_timestamp_ms = 0

    def read(self) -> Optional[FaceResult]:
        """Read a webcam frame and return the first face's landmarks if present."""
        success, frame = self.cap.read()
        if not success:
            return None

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width = frame.shape[:2]

        landmarks = None
        if self.use_solutions_api:
            results = self.face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.face_landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
            self.frame_timestamp_ms += 33
            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

        return FaceResult(
            frame=frame,
            rgb_frame=rgb_frame,
            landmarks=landmarks,
            image_width=image_width,
            image_height=image_height,
        )

    def draw_landmarks(self, frame: np.ndarray, landmarks: Optional[list]) -> np.ndarray:
        """Overlay face landmarks on the frame for visual feedback."""
        if landmarks is None:
            return frame

        if self.use_solutions_api:
            self.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=self._wrap_landmarks(landmarks),
                connections=self.face_mesh_connections,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.drawing_styles.get_default_face_mesh_tesselation_style(),
            )
            return frame

        self._draw_connections(frame, landmarks, self.FACE_OVAL, (0, 220, 255))
        self._draw_connections(frame, landmarks, self.LEFT_EYE, (0, 255, 0))
        self._draw_connections(frame, landmarks, self.RIGHT_EYE, (0, 255, 0))
        self._draw_connections(frame, landmarks, self.MOUTH, (255, 120, 0))

        for point in landmarks:
            x = int(point.x * frame.shape[1])
            y = int(point.y * frame.shape[0])
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)
        return frame

    def _draw_connections(
        self,
        frame: np.ndarray,
        landmarks: list,
        indices: list[int],
        color: tuple[int, int, int],
    ) -> None:
        for start, end in zip(indices, indices[1:]):
            start_point = landmarks[start]
            end_point = landmarks[end]
            start_xy = (int(start_point.x * frame.shape[1]), int(start_point.y * frame.shape[0]))
            end_xy = (int(end_point.x * frame.shape[1]), int(end_point.y * frame.shape[0]))
            cv2.line(frame, start_xy, end_xy, color, 1)

    @staticmethod
    def _wrap_landmarks(landmarks: list):
        """Build a landmark container for the legacy drawing utilities."""
        from mediapipe.framework.formats import landmark_pb2

        landmark_list = landmark_pb2.NormalizedLandmarkList()
        landmark_list.landmark.extend(landmarks)
        return landmark_list

    def release(self) -> None:
        """Release OpenCV and MediaPipe resources."""
        self.cap.release()
        if self.use_solutions_api:
            self.face_mesh.close()
        else:
            self.face_landmarker.close()
