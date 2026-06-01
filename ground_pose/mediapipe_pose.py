from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "src" / "models" / "pose_landmarker_heavy.task"

POSE_LANDMARK_NAMES = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32),
]


@dataclass
class PoseDetection:
    landmarks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    person_mask: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MediaPipePoseEstimator:
    """MediaPipe Pose Landmarker wrapper that only outputs 2D landmark indices."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        running_mode: str = "image",
        num_poses: int = 1,
        output_segmentation_masks: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.running_mode = running_mode
        self.num_poses = int(num_poses)
        self.output_segmentation_masks = output_segmentation_masks
        self._landmarker = None

    def __enter__(self) -> "MediaPipePoseEstimator":
        self._ensure_landmarker()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def detect_image(self, image_bgr: np.ndarray) -> PoseDetection:
        if self.running_mode != "image":
            raise RuntimeError("MediaPipePoseEstimator was not created in image mode.")
        self._ensure_landmarker()
        mp_image = self._to_mp_image(image_bgr)
        result = self._landmarker.detect(mp_image)
        return self._extract(result, image_bgr.shape[:2])

    def detect_video_frame(
        self,
        image_bgr: np.ndarray,
        timestamp_ms: int,
    ) -> PoseDetection:
        if self.running_mode != "video":
            raise RuntimeError("MediaPipePoseEstimator was not created in video mode.")
        self._ensure_landmarker()
        mp_image = self._to_mp_image(image_bgr)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        return self._extract(result, image_bgr.shape[:2])

    def _ensure_landmarker(self) -> None:
        if self._landmarker is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"MediaPipe model not found: {self.model_path}")

        running_mode = (
            vision.RunningMode.VIDEO
            if self.running_mode == "video"
            else vision.RunningMode.IMAGE
        )
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=running_mode,
            num_poses=self.num_poses,
            output_segmentation_masks=self.output_segmentation_masks,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    @staticmethod
    def _to_mp_image(image_bgr: np.ndarray) -> mp.Image:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    @staticmethod
    def _extract(result: Any, image_shape: tuple[int, int]) -> PoseDetection:
        height, width = image_shape
        metadata = {
            "landmark_source": "mediapipe_pose_landmarker",
            "warning": (
                "Only MediaPipe 2D image landmark pixels are used as indices "
                "into the MoGe/Metric3D point map. pose_world_landmarks are not "
                "used as output 3D coordinates."
            ),
        }

        if not result.pose_landmarks:
            return PoseDetection(landmarks={}, metadata={**metadata, "poses": 0})

        pose_landmarks = result.pose_landmarks[0]
        landmarks: Dict[str, Dict[str, Any]] = {}
        for index, landmark in enumerate(pose_landmarks):
            name = POSE_LANDMARK_NAMES.get(index, f"landmark_{index}")
            landmarks[name] = {
                "index": index,
                "x_norm": float(landmark.x),
                "y_norm": float(landmark.y),
                "z_norm": float(landmark.z),
                "x_px": float(landmark.x * width),
                "y_px": float(landmark.y * height),
                "visibility": float(getattr(landmark, "visibility", 0.0)),
                "presence": float(getattr(landmark, "presence", 0.0)),
            }

        person_mask = None
        if getattr(result, "segmentation_masks", None):
            masks = []
            for mask in result.segmentation_masks:
                mask_np = mask.numpy_view()
                masks.append(mask_np > 0.5)
            if masks:
                person_mask = np.any(np.stack(masks, axis=0), axis=0)

        return PoseDetection(
            landmarks=landmarks,
            person_mask=person_mask,
            metadata={**metadata, "poses": len(result.pose_landmarks)},
        )
