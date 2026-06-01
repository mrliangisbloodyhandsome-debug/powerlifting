from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


@dataclass
class SegmentationResult:
    mask: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mask = np.asarray(self.mask).astype(bool)
        if self.mask.ndim != 2:
            raise ValueError("Segmentation mask must be H x W")


class GroundedSAMBackend:
    """Grounded SAM adapter placeholder with an explicit dependency error.

    Grounded SAM and Grounded SAM 2 are distributed as external research repos
    with model-specific setup. This backend intentionally refuses to fake a mask
    when those adapters are not installed.
    """

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt

    def predict(self, image_bgr: np.ndarray) -> SegmentationResult:
        raise NotImplementedError(
            "Grounded SAM backend is not installed in this project. Install and "
            "wire an official Grounded-Segment-Anything or Grounded-SAM-2 adapter, "
            "or rerun with --seg_backend manual_mask --manual_mask_path PATH, "
            "or --seg_backend lower_region. The CLI default will fall back to "
            "lower_region and mark that output as rough."
        )


class ManualMaskBackend:
    def __init__(self, mask_path: str | Path) -> None:
        self.mask_path = Path(mask_path)

    def predict(self, image_bgr: np.ndarray) -> SegmentationResult:
        if not self.mask_path.exists():
            raise FileNotFoundError(f"Ground mask not found: {self.mask_path}")
        mask = cv2.imread(str(self.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read ground mask: {self.mask_path}")
        height, width = image_bgr.shape[:2]
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return SegmentationResult(
            mask=mask > 0,
            metadata={
                "backend": "manual_mask",
                "mask_path": str(self.mask_path),
            },
        )


class LowerRegionBackend:
    def __init__(self, lower_region_ratio: float = 0.40) -> None:
        if not 0.0 < lower_region_ratio <= 1.0:
            raise ValueError("lower_region_ratio must be in (0, 1]")
        self.lower_region_ratio = float(lower_region_ratio)

    def predict(self, image_bgr: np.ndarray) -> SegmentationResult:
        height, width = image_bgr.shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        start_y = int(round(height * (1.0 - self.lower_region_ratio)))
        mask[start_y:, :] = True
        return SegmentationResult(
            mask=mask,
            metadata={
                "backend": "lower_region",
                "lower_region_ratio": self.lower_region_ratio,
                "warning": (
                    "This is a rough fallback that assumes the lower image "
                    "region contains ground. Use Grounded SAM or a manual mask "
                    "for real analysis."
                ),
            },
        )


def resize_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    if mask.shape == (height, width):
        return mask.astype(bool)
    resized = cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def remove_person_from_ground_mask(
    ground_mask: np.ndarray,
    person_mask: Optional[np.ndarray] = None,
    pose_landmarks: Optional[Dict[str, Any]] = None,
    image_shape: Optional[tuple[int, int]] = None,
    dilation_px: int = 15,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Remove MediaPipe person mask or pose bbox from candidate ground pixels."""

    refined = ground_mask.astype(bool).copy()
    metadata: Dict[str, Any] = {"person_exclusion": "none"}

    if person_mask is not None:
        person = resize_mask(person_mask, refined.shape)
        kernel_size = max(1, int(dilation_px))
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        person = cv2.dilate(person.astype(np.uint8), kernel, iterations=1).astype(bool)
        refined &= ~person
        metadata["person_exclusion"] = "mediapipe_segmentation_mask"
        metadata["excluded_pixels"] = int(person.sum())
        return refined, metadata

    if pose_landmarks and image_shape is not None:
        bbox = landmark_bbox(pose_landmarks, image_shape)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            scale_y = refined.shape[0] / image_shape[0]
            scale_x = refined.shape[1] / image_shape[1]
            x1 = int(np.clip(round(x1 * scale_x), 0, refined.shape[1] - 1))
            x2 = int(np.clip(round(x2 * scale_x), 0, refined.shape[1] - 1))
            y1 = int(np.clip(round(y1 * scale_y), 0, refined.shape[0] - 1))
            y2 = int(np.clip(round(y2 * scale_y), 0, refined.shape[0] - 1))
            refined[y1 : y2 + 1, x1 : x2 + 1] = False
            metadata["person_exclusion"] = "mediapipe_landmark_bbox"
            metadata["bbox_px"] = [x1, y1, x2, y2]

    return refined, metadata


def landmark_bbox(
    landmarks: Dict[str, Any],
    image_shape: tuple[int, int],
    visibility_min: float = 0.2,
    expansion_ratio: float = 0.08,
) -> Optional[list[int]]:
    points = []
    for item in landmarks.values():
        if float(item.get("visibility", 1.0)) < visibility_min:
            continue
        x = float(item.get("x_px", np.nan))
        y = float(item.get("y_px", np.nan))
        if np.isfinite(x) and np.isfinite(y):
            points.append((x, y))
    if not points:
        return None

    height, width = image_shape
    xs = np.array([point[0] for point in points])
    ys = np.array([point[1] for point in points])
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    pad_x = (x2 - x1) * expansion_ratio + 10.0
    pad_y = (y2 - y1) * expansion_ratio + 10.0

    return [
        int(np.clip(x1 - pad_x, 0, width - 1)),
        int(np.clip(y1 - pad_y, 0, height - 1)),
        int(np.clip(x2 + pad_x, 0, width - 1)),
        int(np.clip(y2 + pad_y, 0, height - 1)),
    ]
