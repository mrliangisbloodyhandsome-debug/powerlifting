from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .mediapipe_pose import POSE_CONNECTIONS, POSE_LANDMARK_NAMES


def save_visualization(
    image_bgr: np.ndarray,
    ground_mask: Optional[np.ndarray],
    landmarks_ground: Dict[str, Dict[str, Any]],
    output_path: str | Path,
) -> np.ndarray:
    visualization = draw_visualization(image_bgr, ground_mask, landmarks_ground)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), visualization)
    return visualization


def draw_visualization(
    image_bgr: np.ndarray,
    ground_mask: Optional[np.ndarray],
    landmarks_ground: Dict[str, Dict[str, Any]],
) -> np.ndarray:
    canvas = image_bgr.copy()

    if ground_mask is not None:
        mask = ground_mask.astype(bool)
        if mask.shape != canvas.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (canvas.shape[1], canvas.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        overlay = canvas.copy()
        overlay[mask] = (0, 180, 80)
        canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)

    z_values = [
        item["ground"]["Zg"]
        for item in landmarks_ground.values()
        if item.get("valid") and item.get("ground") is not None
    ]
    z_min = min(z_values) if z_values else 0.0
    z_max = max(z_values) if z_values else 1.0
    if abs(z_max - z_min) < 1e-8:
        z_max = z_min + 1.0

    points_by_index = {
        int(item["index"]): item
        for item in landmarks_ground.values()
        if "index" in item and np.isfinite(item.get("x_px", np.nan))
    }

    for start, end in POSE_CONNECTIONS:
        a = points_by_index.get(start)
        b = points_by_index.get(end)
        if a is None or b is None:
            continue
        if not a.get("valid") or not b.get("valid"):
            continue
        p1 = (int(round(a["x_px"])), int(round(a["y_px"])))
        p2 = (int(round(b["x_px"])), int(round(b["y_px"])))
        cv2.line(canvas, p1, p2, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    for item in landmarks_ground.values():
        x = int(round(item.get("x_px", -1)))
        y = int(round(item.get("y_px", -1)))
        if x < 0 or y < 0 or x >= canvas.shape[1] or y >= canvas.shape[0]:
            continue
        if item.get("valid") and item.get("ground") is not None:
            color = _height_color(item["ground"]["Zg"], z_min, z_max)
            cv2.circle(canvas, (x, y), 5, color, -1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(canvas, (x, y), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    _draw_height_legend(canvas, z_min, z_max)
    return canvas


def save_depth_png(depth: np.ndarray, output_path: str | Path) -> None:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth)
    if not valid.any():
        normalized = np.zeros(depth.shape, dtype=np.uint8)
    else:
        lo, hi = np.nanpercentile(depth[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        normalized = np.clip((depth - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_MAGMA)
    cv2.imwrite(str(output_path), colored)


def save_normal_png(normal: np.ndarray, output_path: str | Path) -> None:
    normal = np.asarray(normal, dtype=np.float32)
    image = ((normal + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _height_color(z_value: float, z_min: float, z_max: float) -> tuple[int, int, int]:
    t = float(np.clip((z_value - z_min) / (z_max - z_min), 0.0, 1.0))
    color = cv2.applyColorMap(np.array([[int(t * 255)]], dtype=np.uint8), cv2.COLORMAP_TURBO)
    return tuple(int(v) for v in color[0, 0])


def _draw_height_legend(canvas: np.ndarray, z_min: float, z_max: float) -> None:
    height = 120
    width = 14
    x0 = canvas.shape[1] - 30
    y0 = 20
    gradient = np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1)
    colorbar = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
    canvas[y0 : y0 + height, x0 : x0 + width] = colorbar
    cv2.putText(
        canvas,
        "Zg",
        (x0 - 4, y0 + height + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{z_max:.2f}",
        (x0 - 58, y0 + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{z_min:.2f}",
        (x0 - 58, y0 + height),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
