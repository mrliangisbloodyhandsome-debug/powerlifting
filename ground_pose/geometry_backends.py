from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import cv2
import numpy as np


@dataclass
class GeometryPrediction:
    """Dense scene geometry aligned to an image or frame."""

    points: np.ndarray
    depth: Optional[np.ndarray] = None
    normal: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    intrinsics: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float32)
        if self.points.ndim != 3 or self.points.shape[2] != 3:
            raise ValueError("points must have shape H x W x 3")
        if self.depth is not None:
            self.depth = np.asarray(self.depth, dtype=np.float32)
        if self.normal is not None:
            self.normal = np.asarray(self.normal, dtype=np.float32)
        if self.mask is not None:
            self.mask = np.asarray(self.mask).astype(bool)
        if self.intrinsics is not None:
            self.intrinsics = np.asarray(self.intrinsics, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        return self.points.shape[:2]

    def valid_mask(self) -> np.ndarray:
        finite_points = np.isfinite(self.points).all(axis=2)
        mask = finite_points
        if self.mask is not None:
            mask = mask & self.mask
        if self.depth is not None:
            mask = mask & np.isfinite(self.depth) & (self.depth > 0)
        return mask


class MoGeBackend:
    """MoGe-2 backend using the official `moge.model.v2.MoGeModel` API.

    The official MoGe README exposes:
    `MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").infer(image)`.
    This class only calls that real API. If the package or weights are missing it
    raises NotImplementedError with installation guidance instead of fabricating
    geometry.
    """

    def __init__(
        self,
        pretrained: str = "Ruicheng/moge-2-vitl-normal",
        device: Optional[str] = None,
    ) -> None:
        self.pretrained = pretrained
        self.device = device
        self._model = None
        self._torch = None

    def predict(self, image_bgr: np.ndarray) -> GeometryPrediction:
        self._ensure_model()
        torch = self._torch
        assert torch is not None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = (
            torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .to(self.device)
        )

        with torch.no_grad():
            output = self._model.infer(image_tensor)

        points = _tensor_to_numpy(_first_present(output, ["points", "point_map"]))
        if points is None:
            raise RuntimeError("MoGe output did not include `points` or `point_map`.")

        depth = _tensor_to_numpy(_first_present(output, ["depth"]))
        normal = _tensor_to_numpy(_first_present(output, ["normal", "normals"]))
        mask = _tensor_to_numpy(_first_present(output, ["mask", "valid_mask"]))
        intrinsics = _tensor_to_numpy(_first_present(output, ["intrinsics", "K"]))

        return GeometryPrediction(
            points=points,
            depth=depth,
            normal=normal,
            mask=mask.astype(bool) if mask is not None else None,
            intrinsics=intrinsics,
            metadata={
                "backend": "moge",
                "model": self.pretrained,
                "notes": [
                    "MoGe/Metric3D scene points are sampled at MediaPipe 2D landmark pixels.",
                    "MediaPipe pose_world_landmarks are intentionally not mixed with scene geometry.",
                ],
            },
        )

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from moge.model.v2 import MoGeModel
        except Exception as exc:  # pragma: no cover - depends on optional deps.
            raise NotImplementedError(
                "MoGe-2 backend requires PyTorch and the official MoGe package. "
                "Install in this project with: "
                ".\\.venv\\Scripts\\python.exe -m pip install torch torchvision "
                "git+https://github.com/microsoft/MoGe.git"
            ) from exc

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            model = MoGeModel.from_pretrained(self.pretrained)
        except Exception as exc:  # pragma: no cover - depends on network/weights.
            raise NotImplementedError(
                f"Could not load MoGe-2 weights `{self.pretrained}`. "
                "Check network access and the Hugging Face model name."
            ) from exc

        self._torch = torch
        self._model = model.to(self.device).eval()


class Metric3DBackend:
    """Metric3D v2 fallback backend using the official torch.hub entry point."""

    def __init__(
        self,
        model_name: str = "metric3d_vit_small",
        device: Optional[str] = None,
        focal_length_px: Optional[float] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.focal_length_px = focal_length_px
        self._model = None
        self._torch = None

    def predict(self, image_bgr: np.ndarray) -> GeometryPrediction:
        self._ensure_model()
        torch = self._torch
        assert torch is not None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_height, original_width = image_bgr.shape[:2]
        intrinsics = make_approximate_intrinsics(
            width=original_width,
            height=original_height,
            focal_length_px=self.focal_length_px,
        )
        tensor, pad_info, scaled_intrinsics = _prepare_metric3d_input(
            image_rgb=image_rgb,
            intrinsics=intrinsics,
            torch=torch,
            model_name=self.model_name,
        )
        tensor = tensor.to(self.device)

        with torch.no_grad():
            try:
                pred_depth, confidence, output_dict = self._model.inference(
                    {"input": tensor[None]}
                )
            except Exception as exc:  # pragma: no cover - optional model path.
                raise RuntimeError(
                    "Metric3D inference failed. The official model API may have "
                    "changed; inspect the Metric3D README for the current call."
                ) from exc

        depth_tensor = _unpad_depth_tensor(pred_depth, pad_info)
        depth_tensor = torch.nn.functional.interpolate(
            depth_tensor[None, None],
            (original_height, original_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        canonical_to_real_scale = float(scaled_intrinsics[0, 0] / 1000.0)
        depth_tensor = torch.clamp(depth_tensor * canonical_to_real_scale, 0, 300)
        depth = _tensor_to_numpy(depth_tensor).astype(np.float32)

        normal = None
        if isinstance(output_dict, dict):
            normal_tensor = output_dict.get("prediction_normal")
            if normal_tensor is not None:
                normal_tensor = normal_tensor[:, :3]
                normal_tensor = _unpad_normal_tensor(normal_tensor, pad_info)
                normal_tensor = torch.nn.functional.interpolate(
                    normal_tensor,
                    (original_height, original_width),
                    mode="bilinear",
                    align_corners=False,
                )
                normal = _tensor_to_numpy(normal_tensor.squeeze(0))
                normal = np.moveaxis(normal, 0, 2).astype(np.float32)

        points = backproject_depth(depth, intrinsics)

        return GeometryPrediction(
            points=points,
            depth=depth,
            normal=normal,
            mask=np.isfinite(depth) & (depth > 0),
            intrinsics=intrinsics,
            metadata={
                "backend": "metric3d",
                "model": self.model_name,
                "intrinsics_source": "approximate_pinhole_from_image_size",
                "canonical_to_real_scale": canonical_to_real_scale,
                "intrinsics_warning": (
                    "No real camera intrinsics were provided. Depth was "
                    "backprojected with an approximate pinhole camera, so the "
                    "3D coordinates are suitable for relative geometry only."
                ),
                "notes": [
                    "MediaPipe pose_world_landmarks are intentionally not mixed with Metric3D points.",
                ],
            },
        )

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional deps.
            raise NotImplementedError(
                "Metric3D v2 backend requires PyTorch. Install with: "
                ".\\.venv\\Scripts\\python.exe -m pip install torch torchvision"
            ) from exc

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            model = torch.hub.load(
                "yvanyin/metric3d",
                self.model_name,
                pretrain=True,
                trust_repo=True,
            )
        except Exception as exc:  # pragma: no cover - network/model dependent.
            raise NotImplementedError(
                "Could not load Metric3D v2 from torch.hub. Install git, ensure "
                "network access, and see https://github.com/YvanYin/Metric3D."
            ) from exc

        self._torch = torch
        self._model = model.to(self.device).eval()
        if self.device == "cpu":
            self._patch_metric3d_cpu_decoder()

    def _patch_metric3d_cpu_decoder(self) -> None:
        """Patch a Metric3D v2 helper that hard-codes CUDA in torch.hub code."""

        import math
        import types

        torch = self._torch
        decoder = getattr(getattr(self._model, "depth_model", None), "decoder", None)
        if decoder is None or not hasattr(decoder, "get_bins"):
            return

        def get_bins_on_model_device(decoder_self: Any, bins_num: int) -> Any:
            first_param = next(decoder_self.parameters(), None)
            device = first_param.device if first_param is not None else torch.device("cpu")
            depth_bins_vec = torch.linspace(
                math.log(decoder_self.min_val),
                math.log(decoder_self.max_val),
                bins_num,
                device=device,
            )
            return torch.exp(depth_bins_vec)

        decoder.get_bins = types.MethodType(get_bins_on_model_device, decoder)


def make_approximate_intrinsics(
    width: int,
    height: int,
    focal_length_px: Optional[float] = None,
) -> np.ndarray:
    """Build a rough pinhole camera matrix from image dimensions."""

    if focal_length_px is None:
        focal_length_px = float(max(width, height))
    return np.array(
        [
            [focal_length_px, 0.0, (width - 1) / 2.0],
            [0.0, focal_length_px, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def backproject_depth(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Backproject H x W depth into a dense H x W x 3 point map."""

    depth = np.asarray(depth, dtype=np.float32)
    height, width = depth.shape
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    z = depth
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    points = np.stack([x, y, z], axis=2).astype(np.float32)
    points[~np.isfinite(points).all(axis=2)] = np.nan
    return points


def _metric3d_input_size(model_name: str) -> tuple[int, int]:
    if "convnext" in model_name:
        return (544, 1216)
    return (616, 1064)


def _prepare_metric3d_input(
    image_rgb: np.ndarray,
    intrinsics: np.ndarray,
    torch: Any,
    model_name: str,
) -> tuple[Any, list[int], np.ndarray]:
    input_height, input_width = _metric3d_input_size(model_name)
    height, width = image_rgb.shape[:2]
    scale = min(input_height / height, input_width / width)
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    resized = cv2.resize(
        image_rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    scaled_intrinsics = intrinsics.copy()
    scaled_intrinsics[0, :] *= scale
    scaled_intrinsics[1, :] *= scale

    padding_value = [123.675, 116.28, 103.53]
    pad_h = input_height - resized_height
    pad_w = input_width - resized_width
    pad_h_top = pad_h // 2
    pad_h_bottom = pad_h - pad_h_top
    pad_w_left = pad_w // 2
    pad_w_right = pad_w - pad_w_left
    padded = cv2.copyMakeBorder(
        resized,
        pad_h_top,
        pad_h_bottom,
        pad_w_left,
        pad_w_right,
        cv2.BORDER_CONSTANT,
        value=padding_value,
    )

    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)[:, None, None]
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)[:, None, None]
    tensor_np = padded.transpose(2, 0, 1).astype(np.float32)
    tensor_np = (tensor_np - mean) / std
    tensor = torch.from_numpy(tensor_np).float()
    pad_info = [pad_h_top, pad_h_bottom, pad_w_left, pad_w_right]
    return tensor, pad_info, scaled_intrinsics


def _unpad_depth_tensor(depth: Any, pad_info: list[int]) -> Any:
    depth = depth.squeeze()
    top, bottom, left, right = pad_info
    h_end = depth.shape[-2] - bottom if bottom else depth.shape[-2]
    w_end = depth.shape[-1] - right if right else depth.shape[-1]
    return depth[top:h_end, left:w_end]


def _unpad_normal_tensor(normal: Any, pad_info: list[int]) -> Any:
    top, bottom, left, right = pad_info
    h_end = normal.shape[-2] - bottom if bottom else normal.shape[-2]
    w_end = normal.shape[-1] - right if right else normal.shape[-1]
    return normal[:, :, top:h_end, left:w_end]


def _tensor_to_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _first_present(mapping: Any, keys: list[str]) -> Any:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None
