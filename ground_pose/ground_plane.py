from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class PlaneFitResult:
    plane_model: np.ndarray
    unit_normal: np.ndarray
    inlier_indices: np.ndarray
    inlier_ratio: float
    candidate_count: int
    ransac_distance_threshold: float
    ransac_num_iterations: int
    backend: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def inlier_count(self) -> int:
        return int(len(self.inlier_indices))

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "plane_model": self.plane_model.astype(float).tolist(),
            "unit_normal": self.unit_normal.astype(float).tolist(),
            "ground_inlier_ratio": float(self.inlier_ratio),
            "number_of_ground_candidate_points": int(self.candidate_count),
            "number_of_ransac_inliers": self.inlier_count,
            "ransac_parameters": {
                "distance_threshold": float(self.ransac_distance_threshold),
                "num_iterations": int(self.ransac_num_iterations),
            },
            "fit_backend": self.backend,
            "metadata": self.metadata,
        }


def fit_plane_open3d(
    points: np.ndarray,
    distance_threshold: float = 0.03,
    num_iterations: int = 1000,
    ransac_n: int = 3,
    min_points: int = 50,
) -> PlaneFitResult:
    """Fit a plane with Open3D if available, otherwise with NumPy RANSAC."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]
    candidate_count = int(points.shape[0])

    if candidate_count < min_points:
        raise RuntimeError(
            f"Ground plane fitting needs at least {min_points} valid points; "
            f"got {candidate_count}."
        )

    try:
        import open3d as o3d

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        plane_model, inliers = cloud.segment_plane(
            distance_threshold=float(distance_threshold),
            ransac_n=int(ransac_n),
            num_iterations=int(num_iterations),
        )
        plane_model = np.asarray(plane_model, dtype=np.float64)
        inlier_indices = np.asarray(inliers, dtype=np.int64)
        backend = "open3d"
    except Exception as exc:
        plane_model, inlier_indices = _fit_plane_ransac_numpy(
            points=points,
            distance_threshold=distance_threshold,
            num_iterations=num_iterations,
            ransac_n=ransac_n,
        )
        backend = "numpy_ransac_fallback"
        fallback_exc = f"{type(exc).__name__}: {exc}"

    plane_model = normalize_plane(plane_model)
    normal = plane_model[:3].copy()
    inlier_ratio = float(len(inlier_indices) / candidate_count) if candidate_count else 0.0
    metadata: Dict[str, Any] = {}
    if backend == "numpy_ransac_fallback":
        metadata["open3d_fallback_reason"] = fallback_exc

    return PlaneFitResult(
        plane_model=plane_model,
        unit_normal=normal,
        inlier_indices=inlier_indices,
        inlier_ratio=inlier_ratio,
        candidate_count=candidate_count,
        ransac_distance_threshold=float(distance_threshold),
        ransac_num_iterations=int(num_iterations),
        backend=backend,
        metadata=metadata,
    )


def orient_plane_with_body(
    plane_model: np.ndarray,
    unit_normal: np.ndarray,
    mid_hip: Optional[np.ndarray],
    mid_shoulder: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Flip plane normal so torso direction points toward positive ground Z."""

    plane_model = normalize_plane(np.asarray(plane_model, dtype=np.float64))
    normal = normalize_vector(unit_normal)
    metadata = {
        "oriented_with_body": False,
        "flipped": False,
        "reason": None,
    }

    if mid_hip is None or mid_shoulder is None:
        metadata["reason"] = "missing_mid_hip_or_mid_shoulder"
        return plane_model, normal, metadata

    mid_hip = np.asarray(mid_hip, dtype=np.float64)
    mid_shoulder = np.asarray(mid_shoulder, dtype=np.float64)
    if not np.isfinite(mid_hip).all() or not np.isfinite(mid_shoulder).all():
        metadata["reason"] = "non_finite_body_points"
        return plane_model, normal, metadata

    torso_vector = mid_shoulder - mid_hip
    if np.linalg.norm(torso_vector) < 1e-8:
        metadata["reason"] = "zero_torso_vector"
        return plane_model, normal, metadata

    dot_value = float(np.dot(normal, torso_vector))
    metadata["oriented_with_body"] = True
    metadata["normal_dot_torso"] = dot_value

    if dot_value < 0.0:
        normal = -normal
        plane_model = -plane_model
        metadata["flipped"] = True
    return plane_model, normal, metadata


def build_ground_coordinate_frame(
    plane_model: np.ndarray,
    inlier_points: np.ndarray,
    landmark_scene_points: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    plane_model = normalize_plane(np.asarray(plane_model, dtype=np.float64))
    inlier_points = np.asarray(inlier_points, dtype=np.float64).reshape(-1, 3)
    inlier_points = inlier_points[np.isfinite(inlier_points).all(axis=1)]
    if len(inlier_points) == 0:
        raise RuntimeError("Cannot build ground frame without inlier ground points.")

    p0 = inlier_points.mean(axis=0)
    ez = normalize_vector(plane_model[:3])

    ex = None
    source = "camera_x_axis"
    if landmark_scene_points:
        left_hip = landmark_scene_points.get("left_hip")
        right_hip = landmark_scene_points.get("right_hip")
        if is_valid_point(left_hip) and is_valid_point(right_hip):
            hip_right_left = np.asarray(right_hip) - np.asarray(left_hip)
            projected = project_to_plane(hip_right_left, ez)
            if np.linalg.norm(projected) > 1e-8:
                ex = normalize_vector(projected)
                source = "right_hip_minus_left_hip_projected_to_ground"

    if ex is None:
        camera_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        projected = project_to_plane(camera_x, ez)
        if np.linalg.norm(projected) < 1e-8:
            projected = project_to_plane(np.array([0.0, 1.0, 0.0]), ez)
            source = "camera_y_axis_projected_to_ground"
        ex = normalize_vector(projected)

    ey = normalize_vector(np.cross(ez, ex))

    return {
        "P0": p0,
        "ex": ex,
        "ey": ey,
        "ez": ez,
        "x_axis_source": source,
    }


def transform_landmarks_to_ground_frame(
    landmarks_2d: Dict[str, Dict[str, Any]],
    point_map: np.ndarray,
    valid_mask: Optional[np.ndarray],
    frame: Dict[str, Any],
    image_shape: tuple[int, int],
) -> Dict[str, Dict[str, Any]]:
    """Sample scene points at MediaPipe 2D pixels and transform them to ground axes."""

    results: Dict[str, Dict[str, Any]] = {}
    p0 = np.asarray(frame["P0"], dtype=np.float64)
    ex = np.asarray(frame["ex"], dtype=np.float64)
    ey = np.asarray(frame["ey"], dtype=np.float64)
    ez = np.asarray(frame["ez"], dtype=np.float64)

    for name, landmark in landmarks_2d.items():
        sampled = sample_point_for_landmark(landmark, point_map, valid_mask, image_shape)
        row = {
            "index": int(landmark.get("index", -1)),
            "x_px": float(landmark.get("x_px", np.nan)),
            "y_px": float(landmark.get("y_px", np.nan)),
            "visibility": float(landmark.get("visibility", 0.0)),
            "presence": float(landmark.get("presence", 0.0)),
            "valid": sampled["valid"],
            "invalid_reason": sampled["invalid_reason"],
            "scene_point": None,
            "ground": None,
        }
        if sampled["valid"]:
            point = sampled["point"]
            pg = point - p0
            row["scene_point"] = point.astype(float).tolist()
            row["point_map_pixel"] = [int(sampled["u_pm"]), int(sampled["v_pm"])]
            row["ground"] = {
                "Xg": float(np.dot(pg, ex)),
                "Yg": float(np.dot(pg, ey)),
                "Zg": float(np.dot(pg, ez)),
            }
        results[name] = row
    return results


def sample_landmark_scene_points(
    landmarks_2d: Dict[str, Dict[str, Any]],
    point_map: np.ndarray,
    valid_mask: Optional[np.ndarray],
    image_shape: tuple[int, int],
) -> Dict[str, np.ndarray]:
    sampled: Dict[str, np.ndarray] = {}
    for name, landmark in landmarks_2d.items():
        result = sample_point_for_landmark(landmark, point_map, valid_mask, image_shape)
        if result["valid"]:
            sampled[name] = result["point"]
    return sampled


def midpoint(
    points: Dict[str, np.ndarray],
    left_name: str,
    right_name: str,
) -> Optional[np.ndarray]:
    left = points.get(left_name)
    right = points.get(right_name)
    if not is_valid_point(left) or not is_valid_point(right):
        return None
    return (np.asarray(left) + np.asarray(right)) / 2.0


def sample_point_for_landmark(
    landmark: Dict[str, Any],
    point_map: np.ndarray,
    valid_mask: Optional[np.ndarray],
    image_shape: tuple[int, int],
) -> Dict[str, Any]:
    image_h, image_w = image_shape
    map_h, map_w = point_map.shape[:2]
    x_px = float(landmark.get("x_px", np.nan))
    y_px = float(landmark.get("y_px", np.nan))
    if not np.isfinite(x_px) or not np.isfinite(y_px):
        return _invalid("non_finite_2d_landmark")
    if x_px < 0 or y_px < 0 or x_px >= image_w or y_px >= image_h:
        return _invalid("2d_landmark_outside_image")

    u_pm = int(np.clip(round(x_px * map_w / image_w), 0, map_w - 1))
    v_pm = int(np.clip(round(y_px * map_h / image_h), 0, map_h - 1))
    if valid_mask is not None and not bool(valid_mask[v_pm, u_pm]):
        return _invalid("point_outside_valid_geometry_mask", u_pm=u_pm, v_pm=v_pm)

    point = np.asarray(point_map[v_pm, u_pm], dtype=np.float64)
    if not np.isfinite(point).all():
        return _invalid("scene_point_nan_or_inf", u_pm=u_pm, v_pm=v_pm)

    return {
        "valid": True,
        "invalid_reason": None,
        "point": point,
        "u_pm": u_pm,
        "v_pm": v_pm,
    }


def normalize_plane(plane_model: np.ndarray) -> np.ndarray:
    plane_model = np.asarray(plane_model, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(plane_model[:3])
    if norm < 1e-12:
        raise ValueError("Plane normal has zero length.")
    return plane_model / norm


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("Cannot normalize zero-length vector.")
    return vector / norm


def project_to_plane(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    normal = normalize_vector(normal)
    return vector - np.dot(vector, normal) * normal


def is_valid_point(point: Optional[np.ndarray]) -> bool:
    if point is None:
        return False
    point = np.asarray(point)
    return point.shape == (3,) and np.isfinite(point).all()


def _invalid(reason: str, **kwargs: Any) -> Dict[str, Any]:
    return {
        "valid": False,
        "invalid_reason": reason,
        "point": None,
        **kwargs,
    }


def _fit_plane_ransac_numpy(
    points: np.ndarray,
    distance_threshold: float,
    num_iterations: int,
    ransac_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    if ransac_n != 3:
        raise ValueError("NumPy fallback currently supports ransac_n=3 only.")

    rng = np.random.default_rng(0)
    best_inliers: np.ndarray = np.array([], dtype=np.int64)
    best_plane: Optional[np.ndarray] = None
    count = points.shape[0]

    for _ in range(int(num_iterations)):
        sample_idx = rng.choice(count, size=3, replace=False)
        p1, p2, p3 = points[sample_idx]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm
        d = -float(np.dot(normal, p1))
        distances = np.abs(points @ normal + d)
        inliers = np.flatnonzero(distances <= distance_threshold)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = np.array([normal[0], normal[1], normal[2], d])

    if best_plane is None or len(best_inliers) == 0:
        raise RuntimeError("RANSAC failed to find a valid ground plane.")

    refined_plane = _least_squares_plane(points[best_inliers])
    distances = np.abs(points @ refined_plane[:3] + refined_plane[3])
    refined_inliers = np.flatnonzero(distances <= distance_threshold)
    return refined_plane, refined_inliers


def _least_squares_plane(points: np.ndarray) -> np.ndarray:
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    d = -float(np.dot(normal, centroid))
    return np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)
