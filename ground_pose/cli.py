from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .geometry_backends import GeometryPrediction, Metric3DBackend, MoGeBackend
from .ground_plane import (
    build_ground_coordinate_frame,
    fit_plane_open3d,
    midpoint,
    orient_plane_with_body,
    sample_landmark_scene_points,
    transform_landmarks_to_ground_frame,
)
from .mediapipe_pose import DEFAULT_MODEL_PATH, MediaPipePoseEstimator, PoseDetection
from .segmentation_backends import (
    GroundedSAMBackend,
    LowerRegionBackend,
    ManualMaskBackend,
    remove_person_from_ground_mask,
    resize_mask,
)
from .visualize import save_depth_png, save_normal_png, save_visualization


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    input_path = Path(args.path or args.image_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.suffix.lower() in IMAGE_EXTENSIONS:
        run_image(input_path, output_dir, args)
    else:
        run_video(input_path, output_dir, args)


def run_image(input_path: Path, output_dir: Path, args: argparse.Namespace) -> None:
    image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {input_path}")

    with MediaPipePoseEstimator(
        model_path=args.mediapipe_model,
        running_mode="image",
        output_segmentation_masks=True,
    ) as pose_estimator:
        pose = pose_estimator.detect_image(image_bgr)

    analysis = analyze_frame(
        image_bgr=image_bgr,
        pose=pose,
        args=args,
        output_dir=output_dir,
        frame_index=None,
    )
    save_json(output_dir / "result_landmarks_ground.json", analysis["landmarks_json"])
    save_json(output_dir / "ground_plane.json", analysis["ground_plane_json"])
    save_optional_geometry_outputs(output_dir, analysis, args)
    save_visualization(
        image_bgr=image_bgr,
        ground_mask=analysis.get("ground_mask_for_visualization"),
        landmarks_ground=analysis["landmarks_json"].get("landmarks", {}),
        output_path=output_dir / "visualization.png",
    )


def run_video(input_path: Path, output_dir: Path, args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_dir / "visualization.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frames_landmarks = []
    frames_planes = []
    first_visualization_saved = False

    with MediaPipePoseEstimator(
        model_path=args.mediapipe_model,
        running_mode="video",
        output_segmentation_masks=True,
    ) as pose_estimator:
        frame_index = 0
        processed = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_index % args.frame_stride != 0:
                frame_index += 1
                continue
            if args.max_video_frames and processed >= args.max_video_frames:
                break

            timestamp_ms = int(frame_index * 1000 / fps)
            pose = pose_estimator.detect_video_frame(frame_bgr, timestamp_ms)
            frame_output_dir = output_dir / "frames" / f"frame_{frame_index:06d}"
            frame_output_dir.mkdir(parents=True, exist_ok=True)
            analysis = analyze_frame(
                image_bgr=frame_bgr,
                pose=pose,
                args=args,
                output_dir=frame_output_dir,
                frame_index=frame_index,
            )
            save_optional_geometry_outputs(frame_output_dir, analysis, args)
            frames_landmarks.append(analysis["landmarks_json"])
            frames_planes.append(analysis["ground_plane_json"])

            visualization = save_visualization(
                image_bgr=frame_bgr,
                ground_mask=analysis.get("ground_mask_for_visualization"),
                landmarks_ground=analysis["landmarks_json"].get("landmarks", {}),
                output_path=frame_output_dir / "visualization.png",
            )
            writer.write(visualization)
            if not first_visualization_saved:
                cv2.imwrite(str(output_dir / "visualization.png"), visualization)
                first_visualization_saved = True

            processed += 1
            frame_index += 1

    cap.release()
    writer.release()

    save_json(
        output_dir / "result_landmarks_ground.json",
        {"input_type": "video", "frames": frames_landmarks},
    )
    save_json(
        output_dir / "ground_plane.json",
        {"input_type": "video", "frames": frames_planes},
    )


def analyze_frame(
    image_bgr: np.ndarray,
    pose: PoseDetection,
    args: argparse.Namespace,
    output_dir: Path,
    frame_index: Optional[int],
) -> Dict[str, Any]:
    geometry = predict_geometry(image_bgr, args)
    ground_mask, segmentation_metadata = predict_ground_mask(image_bgr, pose, args)
    ground_mask = resize_mask(ground_mask, geometry.shape)
    valid_geometry_mask = geometry.valid_mask()
    ground_mask, exclusion_metadata = remove_person_from_ground_mask(
        ground_mask,
        person_mask=pose.person_mask,
        pose_landmarks=pose.landmarks,
        image_shape=image_bgr.shape[:2],
    )

    candidate_mask = ground_mask & valid_geometry_mask
    candidate_flat_indices = np.flatnonzero(candidate_mask.reshape(-1))
    if args.max_ground_points and len(candidate_flat_indices) > args.max_ground_points:
        rng = np.random.default_rng(0)
        sampled_positions = np.sort(
            rng.choice(
                len(candidate_flat_indices),
                size=int(args.max_ground_points),
                replace=False,
            )
        )
        sampled_candidate_flat_indices = candidate_flat_indices[sampled_positions]
        candidate_points = geometry.points.reshape(-1, 3)[sampled_candidate_flat_indices]
        ground_point_sampling = {
            "sampled_for_ransac": True,
            "raw_candidate_points": int(len(candidate_flat_indices)),
            "sampled_candidate_points": int(len(sampled_candidate_flat_indices)),
        }
    else:
        sampled_candidate_flat_indices = candidate_flat_indices
        candidate_points = geometry.points.reshape(-1, 3)[sampled_candidate_flat_indices]
        ground_point_sampling = {
            "sampled_for_ransac": False,
            "raw_candidate_points": int(len(candidate_flat_indices)),
            "sampled_candidate_points": int(len(sampled_candidate_flat_indices)),
        }

    ground_plane_json: Dict[str, Any] = {
        "frame_index": frame_index,
        "status": "failed",
        "geometry_metadata": geometry.metadata,
        "segmentation_metadata": segmentation_metadata,
        "person_exclusion_metadata": exclusion_metadata,
        "ground_point_sampling": ground_point_sampling,
        "important_limitations": IMPORTANT_LIMITATIONS,
    }
    landmarks_json: Dict[str, Any] = {
        "frame_index": frame_index,
        "pose_metadata": pose.metadata,
        "geometry_metadata": geometry.metadata,
        "segmentation_metadata": segmentation_metadata,
        "landmarks": {},
        "important_note": (
            "MediaPipe pose_world_landmarks are not used. MediaPipe 2D pixels "
            "index into the dense MoGe/Metric3D scene point map."
        ),
    }

    try:
        fit = fit_plane_open3d(
            candidate_points,
            distance_threshold=args.ransac_distance_threshold,
            num_iterations=args.ransac_num_iterations,
        )
        sampled_points = sample_landmark_scene_points(
            pose.landmarks,
            geometry.points,
            valid_geometry_mask,
            image_bgr.shape[:2],
        )
        mid_hip = midpoint(sampled_points, "left_hip", "right_hip")
        mid_shoulder = midpoint(sampled_points, "left_shoulder", "right_shoulder")
        plane_model, normal, orient_meta = orient_plane_with_body(
            fit.plane_model,
            fit.unit_normal,
            mid_hip,
            mid_shoulder,
        )
        fit.plane_model = plane_model
        fit.unit_normal = normal
        fit.metadata.update(orient_meta)

        inlier_points = candidate_points[fit.inlier_indices]
        frame = build_ground_coordinate_frame(
            plane_model=fit.plane_model,
            inlier_points=inlier_points,
            landmark_scene_points=sampled_points,
        )
        landmarks_ground = transform_landmarks_to_ground_frame(
            landmarks_2d=pose.landmarks,
            point_map=geometry.points,
            valid_mask=valid_geometry_mask,
            frame=frame,
            image_shape=image_bgr.shape[:2],
        )
        ground_plane_json.update(fit.to_json_dict())
        ground_plane_json["ground_point_sampling"] = ground_point_sampling
        ground_plane_json.update(
            {
                "status": "ok",
                "P0": frame["P0"].astype(float).tolist(),
                "ex": frame["ex"].astype(float).tolist(),
                "ey": frame["ey"].astype(float).tolist(),
                "ez": frame["ez"].astype(float).tolist(),
                "x_axis_source": frame["x_axis_source"],
            }
        )
        landmarks_json["landmarks"] = landmarks_ground
        landmarks_json["ground_frame"] = {
            "P0": frame["P0"].astype(float).tolist(),
            "ex": frame["ex"].astype(float).tolist(),
            "ey": frame["ey"].astype(float).tolist(),
            "ez": frame["ez"].astype(float).tolist(),
        }
        ground_inlier_mask = np.zeros(candidate_mask.shape, dtype=bool)
        inlier_flat = sampled_candidate_flat_indices[fit.inlier_indices]
        ground_inlier_mask.reshape(-1)[inlier_flat] = True
    except Exception as exc:
        ground_plane_json["error"] = str(exc)
        landmarks_json["error"] = "ground_plane_fit_failed"
        landmarks_json["landmarks"] = _invalid_landmark_rows(pose.landmarks, str(exc))
        ground_inlier_mask = np.zeros(candidate_mask.shape, dtype=bool)

    return {
        "geometry": geometry,
        "landmarks_json": landmarks_json,
        "ground_plane_json": ground_plane_json,
        "ground_mask_for_visualization": resize_mask(ground_mask, image_bgr.shape[:2]),
        "candidate_mask": candidate_mask,
        "ground_inlier_mask": ground_inlier_mask,
        "output_dir": output_dir,
    }


def predict_geometry(image_bgr: np.ndarray, args: argparse.Namespace) -> GeometryPrediction:
    if args.geometry_backend == "moge":
        try:
            return MoGeBackend().predict(image_bgr)
        except NotImplementedError as moge_error:
            try:
                prediction = Metric3DBackend(model_name=args.metric3d_model).predict(
                    image_bgr
                )
                prediction.metadata["fallback_from"] = "moge"
                prediction.metadata["moge_error"] = str(moge_error)
                return prediction
            except NotImplementedError as metric_error:
                raise NotImplementedError(
                    "Neither MoGe-2 nor Metric3D v2 could be loaded. "
                    f"MoGe error: {moge_error} Metric3D error: {metric_error}"
                ) from metric_error

    if args.geometry_backend == "metric3d":
        return Metric3DBackend(model_name=args.metric3d_model).predict(image_bgr)

    raise ValueError(f"Unknown geometry backend: {args.geometry_backend}")


def predict_ground_mask(
    image_bgr: np.ndarray,
    pose: PoseDetection,
    args: argparse.Namespace,
) -> tuple[np.ndarray, Dict[str, Any]]:
    if args.seg_backend == "grounded_sam":
        try:
            result = GroundedSAMBackend(prompt=args.ground_prompt).predict(image_bgr)
            result.metadata["ground_prompt"] = args.ground_prompt
            return result.mask, result.metadata
        except NotImplementedError as exc:
            fallback = LowerRegionBackend(args.lower_region_ratio).predict(image_bgr)
            fallback.metadata["fallback_from"] = "grounded_sam"
            fallback.metadata["grounded_sam_error"] = str(exc)
            fallback.metadata["ground_prompt"] = args.ground_prompt
            return fallback.mask, fallback.metadata

    if args.seg_backend == "manual_mask":
        result = ManualMaskBackend(args.manual_mask_path).predict(image_bgr)
        return result.mask, result.metadata

    if args.seg_backend == "lower_region":
        result = LowerRegionBackend(args.lower_region_ratio).predict(image_bgr)
        return result.mask, result.metadata

    raise ValueError(f"Unknown segmentation backend: {args.seg_backend}")


def save_optional_geometry_outputs(
    output_dir: Path,
    analysis: Dict[str, Any],
    args: Optional[argparse.Namespace] = None,
) -> None:
    geometry: GeometryPrediction = analysis["geometry"]
    if geometry.depth is not None:
        save_depth_png(geometry.depth, output_dir / "depth.png")
    if geometry.normal is not None:
        save_normal_png(geometry.normal, output_dir / "normal.png")
    if args is not None and args.save_point_cloud:
        valid_mask = geometry.valid_mask()
        write_ply(output_dir / "point_cloud.ply", geometry.points[valid_mask])
        inlier_mask = analysis.get("ground_inlier_mask")
        if inlier_mask is not None and bool(np.asarray(inlier_mask).any()):
            write_ply(output_dir / "ground_inliers.ply", geometry.points[inlier_mask])


def write_ply(path: Path, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")
        for x, y, z in points:
            file.write(f"{x:.8f} {y:.8f} {z:.8f}\n")


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _invalid_landmark_rows(
    landmarks: Dict[str, Dict[str, Any]],
    reason: str,
) -> Dict[str, Dict[str, Any]]:
    rows = {}
    for name, landmark in landmarks.items():
        rows[name] = {
            "index": int(landmark.get("index", -1)),
            "x_px": float(landmark.get("x_px", np.nan)),
            "y_px": float(landmark.get("y_px", np.nan)),
            "visibility": float(landmark.get("visibility", 0.0)),
            "presence": float(landmark.get("presence", 0.0)),
            "valid": False,
            "invalid_reason": f"ground_plane_fit_failed: {reason}",
            "scene_point": None,
            "ground": None,
        }
    return rows


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate ground-referenced 3D pose coordinates from image/video."
    )
    parser.add_argument("--path", help="Input image or video path.")
    parser.add_argument("--image_path", help="Alias for --path for single-image scripts.")
    parser.add_argument("--output_dir", required=True, help="Directory for outputs.")
    parser.add_argument(
        "--geometry_backend",
        choices=["moge", "metric3d"],
        default="moge",
        help="Dense geometry backend. MoGe falls back to Metric3D if unavailable.",
    )
    parser.add_argument(
        "--seg_backend",
        choices=["grounded_sam", "manual_mask", "lower_region"],
        default="grounded_sam",
        help="Ground segmentation backend.",
    )
    parser.add_argument(
        "--ground_prompt",
        default="floor. ground. road. gym floor.",
        help="Prompt for Grounded SAM-style ground segmentation.",
    )
    parser.add_argument(
        "--manual_mask_path",
        help="Binary ground mask path when --seg_backend manual_mask is used.",
    )
    parser.add_argument(
        "--lower_region_ratio",
        type=float,
        default=0.40,
        help="Fallback lower-image ratio used as rough ground candidate mask.",
    )
    parser.add_argument(
        "--ransac_distance_threshold",
        type=float,
        default=0.03,
        help="Open3D/NumPy RANSAC plane distance threshold.",
    )
    parser.add_argument(
        "--ransac_num_iterations",
        type=int,
        default=1000,
        help="Open3D/NumPy RANSAC iteration count.",
    )
    parser.add_argument(
        "--mediapipe_model",
        default=str(DEFAULT_MODEL_PATH),
        help="MediaPipe Pose Landmarker .task model path.",
    )
    parser.add_argument(
        "--metric3d_model",
        default="metric3d_vit_small",
        help=(
            "Torch Hub Metric3D model name. Use metric3d_vit_small for CPU "
            "smoke tests; metric3d_vit_giant2 is heavier."
        ),
    )
    parser.add_argument(
        "--max_ground_points",
        type=int,
        default=200000,
        help="Maximum ground candidate points sampled before RANSAC.",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Process every Nth frame for video input.",
    )
    parser.add_argument(
        "--max_video_frames",
        type=int,
        default=0,
        help="Maximum processed video frames; 0 means all frames.",
    )
    parser.add_argument(
        "--save_point_cloud",
        action="store_true",
        help="Optionally save point_cloud.ply and ground_inliers.ply.",
    )
    args = parser.parse_args(argv)

    if not args.path and not args.image_path:
        parser.error("Provide --path or --image_path.")
    if args.seg_backend == "manual_mask" and not args.manual_mask_path:
        parser.error("--manual_mask_path is required with --seg_backend manual_mask.")
    if args.frame_stride < 1:
        parser.error("--frame_stride must be >= 1.")
    return args


IMPORTANT_LIMITATIONS = [
    "Monocular geometry estimation can be wrong and may not be metrically scaled.",
    "If the real floor is heavily occluded, RANSAC may fail or fit the wrong plane.",
    "If Grounded SAM or the selected segmentation backend misses the floor, the ground normal will be wrong.",
    "Do not mix MediaPipe pose_world_landmarks with MoGe/Metric3D scene geometry.",
    "The stable approach is: MediaPipe provides 2D joint pixels; MoGe/Metric3D provides scene 3D points at those same pixels.",
]


if __name__ == "__main__":
    main()
