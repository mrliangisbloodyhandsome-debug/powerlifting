from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .pose_landmarker import DEFAULT_MODEL_PATH, PoseLandmarker
    from .squat_analyzer import (
        AUTO_LEVEL_AUTOMATIC,
        UNCERTAIN,
        RuleResult,
        SquatAnalyzerConfig,
        SquatDoubleBounceRule,
        SquatFinishKneesLockedRule,
        SquatFootMovementRule,
        SquatStartKneesLockedRule,
        detect_squat_events,
        evaluate_ground_depth,
        extract_squat_features,
        load_pose_json,
        make_final_decision,
    )
except ImportError:  # Allows: .\.venv\Scripts\python.exe src\squat_pipeline.py
    from pose_landmarker import DEFAULT_MODEL_PATH, PoseLandmarker
    from squat_analyzer import (
        AUTO_LEVEL_AUTOMATIC,
        UNCERTAIN,
        RuleResult,
        SquatAnalyzerConfig,
        SquatDoubleBounceRule,
        SquatFinishKneesLockedRule,
        SquatFootMovementRule,
        SquatStartKneesLockedRule,
        detect_squat_events,
        evaluate_ground_depth,
        extract_squat_features,
        load_pose_json,
        make_final_decision,
    )

from ground_pose import cli as ground_pose_cli
from ground_pose.mediapipe_pose import PoseDetection
from ground_pose.visualize import save_visualization


DEFAULT_INPUT_VIDEO = REPO_ROOT / "data" / "your_squat_video.mp4"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "runs"
REPORT_FILENAME = "report.json"
POSE_LANDMARKS_FILENAME = "pose_landmarks.json"
POSE_OVERLAY_FILENAME = "pose_overlay.mp4"


def run_squat_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    input_video_path = Path(args.video)
    output_dir, run_id = build_run_output_dir(
        input_path=input_video_path,
        lift_type="squat",
        output_root=Path(args.output_dir),
        run_id=getattr(args, "run_id", None),
    )
    report_path = output_dir / REPORT_FILENAME

    pose_json_path = output_dir / POSE_LANDMARKS_FILENAME
    pose_video_path = (
        output_dir / POSE_OVERLAY_FILENAME
        if args.draw_pose_video and not args.pose_json
        else None
    )

    if args.pose_json:
        pose_source = "existing_pose_json_copied_to_run_dir"
        copy_pose_json_into_run_dir(Path(args.pose_json), pose_json_path)
    else:
        pose_source = "mediapipe_video"
        run_mediapipe_video_branch(
            input_video_path=input_video_path,
            output_json_path=pose_json_path,
            output_video_path=pose_video_path,
            model_path=Path(args.mediapipe_model),
            draw=bool(args.draw_pose_video),
        )

    config = SquatAnalyzerConfig(
        pose_confidence_min=args.pose_confidence_min,
        depth_margin_m=args.depth_margin_m,
        knee_locked_angle_deg=args.knee_locked_angle_deg,
        lock_standing_window_frames=args.lock_standing_window_frames,
        lock_angle_percentile=args.lock_angle_percentile,
        lock_near_side_weight=args.lock_near_side_weight,
        event_flexion_angle_drop_deg=args.event_flexion_angle_drop_deg,
        event_onset_angle_drop_deg=args.event_onset_angle_drop_deg,
        event_onset_min_consecutive_frames=args.event_onset_min_consecutive_frames,
        event_start_preroll_frames=args.event_start_preroll_frames,
        event_extension_angle_tolerance_deg=args.event_extension_angle_tolerance_deg,
        min_flexion_angle_travel_deg=args.min_flexion_angle_travel_deg,
        foot_movement_px=args.foot_movement_px,
        double_bounce_rebound_px=args.double_bounce_rebound_px,
        double_bounce_second_drop_px=args.double_bounce_second_drop_px,
    )

    pose_branch = analyze_full_video_pose_branch(
        pose_json_path=pose_json_path,
        config=config,
        pose_source=pose_source,
    )

    ground_branch = analyze_ground_depth_branch(
        input_video_path=input_video_path,
        output_dir=output_dir / "bottom_ground_depth",
        events=pose_branch["events_object"],
        pose_frames_data=pose_branch["pose_frames_data"],
        config=config,
        args=args,
    )

    rule_results = ground_branch["rule_results"] + pose_branch["rule_results"]
    final_result = {
        "schema_version": "squat_adjudication_v1",
        "run_id": run_id,
        "input": {
            "video_path": str(input_video_path),
            "lift_type": "squat",
        },
        "output": {
            "run_id": run_id,
            "run_dir": str(output_dir),
            "report_path": str(report_path),
            "layout": "outputs/runs/<run_id>/report.json",
        },
        "runtime_flow": [
            "1. MediaPipe detects pose landmarks on the full test video.",
            "2. Independent branches run from the shared pose timeline.",
            (
                "2.1 The bottom squat frame is saved for dense geometry, while "
                "the landmark pixels are reused from the full-video MediaPipe result."
            ),
            "2.2 Full-video pose landmarks are used for double-bounce, knee-lock, and foot-movement rules.",
            "2.3 Start/end timestamps are exported as the future barbell-tracking window.",
            "3. All rule outputs are written into this single JSON report.",
        ],
        "branch_model": "logical_parallel_after_full_video_pose",
        "decision": make_final_decision(rule_results),
        "events": asdict(pose_branch["events_object"]),
        "barbell_tracking_window": build_barbell_tracking_window(pose_branch["events_object"]),
        "rule_results": [asdict(result) for result in rule_results],
        "branches": {
            "full_video_mediapipe_pose": {
                "status": "ok",
                "source": pose_source,
                "pose_json_path": str(pose_json_path),
                "pose_video_path": str(pose_video_path) if pose_video_path else None,
                "frame_summary": pose_branch["frame_summary"],
            },
            "bottom_frame_ground_depth": ground_branch["branch_summary"],
            "full_video_pose_rules": pose_branch["branch_summary"],
            "barbell_tracking_window": {
                "status": "ready_for_future_barbell_model",
                "description": (
                    "Use these timestamps to run a future barbell detector/tracker "
                    "and check whether barbell y descends during ascent."
                ),
            },
        },
        "artifacts": {
            "report_json_path": str(report_path),
            "output_json_path": str(report_path),
            "pose_json_path": str(pose_json_path),
            "pose_video_path": str(pose_video_path) if pose_video_path else None,
            **ground_branch["artifacts"],
        },
    }

    save_json(report_path, final_result)
    return final_result


def build_run_output_dir(
    input_path: Path,
    lift_type: str,
    output_root: Path,
    run_id: Optional[str] = None,
) -> tuple[Path, str]:
    run_id = run_id or build_run_id(input_path=input_path, lift_type=lift_type)
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, run_id


def build_run_id(
    input_path: Path,
    lift_type: str,
    now: Optional[datetime] = None,
) -> str:
    now = now or datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    milliseconds = f"{now.microsecond // 1000:03d}"
    safe_lift_type = sanitize_run_id_part(lift_type)
    safe_video_stem = sanitize_run_id_part(input_path.stem)
    hash8 = input_hash8(input_path=input_path, lift_type=safe_lift_type)
    return f"{timestamp}_{milliseconds}_{safe_lift_type}_{safe_video_stem}_{hash8}"


def sanitize_run_id_part(value: str) -> str:
    raw = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    sanitized = re.sub(r"_+", "_", raw).strip("_")
    return sanitized or "unknown"


def input_hash8(input_path: Path, lift_type: str) -> str:
    parts = [str(input_path.resolve()), lift_type]
    if input_path.exists():
        stat = input_path.stat()
        parts.extend([str(stat.st_size), str(stat.st_mtime_ns)])
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:8]


def copy_pose_json_into_run_dir(source_path: Path, target_path: Path) -> None:
    source_path = source_path.resolve()
    target_path = target_path.resolve()
    if source_path == target_path:
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)


def run_mediapipe_video_branch(
    input_video_path: Path,
    output_json_path: Path,
    output_video_path: Optional[Path],
    model_path: Path,
    draw: bool,
) -> None:
    pose_landmarker = PoseLandmarker(model_path=model_path)
    pose_landmarker.process_video(
        input_video_path=input_video_path,
        output_json_path=output_json_path,
        output_video_path=output_video_path,
        draw=draw,
    )


def analyze_full_video_pose_branch(
    pose_json_path: Path,
    config: SquatAnalyzerConfig,
    pose_source: str,
) -> Dict[str, Any]:
    frames_data = load_pose_json(pose_json_path)
    frames = extract_squat_features(frames_data, config)
    events = detect_squat_events(frames, config)
    rules = [
        SquatDoubleBounceRule(),
        SquatStartKneesLockedRule(),
        SquatFinishKneesLockedRule(),
        SquatFootMovementRule(),
    ]
    rule_results = [rule.evaluate(frames, config, events) for rule in rules]

    return {
        "events_object": events,
        "rule_results": rule_results,
        "frame_summary": {
            "pose_source": pose_source,
            "total_frames": len(frames_data),
            "valid_squat_frames": len(frames),
            "pose_confidence_min": config.pose_confidence_min,
        },
        "branch_summary": {
            "status": "ok" if frames else "no_valid_pose_frames",
            "rule_ids": [result.rule_id for result in rule_results],
        },
        "pose_frames_data": frames_data,
    }


def analyze_ground_depth_branch(
    input_video_path: Path,
    output_dir: Path,
    events: Any,
    pose_frames_data: Sequence[Dict[str, Any]],
    config: SquatAnalyzerConfig,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    artifacts = {
        "bottom_frame_image_path": None,
        "ground_landmarks_json_path": None,
        "ground_plane_json_path": None,
        "ground_visualization_path": None,
    }

    if events.bottom_frame is None:
        result = RuleResult(
            rule_id="SQ_DEPTH_GROUND",
            status=UNCERTAIN,
            auto_level=AUTO_LEVEL_AUTOMATIC,
            reason="没有可靠最低点帧，无法截取单帧进行地面三维深度判断",
            confidence=0.0,
        )
        return {
            "rule_results": [result],
            "branch_summary": {"status": "skipped", "reason": result.reason},
            "artifacts": artifacts,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    bottom_image_path = output_dir / f"bottom_frame_{events.bottom_frame:06d}.png"
    artifacts["bottom_frame_image_path"] = str(bottom_image_path)

    try:
        image_bgr = save_video_frame(input_video_path, events.bottom_frame, bottom_image_path)
        bottom_pose_frame = find_pose_frame_data(pose_frames_data, events.bottom_frame)
        pose = pose_detection_from_full_video_frame(bottom_pose_frame)

        ground_args = build_ground_args(args)
        analysis = ground_pose_cli.analyze_frame(
            image_bgr=image_bgr,
            pose=pose,
            args=ground_args,
            output_dir=output_dir,
            frame_index=events.bottom_frame,
        )
        landmarks_json = analysis["landmarks_json"]
        ground_plane_json = analysis["ground_plane_json"]

        landmarks_path = output_dir / "result_landmarks_ground.json"
        plane_path = output_dir / "ground_plane.json"
        visualization_path = output_dir / "visualization.png"
        save_json(landmarks_path, landmarks_json)
        save_json(plane_path, ground_plane_json)
        ground_pose_cli.save_optional_geometry_outputs(output_dir, analysis, ground_args)
        save_visualization(
            image_bgr=image_bgr,
            ground_mask=analysis.get("ground_mask_for_visualization"),
            landmarks_ground=landmarks_json.get("landmarks", {}),
            output_path=visualization_path,
        )

        artifacts.update(
            {
                "ground_landmarks_json_path": str(landmarks_path),
                "ground_plane_json_path": str(plane_path),
                "ground_visualization_path": str(visualization_path),
            }
        )
        depth_result = evaluate_ground_depth(
            landmarks_ground=landmarks_json.get("landmarks", {}),
            config=config,
            bottom_frame_index=events.bottom_frame,
        )
        return {
            "rule_results": [depth_result],
            "branch_summary": {
                "status": "ok",
                "bottom_frame": events.bottom_frame,
                "pose_landmark_source": "full_video_mediapipe_pose_json",
                "ground_plane_status": ground_plane_json.get("status"),
                "rule_ids": [depth_result.rule_id],
            },
            "artifacts": artifacts,
        }
    except Exception as exc:
        result = RuleResult(
            rule_id="SQ_DEPTH_GROUND",
            status=UNCERTAIN,
            auto_level=AUTO_LEVEL_AUTOMATIC,
            reason=f"最低点地面三维深度分支失败：{type(exc).__name__}: {exc}",
            evidence_frames=[events.bottom_frame],
            confidence=0.0,
            details={"bottom_frame_image_path": str(bottom_image_path)},
        )
        return {
            "rule_results": [result],
            "branch_summary": {
                "status": "failed",
                "bottom_frame": events.bottom_frame,
                "error": result.reason,
            },
            "artifacts": artifacts,
        }


def build_ground_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        geometry_backend=args.geometry_backend,
        seg_backend=args.seg_backend,
        ground_prompt=args.ground_prompt,
        manual_mask_path=args.manual_mask_path,
        lower_region_ratio=args.lower_region_ratio,
        ransac_distance_threshold=args.ransac_distance_threshold,
        ransac_num_iterations=args.ransac_num_iterations,
        mediapipe_model=args.mediapipe_model,
        metric3d_model=args.metric3d_model,
        max_ground_points=args.max_ground_points,
        save_point_cloud=args.save_point_cloud,
    )


def find_pose_frame_data(
    frames_data: Sequence[Dict[str, Any]],
    frame_index: int,
) -> Dict[str, Any]:
    for frame_data in frames_data:
        if int(frame_data.get("frame_index", -1)) == int(frame_index):
            return frame_data
    raise RuntimeError(f"Full-video pose JSON does not contain frame {frame_index}.")


def pose_detection_from_full_video_frame(frame_data: Dict[str, Any]) -> PoseDetection:
    poses = frame_data.get("poses") or []
    metadata = {
        "landmark_source": "full_video_mediapipe_pose_json",
        "frame_index": int(frame_data.get("frame_index", -1)),
        "timestamp_ms": int(frame_data.get("timestamp_ms", 0)),
        "poses": len(poses),
        "person_mask_source": None,
        "person_mask_note": (
            "The bottom-frame ground branch reuses tracked full-video landmarks "
            "instead of rerunning MediaPipe on the still image. Person exclusion "
            "therefore falls back to the pose landmark bbox."
        ),
    }

    if not poses:
        return PoseDetection(landmarks={}, person_mask=None, metadata=metadata)

    pose = poses[0]
    landmarks = {
        name: dict(landmark)
        for name, landmark in pose.get("landmarks", {}).items()
    }
    metadata["pose_id"] = int(pose.get("pose_id", 0))
    return PoseDetection(
        landmarks=landmarks,
        person_mask=None,
        metadata=metadata,
    )


def save_video_frame(video_path: Path, frame_index: int, output_path: Path) -> Any:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame_bgr)
        return frame_bgr
    finally:
        cap.release()


def build_barbell_tracking_window(events: Any) -> Dict[str, Any]:
    return {
        "lift_start_frame": events.lift_start_frame,
        "lift_start_timestamp_ms": events.lift_start_timestamp_ms,
        "bottom_frame": events.bottom_frame,
        "bottom_timestamp_ms": events.bottom_timestamp_ms,
        "lift_end_frame": events.lift_end_frame,
        "lift_end_timestamp_ms": events.lift_end_timestamp_ms,
        "future_rule": "SQ_BAR_DOWNWARD_ASCENT",
        "future_input_needed": "barbell_y_series_between_bottom_and_lift_end",
    }


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the squat MediaPipe -> ground-depth -> rule JSON pipeline."
    )
    parser.add_argument(
        "--video",
        default=str(DEFAULT_INPUT_VIDEO),
        help="Input squat test video.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Root directory where timestamped run folders are created.",
    )
    parser.add_argument(
        "--run_id",
        help=(
            "Optional run folder name. Defaults to "
            "YYYYMMDD_HHMMSS_mmm_<lift_type>_<video_stem>_<hash8>."
        ),
    )
    parser.add_argument(
        "--pose_json",
        help="Reuse an existing full-video MediaPipe pose JSON instead of rerunning pose detection.",
    )
    parser.add_argument(
        "--draw_pose_video",
        dest="draw_pose_video",
        action="store_true",
        default=True,
        help="Save a full-video MediaPipe pose overlay.",
    )
    parser.add_argument(
        "--no_draw_pose_video",
        dest="draw_pose_video",
        action="store_false",
        help="Skip the full-video MediaPipe pose overlay.",
    )
    parser.add_argument(
        "--mediapipe_model",
        default=str(DEFAULT_MODEL_PATH),
        help="MediaPipe Pose Landmarker .task model path.",
    )
    parser.add_argument("--pose_confidence_min", type=float, default=0.50)
    parser.add_argument("--depth_margin_m", type=float, default=0.0)
    parser.add_argument("--knee_locked_angle_deg", type=float, default=155.0)
    parser.add_argument(
        "--lock_standing_window_frames",
        type=int,
        default=15,
        help="Frames used to judge knee lock from the standing window before/after the squat.",
    )
    parser.add_argument(
        "--lock_angle_percentile",
        type=float,
        default=0.85,
        help="Upper percentile knee angle used for each standing-window side.",
    )
    parser.add_argument(
        "--lock_near_side_weight",
        type=float,
        default=2.0,
        help="Weight multiplier for the side closer to the camera.",
    )
    parser.add_argument(
        "--event_flexion_angle_drop_deg",
        type=float,
        default=10.0,
        help="Confirm obvious flexion when both hip and knee angles drop this much from standing.",
    )
    parser.add_argument(
        "--event_onset_angle_drop_deg",
        type=float,
        default=3.0,
        help="Backtrack start to sustained hip and knee angle drops at this lower threshold.",
    )
    parser.add_argument(
        "--event_onset_min_consecutive_frames",
        type=int,
        default=2,
        help="Minimum consecutive frames required for lower-threshold flexion onset.",
    )
    parser.add_argument(
        "--event_start_preroll_frames",
        type=int,
        default=3,
        help="Export the squat start this many frames before sustained flexion onset.",
    )
    parser.add_argument(
        "--event_extension_angle_tolerance_deg",
        type=float,
        default=12.0,
        help="End squat when both hip and knee angles return within this much of standing.",
    )
    parser.add_argument(
        "--min_flexion_angle_travel_deg",
        type=float,
        default=20.0,
        help="Minimum hip and knee angle travel required before using angle-based event detection.",
    )
    parser.add_argument("--foot_movement_px", type=float, default=25.0)
    parser.add_argument("--double_bounce_rebound_px", type=float, default=10.0)
    parser.add_argument("--double_bounce_second_drop_px", type=float, default=8.0)

    parser.add_argument(
        "--geometry_backend",
        choices=["moge", "metric3d"],
        default="moge",
        help="Dense geometry backend for bottom-frame ground coordinates.",
    )
    parser.add_argument(
        "--seg_backend",
        choices=["grounded_sam", "manual_mask", "lower_region"],
        default="lower_region",
        help="Ground segmentation backend for the bottom frame.",
    )
    parser.add_argument(
        "--ground_prompt",
        default="floor. ground. road. gym floor.",
        help="Prompt for Grounded SAM-style ground segmentation.",
    )
    parser.add_argument("--manual_mask_path")
    parser.add_argument("--lower_region_ratio", type=float, default=0.40)
    parser.add_argument("--ransac_distance_threshold", type=float, default=0.03)
    parser.add_argument("--ransac_num_iterations", type=int, default=1000)
    parser.add_argument("--metric3d_model", default="metric3d_vit_small")
    parser.add_argument("--max_ground_points", type=int, default=200000)
    parser.add_argument("--save_point_cloud", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    result = run_squat_pipeline(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
