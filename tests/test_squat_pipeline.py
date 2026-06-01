import argparse
import json
import re
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.squat_analyzer import PASS, SquatAnalyzerConfig, SquatEvents
from src.squat_pipeline import (
    analyze_ground_depth_branch,
    build_run_id,
    build_run_output_dir,
    find_pose_frame_data,
    pose_detection_from_full_video_frame,
    run_squat_pipeline,
)


def pose_frame(frame_index=7):
    return {
        "frame_index": frame_index,
        "timestamp_ms": frame_index * 100,
        "poses": [
            {
                "pose_id": 0,
                "landmarks": {
                    "left_hip": {
                        "index": 23,
                        "x_px": 123.0,
                        "y_px": 456.0,
                        "visibility": 0.9,
                        "presence": 0.8,
                    },
                    "left_knee": {
                        "index": 25,
                        "x_px": 124.0,
                        "y_px": 567.0,
                        "visibility": 0.95,
                        "presence": 0.85,
                    },
                },
            }
        ],
    }


def ground_args():
    return argparse.Namespace(
        geometry_backend="metric3d",
        seg_backend="lower_region",
        ground_prompt="floor",
        manual_mask_path=None,
        lower_region_ratio=0.40,
        ransac_distance_threshold=0.03,
        ransac_num_iterations=1000,
        mediapipe_model="unused.task",
        metric3d_model="metric3d_vit_small",
        max_ground_points=100,
        save_point_cloud=False,
    )


def pipeline_args(video, output_dir, pose_json):
    return argparse.Namespace(
        video=str(video),
        output_dir=str(output_dir),
        run_id="test_run",
        pose_json=str(pose_json),
        draw_pose_video=True,
        mediapipe_model="unused.task",
        pose_confidence_min=0.50,
        depth_margin_m=0.0,
        knee_locked_angle_deg=155.0,
        lock_standing_window_frames=15,
        lock_angle_percentile=0.85,
        lock_near_side_weight=2.0,
        event_flexion_angle_drop_deg=10.0,
        event_onset_angle_drop_deg=3.0,
        event_onset_min_consecutive_frames=2,
        event_start_preroll_frames=3,
        event_extension_angle_tolerance_deg=12.0,
        min_flexion_angle_travel_deg=20.0,
        foot_movement_px=25.0,
        double_bounce_rebound_px=10.0,
        double_bounce_second_drop_px=8.0,
        geometry_backend="metric3d",
        seg_backend="lower_region",
        ground_prompt="floor",
        manual_mask_path=None,
        lower_region_ratio=0.40,
        ransac_distance_threshold=0.03,
        ransac_num_iterations=1000,
        metric3d_model="metric3d_vit_small",
        max_ground_points=100,
        save_point_cloud=False,
    )


class SquatPipelineTests(unittest.TestCase):
    def test_build_run_id_uses_requested_default_format(self):
        run_id = build_run_id(
            input_path=Path("Test Videos/My Squat Attempt.mp4"),
            lift_type="squat",
            now=datetime(2026, 6, 1, 22, 44, 22, 123000),
        )

        self.assertRegex(
            run_id,
            re.compile(r"^20260601_224422_123_squat_my_squat_attempt_[0-9a-f]{8}$"),
        )

    def test_build_run_output_dir_places_run_under_output_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir, run_id = build_run_output_dir(
                input_path=Path("sample.mp4"),
                lift_type="squat",
                output_root=Path(tmp),
                run_id="custom_run",
            )

            self.assertEqual(run_id, "custom_run")
            self.assertEqual(output_dir, Path(tmp) / "custom_run")
            self.assertTrue(output_dir.exists())

    def test_run_squat_pipeline_writes_main_report_in_run_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_pose_json = tmp_path / "source_pose.json"
            source_pose_json.write_text(
                json.dumps([pose_frame()], ensure_ascii=False),
                encoding="utf-8",
            )

            result = run_squat_pipeline(
                pipeline_args(
                    video=tmp_path / "squat.mp4",
                    output_dir=tmp_path / "runs",
                    pose_json=source_pose_json,
                )
            )

            run_dir = tmp_path / "runs" / "test_run"
            self.assertEqual(result["run_id"], "test_run")
            self.assertTrue((run_dir / "report.json").exists())
            self.assertTrue((run_dir / "pose_landmarks.json").exists())
            self.assertEqual(result["artifacts"]["report_json_path"], str(run_dir / "report.json"))
            self.assertIsNone(result["artifacts"]["pose_video_path"])

    def test_pose_detection_reuses_full_video_frame_landmarks(self):
        pose = pose_detection_from_full_video_frame(pose_frame())

        self.assertEqual(pose.metadata["landmark_source"], "full_video_mediapipe_pose_json")
        self.assertIsNone(pose.person_mask)
        self.assertEqual(pose.landmarks["left_hip"]["x_px"], 123.0)

    def test_ground_depth_branch_passes_full_video_landmarks_to_ground_pose(self):
        events = SquatEvents(
            status="ok",
            reason="test",
            bottom_frame=7,
            bottom_timestamp_ms=700,
        )
        captured = {}

        def fake_analyze_frame(image_bgr, pose, args, output_dir, frame_index):
            captured["pose"] = pose
            return {
                "geometry": object(),
                "landmarks_json": {
                    "landmarks": {
                        "left_hip": {
                            "valid": True,
                            "visibility": 0.9,
                            "presence": 0.8,
                            "ground": {"Zg": 0.40},
                        },
                        "left_knee": {
                            "valid": True,
                            "visibility": 0.95,
                            "presence": 0.85,
                            "ground": {"Zg": 0.55},
                        },
                    }
                },
                "ground_plane_json": {"status": "ok"},
                "ground_mask_for_visualization": np.zeros((2, 2), dtype=bool),
            }

        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.squat_pipeline.save_video_frame", return_value=np.zeros((2, 2, 3), dtype=np.uint8)):
                with patch("src.squat_pipeline.ground_pose_cli.analyze_frame", side_effect=fake_analyze_frame):
                    with patch("src.squat_pipeline.ground_pose_cli.save_optional_geometry_outputs"):
                        with patch("src.squat_pipeline.save_visualization"):
                            result = analyze_ground_depth_branch(
                                input_video_path=Path("input.mp4"),
                                output_dir=Path(tmp),
                                events=events,
                                pose_frames_data=[pose_frame()],
                                config=SquatAnalyzerConfig(),
                                args=ground_args(),
                            )

        self.assertEqual(
            captured["pose"].metadata["landmark_source"],
            "full_video_mediapipe_pose_json",
        )
        self.assertEqual(captured["pose"].landmarks["left_hip"]["x_px"], 123.0)
        self.assertEqual(result["rule_results"][0].status, PASS)
        self.assertEqual(
            result["branch_summary"]["pose_landmark_source"],
            "full_video_mediapipe_pose_json",
        )

    def test_find_pose_frame_data_requires_exact_frame(self):
        with self.assertRaisesRegex(RuntimeError, "does not contain frame 99"):
            find_pose_frame_data([pose_frame(7)], 99)


if __name__ == "__main__":
    unittest.main()
