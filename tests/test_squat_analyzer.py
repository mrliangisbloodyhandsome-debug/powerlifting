import unittest
import math

from src.squat_analyzer import (
    FAIL,
    PASS,
    LandmarkPoint,
    SquatAnalyzerConfig,
    SquatDoubleBounceRule,
    SquatEvents,
    SquatFinishKneesLockedRule,
    SquatFrameFeatures,
    SquatSideFeatures,
    SquatStartKneesLockedRule,
    detect_squat_events,
    evaluate_ground_depth,
)


def point(name, x_px, y_px, z_norm=None):
    return LandmarkPoint(
        name=name,
        x_px=float(x_px),
        y_px=float(y_px),
        visibility=1.0,
        presence=1.0,
        z_norm=z_norm,
    )


def squat_frame(frame_index, hip_y):
    side = SquatSideFeatures(
        side="left",
        shoulder=point("left_shoulder", 0.0, hip_y - 120.0),
        hip=point("left_hip", 0.0, hip_y),
        knee=point("left_knee", 0.0, hip_y + 90.0),
        ankle=point("left_ankle", 0.0, hip_y + 180.0),
        heel=point("left_heel", -10.0, hip_y + 185.0),
        foot_index=point("left_foot_index", 20.0, hip_y + 185.0),
        confidence=1.0,
    )
    return SquatFrameFeatures(
        frame_index=frame_index,
        timestamp_ms=frame_index * 100,
        sides={"left": side},
    )


def angled_side(side, hip_angle_deg, knee_angle_deg, x_offset=0.0, z_norm=None):
    hip = point(f"{side}_hip", x_offset, 100.0, z_norm)
    shoulder = point(f"{side}_shoulder", x_offset, 0.0, z_norm)

    hip_rad = math.radians(hip_angle_deg)
    thigh = (math.sin(hip_rad), -math.cos(hip_rad))
    knee = point(
        f"{side}_knee",
        x_offset + thigh[0] * 100.0,
        100.0 + thigh[1] * 100.0,
        z_norm,
    )

    knee_to_hip = ((hip.x_px - knee.x_px) / 100.0, (hip.y_px - knee.y_px) / 100.0)
    knee_rad = math.radians(knee_angle_deg)
    shank = (
        knee_to_hip[0] * math.cos(knee_rad) - knee_to_hip[1] * math.sin(knee_rad),
        knee_to_hip[0] * math.sin(knee_rad) + knee_to_hip[1] * math.cos(knee_rad),
    )
    ankle = point(
        f"{side}_ankle",
        knee.x_px + shank[0] * 100.0,
        knee.y_px + shank[1] * 100.0,
        z_norm,
    )

    return SquatSideFeatures(
        side=side,
        shoulder=shoulder,
        hip=hip,
        knee=knee,
        ankle=ankle,
        heel=point(f"{side}_heel", ankle.x_px - 10.0, ankle.y_px + 5.0, z_norm),
        foot_index=point(f"{side}_foot_index", ankle.x_px + 20.0, ankle.y_px + 5.0, z_norm),
        confidence=1.0,
    )


def angled_squat_frame(frame_index, hip_angle_deg, knee_angle_deg):
    side = angled_side("left", hip_angle_deg, knee_angle_deg)
    return SquatFrameFeatures(
        frame_index=frame_index,
        timestamp_ms=frame_index * 100,
        sides={"left": side},
    )


def two_side_angled_frame(frame_index, left_knee_angle_deg, right_knee_angle_deg):
    left = angled_side("left", 178, left_knee_angle_deg, x_offset=0.0, z_norm=-0.30)
    right = angled_side("right", 178, right_knee_angle_deg, x_offset=200.0, z_norm=0.30)
    return SquatFrameFeatures(
        frame_index=frame_index,
        timestamp_ms=frame_index * 100,
        sides={"left": left, "right": right},
    )


class SquatAnalyzerTests(unittest.TestCase):
    def test_detect_squat_events_marks_start_bottom_and_end(self):
        frames = [squat_frame(index, hip_y) for index, hip_y in enumerate(
            [100, 102, 104, 125, 155, 190, 170, 140, 112, 101]
        )]
        config = SquatAnalyzerConfig(event_smoothing_window=1, min_squat_travel_px=20)

        events = detect_squat_events(frames, config)

        self.assertEqual(events.status, "ok")
        self.assertEqual(events.bottom_frame, 5)
        self.assertIsNotNone(events.lift_start_frame)
        self.assertIsNotNone(events.lift_end_frame)
        self.assertLess(events.lift_start_frame, events.bottom_frame)
        self.assertGreater(events.lift_end_frame, events.bottom_frame)

    def test_detect_squat_events_uses_hip_and_knee_flexion_angles(self):
        frames = [
            angled_squat_frame(0, 178, 176),
            angled_squat_frame(1, 177, 175),
            angled_squat_frame(2, 174, 172),
            angled_squat_frame(3, 166, 163),
            angled_squat_frame(4, 145, 132),
            angled_squat_frame(5, 108, 92),
            angled_squat_frame(6, 125, 112),
            angled_squat_frame(7, 156, 150),
            angled_squat_frame(8, 170, 166),
            angled_squat_frame(9, 176, 174),
        ]
        config = SquatAnalyzerConfig(
            min_valid_frames=2,
            event_smoothing_window=1,
            min_flexion_angle_travel_deg=20.0,
            event_flexion_angle_drop_deg=10.0,
            event_onset_angle_drop_deg=3.0,
            event_onset_min_consecutive_frames=2,
            event_start_preroll_frames=2,
            event_extension_angle_tolerance_deg=12.0,
        )

        events = detect_squat_events(frames, config)

        self.assertEqual(events.status, "ok")
        self.assertEqual(events.details["event_detection_method"], "hip_knee_angle_flexion_extension")
        self.assertTrue(events.details["start_backtracked_from_obvious_flexion"])
        self.assertTrue(events.details["start_backtracked_from_flexion_onset"])
        self.assertEqual(events.details["flexion_onset_sequence_index"], 2)
        self.assertEqual(events.details["obvious_flexion_sequence_index"], 3)
        self.assertEqual(events.lift_start_frame, 0)
        self.assertEqual(events.bottom_frame, 5)
        self.assertEqual(events.lift_end_frame, 8)

    def test_ground_depth_uses_z_up_height_order(self):
        landmarks_ground = {
            "left_hip": {
                "valid": True,
                "visibility": 1.0,
                "presence": 1.0,
                "ground": {"Zg": 0.40},
            },
            "left_knee": {
                "valid": True,
                "visibility": 1.0,
                "presence": 1.0,
                "ground": {"Zg": 0.50},
            },
        }

        result = evaluate_ground_depth(landmarks_ground, bottom_frame_index=12)

        self.assertEqual(result.status, PASS)
        self.assertEqual(result.rule_id, "SQ_DEPTH_GROUND")
        self.assertEqual(result.evidence_frames, [12])

    def test_start_knee_lock_uses_standing_window_before_flexion(self):
        frames = [
            angled_squat_frame(0, 178, 176),
            angled_squat_frame(1, 178, 176),
            angled_squat_frame(2, 177, 175),
            angled_squat_frame(3, 176, 174),
            angled_squat_frame(4, 175, 173),
            angled_squat_frame(5, 150, 135),
        ]
        events = SquatEvents(
            status="ok",
            reason="test",
            lift_start_frame=5,
            lift_start_timestamp_ms=500,
            details={
                "lift_start_sequence_index": 5,
                "flexion_onset_sequence_index": 5,
            },
        )
        config = SquatAnalyzerConfig(
            knee_locked_angle_deg=155,
            lock_standing_window_frames=5,
            lock_check_offset_frames=0,
        )

        result = SquatStartKneesLockedRule().evaluate(frames, config, events)

        self.assertEqual(result.status, PASS)
        self.assertEqual(result.rule_id, "SQ_START_KNEES_LOCKED")
        self.assertEqual(result.details["window_end_sequence_index"], 4)
        self.assertGreaterEqual(result.details["minimum_lock_angle_deg"], 175.0)
        self.assertEqual(result.details["side_summaries"]["left"]["max_angle_deg"], 176.0)

    def test_finish_knee_lock_uses_standing_window_after_extension(self):
        frames = [
            angled_squat_frame(0, 108, 92),
            angled_squat_frame(1, 130, 115),
            angled_squat_frame(2, 150, 135),
            angled_squat_frame(3, 165, 150),
            angled_squat_frame(4, 168, 145),
            angled_squat_frame(5, 176, 174),
            angled_squat_frame(6, 177, 175),
            angled_squat_frame(7, 178, 176),
        ]
        events = SquatEvents(
            status="ok",
            reason="test",
            lift_end_frame=4,
            lift_end_timestamp_ms=400,
            details={"lift_end_sequence_index": 4},
        )
        config = SquatAnalyzerConfig(
            knee_locked_angle_deg=155,
            lock_standing_window_frames=4,
        )

        result = SquatFinishKneesLockedRule().evaluate(frames, config, events)

        self.assertEqual(result.status, PASS)
        self.assertEqual(result.rule_id, "SQ_FINISH_KNEES_LOCKED")
        self.assertEqual(result.details["window_start_sequence_index"], 4)
        self.assertEqual(result.details["window_end_sequence_index"], 7)
        self.assertGreaterEqual(result.details["minimum_lock_angle_deg"], 155.0)

    def test_knee_lock_weights_near_camera_side_and_reports_problem_side(self):
        frames = [
            two_side_angled_frame(0, 174, 139),
            two_side_angled_frame(1, 176, 142),
            two_side_angled_frame(2, 178, 145),
            two_side_angled_frame(3, 177, 143),
        ]
        events = SquatEvents(
            status="ok",
            reason="test",
            lift_start_frame=4,
            lift_start_timestamp_ms=400,
            details={
                "lift_start_sequence_index": 4,
                "flexion_onset_sequence_index": 4,
            },
        )
        config = SquatAnalyzerConfig(
            knee_locked_angle_deg=155,
            lock_standing_window_frames=4,
            lock_angle_percentile=1.0,
            lock_near_side_weight=2.0,
        )

        result = SquatStartKneesLockedRule().evaluate(frames, config, events)

        self.assertEqual(result.status, PASS)
        self.assertEqual(result.details["near_camera_side"], "left")
        self.assertEqual(result.details["side_summaries"]["left"]["side_weight"], 2.0)
        self.assertEqual(result.details["side_summaries"]["right"]["side_weight"], 1.0)
        self.assertEqual(result.details["problem_sides"], ["right"])
        self.assertIn("右侧", result.reason)
        self.assertGreaterEqual(result.details["weighted_lock_angle_deg"], 155.0)

    def test_double_bounce_rule_flags_second_drop_after_rebound(self):
        frames = [squat_frame(index, hip_y) for index, hip_y in enumerate(
            [100, 110, 145, 190, 178, 188, 160, 130, 105]
        )]
        config = SquatAnalyzerConfig(
            event_smoothing_window=1,
            min_squat_travel_px=20,
            double_bounce_rebound_px=8,
            double_bounce_second_drop_px=6,
        )
        events = detect_squat_events(frames, config)

        result = SquatDoubleBounceRule().evaluate(frames, config, events)

        self.assertEqual(result.status, FAIL)
        self.assertEqual(result.rule_id, "SQ_DOUBLE_BOUNCE")


if __name__ == "__main__":
    unittest.main()
