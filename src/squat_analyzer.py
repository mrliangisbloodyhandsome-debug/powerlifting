from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


PASS = "pass"
FAIL = "fail"
UNCERTAIN = "uncertain"
NOT_APPLICABLE = "not_applicable"

AUTO_LEVEL_AUTOMATIC = "可自动判断"
AUTO_LEVEL_MULTI_VIEW = "需要多角度"


@dataclass
class LandmarkPoint:
    name: str
    x_px: float
    y_px: float
    visibility: float
    presence: float
    z_norm: Optional[float] = None

    @classmethod
    def from_landmark(cls, name: str, data: Dict[str, Any]) -> "LandmarkPoint":
        z_norm = data.get("z_norm")
        return cls(
            name=name,
            x_px=float(data["x_px"]),
            y_px=float(data["y_px"]),
            visibility=float(data.get("visibility", 0.0)),
            presence=float(data.get("presence", 0.0)),
            z_norm=None if z_norm is None else float(z_norm),
        )

    @property
    def score(self) -> float:
        return min(self.visibility, self.presence)


@dataclass
class SquatSideFeatures:
    side: str
    hip: LandmarkPoint
    knee: LandmarkPoint
    confidence: float
    shoulder: Optional[LandmarkPoint] = None
    ankle: Optional[LandmarkPoint] = None
    heel: Optional[LandmarkPoint] = None
    foot_index: Optional[LandmarkPoint] = None

    @property
    def knee_angle_deg(self) -> Optional[float]:
        if self.ankle is None:
            return None
        return angle_degrees(self.hip, self.knee, self.ankle)

    @property
    def hip_angle_deg(self) -> Optional[float]:
        if self.shoulder is None:
            return None
        return angle_degrees(self.shoulder, self.hip, self.knee)

    def foot_points(self) -> Dict[str, LandmarkPoint]:
        points: Dict[str, LandmarkPoint] = {}
        if self.ankle is not None:
            points[f"{self.side}_ankle"] = self.ankle
        if self.heel is not None:
            points[f"{self.side}_heel"] = self.heel
        if self.foot_index is not None:
            points[f"{self.side}_foot_index"] = self.foot_index
        return points


@dataclass
class SquatFrameFeatures:
    frame_index: int
    timestamp_ms: int
    sides: Dict[str, SquatSideFeatures]

    @property
    def best_side(self) -> SquatSideFeatures:
        return max(self.sides.values(), key=lambda side: side.confidence)

    @property
    def side(self) -> str:
        return self.best_side.side

    @property
    def hip(self) -> LandmarkPoint:
        return self.best_side.hip

    @property
    def knee(self) -> LandmarkPoint:
        return self.best_side.knee

    @property
    def ankle(self) -> Optional[LandmarkPoint]:
        return self.best_side.ankle

    @property
    def heel(self) -> Optional[LandmarkPoint]:
        return self.best_side.heel

    @property
    def foot_index(self) -> Optional[LandmarkPoint]:
        return self.best_side.foot_index

    @property
    def confidence(self) -> float:
        return self.best_side.confidence

    @property
    def hip_y(self) -> float:
        return self.hip.y_px

    @property
    def knee_y(self) -> float:
        return self.knee.y_px

    @property
    def depth_margin_px(self) -> float:
        return self.hip_y - self.knee_y


@dataclass
class SquatEvents:
    status: str
    reason: str
    lift_start_frame: Optional[int] = None
    lift_start_timestamp_ms: Optional[int] = None
    bottom_frame: Optional[int] = None
    bottom_timestamp_ms: Optional[int] = None
    lift_end_frame: Optional[int] = None
    lift_end_timestamp_ms: Optional[int] = None
    confidence: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleResult:
    rule_id: str
    status: str
    auto_level: str
    reason: str
    evidence_frames: List[int] = field(default_factory=list)
    confidence: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SquatAnalyzerConfig:
    pose_confidence_min: float = 0.50
    depth_margin_px: float = 0.0
    depth_margin_m: float = 0.0
    min_valid_frames: int = 5
    knee_locked_angle_deg: float = 155.0
    event_smoothing_window: int = 7
    min_squat_travel_px: float = 25.0
    event_start_drop_ratio: float = 0.15
    event_end_return_ratio: float = 0.20
    event_min_threshold_px: float = 8.0
    event_flexion_angle_drop_deg: float = 10.0
    event_onset_angle_drop_deg: float = 3.0
    event_onset_min_consecutive_frames: int = 2
    event_start_preroll_frames: int = 3
    event_extension_angle_tolerance_deg: float = 12.0
    min_flexion_angle_travel_deg: float = 20.0
    lock_standing_window_frames: int = 15
    lock_angle_percentile: float = 0.85
    lock_near_side_weight: float = 2.0
    lock_check_offset_frames: int = 3
    double_bounce_rebound_px: float = 10.0
    double_bounce_second_drop_px: float = 8.0
    foot_movement_px: float = 25.0
    foot_movement_body_scale_ratio: float = 0.08


class SquatRule:
    rule_id = ""
    auto_level = ""

    def evaluate(
        self,
        frames: Sequence[SquatFrameFeatures],
        config: SquatAnalyzerConfig,
        events: Optional[SquatEvents] = None,
    ) -> RuleResult:
        raise NotImplementedError


class SquatDepthRule(SquatRule):
    """Legacy 2D-pixel depth rule kept as a fallback and smoke-test surface."""

    rule_id = "SQ_DEPTH"
    auto_level = AUTO_LEVEL_AUTOMATIC

    def evaluate(
        self,
        frames: Sequence[SquatFrameFeatures],
        config: SquatAnalyzerConfig,
        events: Optional[SquatEvents] = None,
    ) -> RuleResult:
        if len(frames) < config.min_valid_frames:
            return RuleResult(
                rule_id=self.rule_id,
                status=UNCERTAIN,
                auto_level=self.auto_level,
                reason="可用髋点和膝点帧数不足，不能可靠判断深蹲深度",
                confidence=0.0,
                details={"valid_frame_count": len(frames)},
            )

        bottom_frame = max(frames, key=lambda frame: frame.hip_y)
        margin = bottom_frame.depth_margin_px

        details = {
            "coordinate_space": "image_pixels_y_down",
            "bottom_frame": bottom_frame.frame_index,
            "side": bottom_frame.side,
            "hip_name": bottom_frame.hip.name,
            "knee_name": bottom_frame.knee.name,
            "hip_y_px": round(bottom_frame.hip_y, 3),
            "knee_y_px": round(bottom_frame.knee_y, 3),
            "depth_margin_px": round(margin, 3),
            "required_depth_margin_px": config.depth_margin_px,
        }

        if margin > config.depth_margin_px:
            return RuleResult(
                rule_id=self.rule_id,
                status=PASS,
                auto_level=self.auto_level,
                reason="最低点时髋点低于膝点，深蹲深度可能足够",
                evidence_frames=[bottom_frame.frame_index],
                confidence=bottom_frame.confidence,
                details=details,
            )

        return RuleResult(
            rule_id=self.rule_id,
            status=FAIL,
            auto_level=self.auto_level,
            reason="最低点时髋点没有低于膝点，深蹲深度可能不足",
            evidence_frames=[bottom_frame.frame_index],
            confidence=bottom_frame.confidence,
            details=details,
        )


class SquatGroundDepthRule(SquatRule):
    """Depth rule using ground-referenced 3D coordinates from ground_pose."""

    rule_id = "SQ_DEPTH_GROUND"
    auto_level = AUTO_LEVEL_AUTOMATIC

    def __init__(
        self,
        landmarks_ground: Dict[str, Dict[str, Any]],
        bottom_frame_index: Optional[int] = None,
    ) -> None:
        self.landmarks_ground = landmarks_ground
        self.bottom_frame_index = bottom_frame_index

    def evaluate(
        self,
        frames: Sequence[SquatFrameFeatures],
        config: SquatAnalyzerConfig,
        events: Optional[SquatEvents] = None,
    ) -> RuleResult:
        candidates = []
        for side in ("left", "right"):
            hip = self.landmarks_ground.get(f"{side}_hip")
            knee = self.landmarks_ground.get(f"{side}_knee")
            if not _valid_ground_landmark(hip) or not _valid_ground_landmark(knee):
                continue
            hip_z = float(hip["ground"]["Zg"])
            knee_z = float(knee["ground"]["Zg"])
            confidence = min(
                float(hip.get("visibility", 0.0)),
                float(hip.get("presence", 0.0)),
                float(knee.get("visibility", 0.0)),
                float(knee.get("presence", 0.0)),
            )
            candidates.append((side, hip, knee, hip_z, knee_z, knee_z - hip_z, confidence))

        if not candidates:
            invalid_reasons = {
                name: item.get("invalid_reason")
                for name, item in self.landmarks_ground.items()
                if name.endswith("_hip") or name.endswith("_knee")
            }
            return RuleResult(
                rule_id=self.rule_id,
                status=UNCERTAIN,
                auto_level=self.auto_level,
                reason="最低点单帧没有可用的地面三维髋点/膝点，不能可靠判断深度",
                evidence_frames=_optional_frame_list(self.bottom_frame_index),
                confidence=0.0,
                details={
                    "coordinate_space": "ground_frame_z_up",
                    "invalid_reasons": invalid_reasons,
                },
            )

        side, hip, knee, hip_z, knee_z, margin, confidence = max(
            candidates,
            key=lambda item: item[-1],
        )
        details = {
            "coordinate_space": "ground_frame_z_up",
            "side": side,
            "hip_name": f"{side}_hip",
            "knee_name": f"{side}_knee",
            "hip_z_m": round(hip_z, 6),
            "knee_z_m": round(knee_z, 6),
            "knee_minus_hip_z_m": round(margin, 6),
            "required_knee_minus_hip_z_m": config.depth_margin_m,
        }

        if margin > config.depth_margin_m:
            return RuleResult(
                rule_id=self.rule_id,
                status=PASS,
                auto_level=self.auto_level,
                reason="最低点地面坐标中髋点低于膝点，深蹲深度可能足够",
                evidence_frames=_optional_frame_list(self.bottom_frame_index),
                confidence=confidence,
                details=details,
            )

        return RuleResult(
            rule_id=self.rule_id,
            status=FAIL,
            auto_level=self.auto_level,
            reason="最低点地面坐标中髋点没有低于膝点，深蹲深度可能不足",
            evidence_frames=_optional_frame_list(self.bottom_frame_index),
            confidence=confidence,
            details=details,
        )


class SquatStartKneesLockedRule(SquatRule):
    rule_id = "SQ_START_KNEES_LOCKED"
    auto_level = AUTO_LEVEL_AUTOMATIC

    def evaluate(
        self,
        frames: Sequence[SquatFrameFeatures],
        config: SquatAnalyzerConfig,
        events: Optional[SquatEvents] = None,
    ) -> RuleResult:
        if events is None or events.lift_start_frame is None:
            return _event_missing_result(self.rule_id, self.auto_level, "起始时间戳不足")
        window_start, window_end = start_lock_window_indices(events, len(frames), config)
        return evaluate_knee_lock_in_window(
            rule_id=self.rule_id,
            auto_level=self.auto_level,
            frames=frames,
            start_index=window_start,
            end_index=window_end,
            config=config,
            threshold_deg=config.knee_locked_angle_deg,
            pass_reason="下蹲前站定窗口内膝角接近伸直，起始锁膝可能合格",
            fail_reason="下蹲前站定窗口内膝角明显弯曲，起始锁膝可能不合格",
            window_role="pre_squat_standing_before_flexion",
        )


class SquatFinishKneesLockedRule(SquatRule):
    rule_id = "SQ_FINISH_KNEES_LOCKED"
    auto_level = AUTO_LEVEL_AUTOMATIC

    def evaluate(
        self,
        frames: Sequence[SquatFrameFeatures],
        config: SquatAnalyzerConfig,
        events: Optional[SquatEvents] = None,
    ) -> RuleResult:
        if events is None or events.lift_end_frame is None:
            return _event_missing_result(self.rule_id, self.auto_level, "完成时间戳不足")
        window_start, window_end = finish_lock_window_indices(events, len(frames), config)
        return evaluate_knee_lock_in_window(
            rule_id=self.rule_id,
            auto_level=self.auto_level,
            frames=frames,
            start_index=window_start,
            end_index=window_end,
            config=config,
            threshold_deg=config.knee_locked_angle_deg,
            pass_reason="蹲起后站定窗口内膝角接近伸直，完成锁膝可能合格",
            fail_reason="蹲起后站定窗口内膝角明显弯曲，完成锁膝可能不合格",
            window_role="post_squat_standing_after_extension",
        )


class SquatDoubleBounceRule(SquatRule):
    rule_id = "SQ_DOUBLE_BOUNCE"
    auto_level = AUTO_LEVEL_AUTOMATIC

    def evaluate(
        self,
        frames: Sequence[SquatFrameFeatures],
        config: SquatAnalyzerConfig,
        events: Optional[SquatEvents] = None,
    ) -> RuleResult:
        if events is None or events.bottom_frame is None or events.lift_end_frame is None:
            return _event_missing_result(self.rule_id, self.auto_level, "底部或完成时间戳不足")

        bottom_index = int(events.details.get("bottom_sequence_index", 0))
        end_index = int(events.details.get("lift_end_sequence_index", len(frames) - 1))
        if end_index <= bottom_index + 2:
            return RuleResult(
                rule_id=self.rule_id,
                status=UNCERTAIN,
                auto_level=self.auto_level,
                reason="最低点之后可分析帧数不足，不能可靠判断二次下沉",
                evidence_frames=[events.bottom_frame],
                confidence=events.confidence,
                details={"bottom_sequence_index": bottom_index, "lift_end_sequence_index": end_index},
            )

        hip_y = smooth_series(
            [frame.hip_y for frame in frames],
            window=config.event_smoothing_window,
        )
        bottom_y = hip_y[bottom_index]
        rebound_low_y = bottom_y
        rebound_low_index = bottom_index

        for sequence_index in range(bottom_index + 1, end_index + 1):
            current_y = hip_y[sequence_index]
            if current_y < rebound_low_y:
                rebound_low_y = current_y
                rebound_low_index = sequence_index
                continue

            rebound_px = bottom_y - rebound_low_y
            second_drop_px = current_y - rebound_low_y
            if (
                rebound_px >= config.double_bounce_rebound_px
                and second_drop_px >= config.double_bounce_second_drop_px
            ):
                return RuleResult(
                    rule_id=self.rule_id,
                    status=FAIL,
                    auto_level=self.auto_level,
                    reason="最低点附近出现上升后再次下沉的明显二次下沉",
                    evidence_frames=[
                        frames[bottom_index].frame_index,
                        frames[rebound_low_index].frame_index,
                        frames[sequence_index].frame_index,
                    ],
                    confidence=min(events.confidence, frames[sequence_index].confidence),
                    details={
                        "coordinate_space": "image_pixels_y_down",
                        "bottom_y_px": round(bottom_y, 3),
                        "rebound_low_y_px": round(rebound_low_y, 3),
                        "second_drop_y_px": round(current_y, 3),
                        "rebound_px": round(rebound_px, 3),
                        "second_drop_px": round(second_drop_px, 3),
                        "required_rebound_px": config.double_bounce_rebound_px,
                        "required_second_drop_px": config.double_bounce_second_drop_px,
                    },
                )

        return RuleResult(
            rule_id=self.rule_id,
            status=PASS,
            auto_level=self.auto_level,
            reason="最低点后未检测到超过阈值的明显二次下沉",
            evidence_frames=[events.bottom_frame, events.lift_end_frame],
            confidence=events.confidence,
            details={
                "coordinate_space": "image_pixels_y_down",
                "required_rebound_px": config.double_bounce_rebound_px,
                "required_second_drop_px": config.double_bounce_second_drop_px,
            },
        )


class SquatFootMovementRule(SquatRule):
    rule_id = "SQ_FOOT_MOVEMENT"
    auto_level = AUTO_LEVEL_MULTI_VIEW

    def evaluate(
        self,
        frames: Sequence[SquatFrameFeatures],
        config: SquatAnalyzerConfig,
        events: Optional[SquatEvents] = None,
    ) -> RuleResult:
        if events is None or events.lift_start_frame is None or events.lift_end_frame is None:
            return _event_missing_result(self.rule_id, self.auto_level, "起止时间戳不足")

        start_index = int(events.details.get("lift_start_sequence_index", 0))
        end_index = int(events.details.get("lift_end_sequence_index", len(frames) - 1))
        if end_index <= start_index:
            return _event_missing_result(self.rule_id, self.auto_level, "起止时间戳顺序异常")

        start_frame = frames[start_index]
        threshold_px = max(
            config.foot_movement_px,
            body_scale_px(start_frame) * config.foot_movement_body_scale_ratio,
        )
        baselines: Dict[str, LandmarkPoint] = {}
        for side_features in start_frame.sides.values():
            for name, point in side_features.foot_points().items():
                if point.score >= config.pose_confidence_min:
                    baselines[name] = point

        if not baselines:
            return RuleResult(
                rule_id=self.rule_id,
                status=UNCERTAIN,
                auto_level=self.auto_level,
                reason="起始帧没有可靠足部关键点，不能判断脚是否移动",
                evidence_frames=[start_frame.frame_index],
                confidence=0.0,
                details={"threshold_px": round(threshold_px, 3)},
            )

        max_name: Optional[str] = None
        max_distance = 0.0
        max_frame = start_frame.frame_index
        observations = 0
        for frame in frames[start_index : end_index + 1]:
            for side_features in frame.sides.values():
                for name, point in side_features.foot_points().items():
                    baseline = baselines.get(name)
                    if baseline is None or point.score < config.pose_confidence_min:
                        continue
                    observations += 1
                    distance = distance_px(baseline, point)
                    if distance > max_distance:
                        max_distance = distance
                        max_name = name
                        max_frame = frame.frame_index

        details = {
            "coordinate_space": "image_pixels",
            "max_moved_point": max_name,
            "max_distance_px": round(max_distance, 3),
            "threshold_px": round(threshold_px, 3),
            "observations": observations,
            "note": "单机位只能初步判断足部关键点平移，脚掌/脚跟间摇动需要人工复核。",
        }
        confidence = events.confidence if observations else 0.0

        if observations < config.min_valid_frames:
            return RuleResult(
                rule_id=self.rule_id,
                status=UNCERTAIN,
                auto_level=self.auto_level,
                reason="足部关键点可用观测太少，不能可靠判断脚是否移动",
                evidence_frames=[start_frame.frame_index, frames[end_index].frame_index],
                confidence=confidence,
                details=details,
            )

        if max_distance > threshold_px:
            return RuleResult(
                rule_id=self.rule_id,
                status=FAIL,
                auto_level=self.auto_level,
                reason="动作期间足部关键点平移超过阈值，脚步移动可能不合格",
                evidence_frames=[start_frame.frame_index, max_frame],
                confidence=confidence,
                details=details,
            )

        return RuleResult(
            rule_id=self.rule_id,
            status=PASS,
            auto_level=self.auto_level,
            reason="动作期间未检测到超过阈值的明显脚步移动",
            evidence_frames=[start_frame.frame_index, frames[end_index].frame_index],
            confidence=confidence,
            details=details,
        )


class SquatAnalyzer:
    def __init__(
        self,
        config: Optional[SquatAnalyzerConfig] = None,
        rules: Optional[Iterable[SquatRule]] = None,
    ):
        self.config = config or SquatAnalyzerConfig()
        self.rules: List[SquatRule] = list(
            rules
            or [
                SquatDepthRule(),
                SquatDoubleBounceRule(),
                SquatStartKneesLockedRule(),
                SquatFinishKneesLockedRule(),
                SquatFootMovementRule(),
            ]
        )

    def add_rule(self, rule: SquatRule) -> None:
        self.rules.append(rule)

    def analyze_pose_json(self, pose_json_path: str | Path) -> Dict[str, Any]:
        frames_data = load_pose_json(pose_json_path)
        frames = extract_squat_features(frames_data, self.config)
        events = detect_squat_events(frames, self.config)
        rule_results = [
            rule.evaluate(frames, self.config, events)
            for rule in self.rules
        ]
        return {
            "decision": make_final_decision(rule_results),
            "events": asdict(events),
            "rule_results": [asdict(result) for result in rule_results],
            "frame_summary": {
                "total_frames": len(frames_data),
                "valid_squat_frames": len(frames),
                "pose_confidence_min": self.config.pose_confidence_min,
            },
        }


def load_pose_json(pose_json_path: str | Path) -> List[Dict[str, Any]]:
    pose_json_path = Path(pose_json_path)
    with open(pose_json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_squat_features(
    frames_data: Sequence[Dict[str, Any]],
    config: SquatAnalyzerConfig,
) -> List[SquatFrameFeatures]:
    frames: List[SquatFrameFeatures] = []

    for frame_data in frames_data:
        if not frame_data.get("poses"):
            continue

        landmarks = frame_data["poses"][0].get("landmarks", {})
        frame_features = build_frame_features(
            landmarks=landmarks,
            frame_index=int(frame_data["frame_index"]),
            timestamp_ms=int(frame_data["timestamp_ms"]),
            config=config,
        )

        if frame_features is not None:
            frames.append(frame_features)

    return frames


def build_frame_features(
    landmarks: Dict[str, Any],
    frame_index: int,
    timestamp_ms: int,
    config: SquatAnalyzerConfig,
) -> Optional[SquatFrameFeatures]:
    sides: Dict[str, SquatSideFeatures] = {}
    for side in ("left", "right"):
        side_features = build_side_features(side, landmarks)
        if side_features is None:
            continue
        if side_features.confidence >= config.pose_confidence_min:
            sides[side] = side_features

    if not sides:
        return None

    return SquatFrameFeatures(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        sides=sides,
    )


def choose_best_side(
    landmarks: Dict[str, Any],
    frame_index: int,
    timestamp_ms: int,
    config: SquatAnalyzerConfig,
) -> Optional[SquatFrameFeatures]:
    return build_frame_features(landmarks, frame_index, timestamp_ms, config)


def build_side_features(
    side: str,
    landmarks: Dict[str, Any],
    frame_index: Optional[int] = None,
    timestamp_ms: Optional[int] = None,
) -> Optional[SquatSideFeatures]:
    del frame_index, timestamp_ms
    hip = get_point(landmarks, f"{side}_hip")
    knee = get_point(landmarks, f"{side}_knee")

    if hip is None or knee is None:
        return None

    confidence = min(hip.visibility, hip.presence, knee.visibility, knee.presence)

    return SquatSideFeatures(
        side=side,
        hip=hip,
        knee=knee,
        confidence=confidence,
        shoulder=get_point(landmarks, f"{side}_shoulder"),
        ankle=get_point(landmarks, f"{side}_ankle"),
        heel=get_point(landmarks, f"{side}_heel"),
        foot_index=get_point(landmarks, f"{side}_foot_index"),
    )


def get_point(landmarks: Dict[str, Any], name: str) -> Optional[LandmarkPoint]:
    data = landmarks.get(name)
    if data is None:
        return None
    return LandmarkPoint.from_landmark(name, data)


def detect_squat_events(
    frames: Sequence[SquatFrameFeatures],
    config: SquatAnalyzerConfig,
) -> SquatEvents:
    if len(frames) < config.min_valid_frames:
        return SquatEvents(
            status=UNCERTAIN,
            reason="可用姿态帧数不足，不能标注深蹲起止时间戳",
            details={"valid_frame_count": len(frames)},
        )

    angle_events = detect_squat_events_from_joint_angles(frames, config)
    if angle_events is not None:
        return angle_events

    return detect_squat_events_from_hip_y(frames, config)


def detect_squat_events_from_joint_angles(
    frames: Sequence[SquatFrameFeatures],
    config: SquatAnalyzerConfig,
) -> Optional[SquatEvents]:
    angle_rows = [frame_squat_angles(frame) for frame in frames]
    pre_count = min(max(config.min_valid_frames, len(frames) // 10), 30)
    baseline_hip_values = [
        row["hip_angle_deg"]
        for row in angle_rows[:pre_count]
        if row is not None
    ]
    baseline_knee_values = [
        row["knee_angle_deg"]
        for row in angle_rows[:pre_count]
        if row is not None
    ]
    if (
        len(baseline_hip_values) < config.min_valid_frames
        or len(baseline_knee_values) < config.min_valid_frames
    ):
        return None

    baseline_hip_angle = median(baseline_hip_values)
    baseline_knee_angle = median(baseline_knee_values)
    hip_angles = smooth_optional_series(
        [None if row is None else row["hip_angle_deg"] for row in angle_rows],
        window=config.event_smoothing_window,
    )
    knee_angles = smooth_optional_series(
        [None if row is None else row["knee_angle_deg"] for row in angle_rows],
        window=config.event_smoothing_window,
    )

    flexion_scores: List[Optional[float]] = []
    hip_drops: List[Optional[float]] = []
    knee_drops: List[Optional[float]] = []
    for hip_angle, knee_angle in zip(hip_angles, knee_angles):
        if hip_angle is None or knee_angle is None:
            flexion_scores.append(None)
            hip_drops.append(None)
            knee_drops.append(None)
            continue
        hip_drop = max(0.0, baseline_hip_angle - hip_angle)
        knee_drop = max(0.0, baseline_knee_angle - knee_angle)
        hip_drops.append(hip_drop)
        knee_drops.append(knee_drop)
        flexion_scores.append(hip_drop + knee_drop)

    valid_score_indices = [
        index
        for index, score in enumerate(flexion_scores)
        if score is not None
    ]
    if len(valid_score_indices) < config.min_valid_frames:
        return None

    obvious_start_index = None
    for index in valid_score_indices:
        if _has_started_flexion(hip_drops[index], knee_drops[index], config):
            obvious_start_index = index
            break
    if obvious_start_index is None:
        return None

    flexion_onset_index = find_flexion_onset_index(
        hip_drops=hip_drops,
        knee_drops=knee_drops,
        obvious_start_index=obvious_start_index,
        config=config,
    )
    start_index = max(0, flexion_onset_index - max(0, int(config.event_start_preroll_frames)))
    bottom_index = start_index
    max_hip_drop = 0.0
    max_knee_drop = 0.0
    max_flexion_score = 0.0
    end_index = None
    for index in range(start_index, len(frames)):
        flexion_score = float(flexion_scores[index] or 0.0)
        if flexion_score > max_flexion_score:
            max_flexion_score = flexion_score
            bottom_index = index
        max_hip_drop = max(max_hip_drop, float(hip_drops[index] or 0.0))
        max_knee_drop = max(max_knee_drop, float(knee_drops[index] or 0.0))

        hip_angle = hip_angles[index]
        knee_angle = knee_angles[index]
        if hip_angle is None or knee_angle is None:
            continue
        has_full_flexion = (
            max_hip_drop >= config.min_flexion_angle_travel_deg
            and max_knee_drop >= config.min_flexion_angle_travel_deg
        )
        if (
            has_full_flexion
            and index > bottom_index
            and hip_angle >= baseline_hip_angle - config.event_extension_angle_tolerance_deg
            and knee_angle >= baseline_knee_angle - config.event_extension_angle_tolerance_deg
        ):
            end_index = index
            break

    if (
        max_hip_drop < config.min_flexion_angle_travel_deg
        or max_knee_drop < config.min_flexion_angle_travel_deg
    ):
        return None

    if end_index is None:
        end_index = valid_score_indices[-1]

    start_frame = frames[start_index]
    bottom_frame = frames[bottom_index]
    end_frame = frames[end_index]
    confidence = min(start_frame.confidence, bottom_frame.confidence, end_frame.confidence)

    return SquatEvents(
        status="ok",
        reason="已根据从站定到持续屈髋屈膝、再到伸髋伸膝恢复标注深蹲启动、最低点和结束时间戳",
        lift_start_frame=start_frame.frame_index,
        lift_start_timestamp_ms=start_frame.timestamp_ms,
        bottom_frame=bottom_frame.frame_index,
        bottom_timestamp_ms=bottom_frame.timestamp_ms,
        lift_end_frame=end_frame.frame_index,
        lift_end_timestamp_ms=end_frame.timestamp_ms,
        confidence=confidence,
        details={
            "event_detection_method": "hip_knee_angle_flexion_extension",
            "lift_start_sequence_index": start_index,
            "flexion_onset_sequence_index": flexion_onset_index,
            "obvious_flexion_sequence_index": obvious_start_index,
            "start_backtracked_from_flexion_onset": start_index < flexion_onset_index,
            "start_backtracked_from_obvious_flexion": start_index < obvious_start_index,
            "bottom_sequence_index": bottom_index,
            "lift_end_sequence_index": end_index,
            "baseline_hip_angle_deg": round(baseline_hip_angle, 3),
            "baseline_knee_angle_deg": round(baseline_knee_angle, 3),
            "start_hip_drop_deg": round(float(hip_drops[start_index] or 0.0), 3),
            "start_knee_drop_deg": round(float(knee_drops[start_index] or 0.0), 3),
            "flexion_onset_hip_drop_deg": round(float(hip_drops[flexion_onset_index] or 0.0), 3),
            "flexion_onset_knee_drop_deg": round(float(knee_drops[flexion_onset_index] or 0.0), 3),
            "obvious_start_hip_drop_deg": round(float(hip_drops[obvious_start_index] or 0.0), 3),
            "obvious_start_knee_drop_deg": round(float(knee_drops[obvious_start_index] or 0.0), 3),
            "bottom_hip_drop_deg": round(float(hip_drops[bottom_index] or 0.0), 3),
            "bottom_knee_drop_deg": round(float(knee_drops[bottom_index] or 0.0), 3),
            "max_flexion_score_deg": round(max_flexion_score, 3),
            "required_obvious_angle_drop_deg": config.event_flexion_angle_drop_deg,
            "required_onset_angle_drop_deg": config.event_onset_angle_drop_deg,
            "required_onset_min_consecutive_frames": config.event_onset_min_consecutive_frames,
            "start_preroll_frames": config.event_start_preroll_frames,
            "required_extension_angle_tolerance_deg": config.event_extension_angle_tolerance_deg,
            "smoothing_window": config.event_smoothing_window,
        },
    )


def detect_squat_events_from_hip_y(
    frames: Sequence[SquatFrameFeatures],
    config: SquatAnalyzerConfig,
) -> SquatEvents:
    hip_y = smooth_series(
        [frame.hip_y for frame in frames],
        window=config.event_smoothing_window,
    )
    bottom_index = max(range(len(hip_y)), key=lambda index: hip_y[index])
    pre_count = min(max(config.min_valid_frames, len(frames) // 10), 30)
    baseline_y = median(hip_y[:pre_count])
    travel_px = max(hip_y) - min(hip_y)
    threshold_px = max(
        config.event_min_threshold_px,
        travel_px * config.event_start_drop_ratio,
    )

    if travel_px < config.min_squat_travel_px:
        bottom_frame = frames[bottom_index]
        return SquatEvents(
            status=UNCERTAIN,
            reason="髋点垂直位移不足，不能可靠标注一次完整深蹲",
            bottom_frame=bottom_frame.frame_index,
            bottom_timestamp_ms=bottom_frame.timestamp_ms,
            confidence=bottom_frame.confidence,
            details={
                "coordinate_space": "image_pixels_y_down",
                "bottom_sequence_index": bottom_index,
                "baseline_hip_y_px": round(baseline_y, 3),
                "hip_travel_px": round(travel_px, 3),
                "required_min_travel_px": config.min_squat_travel_px,
            },
        )

    start_threshold_y = baseline_y + threshold_px
    start_index = 0
    for index in range(bottom_index + 1):
        if hip_y[index] >= start_threshold_y:
            start_index = index
            break

    end_threshold_y = baseline_y + max(
        config.event_min_threshold_px,
        travel_px * config.event_end_return_ratio,
    )
    end_index = len(frames) - 1
    for index in range(bottom_index + 1, len(frames)):
        if hip_y[index] <= end_threshold_y:
            end_index = index
            break

    start_frame = frames[start_index]
    bottom_frame = frames[bottom_index]
    end_frame = frames[end_index]
    confidence = min(start_frame.confidence, bottom_frame.confidence, end_frame.confidence)

    return SquatEvents(
        status="ok",
        reason="已根据髋点垂直位移标注深蹲启动、最低点和结束时间戳",
        lift_start_frame=start_frame.frame_index,
        lift_start_timestamp_ms=start_frame.timestamp_ms,
        bottom_frame=bottom_frame.frame_index,
        bottom_timestamp_ms=bottom_frame.timestamp_ms,
        lift_end_frame=end_frame.frame_index,
        lift_end_timestamp_ms=end_frame.timestamp_ms,
        confidence=confidence,
        details={
            "event_detection_method": "hip_vertical_displacement_fallback",
            "coordinate_space": "image_pixels_y_down",
            "lift_start_sequence_index": start_index,
            "bottom_sequence_index": bottom_index,
            "lift_end_sequence_index": end_index,
            "baseline_hip_y_px": round(baseline_y, 3),
            "hip_travel_px": round(travel_px, 3),
            "start_threshold_y_px": round(start_threshold_y, 3),
            "end_threshold_y_px": round(end_threshold_y, 3),
            "smoothing_window": config.event_smoothing_window,
        },
    )


def start_lock_window_indices(
    events: SquatEvents,
    frame_count: int,
    config: SquatAnalyzerConfig,
) -> tuple[int, int]:
    if frame_count <= 0:
        return 0, 0
    onset_index = events.details.get("flexion_onset_sequence_index")
    if onset_index is None:
        onset_index = events.details.get("lift_start_sequence_index", 0)
    end_index = min(frame_count - 1, max(0, int(onset_index) - 1))
    window_frames = max(1, int(config.lock_standing_window_frames))
    start_index = max(0, end_index - window_frames + 1)
    return start_index, end_index


def finish_lock_window_indices(
    events: SquatEvents,
    frame_count: int,
    config: SquatAnalyzerConfig,
) -> tuple[int, int]:
    if frame_count <= 0:
        return 0, 0
    end_index = events.details.get("lift_end_sequence_index", frame_count - 1)
    start_index = min(frame_count - 1, max(0, int(end_index)))
    window_frames = max(1, int(config.lock_standing_window_frames))
    end_index = min(frame_count - 1, start_index + window_frames - 1)
    return start_index, end_index


def evaluate_knee_lock_in_window(
    rule_id: str,
    auto_level: str,
    frames: Sequence[SquatFrameFeatures],
    start_index: int,
    end_index: int,
    config: SquatAnalyzerConfig,
    threshold_deg: float,
    pass_reason: str,
    fail_reason: str,
    window_role: str,
) -> RuleResult:
    if not frames:
        return RuleResult(
            rule_id=rule_id,
            status=UNCERTAIN,
            auto_level=auto_level,
            reason="缺少可用姿态帧，不能可靠计算膝角",
            confidence=0.0,
            details={"threshold_deg": threshold_deg, "window_role": window_role},
        )

    start_index = min(len(frames) - 1, max(0, int(start_index)))
    end_index = min(len(frames) - 1, max(start_index, int(end_index)))
    side_observations: Dict[str, List[Dict[str, float]]] = {}
    for sequence_index in range(start_index, end_index + 1):
        frame = frames[sequence_index]
        for side_features in frame.sides.values():
            angle = side_features.knee_angle_deg
            if angle is None:
                continue
            z_values = [
                point.z_norm
                for point in (side_features.hip, side_features.knee, side_features.ankle)
                if point is not None and point.z_norm is not None
            ]
            z_norm = median(z_values) if z_values else None
            side_observations.setdefault(side_features.side, []).append(
                {
                    "sequence_index": float(sequence_index),
                    "frame_index": float(frame.frame_index),
                    "timestamp_ms": float(frame.timestamp_ms),
                    "angle_deg": float(angle),
                    "confidence": float(side_features.confidence),
                    "z_norm": float(z_norm) if z_norm is not None else math.nan,
                }
            )

    if not side_observations:
        return RuleResult(
            rule_id=rule_id,
            status=UNCERTAIN,
            auto_level=auto_level,
            reason="站定窗口缺少髋、膝、踝关键点，不能可靠计算膝角",
            evidence_frames=[frames[start_index].frame_index, frames[end_index].frame_index],
            confidence=0.0,
            details={
                "window_role": window_role,
                "window_start_sequence_index": start_index,
                "window_end_sequence_index": end_index,
                "window_start_frame": frames[start_index].frame_index,
                "window_end_frame": frames[end_index].frame_index,
                "required_knee_locked_angle_deg": threshold_deg,
            },
        )

    side_summaries: Dict[str, Dict[str, Any]] = {}
    for side, observations in side_observations.items():
        angles = [item["angle_deg"] for item in observations]
        lock_angle = percentile(angles, config.lock_angle_percentile)
        median_angle = median(angles)
        median_observation = min(
            observations,
            key=lambda item: abs(item["angle_deg"] - lock_angle),
        )
        z_values = [item["z_norm"] for item in observations if not math.isnan(item["z_norm"])]
        side_summaries[side] = {
            "lock_angle_deg": lock_angle,
            "median_angle_deg": median_angle,
            "min_angle_deg": min(angles),
            "max_angle_deg": max(angles),
            "observation_count": len(observations),
            "representative_frame": int(median_observation["frame_index"]),
            "representative_timestamp_ms": int(median_observation["timestamp_ms"]),
            "confidence": median([item["confidence"] for item in observations]),
            "depth_z_norm": median(z_values) if z_values else None,
        }

    near_side = choose_near_camera_side(side_summaries)
    side_weights = {
        side: (
            max(1.0, float(config.lock_near_side_weight))
            if near_side is not None and side == near_side
            else 1.0
        )
        for side in side_summaries
    }
    weighted_angle = sum(
        summary["lock_angle_deg"] * side_weights[side]
        for side, summary in side_summaries.items()
    ) / sum(side_weights.values())
    min_side, min_summary = min(
        side_summaries.items(),
        key=lambda item: item[1]["lock_angle_deg"],
    )
    min_angle = float(min_summary["lock_angle_deg"])
    problem_sides = [
        side
        for side, summary in side_summaries.items()
        if summary["lock_angle_deg"] < threshold_deg
    ]
    evidence_frames = sorted(
        {
            frames[start_index].frame_index,
            frames[end_index].frame_index,
            int(min_summary["representative_frame"]),
        }
    )
    details = {
        "window_role": window_role,
        "window_start_sequence_index": start_index,
        "window_end_sequence_index": end_index,
        "window_start_frame": frames[start_index].frame_index,
        "window_end_frame": frames[end_index].frame_index,
        "window_start_timestamp_ms": frames[start_index].timestamp_ms,
        "window_end_timestamp_ms": frames[end_index].timestamp_ms,
        "side_summaries": {
            side: {
                "lock_angle_deg": round(summary["lock_angle_deg"], 3),
                "median_angle_deg": round(summary["median_angle_deg"], 3),
                "min_angle_deg": round(summary["min_angle_deg"], 3),
                "max_angle_deg": round(summary["max_angle_deg"], 3),
                "observation_count": summary["observation_count"],
                "representative_frame": summary["representative_frame"],
                "representative_timestamp_ms": summary["representative_timestamp_ms"],
                "depth_z_norm": (
                    None
                    if summary["depth_z_norm"] is None
                    else round(summary["depth_z_norm"], 6)
                ),
                "side_weight": round(side_weights[side], 3),
            }
            for side, summary in side_summaries.items()
        },
        "near_camera_side": near_side,
        "near_camera_side_weight": round(side_weights.get(near_side, 1.0), 3) if near_side else None,
        "side_weighting_basis": "MediaPipe z_norm; smaller z_norm is treated as closer to the camera.",
        "lock_angle_percentile": config.lock_angle_percentile,
        "weighted_lock_angle_deg": round(weighted_angle, 3),
        "minimum_lock_angle_side": min_side,
        "minimum_lock_angle_deg": round(min_angle, 3),
        "problem_sides": problem_sides,
        "required_knee_locked_angle_deg": threshold_deg,
    }
    confidence = float(min_summary["confidence"])

    problem_text = format_problem_sides(problem_sides)
    near_text = format_side_name(near_side) if near_side else "无法判断"

    if weighted_angle >= threshold_deg:
        reason = pass_reason
        if problem_sides:
            reason = (
                f"{pass_reason}；{problem_text}膝角低于阈值，"
                f"但近摄像头侧（{near_text}）权重更高后综合仍达标"
            )
        else:
            reason = f"{pass_reason}；左右两侧均未低于阈值"
        return RuleResult(
            rule_id=rule_id,
            status=PASS,
            auto_level=auto_level,
            reason=reason,
            evidence_frames=evidence_frames,
            confidence=confidence,
            details=details,
        )

    reason = fail_reason
    if problem_sides:
        reason = f"{fail_reason}；{problem_text}膝角低于阈值"
    return RuleResult(
        rule_id=rule_id,
        status=FAIL,
        auto_level=auto_level,
        reason=reason,
        evidence_frames=evidence_frames,
        confidence=confidence,
        details=details,
    )


def evaluate_knee_lock_at_frame(
    rule_id: str,
    auto_level: str,
    frame: SquatFrameFeatures,
    threshold_deg: float,
    pass_reason: str,
    fail_reason: str,
) -> RuleResult:
    angles = []
    for side_features in frame.sides.values():
        angle = side_features.knee_angle_deg
        if angle is not None:
            angles.append((side_features.side, angle, side_features.confidence))

    if not angles:
        return RuleResult(
            rule_id=rule_id,
            status=UNCERTAIN,
            auto_level=auto_level,
            reason="缺少髋-膝-踝关键点，不能可靠计算膝角",
            evidence_frames=[frame.frame_index],
            confidence=0.0,
            details={"threshold_deg": threshold_deg},
        )

    min_side, min_angle, confidence = min(angles, key=lambda item: item[1])
    details = {
        "frame_index": frame.frame_index,
        "timestamp_ms": frame.timestamp_ms,
        "angles_deg": {side: round(angle, 3) for side, angle, _ in angles},
        "minimum_angle_side": min_side,
        "minimum_angle_deg": round(min_angle, 3),
        "required_knee_locked_angle_deg": threshold_deg,
    }

    if min_angle >= threshold_deg:
        return RuleResult(
            rule_id=rule_id,
            status=PASS,
            auto_level=auto_level,
            reason=pass_reason,
            evidence_frames=[frame.frame_index],
            confidence=confidence,
            details=details,
        )

    return RuleResult(
        rule_id=rule_id,
        status=FAIL,
        auto_level=auto_level,
        reason=fail_reason,
        evidence_frames=[frame.frame_index],
        confidence=confidence,
        details=details,
    )


def evaluate_ground_depth(
    landmarks_ground: Dict[str, Dict[str, Any]],
    config: Optional[SquatAnalyzerConfig] = None,
    bottom_frame_index: Optional[int] = None,
) -> RuleResult:
    return SquatGroundDepthRule(
        landmarks_ground=landmarks_ground,
        bottom_frame_index=bottom_frame_index,
    ).evaluate([], config or SquatAnalyzerConfig(), None)


def make_final_decision(rule_results: Sequence[RuleResult]) -> Dict[str, Any]:
    failed_rules = [result for result in rule_results if result.status == FAIL]
    uncertain_rules = [result for result in rule_results if result.status == UNCERTAIN]

    if failed_rules:
        return {
            "decision": "no_lift_likely",
            "need_human_review": True,
            "failed_rules": [result.rule_id for result in failed_rules],
            "uncertain_rules": [result.rule_id for result in uncertain_rules],
        }

    if uncertain_rules:
        return {
            "decision": "uncertain_need_human_review",
            "need_human_review": True,
            "failed_rules": [],
            "uncertain_rules": [result.rule_id for result in uncertain_rules],
        }

    return {
        "decision": "good_lift_likely",
        "need_human_review": False,
        "failed_rules": [],
        "uncertain_rules": [],
    }


def save_analysis(result: Dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)


def angle_degrees(
    a: LandmarkPoint,
    vertex: LandmarkPoint,
    c: LandmarkPoint,
) -> Optional[float]:
    bax = a.x_px - vertex.x_px
    bay = a.y_px - vertex.y_px
    bcx = c.x_px - vertex.x_px
    bcy = c.y_px - vertex.y_px
    norm_a = math.hypot(bax, bay)
    norm_c = math.hypot(bcx, bcy)
    if norm_a < 1e-8 or norm_c < 1e-8:
        return None
    cosine = (bax * bcx + bay * bcy) / (norm_a * norm_c)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def distance_px(a: LandmarkPoint, b: LandmarkPoint) -> float:
    return math.hypot(a.x_px - b.x_px, a.y_px - b.y_px)


def body_scale_px(frame: SquatFrameFeatures) -> float:
    distances = []
    for side_features in frame.sides.values():
        if side_features.shoulder is not None:
            distances.append(distance_px(side_features.shoulder, side_features.hip))
        if side_features.ankle is not None:
            distances.append(distance_px(side_features.hip, side_features.knee))
            distances.append(distance_px(side_features.knee, side_features.ankle))
    return max(distances) if distances else 0.0


def frame_squat_angles(frame: SquatFrameFeatures) -> Optional[Dict[str, Any]]:
    candidates = []
    for side_features in frame.sides.values():
        hip_angle = side_features.hip_angle_deg
        knee_angle = side_features.knee_angle_deg
        if hip_angle is None or knee_angle is None:
            continue
        candidates.append(
            {
                "side": side_features.side,
                "hip_angle_deg": hip_angle,
                "knee_angle_deg": knee_angle,
                "confidence": side_features.confidence,
            }
        )
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["confidence"])


def smooth_optional_series(
    values: Sequence[Optional[float]],
    window: int,
) -> List[Optional[float]]:
    if window <= 1 or len(values) <= 2:
        return [None if value is None else float(value) for value in values]
    window = max(1, int(window))
    radius = window // 2
    smoothed: List[Optional[float]] = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        local = [float(value) for value in values[start:end] if value is not None]
        smoothed.append(sum(local) / len(local) if local else None)
    return smoothed


def _has_started_flexion(
    hip_drop: Optional[float],
    knee_drop: Optional[float],
    config: SquatAnalyzerConfig,
) -> bool:
    if hip_drop is None or knee_drop is None:
        return False
    return (
        hip_drop >= config.event_flexion_angle_drop_deg
        and knee_drop >= config.event_flexion_angle_drop_deg
    )


def find_flexion_onset_index(
    hip_drops: Sequence[Optional[float]],
    knee_drops: Sequence[Optional[float]],
    obvious_start_index: int,
    config: SquatAnalyzerConfig,
) -> int:
    consecutive = max(1, int(config.event_onset_min_consecutive_frames))
    for index in range(obvious_start_index + 1):
        end = index + consecutive
        if end > obvious_start_index + 1:
            break
        if all(
            _has_onset_flexion(hip_drops[item], knee_drops[item], config)
            for item in range(index, end)
        ):
            return index
    return obvious_start_index


def _has_onset_flexion(
    hip_drop: Optional[float],
    knee_drop: Optional[float],
    config: SquatAnalyzerConfig,
) -> bool:
    if hip_drop is None or knee_drop is None:
        return False
    return (
        hip_drop >= config.event_onset_angle_drop_deg
        and knee_drop >= config.event_onset_angle_drop_deg
    )


def smooth_series(values: Sequence[float], window: int) -> List[float]:
    if window <= 1 or len(values) <= 2:
        return [float(value) for value in values]
    window = max(1, int(window))
    radius = window // 2
    smoothed: List[float] = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        smoothed.append(sum(float(value) for value in values[start:end]) / (end - start))
    return smoothed


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(value) for value in values)
    quantile = max(0.0, min(1.0, float(quantile)))
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(value) for value in values)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[middle]
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2.0


def _valid_ground_landmark(item: Optional[Dict[str, Any]]) -> bool:
    return bool(item and item.get("valid") and item.get("ground") is not None)


def _optional_frame_list(frame_index: Optional[int]) -> List[int]:
    return [] if frame_index is None else [int(frame_index)]


def choose_near_camera_side(side_summaries: Dict[str, Dict[str, Any]]) -> Optional[str]:
    candidates = [
        (side, summary["depth_z_norm"])
        for side, summary in side_summaries.items()
        if summary.get("depth_z_norm") is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[1])[0]


def format_side_name(side: Optional[str]) -> str:
    if side == "left":
        return "左侧"
    if side == "right":
        return "右侧"
    return "未知侧"


def format_problem_sides(sides: Sequence[str]) -> str:
    ordered = [side for side in ("left", "right") if side in set(sides)]
    if not ordered:
        return "没有任一侧"
    return "和".join(format_side_name(side) for side in ordered)


def _event_missing_result(rule_id: str, auto_level: str, reason: str) -> RuleResult:
    return RuleResult(
        rule_id=rule_id,
        status=UNCERTAIN,
        auto_level=auto_level,
        reason=f"{reason}，不能可靠判断该规则",
        confidence=0.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze squat rules from MediaPipe pose JSON."
    )
    parser.add_argument("pose_json", help="Path to pose JSON from pose_landmarker.py")
    parser.add_argument(
        "--output",
        help="Optional path for analysis JSON output.",
    )
    parser.add_argument(
        "--pose-confidence-min",
        type=float,
        default=0.50,
        help="Minimum visibility/presence for required landmarks.",
    )
    parser.add_argument(
        "--depth-margin-px",
        type=float,
        default=0.0,
        help="2D fallback: hip must be this many pixels lower than knee.",
    )
    parser.add_argument(
        "--knee-locked-angle-deg",
        type=float,
        default=155.0,
        help="Minimum knee angle treated as locked.",
    )
    parser.add_argument(
        "--lock-standing-window-frames",
        type=int,
        default=15,
        help="Frames used to judge knee lock from the standing window before/after the squat.",
    )
    parser.add_argument(
        "--lock-angle-percentile",
        type=float,
        default=0.85,
        help="Upper percentile knee angle used for each standing-window side.",
    )
    parser.add_argument(
        "--lock-near-side-weight",
        type=float,
        default=2.0,
        help="Weight multiplier for the side closer to the camera.",
    )
    parser.add_argument(
        "--event-flexion-angle-drop-deg",
        type=float,
        default=10.0,
        help="Confirm obvious flexion when both hip and knee angles drop this much from standing.",
    )
    parser.add_argument(
        "--event-onset-angle-drop-deg",
        type=float,
        default=3.0,
        help="Backtrack start event to sustained hip and knee angle drops at this lower threshold.",
    )
    parser.add_argument(
        "--event-onset-min-consecutive-frames",
        type=int,
        default=2,
        help="Minimum consecutive frames required for the lower-threshold flexion onset.",
    )
    parser.add_argument(
        "--event-start-preroll-frames",
        type=int,
        default=3,
        help="Export the squat start this many frames before sustained flexion onset.",
    )
    parser.add_argument(
        "--event-extension-angle-tolerance-deg",
        type=float,
        default=12.0,
        help="End event when both hip and knee angles return within this much of standing.",
    )
    parser.add_argument(
        "--min-flexion-angle-travel-deg",
        type=float,
        default=20.0,
        help="Minimum hip and knee angle travel required for angle-based event detection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyzer = SquatAnalyzer(
        config=SquatAnalyzerConfig(
            pose_confidence_min=args.pose_confidence_min,
            depth_margin_px=args.depth_margin_px,
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
        )
    )
    result = analyzer.analyze_pose_json(args.pose_json)

    if args.output:
        save_analysis(result, args.output)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
