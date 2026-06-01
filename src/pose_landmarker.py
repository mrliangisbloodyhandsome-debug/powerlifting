import json
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


SRC_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SRC_DIR / "models" / "pose_landmarker_heavy.task"

FACE_LANDMARK_INDICES = set(range(0, 11))

POSE_LANDMARK_NAMES = {
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


class PoseLandmarker:
    """Reusable MediaPipe pose landmarker wrapper for video files."""

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
        ignored_landmark_indices=None,
    ):
        self.model_path = Path(model_path)
        self.num_poses = num_poses
        self.min_pose_detection_confidence = min_pose_detection_confidence
        self.min_pose_presence_confidence = min_pose_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.output_segmentation_masks = output_segmentation_masks
        if ignored_landmark_indices is None:
            ignored_landmark_indices = FACE_LANDMARK_INDICES
        self.ignored_landmark_indices = set(ignored_landmark_indices)

    def process_video(
        self,
        input_video_path,
        output_json_path=None,
        output_video_path=None,
        draw=True,
    ):
        input_video_path = Path(input_video_path)
        output_json_path = Path(output_json_path) if output_json_path else None
        output_video_path = Path(output_video_path) if output_video_path else None

        self._validate_paths(input_video_path)
        self._ensure_parent_dir(output_json_path)
        self._ensure_parent_dir(output_video_path)

        cap = cv2.VideoCapture(str(input_video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频：{input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = self._create_video_writer(output_video_path, fps, width, height)

        all_frames_data = []

        try:
            with vision.PoseLandmarker.create_from_options(self._create_options()) as landmarker:
                frame_index = 0

                while True:
                    success, frame_bgr = cap.read()
                    if not success:
                        break

                    timestamp_ms = int(frame_index * 1000 / fps)
                    result = self.detect_frame(
                        landmarker=landmarker,
                        frame_bgr=frame_bgr,
                        timestamp_ms=timestamp_ms,
                    )

                    frame_data = self.extract_frame_landmarks(
                        result=result,
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        width=width,
                        height=height,
                        ignored_landmark_indices=self.ignored_landmark_indices,
                    )
                    all_frames_data.append(frame_data)

                    if writer is not None:
                        output_frame = frame_bgr.copy()
                        if draw and result.pose_landmarks:
                            output_frame = self.draw_pose(
                                output_frame,
                                result.pose_landmarks[0],
                                ignored_landmark_indices=self.ignored_landmark_indices,
                            )
                        writer.write(output_frame)

                    frame_index += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        if output_json_path is not None:
            self.save_landmarks(all_frames_data, output_json_path)

        return all_frames_data

    def detect_frame(self, landmarker, frame_bgr, timestamp_ms):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )
        return landmarker.detect_for_video(mp_image, timestamp_ms)

    @staticmethod
    def save_landmarks(frames_data, output_json_path):
        output_json_path = Path(output_json_path)
        PoseLandmarker._ensure_parent_dir(output_json_path)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(frames_data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def normalized_to_pixel(landmark, width, height):
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        return x, y

    @staticmethod
    def draw_pose(
        frame,
        landmarks,
        visibility_threshold=0.5,
        ignored_landmark_indices=FACE_LANDMARK_INDICES,
    ):
        height, width = frame.shape[:2]

        for idx, landmark in enumerate(landmarks):
            if idx in ignored_landmark_indices:
                continue

            x, y = PoseLandmarker.normalized_to_pixel(landmark, width, height)
            visibility = getattr(landmark, "visibility", 1.0)

            if visibility >= visibility_threshold:
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            else:
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        for start_idx, end_idx in POSE_CONNECTIONS:
            start = landmarks[start_idx]
            end = landmarks[end_idx]

            if getattr(start, "visibility", 1.0) < visibility_threshold:
                continue
            if getattr(end, "visibility", 1.0) < visibility_threshold:
                continue

            x1, y1 = PoseLandmarker.normalized_to_pixel(start, width, height)
            x2, y2 = PoseLandmarker.normalized_to_pixel(end, width, height)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        return frame

    @staticmethod
    def extract_frame_landmarks(
        result,
        frame_index,
        timestamp_ms,
        width,
        height,
        ignored_landmark_indices=FACE_LANDMARK_INDICES,
    ):
        frame_data = {
            "frame_index": frame_index,
            "timestamp_ms": timestamp_ms,
            "poses": [],
        }

        if not result.pose_landmarks:
            return frame_data

        for pose_id, pose_landmarks in enumerate(result.pose_landmarks):
            pose_data = {
                "pose_id": pose_id,
                "landmarks": {},
            }

            for idx, landmark in enumerate(pose_landmarks):
                if idx in ignored_landmark_indices:
                    continue

                name = POSE_LANDMARK_NAMES.get(idx, f"landmark_{idx}")

                pose_data["landmarks"][name] = {
                    "index": idx,
                    "x_norm": float(landmark.x),
                    "y_norm": float(landmark.y),
                    "z_norm": float(landmark.z),
                    "x_px": float(landmark.x * width),
                    "y_px": float(landmark.y * height),
                    "visibility": float(getattr(landmark, "visibility", 0.0)),
                    "presence": float(getattr(landmark, "presence", 0.0)),
                }

            frame_data["poses"].append(pose_data)

        return frame_data

    def _validate_paths(self, input_video_path):
        if not self.model_path.exists():
            raise FileNotFoundError(f"找不到模型文件：{self.model_path}")
        if not input_video_path.exists():
            raise FileNotFoundError(f"找不到输入视频：{input_video_path}")

    def _create_options(self):
        base_options = python.BaseOptions(model_asset_path=str(self.model_path))

        return vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=self.num_poses,
            min_pose_detection_confidence=self.min_pose_detection_confidence,
            min_pose_presence_confidence=self.min_pose_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=self.output_segmentation_masks,
        )

    @staticmethod
    def _create_video_writer(output_video_path, fps, width, height):
        if output_video_path is None:
            return None

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    @staticmethod
    def _ensure_parent_dir(path):
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
