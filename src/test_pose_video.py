from pathlib import Path

from pose_landmarker import DEFAULT_MODEL_PATH, PoseLandmarker


REPO_ROOT = Path(__file__).resolve().parents[1]

INPUT_VIDEO_PATH = REPO_ROOT / "data" / "your_squat_video.mp4"
OUTPUT_JSON_PATH = REPO_ROOT / "outputs" / "pose_json" / "squat_pose.json"
OUTPUT_VIDEO_PATH = REPO_ROOT / "outputs" / "pose_videos" / "squat_pose.mp4"


def run_pose_landmarker_on_video():
    pose_landmarker = PoseLandmarker(model_path=DEFAULT_MODEL_PATH)
    pose_landmarker.process_video(
        input_video_path=INPUT_VIDEO_PATH,
        output_json_path=OUTPUT_JSON_PATH,
        output_video_path=OUTPUT_VIDEO_PATH,
        draw=True,
    )

    print("处理完成")
    print(f"姿态 JSON 输出到：{OUTPUT_JSON_PATH}")
    print(f"可视化视频输出到：{OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    run_pose_landmarker_on_video()
