import unittest

import numpy as np

from ground_pose.ground_plane import (
    build_ground_coordinate_frame,
    transform_landmarks_to_ground_frame,
)


class CoordinateTransformTests(unittest.TestCase):
    def test_ground_z_ordering_is_preserved(self):
        point_map = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 1.0]],
            ],
            dtype=np.float32,
        )
        landmarks = {
            "low": {"index": 0, "x_px": 0.0, "y_px": 0.0, "visibility": 1.0, "presence": 1.0},
            "mid": {"index": 1, "x_px": 1.0, "y_px": 0.0, "visibility": 1.0, "presence": 1.0},
            "high": {"index": 2, "x_px": 2.0, "y_px": 0.0, "visibility": 1.0, "presence": 1.0},
        }
        frame = {
            "P0": np.array([0.0, 0.0, 0.0]),
            "ex": np.array([1.0, 0.0, 0.0]),
            "ey": np.array([0.0, 1.0, 0.0]),
            "ez": np.array([0.0, 0.0, 1.0]),
        }

        transformed = transform_landmarks_to_ground_frame(
            landmarks,
            point_map,
            valid_mask=np.ones((1, 3), dtype=bool),
            frame=frame,
            image_shape=(1, 3),
        )

        self.assertLess(transformed["low"]["ground"]["Zg"], transformed["mid"]["ground"]["Zg"])
        self.assertLess(transformed["mid"]["ground"]["Zg"], transformed["high"]["ground"]["Zg"])

    def test_frame_uses_projected_hip_axis(self):
        plane_model = np.array([0.0, 0.0, 1.0, 0.0])
        inliers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        landmarks = {
            "left_hip": np.array([0.0, 0.0, 1.0]),
            "right_hip": np.array([2.0, 0.0, 3.0]),
        }

        frame = build_ground_coordinate_frame(plane_model, inliers, landmarks)

        np.testing.assert_allclose(frame["ex"], np.array([1.0, 0.0, 0.0]), atol=1e-6)
        np.testing.assert_allclose(frame["ez"], np.array([0.0, 0.0, 1.0]), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
