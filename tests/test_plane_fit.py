import unittest

import numpy as np

from ground_pose.ground_plane import fit_plane_open3d, orient_plane_with_body


class PlaneFitTests(unittest.TestCase):
    def test_ransac_fits_known_z_zero_plane(self):
        xs, ys = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(-1, 1, 30))
        points = np.stack([xs.reshape(-1), ys.reshape(-1), np.zeros(xs.size)], axis=1)

        result = fit_plane_open3d(
            points,
            distance_threshold=0.001,
            num_iterations=200,
            min_points=20,
        )

        self.assertGreater(abs(result.unit_normal[2]), 0.99)
        self.assertAlmostEqual(abs(result.plane_model[3]), 0.0, places=6)
        self.assertGreater(result.inlier_ratio, 0.99)

    def test_orient_plane_with_body_flips_downward_normal(self):
        plane_model = np.array([0.0, 0.0, -1.0, 0.0])
        unit_normal = np.array([0.0, 0.0, -1.0])
        mid_hip = np.array([0.0, 0.0, 0.0])
        mid_shoulder = np.array([0.0, 0.0, 1.0])

        oriented_plane, oriented_normal, metadata = orient_plane_with_body(
            plane_model,
            unit_normal,
            mid_hip,
            mid_shoulder,
        )

        self.assertTrue(metadata["flipped"])
        self.assertGreater(oriented_normal[2], 0.99)
        self.assertGreater(oriented_plane[2], 0.99)


if __name__ == "__main__":
    unittest.main()
