"""
Planar Homography Projector: Maps 2D Image Pixels (u, v) to Plant Floor Real-World Coordinates (X, Y in meters).
"""
from typing import Optional, Tuple
import numpy as np


class PlanarHomography:
    def __init__(self, matrix: Optional[np.ndarray] = None):
        """
        matrix: 3x3 transformation matrix mapping [u, v, 1]^T to [X, Y, 1]^T
        """
        if matrix is not None:
            self.H = np.array(matrix, dtype=np.float64)
        else:
            self.H = np.eye(3, dtype=np.float64)

    @classmethod
    def from_calibration_points(cls, image_points: np.ndarray, world_points: np.ndarray):
        """
        Computes Homography matrix H using cv2.findHomography on at least 4 corresponding points.
        """
        import cv2
        H, status = cv2.findHomography(image_points, world_points, cv2.RANSAC, 5.0)
        return cls(H)

    def image_to_world(self, u: float, v: float) -> Tuple[float, float]:
        """
        Transforms bottom-center of bounding box (footprint) to floor metric coordinates (X, Y).
        """
        p_img = np.array([u, v, 1.0], dtype=np.float64).reshape(3, 1)
        p_world = np.dot(self.H, p_img)

        # Normalize by scale factor w
        w = p_world[2, 0]
        if abs(w) < 1e-7:
            return (0.0, 0.0)

        X = p_world[0, 0] / w
        Y = p_world[1, 0] / w
        return (float(X), float(Y))
