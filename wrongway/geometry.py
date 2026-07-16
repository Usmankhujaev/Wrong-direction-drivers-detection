"""Pure-numpy geometry helpers: homography calibration and polygon tests."""

from __future__ import annotations

import numpy as np


def compute_homography(image_points, world_points):
    """Estimate the 3x3 homography mapping image points to world (road) points.

    Uses the direct linear transform with SVD. Needs at least 4 point pairs;
    world points are typically measured in meters on the road plane.
    """
    src = np.asarray(image_points, dtype=float)
    dst = np.asarray(world_points, dtype=float)
    if src.shape != dst.shape or src.shape[0] < 4 or src.shape[1] != 2:
        raise ValueError("Need at least 4 corresponding 2D point pairs")

    rows = []
    for (x, y), (X, Y) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
        rows.append([0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    H = vt[-1].reshape(3, 3)
    if abs(H[2, 2]) < 1e-12:
        raise ValueError("Degenerate point configuration")
    return H / H[2, 2]


def apply_homography(H, points):
    """Map an (N, 2) array (or a single (x, y) pair) through homography H."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    homogeneous = np.hstack([pts, np.ones((pts.shape[0], 1))])
    mapped = homogeneous @ H.T
    mapped = mapped[:, :2] / mapped[:, 2:3]
    return mapped[0] if np.asarray(points).ndim == 1 else mapped


def point_in_polygon(point, polygon):
    """Ray-casting point-in-polygon test. Polygon is a sequence of (x, y)."""
    x, y = float(point[0]), float(point[1])
    pts = np.asarray(polygon, dtype=float)
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside
