import numpy as np
import pytest

from wrongway.geometry import apply_homography, compute_homography, point_in_polygon


def test_homography_recovers_affine_map():
    image = [[0, 0], [100, 0], [100, 50], [0, 50], [50, 25]]
    world = [[0, 0], [30, 0], [30, 12], [0, 12], [15, 6]]  # scale x0.3 / x0.24
    H = compute_homography(image, world)
    mapped = apply_homography(H, [50, 0])
    assert np.allclose(mapped, [15, 0], atol=1e-6)


def test_homography_single_point_shape():
    H = compute_homography([[0, 0], [1, 0], [1, 1], [0, 1]],
                           [[0, 0], [2, 0], [2, 2], [0, 2]])
    assert apply_homography(H, [0.5, 0.5]).shape == (2,)
    assert apply_homography(H, [[0.5, 0.5], [1, 1]]).shape == (2, 2)


def test_homography_needs_four_points():
    with pytest.raises(ValueError):
        compute_homography([[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 0], [1, 1]])


def test_point_in_polygon_square():
    square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert point_in_polygon((0.5, 0.5), square)
    assert not point_in_polygon((1.5, 0.5), square)
    assert not point_in_polygon((-0.1, 0.5), square)


def test_point_in_polygon_concave():
    l_shape = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    assert point_in_polygon((0.5, 1.5), l_shape)
    assert not point_in_polygon((1.5, 1.5), l_shape)
