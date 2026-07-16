import numpy as np

from wrongway.direction import (OK, PENDING, SUSPECT, WRONG,
                                DirectionValidator, FlowLearner)
from wrongway.geometry import compute_homography

FRAME = (400, 300)


def drive(validator, track_id, points):
    status = PENDING
    for point in points:
        status = validator.update(track_id, point, FRAME)
    return status


def east_track(n=20, step=5, y=100):
    return [(10 + i * step, y) for i in range(n)]


def west_track(n=20, step=5, y=100):
    return [(390 - i * step, y) for i in range(n)]


def test_with_traffic_is_ok():
    validator = DirectionValidator("east", hysteresis_frames=3)
    assert drive(validator, 1, east_track()) == OK


def test_against_traffic_is_wrong():
    validator = DirectionValidator("east", hysteresis_frames=3)
    assert drive(validator, 1, west_track()) == WRONG
    assert 1 in validator.confirmed


def test_small_jitter_stays_pending():
    validator = DirectionValidator("east", min_displacement=12)
    jitter = [(100 + (i % 2), 100 + (i % 3)) for i in range(30)]
    assert drive(validator, 1, jitter) == PENDING


def test_hysteresis_requires_consecutive_frames():
    validator = DirectionValidator("east", hysteresis_frames=100)
    assert drive(validator, 1, west_track()) == SUSPECT
    assert 1 not in validator.confirmed


def test_u_turn_becomes_wrong():
    validator = DirectionValidator("east", hysteresis_frames=3, history=100)
    forward = east_track(10)
    back = [(55 - i * 8, 100) for i in range(15)]  # returns past the entry point
    assert drive(validator, 1, forward + back) == WRONG


def test_confirmed_is_sticky():
    validator = DirectionValidator("east", hysteresis_frames=3)
    drive(validator, 1, west_track())
    assert validator.update(1, (10, 100), FRAME) == WRONG


def test_perpendicular_motion_is_not_wrong():
    validator = DirectionValidator("east", hysteresis_frames=3)
    vertical = [(200, 10 + i * 8) for i in range(25)]
    assert drive(validator, 1, vertical) == PENDING


def test_homography_uses_world_metric():
    # 100 px == 30 m along the road; 12 "units" threshold now means meters
    H = compute_homography([[0, 0], [100, 0], [100, 50], [0, 50]],
                           [[0, 0], [30, 0], [30, 12], [0, 12]])
    validator = DirectionValidator("east", min_displacement=12.0,
                                   hysteresis_frames=2, homography=H)
    # 30 px of image motion == 9 m: below the 12 m threshold
    assert drive(validator, 1, east_track(n=7, step=5)) == PENDING
    # 95 px == 28.5 m: judged, and moving east == ok
    assert drive(validator, 2, east_track(n=20, step=5)) == OK


def test_flow_learner_flags_counter_flow():
    learner = FlowLearner(grid=(4, 4), min_cell_samples=5, calibration_samples=60)
    validator = DirectionValidator(flow_learner=learner, hysteresis_frames=3,
                                   min_displacement=10, history=100)
    # calibration traffic moving east across the frame
    for track_id in range(10, 14):
        drive(validator, track_id, east_track(n=20, step=19, y=50 + track_id))
    assert learner.ready()
    assert drive(validator, 1, west_track(n=15, step=10, y=60)) == WRONG
    assert drive(validator, 2, east_track(n=15, step=10, y=60)) == OK


def test_flow_learner_persists(tmp_path):
    cache = tmp_path / "flow.npz"
    learner = FlowLearner(grid=(2, 2), min_cell_samples=2, calibration_samples=10,
                          cache_path=cache)
    prev = (0.0, 50.0)
    for i in range(1, 15):
        point = (i * 10.0, 50.0)
        learner.observe(prev, point, FRAME)
        prev = point
    assert cache.exists()
    reloaded = FlowLearner(grid=(2, 2), min_cell_samples=2,
                           calibration_samples=10, cache_path=cache)
    assert reloaded.ready()
    flow = reloaded.flow_at((50, 50), FRAME)
    assert flow is not None and np.dot(flow, [1, 0]) > 0.9
