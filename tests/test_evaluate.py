import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import evaluate  # noqa: E402


def violation(t_s):
    return {"kind": "violation", "t_s": t_s}


GROUND_TRUTH = [
    {"id": "v1", "start_s": 0.0, "end_s": 5.0, "wrong": True},
    {"id": "v2", "start_s": 6.0, "end_s": 10.0, "wrong": False},
    {"id": "v3", "start_s": 11.0, "end_s": 15.0, "wrong": True},
]


def test_perfect_run():
    metrics = evaluate(GROUND_TRUTH, [violation(2.0), violation(12.0)])
    assert (metrics["TP"], metrics["FP"], metrics["FN"], metrics["TN"]) == (2, 0, 0, 1)
    assert metrics["accuracy"] == 1.0


def test_missed_violation_is_fn():
    metrics = evaluate(GROUND_TRUTH, [violation(2.0)])
    assert metrics["FN"] == 1
    assert metrics["recall"] == 0.5


def test_false_alarm_on_correct_vehicle():
    metrics = evaluate(GROUND_TRUTH, [violation(2.0), violation(8.0), violation(12.0)])
    assert metrics["FP"] == 1
    assert metrics["TN"] == 0


def test_spurious_violation_outside_truth_is_fp():
    metrics = evaluate(GROUND_TRUTH, [violation(2.0), violation(12.0), violation(50.0)])
    assert metrics["FP"] == 1


def test_tolerance_window():
    metrics = evaluate(GROUND_TRUTH, [violation(5.9)], tolerance=1.0)
    assert metrics["TP"] == 1  # 5.9 is within 5.0 + 1.0 of v1
