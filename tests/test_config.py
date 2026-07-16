from pathlib import Path

import pytest

from wrongway.config import load_config

EXAMPLE = Path(__file__).resolve().parent.parent / "configs" / "example_camera.yaml"


def test_defaults_without_file():
    cfg = load_config()
    assert cfg.direction.allowed == "west"
    assert cfg.confirmation == "any"
    assert cfg.detection.model == "yolo11n.pt"
    assert cfg.zones is None


def test_example_config_loads():
    cfg = load_config(EXAMPLE)
    assert cfg.camera == "cam01"
    assert cfg.roi == [0.109, 0.0608, 0.78, 1.0]
    assert cfg.direction.hysteresis_frames == 10
    assert cfg.alerts.save_clips is True


def test_overrides():
    cfg = load_config(EXAMPLE, overrides={"source": "video.mp4", "lanes": 4})
    assert cfg.source == "video.mp4"
    assert cfg.lanes == 4


def test_zones_confirmation_requires_zones(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("confirmation: both\n")
    with pytest.raises(ValueError, match="zones"):
        load_config(bad)


def test_learned_mode_rejects_calibration(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "direction:\n  mode: learned\n"
        "calibration:\n"
        "  image_points: [[0, 0], [1, 0], [1, 1], [0, 1]]\n"
        "  world_points: [[0, 0], [2, 0], [2, 2], [0, 2]]\n")
    with pytest.raises(ValueError, match="pixel space"):
        load_config(bad)


def test_bad_roi_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("roi: [0.9, 0.0, 0.1, 1.0]\n")
    with pytest.raises(ValueError, match="roi"):
        load_config(bad)
