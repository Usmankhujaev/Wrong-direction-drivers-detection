"""Per-camera YAML configuration. See configs/example_camera.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DirectionConfig:
    mode: str = "fixed"              # "fixed" or "learned"
    allowed: str = "west"            # used in fixed mode
    min_displacement: float = 12.0   # pixels (or meters when calibrated)
    hysteresis_frames: int = 10
    history: int = 30
    flow_cache: Optional[str] = None  # persist learned flow across restarts


@dataclass
class CalibrationConfig:
    image_points: list = field(default_factory=list)   # [[px, py], ...] pixels
    world_points: list = field(default_factory=list)   # [[X, Y], ...] meters
    min_displacement_m: float = 2.0


@dataclass
class ZonesConfig:
    areas: dict = field(default_factory=dict)          # name -> [[rx, ry], ...]
    wrong_entries: list = field(default_factory=list)
    wrong_transitions: list = field(default_factory=list)


@dataclass
class DetectionConfig:
    model: str = "yolo11n.pt"
    conf: float = 0.3
    iou: float = 0.4
    tracker: str = "bytetrack.yaml"  # or "botsort.yaml" (ReID, better occlusions)
    detect_every: int = 1            # run the detector every Nth frame
    device: Optional[str] = None     # "0", "mps", "cpu"; auto if None
    detect_persons: bool = False


@dataclass
class AlertConfig:
    result_dir: str = "result"
    event_log: str = "result/events.jsonl"
    sqlite: Optional[str] = None
    save_clips: bool = True
    clip_seconds_before: float = 3.0
    clip_seconds_after: float = 3.0
    notifiers: list = field(default_factory=list)      # [{"kind": "webhook", ...}]


@dataclass
class EnhanceConfig:
    low_light: bool = False
    brightness_threshold: float = 60.0


@dataclass
class AppConfig:
    camera: str = "camera"
    source: str = ""
    roi: Optional[list] = None       # [x1, y1, x2, y2] relative
    lanes: int = 3
    confirmation: str = "any"        # displacement | zones | any | both
    direction: DirectionConfig = field(default_factory=DirectionConfig)
    calibration: Optional[CalibrationConfig] = None
    zones: Optional[ZonesConfig] = None
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    enhance: EnhanceConfig = field(default_factory=EnhanceConfig)

    def validate(self):
        if self.confirmation not in ("displacement", "zones", "any", "both"):
            raise ValueError(f"Bad confirmation mode: {self.confirmation!r}")
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError(f"Bad roi {self.roi}: need relative "
                                 "[x1, y1, x2, y2] with x1<x2, y1<y2 in [0,1]")
        if self.confirmation in ("zones", "both") and self.zones is None:
            raise ValueError(f"confirmation={self.confirmation!r} requires a "
                             "zones section in the config")
        if self.direction.mode not in ("fixed", "learned"):
            raise ValueError(f"Bad direction mode: {self.direction.mode!r}")
        if self.direction.mode == "learned" and self.calibration is not None:
            raise ValueError("Learned flow works in pixel space and cannot be "
                             "combined with a calibration section")
        return self


def _build(cls, data):
    if data is None:
        return None
    return cls(**data)


def load_config(path=None, overrides=None):
    """Load an AppConfig from YAML; ``overrides`` are top-level replacements."""
    data = {}
    if path:
        data = yaml.safe_load(Path(path).read_text()) or {}
    config = AppConfig(
        camera=data.get("camera", "camera"),
        source=data.get("source", ""),
        roi=data.get("roi"),
        lanes=data.get("lanes", 3),
        confirmation=data.get("confirmation", "any"),
        direction=_build(DirectionConfig, data.get("direction")) or DirectionConfig(),
        calibration=_build(CalibrationConfig, data.get("calibration")),
        zones=_build(ZonesConfig, data.get("zones")),
        detection=_build(DetectionConfig, data.get("detection")) or DetectionConfig(),
        alerts=_build(AlertConfig, data.get("alerts")) or AlertConfig(),
        enhance=_build(EnhanceConfig, data.get("enhance")) or EnhanceConfig(),
    )
    for key, value in (overrides or {}).items():
        if value is not None:
            setattr(config, key, value)
    return config.validate()
