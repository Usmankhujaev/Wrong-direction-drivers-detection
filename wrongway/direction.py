"""Direction validation: displacement model, learned flow, and hysteresis.

The displacement model is the paper's entry-exit direction estimation: the
vector from a track's entry point to its current position, projected onto the
allowed travel direction. Optionally the projection happens in world (road)
coordinates through a homography, making the threshold camera-independent.

The learned-flow model removes the fixed direction entirely: during a
calibration phase it accumulates per-cell motion vectors on a grid, then flags
tracks that move against the local dominant flow.

Both feed a hysteresis stage: a violation is confirmed only after the raw
"wrong" verdict persists for N consecutive observations, which suppresses
single-frame false positives (e.g., from nighttime illumination).
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from .geometry import apply_homography

# Allowed traffic direction -> unit vector (x grows right, y grows down)
DIRECTION_VECTORS = {
    "east": (1.0, 0.0),
    "west": (-1.0, 0.0),
    "north": (0.0, -1.0),
    "south": (0.0, 1.0),
}

# Track statuses, from least to most severe
PENDING = "pending"      # not enough motion (or still calibrating) to judge
OK = "ok"                # moving with traffic
SUSPECT = "suspect"      # raw wrong verdict, hysteresis not yet satisfied
WRONG = "wrong"          # confirmed violation


class FlowLearner:
    """Learn the dominant motion direction per grid cell from observed tracks."""

    def __init__(self, grid=(8, 8), min_cell_samples=20, calibration_samples=2000,
                 cache_path=None):
        self.grid = grid
        self.min_cell_samples = min_cell_samples
        self.calibration_samples = calibration_samples
        self.cache_path = Path(cache_path) if cache_path else None
        self.sums = np.zeros((grid[1], grid[0], 2), dtype=float)
        self.counts = np.zeros((grid[1], grid[0]), dtype=int)
        if self.cache_path and self.cache_path.exists():
            data = np.load(self.cache_path)
            if data["sums"].shape == self.sums.shape:
                self.sums = data["sums"]
                self.counts = data["counts"]

    def _cell(self, point, frame_size):
        w, h = frame_size
        col = min(int(point[0] / max(w, 1) * self.grid[0]), self.grid[0] - 1)
        row = min(int(point[1] / max(h, 1) * self.grid[1]), self.grid[1] - 1)
        return max(row, 0), max(col, 0)

    def observe(self, prev_point, point, frame_size):
        """Record one per-frame motion step (only while calibrating)."""
        if self.ready():
            return
        step = np.asarray(point, float) - np.asarray(prev_point, float)
        if np.hypot(*step) < 0.5:  # ignore stationary jitter
            return
        row, col = self._cell(point, frame_size)
        self.sums[row, col] += step
        self.counts[row, col] += 1
        if self.ready() and self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.cache_path, sums=self.sums, counts=self.counts)

    def ready(self):
        return int(self.counts.sum()) >= self.calibration_samples

    def flow_at(self, point, frame_size):
        """Unit flow vector at a point, or None if the cell is unlearned."""
        row, col = self._cell(point, frame_size)
        if self.counts[row, col] < self.min_cell_samples:
            return None
        vec = self.sums[row, col] / self.counts[row, col]
        norm = np.hypot(*vec)
        return vec / norm if norm > 1e-6 else None


class DirectionValidator:
    """Per-track direction verdicts with hysteresis confirmation."""

    def __init__(self, allowed_direction=None, min_displacement=12.0, history=30,
                 hysteresis_frames=10, homography=None, flow_learner=None,
                 opposition_cos=-0.5):
        if allowed_direction is None and flow_learner is None:
            raise ValueError("Provide allowed_direction or a FlowLearner")
        if flow_learner is not None and homography is not None:
            raise ValueError("Learned-flow mode operates in pixel space; "
                             "it cannot be combined with a homography")
        self.allowed = (np.asarray(DIRECTION_VECTORS[allowed_direction], float)
                        if allowed_direction else None)
        self.min_displacement = min_displacement
        self.hysteresis_frames = hysteresis_frames
        self.homography = homography
        self.flow = flow_learner
        self.opposition_cos = opposition_cos
        self.traces = defaultdict(lambda: deque(maxlen=history))
        self.wrong_streak = defaultdict(int)
        self.confirmed = set()

    def update(self, track_id, centroid, frame_size=None):
        """Add a centroid observation and return the track's status."""
        point = (apply_homography(self.homography, centroid)
                 if self.homography is not None else np.asarray(centroid, float))
        trace = self.traces[track_id]
        if self.flow is not None and len(trace) > 0:
            self.flow.observe(trace[-1], point, frame_size)
        trace.append(point)

        if track_id in self.confirmed:
            return WRONG

        raw = self._raw_verdict(track_id, frame_size)
        if raw == WRONG:
            self.wrong_streak[track_id] += 1
            if self.wrong_streak[track_id] >= self.hysteresis_frames:
                self.confirmed.add(track_id)
                return WRONG
            return SUSPECT
        self.wrong_streak[track_id] = 0
        return raw

    def _raw_verdict(self, track_id, frame_size):
        trace = self.traces[track_id]
        if len(trace) < 3:
            return PENDING
        displacement = trace[-1] - trace[0]
        magnitude = float(np.hypot(*displacement))
        if magnitude < self.min_displacement:
            return PENDING

        if self.flow is not None:
            if not self.flow.ready():
                return PENDING
            flow = self.flow.flow_at(trace[-1], frame_size)
            if flow is None:
                return PENDING
            cos = float(np.dot(displacement / magnitude, flow))
            return WRONG if cos < self.opposition_cos else OK

        travel = float(np.dot(displacement, self.allowed))
        if abs(travel) < self.min_displacement:
            return PENDING
        return WRONG if travel < 0 else OK

    def trace_of(self, track_id):
        return self.traces[track_id]

    def forget(self, track_id):
        self.traces.pop(track_id, None)
        self.wrong_streak.pop(track_id, None)
