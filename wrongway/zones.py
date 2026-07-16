"""Zone-based entry-exit validation, as described in the paper.

Named areas (polygons in relative coordinates) are placed on the frame. A
track's *entry zone* is the first area its centroid is observed in. Violations
are declared by two configurable rules:

- ``wrong_entries``: a vehicle whose entry zone is in this list is wrong-way
  the moment it appears (e.g., entering the frame through exit area C).
- ``wrong_transitions``: (entry, current) zone pairs that are violations
  (e.g., entering from B and reaching C).
"""

from __future__ import annotations

import numpy as np

from .geometry import point_in_polygon

PENDING = "pending"
OK = "ok"
WRONG = "wrong"


class ZoneValidator:
    def __init__(self, areas, wrong_entries=(), wrong_transitions=()):
        """``areas``: {name: [(rx, ry), ...]} polygons in relative [0,1] coords."""
        self.areas = {name: np.asarray(poly, float) for name, poly in areas.items()}
        self.wrong_entries = set(wrong_entries)
        self.wrong_transitions = {tuple(t) for t in wrong_transitions}
        self.entry_zone = {}
        self.current_zone = {}
        self.flagged = set()

    def zone_at(self, centroid, frame_size):
        w, h = frame_size
        rel = (centroid[0] / max(w, 1), centroid[1] / max(h, 1))
        for name, polygon in self.areas.items():
            if point_in_polygon(rel, polygon):
                return name
        return None

    def update(self, track_id, centroid, frame_size):
        zone = self.zone_at(centroid, frame_size)
        if zone is not None:
            if track_id not in self.entry_zone:
                self.entry_zone[track_id] = zone
                if zone in self.wrong_entries:
                    self.flagged.add(track_id)
            self.current_zone[track_id] = zone

        if track_id in self.flagged:
            return WRONG

        entry = self.entry_zone.get(track_id)
        current = self.current_zone.get(track_id)
        if entry is None:
            return PENDING
        if current != entry and (entry, current) in self.wrong_transitions:
            self.flagged.add(track_id)
            return WRONG
        return OK if current != entry else PENDING

    def forget(self, track_id):
        self.entry_zone.pop(track_id, None)
        self.current_zone.pop(track_id, None)
