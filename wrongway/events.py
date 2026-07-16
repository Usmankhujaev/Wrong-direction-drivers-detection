"""Structured event logging: JSONL always, SQLite optionally.

Two event kinds are emitted by the pipeline:
- ``violation``: a confirmed wrong-way vehicle (once per track).
- ``track_summary``: emitted when a track leaves the scene, so total traffic
  counts and per-track verdicts are available for evaluation.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Event:
    kind: str
    t_s: float                     # seconds from the start of the video/stream
    frame: int
    camera: str
    track_id: int
    label: str = ""
    lane: int = 0
    detail: dict = field(default_factory=dict)
    snapshot: str = ""
    clip: str = ""
    wall_time: str = field(
        default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))


class EventLog:
    def __init__(self, jsonl_path, sqlite_path=None):
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._jsonl = open(self.jsonl_path, "a", encoding="utf-8")
        self._db = None
        if sqlite_path:
            Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(sqlite_path)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS events ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, t_s REAL, "
                "frame INTEGER, camera TEXT, track_id INTEGER, label TEXT, "
                "lane INTEGER, detail TEXT, snapshot TEXT, clip TEXT, "
                "wall_time TEXT)")
            self._db.commit()

    def log(self, event: Event):
        record = asdict(event)
        self._jsonl.write(json.dumps(record) + "\n")
        self._jsonl.flush()
        if self._db is not None:
            self._db.execute(
                "INSERT INTO events (kind, t_s, frame, camera, track_id, label,"
                " lane, detail, snapshot, clip, wall_time)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (event.kind, event.t_s, event.frame, event.camera,
                 event.track_id, event.label, event.lane,
                 json.dumps(event.detail), event.snapshot, event.clip,
                 event.wall_time))
            self._db.commit()

    def close(self):
        self._jsonl.close()
        if self._db is not None:
            self._db.close()
