import json
import sqlite3

from wrongway.events import Event, EventLog


def test_jsonl_logging(tmp_path):
    log_path = tmp_path / "events.jsonl"
    log = EventLog(log_path)
    log.log(Event(kind="violation", t_s=1.5, frame=45, camera="cam01",
                  track_id=7, label="car", lane=2, snapshot="snap.jpg"))
    log.log(Event(kind="track_summary", t_s=3.0, frame=90, camera="cam01",
                  track_id=7, detail={"status": "wrong"}))
    log.close()

    lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["kind"] == "violation"
    assert lines[0]["track_id"] == 7
    assert lines[1]["detail"]["status"] == "wrong"


def test_sqlite_logging(tmp_path):
    db_path = tmp_path / "events.db"
    log = EventLog(tmp_path / "events.jsonl", sqlite_path=db_path)
    log.log(Event(kind="violation", t_s=2.0, frame=60, camera="cam01",
                  track_id=3, label="truck", lane=1))
    log.close()

    rows = sqlite3.connect(db_path).execute(
        "SELECT kind, track_id, label, lane FROM events").fetchall()
    assert rows == [("violation", 3, "truck", 1)]
