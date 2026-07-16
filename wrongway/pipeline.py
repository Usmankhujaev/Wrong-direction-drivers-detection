"""Main processing loop: detect -> track -> validate -> alert.

Detection/tracking is Ultralytics YOLO11 + ByteTrack (or BoT-SORT). Validation
combines the displacement model (with optional homography calibration or
learned flow) and the paper's zone-based entry-exit rules, gated by the
``confirmation`` mode. Confirmed violations produce an event-log record, a
snapshot, a pre/post video clip, and notifier calls.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import cv2
import numpy as np

from .config import AppConfig
from .direction import (OK, PENDING, SUSPECT, WRONG, DirectionValidator,
                        FlowLearner)
from .events import Event, EventLog
from .geometry import compute_homography
from .notify import build_notifiers
from .video import ClipRecorder, VideoSource, enhance_low_light
from .zones import ZoneValidator

# COCO class ids used by the pretrained model
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PERSON_CLASS = 0

GREEN = (0, 255, 0)
RED = (0, 0, 255)
ORANGE = (0, 140, 255)
YELLOW = (0, 255, 255)
STATUS_COLORS = {WRONG: RED, SUSPECT: ORANGE, OK: GREEN, PENDING: YELLOW}

STALE_SECONDS = 3.0  # a track unseen this long is summarized and forgotten


def lane_number(y, frame_height, num_lanes):
    if num_lanes <= 1:
        return 1
    lane = int(y / max(frame_height, 1) * num_lanes) + 1
    return min(max(lane, 1), num_lanes)


def combine_status(disp_status, zone_status, mode):
    """Merge the two validators' verdicts according to the confirmation mode."""
    if zone_status is None:
        mode = "displacement"
    wrong_d = disp_status == WRONG
    wrong_z = zone_status == WRONG
    if mode == "displacement":
        confirmed = wrong_d
    elif mode == "zones":
        confirmed = wrong_z
    elif mode == "both":
        confirmed = wrong_d and wrong_z
    else:  # any
        confirmed = wrong_d or wrong_z
    if confirmed:
        return WRONG
    if disp_status == SUSPECT or wrong_d or wrong_z:
        return SUSPECT
    if OK in (disp_status, zone_status):
        return OK
    return PENDING


def _build_validators(cfg: AppConfig):
    homography = None
    min_displacement = cfg.direction.min_displacement
    if cfg.calibration is not None:
        homography = compute_homography(cfg.calibration.image_points,
                                        cfg.calibration.world_points)
        min_displacement = cfg.calibration.min_displacement_m

    flow = None
    allowed = cfg.direction.allowed
    if cfg.direction.mode == "learned":
        flow = FlowLearner(cache_path=cfg.direction.flow_cache)
        allowed = None

    direction = DirectionValidator(
        allowed_direction=allowed,
        min_displacement=min_displacement,
        history=cfg.direction.history,
        hysteresis_frames=cfg.direction.hysteresis_frames,
        homography=homography,
        flow_learner=flow,
    )
    zones = None
    if cfg.zones is not None and cfg.zones.areas:
        zones = ZoneValidator(cfg.zones.areas, cfg.zones.wrong_entries,
                              cfg.zones.wrong_transitions)
    return direction, zones


def _draw_zones(frame, zones: ZoneValidator):
    h, w = frame.shape[:2]
    for name, polygon in zones.areas.items():
        pts = (polygon * [w, h]).astype(np.int32)
        color = RED if name in zones.wrong_entries else (180, 180, 180)
        cv2.polylines(frame, [pts], True, color, 1, cv2.LINE_AA)
        cv2.putText(frame, name, (int(pts[0][0]), int(pts[0][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def run(cfg: AppConfig, output="", show=True):
    from ultralytics import YOLO  # deferred: keeps core modules torch-free

    model = YOLO(cfg.detection.model)
    direction, zones = _build_validators(cfg)
    notifiers = build_notifiers(cfg.alerts.notifiers)
    event_log = EventLog(cfg.alerts.event_log, cfg.alerts.sqlite)
    result_dir = Path(cfg.alerts.result_dir)

    source = VideoSource(cfg.source)
    fps_in = source.fps
    clips = (ClipRecorder(result_dir / "clips", fps_in,
                          cfg.alerts.clip_seconds_before,
                          cfg.alerts.clip_seconds_after)
             if cfg.alerts.save_clips else None)

    classes = list(VEHICLE_CLASSES)
    if cfg.detection.detect_persons:
        classes.append(PERSON_CLASS)

    writer = None
    frame_idx = -1
    last_results = None
    active_tracks = {}   # id -> {label, first_t, last_t, last_xy, status}
    fps_timer = cv2.getTickCount()

    def finish_track(track_id, info):
        event_log.log(Event(
            kind="track_summary", t_s=info["last_t"], frame=frame_idx,
            camera=cfg.camera, track_id=track_id, label=info["label"],
            lane=info["lane"],
            detail={"first_t": info["first_t"], "status": info["status"]}))
        direction.forget(track_id)
        if zones is not None:
            zones.forget(track_id)

    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            frame_idx += 1
            t_s = frame_idx / fps_in

            if cfg.roi:
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = cfg.roi
                frame = frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
            if cfg.enhance.low_light:
                frame = enhance_low_light(frame, cfg.enhance.brightness_threshold)
            h, w = frame.shape[:2]

            detect_now = frame_idx % cfg.detection.detect_every == 0
            if detect_now:
                last_results = model.track(
                    frame, persist=True, conf=cfg.detection.conf,
                    iou=cfg.detection.iou, classes=classes,
                    tracker=cfg.detection.tracker,
                    device=cfg.detection.device, verbose=False)[0]
            results = last_results

            vehicle_count = 0
            if (results is not None and results.boxes is not None
                    and results.boxes.id is not None):
                boxes = results.boxes.xyxy.cpu().numpy()
                ids = results.boxes.id.int().cpu().tolist()
                cls_ids = results.boxes.cls.int().cpu().tolist()

                for box, track_id, cls_id in zip(boxes, ids, cls_ids):
                    bx1, by1, bx2, by2 = box.astype(int)
                    cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2

                    if cls_id == PERSON_CLASS:
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), RED, 1)
                        cv2.putText(frame, "Person", (bx1, by1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED, 1)
                        continue

                    vehicle_count += 1
                    label = VEHICLE_CLASSES.get(cls_id, "vehicle")
                    if detect_now:
                        disp_status = direction.update(track_id, (cx, cy), (w, h))
                        zone_status = (zones.update(track_id, (cx, cy), (w, h))
                                       if zones is not None else None)
                        status = combine_status(disp_status, zone_status,
                                                cfg.confirmation)
                    else:
                        status = active_tracks.get(track_id, {}).get(
                            "status", PENDING)

                    lane = lane_number(cy, h, cfg.lanes)
                    newly_confirmed = (
                        status == WRONG
                        and active_tracks.get(track_id, {}).get("status") != WRONG)
                    info = active_tracks.setdefault(
                        track_id, {"label": label, "first_t": t_s,
                                   "status": PENDING, "lane": lane})
                    info.update(last_t=t_s, status=status, lane=lane)

                    color = STATUS_COLORS[status]
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1)
                    trace = direction.trace_of(track_id)
                    if len(trace) > 1 and direction.homography is None:
                        cv2.arrowedLine(frame, tuple(map(int, trace[0])),
                                        tuple(map(int, trace[-1])), color, 1,
                                        line_type=cv2.LINE_AA)
                    text = ("WRONG DIRECTION" if status == WRONG
                            else f"{label} #{track_id}")
                    cv2.putText(frame, text, (bx1, by1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5 if status == WRONG else 0.4, color,
                                2 if status == WRONG else 1)

                    if newly_confirmed:
                        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                        name = f"wrong_direction_{stamp}_lane{lane}_id{track_id}"
                        result_dir.mkdir(parents=True, exist_ok=True)
                        snapshot = str(result_dir / f"{name}.jpg")
                        cv2.imwrite(snapshot, frame)
                        clip_path = clips.trigger(name, frame) if clips else ""
                        event = Event(
                            kind="violation", t_s=t_s, frame=frame_idx,
                            camera=cfg.camera, track_id=track_id, label=label,
                            lane=lane, snapshot=snapshot, clip=clip_path,
                            detail={"confirmation": cfg.confirmation})
                        event_log.log(event)
                        print(f"[{t_s:7.1f}s] WRONG-WAY {label} "
                              f"track {track_id} lane {lane} -> {snapshot}")
                        for notifier in notifiers:
                            notifier.send(event.__dict__)

            # summarize and drop tracks that left the scene
            for track_id in [tid for tid, info in active_tracks.items()
                             if t_s - info["last_t"] > STALE_SECONDS]:
                finish_track(track_id, active_tracks.pop(track_id))

            if zones is not None:
                _draw_zones(frame, zones)

            now = cv2.getTickCount()
            fps = cv2.getTickFrequency() / max(now - fps_timer, 1)
            fps_timer = now
            wrong_total = len(direction.confirmed | (zones.flagged if zones else set()))
            cv2.putText(frame, f"FPS: {fps:.1f}", (3, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1)
            cv2.putText(frame,
                        f"Vehicles: {vehicle_count}  Wrong-way: {wrong_total}",
                        (3, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1)

            if clips is not None:
                clips.add_frame(frame.copy())
            if output:
                if writer is None:
                    writer = cv2.VideoWriter(
                        output, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
                writer.write(frame)
            if show:
                cv2.imshow("Wrong Direction Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    while cv2.waitKey(30) & 0xFF != ord("p"):
                        pass
    finally:
        for track_id, info in list(active_tracks.items()):
            finish_track(track_id, info)
        source.release()
        if writer is not None:
            writer.release()
        if clips is not None:
            clips.close()
        event_log.close()
        cv2.destroyAllWindows()

    total_wrong = len(direction.confirmed | (zones.flagged if zones else set()))
    print(f"Done. {total_wrong} wrong-way vehicle(s); "
          f"events logged to {cfg.alerts.event_log}")
