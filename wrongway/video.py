"""Video I/O: sources with RTSP reconnection and pre/post violation clips."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2


class VideoSource:
    """cv2.VideoCapture wrapper. Live streams reconnect with backoff; files end."""

    def __init__(self, source, max_retries=5, retry_delay=2.0):
        self.source = 0 if source == "camera" else source
        self.is_stream = (isinstance(self.source, int)
                          or str(self.source).startswith(("rtsp://", "http://", "https://")))
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cap = self._open()

    def _open(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise IOError(f"Couldn't open video source: {self.source}")
        return cap

    @property
    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def read(self):
        """Return the next frame, or None when the source is exhausted."""
        ok, frame = self.cap.read()
        if ok:
            return frame
        if not self.is_stream:
            return None
        for attempt in range(1, self.max_retries + 1):
            print(f"Stream dropped; reconnecting ({attempt}/{self.max_retries})...")
            self.cap.release()
            time.sleep(self.retry_delay * attempt)
            try:
                self.cap = self._open()
            except IOError:
                continue
            ok, frame = self.cap.read()
            if ok:
                return frame
        return None

    def release(self):
        self.cap.release()


class ClipRecorder:
    """Ring-buffers recent frames; on trigger writes a pre+post violation clip."""

    def __init__(self, directory, fps, seconds_before=3.0, seconds_after=3.0):
        self.directory = Path(directory)
        self.fps = max(fps, 1.0)
        self.buffer = deque(maxlen=max(int(seconds_before * self.fps), 1))
        self.post_frames = max(int(seconds_after * self.fps), 1)
        self.writer = None
        self.remaining = 0
        self.path = None

    def add_frame(self, frame):
        self.buffer.append(frame)
        if self.writer is not None:
            self.writer.write(frame)
            self.remaining -= 1
            if self.remaining <= 0:
                self.writer.release()
                self.writer = None
                print(f"Violation clip saved: {self.path}")

    def trigger(self, name, frame=None):
        """Start (or extend) a clip. Returns the clip path.

        ``frame`` seeds the clip when the violation lands before any frame
        reached the ring buffer (possible with zone wrong-entries).
        """
        if self.writer is not None:  # already recording: extend the tail
            self.remaining = self.post_frames
            return str(self.path)
        if not self.buffer and frame is None:
            return ""
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"{name}.mp4"
        reference = self.buffer[-1] if self.buffer else frame
        height, width = reference.shape[:2]
        self.writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
        for buffered in self.buffer:
            self.writer.write(buffered)
        if not self.buffer and frame is not None:
            self.writer.write(frame)
        self.remaining = self.post_frames
        return str(self.path)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def enhance_low_light(frame, brightness_threshold=60.0):
    """Apply CLAHE on the L channel when the frame is dark. Cheap and local."""
    gray_mean = frame.mean()
    if gray_mean >= brightness_threshold:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l_channel), a, b)), cv2.COLOR_LAB2BGR)
