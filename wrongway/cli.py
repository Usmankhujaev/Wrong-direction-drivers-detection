"""Command-line entry point."""

from __future__ import annotations

import argparse

from .config import load_config
from .direction import DIRECTION_VECTORS


def main():
    parser = argparse.ArgumentParser(
        description="Real-time wrong-direction driver detection")
    parser.add_argument("--config", default=None,
                        help="Per-camera YAML config (see configs/example_camera.yaml)")
    parser.add_argument("--input", default=None,
                        help="Video file, RTSP/HTTP URL, or 'camera'; overrides config")
    parser.add_argument("--output", default="",
                        help="Save the annotated video to this .mp4 path")
    parser.add_argument("--model", default=None,
                        help="Ultralytics weights; overrides config")
    parser.add_argument("--direction", default=None, choices=DIRECTION_VECTORS,
                        help="Allowed traffic direction; overrides config")
    parser.add_argument("--no-show", action="store_true",
                        help="Headless mode (no display window)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.input:
        cfg.source = args.input
    if args.model:
        cfg.detection.model = args.model
    if args.direction:
        cfg.direction.allowed = args.direction
        cfg.direction.mode = "fixed"
    if not cfg.source:
        parser.error("No video source: pass --input or set 'source' in the config")

    from .pipeline import run  # deferred: import torch only when running
    run(cfg, output=args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
