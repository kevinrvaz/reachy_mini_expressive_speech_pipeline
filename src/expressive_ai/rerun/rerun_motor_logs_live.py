"""Command-line entry point for the reusable Reachy Mini Rerun component."""

import argparse
import logging
import math
import time
from pathlib import Path

import rerun as rr

from .reachy_view import DEFAULT_ASSETS, ReachyMiniView


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        help="Optional external Reachy Mini checkout; defaults to bundled assets",
    )
    parser.add_argument(
        "--url", default="http://localhost:8000", help="Reachy daemon URL"
    )
    parser.add_argument("--hz", type=float, default=30)
    parser.add_argument("--save", type=Path, help="Save an .rrd recording")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO"
    )
    args = parser.parse_args()
    if not math.isfinite(args.hz) or args.hz <= 0:
        parser.error("--hz must be finite and positive")
    if args.no_viewer and not args.save:
        parser.error("--no-viewer requires --save")
    source = DEFAULT_ASSETS / "robot_no_collision.urdf"
    config = DEFAULT_ASSETS / "kinematics_data.json"
    if args.repo is not None:
        source = (
            args.repo
            / "src/reachy_mini/descriptions/reachy_mini/urdf/robot_no_collision.urdf"
        )
        config = args.repo / "src/reachy_mini/assets/kinematics_data.json"
    logging.basicConfig(level=logging.INFO)
    with rr.RecordingStream("reachy_mini_motors") as rec:
        if not args.no_viewer:
            rec.spawn()
        if args.save:
            rec.save(args.save)
        with ReachyMiniView(rec, urdf=source, kinematics=config) as reachy:
            rec.send_blueprint(reachy.blueprint())
            with reachy.forward_logs(getattr(logging, args.log_level)):
                logging.info(
                    "Reading measured state from %s. Ctrl+C to stop.", args.url
                )
                reachy.start(url=args.url, hz=args.hz)
                try:
                    while True:
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    reachy.stop()
                    logging.info("Stopped.")


if __name__ == "__main__":
    main()
