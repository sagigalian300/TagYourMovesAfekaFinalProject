"""
Runs only the frame-list repair on a frames JSON file, without starting the server.

It uses the server's own parsing and response code (main.py) and calls the same repair, so the
output is exactly what POST /repair-ball would return for that file.

Usage:
    python src/repairers/tests/repair_only.py frames.json [width height] [--max-gap N]
Writes <name>.repaired.json next to the input and prints a summary.
"""
import os
import sys
import json
import argparse
from typing import Optional, Tuple

# main.py lives in the service root and the repair code imports its modules by bare name
# (e.g. `from Ball import Ball` inside Frame.py), so all three folders go on the import path.
# They are found from this file's location, so the script works from any folder.
REPAIRERS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVICE_DIR = os.path.dirname(os.path.dirname(REPAIRERS_DIR))
sys.path[:0] = [SERVICE_DIR, os.path.join(REPAIRERS_DIR, "contracts"), os.path.join(REPAIRERS_DIR, "offline")]

from main import parse_frames, build_response
from pipeline import repair_frames


def repair_file(path: str, frame_size: Optional[Tuple[int, int]] = None, max_gap_frames: int = 45) -> str:
    """
    Repairs the missing balls in a frames JSON file and writes the result next to it.

    Args:
        path (str): Path to a JSON file holding a list of frames, as the service receives it.
        frame_size (Optional[Tuple[int, int]], optional): (width, height) of the video, used to
            keep estimates inside the frame. Defaults to None.
        max_gap_frames (int, optional): Longest gap that is filled end to end. Defaults to 45.

    Returns:
        str: Path of the written <name>.repaired.json file.

    Raises:
        Exception: If the file is not a valid frame list (the same checks as the service, which
            answers 400 for them).
    """
    with open(path, encoding="utf-8") as fh:
        raw, frames = parse_frames(fh.read())

    # Exactly what the endpoint does, minus HTTP: repair in time order, then write back only
    # the balls that were filled, keeping everything else as it was in the file
    received = [f.ball for f in frames]
    repair_frames(sorted(frames, key=lambda f: f.frame_number), frame_size, max_gap_frames)
    repaired = build_response(raw, frames, received)

    out_path = os.path.splitext(path)[0] + ".repaired.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(repaired, fh, indent=2)

    missing = sum(1 for b in received if b is None)
    filled = sum(1 for f, before in zip(frames, received) if before is None and f.ball is not None)
    rim_frames = sum(1 for f in frames if f.rims)
    print(f"{len(frames)} frames, {missing} had no ball, {filled} filled, {missing - filled} left empty")
    # Camera motion needs rims in at least two frames; say so, since it changes the result
    print(f"rims in {rim_frames} frames (camera motion {'used' if rim_frames >= 2 else 'not used'})")
    print(f"written to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repair the missing balls in a frames JSON file, without the server.")
    parser.add_argument("frames", help="frames JSON file")
    parser.add_argument("width", type=int, nargs="?", help="video width in pixels (optional)")
    parser.add_argument("height", type=int, nargs="?", help="video height in pixels (optional)")
    parser.add_argument("--max-gap", type=int, default=45, help="longest gap filled end to end (default 45)")
    args = parser.parse_args()

    size = (args.width, args.height) if args.width and args.height else None
    # Report a bad file the way the service would (one line saying what's wrong) instead of a
    # traceback, since a malformed input is the user's to fix, not a bug in the script
    try:
        repair_file(args.frames, size, args.max_gap)
    except Exception as e:
        sys.exit(f"Invalid frames file: {e}")
