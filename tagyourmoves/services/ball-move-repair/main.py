"""
Main entry point for the ball-move-repair microservice.

Runs a FastAPI server that receives the detected frames of a video (and optionally the video
itself), estimates the ball's location in frames where it was obscured or not detected, and
returns the repaired frames.
"""

import os
import sys
import json
import inspect
import tempfile
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool

# The repair code and its Frame/Ball/Player/Rim classes import each other by bare module
# name (e.g. `from Ball import Ball`), so their folder has to be on the import path
OFFLINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "repairers", "offline")
sys.path.insert(0, OFFLINE_DIR)

from Ball import Ball  # noqa: E402
from Frame import Frame  # noqa: E402
from Player import Player  # noqa: E402
from Rim import Rim  # noqa: E402
from pipeline import repair_frames, render_video, video_frame_size  # noqa: E402

DEBUG_DIR = "mock/debug_outputs"
UPLOAD_CHUNK_SIZE = 1024 * 1024

app = FastAPI(title="Ball Move Repair Microservice")


def _from_dict(cls, data: dict):
    """Builds a contract object from its JSON dict.

    Keys matching the constructor are passed to it; any others (e.g. Player.has_ball, which
    the constructor always sets to False) are set as attributes afterwards, so a Frame that
    went through other services keeps all of its fields.

    Args:
        cls (type): Ball, Player, Rim or Frame.
        data (dict): The object's fields.

    Returns:
        object: An instance of cls.
    """
    params = inspect.signature(cls.__init__).parameters
    obj = cls(**{k: v for k, v in data.items() if k in params})
    for key, value in data.items():
        if key not in params:
            setattr(obj, key, value)
    return obj


def parse_frames(frames_data: str) -> List[Frame]:
    """Parses and validates the frames sent with a request.

    Args:
        frames_data (str): JSON list of Frame objects. The ball is an object or null;
            players and rims are lists (may be omitted).

    Returns:
        List[Frame]: The frames sorted by frame_number.

    Raises:
        ValueError: If the JSON is malformed, empty, or the frame numbers are not
            consecutive. The estimator uses the position in the list as time, so every
            frame must be present, including those without a ball.
    """
    raw = json.loads(frames_data)
    if not isinstance(raw, list) or not raw:
        raise ValueError("frames_data must be a non-empty JSON list of frames")

    frames = []
    for d in raw:
        frames.append(_from_dict(Frame, {
            **d,
            "ball": _from_dict(Ball, d["ball"]) if d.get("ball") else None,
            "players": [_from_dict(Player, p) for p in d.get("players") or []],
            "rims": [_from_dict(Rim, r) for r in d.get("rims") or []],
        }))

    frames.sort(key=lambda f: f.frame_number)
    numbers = [f.frame_number for f in frames]
    if any(b - a != 1 for a, b in zip(numbers[:-1], numbers[1:])):
        raise ValueError("frame_number values must be consecutive, with no missing or duplicate frames")
    return frames


@app.get("/health")
async def health_endpoint():
    """Liveness check for the container orchestration.

    Returns:
        dict: {"status": "ok"}.
    """
    return {"status": "ok"}


@app.post("/repair-ball")
async def repair_ball_endpoint(
    frames_data: UploadFile = File(...),
    video: Optional[UploadFile] = File(None),
    frame_width: Optional[int] = Form(None),
    frame_height: Optional[int] = Form(None),
    max_gap_frames: int = Form(45),
    debug: bool = Form(False),
):
    """Endpoint to estimate the ball in every frame where it is missing.

    Accepts multipart/form-data with the frames as a JSON file and, optionally, the video
    they came from. Without the video only the detections are used; with it, camera motion
    is compensated (important for a moving camera) and optical-flow hints are added.

    Args:
        frames_data (UploadFile): JSON file holding a list of every Frame of the video, in
            any order. Sent as a file rather than a text field because text fields are capped
            at 1 MB, which a frame list passes after about 600 frames.
        video (Optional[UploadFile]): The video file. Its first frame must be frame_number 0.
        frame_width (Optional[int]): Video width, to keep estimates inside the frame when no
            video is sent.
        frame_height (Optional[int]): Video height, as above.
        max_gap_frames (int): Longest gap that is filled end to end. Defaults to 45.
        debug (bool): With a video, also save a rendered video to mock/debug_outputs.

    Returns:
        List[Frame]: The frames with estimated balls filled in (confidence 0.0 marks an
            estimate). Detections rejected as not being the ball are removed or replaced.

    Raises:
        HTTPException: If the frames JSON is invalid, or the video can't be read or doesn't
            start at frame 0.
    """
    # Starlette caps text form fields at 1 MB and FastAPI offers no way to raise it, while file
    # parts are spooled to disk with no cap, so the frame list must arrive as a file
    try:
        frames = parse_frames((await frames_data.read()).decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid frames JSON data: {e}")

    frame_size = (frame_width, frame_height) if frame_width and frame_height else None

    if video is None:
        # The repair is CPU-bound; run it off the event loop so the server stays responsive
        return await run_in_threadpool(repair_frames, frames, None, frame_size, max_gap_frames)

    if frames[0].frame_number != 0:
        raise HTTPException(status_code=400, detail="With a video, the frames must start at frame_number 0")

    # OpenCV reads videos from disk, so the upload is written to a temporary file
    with tempfile.TemporaryDirectory() as tmp_dir:
        suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
        video_path = os.path.join(tmp_dir, f"input{suffix}")
        with open(video_path, "wb") as fh:
            while chunk := await video.read(UPLOAD_CHUNK_SIZE):
                fh.write(chunk)

        try:
            video_frame_size(video_path)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid video file: it could not be opened")

        repaired = await run_in_threadpool(repair_frames, frames, video_path, frame_size, max_gap_frames)

        if debug:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            debug_path = os.path.join(DEBUG_DIR, f"repair_{timestamp}.mp4")
            await run_in_threadpool(render_video, video_path, repaired, debug_path)
            print(f"[Debug] Rendered video saved to: {debug_path}")

    return repaired
