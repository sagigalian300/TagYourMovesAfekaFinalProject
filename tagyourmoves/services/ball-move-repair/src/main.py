"""
Main entry point for the ball-move-repair microservice.

Runs a FastAPI server that receives the detected frames of a video, estimates the ball's location
in the frames where it is missing, and returns the same frames JSON with only those balls filled.
Every object in the frames is presumed accurate; the service's only job is the missing balls.
"""

import os
import sys
import json
import inspect
from typing import List, Optional, Tuple

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool

# The repair code and its Frame/Ball/Player/Rim classes import each other by bare module
# name (e.g. `from Ball import Ball` inside Frame.py, which can't be changed), so both their
# folders have to be on the import path: contracts/ for the shared classes, offline/ for the algorithm
REPAIRERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repairers")
sys.path.insert(0, os.path.join(REPAIRERS_DIR, "contracts"))
sys.path.insert(0, os.path.join(REPAIRERS_DIR, "offline"))

from Ball import Ball
from Frame import Frame 
from Player import Player
from Rim import Rim 
from Offline import repair_frames

app =FastAPI(title="Ball Move Repair Microservice")


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


def parse_frames(frames_data: str) -> Tuple[List[dict], List[Frame]]:
    """Parses and validates the frames sent with a request.

    Args:
        frames_data (str): JSON list of Frame objects. The ball is an object or null;
            players and rims are lists (may be omitted).

    Returns:
        Tuple[List[dict], List[Frame]]: The frames exactly as received, and a Frame object
            for each of them, in the same order (frames[i] belongs to raw[i]).

    Raises:
        ValueError: If the JSON is malformed, empty, or the frame numbers are not
            consecutive. The estimator uses the position in the list as time, so every
            frame must be present, including those without a ball.
    """
    raw = json.loads(frames_data)
    if not isinstance(raw, list) or not raw or not all(isinstance(d, dict) for d in raw):
        raise ValueError("frames_data must be a non-empty JSON list of frame objects")

    frames = []
    for d in raw:
        frames.append(_from_dict(Frame, {
            **d,
            "ball": _from_dict(Ball, d["ball"]) if d.get("ball") else None,
            "players": [_from_dict(Player, p) for p in d.get("players") or []],
            "rims": [_from_dict(Rim, r) for r in d.get("rims") or []],
        }))

    # Checked on a sorted copy: the caller may send frames in any order, and the response
    # keeps their order, so the list itself is not reordered
    numbers = sorted(f.frame_number for f in frames)
    if any(b - a != 1 for a, b in zip(numbers[:-1], numbers[1:])):
        raise ValueError("frame_number values must be consecutive, with no missing or duplicate frames")
    return raw, frames


def build_response(raw: List[dict], frames: List[Frame], received_balls: List[Optional[Ball]]) -> List[dict]:
    """Writes the repaired balls back into the frames JSON as it was received.

    Only the "ball" of a frame whose ball the service filled is written. Everything else
    (frame order, key order, players, rims, fields only other services know about, number
    formatting of the balls that were sent) is returned exactly as the caller sent it, so the
    response can be used anywhere the request's JSON could.

    Args:
        raw (List[dict]): The frames as received; updated in place.
        frames (List[Frame]): The repaired Frame objects, aligned with raw.
        received_balls (List[Optional[Ball]]): Each frame's ball object before the repair,
            aligned with raw.

    Returns:
        List[dict]: raw, with the repaired balls.
    """
    for d, f, received in zip(raw, frames, received_balls):
        # The repair only sets a new ball object where the ball was missing; a ball that was sent
        # is the very same object, and keeping its original dict avoids rewriting it (e.g. 10
        # becoming 10.0, or dropping extra keys)
        if f.ball is received:
            continue
        d["ball"] = None if f.ball is None else {
            "confidence": f.ball.confidence, "x1": f.ball.x1, "y1": f.ball.y1, "x2": f.ball.x2, "y2": f.ball.y2,
        }
    return raw


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
    frame_width: Optional[int] = Form(None),
    frame_height: Optional[int] = Form(None),
    max_gap_frames: int = Form(45),
):
    """Endpoint to estimate the ball in every frame where it is missing.

    Accepts multipart/form-data with the frames as a JSON file. The camera's movement is
    measured from the rims in the frames, so the video itself is not needed.

    Args:
        frames_data (UploadFile): JSON file holding a list of every Frame of the video, in
            any order. Sent as a file rather than a text field because text fields are capped
            at 1 MB, which a frame list passes after about 600 frames.
        frame_width (Optional[int]): Video width, to keep estimates inside the frame.
        frame_height (Optional[int]): Video height, as above.
        max_gap_frames (int): Longest gap that is filled end to end. Defaults to 45.

    Returns:
        List[dict]: The received frames JSON, in the same order and format, where each
            "ball": null was filled with an estimate (confidence 0.0) when one could be made.
            Balls that were sent come back unchanged.

    Raises:
        HTTPException: If the frames JSON is invalid.
    """
    # Starlette caps text form fields at 1 MB and FastAPI offers no way to raise it, while file
    # parts are spooled to disk with no cap, so the frame list must arrive as a file
    try:
        raw, frames = parse_frames((await frames_data.read()).decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid frames JSON data: {e}")

    # The repair needs the frames in time order and sets a new ball object where it fills one,
    # so remember what each frame held before, to tell the filled ones apart afterwards
    ordered = sorted(frames, key=lambda f: f.frame_number)
    received_balls = [f.ball for f in frames]
    frame_size = (frame_width, frame_height) if frame_width and frame_height else None

    # The repair is CPU-bound; run it off the event loop so the server stays responsive
    await run_in_threadpool(repair_frames, ordered, frame_size, max_gap_frames)
    return build_response(raw, frames, received_balls)
