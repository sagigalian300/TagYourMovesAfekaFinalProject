import os
import re
import json
import cv2
import numpy as np
from typing import List, Optional
from ultralytics import YOLO

from Ball import Ball
from Player import Player
from Rim import Rim
from Frame import Frame
from Offline import estimate_missing_balls_offline
from CameraMotion import estimate_camera_motion
from pipeline import apply_optical_flow, render_video  # also imported from here by evaluate_offline.py

# Detector used by the local test scripts. Any Ultralytics model works, since classes are
# matched by name below. 1280 instead of the default 640 because the ball is small: at 640 a
# 1080p ball shrinks to ~17px and is missed far more often.
DEFAULT_MODEL = "yolo26s.pt"
DEFAULT_IMGSZ = 1280


# Classes are matched by name, so COCO models ('sports ball', 'person') and basketball
# models ('ball', 'human', 'rim', or per-player classes like 'Red_23') all work unchanged
BALL_NAMES = {'sports ball', 'ball', 'basketball'}
RIM_NAMES = {'rim', 'hoop', 'basket'}
PLAYER_NAMES = {'person', 'human', 'player', 'referee'}
PLAYER_PATTERN = re.compile(r'[a-z]+_\d+')  # jersey / referee classes, e.g. 'red_23', 'refree_1'


def class_role(name: str) -> Optional[str]:
    """
    Maps a model's class name to the role it plays in the pipeline.

    Args:
        name (str): Class name from the model (case-insensitive), e.g. 'sports ball' or 'Red_23'.

    Returns:
        Optional[str]: 'ball', 'rim' or 'player', or None for classes the pipeline ignores.
    """
    name = name.lower()
    if name in BALL_NAMES:
        return 'ball'
    if name in RIM_NAMES:
        return 'rim'
    if name in PLAYER_NAMES or PLAYER_PATTERN.fullmatch(name):
        return 'player'
    return None


def detect_objects_yolo(model: YOLO, image_frame, confidence_threshold: float = 0.25, imgsz: Optional[int] = None):
    """
    Runs the detector on one frame.

    Args:
        model (YOLO): The loaded detector.
        image_frame (np.ndarray): The frame (BGR).
        confidence_threshold (float, optional): Detections below this are ignored. Defaults to 0.25.
        imgsz (Optional[int], optional): Inference resolution. None uses the model's default.
            Defaults to None.

    Returns:
        Tuple[Optional[Ball], List[Player], List[Rim]]: The highest-confidence ball (or None),
            all players and all rims.
    """
    results =model(image_frame, verbose=False, **({'imgsz': imgsz} if imgsz else {}))[0]
    best_ball = None
    highest_conf_ball = 0.0
    players = []
    rims = []

    for box in results.boxes:
        role = class_role(results.names[int(box.cls[0])])
        conf = float(box.conf[0])

        if conf >= confidence_threshold:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if role == 'ball' and conf > highest_conf_ball:
                highest_conf_ball = conf
                best_ball = Ball(confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2)
            elif role == 'player':
                players.append(Player(confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2))
            elif role == 'rim':
                rims.append(Rim(confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2))

    return best_ball, players, rims


def default_cache_path(video_path: str, model_path: str, imgsz: Optional[int] = None) -> str:
    """
    Builds the detection cache path, unique per video, model and inference size.

    Args:
        video_path (str): Path to the video.
        model_path (str): Path to the detector weights.
        imgsz (Optional[int], optional): Inference resolution, added to the name when given.
            Defaults to None.

    Returns:
        str: e.g. 'test_vod2.yolo26s-1280.detections.json' next to the video.
    """
    model_name =os.path.splitext(os.path.basename(model_path))[0]
    if imgsz:
        model_name += f"-{imgsz}"
    return f"{os.path.splitext(video_path)[0]}.{model_name}.detections.json"


def save_detections(frames: List[Frame], cache_path: str):
    """
    Saves the detections of every frame as JSON.

    Args:
        frames (List[Frame]): Frames to save (ball, players and rims).
        cache_path (str): Output JSON path.
    """
    data = []
    for f in frames:
        b = f.ball
        data.append({
            'frame_number': f.frame_number,
            'timestamp_ms': f.timestamp_ms,
            'ball': [b.confidence, b.x1, b.y1, b.x2, b.y2] if b is not None else None,
            'players': [[p.confidence, p.x1, p.y1, p.x2, p.y2] for p in f.players],
            'rims': [[r.confidence, r.x1, r.y1, r.x2, r.y2] for r in f.rims],
        })
    with open(cache_path, 'w') as fh:
        json.dump(data, fh)


def load_detections(cache_path: str) -> List[Frame]:
    """
    Loads detections saved by save_detections.

    Builds fresh Frame objects every call, so callers may mutate them freely.

    Args:
        cache_path (str): JSON path written by save_detections.

    Returns:
        List[Frame]: Frames in video order.
    """
    with open(cache_path) as fh:
        data = json.load(fh)
    return [Frame(
        frame_number=d['frame_number'],
        timestamp_ms=d['timestamp_ms'],
        ball=Ball(*d['ball']) if d['ball'] is not None else None,
        players=[Player(*p) for p in d['players']],
        rims=[Rim(*r) for r in d.get('rims', [])],
    ) for d in data]


def detect_video(input_video_path: str, model_path: str, cache_path: Optional[str] = None,
                 imgsz: Optional[int] = None) -> List[Frame]:
    """
    Pass 1: runs YOLO on every frame. Results are cached as JSON, so re-runs and the
    evaluation harness don't pay for inference again.

    Args:
        input_video_path (str): Path to the video.
        model_path (str): Path to the detector weights (.pt, .onnx, ...).
        cache_path (Optional[str], optional): JSON cache. Loaded if it exists, written otherwise.
            None disables caching. Defaults to None.
        imgsz (Optional[int], optional): Inference resolution. The ball is small, so a COCO model
            gains from 1280 over its default 640; ONNX exports only accept the size they were
            exported with. Defaults to None.

    Returns:
        List[Frame]: One Frame per video frame with its ball, players and rims.

    Raises:
        IOError: If the video can't be opened.
    """
    if cache_path and os.path.exists(cache_path):
        return load_detections(cache_path)

    model = YOLO(model_path, task='detect')
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video {input_video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Pass 1: Scanning {total_frames} frames with YOLO...")

    frames: List[Frame] = []
    frame_number = 0
    while True:
        ret, image_frame = cap.read()
        if not ret:
            break

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        ball, players, rims = detect_objects_yolo(model, image_frame, imgsz=imgsz)
        frames.append(Frame(frame_number=frame_number, timestamp_ms=timestamp_ms, ball=ball, players=players, rims=rims))
        frame_number += 1
        if frame_number % 100 == 0:
            print(f"  {frame_number}/{total_frames}")

    cap.release()
    if cache_path:
        save_detections(frames, cache_path)
    return frames


def camera_motion_for(input_video_path: str, frames: List[Frame], cache_path: Optional[str] = None) -> np.ndarray:
    """
    Gets the camera motion per frame (see CameraMotion.py), cached since it only depends on
    the video and the player boxes.

    Args:
        input_video_path (str): Path to the video.
        frames (List[Frame]): Frames in video order; their player boxes are masked out.
        cache_path (Optional[str], optional): .npy cache. Loaded if it exists, written otherwise.
            None disables caching. Defaults to None.

    Returns:
        np.ndarray: (n, 2, 3) transforms from estimate_camera_motion.
    """
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)
    motion = estimate_camera_motion(input_video_path, frames)
    if cache_path:
        np.save(cache_path, motion)
    return motion


def default_camera_cache_path(video_path: str) -> str:
    """
    Builds the camera motion cache path for a video.

    Args:
        video_path (str): Path to the video.

    Returns:
        str: e.g. 'test_vod2.camera.npy' next to the video.
    """
    return f"{os.path.splitext(video_path)[0]}.camera.npy"


def require_model(model_path: str):
    """
    Checks that the detector weights exist before running.

    Args:
        model_path (str): Path to the detector weights.

    Raises:
        FileNotFoundError: If the weights file is missing.
    """
    # Fail early with a clear message; otherwise Ultralytics may try to download a model of
    # the same name or fail deep inside loading
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Detector weights not found: {model_path}\n"
            f"Put the weights file next to test_offline.py or change DEFAULT_MODEL"
        )


def test_offline_estimation(input_video_path: str, output_video_path: str, model_path: str, imgsz: Optional[int] = None):
    """
    Runs the whole pipeline: detection, camera motion, optical-flow hints, offline estimation
    and rendering.

    Args:
        input_video_path (str): Path to the video.
        output_video_path (str): Path of the rendered .mp4.
        model_path (str): Path to the detector weights.
        imgsz (Optional[int], optional): Inference resolution. Defaults to None.
    """
    cache_path =default_cache_path(input_video_path, model_path, imgsz)
    if not os.path.exists(cache_path):
        require_model(model_path)
    frames_metadata = detect_video(input_video_path, model_path, cache_path, imgsz=imgsz)

    print("Estimating camera motion...")
    camera_motion = camera_motion_for(input_video_path, frames_metadata, default_camera_cache_path(input_video_path))

    print("Applying Lucas-Kanade hints...")
    frames_metadata = apply_optical_flow(input_video_path, frames_metadata)

    cap = cv2.VideoCapture(input_video_path)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()

    print("Offline Processing: Estimating missing balls...")
    frames_metadata = estimate_missing_balls_offline(
        frames_metadata, max_gap_frames=45, frame_size=frame_size, camera_motion=camera_motion
    )

    print("Pass 2: Rendering output video...")
    render_video(input_video_path, frames_metadata, output_video_path)
    print(f"Success! Output saved to: {output_video_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "test_vod2.mp4")
    output_path = os.path.join(script_dir, "output_interpolated.mp4")
    weights_path = os.path.join(script_dir, DEFAULT_MODEL)

    test_offline_estimation(input_path, output_path, weights_path, imgsz=DEFAULT_IMGSZ)
