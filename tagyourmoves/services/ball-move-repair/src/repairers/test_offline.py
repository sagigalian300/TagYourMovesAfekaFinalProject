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
from OpticalFlowTracker import BallOpticalFlowTracker # Import the LK Tracker
from CameraMotion import estimate_camera_motion

# Basketball-specific detector (classes: ball, human, rim).
DEFAULT_MODEL = "basketball_yolov8n.onnx"
DEFAULT_MODEL_URL = "https://huggingface.co/MTerryJack/Element-Detect-Basketball"
DEFAULT_IMGSZ = 640  # the ONNX export only accepts its export size

"""
DEFAULT_MODEL = "yolov8s.pt"
DEFAULT_IMGSZ = 1280
"""


# Classes are matched by name, so COCO models ('sports ball', 'person') and basketball
# models ('ball', 'human', 'rim', or per-player classes like 'Red_23') all work unchanged
BALL_NAMES = {'sports ball', 'ball', 'basketball'}
RIM_NAMES = {'rim', 'hoop', 'basket'}
PLAYER_NAMES = {'person', 'human', 'player', 'referee'}
PLAYER_PATTERN = re.compile(r'[a-z]+_\d+')  # jersey / referee classes, e.g. 'red_23', 'refree_1'


def class_role(name: str) -> Optional[str]:
    name = name.lower()
    if name in BALL_NAMES:
        return 'ball'
    if name in RIM_NAMES:
        return 'rim'
    if name in PLAYER_NAMES or PLAYER_PATTERN.fullmatch(name):
        return 'player'
    return None


def detect_objects_yolo(model: YOLO, image_frame, confidence_threshold: float = 0.25, imgsz: Optional[int] = None):
    results = model(image_frame, verbose=False, **({'imgsz': imgsz} if imgsz else {}))[0]
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
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    if imgsz:
        model_name += f"-{imgsz}"
    return f"{os.path.splitext(video_path)[0]}.{model_name}.detections.json"


def save_detections(frames: List[Frame], cache_path: str):
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
    """Builds fresh Frame objects every call, so callers may mutate them freely."""
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

    imgsz: inference resolution. The ball is small, so a COCO model gains from 1280
    over its default 640; ONNX exports only accept the size they were exported with.
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
    """Camera pan/zoom per frame (see CameraMotion.py). Depends only on the video and player boxes, so it is cached."""
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)
    motion = estimate_camera_motion(input_video_path, frames)
    if cache_path:
        np.save(cache_path, motion)
    return motion


def default_camera_cache_path(video_path: str) -> str:
    return f"{os.path.splitext(video_path)[0]}.camera.npy"


def apply_optical_flow(input_video_path: str, frames: List[Frame], max_frames: int = 8) -> List[Frame]:
    """
    Fills frames YOLO missed with short-range Lucas-Kanade hints (confidence 0.0).
    The offline estimator treats these as weak measurements, not anchors.
    """
    lk_tracker = BallOpticalFlowTracker(max_frames=max_frames)
    cap = cv2.VideoCapture(input_video_path)

    for f in frames:
        ret, image_frame = cap.read()
        if not ret:
            break

        if f.ball is not None:
            # YOLO detected the ball -> Re-initialize optical flow feature points
            lk_tracker.initialize(image_frame, f.ball)
        else:
            # YOLO missed the ball -> Use Lucas-Kanade Optical Flow
            f.ball = lk_tracker.update(image_frame)

    cap.release()
    return frames


def render_video(input_video_path: str, frames: List[Frame], output_video_path: str):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_obj in frames:
        ret, image_frame = cap.read()
        if not ret:
            break

        if frame_obj.ball is not None:
            b = frame_obj.ball
            x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)

            color = (0, 255, 0) if b.confidence > 0.0 else (0, 0, 255)
            label = f"Detected: {b.confidence:.2f}" if b.confidence > 0.0 else "Estimated"

            cv2.rectangle(image_frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(image_frame, (x1, max(0, y1 - 20)), (x1 + 110, y1), color, -1)
            cv2.putText(image_frame, label, (x1 + 2, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        out.write(image_frame)

    cap.release()
    out.release()


def require_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Detector weights not found: {model_path}\n"
            f"Download weights.onnx from {DEFAULT_MODEL_URL} and save it as {DEFAULT_MODEL}"
        )


def test_offline_estimation(input_video_path: str, output_video_path: str, model_path: str, imgsz: Optional[int] = None):
    cache_path = default_cache_path(input_video_path, model_path, imgsz)
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
