"""
The ball repair pipeline: optional video passes (camera motion, optical-flow hints) around the
offline estimator. Shared by the FastAPI service (main.py) and the local test scripts, and kept
free of YOLO/ultralytics so the service doesn't need a detector installed.
"""
import cv2
from typing import List, Optional, Tuple

from Frame import Frame
from Offline import estimate_missing_balls_offline
from OpticalFlowTracker import BallOpticalFlowTracker # Import the LK Tracker
from CameraMotion import estimate_camera_motion


def video_frame_size(input_video_path: str) -> Tuple[int, int]:
    """
    Reads a video's frame size.

    Args:
        input_video_path (str): Path to the video.

    Returns:
        Tuple[int, int]: (width, height) in pixels.

    Raises:
        ValueError: If the video can't be opened.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {input_video_path}")
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return frame_size


def apply_optical_flow(input_video_path: str, frames: List[Frame], max_frames: int = 8) -> List[Frame]:
    """
    Fills frames YOLO missed with short-range Lucas-Kanade hints (confidence 0.0).
    The offline estimator treats these as weak measurements, not anchors.

    Args:
        input_video_path (str): Path to the video.
        frames (List[Frame]): Frames in video order. Modified in place.
        max_frames (int, optional): Max consecutive frames to track after a detection. Defaults to 8.

    Returns:
        List[Frame]: The same frames, with hints where the tracker succeeded.
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
    """
    Writes a copy of the video with the ball drawn on it: green for detections,
    red for estimates.

    Args:
        input_video_path (str): Path to the source video.
        frames (List[Frame]): Frames in video order, with their final balls.
        output_video_path (str): Path of the .mp4 to write.
    """
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


def repair_frames(
    frames: List[Frame],
    input_video_path: Optional[str] = None,
    frame_size: Optional[Tuple[int, int]] = None,
    max_gap_frames: int = 45,
) -> List[Frame]:
    """
    Estimates the ball in every frame where it is missing.

    Without a video only the detections are used. With the video, camera motion is
    compensated (important for a moving camera) and optical-flow hints are added.

    Args:
        frames (List[Frame]): All frames in video order, one per video frame. Modified in place.
        input_video_path (Optional[str], optional): The video the frames came from; frames[0]
            must be its first frame. Defaults to None.
        frame_size (Optional[Tuple[int, int]], optional): (width, height), used to keep estimates
            inside the frame. Read from the video when one is given. Defaults to None.
        max_gap_frames (int, optional): Longest gap that is filled end to end. Defaults to 45.

    Returns:
        List[Frame]: The same frames with estimated balls (confidence 0.0) filled in.

    Raises:
        ValueError: If the video can't be opened.
    """
    camera_motion = None
    if input_video_path is not None:
        frame_size = video_frame_size(input_video_path)
        camera_motion = estimate_camera_motion(input_video_path, frames)
        frames = apply_optical_flow(input_video_path, frames)

    return estimate_missing_balls_offline(
        frames, max_gap_frames=max_gap_frames, frame_size=frame_size, camera_motion=camera_motion
    )
