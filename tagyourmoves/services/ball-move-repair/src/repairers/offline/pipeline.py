"""
The ball repair pipeline: camera motion from the rims, then the offline estimator.

Works from the frame list alone. Every object in it is presumed accurate, so no video is needed:
the rims stand in for the background as the reference for the camera's movement.
"""
from typing import List, Optional, Tuple

from Frame import Frame
from Offline import estimate_missing_balls_offline
from CameraMotion import estimate_camera_motion


def repair_frames(
    frames: List[Frame],
    frame_size: Optional[Tuple[int, int]] = None,
    max_gap_frames: int = 45,
) -> List[Frame]:
    """
    Estimates the ball in every frame where it is missing.

    Args:
        frames (List[Frame]): All frames in video order, one per video frame. Modified in place.
        frame_size (Optional[Tuple[int, int]], optional): (width, height), used to keep estimates
            inside the frame. Defaults to None.
        max_gap_frames (int, optional): Longest gap that is filled end to end. Defaults to 45.

    Returns:
        List[Frame]: The same frames with estimated balls (confidence 0.0) in place of the
            missing ones; balls that were present are left untouched.
    """
    # None when fewer than two frames show a rim: the estimator then works in image coordinates,
    # which is right for a still camera and the best that can be done for a moving one
    camera_motion = estimate_camera_motion(frames)
    return estimate_missing_balls_offline(
        frames, max_gap_frames=max_gap_frames, frame_size=frame_size, camera_motion=camera_motion
    )
