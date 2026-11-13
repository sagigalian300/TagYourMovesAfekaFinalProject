"""
Camera motion measured from the rims in the frame list.

Rims are fixed to the court, so when a rim moves on screen it is the camera that moved. That
needs nothing but the detections the service already receives, so no video has to be sent.
The rims are presumed accurate, like every object in the frame list.
"""
import numpy as np
from typing import List, Optional

from Frame import Frame
from Rim import Rim

# How far a rim may shift between two looks at it, in rim widths per frame. A fast pan moves the
# image about one rim width per frame; the court's two rims are many widths apart, so this gate
# never pairs the near rim with the far one.
MAX_RIM_SHIFT = 1.5
# Longest stretch without any rim in view that is bridged by spreading the shift evenly over it.
# Over a longer stretch the camera may have turned to the other rim, so the shift is not trusted.
MAX_BRIDGE_FRAMES = 15


def estimate_camera_motion(frames: List[Frame]) -> Optional[np.ndarray]:
    """
    Estimates how the camera pans through the video from the rims' positions.

    For every two consecutive frames that show a rim, the same rim is found in both and the
    camera shift is its displacement. Frames without a rim in between get an equal share of
    that shift. Where no rim is in view for longer than MAX_BRIDGE_FRAMES, the camera is
    treated as still, which is also how the estimator works when it gets no camera motion.

    Only the shift is measured: a rim's box is too small for its size change to measure zoom
    reliably, and a phone or broadcast camera mostly pans anyway.

    Args:
        frames (List[Frame]): Frames in video order, one per video frame.

    Returns:
        Optional[np.ndarray]: (n, 2, 3) affine matrices mapping each frame's image points into
            the coordinates of frame 0 ([x_ref, y_ref] = M[i] @ [x, y, 1]), or None when fewer
            than two frames show a rim and there is nothing to measure.
    """
    n = len(frames)
    looks = [i for i, f in enumerate(frames) if f.rims]
    if len(looks) < 2:
        return None

    # steps[i] = shift that maps frame i's points onto frame i-1; their running sum maps onto frame 0
    steps = np.zeros((n, 2))
    for a, b in zip(looks[:-1], looks[1:]):
        if b - a > MAX_BRIDGE_FRAMES:
            continue
        shift = _rim_shift(frames[a].rims, frames[b].rims, b - a)
        if shift is not None:
            steps[a + 1:b + 1] = shift / (b - a)

    motion = np.tile(np.eye(3)[:2], (n, 1, 1))
    motion[:, :, 2] = np.cumsum(steps, axis=0)
    return motion


def _rim_shift(earlier: List[Rim], later: List[Rim], frames_apart: int) -> Optional[np.ndarray]:
    """
    Measures the camera shift between two looks at the rims.

    Each rim is paired with the nearest rim of the other look, and only pairs that choose each
    other and are within the allowed shift count. That way a rim entering or leaving the
    picture is never paired with the wrong one.

    Args:
        earlier (List[Rim]): The rims of the earlier frame.
        later (List[Rim]): The rims of the later frame.
        frames_apart (int): How many frames apart the two looks are.

    Returns:
        Optional[np.ndarray]: [dx, dy] that maps the later frame's points onto the earlier one,
            or None if no rim could be paired.
    """
    a = np.array([[(r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2] for r in earlier])
    b = np.array([[(r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2] for r in later])
    width = float(np.mean([r.x2 - r.x1 for r in earlier + later]))
    dist = np.hypot(*(b[:, None, :] - a[None, :, :]).transpose(2, 0, 1))

    shifts = []
    for j in range(len(b)):
        i = int(np.argmin(dist[j]))
        mutual = int(np.argmin(dist[:, i])) == j
        if mutual and dist[j, i] <= MAX_RIM_SHIFT * width * frames_apart:
            shifts.append(a[i] - b[j])

    # The median keeps one badly placed box from skewing the shift when both rims are in view
    return np.median(np.array(shifts), axis=0) if shifts else None


def to_reference(motion: np.ndarray, i: int, points: np.ndarray) -> np.ndarray:
    """
    Maps image points of a frame into camera-stabilized reference coordinates.

    Args:
        motion (np.ndarray): (n, 2, 3) transforms from estimate_camera_motion.
        i (int): Frame index the points belong to.
        points (np.ndarray): (k, 2) points in the image coordinates of frame i.

    Returns:
        np.ndarray: (k, 2) points in reference coordinates.
    """
    return points @ motion[i][:, :2].T + motion[i][:, 2]


def from_reference(motion: np.ndarray, i: int, points: np.ndarray) -> np.ndarray:
    """
    Maps reference points back into the image coordinates of a frame (the inverse of to_reference).

    Args:
        motion (np.ndarray): (n, 2, 3) transforms from estimate_camera_motion.
        i (int): Frame index to map into.
        points (np.ndarray): (k, 2) points in reference coordinates.

    Returns:
        np.ndarray: (k, 2) points in the image coordinates of frame i.
    """
    A, t = motion[i][:, :2], motion[i][:, 2]
    return (points - t) @ np.linalg.inv(A).T
