import cv2
import numpy as np
from typing import List

from Frame import Frame


def estimate_camera_motion(input_video_path: str, frames: List[Frame], scale: float = 0.5) -> np.ndarray:
    """
    Estimates how the camera moves (pan / tilt) through the video.

    Background corners are tracked with Lucas-Kanade between consecutive frames, with player boxes
    masked out since players move on their own, and the image shift is taken from the RANSAC inliers.
    Chaining these gives, for every frame, a transform into the coordinates of frame 0, so motion
    measured there is the ball's own motion and not the camera's.

    Returns:
        (n, 2, 3) array of affine matrices: [x_ref, y_ref] = M[i] @ [x, y, 1] for a point in frame i.
    """
    n = len(frames)
    motion = np.tile(np.eye(3)[:2], (n, 1, 1))
    cumulative = np.eye(3)

    # Points are found at reduced resolution; S converts between the two resolutions
    S = np.diag([scale, scale, 1.0])
    S_inv = np.diag([1 / scale, 1 / scale, 1.0])

    cap = cv2.VideoCapture(input_video_path)
    prev_gray = None

    for i, f in enumerate(frames):
        ret, image_frame = cap.read()
        if not ret:
            motion[i:] = cumulative[:2]
            break

        gray = cv2.cvtColor(cv2.resize(image_frame, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            step = _frame_to_frame(prev_gray, gray, frames[i - 1], scale)
            if step is not None:
                # step maps current -> previous frame (full resolution); cumulative maps previous -> frame 0
                cumulative = cumulative @ (S_inv @ step @ S)

        motion[i] = cumulative[:2]
        prev_gray = gray

    cap.release()
    return motion


def _frame_to_frame(prev_gray: np.ndarray, gray: np.ndarray, prev_frame: Frame, scale: float):
    """Similarity transform (3x3) mapping points of `gray` onto `prev_gray`, or None if it can't be estimated."""
    mask = np.full_like(prev_gray, 255)
    for p in prev_frame.players:
        cv2.rectangle(mask, (int(p.x1 * scale), int(p.y1 * scale)), (int(p.x2 * scale), int(p.y2 * scale)), 0, -1)

    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.01, minDistance=10, mask=mask)
    if p0 is None or len(p0) < 10:
        return None

    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(21, 21), maxLevel=3)
    if p1 is None:
        return None
    good = st.reshape(-1) == 1
    if good.sum() < 10:
        return None

    # RANSAC separates background points from anything else that moves (the ball, missed players)
    M, inliers = cv2.estimateAffinePartial2D(p1[good], p0[good], method=cv2.RANSAC, ransacReprojThreshold=1.5)
    if M is None or inliers is None or inliers.sum() < 8:
        return None

    # Keep only the translation: small per-frame scale errors multiply when chained over hundreds of
    # frames and wreck the reference coordinates, while translation errors merely add up
    inl = inliers.reshape(-1) == 1
    shift = np.median((p0[good][inl] - p1[good][inl]).reshape(-1, 2), axis=0)
    return np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]], [0.0, 0.0, 1.0]])


def to_reference(motion: np.ndarray, i: int, points: np.ndarray) -> np.ndarray:
    """Maps (k, 2) image points of frame i into reference coordinates."""
    return points @ motion[i][:, :2].T + motion[i][:, 2]


def from_reference(motion: np.ndarray, i: int, points: np.ndarray) -> np.ndarray:
    """Maps (k, 2) reference points back into the image coordinates of frame i."""
    A, t = motion[i][:, :2], motion[i][:, 2]
    return (points - t) @ np.linalg.inv(A).T
