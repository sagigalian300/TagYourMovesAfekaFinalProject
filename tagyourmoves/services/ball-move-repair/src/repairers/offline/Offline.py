import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from Frame import Frame
from Ball import Ball
from CameraMotion import to_reference, from_reference

# All distances below are expressed in ball diameters, so they scale with zoom / resolution.
MAX_SPEED = 2.5             # Max plausible ball motion between consecutive frames (sets the Kalman's starting doubt)
DETECTION_NOISE = 0.1       # Std of a given ball's center: small, since the input is presumed accurate
ACCELERATION_NOISE = 0.15   # Std of unmodelled acceleration per frame in the Kalman model
POSSESSION_CONTEXT = 10     # Frames around a gap used to check whether the ball moves with the player
PLAYER_MARGIN = 0.5         # How far outside a player's box the ball may be and still be "held"
ONE_SIDED_FRAMES = 10       # Frames to follow the holder when only one side of a gap is known
PLAYER_IOU = 0.3            # Min IoU to link a player box to an existing player track
PLAYER_MAX_MISS = 5         # Frames a player track survives without a matching detection
PLAYER_SMOOTHING = 9        # Moving-average window for player boxes (they jitter with arms and legs)


def estimate_missing_balls_offline(
    frames: List[Frame],
    max_gap_frames: int = 15,
    frame_size: Optional[Tuple[int, int]] = None,
    camera_motion: Optional[np.ndarray] = None,
    use_possession: bool = True,
) -> List[Frame]:
    """
    Estimates the ball's location in frames where it is missing (ball is None).

    Contract: every object in the frames is presumed accurate, including every ball that is
    present, whatever its confidence. Checking detections is the detection service's job, so
    present balls are used as they are and never changed; only missing balls are filled.
    Every estimate written back has confidence 0.0.

    Each gap between two known balls is filled using both of its ends:
      * possession - the same player holds the ball on both sides and the ball moves with
                     that player around the gap: the ball follows the player's box
      * otherwise  - a Kalman RTS smoother over the known balls
    Gaps longer than `max_gap_frames` are left empty apart from a short stretch near each end.

    Args:
        frames (List[Frame]): Frames in video order. Modified in place.
        max_gap_frames (int, optional): Longest gap that is filled end to end. Defaults to 15.
        frame_size (Optional[Tuple[int, int]], optional): (width, height) of the video. When given,
            estimates are clamped inside the frame. Defaults to None.
        camera_motion (Optional[np.ndarray], optional): Per-frame transforms from
            CameraMotion.estimate_camera_motion. When given, ball motion is modelled in
            camera-stabilized coordinates, so pans don't look like ball motion. Defaults to None.
        use_possession (bool, optional): Fill gaps by following the player holding the ball.
            Switch off for evaluation. Defaults to True.

    Returns:
        List[Frame]: The same frames, with estimated balls (confidence 0.0) in the gaps.
    """
    n = len(frames)
    if n == 0:
        return frames

    # =================================================================
    # STEP 1: Extract the known balls (the anchors every gap is filled from)
    # =================================================================
    centers = np.full((n, 2), np.nan)
    sizes = np.full((n, 2), np.nan)
    for i, f in enumerate(frames):
        b = f.ball
        if b is not None:
            centers[i] = ((b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2)
            sizes[i] = (b.x2 - b.x1, b.y2 - b.y1)

    anchors = np.flatnonzero(~np.isnan(centers[:, 0]))
    if len(anchors) == 0:
        return frames

    # Ball motion is modelled in camera-stabilized coordinates when the camera motion is known
    stable = centers.copy()
    if camera_motion is not None:
        for i in anchors:
            stable[i] = to_reference(camera_motion, i, centers[i:i + 1])[0]

    # Ball diameter for every frame, interpolated between anchors
    anchor_diameters = sizes[anchors].mean(axis=1)
    diameters = np.interp(np.arange(n), anchors, anchor_diameters)

    # =================================================================
    # STEP 2: Track players so "the same player" means something across frames
    # =================================================================
    players = _PlayerTracks(frames) if use_possession else None

    # =================================================================
    # STEP 3: Kalman RTS smoother over the anchors as the general fallback
    # =================================================================
    smoothed = _kalman_smooth(n, anchors, stable, diameters, max_gap_frames)
    if camera_motion is not None:
        for i in np.flatnonzero(np.isfinite(smoothed[:, 0])):
            smoothed[i] = from_reference(camera_motion, i, smoothed[i:i + 1])[0]

    # =================================================================
    # STEP 4: Fill every gap
    # =================================================================
    estimates: Dict[int, np.ndarray] = {}

    def owner(i: int) -> Optional[int]:
        """
        Finds the player holding the ball in frame i.

        Args:
            i (int): Frame index; must hold a ball center.

        Returns:
            Optional[int]: Player track id, or None if nobody holds it or possession is off.
        """
        return players.owner(i, centers[i], diameters[i]) if players is not None else None

    def follow_owner(anchor: int, frame_range: range):
        """
        Fills a one-sided gap by keeping the ball at a fixed spot relative to whoever held it.

        Args:
            anchor (int): Frame of the known detection next to the gap.
            frame_range (range): Frames to fill, walking away from the anchor. Stops early
                if the player's track ends.
        """
        tid = owner(anchor)
        if tid is None:
            return
        offset = players.relative(tid, anchor, centers[anchor])
        for i in frame_range:
            pos = players.absolute(tid, i, offset)
            if pos is None:
                break
            estimates[i] = pos

    for a, b in zip(anchors[:-1], anchors[1:]):
        gap = range(a + 1, b)
        if len(gap) == 0:
            continue

        if len(gap) > max_gap_frames:
            follow_owner(a, range(a + 1, min(b, a + 1 + ONE_SIDED_FRAMES)))
            follow_owner(b, range(b - 1, max(a, b - 1 - ONE_SIDED_FRAMES), -1))
            continue

        filled = None
        if players is not None:
            tid = owner(a)
            if tid is not None and tid == owner(b):
                near = anchors[(anchors >= a - POSSESSION_CONTEXT) & (anchors <= b + POSSESSION_CONTEXT)]
                if players.moves_with(tid, near, centers, stable):
                    filled = players.interpolate(tid, a, b, centers[a], centers[b])
        if filled is None:
            filled = smoothed[a + 1:b]

        for i, pos in zip(gap, filled):
            estimates[i] = pos

    # Before the first and after the last detection only one side is known
    follow_owner(anchors[0], range(anchors[0] - 1, max(-1, anchors[0] - 1 - ONE_SIDED_FRAMES), -1))
    follow_owner(anchors[-1], range(anchors[-1] + 1, min(n, anchors[-1] + 1 + ONE_SIDED_FRAMES)))

    # =================================================================
    # STEP 5: Write estimates into the frames whose ball is missing
    # =================================================================
    box_sizes = _interpolate_sizes(n, anchors, sizes)

    # Frames that already hold a ball are skipped: under the contract their ball is correct,
    # and the caller gets it back exactly as sent
    for i, f in enumerate(frames):
        if f.ball is None and i in estimates and np.all(np.isfinite(estimates[i])):
            f.ball = _make_ball(estimates[i], box_sizes[i], frame_size)

    return frames


def _kalman_smooth(
    n: int,
    anchors: np.ndarray,
    centers: np.ndarray,
    diameters: np.ndarray,
    max_gap_frames: int,
) -> np.ndarray:
    """
    Smooths the ball center with a constant-velocity Kalman filter and an RTS smoother.

    Runs separately on each segment of the video (segments are split at gaps longer than
    max_gap_frames) so no velocity is carried across a long gap.

    Args:
        n (int): Number of frames in the video.
        anchors (np.ndarray): Sorted frame indices of the known balls.
        centers (np.ndarray): (n, 2) ball centers per frame (camera-stabilized when available).
        diameters (np.ndarray): (n,) ball diameter per frame, used to scale the noise.
        max_gap_frames (int): Gaps longer than this split the video into separate segments.

    Returns:
        np.ndarray: (n, 2) smoothed centers, NaN outside the segments.
    """
    out = np.full((n, 2), np.nan)
    is_anchor = np.zeros(n, dtype=bool)
    is_anchor[anchors] = True

    breaks = np.flatnonzero(np.diff(anchors) > max_gap_frames + 1)
    for segment in np.split(anchors, breaks + 1):
        start, end = segment[0], segment[-1]
        scale = float(np.median(diameters[segment]))

        kf = KalmanFilter(dim_x=4, dim_z=2)  # state: x, vx, y, vy
        kf.F = np.array([[1, 1, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 0, 1]], dtype=float)
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0]], dtype=float)
        kf.Q = Q_discrete_white_noise(dim=2, dt=1.0, var=(ACCELERATION_NOISE * scale) ** 2, block_size=2)
        kf.x = np.array([[centers[start, 0]], [0.0], [centers[start, 1]], [0.0]])
        kf.P = np.diag([scale ** 2, (MAX_SPEED * scale) ** 2, scale ** 2, (MAX_SPEED * scale) ** 2])

        mu, cov = [], []
        for i in range(start, end + 1):
            kf.predict()
            if is_anchor[i]:
                kf.update(centers[i], R=np.eye(2) * (DETECTION_NOISE * diameters[i]) ** 2)
            mu.append(kf.x.copy())
            cov.append(kf.P.copy())

        M, _, _, _ = kf.rts_smoother(np.array(mu), np.array(cov))
        out[start:end + 1, 0] = M[:, 0, 0]
        out[start:end + 1, 1] = M[:, 2, 0]

    return out


def _linear_residual(t: np.ndarray, points: np.ndarray) -> float:
    """
    Measures how far points are from moving at constant velocity.

    Args:
        t (np.ndarray): (k,) frame times of the points. Needs at least 2 distinct values.
        points (np.ndarray): (k, 2) positions.

    Returns:
        float: RMS distance of the points from the best-fitting straight-line motion.
    """
    fit = np.stack([np.polyval(np.polyfit(t, points[:, c], 1), t) for c in range(2)], axis=1)
    return float(np.sqrt(np.mean(np.sum((fit - points) ** 2, axis=1))))


class _PlayerTracks:
    """
    Links player boxes across frames into tracks and smooths each track. Uses the players'
    tracking_id when the input provides one for every player, and IoU (Hungarian) matching
    otherwise. Kept internal so the Player objects themselves are not modified.

    Attributes:
        tracks (List[Tuple[int, np.ndarray]]): Per track id: (first frame, (k, 4) smoothed boxes
            for consecutive frames from there).
        present (Dict[int, List[int]]): Per frame: ids of the tracks with a real detection in it.
    """

    def __init__(self, frames: List[Frame]):
        """
        Builds the player tracks.

        Args:
            frames (List[Frame]): Frames in video order; only their players are read.
        """
        # The detection service's tracker is presumed accurate, so when it labelled every player
        # its ids are used as they are; matching boxes by overlap is only the fallback, since it
        # can swap two players who cross each other
        all_players = [p for f in frames for p in f.players]
        if all_players and all(p.tracking_id is not None for p in all_players):
            raw = self._group_by_id(frames)
        else:
            raw = self._match_by_overlap(frames)

        # Per track: fill short holes by interpolation, then smooth with a moving average
        self.tracks: List[Tuple[int, np.ndarray]] = []
        self.present: Dict[int, List[int]] = {}
        for track in self._split_at_long_holes(raw):
            idx = np.array(sorted(track))
            boxes = np.array([track[i] for i in idx])
            frames_range = np.arange(idx[0], idx[-1] + 1)
            filled = np.stack([np.interp(frames_range, idx, boxes[:, c]) for c in range(4)], axis=1)
            if len(filled) >= PLAYER_SMOOTHING:
                kernel = np.ones(PLAYER_SMOOTHING) / PLAYER_SMOOTHING
                pad = PLAYER_SMOOTHING // 2
                padded = np.pad(filled, ((pad, pad), (0, 0)), mode='edge')
                filled = np.stack([np.convolve(padded[:, c], kernel, mode='valid') for c in range(4)], axis=1)

            tid = len(self.tracks)
            self.tracks.append((int(idx[0]), filled))
            for i in idx:
                self.present.setdefault(int(i), []).append(tid)

    @staticmethod
    def _group_by_id(frames: List[Frame]) -> List[Dict[int, np.ndarray]]:
        """
        Groups player boxes into tracks by their tracking_id.

        Args:
            frames (List[Frame]): Frames in video order, every player carrying a tracking_id.

        Returns:
            List[Dict[int, np.ndarray]]: One {frame index: [x1, y1, x2, y2]} per tracking_id.
        """
        by_id: Dict[object, Dict[int, np.ndarray]] = {}
        for i, f in enumerate(frames):
            for p in f.players:
                by_id.setdefault(p.tracking_id, {})[i] = np.array([p.x1, p.y1, p.x2, p.y2], dtype=float)
        return list(by_id.values())

    @staticmethod
    def _match_by_overlap(frames: List[Frame]) -> List[Dict[int, np.ndarray]]:
        """
        Links player boxes into tracks by IoU with each track's last box (Hungarian matching).

        Args:
            frames (List[Frame]): Frames in video order.

        Returns:
            List[Dict[int, np.ndarray]]: One {frame index: [x1, y1, x2, y2]} per track.
        """
        raw: List[Dict[int, np.ndarray]] = []
        active: List[int] = []  # indices into raw

        for i, f in enumerate(frames):
            boxes = np.array([[p.x1, p.y1, p.x2, p.y2] for p in f.players], dtype=float).reshape(-1, 4)
            active = [t for t in active if i - max(raw[t]) <= PLAYER_MAX_MISS]
            matched = set()

            if active and len(boxes):
                last_boxes = np.array([raw[t][max(raw[t])] for t in active])
                iou = _iou_matrix(last_boxes, boxes)
                rows, cols = linear_sum_assignment(-iou)
                for r, c in zip(rows, cols):
                    if iou[r, c] >= PLAYER_IOU:
                        raw[active[r]][i] = boxes[c]
                        matched.add(c)

            for c in range(len(boxes)):
                if c not in matched:
                    raw.append({i: boxes[c]})
                    active.append(len(raw) - 1)
        return raw

    @staticmethod
    def _split_at_long_holes(raw: List[Dict[int, np.ndarray]]) -> List[Dict[int, np.ndarray]]:
        """
        Splits each track where the player is missing for more than PLAYER_MAX_MISS frames.

        Short holes are later filled by interpolation, but a player who left the picture and
        came back must not get invented boxes for the time they were away: the ball could be
        placed in a box that never existed.

        Args:
            raw (List[Dict[int, np.ndarray]]): Tracks as {frame index: box}.

        Returns:
            List[Dict[int, np.ndarray]]: The tracks, each holding no hole longer than PLAYER_MAX_MISS.
        """
        parts = []
        for track in raw:
            part: Dict[int, np.ndarray] = {}
            last = None
            for i in sorted(track):
                if last is not None and i - last > PLAYER_MAX_MISS:
                    parts.append(part)
                    part = {}
                part[i] = track[i]
                last = i
            parts.append(part)
        return parts

    def box(self, tid: int, i: int) -> Optional[np.ndarray]:
        """
        Gets a player's smoothed box in a frame.

        Args:
            tid (int): Player track id.
            i (int): Frame index.

        Returns:
            Optional[np.ndarray]: [x1, y1, x2, y2], or None if the track doesn't cover frame i.
        """
        start, boxes = self.tracks[tid]
        if start <= i < start + len(boxes):
            return boxes[i - start]
        return None

    def owner(self, i: int, center: np.ndarray, diameter: float) -> Optional[int]:
        """
        Finds the player holding the ball: the closest box (relative to its size) that contains it.

        Args:
            i (int): Frame index.
            center (np.ndarray): Ball center [x, y] in image coordinates.
            diameter (float): Ball diameter, which sets how far outside a box still counts.

        Returns:
            Optional[int]: Track id of the holder, or None if no player box contains the ball.
        """
        best, best_score = None, np.inf
        for tid in self.present.get(i, []):
            x1, y1, x2, y2 = self.box(tid, i)
            margin = PLAYER_MARGIN * diameter
            if not (x1 - margin <= center[0] <= x2 + margin and y1 - margin <= center[1] <= y2 + margin):
                continue
            score = abs(center[0] - (x1 + x2) / 2) / (x2 - x1) + abs(center[1] - (y1 + y2) / 2) / (y2 - y1)
            if score < best_score:
                best, best_score = tid, score
        return best

    def moves_with(self, tid: int, idx: np.ndarray, centers: np.ndarray, stable: np.ndarray) -> bool:
        """
        Checks whether the ball moves with the player around a gap.

        It does when its path relative to the player's box is steadier than its own
        (camera-stabilized) path, which is what the Kalman smoother assumes.

        Args:
            tid (int): Player track id.
            idx (np.ndarray): Frame indices of the detections around the gap.
            centers (np.ndarray): (n, 2) ball centers in image coordinates.
            stable (np.ndarray): (n, 2) ball centers in camera-stabilized coordinates.

        Returns:
            bool: True if the ball follows the player. False with fewer than 3 usable
                detections, since there's nothing to compare.
        """
        idx = np.array([i for i in idx if self.box(tid, i) is not None])
        if len(idx) < 3:
            return False
        boxes = np.array([self.box(tid, i) for i in idx])
        box_centers = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
        t = idx.astype(float)
        return _linear_residual(t, centers[idx] - box_centers) < _linear_residual(t, stable[idx])

    def relative(self, tid: int, i: int, center: np.ndarray) -> np.ndarray:
        """
        Expresses a point relative to a player's box.

        Args:
            tid (int): Player track id; its track must cover frame i.
            i (int): Frame index.
            center (np.ndarray): Point [x, y] in image coordinates.

        Returns:
            np.ndarray: [u, v] where (0, 0) is the box's top-left and (1, 1) its bottom-right.
        """
        x1, y1, x2, y2 = self.box(tid, i)
        return np.array([(center[0] - x1) / (x2 - x1), (center[1] - y1) / (y2 - y1)])

    def absolute(self, tid: int, i: int, offset: np.ndarray) -> Optional[np.ndarray]:
        """
        Converts a box-relative offset back to image coordinates (the inverse of `relative`).

        Args:
            tid (int): Player track id.
            i (int): Frame index.
            offset (np.ndarray): [u, v] relative to the player's box.

        Returns:
            Optional[np.ndarray]: Point [x, y] in image coordinates, or None if the track
                doesn't cover frame i.
        """
        box = self.box(tid, i)
        if box is None:
            return None
        x1, y1, x2, y2 = box
        return np.array([x1 + offset[0] * (x2 - x1), y1 + offset[1] * (y2 - y1)])

    def interpolate(self, tid: int, a: int, b: int, center_a: np.ndarray, center_b: np.ndarray) -> Optional[np.ndarray]:
        """
        Moves the ball with the player, blending its offset inside the player's box from a to b.

        Args:
            tid (int): Track id of the player holding the ball.
            a (int): Frame of the detection before the gap.
            b (int): Frame of the detection after the gap.
            center_a (np.ndarray): Ball center [x, y] at frame a.
            center_b (np.ndarray): Ball center [x, y] at frame b.

        Returns:
            Optional[np.ndarray]: (b - a - 1, 2) ball centers for the gap frames, or None if
                the player's track doesn't cover the whole gap.
        """
        offset_a = self.relative(tid, a, center_a)
        offset_b = self.relative(tid, b, center_b)
        path = []
        for i in range(a + 1, b):
            t = (i - a) / (b - a)
            pos = self.absolute(tid, i, (1 - t) * offset_a + t * offset_b)
            if pos is None:
                return None
            path.append(pos)
        return np.array(path)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes the intersection-over-union of every pair of boxes.

    Args:
        a (np.ndarray): (m, 4) boxes as [x1, y1, x2, y2].
        b (np.ndarray): (k, 4) boxes as [x1, y1, x2, y2].

    Returns:
        np.ndarray: (m, k) IoU values between 0 and 1.
    """
    x1 =np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _interpolate_sizes(n: int, anchors: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """
    Computes a ball box size for every frame.

    Sizes are linear between anchors, using a short median over neighbouring anchors to
    remove detection jitter.

    Args:
        n (int): Number of frames in the video.
        anchors (np.ndarray): Sorted frame indices of the accepted detections.
        sizes (np.ndarray): (n, 2) box width/height per frame.

    Returns:
        np.ndarray: (n, 2) box width/height for every frame.
    """
    smoothed = np.array([np.median(sizes[anchors[max(0, k - 1):k + 2]], axis=0) for k in range(len(anchors))])
    frame_idx = np.arange(n)
    return np.stack([np.interp(frame_idx, anchors, smoothed[:, c]) for c in range(2)], axis=1)


def _make_ball(center: np.ndarray, size: np.ndarray, frame_size: Optional[Tuple[int, int]]) -> Ball:
    """
    Builds an estimated Ball from a center and box size.

    Args:
        center (np.ndarray): Ball center [x, y] in image coordinates.
        size (np.ndarray): Box [width, height].
        frame_size (Optional[Tuple[int, int]]): (width, height) of the video. When given, the
            center and box are clamped inside the frame.

    Returns:
        Ball: The estimate, with confidence 0.0 to mark it as not detected.
    """
    cx, cy = float(center[0]), float(center[1])
    w, h = float(size[0]), float(size[1])
    if frame_size is not None:
        cx = min(max(cx, 0.0), float(frame_size[0]))
        cy = min(max(cy, 0.0), float(frame_size[1]))
    x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x2, y2 = cx + w / 2, cy + h / 2
    if frame_size is not None:
        x2, y2 = min(x2, float(frame_size[0])), min(y2, float(frame_size[1]))
    return Ball(confidence=0.0, x1=x1, y1=y1, x2=x2, y2=y2)
