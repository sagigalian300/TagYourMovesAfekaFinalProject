import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from Frame import Frame
from Ball import Ball
from CameraMotion import to_reference, from_reference

# All distances below are expressed in ball diameters, so they scale with zoom / resolution.
MAX_SPEED = 2.5             # Max plausible ball motion between consecutive frames
MAX_SIZE_RATIO = 2.0        # Two detections whose sizes differ more than this aren't the same ball
TRACKLET_GAP = 5            # Max frames between detections linked into the same tracklet
DETECTION_NOISE = 0.1       # Std of a YOLO detection's center
HINT_NOISE = 0.1            # Std of an optical-flow hint, multiplied by (1 + frames since detection)
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
    use_hints: bool = True,
) -> List[Frame]:
    """
    Estimates the ball's location in frames where it is obscured or was not detected.

    Input convention: a ball with confidence > 0 is a detection; a ball with confidence 0.0
    is a weak hint from an earlier stage (e.g. Lucas-Kanade optical flow). Every estimate
    written back has confidence 0.0.

    Each gap between two trusted detections is filled using both of its ends:
      * possession - the same player holds the ball on both sides and the ball moves with
                     that player around the gap: the ball follows the player's box
      * otherwise  - a Kalman RTS smoother over the detections (and optical-flow hints)
    Gaps longer than `max_gap_frames` are left empty apart from a short stretch near each end.

    Args:
        frames: Frames in video order.
        max_gap_frames: Longest gap that is filled end to end.
        frame_size: (width, height); when given, estimates are clamped inside the frame.
        camera_motion: Per-frame transforms from CameraMotion.estimate_camera_motion. When given,
            ball motion is modelled in camera-stabilized coordinates, so pans don't look like ball motion.
        use_possession / use_hints: Switch individual stages off (for evaluation).
    """
    n = len(frames)
    if n == 0:
        return frames

    # =================================================================
    # STEP 1: Extract detections and hints
    # =================================================================
    centers = np.full((n, 2), np.nan)
    sizes = np.full((n, 2), np.nan)
    is_detection = np.zeros(n, dtype=bool)
    is_hint = np.zeros(n, dtype=bool)

    for i, f in enumerate(frames):
        b = f.ball
        if b is None:
            continue
        centers[i] = ((b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2)
        sizes[i] = (b.x2 - b.x1, b.y2 - b.y1)
        if b.confidence > 0.0:
            is_detection[i] = True
        else:
            is_hint[i] = True

    if not is_detection.any():
        return frames

    # Ball motion is modelled in camera-stabilized coordinates when the camera motion is known
    stable = centers.copy()
    if camera_motion is not None:
        for i in np.flatnonzero(is_detection | is_hint):
            stable[i] = to_reference(camera_motion, i, centers[i:i + 1])[0]

    # =================================================================
    # STEP 2: Keep only detections that belong to the ball's track (not heads, lights, a 2nd ball)
    # =================================================================
    confidences = np.array([f.ball.confidence if f.ball is not None else 0.0 for f in frames])
    anchors = _select_ball_track(stable, sizes, confidences, np.flatnonzero(is_detection), max_gap_frames)
    is_anchor = np.zeros(n, dtype=bool)
    is_anchor[anchors] = True

    # Ball diameter for every frame, interpolated between anchors
    anchor_diameters = sizes[anchors].mean(axis=1)
    diameters = np.interp(np.arange(n), anchors, anchor_diameters)

    # =================================================================
    # STEP 3: Track players so "the same player" means something across frames
    # =================================================================
    players = _PlayerTracks(frames) if use_possession else None

    # =================================================================
    # STEP 4: Kalman RTS smoother over anchors (+ hints) as the general fallback
    # =================================================================
    # Optical flow ran before track selection, so drop hints that continue a rejected detection
    last_detection = -1
    for i in range(n):
        if is_detection[i]:
            last_detection = i
        elif is_hint[i] and (not use_hints or last_detection < 0 or not is_anchor[last_detection]):
            is_hint[i] = False

    hint_idx = np.flatnonzero(is_hint)
    smoothed = _kalman_smooth(n, anchors, hint_idx, stable, diameters, max_gap_frames)
    if camera_motion is not None:
        for i in np.flatnonzero(np.isfinite(smoothed[:, 0])):
            smoothed[i] = from_reference(camera_motion, i, smoothed[i:i + 1])[0]

    # =================================================================
    # STEP 5: Fill every gap
    # =================================================================
    estimates: Dict[int, np.ndarray] = {}

    def owner(i: int) -> Optional[int]:
        return players.owner(i, centers[i], diameters[i]) if players is not None else None

    def follow_owner(anchor: int, frame_range: range):
        # One-sided gap: keep the ball at a fixed spot relative to whoever held it
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
    # STEP 6: Write estimates back to the frames
    # =================================================================
    box_sizes = _interpolate_sizes(n, anchors, sizes)

    for i, f in enumerate(frames):
        if is_anchor[i]:
            continue
        if i in estimates and np.all(np.isfinite(estimates[i])):
            f.ball = _make_ball(estimates[i], box_sizes[i], frame_size)
        elif is_hint[i]:
            continue  # nothing better than the optical-flow hint (e.g. inside a long gap)
        else:
            f.ball = None

    return frames


def _select_ball_track(
    centers: np.ndarray,
    sizes: np.ndarray,
    confidences: np.ndarray,
    detections: np.ndarray,
    max_gap_frames: int,
) -> np.ndarray:
    """
    YOLO's ball class also fires on other objects, sometimes for many frames in a row, so not
    every detection is the ball.
      1. Split detections into tracklets: runs whose consecutive detections are physically
         consistent (plausible speed, similar size).
      2. Pick the chain of tracklets with the highest total confidence in which every jump between
         consecutive tracklets is also physically possible (longest path in a DAG, by dynamic
         programming). After a gap longer than max_gap_frames anything is possible.
    """
    diameters = sizes.mean(axis=1)

    def consistent(i: int, j: int) -> bool:
        if max(diameters[i], diameters[j]) / min(diameters[i], diameters[j]) > MAX_SIZE_RATIO:
            return False
        diameter = (diameters[i] + diameters[j]) / 2
        return np.hypot(*(centers[j] - centers[i])) <= diameter * (MAX_SPEED * abs(j - i) + 1.0)

    tracklets = [[int(detections[0])]]
    for i in detections[1:]:
        last = tracklets[-1][-1]
        if i - last <= TRACKLET_GAP and consistent(last, i):
            tracklets[-1].append(int(i))
        else:
            tracklets.append([int(i)])

    score = [float(confidences[t].sum()) for t in tracklets]
    best = list(score)
    parent = [-1] * len(tracklets)
    for k in range(len(tracklets)):
        for j in range(k):
            end, start = tracklets[j][-1], tracklets[k][0]
            if (start - end > max_gap_frames + 1 or consistent(end, start)) and best[j] + score[k] > best[k]:
                best[k] = best[j] + score[k]
                parent[k] = j

    chain = []
    k = int(np.argmax(best))
    while k != -1:
        chain.extend(tracklets[k])
        k = parent[k]
    return np.array(sorted(chain), dtype=int)


def _kalman_smooth(
    n: int,
    anchors: np.ndarray,
    hints: np.ndarray,
    centers: np.ndarray,
    diameters: np.ndarray,
    max_gap_frames: int,
) -> np.ndarray:
    """
    Constant-velocity Kalman filter + RTS smoother over the ball center. Runs separately on each
    segment of the video (segments are split at gaps longer than max_gap_frames) so no velocity
    is carried across a long gap. Returns an (n, 2) array, NaN outside the segments.
    """
    out = np.full((n, 2), np.nan)
    is_anchor = np.zeros(n, dtype=bool)
    is_anchor[anchors] = True
    is_hint = np.zeros(n, dtype=bool)
    is_hint[hints] = True

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
        last_anchor = start
        for i in range(start, end + 1):
            kf.predict()
            if is_anchor[i]:
                kf.update(centers[i], R=np.eye(2) * (DETECTION_NOISE * diameters[i]) ** 2)
                last_anchor = i
            elif is_hint[i]:
                # Optical flow drifts, so trust each hint less the further it is from a detection
                age = i - last_anchor
                kf.update(centers[i], R=np.eye(2) * (HINT_NOISE * diameters[i] * (1 + age)) ** 2)
            mu.append(kf.x.copy())
            cov.append(kf.P.copy())

        M, _, _, _ = kf.rts_smoother(np.array(mu), np.array(cov))
        out[start:end + 1, 0] = M[:, 0, 0]
        out[start:end + 1, 1] = M[:, 2, 0]

    return out


def _linear_residual(t: np.ndarray, points: np.ndarray) -> float:
    """RMS distance of points from the best constant-velocity line through them."""
    fit = np.stack([np.polyval(np.polyfit(t, points[:, c], 1), t) for c in range(2)], axis=1)
    return float(np.sqrt(np.mean(np.sum((fit - points) ** 2, axis=1))))


class _PlayerTracks:
    """
    Links player boxes across frames by IoU (Hungarian matching) and smooths each track.
    Kept internal so the Player objects themselves are not modified.
    """

    def __init__(self, frames: List[Frame]):
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

        # Per track: fill short holes by interpolation, then smooth with a moving average
        self.tracks: List[Tuple[int, np.ndarray]] = []
        self.present: Dict[int, List[int]] = {}
        for track in raw:
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

    def box(self, tid: int, i: int) -> Optional[np.ndarray]:
        start, boxes = self.tracks[tid]
        if start <= i < start + len(boxes):
            return boxes[i - start]
        return None

    def owner(self, i: int, center: np.ndarray, diameter: float) -> Optional[int]:
        """The player holding the ball in frame i: the closest box (relative to its size) that contains it."""
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
        Whether the ball moves with the player around a gap: its path relative to the player's box
        is steadier than its own (camera-stabilized) path, which is what the Kalman smoother assumes.
        With fewer than 3 detections there's nothing to compare, so the answer is no.
        """
        idx = np.array([i for i in idx if self.box(tid, i) is not None])
        if len(idx) < 3:
            return False
        boxes = np.array([self.box(tid, i) for i in idx])
        box_centers = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
        t = idx.astype(float)
        return _linear_residual(t, centers[idx] - box_centers) < _linear_residual(t, stable[idx])

    def relative(self, tid: int, i: int, center: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = self.box(tid, i)
        return np.array([(center[0] - x1) / (x2 - x1), (center[1] - y1) / (y2 - y1)])

    def absolute(self, tid: int, i: int, offset: np.ndarray) -> Optional[np.ndarray]:
        box = self.box(tid, i)
        if box is None:
            return None
        x1, y1, x2, y2 = box
        return np.array([x1 + offset[0] * (x2 - x1), y1 + offset[1] * (y2 - y1)])

    def interpolate(self, tid: int, a: int, b: int, center_a: np.ndarray, center_b: np.ndarray) -> Optional[np.ndarray]:
        """Moves the ball with the player, blending its offset inside the player's box from a to b."""
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
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _interpolate_sizes(n: int, anchors: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Box width/height for every frame: linear between anchors, using a short median to remove detection jitter."""
    smoothed = np.array([np.median(sizes[anchors[max(0, k - 1):k + 2]], axis=0) for k in range(len(anchors))])
    frame_idx = np.arange(n)
    return np.stack([np.interp(frame_idx, anchors, smoothed[:, c]) for c in range(2)], axis=1)


def _make_ball(center: np.ndarray, size: np.ndarray, frame_size: Optional[Tuple[int, int]]) -> Ball:
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
