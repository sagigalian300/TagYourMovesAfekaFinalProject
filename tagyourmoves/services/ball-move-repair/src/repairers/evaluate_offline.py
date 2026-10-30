"""
Evaluation harness for the offline ball estimator.

There is no labelled ground truth, so we make our own from the video: take stretches
where YOLO saw the ball confidently, hide them, run the full pipeline (optical-flow hints
+ offline estimator) and compare the estimates against the detections we hid.

Usage:
    python evaluate_offline.py                      # both test videos, 5 trials each
    python evaluate_offline.py --videos test_vod2.mp4 --trials 10 --no-lk
"""
import os
import argparse
import random
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2

from Frame import Frame
from Offline import estimate_missing_balls_offline
from test_offline import (detect_video, load_detections, apply_optical_flow, default_cache_path,
                          camera_motion_for, default_camera_cache_path, DEFAULT_MODEL, DEFAULT_IMGSZ)


@dataclass
class VideoInfo:
    frame_size: Tuple[int, int]
    camera_motion: Optional[np.ndarray]


Estimator = Callable[[List[Frame], VideoInfo], List[Frame]]
HintPass = Callable[[str, List[Frame]], List[Frame]]

GAP_BUCKETS = [(3, 7), (8, 15), (16, 30), (31, 45)]


def _center_and_size(ball) -> Tuple[float, float, float]:
    return (ball.x1 + ball.x2) / 2, (ball.y1 + ball.y2) / 2, ((ball.x2 - ball.x1) + (ball.y2 - ball.y1)) / 2


def sample_masks(
    frames: List[Frame],
    rng: random.Random,
    gap_range: Tuple[int, int] = (3, 30),
    min_gt_conf: float = 0.4,
    min_gt_ratio: float = 0.6,
    spacing: int = 10,
    masks_per_100_frames: float = 2.0,
) -> List[Tuple[int, int]]:
    """
    Picks non-overlapping windows [start, end) to hide. A window is valid when both of its
    neighbours hold a detection (so the gap has anchors on both sides, like a real occlusion),
    at least `min_gt_ratio` of its frames have a confident detection to score against, and all
    those detections form one continuous track. YOLO's ball class also fires on other objects,
    and scoring against those would make the ground truth itself wrong.
    """
    n = len(frames)
    confident = np.array([f.ball is not None and f.ball.confidence >= min_gt_conf for f in frames])
    has_ball = np.array([f.ball is not None for f in frames])

    target = max(1, int(n / 100 * masks_per_100_frames))
    taken = np.zeros(n, dtype=bool)
    masks = []
    starts = list(range(1, n - gap_range[0] - 1))
    rng.shuffle(starts)

    for start in starts:
        if len(masks) >= target:
            break
        length = rng.randint(*gap_range)
        end = start + length
        if end >= n or not has_ball[start - 1] or not has_ball[end]:
            continue
        if confident[start:end].mean() < min_gt_ratio:
            continue
        if not _is_continuous_track(frames, start - 1, end):
            continue
        lo, hi = max(0, start - spacing), min(n, end + spacing)
        if taken[lo:hi].any():
            continue
        taken[start:end] = True
        masks.append((start, end))

    return sorted(masks)


def _is_continuous_track(frames: List[Frame], first: int, last: int,
                         max_speed: float = 2.5, max_size_ratio: float = 1.5) -> bool:
    """Whether the detections in frames[first..last] plausibly belong to one moving ball."""
    dets = [(i, _center_and_size(frames[i].ball)) for i in range(first, last + 1) if frames[i].ball is not None]
    for (i, (x1, y1, s1)), (j, (x2, y2, s2)) in zip(dets[:-1], dets[1:]):
        if max(s1, s2) / max(min(s1, s2), 1.0) > max_size_ratio:
            return False
        if np.hypot(x2 - x1, y2 - y1) > (s1 + s2) / 2 * (max_speed * (j - i) + 1.0):
            return False
    return True


def run_trial(
    video_path: str,
    cache_path: str,
    estimator: Estimator,
    hint_pass: Optional[HintPass],
    rng: random.Random,
    info: VideoInfo,
    gap_range: Tuple[int, int],
    min_gt_conf: float,
) -> List[Dict]:
    frames = load_detections(cache_path)
    masks = sample_masks(frames, rng, gap_range=gap_range, min_gt_conf=min_gt_conf)

    ground_truth = {}
    for start, end in masks:
        for i in range(start, end):
            b = frames[i].ball
            if b is not None and b.confidence >= min_gt_conf:
                ground_truth[i] = (_center_and_size(b), end - start)
            frames[i].ball = None

    if hint_pass is not None:
        frames = hint_pass(video_path, frames)
    frames = estimator(frames, info)

    results = []
    for i, ((gx, gy, gsize), gap_len) in ground_truth.items():
        est = frames[i].ball
        if est is None:
            results.append({'gap': gap_len, 'err': None, 'size': gsize})
        else:
            ex, ey, _ = _center_and_size(est)
            results.append({'gap': gap_len, 'err': float(np.hypot(ex - gx, ey - gy)), 'size': gsize})
    return results


def summarize(results: List[Dict]) -> Dict:
    if not results:
        return {'n': 0}
    filled = [r for r in results if r['err'] is not None]
    errs = np.array([r['err'] for r in filled]) if filled else np.array([np.nan])
    norm = np.array([r['err'] / max(r['size'], 1.0) for r in filled]) if filled else np.array([np.nan])
    return {
        'n': len(results),
        'coverage': len(filled) / len(results),
        'mean_px': float(np.nanmean(errs)),
        'median_px': float(np.nanmedian(errs)),
        'p90_px': float(np.nanpercentile(errs, 90)),
        # Share of *all* hidden frames estimated within one ball diameter (misses count as failures)
        'within_1d': float(np.sum(norm <= 1.0) / len(results)),
    }


def format_summary(name: str, s: Dict) -> str:
    if s['n'] == 0:
        return f"{name:<14} n=0"
    return (f"{name:<14} n={s['n']:<5} coverage={s['coverage']:6.1%}  mean={s['mean_px']:6.1f}px  "
            f"median={s['median_px']:6.1f}px  p90={s['p90_px']:6.1f}px  within 1 ball={s['within_1d']:6.1%}")


def evaluate(
    video_path: str,
    model_path: str,
    estimator: Estimator,
    hint_pass: Optional[HintPass] = apply_optical_flow,
    trials: int = 5,
    seed: int = 0,
    gap_range: Tuple[int, int] = (3, 30),
    min_gt_conf: float = 0.4,
    verbose: bool = True,
    imgsz: Optional[int] = None,
) -> Dict:
    cache_path = default_cache_path(video_path, model_path, imgsz)
    frames = detect_video(video_path, model_path, cache_path, imgsz=imgsz)

    cap = cv2.VideoCapture(video_path)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    # Camera motion only depends on the background and the player boxes, which masking doesn't touch
    info = VideoInfo(frame_size, camera_motion_for(video_path, frames, default_camera_cache_path(video_path)))

    results = []
    for t in range(trials):
        rng = random.Random(seed * 1000 + t)
        results += run_trial(video_path, cache_path, estimator, hint_pass, rng, info, gap_range, min_gt_conf)

    overall = summarize(results)
    if verbose:
        print(f"\n== {os.path.basename(video_path)} ({trials} trials) ==")
        print(format_summary('all gaps', overall))
        for lo, hi in GAP_BUCKETS:
            bucket = [r for r in results if lo <= r['gap'] <= hi]
            if bucket:
                print(format_summary(f'gap {lo}-{hi}', summarize(bucket)))
    return {'overall': overall, 'results': results}


def default_estimator(max_gap_frames: int = 45, **options) -> Estimator:
    return lambda frames, info: estimate_missing_balls_offline(
        frames, max_gap_frames=max_gap_frames, frame_size=info.frame_size, camera_motion=info.camera_motion, **options
    )


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Evaluate the offline ball estimator on artificially hidden detections.")
    parser.add_argument('--videos', nargs='+', default=['test_vod.mp4', 'test_vod2.mp4'])
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--imgsz', type=int, default=None,
                        help=f"Inference size (default: {DEFAULT_IMGSZ} for the default model, the model's own otherwise)")
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-gap', type=int, default=45)
    parser.add_argument('--no-lk', action='store_true', help="Skip the Lucas-Kanade hint pass")
    args = parser.parse_args()

    for video in args.videos:
        evaluate(
            os.path.join(script_dir, video),
            os.path.join(script_dir, args.model),
            default_estimator(args.max_gap),
            hint_pass=None if args.no_lk else apply_optical_flow,
            trials=args.trials,
            seed=args.seed,
            imgsz=args.imgsz or (DEFAULT_IMGSZ if args.model == DEFAULT_MODEL else None),
        )
