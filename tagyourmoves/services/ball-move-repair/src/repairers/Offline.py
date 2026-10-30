import numpy as np
import pandas as pd
from typing import List
from filterpy.kalman import KalmanFilter

from Frame import Frame
from Ball import Ball

def estimate_missing_balls_offline(frames: List[Frame], max_gap_frames: int = 15) -> List[Frame]:
    if not frames:
        return frames

    # =================================================================
    # STEP 1: Extract data for the Kalman Filter
    # (Player anchoring is removed; we rely on LK Flow + YOLO detections)
    # =================================================================
    data = []
    measurements = []
    confidences = [] 
    
    for f in frames:
        if f.ball is not None:
            data.append({'frame_number': f.frame_number, 'x1': f.ball.x1, 'y1': f.ball.y1})
            bx = (f.ball.x1 + f.ball.x2) / 2
            by = (f.ball.y1 + f.ball.y2) / 2
            bw = f.ball.x2 - f.ball.x1
            bh = f.ball.y2 - f.ball.y1
            measurements.append([bx, by, bw, bh])
            confidences.append(f.ball.confidence)
        else:
            data.append({'frame_number': f.frame_number, 'x1': None, 'y1': None})
            measurements.append(None)
            confidences.append(None)

    df = pd.DataFrame(data).set_index('frame_number')
    if df['x1'].isna().all():
        return frames

    # =================================================================
    # STEP 2: Kalman Filter Forward Pass & RTS Backward Smoother
    # =================================================================
    kf = KalmanFilter(dim_x=8, dim_z=4)
    dt = 1.0

    kf.F = np.array([
        [1, 0, 0, 0, dt, 0,  0,  0],
        [0, 1, 0, 0,  0, dt, 0,  0],
        [0, 0, 1, 0,  0, 0, dt,  0],
        [0, 0, 0, 1,  0, 0,  0, dt],
        [0, 0, 0, 0,  1, 0,  0,  0],
        [0, 0, 0, 0,  0, 1,  0,  0],
        [0, 0, 0, 0,  0, 0,  1,  0],
        [0, 0, 0, 0,  0, 0,  0,  1]
    ])

    kf.H = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0]
    ])

    kf.P *= 1000.          
    kf.Q = np.eye(8) * 0.1 
    kf.Q[4:, 4:] *= 10.0   

    first_valid_idx = next(i for i, m in enumerate(measurements) if m is not None)
    kf.x[:4, 0] = measurements[first_valid_idx]
    
    mu, cov = [], []
    for i, m in enumerate(measurements):
        kf.predict()
        if m is not None:
            # DYNAMIC NOISE: 
            # High trust for YOLO detections (noise = 5)
            # Medium trust for Lucas-Kanade estimates (noise = 50) allows Kalman to smooth LK jitters
            noise_level = 5.0 if confidences[i] > 0.0 else 50.0
            kf.R = np.eye(4) * noise_level
            
            kf.update(m)
        
        mu.append(kf.x.copy())
        cov.append(kf.P.copy())

    M, _, _, _ = kf.rts_smoother(np.array(mu), np.array(cov))

    df['smooth_x1'] = M[:, 0, 0] - M[:, 2, 0] / 2
    df['smooth_y1'] = M[:, 1, 0] - M[:, 3, 0] / 2
    df['smooth_x2'] = M[:, 0, 0] + M[:, 2, 0] / 2
    df['smooth_y2'] = M[:, 1, 0] + M[:, 3, 0] / 2

    # =================================================================
    # STEP 3: Apply Smoothing and Bridge Complete Gaps
    # =================================================================
    missing = df['x1'].isna()
    gap_ids = (~missing).cumsum()
    gap_counts = gap_ids[missing].map(gap_ids[missing].value_counts())
    
    valid_gap_mask = pd.Series(False, index=df.index)
    valid_gap_mask[missing] = gap_counts <= max_gap_frames

    for i, f in enumerate(frames):
        # 1. Fill in pure gaps (where both YOLO and LK Tracker failed)
        if f.ball is None and valid_gap_mask.iloc[i]:
            row = df.iloc[i]
            x1 = max(0.0, float(row['smooth_x1']))
            y1 = max(0.0, float(row['smooth_y1']))
            x2 = max(0.0, float(row['smooth_x2']))
            y2 = max(0.0, float(row['smooth_y2']))

            f.ball = Ball(confidence=0.0, x1=x1, y1=y1, x2=x2, y2=y2)
            
        # 2. Smooth out the Lucas-Kanade estimates
        elif f.ball is not None and f.ball.confidence == 0.0:
            row = df.iloc[i]
            f.ball.x1 = max(0.0, float(row['smooth_x1']))
            f.ball.y1 = max(0.0, float(row['smooth_y1']))
            f.ball.x2 = max(0.0, float(row['smooth_x2']))
            f.ball.y2 = max(0.0, float(row['smooth_y2']))

    return frames