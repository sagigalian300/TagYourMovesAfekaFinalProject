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
    # STEP 1: Handle Player Possession (Occlusion Anchoring)
    # =================================================================
    last_known_ball_center = None
    player_holding_ball = False
    last_player_center = None

    for f in frames:
        if f.ball is not None:
            bx = (f.ball.x1 + f.ball.x2) / 2
            by = (f.ball.y1 + f.ball.y2) / 2
            
            player_holding_ball = False
            for p in f.players:
                if p.x1 - 15 <= bx <= p.x2 + 15 and p.y1 - 15 <= by <= p.y2 + 15:
                    player_holding_ball = True
                    last_player_center = ((p.x1 + p.x2) / 2, (p.y1 + p.y2) / 2)
                    break
        else:
            if player_holding_ball and last_player_center is not None:
                best_dist = float('inf')
                best_player = None
                for p in f.players:
                    px = (p.x1 + p.x2) / 2
                    py = (p.y1 + p.y2) / 2
                    dist = (px - last_player_center[0])**2 + (py - last_player_center[1])**2
                    if dist < best_dist:
                        best_dist = dist
                        best_player = p
                
                if best_player and best_dist < 15000: 
                    px = (best_player.x1 + best_player.x2) / 2
                    py = (best_player.y1 + best_player.y2) / 2
                    last_player_center = (px, py)
                    
                    bw, bh = 25, 25 
                    f.ball = Ball(
                        confidence=0.0, 
                        x1=px - bw/2, y1=py - bh/2, 
                        x2=px + bw/2, y2=py + bh/2
                    )

    # =================================================================
    # STEP 2: Extract data for the Kalman Filter
    # =================================================================
    data = []
    measurements = []
    for f in frames:
        if f.ball is not None:
            data.append({'frame_number': f.frame_number, 'x1': f.ball.x1, 'y1': f.ball.y1})
            
            # We track the center coordinates and dimensions for smoother physics
            bx = (f.ball.x1 + f.ball.x2) / 2
            by = (f.ball.y1 + f.ball.y2) / 2
            bw = f.ball.x2 - f.ball.x1
            bh = f.ball.y2 - f.ball.y1
            measurements.append([bx, by, bw, bh])
        else:
            data.append({'frame_number': f.frame_number, 'x1': None, 'y1': None})
            measurements.append(None)

    df = pd.DataFrame(data).set_index('frame_number')
    if df['x1'].isna().all():
        return frames

    # =================================================================
    # STEP 3: Kalman Filter Forward Pass & RTS Backward Smoother
    # =================================================================
    # State Vector: [x, y, w, h, velocity_x, velocity_y, velocity_w, velocity_h]
    kf = KalmanFilter(dim_x=8, dim_z=4)
    dt = 1.0 # 1 frame timestep

    # State Transition Matrix (how physics progresses frame-to-frame)
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

    # Measurement Matrix (how to extract x,y,w,h from the state vector)
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0]
    ])

    # Tuning Matrices
    kf.P *= 1000.          # Initial uncertainty
    kf.R *= 5.0            # Measurement noise (accounts for YOLO bounding box jitter)
    kf.Q = np.eye(8) * 0.1 # Process noise
    kf.Q[4:, 4:] *= 10.0   # Higher noise for velocity since the ball changes speed/direction

    # Initialize state with the first valid measurement
    first_valid_idx = next(i for i, m in enumerate(measurements) if m is not None)
    kf.x[:4, 0] = measurements[first_valid_idx]
    
    mu, cov = [], []
    for m in measurements:
        kf.predict()
        if m is not None:
            kf.update(m) # Correct prediction using actual YOLO detection
        
        # Save states for the time-traveling backward pass
        mu.append(kf.x.copy())
        cov.append(kf.P.copy())

    # Run the RTS Smoother (smoothes the trajectory from future to past)
    M, _, _, _ = kf.rts_smoother(np.array(mu), np.array(cov))

    # Convert smoothed states [x, y, w, h] back to bounding box [x1, y1, x2, y2]
    df['smooth_x1'] = M[:, 0, 0] - M[:, 2, 0] / 2
    df['smooth_y1'] = M[:, 1, 0] - M[:, 3, 0] / 2
    df['smooth_x2'] = M[:, 0, 0] + M[:, 2, 0] / 2
    df['smooth_y2'] = M[:, 1, 0] + M[:, 3, 0] / 2

    # =================================================================
    # STEP 4: Gap Validation & Object Reconstruction
    # =================================================================
    # Accurately identify gaps that are <= max_gap_frames
    missing = df['x1'].isna()
    gap_ids = (~missing).cumsum()
    gap_counts = gap_ids[missing].map(gap_ids[missing].value_counts())
    
    valid_gap_mask = pd.Series(False, index=df.index)
    valid_gap_mask[missing] = gap_counts <= max_gap_frames

    for i, f in enumerate(frames):
        if f.ball is None and valid_gap_mask.iloc[i]:
            row = df.iloc[i]
            
            # Clamp coordinates to ensure bounding boxes don't shoot off-screen
            x1 = max(0.0, float(row['smooth_x1']))
            y1 = max(0.0, float(row['smooth_y1']))
            x2 = max(0.0, float(row['smooth_x2']))
            y2 = max(0.0, float(row['smooth_y2']))

            f.ball = Ball(confidence=0.0, x1=x1, y1=y1, x2=x2, y2=y2)

    return frames