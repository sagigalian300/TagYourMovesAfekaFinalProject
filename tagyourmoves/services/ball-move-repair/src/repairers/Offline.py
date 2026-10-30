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
    # STEP 1: Player Displacement Vector Tracking
    # Move missing balls using the holding player's delta (dx, dy)
    # =================================================================
    last_player_pos = None
    holding_player_ref = None

    for f in frames:
        if f.ball is not None:
            bx = (f.ball.x1 + f.ball.x2) / 2
            by = (f.ball.y1 + f.ball.y2) / 2
            
            holding_player_ref = None
            last_player_pos = None
            
            # Check if ball is near/held by a player
            for p in f.players:
                if p.x1 - 20 <= bx <= p.x2 + 20 and p.y1 - 20 <= by <= p.y2 + 20:
                    holding_player_ref = p
                    px = (p.x1 + p.x2) / 2
                    py = (p.y1 + p.y2) / 2
                    last_player_pos = (px, py)
                    break
        else:
            # Ball is missing (YOLO + LK failed)
            if holding_player_ref is not None and last_player_pos is not None:
                # Find the same player in the current frame (closest to last position)
                best_dist = float('inf')
                curr_player = None
                
                for p in f.players:
                    px = (p.x1 + p.x2) / 2
                    py = (p.y1 + p.y2) / 2
                    dist = (px - last_player_pos[0])**2 + (py - last_player_pos[1])**2
                    if dist < best_dist:
                        best_dist = dist
                        curr_player = p
                
                # Match found within reasonable threshold
                if curr_player and best_dist < 12000:
                    c_px = (curr_player.x1 + curr_player.x2) / 2
                    c_py = (curr_player.y1 + curr_player.y2) / 2
                    
                    # Calculate player displacement vector (dx, dy)
                    dx = c_px - last_player_pos[0]
                    dy = c_py - last_player_pos[1]
                    
                    # Apply displacement to the previous frame's ball position
                    prev_ball = frames[f.frame_number - 1].ball
                    if prev_ball is not None:
                        new_x1 = prev_ball.x1 + dx
                        new_y1 = prev_ball.y1 + dy
                        new_x2 = prev_ball.x2 + dx
                        new_y2 = prev_ball.y2 + dy
                        
                        f.ball = Ball(confidence=0.0, x1=new_x1, y1=new_y1, x2=new_x2, y2=new_y2)
                    
                    last_player_pos = (c_px, c_py)
                else:
                    # Player was lost or left screen
                    holding_player_ref = None
                    last_player_pos = None

    # =================================================================
    # STEP 2: Extract Data for Kalman RTS Smoother
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
    # STEP 3: Kalman Forward Pass & RTS Backward Smoother
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
            # High trust for YOLO detections (noise=5), lower trust for estimated (noise=40)
            noise_level = 5.0 if confidences[i] > 0.0 else 40.0
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
    # STEP 4: Apply Smooth Coordinates to Frames
    # =================================================================
    missing = df['x1'].isna()
    gap_ids = (~missing).cumsum()
    gap_counts = gap_ids[missing].map(gap_ids[missing].value_counts())
    
    valid_gap_mask = pd.Series(False, index=df.index)
    valid_gap_mask[missing] = gap_counts <= max_gap_frames

    for i, f in enumerate(frames):
        if f.ball is None and valid_gap_mask.iloc[i]:
            row = df.iloc[i]
            x1 = max(0.0, float(row['smooth_x1']))
            y1 = max(0.0, float(row['smooth_y1']))
            x2 = max(0.0, float(row['smooth_x2']))
            y2 = max(0.0, float(row['smooth_y2']))

            f.ball = Ball(confidence=0.0, x1=x1, y1=y1, x2=x2, y2=y2)
            
        elif f.ball is not None and f.ball.confidence == 0.0:
            row = df.iloc[i]
            f.ball.x1 = max(0.0, float(row['smooth_x1']))
            f.ball.y1 = max(0.0, float(row['smooth_y1']))
            f.ball.x2 = max(0.0, float(row['smooth_x2']))
            f.ball.y2 = max(0.0, float(row['smooth_y2']))

    return frames