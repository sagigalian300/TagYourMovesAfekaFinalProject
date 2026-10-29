import pandas as pd
from typing import List
from Frame import Frame
from Ball import Ball

def estimate_missing_balls_offline(frames: List[Frame], max_gap_frames: int = 15) -> List[Frame]:
    if not frames:
        return frames

    # =================================================================
    # STEP 1: Handle Player Possession (Occlusion)
    # If the ball disappears while held, anchor it to the moving player.
    # =================================================================
    last_known_ball_center = None
    player_holding_ball = False
    last_player_center = None

    for f in frames:
        if f.ball is not None:
            # Ball is visible. Update our tracking variables.
            bx = (f.ball.x1 + f.ball.x2) / 2
            by = (f.ball.y1 + f.ball.y2) / 2
            
            player_holding_ball = False
            for p in f.players:
                # Check if ball center is inside/near the player's bounding box (padding of 15 pixels)
                if p.x1 - 15 <= bx <= p.x2 + 15 and p.y1 - 15 <= by <= p.y2 + 15:
                    player_holding_ball = True
                    last_player_center = ((p.x1 + p.x2) / 2, (p.y1 + p.y2) / 2)
                    break
        else:
            # Ball is missing. Was it held by a player right before it disappeared?
            if player_holding_ball and last_player_center is not None:
                # Find the closest player in the current frame to the last known holding player
                best_dist = float('inf')
                best_player = None
                
                for p in f.players:
                    px = (p.x1 + p.x2) / 2
                    py = (p.y1 + p.y2) / 2
                    dist = (px - last_player_center[0])**2 + (py - last_player_center[1])**2
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_player = p
                
                # If a matching player is found (heuristic: didn't teleport across the court)
                if best_player and best_dist < 15000: 
                    px = (best_player.x1 + best_player.x2) / 2
                    py = (best_player.y1 + best_player.y2) / 2
                    last_player_center = (px, py) # Update anchor for the next frame
                    
                    # Estimate the ball at the player's center (chest area)
                    bw, bh = 25, 25 # Estimated ball width/height
                    f.ball = Ball(
                        confidence=0.0, 
                        x1=px - bw/2, y1=py - bh/2, 
                        x2=px + bw/2, y2=py + bh/2
                    )

    # =================================================================
    # STEP 2: Extract data into DataFrame for mid-air gaps
    # =================================================================
    data = []
    for f in frames:
        if f.ball is not None:
            data.append({
                'frame_number': f.frame_number,
                'x1': f.ball.x1, 'y1': f.ball.y1,
                'x2': f.ball.x2, 'y2': f.ball.y2,
            })
        else:
            data.append({
                'frame_number': f.frame_number,
                'x1': None, 'y1': None, 'x2': None, 'y2': None
            })

    df = pd.DataFrame(data).set_index('frame_number')

    if df['x1'].isna().all():
        return frames

    # =================================================================
    # STEP 3: Interpolate remaining missing bounding box coordinates
    # (These are the gaps where the ball was in free-flight)
    # =================================================================
    try:
        interpolated_df = df[['x1', 'y1', 'x2', 'y2']].interpolate(
            method='quadratic', limit=max_gap_frames, limit_direction='both'
        )
    except ValueError:
        interpolated_df = df[['x1', 'y1', 'x2', 'y2']].interpolate(
            method='linear', limit=max_gap_frames, limit_direction='both'
        )

    # =================================================================
    # STEP 4: Reconstruct the missing mid-air Ball objects
    # =================================================================
    for f in frames:
        if f.ball is None:
            row = interpolated_df.loc[f.frame_number]
            if not pd.isna(row['x1']):
                x1 = max(0.0, float(row['x1']))
                y1 = max(0.0, float(row['y1']))
                x2 = max(0.0, float(row['x2']))
                y2 = max(0.0, float(row['y2']))

                f.ball = Ball(confidence=0.0, x1=x1, y1=y1, x2=x2, y2=y2)

    return frames