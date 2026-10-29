import pandas as pd
from typing import List
from .Frame import Frame
from .Ball import Ball

def estimate_missing_balls_offline(frames: List[Frame], max_gap_frames: int = 15) -> List[Frame]:
    """
    Fills in missing ball detections using offline interpolation.
    
    Args:
        frames (List[Frame]): The complete list of frames for the video/chunk.
        max_gap_frames (int): The maximum number of consecutive missing frames to interpolate. 
                              If the ball is missing for longer, it likely left the screen.
                              
    Returns:
        List[Frame]: The updated list of frames with interpolated balls.
    """
    if not frames:
        return frames

    # 1. Extract data into a structured format
    data = []
    for f in frames:
        if f.ball is not None:
            data.append({
                'frame_number': f.frame_number,
                'x1': f.ball.x1,
                'y1': f.ball.y1,
                'x2': f.ball.x2,
                'y2': f.ball.y2,
                'confidence': f.ball.confidence
            })
        else:
            data.append({
                'frame_number': f.frame_number,
                'x1': None,
                'y1': None,
                'x2': None,
                'y2': None,
                'confidence': None
            })

    # Create a DataFrame and set the index to the frame number
    df = pd.DataFrame(data).set_index('frame_number')

    # If the ball was NEVER detected in this chunk, return early
    if df['x1'].isna().all():
        return frames

    # 2. Interpolate the missing bounding box coordinates
    # We use 'linear' as a safe baseline. For basketballs, you can later experiment 
    # with method='quadratic' because balls travel in parabolic arcs.
    # We use 'limit' to prevent interpolating when the ball goes off-screen for a long time.
    interpolated_df = df[['x1', 'y1', 'x2', 'y2']].interpolate(
        method='linear', 
        limit=max_gap_frames, 
        limit_direction='both' # Interpolate forward and backward
    )

    # 3. Reconstruct the missing Ball objects in the frames
    for f in frames:
        if f.ball is None:
            row = interpolated_df.loc[f.frame_number]
            
            # If the row is still NaN, it means it exceeded our max_gap_frames limit
            # or it was missing at the very beginning/end of the video chunk.
            if not pd.isna(row['x1']):
                # Create a synthetic ball. 
                # We set confidence to 0.0 (or a negative flag like -1.0) 
                # so upstream services know this was an algorithm estimate, not a model detection.
                f.ball = Ball(
                    confidence=0.0, 
                    x1=float(row['x1']),
                    y1=float(row['y1']),
                    x2=float(row['x2']),
                    y2=float(row['y2'])
                )

    return frames