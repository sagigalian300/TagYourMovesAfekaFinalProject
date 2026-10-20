"""
Repairs missing ball detections in a sequence of video frames by interpolating the ball's 
position based on known detections.
This module provides the BallInterpolator class, which processes a list of Frame objects.
Each Frame may contain a detected Ball object or None if the ball was not detected in that frame.
The BallInterpolator identifies gaps in the ball detections and fills them using
linear interpolation based on the timestamps of the frames.
"""


from typing import List
from Ball import Ball
from Frame import Frame

class BallInterpolator:
    """
    Analyzes a list of sequential frames and estimates the location of the ball 
    for frames where the ball detection is missing (None).
    """

    def __init__(self, frames: List[Frame]):
        """
        Initializes the BallInterpolator.

        Args:
            frames (List[Frame]): A list of frames to process. Must contain at least 1 frame.
        """
        if not frames:
            raise ValueError("The frames list must contain at least 1 frame.")
        
        # Sort frames by frame_number to ensure they are strictly sequential
        self.frames = sorted(frames, key=lambda f: f.frame_number)

    def repair_missing_balls(self):
        """
        Iterates through the frames and replaces missing balls (None) with 
        new Ball objects using linear interpolation based on timestamps.
        """
        # Find the indices of all frames where the ball was successfully detected
        known_indices = [i for i, f in enumerate(self.frames) if f.ball is not None]

        # Edge case: No ball was detected in the entire list of frames
        if not known_indices:
            print("Warning: No ball detected in any frame. Cannot interpolate.")
            return

        # 1. Back-fill frames before the first detection (copy the first known ball)
        first_known = known_indices[0]
        for i in range(first_known):
            self._copy_ball(source_ball=self.frames[first_known].ball, target_frame=self.frames[i])

        # 2. Forward-fill frames after the last detection (copy the last known ball)
        last_known = known_indices[-1]
        for i in range(last_known + 1, len(self.frames)):
            self._copy_ball(source_ball=self.frames[last_known].ball, target_frame=self.frames[i])

        # 3. Interpolate the gaps between known detections
        for k in range(len(known_indices) - 1):
            start_idx = known_indices[k]
            end_idx = known_indices[k + 1]

            # If there is a gap of at least 1 missing frame between detections
            if end_idx - start_idx > 1:
                self._interpolate_gap(start_idx, end_idx)

    def _copy_ball(self, source_ball: Ball, target_frame: Frame):
        """
        Creates a static copy of a ball for extrapolation (beginning/end of video).
        Sets confidence to 0.0 to indicate it's an estimated ball.
        """
        target_frame.ball = Ball(
            confidence=0.0, 
            x1=source_ball.x1,
            y1=source_ball.y1,
            x2=source_ball.x2,
            y2=source_ball.y2
        )

    def _interpolate_gap(self, start_idx: int, end_idx: int):
        """
        Linearly interpolates the bounding box coordinates for missing frames 
        between two known frames based on timestamp differences.
        """
        start_frame = self.frames[start_idx]
        end_frame = self.frames[end_idx]
        
        start_ball = start_frame.ball
        end_ball = end_frame.ball

        start_time = start_frame.timestamp_ms
        end_time = end_frame.timestamp_ms
        time_diff = end_time - start_time

        # Loop through the missing frames in the gap
        for i in range(start_idx + 1, end_idx):
            target_frame = self.frames[i]

            # Calculate how far along the timeline this missing frame is (0.0 to 1.0)
            if time_diff > 0:
                ratio = (target_frame.timestamp_ms - start_time) / time_diff
            else:
                # Fallback to frame numbers if timestamps are identical/missing
                frame_diff = end_frame.frame_number - start_frame.frame_number
                ratio = (target_frame.frame_number - start_frame.frame_number) / frame_diff

            # Calculate interpolated coordinates
            new_x1 = start_ball.x1 + (end_ball.x1 - start_ball.x1) * ratio
            new_y1 = start_ball.y1 + (end_ball.y1 - start_ball.y1) * ratio
            new_x2 = start_ball.x2 + (end_ball.x2 - start_ball.x2) * ratio
            new_y2 = start_ball.y2 + (end_ball.y2 - start_ball.y2) * ratio

            # Create the repaired ball. Confidence is 0.0 to indicate it was artificially generated.
            target_frame.ball = Ball(
                confidence=0.0,
                x1=new_x1,
                y1=new_y1,
                x2=new_x2,
                y2=new_y2
            )