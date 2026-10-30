import cv2
import numpy as np
from Ball import Ball

class BallOpticalFlowTracker:
    def __init__(self):
        # Shi-Tomasi Corner Detector Parameters
        self.feature_params = dict(
            maxCorners=15,      # Maximum keypoints to track inside the ball
            qualityLevel=0.2,   # Minimum quality of corner detection
            minDistance=3,      # Minimum distance between tracked points
            blockSize=3
        )
        
        # Lucas-Kanade Optical Flow Parameters
        self.lk_params = dict(
            winSize=(15, 15),   # Size of search window at each pyramid level
            maxLevel=2,         # Multi-scale image pyramid depth
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        self.prev_gray = None
        self.p0 = None
        self.last_ball = None

    def initialize(self, frame_bgr, ball: Ball) -> bool:
        """
        Called when YOLO detects the ball. Extracts feature points inside the bounding box.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Create a binary mask restricted ONLY to the ball bounding box
        mask = np.zeros_like(gray)
        h, w = gray.shape
        x1 = max(0, int(ball.x1))
        y1 = max(0, int(ball.y1))
        x2 = min(w, int(ball.x2))
        y2 = min(h, int(ball.y2))
        
        mask[y1:y2, x1:x2] = 255
        
        # Extract trackable points inside the ball
        pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        
        if pts is not None and len(pts) > 0:
            self.p0 = pts
            self.prev_gray = gray
            self.last_ball = ball
            return True
        else:
            self.reset()
            return False

    def update(self, frame_bgr) -> Ball | None:
        """
        Called when YOLO misses the ball. Calculates Lucas-Kanade flow to predict displacement.
        """
        if self.prev_gray is None or self.p0 is None or len(self.p0) == 0:
            return None

        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Run Pyramidal Lucas-Kanade Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, self.p0, None, **self.lk_params
        )

        if p1 is None:
            self.reset()
            return None

        # Filter out points that failed to track (st == 1 means success)
        good_new = p1[st == 1]
        good_old = self.p0[st == 1]

        # Require at least 2 valid points to prevent erratic jumping
        if len(good_new) < 2:
            self.reset()
            return None

        # Calculate average displacement (dx, dy) across all tracked points
        displacement = good_new - good_old
        dx = float(np.mean(displacement[:, 0]))
        dy = float(np.mean(displacement[:, 1]))

        # Shift the bounding box coordinates
        new_x1 = self.last_ball.x1 + dx
        new_y1 = self.last_ball.y1 + dy
        new_x2 = self.last_ball.x2 + dx
        new_y2 = self.last_ball.y2 + dy

        estimated_ball = Ball(
            confidence=0.0, # Flag as estimated
            x1=new_x1, y1=new_y1, x2=new_x2, y2=new_y2
        )

        # Update state for the next frame
        self.prev_gray = curr_gray
        self.p0 = good_new.reshape(-1, 1, 2)
        self.last_ball = estimated_ball

        return estimated_ball

    def reset(self):
        self.prev_gray = None
        self.p0 = None
        self.last_ball = None