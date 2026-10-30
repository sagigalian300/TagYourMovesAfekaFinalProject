import cv2
import numpy as np
from Ball import Ball

class BallOpticalFlowTracker:
    """
    Short-range Lucas-Kanade tracker that bridges a few frames after YOLO loses the ball.
    Its output is a weak hint (confidence 0.0); the offline estimator decides how much to trust it.
    """

    def __init__(self, max_frames: int = 8, fb_threshold: float = 1.0, inner_radius_ratio: float = 0.7):
        """
        Args:
            max_frames: Stop tracking after this many consecutive frames without a detection.
                LK drifts quickly on a low-texture ball, so long chains do more harm than good.
            fb_threshold: Max forward-backward error (px) for a point to count as reliably tracked.
            inner_radius_ratio: Features are only taken from a circle of this fraction of the
                ball radius, so corners of the background inside the bounding box are ignored.
        """
        # Shi-Tomasi Corner Detector Parameters
        self.feature_params = dict(
            maxCorners=20,      # Maximum keypoints to track inside the ball
            qualityLevel=0.05,  # The ball has little texture, so accept weaker corners
            minDistance=2,      # Minimum distance between tracked points
            blockSize=3
        )

        # Lucas-Kanade Optical Flow Parameters
        self.lk_params = dict(
            winSize=(15, 15),   # Size of search window at each pyramid level
            maxLevel=2,         # Multi-scale image pyramid depth
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        self.max_frames = max_frames
        self.fb_threshold = fb_threshold
        self.inner_radius_ratio = inner_radius_ratio
        self.reset()

    def initialize(self, frame_bgr, ball: Ball) -> bool:
        """
        Called when YOLO detects the ball. Extracts feature points inside the ball itself.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Circular mask inside the bounding box: the box corners are background, not ball
        mask = np.zeros_like(gray)
        cx = int(round((ball.x1 + ball.x2) / 2))
        cy = int(round((ball.y1 + ball.y2) / 2))
        radius = int(min(ball.x2 - ball.x1, ball.y2 - ball.y1) / 2 * self.inner_radius_ratio)
        cv2.circle(mask, (cx, cy), max(radius, 2), 255, -1)

        # Extract trackable points inside the ball
        pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)

        if pts is not None and len(pts) >= 2:
            self.p0 = pts
            self.prev_gray = gray
            self.last_ball = ball
            self.frames_tracked = 0
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
        if self.frames_tracked >= self.max_frames:
            self.reset()
            return None

        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Forward pass, then track the result back: points that don't return to where
        # they started latched onto something else (an occluding arm, the floor, ...)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, self.p0, None, **self.lk_params)
        if p1 is None:
            self.reset()
            return None
        p0_back, st_back, _ = cv2.calcOpticalFlowPyrLK(curr_gray, self.prev_gray, p1, None, **self.lk_params)
        if p0_back is None:
            self.reset()
            return None

        fb_error = np.linalg.norm((self.p0 - p0_back).reshape(-1, 2), axis=1)
        good = (st.reshape(-1) == 1) & (st_back.reshape(-1) == 1) & (fb_error < self.fb_threshold)
        good_new = p1.reshape(-1, 2)[good]
        good_old = self.p0.reshape(-1, 2)[good]

        # Require at least 2 valid points to prevent erratic jumping
        if len(good_new) < 2:
            self.reset()
            return None

        # Median is robust to the few points that still drifted
        displacement = good_new - good_old
        dx = float(np.median(displacement[:, 0]))
        dy = float(np.median(displacement[:, 1]))

        # Shift the bounding box coordinates
        estimated_ball = Ball(
            confidence=0.0, # Flag as estimated
            x1=self.last_ball.x1 + dx, y1=self.last_ball.y1 + dy,
            x2=self.last_ball.x2 + dx, y2=self.last_ball.y2 + dy
        )

        # Update state for the next frame
        self.prev_gray = curr_gray
        self.p0 = good_new.reshape(-1, 1, 2)
        self.last_ball = estimated_ball
        self.frames_tracked += 1

        return estimated_ball

    def reset(self):
        self.prev_gray = None
        self.p0 = None
        self.last_ball = None
        self.frames_tracked = 0
