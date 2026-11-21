from typing import List, Optional
import numpy as np
from filterpy.kalman import KalmanFilter


class Track:
    """Represents a single object track utilizing a Kalman Filter and Re-ID signature.

    The state space is defined as [u, v, s, r, u_dot, v_dot, s_dot], where:
    - (u, v) is the center of the bounding box.
    - s is the scale (area) of the bounding box.
    - r is the aspect ratio.
    - *_dot represent the respective velocities.

    Attributes:
        track_id (int): Unique identifier assigned to the tracked object.
        kf (KalmanFilter): The Kalman Filter instance managing the kinematic state.
        signature (Optional[np.ndarray]): L2-normalized EMA appearance embedding.
        time_since_update (int): Consecutive frames without a matched detection.
        hits (int): Total frames where this track was matched with a detection.
        hit_streak (int): Consecutive frames this track has been matched.
        age (int): Total frames since the track was instantiated.
    """

    def __init__(
        self,
        bbox: list,
        track_id: int,
        signature: Optional[List[float]] = None
    ):
        """Initializes the Track instance with an initial bounding box and optional signature.

        Args:
            bbox (list): Initial bounding box in [x1, y1, x2, y2] format.
            track_id (int): Unique tracking ID assigned to this object.
            signature (Optional[List[float]], optional): Initial L2-normalized Re-ID
                embedding vector. Defaults to None.
        """
        self.track_id: int = track_id
        self.time_since_update: int = 0
        self.hits: int = 1
        self.hit_streak: int = 1
        self.age: int = 1
        self.signature: Optional[np.ndarray] = (
            np.asarray(signature, dtype=np.float32) if signature is not None else None
        )

        # Initialize Kalman Filter: 7 state variables, 4 measurement variables
        self.kf: KalmanFilter = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix (F)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])

        # Measurement function (H)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        # Initial state covariance matrix (P) - High uncertainty for unobservable velocities
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.

        # Process noise (Q) and Measurement noise (R)
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.R[2:, 2:] *= 10.

        # Set initial state (x) based on the first observation
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)

    def predict(self) -> None:
        """Advances the state vector and covariance by one time step."""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0.0

        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def apply_camera_shift(self, dx: float, dy: float) -> None:
        """Shifts the Kalman Filter's internal position state to compensate for camera motion.

        Args:
            dx (float): Positional shift along the x-axis in pixels.
            dy (float): Positional shift along the y-axis in pixels.
        """
        self.kf.x[0] += dx
        self.kf.x[1] += dy

    def update(
        self,
        bbox: list,
        signature: Optional[List[float]] = None,
        ema_alpha: float = 0.9
    ) -> None:
        """Updates the kinematic state and smooths the appearance signature via EMA.

        Args:
            bbox (list): Matched bounding box in [x1, y1, x2, y2] format.
            signature (Optional[List[float]], optional): New L2-normalized Re-ID
                embedding vector. Defaults to None.
            ema_alpha (float, optional): Smoothing factor for exponential moving
                average (higher retains more history). Defaults to 0.9.
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))

        if signature is not None:
            new_sig = np.asarray(signature, dtype=np.float32)
            if self.signature is None:
                self.signature = new_sig
            else:
                blended = ema_alpha * self.signature + (1.0 - ema_alpha) * new_sig
                norm = np.linalg.norm(blended)
                if norm > 0:
                    self.signature = blended / norm

    def get_state(self) -> list:
        """Returns the current bounding box estimate based on the internal state.

        Returns:
            list: Predicted bounding box in [x1, y1, x2, y2] format.
        """
        if self.kf.x[2] < 0:
            self.kf.x[2] = 0

        u, v, s, r = self.kf.x[:4].reshape((4,))
        w = np.sqrt(s * r)
        h = s / w if w > 0 else 0
        x1 = u - w / 2.0
        y1 = v - h / 2.0
        x2 = u + w / 2.0
        y2 = v + h / 2.0

        return [x1, y1, x2, y2]

    def _convert_bbox_to_z(self, bbox: list) -> np.ndarray:
        """Converts a bounding box to the observation format required by the Kalman Filter.

        Args:
            bbox (list): Bounding box in [x1, y1, x2, y2] format.

        Returns:
            np.ndarray: Measurement vector [u, v, s, r] formatted as a column vector.
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        u = bbox[0] + w / 2.0
        v = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h) if h > 0 else 0
        return np.array([u, v, s, r]).reshape((4, 1))