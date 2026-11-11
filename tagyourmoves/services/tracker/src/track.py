import numpy as np
from filterpy.kalman import KalmanFilter

class Track:
    """
    Represents a single object track utilizing a Kalman Filter for state estimation.

    The state space is defined as [u, v, s, r, u_dot, v_dot, s_dot], where:
    - (u, v) is the center of the bounding box.
    - s is the scale (area) of the bounding box.
    - r is the aspect ratio.
    - *_dot represent the respective velocities.

    Attributes:
        track_id (int): Unique identifier assigned to the tracked object.
        kf (KalmanFilter): The Kalman Filter instance managing the kinematic state.
        time_since_update (int): Number of consecutive frames without a matched detection.
        hits (int): Total number of frames where this track was matched with a detection.
        hit_streak (int): Number of consecutive frames this track has been matched.
        age (int): Total number of frames since the track was instantiated.
    """

    def __init__(self, bbox: list, track_id: int):
        """
        Initializes the Track instance with an initial bounding box.

        Args:
            bbox (list): The initial bounding box in [x1, y1, x2, y2] format.
            track_id (int): The unique tracking ID to assign to this object.
        """
        self.track_id: int = track_id
        self.time_since_update: int = 0
        self.hits: int = 1
        self.hit_streak: int = 1
        self.age: int = 1

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
        """
        Advances the state vector and covariance by one time step.
        """
        # Ensure bounding box area estimation does not become negative
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0.0
            
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox: list) -> None:
        """
        Updates the state vector with the newly matched bounding box detection.

        Args:
            bbox (list): The matched bounding box in [x1, y1, x2, y2] format.
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))

    def get_state(self) -> list:
        """
        Returns the current bounding box estimate based on the internal state.

        Returns:
            list: The predicted bounding box in [x1, y1, x2, y2] format.
        """
        if self.kf.x[2] < 0:
            self.kf.x[2] = 0
            
        # Convert state [u, v, s, r] back to [x1, y1, x2, y2]
        u, v, s, r = self.kf.x[:4].reshape((4, ))
        w = np.sqrt(s * r)
        h = s / w
        x1 = u - w / 2.0
        y1 = v - h / 2.0
        x2 = u + w / 2.0
        y2 = v + h / 2.0
        
        return [x1, y1, x2, y2]

    def _convert_bbox_to_z(self, bbox: list) -> np.ndarray:
        """
        Converts a bounding box to the observation format required by the Kalman Filter.

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
        # Prevent division by zero if width is extremely small
        r = w / float(h) if h > 0 else 0
        return np.array([u, v, s, r]).reshape((4, 1))
    
    def apply_camera_shift(self, dx: float, dy: float) -> None:
        """
        Shifts the Kalman Filter's internal position state to compensate for camera movement.
        This ensures predictions remain accurate even when the background shifts.

        Args:
            dx (float): The positional shift along the x-axis in pixels.
            dy (float): The positional shift along the y-axis in pixels.
        """
        self.kf.x[0] += dx  # Update the u coordinate (center x)
        self.kf.x[1] += dy  # Update the v coordinate (center y)