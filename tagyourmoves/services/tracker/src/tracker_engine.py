import numpy as np
from scipy.optimize import linear_sum_assignment
from contract.frame.Frame import Frame
from src.track import Track

def calculate_iou(box_a: list, box_b: list) -> float:
    """
    Computes the Intersection over Union (IoU) metric between two bounding boxes.

    Args:
        box_a (list): First bounding box [x1, y1, x2, y2].
        box_b (list): Second bounding box [x1, y1, x2, y2].

    Returns:
        float: The IoU score ranging from 0.0 (no overlap) to 1.0 (perfect overlap).
    """
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    iou = intersection_area / float(box_a_area + box_b_area - intersection_area + 1e-6)
    
    return iou

def calculate_center_distance(box_a: list, box_b: list) -> float:
    """
    Computes the Euclidean distance between the centers of two bounding boxes.

    Args:
        box_a (list): First bounding box [x1, y1, x2, y2].
        box_b (list): Second bounding box [x1, y1, x2, y2].

    Returns:
        float: The Euclidean distance in pixels.
    """
    cx_a = (box_a[0] + box_a[2]) / 2.0
    cy_a = (box_a[1] + box_a[3]) / 2.0
    cx_b = (box_b[0] + box_b[2]) / 2.0
    cy_b = (box_b[1] + box_b[3]) / 2.0
    
    return float(np.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2))

class TrackerEngine:
    """
    Core tracking engine implementing SORT logic with Camera Motion Compensation (CMC)
    using rim anchor points and a hybrid IoU/Distance cost matrix.

    Attributes:
        active_tracks (list[Track]): A list of currently active tracked objects.
        next_id (int): The next available unique tracking ID.
        max_cost_threshold (float): Maximum allowed cost to validate an assignment.
        max_age (int): Maximum frames a track is kept alive without a measurement update.
        prev_rim_center (tuple): Center coordinates of the rim in the previous frame.
    """

    def __init__(self):
        """Initializes the TrackerEngine with default configuration parameters."""
        self.active_tracks: list[Track] = []
        self.next_id: int = 1
        self.max_cost_threshold: float = 0.85
        self.max_age: int = 30
        self.prev_rim_center: tuple = None

    def update(self, frame: Frame) -> Frame:
        """
        Processes a new frame by compensating for camera motion, predicting track states,
        matching them with new detections, and updating the internal tracking state.

        Args:
            frame (Frame): The incoming frame containing raw object detections.

        Returns:
            Frame: The updated frame with tracking IDs assigned to the players.
        """
        detections = frame.players
        det_boxes = [[p.x1, p.y1, p.x2, p.y2] for p in detections]

        # 1. Camera Motion Compensation (CMC) Using Rim as an Anchor
        current_rim_center = None
        if frame.rims and len(frame.rims) > 0:
            # We assume the first detected rim is the main anchor
            rim = frame.rims[0]
            current_rim_center = ((rim.x1 + rim.x2) / 2.0, (rim.y1 + rim.y2) / 2.0)

        dx, dy = 0.0, 0.0
        if self.prev_rim_center and current_rim_center:
            dx = current_rim_center[0] - self.prev_rim_center[0]
            dy = current_rim_center[1] - self.prev_rim_center[1]
            
            # Sanity Check: If the shift is massive, it might be a false positive rim or a camera cut
            if abs(dx) > 300 or abs(dy) > 300:
                dx, dy = 0.0, 0.0

        self.prev_rim_center = current_rim_center

        # 2. Predict and apply camera shift to existing tracks
        predicted_boxes = []
        for track in self.active_tracks:
            track.predict()
            if dx != 0.0 or dy != 0.0:
                track.apply_camera_shift(dx, dy)
            predicted_boxes.append(track.get_state())

        if len(self.active_tracks) == 0:
            self._create_new_tracks(detections, list(range(len(detections))))
            return frame

        if len(detections) == 0:
            self._cleanup_stale_tracks()
            return frame

        # 3. Construct Hybrid Cost Matrix (50% IoU + 50% Euclidean Distance)
        cost_matrix = np.zeros((len(predicted_boxes), len(det_boxes)), dtype=np.float32)
        for t, pred_box in enumerate(predicted_boxes):
            for d, det_box in enumerate(det_boxes):
                iou = calculate_iou(pred_box, det_box)
                dist = calculate_center_distance(pred_box, det_box)
                
                # Normalize distance (Assuming ~400 pixels is a huge jump for a player in 1 frame)
                norm_dist = min(dist / 400.0, 1.0)
                
                # Hybrid Cost: 0 = Perfect Match, 1 = Terrible Match
                cost = 0.5 * (1.0 - iou) + 0.5 * norm_dist
                cost_matrix[t, d] = cost

        # 4. Solve bipartite matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 5. Process assignments
        unmatched_detections = set(range(len(det_boxes)))
        matched_track_indices = set()

        for t_idx, d_idx in zip(row_ind, col_ind):
            cost_score = cost_matrix[t_idx, d_idx]
            
            if cost_score <= self.max_cost_threshold:
                # Valid match
                self.active_tracks[t_idx].update(det_boxes[d_idx])
                detections[d_idx].tracking_id = self.active_tracks[t_idx].track_id
                
                unmatched_detections.discard(d_idx)
                matched_track_indices.add(t_idx)

        # Unmatched Tracks get hit_streak reset
        for t_idx, track in enumerate(self.active_tracks):
            if t_idx not in matched_track_indices:
                track.hit_streak = 0

        # Unmatched Detections become new tracks
        self._create_new_tracks(detections, list(unmatched_detections))

        # 6. Cleanup
        self._cleanup_stale_tracks()

        return frame

    def _create_new_tracks(self, detections: list, unmatched_indices: list) -> None:
        """
        Instantiates new tracks for unassigned detections and assigns them tracking IDs.

        Args:
            detections (list): The list of Player domain objects.
            unmatched_indices (list): Indices pointing to unassigned detections.
        """
        for d_idx in unmatched_indices:
            player = detections[d_idx]
            bbox = [player.x1, player.y1, player.x2, player.y2]
            
            new_track = Track(bbox, self.next_id)
            self.active_tracks.append(new_track)
            
            player.tracking_id = self.next_id
            self.next_id += 1

    def _cleanup_stale_tracks(self) -> None:
        """Removes tracks that have exceeded the maximum allowed frames without an update."""
        self.active_tracks = [t for t in self.active_tracks if t.time_since_update <= self.max_age]