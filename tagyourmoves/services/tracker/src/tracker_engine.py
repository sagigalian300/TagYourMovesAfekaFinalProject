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

class TrackerEngine:
    """
    Core tracking engine implementing the SORT (Simple Online and Realtime Tracking) logic.

    Attributes:
        active_tracks (list[Track]): A list of currently active tracked objects.
        next_id (int): The next available unique tracking ID.
        iou_threshold (float): Minimum IoU overlap required to validate an assignment.
        max_age (int): Maximum frames a track is kept alive without a measurement update.
    """

    def __init__(self):
        """Initializes the TrackerEngine with default configuration parameters."""
        self.active_tracks: list[Track] = []
        self.next_id: int = 1
        self.iou_threshold: float = 0.3
        self.max_age: int = 30  # Allows tracks to survive brief occlusions (1 second at 30 FPS)

    def update(self, frame: Frame) -> Frame:
        """
        Processes a new frame by predicting track states, matching them with new detections,
        and updating the internal tracking state.

        Args:
            frame (Frame): The incoming frame containing raw object detections.

        Returns:
            Frame: The updated frame with tracking IDs assigned to the players.
        """
        detections = frame.players
        det_boxes = [[p.x1, p.y1, p.x2, p.y2] for p in detections]

        # 1. Predict the new locations of all existing tracks
        predicted_boxes = []
        for track in self.active_tracks:
            track.predict()
            predicted_boxes.append(track.get_state())

        # 2. Handle empty initial state or empty detections edge-cases
        if len(self.active_tracks) == 0:
            self._create_new_tracks(detections, list(range(len(detections))))
            return frame

        if len(detections) == 0:
            self._cleanup_stale_tracks()
            return frame

        # 3. Construct the cost matrix based on IoU between predictions and actual detections
        cost_matrix = np.zeros((len(predicted_boxes), len(det_boxes)), dtype=np.float32)
        for t, pred_box in enumerate(predicted_boxes):
            for d, det_box in enumerate(det_boxes):
                cost_matrix[t, d] = 1.0 - calculate_iou(pred_box, det_box)

        # 4. Solve the bipartite matching problem using the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 5. Process matched pairs, unassigned tracks, and unassigned detections
        unmatched_detections = set(range(len(det_boxes)))
        matched_track_indices = set()

        for t_idx, d_idx in zip(row_ind, col_ind):
            iou_score = 1.0 - cost_matrix[t_idx, d_idx]
            if iou_score >= self.iou_threshold:
                # Valid match: Update track with the actual detection bounding box
                self.active_tracks[t_idx].update(det_boxes[d_idx])
                detections[d_idx].tracking_id = self.active_tracks[t_idx].track_id
                
                unmatched_detections.discard(d_idx)
                matched_track_indices.add(t_idx)

        # Handle tracks that were not matched (lost or occluded)
        for t_idx, track in enumerate(self.active_tracks):
            if t_idx not in matched_track_indices:
                track.hit_streak = 0

        # Create new tracks for detections that were not matched to any existing track
        self._create_new_tracks(detections, list(unmatched_detections))

        # 6. Remove tracks that have been lost for too long
        self._cleanup_stale_tracks()

        return frame

    def _create_new_tracks(self, detections: list, unmatched_indices: list) -> None:
        """
        Instantiates new tracks for unassigned detections and assigns them tracking IDs.

        Args:
            detections (list): The list of Player domain objects from the current frame.
            unmatched_indices (list): List of indices pointing to unassigned detections.
        """
        for d_idx in unmatched_indices:
            player = detections[d_idx]
            bbox = [player.x1, player.y1, player.x2, player.y2]
            
            new_track = Track(bbox, self.next_id)
            self.active_tracks.append(new_track)
            
            player.tracking_id = self.next_id
            self.next_id += 1

    def _cleanup_stale_tracks(self) -> None:
        """
        Removes tracks that have exceeded the maximum allowed frames without an update.
        """
        self.active_tracks = [t for t in self.active_tracks if t.time_since_update <= self.max_age]