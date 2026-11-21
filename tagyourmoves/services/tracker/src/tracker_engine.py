from typing import Optional
import numpy as np
from scipy.optimize import linear_sum_assignment
from contract.frame.Frame import Frame
from src.track import Track


def calculate_iou(box_a: list, box_b: list) -> float:
    """Computes the Intersection over Union (IoU) metric between two bounding boxes.

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
    """Computes the Euclidean distance between the centers of two bounding boxes.

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

    return float(np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2))


def calculate_cosine_distance(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Computes the cosine distance between two L2-normalized embedding vectors.

    Args:
        sig_a (np.ndarray): First L2-normalized feature vector.
        sig_b (np.ndarray): Second L2-normalized feature vector.

    Returns:
        float: Cosine distance clipped to [0.0, 1.0], where 0.0 is identical.
    """
    similarity = float(np.dot(sig_a, sig_b))
    return float(np.clip(1.0 - similarity, 0.0, 1.0))


class TrackerEngine:
    """Core tracking engine combining SORT, Rim-based CMC, and Re-ID signatures.

    Attributes:
        active_tracks (list[Track]): Currently active tracked objects.
        next_id (int): Next available unique tracking ID.
        max_cost_threshold (float): Maximum allowed cost to validate an assignment.
        max_age (int): Maximum frames a track is kept alive without an update.
        prev_rim_center (Optional[tuple]): Rim center coordinates from the previous frame.
    """

    def __init__(self):
        """Initializes the TrackerEngine with default configuration parameters."""
        self.active_tracks: list[Track] = []
        self.next_id: int = 1
        self.max_cost_threshold: float = 0.85
        self.max_age: int = 30
        self.prev_rim_center: Optional[tuple] = None

    def update(self, frame: Frame) -> Frame:
        """Processes a new frame using CMC, Kalman prediction, and hybrid matching.

        Args:
            frame (Frame): Incoming frame containing player detections and optional signatures.

        Returns:
            Frame: Updated frame with tracking IDs assigned to players.
        """
        detections = frame.players
        det_boxes = [[p.x1, p.y1, p.x2, p.y2] for p in detections]
        det_sigs = [
            np.asarray(p.signature, dtype=np.float32) if getattr(p, "signature", None) is not None else None
            for p in detections
        ]

        # 1. Camera Motion Compensation (CMC) using Rim as an anchor
        current_rim_center = None
        if frame.rims and len(frame.rims) > 0:
            rim = frame.rims[0]
            current_rim_center = ((rim.x1 + rim.x2) / 2.0, (rim.y1 + rim.y2) / 2.0)

        dx, dy = 0.0, 0.0
        if self.prev_rim_center and current_rim_center:
            dx = current_rim_center[0] - self.prev_rim_center[0]
            dy = current_rim_center[1] - self.prev_rim_center[1]

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

        # 3. Construct Hybrid Cost Matrix (IoU + Euclidean Distance + Cosine Distance)
        cost_matrix = np.zeros((len(predicted_boxes), len(det_boxes)), dtype=np.float32)
        for t, (track, pred_box) in enumerate(zip(self.active_tracks, predicted_boxes)):
            for d, (det_box, det_sig) in enumerate(zip(det_boxes, det_sigs)):
                iou_cost = 1.0 - calculate_iou(pred_box, det_box)
                dist = calculate_center_distance(pred_box, det_box)
                norm_dist = min(dist / 400.0, 1.0)

                if track.signature is not None and det_sig is not None:
                    cos_cost = calculate_cosine_distance(track.signature, det_sig)
                    cost = 0.35 * iou_cost + 0.35 * norm_dist + 0.30 * cos_cost
                else:
                    cost = 0.50 * iou_cost + 0.50 * norm_dist

                cost_matrix[t, d] = cost

        # 4. Solve bipartite matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 5. Process assignments
        unmatched_detections = set(range(len(det_boxes)))
        matched_track_indices = set()

        for t_idx, d_idx in zip(row_ind, col_ind):
            if cost_matrix[t_idx, d_idx] <= self.max_cost_threshold:
                det_signature = getattr(detections[d_idx], "signature", None)
                self.active_tracks[t_idx].update(det_boxes[d_idx], signature=det_signature)
                detections[d_idx].tracking_id = self.active_tracks[t_idx].track_id

                unmatched_detections.discard(d_idx)
                matched_track_indices.add(t_idx)

        for t_idx, track in enumerate(self.active_tracks):
            if t_idx not in matched_track_indices:
                track.hit_streak = 0

        self._create_new_tracks(detections, list(unmatched_detections))

        # 6. Cleanup stale tracks
        self._cleanup_stale_tracks()

        return frame

    def _create_new_tracks(self, detections: list, unmatched_indices: list) -> None:
        """Instantiates new tracks for unassigned detections and assigns tracking IDs.

        Args:
            detections (list): List of Player domain objects.
            unmatched_indices (list): Indices pointing to unassigned detections.
        """
        for d_idx in unmatched_indices:
            player = detections[d_idx]
            bbox = [player.x1, player.y1, player.x2, player.y2]
            signature = getattr(player, "signature", None)

            new_track = Track(bbox, self.next_id, signature=signature)
            self.active_tracks.append(new_track)

            player.tracking_id = self.next_id
            self.next_id += 1

    def _cleanup_stale_tracks(self) -> None:
        """Removes tracks that have exceeded the maximum allowed frames without an update."""
        self.active_tracks = [t for t in self.active_tracks if t.time_since_update <= self.max_age]