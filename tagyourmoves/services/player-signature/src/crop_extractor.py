"""Utility for safely extracting player image crops from a video frame."""

from typing import List, Tuple
import numpy as np

from contracts.player.Player import Player


def extract_player_crops(
    image: np.ndarray,
    players: List[Player]
) -> Tuple[List[Player], List[np.ndarray]]:
    """Extracts valid RGB image crops for all detected players in a frame.

    Clips bounding box coordinates to image boundaries and skips degenerate
    or empty crops. Converts crops from OpenCV BGR format to RGB format
    expected by Re-ID models.

    Args:
        image (np.ndarray): The raw full video frame in BGR format.
        players (List[Player]): List of detected Player objects in the frame.

    Returns:
        Tuple[List[Player], List[np.ndarray]]: A tuple containing:
            - valid_players: List of Player objects that produced valid crops.
            - crops_rgb: Corresponding list of cropped player images in RGB format.
    """
    img_height, img_width = image.shape[:2]
    valid_players: List[Player] = []
    crops_rgb: List[np.ndarray] = []

    for player in players:
        x1 = max(0, int(player.x1))
        y1 = max(0, int(player.y1))
        x2 = min(img_width, int(player.x2))
        y2 = min(img_height, int(player.y2))

        crop_bgr = image[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            continue

        crop_rgb = crop_bgr[:, :, ::-1].copy()
        valid_players.append(player)
        crops_rgb.append(crop_rgb)

    return valid_players, crops_rgb