"""
Module for isolating the shirt region from a cropped player image.
"""

import numpy as np


def extract_shirt_from_player(
    player_crop: np.ndarray,
    pred_seg: np.ndarray,
    min_shirt_pixels: int = 20
) -> np.ndarray:
    """Isolates the shirt region from a cropped player image using a segmentation mask.

    Args:
        player_crop (np.ndarray): The cropped BGR image of the player.
        pred_seg (np.ndarray): Pre-computed 2D segmentation mask with class IDs.
        min_shirt_pixels (int): Minimum number of detected shirt pixels required
            to consider the extraction valid.

    Returns:
        np.ndarray: A cropped image of the player where all non-shirt
            pixels are colored black. Returns an empty array if too few
            shirt pixels are detected.
    """
    if player_crop.size == 0 or pred_seg.size == 0:
        return np.array([])

    shirt_mask = (pred_seg == 4).astype(np.uint8)

    if np.count_nonzero(shirt_mask) < min_shirt_pixels:
        return np.array([])

    shirt_only_img = np.zeros_like(player_crop)
    shirt_only_img[shirt_mask == 1] = player_crop[shirt_mask == 1]

    return shirt_only_img