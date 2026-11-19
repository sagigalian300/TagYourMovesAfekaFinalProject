"""
Module for isolating clean shirt pixels from a player crop using a segmentation mask,
with skin-pixel filtering, edge erosion, and a geometric torso fallback.
"""

import cv2
import numpy as np


def _get_skin_mask(bgr_img: np.ndarray) -> np.ndarray:
    """Builds a binary mask of skin-colored pixels in HSV space.

    Args:
        bgr_img (np.ndarray): Input BGR image crop.

    Returns:
        np.ndarray: Binary uint8 mask (0 or 255) where 255 indicates skin pixels.
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    # Standard HSV ranges covering light to dark skin tones under indoor arena lighting
    lower_skin_1 = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin_1 = np.array([25, 220, 255], dtype=np.uint8)
    lower_skin_2 = np.array([165, 30, 60], dtype=np.uint8)
    upper_skin_2 = np.array([180, 220, 255], dtype=np.uint8)

    mask_1 = cv2.inRange(hsv, lower_skin_1, upper_skin_1)
    mask_2 = cv2.inRange(hsv, lower_skin_2, upper_skin_2)
    return cv2.bitwise_or(mask_1, mask_2)


def extract_shirt_from_player(
    player_crop: np.ndarray,
    segmentation_mask: np.ndarray,
    shirt_class_id: int = 4,
    min_shirt_pixels: int = 30,
) -> np.ndarray:
    """Extracts isolated shirt pixels from a player crop.

    Applies light morphological erosion to peel off background bleed and removes
    skin pixels (arms/neck) that leaked into the shirt mask. If too few pixels
    remain (e.g., white jersey missed by SegFormer), falls back to a tight inner
    torso region with skin pixels filtered out.

    Args:
        player_crop (np.ndarray): Cropped BGR image of the player.
        segmentation_mask (np.ndarray): 2D class-ID mask matching player_crop (H, W).
        shirt_class_id (int): Class ID representing upper clothing (default: 4).
        min_shirt_pixels (int): Minimum valid shirt pixels required before triggering
            the geometric torso fallback.

    Returns:
        np.ndarray: Masked BGR image containing only the cleaned shirt pixels
            (background set to [0, 0, 0]).
    """
    if player_crop.size == 0 or segmentation_mask.size == 0:
        return np.zeros_like(player_crop)

    skin_mask = _get_skin_mask(player_crop)
    non_skin_mask = cv2.bitwise_not(skin_mask)

    # 1. Start from SegFormer shirt mask
    shirt_mask = np.where(segmentation_mask == shirt_class_id, 255, 0).astype(np.uint8)

    # 2. Light erosion to remove court/background edge bleed
    if cv2.countNonZero(shirt_mask) >= min_shirt_pixels:
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(shirt_mask, kernel, iterations=1)
        if cv2.countNonZero(eroded) >= min_shirt_pixels:
            shirt_mask = eroded

    # 3. Remove leaked skin pixels (arms/shoulders)
    cleaned_mask = cv2.bitwise_and(shirt_mask, non_skin_mask)

    # 4. Geometric torso fallback if SegFormer missed the jersey (e.g., white shirts)
    if cv2.countNonZero(cleaned_mask) < min_shirt_pixels:
        h, w = player_crop.shape[:2]
        y1, y2 = int(h * 0.25), int(h * 0.50)
        x1, x2 = int(w * 0.35), int(w * 0.65)

        fallback_mask = np.zeros((h, w), dtype=np.uint8)
        fallback_mask[y1:y2, x1:x2] = 255
        cleaned_mask = cv2.bitwise_and(fallback_mask, non_skin_mask)

    return cv2.bitwise_and(player_crop, player_crop, mask=cleaned_mask)