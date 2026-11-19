"""
Module for detecting if a person is wearing long pants (identifying referees).
"""

import cv2
import numpy as np


def has_long_pants(
    player_crop: np.ndarray,
    pred_seg: np.ndarray
) -> tuple[bool, float, float, np.ndarray]:
    """Determines if a person is wearing long dark pants using a hybrid approach.

    Uses the pre-computed SegFormer mask to isolate the person from the background,
    then inspects the shin/calf window (60% to 88% of crop height, excluding shoes)
    for both low skin-tone ratio and high dark-pixel ratio.

    Args:
        player_crop (np.ndarray): The cropped BGR image of the person.
        pred_seg (np.ndarray): Pre-computed 2D segmentation mask with class IDs.

    Returns:
        tuple[bool, float, float, np.ndarray]: A tuple containing:
            - bool: True if the person is wearing long dark pants (referee), False otherwise.
            - float: The calculated skin-to-lower-body pixel ratio (0.0 to 1.0).
            - float: The calculated dark-to-lower-body pixel ratio (0.0 to 1.0).
            - np.ndarray: Binary mask of detected skin pixels in the 60%–88% region.
    """
    if player_crop.size == 0 or pred_seg.size == 0:
        return False, 0.0, 0.0, np.array([])

    person_mask = (pred_seg != 0).astype(np.uint8)

    h, w = player_crop.shape[:2]
    split_top = int(h * 0.60)
    split_bottom = int(h * 0.88)

    if split_bottom <= split_top:
        return False, 0.0, 0.0, np.array([])

    lower_body_img = player_crop[split_top:split_bottom, :]
    lower_person_mask = person_mask[split_top:split_bottom, :]

    lower_person_pixels = np.sum(lower_person_mask)
    if lower_person_pixels == 0:
        return False, 0.0, 0.0, np.array([])

    hsv_lower = cv2.cvtColor(lower_body_img, cv2.COLOR_BGR2HSV)

    # 1. Skin-tone detection inside the person's silhouette
    lower_skin = np.array([0, 20, 50], dtype=np.uint8)
    upper_skin = np.array([25, 170, 255], dtype=np.uint8)

    skin_mask_hsv = cv2.inRange(hsv_lower, lower_skin, upper_skin)
    valid_skin_mask = cv2.bitwise_and(skin_mask_hsv, skin_mask_hsv, mask=lower_person_mask)
    skin_pixels = cv2.countNonZero(valid_skin_mask)
    skin_ratio = float(skin_pixels / lower_person_pixels)

    # 2. Dark/black pants detection (low Value channel in HSV) inside the silhouette
    v_channel = hsv_lower[:, :, 2]
    dark_mask = ((v_channel < 80) & (lower_person_mask > 0)).astype(np.uint8)
    dark_pixels = int(np.sum(dark_mask))
    dark_ratio = float(dark_pixels / lower_person_pixels)

    is_referee = (skin_ratio < 0.15) and (dark_ratio > 0.45)

    return is_referee, skin_ratio, dark_ratio, valid_skin_mask