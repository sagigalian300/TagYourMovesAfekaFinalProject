"""
Module for detecting if a person is wearing long pants (identifying referees).
"""

import cv2
import numpy as np
from src.clothing_segmentation import ClothingSegmenter


def has_long_pants(player_crop: np.ndarray, segmenter: ClothingSegmenter) -> bool:
    """Determines if a person is wearing long pants using a hybrid approach.

    Uses SegFormer strictly to isolate the person from the background (floor/crowd).
    Then applies HSV skin-tone detection on the lower 40% of the isolated person 
    to robustly check for bare legs, ignoring the model's confused clothing labels.

    Args:
        player_crop (np.ndarray): The cropped BGR image of the person.
        segmenter (ClothingSegmenter): The shared segmentation model instance.

    Returns:
        bool: True if the person is wearing long pants (referee), False otherwise.
    """
    if player_crop.size == 0:
        return False

    # 1. Get the segmentation mask
    pred_seg = segmenter.get_segmentation_mask(player_crop)
    if pred_seg.size == 0:
        return False
        
    # 2. Create a binary mask of the PERSON (Class 0 is background)
    # We don't care if it's pants, shirt, or skin - just get the human silhouette
    person_mask = (pred_seg != 0).astype(np.uint8)
    
    # 3. Focus on the bottom 40% of the crop (where the legs are)
    h, w = player_crop.shape[:2]
    split_y = int(h * 0.6)
    
    lower_body_img = player_crop[split_y:, :]
    lower_person_mask = person_mask[split_y:, :]
    
    # Count how many pixels actually belong to the person in this lower section
    lower_person_pixels = np.sum(lower_person_mask)
    if lower_person_pixels == 0:
        return False
        
    # 4. Detect Skin Tones (HSV) ONLY inside the person's silhouette
    hsv_lower = cv2.cvtColor(lower_body_img, cv2.COLOR_BGR2HSV)
    
    # Broad HSV range for skin tones (covers light to dark skin)
    lower_skin = np.array([0, 20, 50], dtype=np.uint8)
    upper_skin = np.array([25, 170, 255], dtype=np.uint8)
    
    skin_mask_hsv = cv2.inRange(hsv_lower, lower_skin, upper_skin)
    
    # INTERSECTION: Pixels that are skin color AND belong to the person (not the floor)
    valid_skin_mask = cv2.bitwise_and(skin_mask_hsv, skin_mask_hsv, mask=lower_person_mask)
    skin_pixels = cv2.countNonZero(valid_skin_mask)
    
    # 5. Calculate the ratio of skin relative to the person's lower body area
    skin_ratio = skin_pixels / lower_person_pixels
    
    # Referees (long pants) will have almost 0 skin. Players will have a high ratio.
    # If skin is less than 5% of their visible lower body, they wear long pants.
    return skin_ratio < 0.05