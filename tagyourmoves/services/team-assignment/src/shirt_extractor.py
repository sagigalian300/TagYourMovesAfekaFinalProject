"""
Module for extracting the player crop and isolating the shirt region.
"""

import numpy as np
from contract.player.Player import Player
from src.clothing_segmentation import ClothingSegmenter


def extract_shirt_from_player(image: np.ndarray, player: Player, segmenter: ClothingSegmenter) -> np.ndarray:
    """Extracts the player from the frame and isolates the shirt.

    Args:
        image (np.ndarray): The full original video frame/image.
        player (Player): The player object containing bounding box coordinates.
        segmenter (ClothingSegmenter): The shared segmentation model instance.

    Returns:
        np.ndarray: A cropped image of the player where all non-shirt 
            pixels are colored black.
    """
    img_height, img_width = image.shape[:2]
    
    p_x1, p_y1 = max(0, int(player.x1)), max(0, int(player.y1))
    p_x2, p_y2 = min(img_width, int(player.x2)), min(img_height, int(player.y2))
    
    player_crop = image[p_y1:p_y2, p_x1:p_x2]
    if player_crop.size == 0:
        return np.array([])
        
    # Get the raw mask and isolate the shirt (Class ID 4)
    pred_seg = segmenter.get_segmentation_mask(player_crop)
    if pred_seg.size == 0:
        return np.array([])

    shirt_mask = (pred_seg == 4).astype(np.uint8)
    
    shirt_only_img = np.zeros_like(player_crop)
    shirt_only_img[shirt_mask == 1] = player_crop[shirt_mask == 1]
    
    return shirt_only_img