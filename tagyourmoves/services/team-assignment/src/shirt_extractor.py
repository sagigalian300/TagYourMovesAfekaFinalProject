"""
Module for extracting the player crop and isolating the shirt region.
"""

import numpy as np
from contract.player.Player import Player
from src.shirt_segmentation import ShirtSegmenter

def extract_shirt_from_player(image: np.ndarray, player: Player, segmenter: ShirtSegmenter) -> np.ndarray:
    """Extracts the player from the frame and isolates the shirt.

    First crops the entire player bounding box from the original image, 
    then uses the semantic segmentation model to isolate shirt pixels.

    Args:
        image (np.ndarray): The full original video frame/image.
        player (Player): The player object containing bounding box coordinates.
        segmenter (ShirtSegmenter): The initialized semantic segmenter.

    Returns:
        np.ndarray: A cropped image of the player where all non-shirt 
            pixels are colored black.
    """
    img_height, img_width = image.shape[:2]
    
    # Extract coordinates and ensure they are within image bounds
    p_x1 = max(0, int(player.x1))
    p_y1 = max(0, int(player.y1))
    p_x2 = min(img_width, int(player.x2))
    p_y2 = min(img_height, int(player.y2))
    
    # Crop the exact player bounding box
    player_crop = image[p_y1:p_y2, p_x1:p_x2]
    
    if player_crop.size == 0:
        return np.array([])
        
    # Isolate shirt pixels using SegFormer
    shirt_only_img = segmenter.extract_shirt(player_crop)
    
    return shirt_only_img