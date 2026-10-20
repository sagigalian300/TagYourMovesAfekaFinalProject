"""
Module for extracting the shirt region from a detected player's bounding box.
"""

import numpy as np
from contract.player.Player import Player

def crop_shirt(image: np.ndarray, player: Player) -> np.ndarray:
    """Extracts the shirt region of interest (ROI) from the full image.

    Calculates a specific sub-region within the player's bounding box that 
    most likely contains the shirt (ignoring the head, legs, and background).

    Args:
        image (np.ndarray): The full original video frame/image.
        player (Player): The player object containing the full bounding box coordinates.

    Returns:
        np.ndarray: A cropped image (numpy array) containing only the player's shirt. 
            Returns an empty array if the bounding box is invalid.
    """
    img_height, img_width = image.shape[:2]
    
    p_x1, p_y1 = int(player.x1), int(player.y1)
    p_x2, p_y2 = int(player.x2), int(player.y2)
    
    box_width = p_x2 - p_x1
    box_height = p_y2 - p_y1
    
    # Shirt area estimation: 15%-55% of height, trim 20% from sides
    s_x1 = int(p_x1 + 0.20 * box_width)
    s_y1 = int(p_y1 + 0.15 * box_height)
    s_x2 = int(p_x2 - 0.20 * box_width)
    s_y2 = int(p_y1 + 0.55 * box_height)
    
    # Ensure coordinates do not exceed image boundaries
    s_x1, s_x2 = max(0, s_x1), min(img_width, s_x2)
    s_y1, s_y2 = max(0, s_y1), min(img_height, s_y2)
    
    # Return the cropped numpy array
    return image[s_y1:s_y2, s_x1:s_x2]