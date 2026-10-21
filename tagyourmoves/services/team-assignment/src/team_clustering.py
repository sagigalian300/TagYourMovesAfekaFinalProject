"""
Module for classifying players into teams based on shirt colors using K-Means clustering.
"""

import cv2
import numpy as np
from typing import List
from sklearn.cluster import KMeans
from contract.player.Player import Player

def extract_dominant_color(shirt_img: np.ndarray) -> np.ndarray:
    """Extracts the dominant color from isolated shirt pixels.

    Filters out the black background pixels created by the segmentation mask, 
    converts the remaining shirt pixels to HSV, and uses K-Means to find the 
    dominant color of the fabric.

    Args:
        shirt_img (np.ndarray): The masked BGR image containing only the shirt.

    Returns:
        np.ndarray: A 1D array representing the dominant HSV color [H, S, V].
            Returns a default [0, 0, 0] if no valid pixels exist.
    """
    # Create a boolean mask of non-black pixels
    mask = np.any(shirt_img != [0, 0, 0], axis=-1)
    
    # Extract valid BGR pixels (flattened to a 1D list of pixels)
    valid_bgr_pixels = shirt_img[mask]
    
    if len(valid_bgr_pixels) == 0:
        return np.array([0, 0, 0])
        
    # Convert pixels to HSV (Requires reshaping to 2D for cv2.cvtColor)
    valid_bgr_pixels_2d = valid_bgr_pixels.reshape(1, -1, 3)
    valid_hsv_pixels = cv2.cvtColor(valid_bgr_pixels_2d, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    
    # Randomly sample pixels if there are too many (speeds up KMeans)
    if len(valid_hsv_pixels) > 500:
        indices = np.random.choice(len(valid_hsv_pixels), 500, replace=False)
        sampled_pixels = valid_hsv_pixels[indices]
    else:
        sampled_pixels = valid_hsv_pixels
        
    # Run Mini K-Means to find the true fabric color (ignoring logos/stripes)
    kmeans = KMeans(n_clusters=min(2, len(sampled_pixels)), random_state=42, n_init='auto')
    labels = kmeans.fit_predict(sampled_pixels)
    
    counts = np.bincount(labels)
    dominant_label = np.argmax(counts)
    
    return kmeans.cluster_centers_[dominant_label]

def assign_teams_to_players(players: List[Player], shirt_crops: List[np.ndarray]) -> None:
    """Analyzes shirt colors and assigns team IDs to players.

    Uses K-Means clustering to find the two dominant team colors across all 
    player crops. Updates the 'team' attribute of each Player object in-place.

    Args:
        players (List[Player]): A list of Player objects to be updated.
        shirt_crops (List[np.ndarray]): A list of cropped shirt images corresponding 
            to the players list.
    """
    if len(players) < 2:
        print("[Clustering] Not enough players detected to form teams.")
        return

    color_features = []
    for crop in shirt_crops:
        color_features.append(extract_dominant_color(crop))
        
    X = np.array(color_features)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X)
    
    for i, player in enumerate(players):
        player.team = int(labels[i]) + 1
        
    print(f"[Clustering] Successfully assigned {len(players)} players into teams.")