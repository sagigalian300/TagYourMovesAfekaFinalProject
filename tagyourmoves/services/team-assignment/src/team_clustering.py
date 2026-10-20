"""
Module for classifying players into teams based on shirt colors using K-Means clustering.
"""

import cv2
import numpy as np
from typing import List
from sklearn.cluster import KMeans
from contract.player.Player import Player


def extract_dominant_color(crop: np.ndarray) -> np.ndarray:
    """Extracts the dominant color from a player's shirt crop.

    Focuses on the central area of the crop to minimize background noise and 
    converts the image to HSV to better handle lighting differences. Uses the 
    median color instead of the mean to reduce the impact of outlier pixels 
    (like logos, numbers, or skin).

    Args:
        crop (np.ndarray): The cropped image array (BGR format from OpenCV).

    Returns:
        np.ndarray: A 1D array representing the median HSV color [H, S, V].
    """
    h, w = crop.shape[:2]
    
    # Take the center 50% of the crop to avoid background and arms
    center_crop = crop[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)]
    
    # Fallback if the crop is extremely small
    if center_crop.size == 0:
        center_crop = crop
        
    # Convert from BGR (OpenCV default) to HSV color space
    hsv_crop = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
    
    # Calculate the median color across the spatial dimensions (axis 0 and 1)
    median_color = np.median(hsv_crop, axis=(0, 1))
    
    return median_color


def assign_teams_to_players(players: List[Player], shirt_crops: List[np.ndarray]) -> None:
    """Analyzes shirt colors and assigns team IDs to players.

    Uses K-Means clustering to find the two dominant team colors across all 
    player crops. Updates the 'team' attribute of each Player object in-place.

    Args:
        players (List[Player]): A list of Player objects to be updated.
        shirt_crops (List[np.ndarray]): A list of cropped shirt images corresponding 
            to the players list (must be in the exact same order).
    """
    # K-Means requires at least as many samples as clusters
    if len(players) < 2:
        print("[Clustering] Not enough players detected to form teams.")
        return

    # Extract feature vectors (dominant colors) for all players
    color_features = []
    for crop in shirt_crops:
        dominant_color = extract_dominant_color(crop)
        color_features.append(dominant_color)
        
    # Convert to a 2D numpy array for Scikit-Learn processing
    X = np.array(color_features)
    
    # Run K-Means clustering
    # n_clusters=2 assumes 2 teams. n_init='auto' suppresses Sklearn warnings.
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X)
    
    # Assign the results back to the Player objects
    for i, player in enumerate(players):
        # K-Means returns labels 0 and 1. We map them to Team 1 and Team 2.
        player.team = int(labels[i]) + 1
        
    print(f"[Clustering] Successfully assigned {len(players)} players into 2 teams.")