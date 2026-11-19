"""
Module for classifying players into teams based on shirt colors using K-Means clustering.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
from contract.player.Player import Player


def extract_dominant_color(shirt_img: np.ndarray) -> np.ndarray:
    """Extracts the dominant color from isolated shirt pixels.

    Filters out the black background pixels created by the segmentation mask,
    converts the remaining shirt pixels to CIE LAB, and uses K-Means to find
    the dominant color of the fabric.

    Args:
        shirt_img (np.ndarray): The masked BGR image containing only the shirt.

    Returns:
        np.ndarray: A 1D array representing the dominant LAB color [L, a, b].
            Returns a default [0, 0, 0] if no valid pixels exist.
    """
    mask = np.any(shirt_img != [0, 0, 0], axis=-1)
    valid_bgr_pixels = shirt_img[mask]

    if len(valid_bgr_pixels) == 0:
        return np.array([0, 0, 0])

    # Convert pixels to CIE LAB (Requires reshaping to 2D for cv2.cvtColor)
    valid_bgr_pixels_2d = valid_bgr_pixels.reshape(1, -1, 3)
    valid_lab_pixels = cv2.cvtColor(valid_bgr_pixels_2d, cv2.COLOR_BGR2LAB).reshape(-1, 3)

    # Randomly sample pixels if there are too many (speeds up KMeans)
    if len(valid_lab_pixels) > 500:
        indices = np.random.choice(len(valid_lab_pixels), 500, replace=False)
        sampled_pixels = valid_lab_pixels[indices]
    else:
        sampled_pixels = valid_lab_pixels

    # Run Mini K-Means to find the true fabric color (ignoring logos/stripes)
    kmeans = KMeans(n_clusters=min(2, len(sampled_pixels)), random_state=42, n_init='auto')
    labels = kmeans.fit_predict(sampled_pixels)

    counts = np.bincount(labels)
    dominant_label = np.argmax(counts)

    return kmeans.cluster_centers_[dominant_label]


def _lab_to_bgr(lab_color: np.ndarray) -> Tuple[int, int, int]:
    """Converts a single CIE LAB color vector to a BGR tuple for OpenCV drawing.

    Args:
        lab_color (np.ndarray): A 1D array representing [L, a, b].

    Returns:
        Tuple[int, int, int]: The corresponding (B, G, R) integer tuple.
    """
    lab_pixel = np.uint8([[np.clip(lab_color, 0, 255)]])
    bgr_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)[0][0]
    return int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])


def assign_teams_to_players(
    players: List[Player],
    shirt_crops: List[np.ndarray]
) -> Dict[int, Tuple[int, int, int]]:
    """Analyzes shirt colors and assigns team IDs to players.

    Uses K-Means clustering to find the two dominant team colors across all
    player crops. Updates the 'team' attribute of each Player object in-place.

    Args:
        players (List[Player]): A list of Player objects to be updated.
        shirt_crops (List[np.ndarray]): A list of cropped shirt images corresponding
            to the players list.

    Returns:
        Dict[int, Tuple[int, int, int]]: Mapping of player object id() to its
            extracted dominant shirt color in BGR format (useful for debugging).
    """
    if len(players) < 2:
        print("[Clustering] Not enough players detected to form teams.")
        return {}

    color_features = []
    dominant_colors_bgr: Dict[int, Tuple[int, int, int]] = {}

    for player, crop in zip(players, shirt_crops):
        dom_lab = extract_dominant_color(crop)
        color_features.append(dom_lab)
        dominant_colors_bgr[id(player)] = _lab_to_bgr(dom_lab)

    X = np.array(color_features)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X)

    for i, player in enumerate(players):
        player.team = int(labels[i]) + 1

    print(f"[Clustering] Successfully assigned {len(players)} players into teams.")
    return dominant_colors_bgr