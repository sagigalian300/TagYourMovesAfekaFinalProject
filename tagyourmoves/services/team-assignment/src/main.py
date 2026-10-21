"""
Main entry point for the team_assignment microservice.
Orchestrates data fetching, shirt extraction, team clustering, and visualization.
"""

import cv2
import numpy as np
from contract.frame.Frame import Frame

from mock.mock import get_mock_data 
from src.shirt_extractor import extract_shirt_from_player
from src.shirt_segmentation import ShirtSegmenter
from src.team_clustering import assign_teams_to_players

def process_frame(image: np.ndarray, current_frame: Frame, segmenter: ShirtSegmenter) -> Frame:
    """Processes a single frame to assign teams to detected players.

    Args:
        image (np.ndarray): The raw image/frame array.
        current_frame (Frame): The Frame object with detected players.
        segmenter (ShirtSegmenter): The loaded segmentation model.

    Returns:
        Frame: The updated Frame object with player 'team' attributes filled.
    """
    shirt_crops = []
    valid_players = []

    # 1. Extract shirt crops for all players (Segmentation happens here)
    for player in current_frame.players:
        shirt_crop = extract_shirt_from_player(image, player, segmenter)
        
        if shirt_crop.size > 0:
            shirt_crops.append(shirt_crop)
            valid_players.append(player)
            
    # 2. Run clustering logic to assign teams
    assign_teams_to_players(valid_players, shirt_crops)
    
    return current_frame

def draw_teams_on_image(image: np.ndarray, frame: Frame, output_path: str) -> None:
    """Draws bounding boxes and team labels on the image for visualization."""
    for player in frame.players:
        if player.team == 1:
            color = (255, 0, 0)  # Blue
        elif player.team == 2:
            color = (0, 0, 255)  # Red
        else:
            color = (0, 255, 0)  # Green

        p_x1, p_y1 = int(player.x1), int(player.y1)
        p_x2, p_y2 = int(player.x2), int(player.y2)

        cv2.rectangle(image, (p_x1, p_y1), (p_x2, p_y2), color, 2)
        label = f"Team {player.team}"
        cv2.putText(image, label, (p_x1, p_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imwrite(output_path, image)
    print(f"Visualized image saved to: {output_path}")

if __name__ == "__main__":
    print("Initializing AI Models (Loading SegFormer)...")
    # Load the model ONCE here
    segmenter = ShirtSegmenter()
    
    print("Fetching mock data...")
    img_path = 'mock/mock-img-team-assignment5.png'
    model_path = 'mock/yolo26s.pt'
    output_result_path = 'mock/final-teams-result.png'
    
    img, frame = get_mock_data(img_path, model_path)
    
    if img is not None and frame is not None:
        print("Processing frame for team assignment...")
        updated_frame = process_frame(img, frame, segmenter)
        
        print("Finished processing. Visualizing results...")
        draw_teams_on_image(img, updated_frame, output_result_path)