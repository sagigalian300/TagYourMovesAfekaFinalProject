"""
Main entry point for the team_assignment microservice.
Orchestrates data fetching, referee filtering, shirt extraction, and clustering.
"""
import cv2
import numpy as np
from contract.frame.Frame import Frame

from mock.mock import get_mock_data 
from src.clothing_segmentation import ClothingSegmenter
from src.long_pants_detector import has_long_pants
from src.shirt_extractor import extract_shirt_from_player
from src.team_clustering import assign_teams_to_players


def process_frame(image: np.ndarray, current_frame: Frame, segmenter: ClothingSegmenter) -> Frame:
    """Processes a single frame to assign teams to detected players.

    Args:
        image (np.ndarray): The raw image/frame array.
        current_frame (Frame): The Frame object with detected players.
        segmenter (ClothingSegmenter): The single shared AI segmentation model.

    Returns:
        Frame: The updated Frame object with player 'team' attributes filled.
    """
    shirt_crops = []
    valid_players = []
    img_height, img_width = image.shape[:2]

    for i, player in enumerate(current_frame.players):
        p_x1, p_y1 = max(0, int(player.x1)), max(0, int(player.y1))
        p_x2, p_y2 = min(img_width, int(player.x2)), min(img_height, int(player.y2))
        
        player_crop = image[p_y1:p_y2, p_x1:p_x2]
        if player_crop.size == 0:
            continue


        # 1. Filter Referees uniformly using the shared segmenter
        if has_long_pants(player_crop, segmenter):
            player.team = 0
            print("[Process] Referee detected (long pants). Assigned to team 0.")
            continue

        # 2. Extract shirts for valid players uniformly using the shared segmenter
        shirt_crop = extract_shirt_from_player(image, player, segmenter)
        if shirt_crop.size > 0:
            shirt_crops.append(shirt_crop)
            valid_players.append(player)
            
    # 3. Clustering is commented out for now to test filtering independently
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
            color = (0, 255, 0)  # Green for Referee

        p_x1, p_y1 = int(player.x1), int(player.y1)
        p_x2, p_y2 = int(player.x2), int(player.y2)

        cv2.rectangle(image, (p_x1, p_y1), (p_x2, p_y2), color, 2)
        label = f"Team {player.team}"
        cv2.putText(image, label, (p_x1, p_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imwrite(output_path, image)
    print(f"Visualized image saved to: {output_path}")


if __name__ == "__main__":
    print("Initializing AI Models (Loading single shared SegFormer)...")
    shared_segmenter = ClothingSegmenter()
    
    print("Fetching mock data...")
    img_path = 'mock/mock-img-team-assignment5.png' 
    model_path = 'mock/yolo26s.pt'
    output_result_path = 'mock/final-teams-result.png'
    
    img, frame = get_mock_data(img_path, model_path)
    
    if img is not None and frame is not None:
        print("Processing frame for team assignment...")
        updated_frame = process_frame(img, frame, shared_segmenter)
        
        print("Finished processing. Visualizing results...")
        draw_teams_on_image(img, updated_frame, output_result_path)