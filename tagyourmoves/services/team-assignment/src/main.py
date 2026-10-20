"""
Main entry point for the team_assignment microservice.
Orchestrates data fetching, shirt extraction, team clustering, and visualization.
"""

import cv2
import numpy as np
from contract.frame.Frame import Frame

# Import the mock generator
from mock.mock import get_mock_data 
from src.shirt_extractor import crop_shirt
from src.team_clustering import assign_teams_to_players


def process_frame(image: np.ndarray, current_frame: Frame) -> Frame:
    """Processes a single frame to assign teams to detected players.

    Args:
        image (np.ndarray): The raw image/frame array.
        current_frame (Frame): The Frame object containing populated Player bounding boxes.

    Returns:
        Frame: The updated Frame object with player 'team' attributes filled.
    """
    shirt_crops = []
    valid_players = []

    # 1. Extract shirt crops for all players
    for player in current_frame.players:
        crop = crop_shirt(image, player)
        
        # Only process valid crops (avoiding edge cases where bounding box is empty)
        if crop.size > 0:
            shirt_crops.append(crop)
            valid_players.append(player)
            
    # 2. Run clustering logic to assign teams
    assign_teams_to_players(valid_players, shirt_crops)
    
    # 3. Return the fully processed frame
    return current_frame


def draw_teams_on_image(image: np.ndarray, frame: Frame, output_path: str) -> None:
    """Draws bounding boxes and team labels on the image for visualization.

    Iterates over all players in the frame and draws a colored bounding box 
    and a text label corresponding to their assigned team.

    Args:
        image (np.ndarray): The original image/frame.
        frame (Frame): The processed Frame object containing players with assigned teams.
        output_path (str): The relative path where the annotated image will be saved.
    """
    for player in frame.players:
        # Determine color based on team (Team 1: Blue, Team 2: Red, Other: Green)
        if player.team == 1:
            color = (255, 0, 0)  # OpenCV uses BGR format (Blue)
        elif player.team == 2:
            color = (0, 0, 255)  # OpenCV uses BGR format (Red)
        else:
            color = (0, 255, 0)  # Green for unassigned / referees

        p_x1, p_y1 = int(player.x1), int(player.y1)
        p_x2, p_y2 = int(player.x2), int(player.y2)

        # Draw the bounding box
        cv2.rectangle(image, (p_x1, p_y1), (p_x2, p_y2), color, 2)
        
        # Draw the team label above the bounding box
        label = f"Team {player.team}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, label, (p_x1, p_y1 - 10), font, 0.8, color, 2)

    # Save the visualized image to the disk
    cv2.imwrite(output_path, image)
    print(f"Visualized image saved to: {output_path}")


if __name__ == "__main__":
    print("Fetching mock data...")
    img_path = 'mock/mock-img-team-assignment2.png'
    model_path = 'mock/yolo26s.pt'
    output_result_path = 'src/final-teams-result2.png'
    
    # Fetch data (make sure get_mock_data is defined in mock.py to return image, frame)
    img, frame = get_mock_data(img_path, model_path)
    
    if img is not None and frame is not None:
        print("Processing frame for team assignment...")
        updated_frame = process_frame(img, frame)
        
        print("Finished processing. Visualizing results...")
        draw_teams_on_image(img, updated_frame, output_result_path)