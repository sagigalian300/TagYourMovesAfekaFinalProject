"""
This file is used to create mock data from an image to test the team_assignment 
microservice without depending on the object-detection microservice.
"""

import cv2
from typing import List
from ultralytics import YOLO

# =====================================================================
# CONTRACT IMPORTS
# =====================================================================
from contract.player.Player import Player
from contract.frame.Frame import Frame


def get_mock_data(img_path: str, model_path: str) -> None:
    """
    Runs YOLO inference on an image, return the bbox + image to the main.

    Args:
        img_path (str): The relative path to the input image.
        model_path (str): The relative path to the YOLO model weights.
    """
    # Load the model and read the image
    model = YOLO(model_path) 
    image = cv2.imread(img_path)
    
    if image is None:
        print(f"Error: Could not load image from {img_path}")
        return

    # Run inference on the image
    # results is a list of Results objects, we take the first one [0] since we pass one image
    results = model(image, conf=0.6)[0]
    
    players_list: List[Player] = []
    
    # YOLO typically uses class ID 0 for 'person' in the standard COCO dataset.
    # Adjust this if your custom YOLO model uses a different ID for players.
    PLAYER_CLASS_ID = 1 
    
    # Extract bounding boxes and populate the Player objects
    for box in results.boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        if cls_id == PLAYER_CLASS_ID:
            # Extract coordinates and unpack them
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            
            # Create a Player instance and append to our list
            player_obj = Player(confidence=confidence, x1=x1, y1=y1, x2=x2, y2=y2)
            players_list.append(player_obj)
            
    # Create the Frame object that holds all detected players
    current_frame = Frame(frame_number=1, timestamp_ms=0.0, players=players_list)
    
    # Draw rectangles using the coordinates saved in our Player objects
    for player in current_frame.players:
        start_point = (int(player.x1), int(player.y1))
        end_point = (int(player.x2), int(player.y2))
        
        # BGR format (Green color)
        color = (0, 255, 0) 
        thickness = 2
        cv2.rectangle(image, start_point, end_point, color, thickness)
        
    return image, current_frame
