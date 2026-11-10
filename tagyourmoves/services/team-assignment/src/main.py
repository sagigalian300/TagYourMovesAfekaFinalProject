"""
Main entry point for the team_assignment microservice.

Runs a FastAPI server that handles image and JSON payloads, processes team 
assignments using a shared AI model, and returns the updated Frame data.
"""

import cv2
import json
from types import SimpleNamespace
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from contract.frame.Frame import Frame
from src.clothing_segmentation import ClothingSegmenter
from src.long_pants_detector import has_long_pants
from src.shirt_extractor import extract_shirt_from_player
from src.team_clustering import assign_teams_to_players

# Global dictionary to hold the AI model in memory while the server runs
shared_resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the lifespan of the FastAPI application.

    Loads the heavy AI models into memory once during startup and cleans them 
    up during shutdown. This prevents out-of-memory (OOM) errors and eliminates 
    loading delays during incoming requests.

    Args:
        app (FastAPI): The FastAPI application instance.
    """
    print("[Startup] Initializing AI Models (Loading single shared SegFormer)...")
    shared_resources["segmenter"] = ClothingSegmenter()
    print("[Startup] Server is ready to accept requests.")
    
    yield
    
    print("[Shutdown] Cleaning up resources...")
    shared_resources.clear()

app = FastAPI(title="Team Assignment Microservice", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Allow Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_frame(image: np.ndarray, current_frame: Frame, segmenter: ClothingSegmenter) -> Frame:
    """Processes a single frame to assign teams to detected players.

    Iterates through the players in the frame, identifies referees using 
    semantic segmentation, extracts shirt crops for actual players, and 
    assigns team IDs using clustering.

    Args:
        image (np.ndarray): The raw image array (BGR format).
        current_frame (Frame): The Frame object containing detected players.
        segmenter (ClothingSegmenter): The shared semantic segmentation model.

    Returns:
        Frame: The updated Frame object with player 'team' attributes populated.
    """
    shirt_crops = []
    valid_players = []
    img_height, img_width = image.shape[:2]

    for player in current_frame.players:
        p_x1, p_y1 = max(0, int(player.x1)), max(0, int(player.y1))
        p_x2, p_y2 = min(img_width, int(player.x2)), min(img_height, int(player.y2))
        
        player_crop = image[p_y1:p_y2, p_x1:p_x2]
        if player_crop.size == 0:
            continue

        # Filter Referees uniformly using the shared segmenter
        if has_long_pants(player_crop, segmenter):
            player.team = 0
            continue

        # Extract shirts for valid players uniformly using the shared segmenter
        shirt_crop = extract_shirt_from_player(image, player, segmenter)
        if shirt_crop.size > 0:
            shirt_crops.append(shirt_crop)
            valid_players.append(player)
            
    # Cluster valid players into teams
    assign_teams_to_players(valid_players, shirt_crops)
    
    return current_frame

@app.post("/assign-teams")
async def assign_teams_endpoint(
    image: UploadFile = File(...),
    frame_data: str = Form(...)
):
    """Endpoint to process a frame, assign teams, and return the updated data.

    Accepts multipart/form-data containing the image file and a JSON string 
    representing the Frame object.

    Args:
        image (UploadFile): The binary image file.
        frame_data (str): A JSON string representing the input Frame object.

    Returns:
        Frame: The processed Frame object with assigned teams.
        
    Raises:
        HTTPException: If the image decoding fails or the JSON is invalid.
    """
    try:
        frame_dict = json.loads(frame_data)

        if "players" in frame_dict:
            frame_dict["players"] = [SimpleNamespace(**p) for p in frame_dict["players"]]    
        current_frame = Frame(**frame_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid frame JSON data: {e}")

    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    segmenter = shared_resources["segmenter"]
    updated_frame = process_frame(img, current_frame, segmenter)

    return updated_frame