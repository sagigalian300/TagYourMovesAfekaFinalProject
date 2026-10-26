"""
Module for object detection using YOLO.
"""

from typing import List
import numpy as np
from ultralytics import YOLO

from contract.frame.Frame import Frame
from contract.ball.Ball import Ball
from contract.player.Player import Player
from contract.rim.Rim import Rim

class ObjectDetector:
    """
    Handles loading the YOLO model and running inference on images to extract 
    detected objects (Ball, Players, Rims).
    """

    # Class IDs based on the trained model
    CLASS_BALL = 0
    CLASS_PLAYER = 1
    CLASS_RIM = 2

    def __init__(self, model_path: str):
        """
        Initializes the ObjectDetector and loads the YOLO model.

        Args:
            model_path (str): The file path to the YOLO model weights (.pt file).
        """
        self.model = YOLO(model_path)

    def detect(self, image: np.ndarray, frame_number: int, timestamp_ms: float) -> Frame:
        """
        Runs object detection on a single image and maps the results to a Frame object.

        Args:
            image (np.ndarray): The raw image array (BGR format) to process.
            frame_number (int): The sequential index of the frame.
            timestamp_ms (float): Time in milliseconds from the start of the video.

        Returns:
            Frame: A Frame object containing the detected ball, players, and rims.
        """
        # Run inference (conf=0.25 is a standard default, can be adjusted)
        results = self.model(image, conf=0.25, verbose=False)
        
        best_ball = None
        players: List[Player] = []
        rims: List[Rim] = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                # Extract box coordinates and metadata
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                if class_id == self.CLASS_BALL:
                    # If multiple balls are detected, keep the one with the highest confidence
                    if best_ball is None or confidence > best_ball.confidence:
                        best_ball = Ball(confidence, x1, y1, x2, y2)
                
                elif class_id == self.CLASS_PLAYER:
                    players.append(Player(confidence, x1, y1, x2, y2))
                
                elif class_id == self.CLASS_RIM:
                    rims.append(Rim(confidence, x1, y1, x2, y2))

        return Frame(
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
            ball=best_ball,
            rims=rims,
            players=players
        )