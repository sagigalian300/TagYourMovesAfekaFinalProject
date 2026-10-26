"""
Main entry point for the object_detection microservice.

Runs a FastAPI server that handles image payloads, runs YOLO-based object 
detection to find players, rims, and the ball, and returns a structured Frame.
"""

import os
import cv2
import numpy as np
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, UploadFile, HTTPException

from src.detector import ObjectDetector
from src.visualization import draw_detections_on_image

# Global dictionary to hold the AI model in memory
shared_resources = {}

# Constants
MODEL_PATH = os.path.join("src", "detectors", "yolo26s.pt")
DEBUG_DIR = os.path.join("mock", "debug_outputs")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifespan of the FastAPI application.

    Loads the YOLO model into memory once during startup to prevent loading delays 
    on incoming requests.

    Args:
        app (FastAPI): The FastAPI application instance.
    """
    print(f"[Startup] Initializing YOLO model from {MODEL_PATH}...")
    try:
        shared_resources["detector"] = ObjectDetector(MODEL_PATH)
        print("[Startup] Model loaded successfully. Server is ready.")
    except Exception as e:
        print(f"[Startup Error] Failed to load model: {e}")
    
    yield
    
    print("[Shutdown] Cleaning up resources...")
    shared_resources.clear()

app = FastAPI(title="Object Detection Microservice", lifespan=lifespan)

@app.post("/detect")
async def detect_objects_endpoint(
    image: UploadFile = File(...),
    frame_number: int = Form(...),
    timestamp_ms: float = Form(...)
):
    """
    Endpoint to process an image, detect basketball entities, and return Frame data.

    Args:
        image (UploadFile): The binary image file.
        frame_number (int): The sequential index of the frame.
        timestamp_ms (float): Time in milliseconds from the start of the video.

    Returns:
        Frame: The processed Frame object populated with detected objects.
        
    Raises:
        HTTPException: If the image decoding fails or the model is not loaded.
    """
    detector: ObjectDetector = shared_resources.get("detector")
    if not detector:
        raise HTTPException(status_code=500, detail="Object detector is not initialized.")

    try:
        # Read and decode the uploaded image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # Run detection
    detected_frame = detector.detect(img, frame_number, timestamp_ms)

    # Save debug visualization
    os.makedirs(DEBUG_DIR, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    debug_path = os.path.join(DEBUG_DIR, f"detection_frame_{frame_number}_{timestamp_str}.png")
    
    draw_detections_on_image(img.copy(), detected_frame, debug_path)

    return detected_frame