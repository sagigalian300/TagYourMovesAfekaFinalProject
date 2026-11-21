"""
Main entry point for the player_signature microservice.

Runs a FastAPI server that handles image and JSON payloads, extracts visual 
appearance signatures (Re-ID embeddings) for detected players using a shared 
OSNet model, and returns the updated Frame data.
"""

import os
import json
from types import SimpleNamespace
from contextlib import asynccontextmanager
import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from contracts.frame.Frame import Frame
from src.signature_extractor import SignatureExtractor
from src.frame_processor import process_frame

# Global dictionary to hold the AI model in memory while the server runs
shared_resources = {}

# Constants
MODEL_PATH = os.path.join("models", "player_signature_model.pth")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the lifespan of the FastAPI application.

    Loads the OSNet Re-ID model with pretrained weights into memory once during
    startup and cleans it up during shutdown to prevent loading delays on
    incoming requests.

    Args:
        app (FastAPI): The FastAPI application instance.
    """
    print(f"[Startup] Initializing AI Model from {MODEL_PATH}...")
    shared_resources["extractor"] = SignatureExtractor(
        model_name="osnet_ain_x1_0",
        model_path=MODEL_PATH
    )

    print(f"[Device Check] Is PyTorch CUDA available? {torch.cuda.is_available()}")
    print(f"[Device Check] Model is loaded on: {shared_resources['extractor'].device}")
    print("[Startup] Server is ready to accept requests.")

    yield

    print("[Shutdown] Cleaning up resources...")
    shared_resources.clear()


app = FastAPI(title="Player Signature Microservice", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/extract-signatures")
async def extract_signatures_endpoint(
    image: UploadFile = File(...),
    frame_data: str = Form(...)
):
    """Endpoint to process a frame, extract player signatures, and return updated data.

    Accepts multipart/form-data containing the image file and a JSON string 
    representing the Frame object.

    Args:
        image (UploadFile): The binary image file.
        frame_data (str): A JSON string representing the input Frame object.

    Returns:
        Frame: The processed Frame object with populated player signatures.

    Raises:
        HTTPException: If the JSON is invalid, image decoding fails, or model is missing.
    """
    extractor: SignatureExtractor = shared_resources.get("extractor")
    if not extractor:
        raise HTTPException(status_code=500, detail="Signature extractor is not initialized.")

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
            raise ValueError("Could not decode image.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    updated_frame = process_frame(img, current_frame, extractor)

    return updated_frame