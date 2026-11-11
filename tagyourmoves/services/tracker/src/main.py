import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from src.tracker_engine import TrackerEngine
from contract.frame.Frame import Frame
from contract.player.Player import Player
from contract.ball.Ball import Ball
from contract.rim.Rim import Rim

# Global dictionary to hold tracking sessions in memory
# Key: session_id (str), Value: TrackerEngine instance
active_sessions = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Tracker MS is ready to accept video streams.")
    yield
    print("[Shutdown] Cleaning up tracker sessions...")
    active_sessions.clear()

app = FastAPI(title="Tracking Microservice", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Schemas for API Payload Validation ---
class BallSchema(BaseModel):
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

class RimSchema(BaseModel):
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

class PlayerSchema(BaseModel):
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    tracking_id: Optional[int] = None
    team: Optional[int] = None
    shirt_number: Optional[int] = None
    has_ball: bool = False

class FramePayload(BaseModel):
    session_id: str
    frame_number: int
    timestamp_ms: float
    ball: Optional[BallSchema] = None
    rims: Optional[List[RimSchema]] = []
    players: Optional[List[PlayerSchema]] = []

# --- API Endpoints ---

@app.post("/track")
async def track_frame_endpoint(payload: FramePayload):
    """
    Receives detection data for a single frame, updates the tracking state,
    and assigns tracking_ids to players.
    """
    # 1. Retrieve or initialize the tracker engine for this specific video session
    session_id = payload.session_id
    if session_id not in active_sessions:
        active_sessions[session_id] = TrackerEngine()
    
    tracker: TrackerEngine = active_sessions[session_id]

    # 2. Convert incoming Pydantic payload to your Domain Objects (contract)
    domain_players = [
        Player(p.confidence, p.x1, p.y1, p.x2, p.y2, p.tracking_id, p.team, p.shirt_number)
        for p in payload.players
    ]
    
    domain_ball = Ball(
        payload.ball.confidence, payload.ball.x1, payload.ball.y1, payload.ball.x2, payload.ball.y2
    ) if payload.ball else None
    
    domain_rims = [
        Rim(r.confidence, r.x1, r.y1, r.x2, r.y2) for r in payload.rims
    ]

    domain_frame = Frame(
        frame_number=payload.frame_number,
        timestamp_ms=payload.timestamp_ms,
        ball=domain_ball,
        rims=domain_rims,
        players=domain_players
    )

    # 3. Run the tracking algorithm (Kalman Filter + Hungarian Algorithm updates IDs)
    tracked_frame = tracker.update(domain_frame)

    # 4. Return the updated frame (FastAPI automatically serializes your objects to JSON)
    return tracked_frame

@app.delete("/session/{session_id}")
async def end_session(session_id: str):
    """
    Clears the tracker state when a video finishes or the user restarts the process.
    """
    if session_id in active_sessions:
        del active_sessions[session_id]
        return {"status": "success", "message": f"Session {session_id} cleared."}
    raise HTTPException(status_code=404, detail="Session not found.")