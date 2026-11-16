from typing import List, Optional
from contracts.ball.Ball import Ball
from contracts.rim.Rim import Rim
from contracts.player.Player import Player

class Frame:
    """
    Represents a single video frame containing all detected objects.
    """
    def __init__(self, frame_number: int, timestamp_ms: float, 
                 ball: Optional[Ball] = None, rims: Optional[List[Rim]] = None, players: Optional[List[Player]] = None):
        self.frame_number: int = frame_number
        self.timestamp_ms: float = timestamp_ms
        self.ball: Optional[Ball] = ball
        self.rims: List[Rim] = rims if rims is not None else []
        self.players: List[Player] = players if players is not None else []
