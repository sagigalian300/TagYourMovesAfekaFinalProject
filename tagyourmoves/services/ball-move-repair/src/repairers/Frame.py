from typing import List, Optional
from Ball import Ball
from Rim import Rim
from Player import Player

class Frame:
    """
    Represents a single video frame containing all detected objects.

    Attributes:
        frame_number (int): The sequential index of the frame in the video.
        timestamp_ms (float): Time in milliseconds from the start of the video.
        ball (Optional[Ball]): The detected ball in the frame, if present.
        rims (List[Rim]): A list of all detected rims in the frame.
        players (List[Player]): A list of all detected players in the frame.
    """

    def __init__(self, 
        frame_number: int, 
        timestamp_ms: float, 
        ball: Optional[Ball] = None, 
        rims: Optional[List[Rim]] = None, 
        players: Optional[List[Player]] = None
    ):
        """
        Initializes the Frame instance.

        Args:
            frame_number (int): The sequential index of the frame in the video.
            timestamp_ms (float): Time in ms from the start of the video. Needed so when 
                we find a move (like a throw) we immediately can tag the millisecond it occurred.
            ball (Optional[Ball], optional): The detected ball object. Defaults to None.
            rims (Optional[List[Rim]], optional): A list of detected rims. Defaults to None.
            players (Optional[List[Player]], optional): A list of detected players. Defaults to None.
        """
        self.frame_number: int = frame_number
        self.timestamp_ms: float = timestamp_ms
        self.ball: Optional[Ball] = ball
        
        # Safely initialize mutable lists to avoid shared state bugs
        self.rims: List[Rim] = rims if rims is not None else []
        self.players: List[Player] = players if players is not None else []