from typing import Optional

class Player:
    """
    Represents a detected player in a frame.
    """
    def __init__(self, confidence: float, x1: float, y1: float, x2: float, y2: float, 
                 tracking_id: Optional[int] = None, team: Optional[int] = None, shirt_number: Optional[int] = None):
        self.confidence: float = confidence
        self.x1: float = x1
        self.y1: float = y1
        self.x2: float = x2
        self.y2: float = y2
        self.tracking_id: Optional[int] = tracking_id
        self.team: Optional[int] = team
        self.shirt_number: Optional[int] = shirt_number
        self.has_ball: bool = False
