from typing import Optional

class Player:
    """
    Represents a detected player in a frame.

    Attributes:
        confidence (float): The confidence score of the detection.
        x1 (float): The x-coordinate of the TOP-LEFT corner of the bounding box.
        y1 (float): The y-coordinate of the TOP-LEFT corner of the bounding box.
        x2 (float): The x-coordinate of the BOTTOM-RIGHT corner of the bounding box.
        y2 (float): The y-coordinate of the BOTTOM-RIGHT corner of the bounding box.
        tracking_id (Optional[int]): The ID assigned by the tracking algorithm.
        team (Optional[int]): The team identifier for the player.
        shirt_number (Optional[int]): The identified shirt number of the player.
        has_ball (bool): Indicates whether the player is currently holding the ball.
    """

    def __init__(self, 
        confidence: float, 
        x1: float, 
        y1: float, 
        x2: float, 
        y2: float, 
        tracking_id: Optional[int] = None, 
        team: Optional[int] = None, 
        shirt_number: Optional[int] = None
    ):
        """
        Initializes the Player instance.

        Args:
            confidence (float): The model's confidence score for this detection.
            x1 (float): TOP-LEFT x-coordinate of the bounding box.
            y1 (float): TOP-LEFT y-coordinate of the bounding box.
            x2 (float): BOTTOM-RIGHT x-coordinate of the bounding box.
            y2 (float): BOTTOM-RIGHT y-coordinate of the bounding box.
            tracking_id (Optional[int], optional): Can be None because YOLO alone does NOT 
                (always) assign IDs. It will only get a value AFTER the tracking algorithm 
                processes the frame. Defaults to None.
            team (Optional[int], optional): None -> no team assigned or judge, 1 -> player related to team1, 
                2 -> player related to team2. Defaults to None.
            shirt_number (Optional[int], optional): Can be None if the model didn't catch 
                the number. Defaults to None.
        """
        self.confidence: float = confidence
        self.x1: float = x1
        self.y1: float = y1
        self.x2: float = x2
        self.y2: float = y2
        
        self.tracking_id: Optional[int] = tracking_id
        self.team: Optional[int] = team
        self.shirt_number: Optional[int] = shirt_number
        
        # Initialized with False by default
        self.has_ball: bool = False