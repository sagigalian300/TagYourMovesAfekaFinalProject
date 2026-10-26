class Ball:
    """
    Represents a detected basketball in a frame.

    Attributes:
        confidence (float): The confidence score of the detection.
                    The origin (0,0) is always the TOP-LEFT of the image/frame.
        x1 (float): The x-coordinate of the TOP-LEFT corner of the bounding box.
        y1 (float): The y-coordinate of the TOP-LEFT corner of the bounding box.
        x2 (float): The x-coordinate of the BOTTOM-RIGHT corner of the bounding box.
        y2 (float): The y-coordinate of the BOTTOM-RIGHT corner of the bounding box.
    """

    def __init__(self, confidence: float, x1: float, y1: float, x2: float, y2: float):
        """
        Initializes the Ball instance.

        Args:
            confidence (float): The model's confidence score for this detection.
            x1 (float): TOP-LEFT x-coordinate of the bounding box.
            y1 (float): TOP-LEFT y-coordinate of the bounding box.
            x2 (float): BOTTOM-RIGHT x-coordinate of the bounding box.
            y2 (float): BOTTOM-RIGHT y-coordinate of the bounding box.
        """
        self.confidence: float = confidence
        self.x1: float = x1
        self.y1: float = y1
        self.x2: float = x2
        self.y2: float = y2