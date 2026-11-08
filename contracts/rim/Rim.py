class Rim:
    """
    Represents a detected basketball rim in a frame.
    """
    def __init__(self, confidence: float, x1: float, y1: float, x2: float, y2: float):
        self.confidence: float = confidence
        self.x1: float = x1
        self.y1: float = y1
        self.x2: float = x2
        self.y2: float = y2
