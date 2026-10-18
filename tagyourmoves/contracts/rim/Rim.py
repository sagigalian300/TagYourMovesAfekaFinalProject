class Rim:
    def __init__(self, confidence, x1, y1, x2, y2):
        self.confidence = confidence
        
        # ==============================================================
        # BBOX COORDINATES EXPLANATION
        # x1, y1 represent the TOP-LEFT corner of the bounding box.
        # x2, y2 represent the BOTTOM-RIGHT corner of the bounding box.
        # ==============================================================
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2