class Player:
    def __init__(self, confidence, x1, y1, x2, y2, tracking_id=None, team=0, shirt_number=None):
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
        
        # 'tracking_id' can be None because YOLO alone does NOT (always) assign IDs.
        # It will only get a value AFTER the tracking algorithm processes the frame.
        self.tracking_id = tracking_id

        # team=0 -> no team assigned or judge
        # team=1 -> player related to team1
        # team=2 -> player related to team2
        self.team = team

        # shirt_number can be None if the model didnt catch the number
        self.shirt_number = shirt_number

        # has_ball initialized with False
        self.has_ball = False
        
        

        