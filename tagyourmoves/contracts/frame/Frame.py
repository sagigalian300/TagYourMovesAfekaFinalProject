from contract.ball.Ball import Ball
from contract.rim.Rim import Rim
from contract.player.Player import Player

class Frame:
    def __init__(self, frame_number, timestamp_ms, ball=None, rims=None, players=None):
        self.frame_number = frame_number
        
        # time in ms from the start of the vid.
        # needed so when we find a move (like a throw)
        # we immidiatly can tag the mili second it occured
        self.timestamp_ms = timestamp_ms

        self.ball = ball
        
        if rims is None:
            self.rims = []
        else:
            self.rims = rims
            
        if players is None:
            self.players = []
        else:
            self.players = players