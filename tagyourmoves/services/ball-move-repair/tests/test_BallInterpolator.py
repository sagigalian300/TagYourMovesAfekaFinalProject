from src.repairers.Ball import Ball
from src.repairers.Frame import Frame
from src.repairers.BallInterpolator import BallInterpolator 

# 1. Create dummy frames with missing balls (None)
frames = [
    # Frame 1: Missing before any detection (Should back-fill using Frame 2)
    Frame(frame_number=1, timestamp_ms=100.0, ball=None), 
    
    # Frame 2: Known Detection
    Frame(frame_number=2, timestamp_ms=200.0, ball=Ball(0.95, 10.0, 10.0, 20.0, 20.0)),
    
    # Frame 3: Missing in the middle (Should interpolate exactly halfway between Frame 2 and 4)
    Frame(frame_number=3, timestamp_ms=300.0, ball=None), 
    
    # Frame 4: Known Detection
    Frame(frame_number=4, timestamp_ms=400.0, ball=Ball(0.88, 30.0, 30.0, 40.0, 40.0)),
    
    # Frame 5: Missing at the end (Should forward-fill using Frame 4)
    Frame(frame_number=5, timestamp_ms=500.0, ball=None)  
]

# 2. Initialize and run the interpolator
interpolator = BallInterpolator(frames)
interpolator.repair_missing_balls()

# 3. Print the results to verify
print("--- Interpolation Results ---")
for f in interpolator.frames:
    if f.ball:
        # Format the coordinates to 1 decimal place for easy reading
        coords = f"({f.ball.x1:.1f}, {f.ball.y1:.1f}) to ({f.ball.x2:.1f}, {f.ball.y2:.1f})"
        print(f"Frame {f.frame_number} | Conf: {f.ball.confidence:.2f} | Coords: {coords}")
    else:
        print(f"Frame {f.frame_number} | Ball: None")