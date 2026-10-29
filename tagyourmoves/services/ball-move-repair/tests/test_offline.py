import cv2
import random
from typing import List

# Import your classes (adjust the import path as needed)
from src.repairers.Ball import Ball
from src.repairers.Frame import Frame
from src.repairers.Offline import estimate_missing_balls_offline
import pandas as pd

def estimate_missing_balls_offline(frames: List[Frame], max_gap_frames: int = 15) -> List[Frame]:
    if not frames: return frames
    data = [{'frame_number': f.frame_number, 
             'x1': f.ball.x1 if f.ball else None, 'y1': f.ball.y1 if f.ball else None, 
             'x2': f.ball.x2 if f.ball else None, 'y2': f.ball.y2 if f.ball else None} 
            for f in frames]
    df = pd.DataFrame(data).set_index('frame_number')
    if df['x1'].isna().all(): return frames
    
    interpolated_df = df.interpolate(method='linear', limit=max_gap_frames, limit_direction='both')
    
    for f in frames:
        if f.ball is None:
            row = interpolated_df.loc[f.frame_number]
            if not pd.isna(row['x1']):
                # Set confidence to 0.0 to flag it as an estimation
                f.ball = Ball(0.0, float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2']))
    return frames
# -----------------------------------------------------------------

def run_mock_detector(frame_number: int, width: int, height: int):
    """
    SIMULATION: Simulates a ball moving diagonally across the screen.
    We intentionally drop detections between frames 30 and 60, 
    and randomly drop 20% of other frames to test interpolation.
    """
    x1 = int(100 + (frame_number * 5) % (width - 50))
    y1 = int(100 + (frame_number * 3) % (height - 50))
    x2, y2 = x1 + 30, y1 + 30

    # Simulate long model failure (e.g., ball goes behind a player)
    if 30 <= frame_number <= 60:
        return None
    
    # Simulate random flickering model failures
    if random.random() < 0.20:
        return None

    return Ball(confidence=0.88, x1=x1, y1=y1, x2=x2, y2=y2)


def test_offline_estimation(input_video_path: str, output_video_path: str):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_metadata: List[Frame] = []
    
    print(f"Pass 1: Scanning {total_frames} frames & extracting detections...")
    frame_number = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
            
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        # ---------------------------------------------------------
        # REPLACE THIS WITH YOUR ACTUAL YOLO/MODEL INFERENCE
        # ball = my_yolo_model.detect_ball(frame)
        ball = run_mock_detector(frame_number, width, height)
        # ---------------------------------------------------------
        
        frames_metadata.append(Frame(
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
            ball=ball
        ))
        frame_number += 1
        
    cap.release()

    print("Offline Processing: Interpolating missing balls...")
    # Using 45 as max gap so it bridges our simulated 30-frame gap
    frames_metadata = estimate_missing_balls_offline(frames_metadata, max_gap_frames=45)

    print("Pass 2: Drawing bounding boxes and rendering output video...")
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, image_frame = cap.read()
        if not ret:
            break
            
        frame_obj = frames_metadata[frame_idx]
        
        if frame_obj.ball is not None:
            b = frame_obj.ball
            
            # Cast coordinates to int for OpenCV drawing
            x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
            
            if b.confidence > 0.0:
                # Actual Detection -> GREEN
                color = (0, 255, 0) # OpenCV uses BGR
                label = f"Detected: {b.confidence:.2f}"
            else:
                # Estimated/Interpolated -> RED
                color = (0, 0, 255) # BGR
                label = "Estimated"
                
            # Draw Bounding Box
            cv2.rectangle(image_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw Label Background
            cv2.rectangle(image_frame, (x1, y1 - 20), (x1 + 110, y1), color, -1)
            # Draw Label Text
            cv2.putText(image_frame, label, (x1 + 2, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        out.write(image_frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Rendered {frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"Success! Output saved to: {output_video_path}")


if __name__ == "__main__":
    # Create a quick dummy video if you don't have one handy, 
    # or replace 'input.mp4' with a real basketball video path.
    test_offline_estimation("input.mp4", "output_interpolated.mp4")