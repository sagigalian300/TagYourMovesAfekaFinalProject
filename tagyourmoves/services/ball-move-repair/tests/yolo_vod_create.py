import sys
from pathlib import Path

# Add the 'ball-move-repair' directory to Python's search path
ball_move_repair_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(ball_move_repair_dir))

import cv2
import math
import pickle
from ultralytics import YOLO
from src.repairers.Ball import Ball
from src.repairers.Frame import Frame

def draw_raw_ball(image, ball):
    """Draws a green box around the raw YOLO ball detection."""
    if ball is None:
        return image
    
    x1, y1, x2, y2 = int(ball.x1), int(ball.y1), int(ball.x2), int(ball.y2)
    color = (0, 255, 0)  # Green
    label = f"Ball [{ball.confidence:.2f}]"
    
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

def main():
    current_dir = Path(__file__).resolve().parent
    model = YOLO(str(current_dir / "yolov8s.pt"))
    SPORTS_BALL_CLASS_ID = 32
    video_path = str(current_dir / "test_vod.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video Writer
# Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Save the video in the same directory as the script
    video_output_path = str(current_dir / 'output_1_raw_yolo.mp4')
    out_raw = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    frames_list = []
    frame_count = 0

    print("--- STEP 1: Running YOLO & Saving Raw Video ---")
    while cap.isOpened():
        ret, cv_frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        # Run YOLO inference
        results = model.track(cv_frame, persist=True, imgsz=1280, conf=0.10, verbose=False)
        
        detected_ball = None
        highest_conf = 0.0
        
        # --- LOGIC FILTER SETTINGS ---
        MAX_BOX_SIZE = 600      # Maximum width/height in pixels a ball should be 
        MAX_PIXEL_SPEED = 200  # Max pixels a ball can move in a single frame
        # -----------------------------

        for box in results[0].boxes:
            if int(box.cls[0]) == SPORTS_BALL_CLASS_ID:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Filter 1: Check Box Size
                box_width = x2 - x1
                box_height = y2 - y1
                if box_width > MAX_BOX_SIZE or box_height > MAX_BOX_SIZE:
                    continue  # Ignore this box, it's too big to be a ball
                
                # Filter 2: Check Distance (Speed Limit)
                # If we have tracked a ball in previous frames, calculate how far this new box is
                current_center_x = (x1 + x2) / 2
                current_center_y = (y1 + y2) / 2
                
                # Look backwards in frames_list to find the last known real ball
                last_known_ball = next((f.ball for f in reversed(frames_list) if f.ball is not None), None)
                
                if last_known_ball is not None:
                    last_center_x = (last_known_ball.x1 + last_known_ball.x2) / 2
                    last_center_y = (last_known_ball.y1 + last_known_ball.y2) / 2
                    
                    # Calculate distance using Pythagorean theorem
                    distance = math.hypot(current_center_x - last_center_x, current_center_y - last_center_y)
                    
                    if distance > MAX_PIXEL_SPEED:
                        continue  # Ignore this box, it teleported too far
                
                # If it passed both filters, check if it's the highest confidence ball so far
                if conf > highest_conf:
                    highest_conf = conf
                    detected_ball = Ball(confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2)
                
        # Store Frame object
        current_frame = Frame(frame_number=frame_count, timestamp_ms=timestamp_ms, ball=detected_ball)
        frames_list.append(current_frame)
        
        # Draw and write to raw YOLO video
        raw_draw_frame = cv_frame.copy()
        draw_raw_ball(raw_draw_frame, detected_ball)
        out_raw.write(raw_draw_frame)

        if frame_count % 50 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    cap.release()
    out_raw.release()

    # Save frames_list to disk so Step 2 can read it instantly
    pickle_filename = str(current_dir / "detected_frames.pkl")
    with open(pickle_filename, "wb") as f:
        pickle.dump(frames_list, f)

    print(f"\nStep 1 Complete! Raw video saved to 'output_1_raw_yolo.mp4'")
    print(f"Frame data saved to '{pickle_filename}'")


main()