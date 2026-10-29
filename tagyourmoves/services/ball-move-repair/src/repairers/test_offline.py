import os
import cv2
from typing import List
from ultralytics import YOLO

from Ball import Ball
from Player import Player 
from Frame import Frame
from Offline import estimate_missing_balls_offline


# Class ID 32 is 'sports ball' in standard COCO YOLO models. 
# Change BALL_CLASS_ID to 0 if your weights (yolov8s.pt) are custom-trained specifically for basketballs.
BALL_CLASS_ID = 32
PERSON_CLASS_ID = 0

def detect_objects_yolo(model: YOLO, image_frame, confidence_threshold: float = 0.25):
    """
    Runs YOLO inference on a single video frame to detect the ball AND players.
    """
    results = model(image_frame, verbose=False)[0]
    
    best_ball = None
    highest_conf_ball = 0.0
    players = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if conf >= confidence_threshold:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            if cls_id == BALL_CLASS_ID:
                if conf > highest_conf_ball:
                    highest_conf_ball = conf
                    best_ball = Ball(confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2)
            
            elif cls_id == PERSON_CLASS_ID:
                players.append(Player(confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2))

    return best_ball, players

def test_offline_estimation(input_video_path: str, output_video_path: str, model_path: str):
    # Initialize YOLO Model
    model = YOLO(model_path)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_metadata: List[Frame] = []
    
    print(f"Pass 1: Scanning {total_frames} frames with YOLO model...")
    frame_number = 0
    while True:
        ret, image_frame = cap.read()
        if not ret:
            break
            
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        # Detect the balland players in the actual video frame
        ball, players = detect_objects_yolo(model, image_frame)
        
        frames_metadata.append(Frame(
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
            ball=ball,
            players=players
        ))
        frame_number += 1
        
        if frame_number % 50 == 0:
            print(f"Scanned {frame_number}/{total_frames} frames...")

    cap.release()

    print("Offline Processing: Interpolating missing ball frames...")
    frames_metadata = estimate_missing_balls_offline(frames_metadata, max_gap_frames=45)

    print("Pass 2: Rendering output video with bounding boxes...")
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
            x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
            
            if b.confidence > 0.0:
                color = (0, 255, 0)  # Green for actual model detection
                label = f"Detected: {b.confidence:.2f}"
            else:
                color = (0, 0, 255)  # Red for algorithm estimation
                label = "Estimated"
                
            cv2.rectangle(image_frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(image_frame, (x1, max(0, y1 - 20)), (x1 + 110, y1), color, -1)
            cv2.putText(image_frame, label, (x1 + 2, max(12, y1 - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        out.write(image_frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Success! Output saved to: {output_video_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_path = os.path.join(script_dir, "test_vod2.mp4")
    output_path = os.path.join(script_dir, "output_interpolated.mp4")
    weights_path = os.path.join(script_dir, "yolov8s.pt")  # or "yolo26s.pt"
    
    test_offline_estimation(input_path, output_path, weights_path)