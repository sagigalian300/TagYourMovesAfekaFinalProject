import sys
from pathlib import Path

# Add the 'ball-move-repair' directory to Python's search path
ball_move_repair_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(ball_move_repair_dir))

import cv2
import pickle
from src.repairers.BallInterpolator import BallInterpolator

def draw_interpolated_ball(image, ball):
    """Draws green for real YOLO detections and red for interpolated detections."""
    if ball is None:
        return image
    
    x1, y1, x2, y2 = int(ball.x1), int(ball.y1), int(ball.x2), int(ball.y2)
    
    if ball.confidence == 0.0:
        color = (0, 0, 255)  # Red for Interpolated
        label = "Ball [Interpolated]"
    else:
        color = (0, 255, 0)  # Green for Real
        label = f"Ball [{ball.confidence:.2f}]"
        
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

def main():
    current_dir = Path(__file__).resolve().parent
    pickle_filename = str(current_dir / "detected_frames.pkl")
    video_path = str(current_dir / "test_vod.mp4")

    # 1. Load the raw frames data saved from Step 1
    print(f"Loading raw frame data from '{pickle_filename}'...")
    try:
        with open(pickle_filename, "rb") as f:
            frames_list = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find '{pickle_filename}'. Run step1_generate_raw_yolo_video.py first.")
        return

    # 2. Run the BallInterpolator
    print("Running Ball Interpolator...")
    interpolator = BallInterpolator(frames_list)
    interpolator.repair_missing_balls()

    # 3. Read original video and save the interpolated output
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Save the video in the same directory as the script
    video_output_path = str(current_dir / 'output_2_interpolated.mp4')
    out_interp = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    print("\n--- STEP 2: Drawing and Saving Interpolated Video ---")
    for i in range(len(interpolator.frames)):
        ret, frame = cap.read()
        if not ret:
            break
            
        interp_ball = interpolator.frames[i].ball
        draw_interpolated_ball(frame, interp_ball)
        out_interp.write(frame)

        if (i + 1) % 50 == 0:
            print(f"Saved {i + 1}/{total_frames} frames...")

    cap.release()
    out_interp.release()
    print("\nStep 2 Complete! Interpolated video saved to 'output_2_interpolated.mp4'")


main()