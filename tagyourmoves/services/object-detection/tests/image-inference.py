import cv2
import os
from ultralytics import YOLO

MODEL_PATH = "src/detectors/yolo26s.pt" 
IMAGE_PATH = "tests/images-for-tests/img1.png"
OUTPUT_PATH = "tests/results_for_tests/img1_result.png"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

print(f"Loading YOLO model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

print(f"Running inference on {IMAGE_PATH}...")
results = model(IMAGE_PATH, conf=0.1)

for result in results:
    annotated_frame = result.plot()
    
    is_saved = cv2.imwrite(OUTPUT_PATH, annotated_frame)
    if is_saved:
        print(f"Success! Result actually saved to {OUTPUT_PATH}")
    else:
        print(f"Error: OpenCV failed to write the image to {OUTPUT_PATH}")