"""
Integration test script for the TagYourMoves microservices pipeline.

Flow:
1. Sends an image to the Object Detection MS (Port 8001) to get bounding boxes.
2. Takes the resulting Frame JSON and the same image, and sends them to 
   the Team Assignment MS (Port 8000) to cluster players into teams.
"""

import os
import json
import requests

def test_full_pipeline(image_path: str) -> None:
    """
    Executes the full pipeline by calling both microservices sequentially.

    Args:
        image_path (str): The local path to the test image file.
    """
    if not os.path.exists(image_path):
        print(f"[Error] Image file not found at: {image_path}")
        return

    url_detection = "http://localhost:8001/detect"
    url_team_assignment = "http://localhost:8000/assign-teams"

    # --- STEP 1: Object Detection ---
    print(f"\n[Step 1] Sending image to Object Detection MS ({url_detection})...")
    detection_data = {
        "frame_number": 1,
        "timestamp_ms": 1000.0
    }
    
    try:
        with open(image_path, "rb") as img_file:
            files_det = {"image": (os.path.basename(image_path), img_file, "image/jpeg")}
            res_det = requests.post(url_detection, data=detection_data, files=files_det)

        if res_det.status_code != 200:
            print(f"[Error] Detection failed: {res_det.text}")
            return
            
        frame_json = res_det.json()
        print(f"[Success] Detected {len(frame_json.get('players', []))} players, "
              f"{len(frame_json.get('rims', []))} rims, "
              f"and {'1' if frame_json.get('ball') else '0'} ball.")
              
    except Exception as e:
        print(f"[Error] Object Detection request failed: {e}")
        return

    # --- STEP 2: Team Assignment ---
    print(f"\n[Step 2] Sending Frame data and image to Team Assignment MS ({url_team_assignment})...")
    
    # The Team Assignment endpoint expects 'frame_data' as a JSON string in Form data
    assignment_data = {
        "frame_data": json.dumps(frame_json)
    }

    try:
        # We open the image again since the previous file pointer was closed
        with open(image_path, "rb") as img_file:
            files_team = {"image": (os.path.basename(image_path), img_file, "image/jpeg")}
            res_team = requests.post(url_team_assignment, data=assignment_data, files=files_team)

        if res_team.status_code != 200:
            print(f"[Error] Team Assignment failed: {res_team.text}")
            return
            
        final_frame = res_team.json()
        print("[Success] Team Assignment completed!")
        print("\n--- Final Frame Output ---")
        print(json.dumps(final_frame, indent=4))
        
        print("\n[Done] Check the 'mock/debug_outputs' folder in your Team Assignment MS to see the visual results.")

    except Exception as e:
        print(f"[Error] Team Assignment request failed: {e}")

if __name__ == "__main__":
    # TODO: Change this to the path of a real image on your computer
    TEST_IMAGE = "img1.png" 
    
    test_full_pipeline(TEST_IMAGE)