"""
Client script to test the object-detection microservice.

This script sends an image along with frame metadata to the FastAPI endpoint
and prints the resulting Frame JSON.
"""

import os
import requests
import json

def test_detect_endpoint(image_path: str, url: str = "http://localhost:8000/detect") -> None:
    """
    Sends an image to the object detection API and prints the response.

    Args:
        image_path (str): The local path to the image file to be tested.
        url (str): The URL of the API endpoint.
    """
    if not os.path.exists(image_path):
        print(f"[Error] Image file not found at: {image_path}")
        return

    # Define the multipart/form-data payload
    data = {
        "frame_number": 42,
        "timestamp_ms": 1500.5
    }

    print(f"Sending request to {url}...")
    
    try:
        # Open the image file in binary mode
        with open(image_path, "rb") as image_file:
            files = {
                "image": (os.path.basename(image_path), image_file, "image/jpeg")
            }
            
            # Send the POST request
            response = requests.post(url, data=data, files=files)

        # Check if the request was successful
        if response.status_code == 200:
            print("[Success] Detection completed!")
            
            # Parse and print the JSON response nicely
            frame_data = response.json()
            print(json.dumps(frame_data, indent=4))
        else:
            print(f"[Error] Server returned status code: {response.status_code}")
            print("Response:", response.text)

    except requests.exceptions.ConnectionError:
        print("[Error] Could not connect to the server. Is the Docker container running?")
    except Exception as e:
        print(f"[Error] An unexpected error occurred: {e}")

if __name__ == "__main__":
    # TODO: Change this to the path of a real image on your computer
    TEST_IMAGE_PATH = "img.png" 
    
    test_detect_endpoint(TEST_IMAGE_PATH)