"""
Module for visualizing detection results on images.
"""

import cv2
import numpy as np
from contract.frame.Frame import Frame

def draw_detections_on_image(image: np.ndarray, frame: Frame, output_path: str) -> None:
    """
    Draws bounding boxes and labels for all detected objects in a Frame 
    and saves the resulting image to disk.

    Args:
        image (np.ndarray): The raw image array (BGR format).
        frame (Frame): The Frame object containing detected objects.
        output_path (str): The file path where the visualized image will be saved.
    """
    # Draw Players (Blue)
    for player in frame.players:
        color = (255, 0, 0)  # Blue in BGR
        p_x1, p_y1 = int(player.x1), int(player.y1)
        p_x2, p_y2 = int(player.x2), int(player.y2)
        cv2.rectangle(image, (p_x1, p_y1), (p_x2, p_y2), color, 2)
        cv2.putText(image, f"Player {player.confidence:.2f}", (p_x1, p_y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw Rims (Red)
    for rim in frame.rims:
        color = (0, 0, 255)  # Red in BGR
        r_x1, r_y1 = int(rim.x1), int(rim.y1)
        r_x2, r_y2 = int(rim.x2), int(rim.y2)
        cv2.rectangle(image, (r_x1, r_y1), (r_x2, r_y2), color, 2)
        cv2.putText(image, f"Rim {rim.confidence:.2f}", (r_x1, r_y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw Ball (Orange)
    if frame.ball is not None:
        color = (0, 165, 255)  # Orange in BGR
        b_x1, b_y1 = int(frame.ball.x1), int(frame.ball.y1)
        b_x2, b_y2 = int(frame.ball.x2), int(frame.ball.y2)
        cv2.rectangle(image, (b_x1, b_y1), (b_x2, b_y2), color, 2)
        cv2.putText(image, f"Ball {frame.ball.confidence:.2f}", (b_x1, b_y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(output_path, image)
    print(f"[Debug] Visualized detection image saved to: {output_path}")