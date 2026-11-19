"""
Module for generating visual debug overlays for team assignment and referee detection.
"""

import os
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Any


def _draw_legend(canvas: np.ndarray) -> None:
    """Draws a semi-transparent color legend on the top-left corner of the frame.

    Args:
        canvas (np.ndarray): The BGR image to draw the legend onto in-place.
    """
    items = [
        ((0, 255, 255), "Mask: Shirt Kept (Class 4)"),
        ((0, 140, 255), "Mask: Filtered Skin/Edge"),
        ((255, 220, 100), "Mask: Torso Fallback"),
        ((255, 0, 255), "Mask: Pants (Class 6)"),
        ((255, 255, 0), "Mask: Skin (60%-88%)"),
        ((0, 255, 0), "Box: Referee (Team 0)"),
        ((255, 0, 0), "Box: Team 1"),
        ((0, 0, 255), "Box: Team 2"),
        ((200, 200, 200), "Box: Unassigned (N/A)"),
        ((255, 255, 255), "Inner Square: Dominant Color"),
    ]

    x0, y0 = 10, 10
    line_h = 22
    box_w = 265
    box_h = len(items) * line_h + 16

    # Semi-transparent dark background for readability
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, dst=canvas)
    cv2.rectangle(canvas, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), 1)

    for idx, (color, text) in enumerate(items):
        cy = y0 + 12 + idx * line_h
        cv2.rectangle(canvas, (x0 + 10, cy), (x0 + 24, cy + 14), color, -1)
        cv2.rectangle(canvas, (x0 + 10, cy), (x0 + 24, cy + 14), (255, 255, 255), 1)
        cv2.putText(
            canvas,
            text,
            (x0 + 32, cy + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


def save_debug_frame(
    image: np.ndarray,
    debug_records: List[Dict[str, Any]],
    output_dir: str = "debug_outputs",
    alpha: float = 0.45
) -> None:
    """Renders segmentation masks, calf-window lines, dominant color swatches, and a legend.

    Args:
        image (np.ndarray): The original full BGR frame.
        debug_records (List[Dict[str, Any]]): List of dicts containing 'player',
            'box' (x1, y1, x2, y2), 'pred_seg', 'skin_ratio', 'dark_ratio',
            'skin_mask', and optionally 'dominant_color_bgr' and 'isolated_shirt'.
        output_dir (str): Directory path where debug images will be saved.
        alpha (float): Transparency factor for the segmentation mask overlay.
    """
    os.makedirs(output_dir, exist_ok=True)
    canvas = image.copy()

    for record in debug_records:
        player = record["player"]
        x1, y1, x2, y2 = record["box"]
        pred_seg = record["pred_seg"]
        skin_ratio = record["skin_ratio"]
        dark_ratio = record["dark_ratio"]

        roi = canvas[y1:y2, x1:x2]
        overlay = roi.copy()

        # Class 4 = Upper-clothes, visualized with skin/edge filtering and fallback
        raw_shirt_mask = (pred_seg == 4)
        isolated_shirt = record.get("isolated_shirt")

        if isolated_shirt is not None and isolated_shirt.size > 0:
            cleaned_shirt_mask = np.any(isolated_shirt > 0, axis=2)

            kept_mask = raw_shirt_mask & cleaned_shirt_mask
            filtered_out_mask = raw_shirt_mask & (~cleaned_shirt_mask)
            fallback_mask = (~raw_shirt_mask) & cleaned_shirt_mask

            overlay[kept_mask] = (0, 255, 255)          # Yellow - Kept shirt pixels
            overlay[filtered_out_mask] = (0, 140, 255)  # Orange - Filtered skin/edge
            overlay[fallback_mask] = (255, 220, 100)    # Light Blue - Torso fallback
        else:
            overlay[raw_shirt_mask] = (0, 255, 255)

        # Class 6 = Pants (Magenta)
        overlay[pred_seg == 6] = (255, 0, 255)

        # Paint detected skin pixels in the 60%-88% window in bright Cyan (BGR)
        box_h = y2 - y1
        split_top = int(box_h * 0.60)
        split_bottom = int(box_h * 0.88)
        skin_mask = record.get("skin_mask")
        if skin_mask is not None and skin_mask.size > 0:
            lower_overlay = overlay[split_top:split_bottom, :]
            lower_overlay[skin_mask > 0] = (255, 255, 0)

        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)

        # Draw the 60% and 88% horizontal lines marking the evaluated window
        cv2.line(canvas, (x1, y1 + split_top), (x2, y1 + split_top), (255, 255, 255), 1)
        cv2.line(canvas, (x1, y1 + split_bottom), (x2, y1 + split_bottom), (255, 255, 255), 1)

        if player.team == 0:
            color = (0, 255, 0)      # Green - Referee
            role_text = "REF"
        elif player.team == 1:
            color = (255, 0, 0)      # Blue - Team 1
            role_text = "T1"
        elif player.team == 2:
            color = (0, 0, 255)      # Red - Team 2
            role_text = "T2"
        else:
            color = (200, 200, 200)  # Gray - Unassigned
            role_text = "N/A"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Draw dominant color swatch in top-left corner of the player box
        dom_bgr = record.get("dominant_color_bgr")
        if dom_bgr is not None:
            swatch_size = 16
            sx2 = min(canvas.shape[1] - 1, x1 + swatch_size)
            sy2 = min(canvas.shape[0] - 1, y1 + swatch_size)
            cv2.rectangle(canvas, (x1 + 2, y1 + 2), (sx2, sy2), dom_bgr, -1)
            cv2.rectangle(canvas, (x1 + 2, y1 + 2), (sx2, sy2), (255, 255, 255), 1)

        label = f"{role_text} | S:{skin_ratio * 100:.0f}% D:{dark_ratio * 100:.0f}%"
        cv2.putText(
            canvas,
            label,
            (x1, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    _draw_legend(canvas)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output_path = os.path.join(output_dir, f"debug_{timestamp}.png")
    cv2.imwrite(output_path, canvas)
    print(f"[Debug] Saved visual debug frame to: {output_path}")