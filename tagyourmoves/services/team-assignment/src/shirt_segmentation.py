"""
Module for semantic segmentation of clothing using SegFormer.
"""

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


class ShirtSegmenter:
    """Handles the isolation of shirt pixels from player images.

    Loads the pre-trained SegFormer model once upon initialization and provides 
    methods to segment clothing from individual player bounding box crops.
    """

    def __init__(self, model_name: str = "mattmdjaga/segformer_b2_clothes"):
        """Initializes the processor and model.

        Args:
            model_name (str): The Hugging Face model repository name.
        """
        print(f"[ShirtSegmenter] Loading model '{model_name}'...")
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        
        # Move to GPU if available for faster inference
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"[ShirtSegmenter] Model loaded successfully on {self.device}.")

    def extract_shirt(self, player_crop: np.ndarray) -> np.ndarray:
        """Isolates the shirt pixels from a player's cropped image.

        Runs the SegFormer model on the provided image, extracts the 
        'Upper-clothes' class mask, and applies it to the original image.

        Args:
            player_crop (np.ndarray): The cropped BGR image of the player.

        Returns:
            np.ndarray: An image of the same size where all non-shirt 
                pixels are colored black ([0, 0, 0]). Returns an empty array 
                if the input crop is invalid.
        """
        if player_crop.size == 0:
            return np.array([])

        # Convert BGR (OpenCV) to RGB (PIL format requirement)
        rgb_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_crop)

        # Prepare inputs and run inference
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits.cpu()
        
        # Resize logits to match original crop size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=pil_image.size[::-1], # (height, width)
            mode="bilinear",
            align_corners=False,
        )
        
        # Get predictions
        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
        
        # Create mask for 'Upper-clothes' (Class ID 4)
        shirt_mask = (pred_seg == 4).astype(np.uint8)
        
        # Apply mask to the original BGR crop
        shirt_only_img = np.zeros_like(player_crop)
        shirt_only_img[shirt_mask == 1] = player_crop[shirt_mask == 1]
        
        return shirt_only_img