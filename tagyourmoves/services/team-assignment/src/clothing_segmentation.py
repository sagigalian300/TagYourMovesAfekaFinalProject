"""
Module for semantic segmentation of clothing using SegFormer.
"""

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


class ClothingSegmenter:
    """Handles the semantic segmentation of players.

    Loads the pre-trained SegFormer model once upon initialization and provides 
    methods to retrieve raw segmentation masks for downstream extractors/detectors.
    """

    def __init__(self, model_name: str = "mattmdjaga/segformer_b2_clothes"):
        """Initializes the processor and model."""
        print(f"[ClothingSegmenter] Loading model '{model_name}'...")
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"[ClothingSegmenter] Model loaded successfully on {self.device}.")

    def get_segmentation_mask(self, player_crop: np.ndarray) -> np.ndarray:
        """Runs inference and returns the full segmentation mask.

        Args:
            player_crop (np.ndarray): The cropped BGR image of the person.

        Returns:
            np.ndarray: A 2D array where each pixel contains the class ID.
                Returns an empty array if the crop is invalid.
        """
        if player_crop.size == 0:
            return np.array([])

        rgb_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_crop)

        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits.cpu()
        
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=pil_image.size[::-1], 
            mode="bilinear",
            align_corners=False,
        )
        
        # Return the raw 2D mask with class IDs for each pixel
        return upsampled_logits.argmax(dim=1)[0].numpy()