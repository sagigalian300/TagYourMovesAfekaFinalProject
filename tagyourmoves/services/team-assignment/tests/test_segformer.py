"""
Proof of Concept for extracting ONLY the shirt pixels using SegFormer.
Adapted for headless Docker environment.
"""

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


def extract_shirt_only(img_path: str, output_path: str) -> None:
    """Uses a Human Parsing model to segment and extract only the shirt.

    Loads the SegFormer model specifically fine-tuned for clothing, runs 
    inference to classify every pixel, and isolates category '4' (Upper-clothes).
    Saves an image where everything except the shirt area is completely black.

    Args:
        img_path (str): The relative path to the input image.
        output_path (str): The relative path where the isolated shirt image will be saved.
    """
    print("Loading SegFormer model (might take a moment to download on the first run)...")
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

    # 1. Load image using PIL (Transformers requirement)
    try:
        image = Image.open(img_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Could not find image at {img_path}")
        return
    
    # 2. Prepare image for the model and run inference
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 3. Process the output mask
    logits = outputs.logits.cpu()
    
    # Resize the output mask back to the original image dimensions
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1], # (height, width)
        mode="bilinear",
        align_corners=False,
    )
    
    # Get the predicted class for each pixel (argmax)
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    
    # 4. Isolate the shirt (Class ID 4 is 'Upper-clothes')
    # Create a binary mask where the shirt is 1 and everything else is 0
    shirt_mask = (pred_seg == 4).astype(np.uint8)
    
    # 5. Apply the mask to the original image using OpenCV
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Create an empty black image of the same size
    shirt_only_img = np.zeros_like(cv_image)
    
    # Copy only the pixels where the mask is 1
    shirt_only_img[shirt_mask == 1] = cv_image[shirt_mask == 1]
    
    # 6. Save the result to disk
    cv2.imwrite(output_path, shirt_only_img)
    print(f"Success! Shirt isolated and saved to: {output_path}")


if __name__ == "__main__":
    # Ensure you have a 'mock-img-team-assignment2.png' file inside the 'mock' folder before running
    extract_shirt_only(
        img_path="mock/mock-img-team-assignment4.png", 
        output_path="tests/shirt_only_result.jpg"
    )