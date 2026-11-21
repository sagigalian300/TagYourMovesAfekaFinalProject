"""Orchestrates player crop extraction and signature assignment for a single frame."""

import numpy as np

from contracts.frame.Frame import Frame
from src.crop_extractor import extract_player_crops
from src.signature_extractor import SignatureExtractor


def process_frame(
    image: np.ndarray,
    current_frame: Frame,
    extractor: SignatureExtractor
) -> Frame:
    """Populates the appearance signature for every valid player in the frame.

    Args:
        image (np.ndarray): The raw image array in BGR format.
        current_frame (Frame): The Frame object containing detected players.
        extractor (SignatureExtractor): The shared Re-ID model instance.

    Returns:
        Frame: The updated Frame object with player 'signature' attributes populated.
    """
    if not current_frame.players:
        return current_frame

    valid_players, crops_rgb = extract_player_crops(image, current_frame.players)
    if not crops_rgb:
        return current_frame

    signatures = extractor.extract_batch(crops_rgb)

    for player, signature in zip(valid_players, signatures):
        player.signature = signature

    return current_frame