"""Re-ID feature extraction module using OSNet for player appearance signatures."""

from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from torchreid.reid.utils import FeatureExtractor


class SignatureExtractor:
    """Wraps a pretrained OSNet Re-ID model to generate normalized appearance vectors.

    Attributes:
        device (str): The compute device ('cuda' or 'cpu').
        extractor (FeatureExtractor): The underlying torchreid feature extractor.
    """

    def __init__(
        self,
        model_name: str = "osnet_ain_x1_0",
        model_path: str = ""
    ):
        """Initializes the SignatureExtractor and loads Re-ID weights onto GPU if available.

        Args:
            model_name (str, optional): The OSNet architecture variant to load.
                Defaults to "osnet_ain_x1_0".
            model_path (str, optional): Path to the pretrained Re-ID .pth weights file.
                Defaults to "".
        """
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            device=self.device
        )

    def extract_batch(self, crops_rgb: List[np.ndarray]) -> List[List[float]]:
        """Computes L2-normalized appearance embeddings for a batch of player crops.

        Args:
            crops_rgb (List[np.ndarray]): List of player image crops in RGB format.

        Returns:
            List[List[float]]: A list of 512-dimensional L2-normalized feature
                vectors, one per input crop.
        """
        if not crops_rgb:
            return []

        with torch.no_grad():
            features = self.extractor(crops_rgb)
            # L2 normalize so cosine similarity reduces to a simple dot product
            normalized_features = F.normalize(features, p=2, dim=1)

        return normalized_features.cpu().numpy().tolist()