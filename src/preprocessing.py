"""
Dual-Stream Clinical Preprocessing Module
Synthesizes Green-Channel CLAHE (Papers 1 & 4) and Canny Boundary Gradient Extraction (Paper 5)
into a rich 3-channel composite representation for robust ocular and cataract disease detection.
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Union


class ClinicalEyePreprocessor:
    """
    Advanced preprocessor for color fundus and ocular imagery.
    Transforms raw RGB images into a 3-channel tensor:
    - Channel 0: Green-Channel with CLAHE (high-contrast vascular & lesion details)
    - Channel 1: Canny Edge Gradient Map (lens opacities & optic disc margins)
    - Channel 2: Contrast-enhanced Luminance / Grayscale (overall anatomical baseline)
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        canny_low_threshold: int = 50,
        canny_high_threshold: int = 150,
        gaussian_blur_ksize: Tuple[int, int] = (3, 3),
        gaussian_blur_sigma: float = 0.5,
    ):
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size
        )
        self.canny_low = canny_low_threshold
        self.canny_high = canny_high_threshold
        self.gaussian_blur_ksize = gaussian_blur_ksize
        self.gaussian_blur_sigma = gaussian_blur_sigma

    def remove_black_margins(self, image: np.ndarray, threshold: int = 10) -> np.ndarray:
        """
        Crops uninformative black margins commonly found in fundus camera captures.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        mask = gray > threshold
        if not np.any(mask):
            return image
        
        # Find non-zero bounding box
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return image[y0:y1, x0:x1]

    def extract_green_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Extracts the green channel (contains the sharpest retinal contrast)
        and applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
        """
        if len(image.shape) == 3:
            # Assuming BGR from cv2
            green = image[:, :, 1]
        else:
            green = image

        # Apply slight Gaussian smoothing to reduce sensor noise
        blurred = cv2.GaussianBlur(
            green, self.gaussian_blur_ksize, self.gaussian_blur_sigma
        )
        # Apply CLAHE
        enhanced = self.clahe.apply(blurred)
        return enhanced

    def extract_canny_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Median & Gaussian filtering followed by Canny edge detection
        to highlight structural boundaries, cataract clouding margins, and optic cup edges.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Multi-stage blur as validated in Diagnostics 2025 (Paper 5)
        median_blurred = cv2.medianBlur(gray, 3)
        gaussian_blurred = cv2.GaussianBlur(
            median_blurred, self.gaussian_blur_ksize, self.gaussian_blur_sigma
        )
        edges = cv2.Canny(gaussian_blurred, self.canny_low, self.canny_high)
        return edges

    def extract_enhanced_gray(self, image: np.ndarray) -> np.ndarray:
        """
        Converts image to LAB color space and applies CLAHE to the L (luminance) channel.
        """
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_enhanced = self.clahe.apply(l_channel)
            return l_enhanced
        else:
            return self.clahe.apply(image)

    def process_image(
        self, image: np.ndarray, return_tensor: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Full preprocessing pipeline:
        1. Circular / Bounding margin crop
        2. Resize to target dimension (224x224)
        3. Multi-channel feature synthesis: [CLAHE-Green, Canny-Edges, Enhanced-Luminance]
        4. Normalize pixel intensity to [0.0, 1.0]
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is invalid or empty.")

        # Step 1: Crop dark background borders
        cropped = self.remove_black_margins(image)

        # Step 2: Resize
        resized = cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_AREA)

        # Step 3: Extract the 3 complementary channels
        ch0_green_clahe = self.extract_green_clahe(resized)
        ch1_canny_edge = self.extract_canny_edges(resized)
        ch2_enhanced_l = self.extract_enhanced_gray(resized)

        # Stack into a 3-channel composite image (Height, Width, 3)
        composite = np.stack([ch0_green_clahe, ch1_canny_edge, ch2_enhanced_l], axis=-1)

        # Normalize to float32 [0, 1]
        composite_norm = composite.astype(np.float32) / 255.0

        if return_tensor:
            # Transpose from (H, W, C) to PyTorch (C, H, W)
            tensor = torch.from_numpy(composite_norm).permute(2, 0, 1).float()
            return tensor

        return composite_norm

    def preprocess_from_file(
        self, file_path: str, return_tensor: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Loads image from disk and processes it.
        """
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Unable to read image from path: {file_path}")
        return self.process_image(image, return_tensor=return_tensor)
