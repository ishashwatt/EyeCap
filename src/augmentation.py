"""
Clinical Image Augmentation Engine
Implements multi-angle rotation, geometric transforms, and tensor normalization
specifically designed for fundus and ocular imagery (validated in PLOS ONE 2024).
"""

import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from typing import Tuple, Optional


class ClinicalAugmentor:
    """
    Applies medically valid augmentations that preserve diagnostic ocular geometry:
    - Discrete multi-angle rotations (15, 30, 45, 90, 180, 270 degrees)
    - Horizontal and vertical flips (simulating left/right eye symmetry)
    - Minor scaling/zooming [0.9, 1.1]
    - Subtle brightness and contrast variations
    """

    def __init__(
        self,
        rotation_angles: Tuple[int, ...] = (15, 30, 45, 90, 180, 270),
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.3,
        zoom_range: Tuple[float, float] = (0.9, 1.1),
        brightness_jitter: float = 0.1,
        contrast_jitter: float = 0.1,
    ):
        self.rotation_angles = rotation_angles
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.zoom_range = zoom_range
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter

    def augment_tensor(self, tensor: torch.Tensor, is_training: bool = True) -> torch.Tensor:
        """
        Applies data augmentation to a PyTorch tensor of shape (C, H, W).
        """
        if not is_training:
            return tensor

        # 1. Random horizontal flip
        if random.random() < self.hflip_prob:
            tensor = TF.hflip(tensor)

        # 2. Random vertical flip
        if random.random() < self.vflip_prob:
            tensor = TF.vflip(tensor)

        # 3. Discrete clinical angle rotation
        if random.random() < 0.7:
            angle = random.choice(self.rotation_angles)
            tensor = TF.rotate(tensor, angle=angle)

        # 4. Random zoom / scale
        if random.random() < 0.5:
            scale = random.uniform(self.zoom_range[0], self.zoom_range[1])
            _, h, w = tensor.shape
            new_h, new_w = int(h * scale), int(w * scale)
            resized = TF.resize(tensor, [new_h, new_w], antialias=True)
            if scale > 1.0:
                # Center crop back to original size
                tensor = TF.center_crop(resized, [h, w])
            else:
                # Pad back to original size
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                tensor = TF.pad(
                    resized, [pad_w, pad_h, w - new_w - pad_w, h - new_h - pad_h]
                )

        # 5. Subtle brightness/contrast jitter
        if random.random() < 0.4:
            b_factor = 1.0 + random.uniform(-self.brightness_jitter, self.brightness_jitter)
            c_factor = 1.0 + random.uniform(-self.contrast_jitter, self.contrast_jitter)
            tensor = TF.adjust_brightness(tensor, b_factor)
            tensor = TF.adjust_contrast(tensor, c_factor)

        # Clamp values to [0, 1]
        tensor = torch.clamp(tensor, 0.0, 1.0)
        return tensor
