"""
Unit Tests for Clinical Eye Preprocessor
"""

import numpy as np
import torch
import pytest
from src.preprocessing import ClinicalEyePreprocessor


def test_preprocessor_output_dimensions():
    preprocessor = ClinicalEyePreprocessor(target_size=(224, 224))
    
    # Create dummy RGB image (256x256x3)
    dummy_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    # Test numpy output
    processed_np = preprocessor.process_image(dummy_img, return_tensor=False)
    assert processed_np.shape == (224, 224, 3)
    assert processed_np.dtype == np.float32
    assert 0.0 <= np.min(processed_np) <= np.max(processed_np) <= 1.0

    # Test PyTorch tensor output
    processed_tensor = preprocessor.process_image(dummy_img, return_tensor=True)
    assert isinstance(processed_tensor, torch.Tensor)
    assert processed_tensor.shape == (3, 224, 224)
    assert processed_tensor.dtype == torch.float32


def test_green_channel_clahe_enhancement():
    preprocessor = ClinicalEyePreprocessor()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_img[:, :, 1] = 120 # Green channel set
    enhanced_green = preprocessor.extract_green_clahe(dummy_img)
    assert enhanced_green.shape == (100, 100)


def test_canny_edge_extraction():
    preprocessor = ClinicalEyePreprocessor()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add a bright square in the center
    dummy_img[30:70, 30:70] = 255
    edges = preprocessor.extract_canny_edges(dummy_img)
    assert edges.shape == (100, 100)
    assert np.any(edges > 0)
