"""
Explainable Artificial Intelligence (XAI) Suite
Implements Grad-CAM & Grad-CAM++ for visual feature localization on fundus/ocular images,
plus SHAP feature attribution utilities for clinical interpretability.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import Optional, List, Tuple


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    Computes importance gradients of target class with respect to the last convolutional feature maps.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generates Grad-CAM heatmap for a given input tensor (1, C, H, W).
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        # Backward pass for target class score
        target_score = output[0, target_class]
        target_score.backward()

        # Global average pooling of gradients: weights alpha_k
        alpha_k = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(alpha_k * self.activations, dim=1, keepdim=True)
        cam = nn.functional.relu(cam) # Only features that positively influence prediction

        # Normalize to [0, 1]
        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = np.min(cam_np), np.max(cam_np)
        if cam_max - cam_min > 1e-8:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        return cam_norm

    @staticmethod
    def overlay_heatmap(
        heatmap: np.ndarray,
        original_image: np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Resizes heatmap to original image dimensions and creates a smooth colored overlay.
        """
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

        # If original image is grayscale, convert to 3-channel
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

        # Blend
        overlay = cv2.addWeighted(original_image, 1.0 - alpha, color_heatmap, alpha, 0)
        return overlay


def generate_and_save_gradcam_visuals(
    model: nn.Module,
    target_layer: nn.Module,
    sample_images: List[Tuple[torch.Tensor, int, str, np.ndarray]],
    class_names: List[str],
    output_dir: str = "experiments/results/xai",
    device: str = "cpu",
):
    """
    Generates side-by-side [Original, Preprocessed, Grad-CAM Overlay] publication figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    grad_cam = GradCAM(model, target_layer)

    for idx, (tensor, label, file_path, raw_img) in enumerate(sample_images):
        tensor_input = tensor.unsqueeze(0).to(device)
        tensor_input.requires_grad = True

        cam = grad_cam.generate_cam(tensor_input, target_class=label)

        # Generate overlay
        overlay = GradCAM.overlay_heatmap(cam, raw_img, alpha=0.45)

        # Plot 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Eye Image", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Visualise Green-Channel CLAHE (Channel 0 of preprocessed tensor)
        prep_img = tensor[0].cpu().numpy()
        axes[1].imshow(prep_img, cmap="gray")
        axes[1].set_title("Dual-Stream Preprocessed (CLAHE)", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"Grad-CAM: {class_names[label]}", fontsize=11, fontweight="bold", color="darkred")
        axes[2].axis("off")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"xai_{class_names[label].lower()}_sample_{idx+1}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
