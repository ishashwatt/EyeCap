"""
Novel 14-Layer Lightweight Attention CNN: OcuNet-CBAM
Engineered specifically for low-complexity, high-accuracy multi-ocular & cataract disease diagnosis.
Parameter Footprint: < 1.5M parameters (mitigates the overfitting observed in 50+ layer models).
"""

import torch
import torch.nn as nn
from src.models.cbam import CBAMBlock
from typing import Tuple, Dict, Any


class ConvBnReluBlock(nn.Module):
    """
    Standard convolutional building block: Conv2D -> BatchNorm -> ReLU.
    """

    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(ConvBnReluBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class OcuNetCBAM(nn.Module):
    """
    14-Layer Optimized Attention Neural Architecture for Eye Disease & Cataract Screening.
    Combines 4 progressive convolutional stages with integrated CBAM attention blocks
    and a regularized multi-layer perceptron head.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        base_filters: int = 32,
        dropout_rate: float = 0.4,
        use_cbam: bool = True,
    ):
        super(OcuNetCBAM, self).__init__()
        self.use_cbam = use_cbam
        f = base_filters

        # Stage 1: Initial Spatial & Edge Feature Extraction (224x224 -> 112x112)
        self.stage1 = nn.Sequential(
            ConvBnReluBlock(in_channels, f, kernel_size=3, padding=1),
            ConvBnReluBlock(f, f, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stage 2: Low-Level Texture & Vessel Extraction (112x112 -> 56x56)
        self.stage2 = nn.Sequential(
            ConvBnReluBlock(f, f * 2, kernel_size=3, padding=1),
            ConvBnReluBlock(f * 2, f * 2, kernel_size=3, padding=1),
        )
        self.cbam2 = CBAMBlock(f * 2) if use_cbam else nn.Identity()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 3: Mid-Level Lesion & Cup Boundary Representation (56x56 -> 28x28)
        self.stage3 = nn.Sequential(
            ConvBnReluBlock(f * 2, f * 4, kernel_size=3, padding=1),
            ConvBnReluBlock(f * 4, f * 4, kernel_size=3, padding=1),
        )
        self.cbam3 = CBAMBlock(f * 4) if use_cbam else nn.Identity()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 4: High-Level Pathological & Opacity Abstraction (28x28 -> 14x14)
        self.stage4 = nn.Sequential(
            ConvBnReluBlock(f * 4, f * 8, kernel_size=3, padding=1),
            ConvBnReluBlock(f * 8, f * 8, kernel_size=3, padding=1),
        )
        self.cbam4 = CBAMBlock(f * 8) if use_cbam else nn.Identity()

        # Global Feature Aggregation
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(f * 8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts deep 256-dimensional semantic feature vector prior to classification head.
        """
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.cbam2(x)
        x = self.pool2(x)

        x = self.stage3(x)
        x = self.cbam3(x)
        x = self.pool3(x)

        x = self.stage4(x)
        x = self.cbam4(x)
        x = self.global_pool(x)
        features = torch.flatten(x, 1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)

        x = self.stage2(x)
        x = self.cbam2(x)
        x = self.pool2(x)

        x = self.stage3(x)
        x = self.cbam3(x)
        x = self.pool3(x)

        x = self.stage4(x)
        x = self.cbam4(x)

        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits

    def count_parameters(self) -> Dict[str, Any]:
        """
        Computes total and trainable parameter counts.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
        }
