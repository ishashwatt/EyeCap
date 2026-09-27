"""
Hybrid Multi-Stream Ensemble Architecture
Extracts high-dimensional deep feature embeddings from both custom OcuNet-CBAM
and pre-trained DenseNet-121 backbones, concatenating them into a gradient-boosted
or soft-voting ensemble classifier for superior clinical reliability.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from src.models.custom_ocunet import OcuNetCBAM
from src.models.transfer_models import TransferEyeModel


class HybridEnsembleModel(nn.Module):
    """
    End-to-End Dual-Stream Deep Feature Ensemble:
    Stream 1: OcuNet-CBAM (Lightweight spatial/channel attention features)
    Stream 2: DenseNet-121 (Dense multi-layer feature reuse representations)
    Fusion: Feature concatenation + Multi-Layer Perceptron / Soft-Voting classification
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout_rate: float = 0.4):
        super(HybridEnsembleModel, self).__init__()
        self.num_classes = num_classes

        # Stream 1
        self.stream_ocunet = OcuNetCBAM(in_channels=3, num_classes=num_classes, use_cbam=True)

        # Stream 2
        self.stream_densenet = TransferEyeModel(
            model_name="densenet121", num_classes=num_classes, pretrained=pretrained
        )
        # Extract features prior to classification head (DenseNet121 has 1024 features)
        self.densenet_features = self.stream_densenet.backbone.features
        self.densenet_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Combined feature dimensionality: 256 (OcuNet) + 1024 (DenseNet) = 1280
        combined_dim = 256 + 1024

        # Fusion Meta-Classifier
        self.fusion_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stream 1 features
        feat_ocu = self.stream_ocunet.extract_features(x)

        # Stream 2 features
        feat_dense = self.densenet_features(x)
        feat_dense = nn.functional.relu(feat_dense, inplace=True)
        feat_dense = self.densenet_pool(feat_dense)
        feat_dense = torch.flatten(feat_dense, 1)

        # Concatenate dual-stream representations
        fused = torch.cat([feat_ocu, feat_dense], dim=1)

        # Final classification logits
        logits = self.fusion_head(fused)
        return logits
