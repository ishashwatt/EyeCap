"""
Pretrained Transfer Learning Architectures for Ocular Disease Benchmarking
Provides modular wrappers for DenseNet-121, EfficientNet-B0, and ResNet-50.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any


class TransferEyeModel(nn.Module):
    """
    Fine-tunable transfer learning backbone for medical fundus classification.
    Supports DenseNet-121, EfficientNet-B0, and ResNet-50.
    """

    def __init__(
        self,
        model_name: str = "densenet121",
        num_classes: int = 4,
        pretrained: bool = True,
        dropout_rate: float = 0.4,
    ):
        super(TransferEyeModel, self).__init__()
        self.model_name = model_name.lower()
        self.num_classes = num_classes

        if "densenet" in self.model_name:
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            self.backbone = models.densenet121(weights=weights)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )

        elif "efficientnet" in self.model_name:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )

        elif "resnet" in self.model_name:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        else:
            raise ValueError(f"Unsupported transfer model architecture: {model_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def count_parameters(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_name": self.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
        }
