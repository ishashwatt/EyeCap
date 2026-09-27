"""
Unit Tests for Deep Learning Architectures and Attention Modules
"""

import torch
import pytest
from src.models.cbam import ChannelAttention, SpatialAttention, CBAMBlock
from src.models.custom_ocunet import OcuNetCBAM
from src.models.transfer_models import TransferEyeModel
from src.models.hybrid_ensemble import HybridEnsembleModel


def test_channel_attention():
    att = ChannelAttention(in_channels=64, reduction_ratio=16)
    x = torch.randn(2, 64, 32, 32)
    out = att(x)
    assert out.shape == (2, 64, 32, 32)


def test_spatial_attention():
    att = SpatialAttention(kernel_size=7)
    x = torch.randn(2, 64, 32, 32)
    out = att(x)
    assert out.shape == (2, 64, 32, 32)


def test_cbam_block():
    cbam = CBAMBlock(in_channels=128)
    x = torch.randn(2, 128, 16, 16)
    out = cbam(x)
    assert out.shape == (2, 128, 16, 16)


def test_ocunet_cbam_forward_and_params():
    model = OcuNetCBAM(in_channels=3, num_classes=4, base_filters=32, use_cbam=True)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 4)

    # Verify parameter efficiency (< 1.5 million parameters)
    params_dict = model.count_parameters()
    assert params_dict["total_parameters"] < 1_500_000
    assert params_dict["total_parameters"] > 100_000


def test_hybrid_ensemble_forward():
    model = HybridEnsembleModel(num_classes=4, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 4)
