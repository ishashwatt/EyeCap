"""
End-to-End Pipeline Sanity and Integration Tests
"""

import os
import shutil
import torch
import pytest
from src.data_loader import SyntheticEyeDataGenerator, build_dataloaders
from src.models.custom_ocunet import OcuNetCBAM
from src.train import ModelTrainer
from src.evaluate import ClinicalEvaluator


@pytest.fixture(scope="session")
def temp_sample_data(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("sample_eye_data")
    classes = ("Cataract", "Diabetic_Retinopathy", "Glaucoma", "Normal")
    SyntheticEyeDataGenerator.generate_sample_dataset(
        output_dir=str(temp_dir), samples_per_class=6, classes=classes
    )
    return str(temp_dir)


def test_dataloader_and_training_step(temp_sample_data, tmp_path):
    classes = ["Cataract", "Diabetic_Retinopathy", "Glaucoma", "Normal"]
    train_loader, val_loader, test_loader, _ = build_dataloaders(
        data_dir=temp_sample_data,
        classes=classes,
        batch_size=4,
        split_ratios=(0.6, 0.2, 0.2),
    )

    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0

    model = OcuNetCBAM(in_channels=3, num_classes=4, base_filters=16)
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=classes,
        device="cpu",
        learning_rate=0.001,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        results_dir=str(tmp_path / "results"),
    )

    results = trainer.train(epochs=2, patience=2)
    assert "best_val_acc" in results
    assert len(results["history"]["train_loss"]) == 2
