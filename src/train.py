"""
Robust Model Training Engine for Medical Eye Disease Classification
Implements AdamW optimization, Cosine Annealing, Early Stopping, and Learning Curve visualizer.
"""

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Tuple
from src.loss_functions import MultiClassFocalLoss, LabelSmoothingCrossEntropy
from src.evaluate import ClinicalEvaluator


class ModelTrainer:
    """
    Orchestrates neural network training with clinical validation safeguards.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_names: List[str],
        device: str = "cuda",
        learning_rate: float = 0.0003,
        weight_decay: float = 0.0001,
        loss_type: str = "focal_loss",
        focal_gamma: float = 2.0,
        checkpoint_dir: str = "experiments/checkpoints",
        results_dir: str = "experiments/results",
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Configure Loss function
        if loss_type == "focal_loss":
            self.criterion = MultiClassFocalLoss(gamma=focal_gamma)
        elif loss_type == "label_smoothing":
            self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer: AdamW
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.evaluator = ClinicalEvaluator(class_names=class_names, output_dir=results_dir)

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Gradient clipping for training stability
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        return epoch_loss, epoch_acc

    def validate_epoch(self) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * images.size(0)
                correct += torch.sum(preds == labels.data).item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        return (
            epoch_loss,
            epoch_acc,
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
        )

    def train(
        self,
        epochs: int = 25,
        patience: int = 7,
        model_save_name: str = "best_model.pth",
    ) -> Dict[str, Any]:
        """
        Runs full training with cosine annealing learning rate schedule and early stopping.
        """
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_model_weights = copy.deepcopy(self.model.state_dict())
        epochs_no_improve = 0

        start_time = time.time()
        print(f"[*] Starting Training on device: {self.device} for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_labels, val_preds, val_probs = self.validate_epoch()
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

            # Checkpoint on best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_model_weights = copy.deepcopy(self.model.state_dict())
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    },
                    os.path.join(self.checkpoint_dir, model_save_name),
                )
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"[!] Early stopping triggered at epoch {epoch}.")
                    break

        elapsed_time = time.time() - start_time
        print(f"[+] Training completed in {elapsed_time:.2f}s. Best Val Acc: {best_val_acc*100:.2f}%")

        # Load best model weights
        self.model.load_state_dict(best_model_weights)

        # Plot training curves
        self.plot_learning_curves(history)

        return {
            "history": history,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "training_time_sec": elapsed_time,
        }

    def plot_learning_curves(self, history: Dict[str, List[float]]):
        epochs_range = range(1, len(history["train_loss"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Loss Plot
        ax1.plot(epochs_range, history["train_loss"], label="Train Loss", color="royalblue", lw=2)
        ax1.plot(epochs_range, history["val_loss"], label="Val Loss", color="crimson", lw=2)
        ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epochs", fontsize=10)
        ax1.set_ylabel("Loss", fontsize=10)
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Accuracy Plot
        ax2.plot(epochs_range, history["train_acc"], label="Train Acc", color="royalblue", lw=2)
        ax2.plot(epochs_range, history["val_acc"], label="Val Acc", color="crimson", lw=2)
        ax2.set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epochs", fontsize=10)
        ax2.set_ylabel("Accuracy", fontsize=10)
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.results_dir, "learning_curves.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
