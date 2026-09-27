"""
Dataset Loader and Data Generator Module
Handles loading Kaggle datasets (ODIR, Cataract, RFMiD) and provides an automated
synthetic clinical eye image generator for instant testing, verification, and reproducibility.
"""

import os
import glob
import math
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
from src.preprocessing import ClinicalEyePreprocessor
from src.augmentation import ClinicalAugmentor


class EyeDiseaseDataset(Dataset):
    """
    PyTorch Dataset for multi-class ocular and cataract disease detection.
    Reads images, applies ClinicalEyePreprocessor, and applies ClinicalAugmentor.
    """

    def __init__(
        self,
        file_paths: List[str],
        labels: List[int],
        classes: List[str],
        preprocessor: Optional[ClinicalEyePreprocessor] = None,
        augmentor: Optional[ClinicalAugmentor] = None,
        is_training: bool = False,
    ):
        self.file_paths = file_paths
        self.labels = labels
        self.classes = classes
        self.preprocessor = preprocessor or ClinicalEyePreprocessor()
        self.augmentor = augmentor or ClinicalAugmentor()
        self.is_training = is_training
        self.cache = {}

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        # Use in-memory cache if already preprocessed
        if idx in self.cache:
            tensor = self.cache[idx]
        else:
            try:
                tensor = self.preprocessor.preprocess_from_file(img_path, return_tensor=True)
            except Exception:
                tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            self.cache[idx] = tensor

        # Clone tensor before applying dynamic augmentations
        if self.is_training and self.augmentor is not None:
            tensor = self.augmentor.augment_tensor(tensor.clone(), is_training=True)

        return tensor, label


class SyntheticEyeDataGenerator:
    """
    Generates synthetic fundus/ocular images simulating:
    1. Normal: Sharp retinal vasculature, clear optic disc.
    2. Cataract: Lens clouding, diffused blurred central opacity.
    3. Diabetic Retinopathy: Microaneurysms (red dots), bright exudates, hemorrhages.
    4. Glaucoma: Enlarged optic cup-to-disc ratio (cupping), neuroretinal rim thinning.
    """

    @staticmethod
    def create_synthetic_image(class_name: str, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        h, w = size
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # 1. Base orange/red retinal background
        center = (w // 2, h // 2)
        radius = min(w, h) // 2 - 10

        # Circular mask for retinal fundus
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)

        # Create gradient fundus background
        for y in range(h):
            for x in range(w):
                if mask[y, x] > 0:
                    dist = math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
                    factor = 1.0 - (dist / radius) * 0.3
                    img[y, x] = [
                        int(20 * factor),   # B
                        int(90 * factor),   # G
                        int(180 * factor),  # R
                    ]

        # 2. Draw Optic Disc
        od_center = (center[0] - 50, center[1])
        od_radius = 28
        cv2.circle(img, od_center, od_radius, (60, 200, 240), -1)

        # 3. Class-specific pathological features
        if class_name == "Cataract":
            # Add widespread foggy central clouding (Media Haze)
            haze = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.circle(haze, center, int(radius * 0.75), (180, 190, 200), -1)
            haze = cv2.GaussianBlur(haze, (45, 45), 0)
            img = cv2.addWeighted(img, 0.45, haze, 0.55, 0)

        elif class_name == "Glaucoma":
            # Significant optic cup enlargement (High Cup-to-Disc Ratio)
            cup_radius = int(od_radius * 0.78)
            cv2.circle(img, od_center, cup_radius, (120, 240, 255), -1)

        elif class_name == "Diabetic_Retinopathy":
            # Microaneurysms (tiny dark red spots) & Cotton wool spots (bright yellow)
            for _ in range(12):
                rx = random.randint(center[0] - 60, center[0] + 60)
                ry = random.randint(center[1] - 60, center[1] + 60)
                if mask[ry, rx] > 0:
                    cv2.circle(img, (rx, ry), 2, (10, 10, 120), -1) # Hemorrhage
            for _ in range(6):
                ex_x = random.randint(center[0] - 50, center[0] + 50)
                ex_y = random.randint(center[1] - 50, center[1] + 50)
                if mask[ex_y, ex_x] > 0:
                    cv2.circle(img, (ex_x, ex_y), 3, (80, 230, 250), -1) # Exudate

        # Add branching blood vessels
        for i in range(4):
            pt1 = od_center
            angle = i * (math.pi / 2) + random.uniform(-0.3, 0.3)
            pt2 = (
                int(od_center[0] + radius * 0.7 * math.cos(angle)),
                int(od_center[1] + radius * 0.7 * math.sin(angle)),
            )
            cv2.line(img, pt1, pt2, (15, 30, 110), 2)

        return img

    @staticmethod
    def generate_sample_dataset(
        output_dir: str = "data/sample_data",
        samples_per_class: int = 25,
        classes: Tuple[str, ...] = ("Cataract", "Diabetic_Retinopathy", "Glaucoma", "Normal"),
    ) -> Dict[str, List[str]]:
        """
        Generates and writes a clean multi-class synthetic dataset to disk.
        """
        os.makedirs(output_dir, exist_ok=True)
        dataset_paths = {}

        for cls in classes:
            cls_dir = os.path.join(output_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            file_list = []
            for i in range(samples_per_class):
                img = SyntheticEyeDataGenerator.create_synthetic_image(cls)
                file_path = os.path.join(cls_dir, f"{cls.lower()}_{i+1:03d}.png")
                cv2.imwrite(file_path, img)
                file_list.append(file_path)
            dataset_paths[cls] = file_list

        return dataset_paths


def build_dataloaders(
    data_dir: str,
    classes: List[str],
    batch_size: int = 16,
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    random_seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Scans data_dir for class subfolders, creates stratified splits, and returns DataLoaders.
    """
    random.seed(random_seed)
    all_files = []
    all_labels = []

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    for cls_name in classes:
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue
        # Match common image formats
        img_paths = glob.glob(os.path.join(cls_dir, "*.png")) + \
                    glob.glob(os.path.join(cls_dir, "*.jpg")) + \
                    glob.glob(os.path.join(cls_dir, "*.jpeg"))
        for p in img_paths:
            all_files.append(p)
            all_labels.append(class_to_idx[cls_name])

    if len(all_files) == 0:
        raise ValueError(f"No images found in {data_dir}. Ensure class subfolders exist.")

    # Group by class to perform stratified split
    cls_indices = {idx: [] for idx in range(len(classes))}
    for idx, label in enumerate(all_labels):
        cls_indices[label].append(idx)

    train_files, train_labels = [], []
    val_files, val_labels = [], []
    test_files, test_labels = [], []

    train_r, val_r, test_r = split_ratios

    for label, indices in cls_indices.items():
        random.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_r)
        n_val = int(n * val_r)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        for i in train_idx:
            train_files.append(all_files[i])
            train_labels.append(all_labels[i])
        for i in val_idx:
            val_files.append(all_files[i])
            val_labels.append(all_labels[i])
        for i in test_idx:
            test_files.append(all_files[i])
            test_labels.append(all_labels[i])

    # Instantiate datasets
    preprocessor = ClinicalEyePreprocessor()
    augmentor = ClinicalAugmentor()

    train_ds = EyeDiseaseDataset(train_files, train_labels, classes, preprocessor, augmentor, is_training=True)
    val_ds = EyeDiseaseDataset(val_files, val_labels, classes, preprocessor, augmentor=None, is_training=False)
    test_ds = EyeDiseaseDataset(test_files, test_labels, classes, preprocessor, augmentor=None, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_to_idx
