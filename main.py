"""
Master CLI Entrypoint for AI-Based Ocular & Cataract Disease Detection (EyeCap MCPS)
Supports: dataset generation, training, evaluation, XAI Grad-CAM generation, and ablation studies.
"""

import os
import argparse
import yaml
import cv2
import numpy as np
import torch
from src.data_loader import SyntheticEyeDataGenerator, build_dataloaders
from src.models.custom_ocunet import OcuNetCBAM
from src.models.transfer_models import TransferEyeModel
from src.models.hybrid_ensemble import HybridEnsembleModel
from src.train import ModelTrainer
from src.evaluate import ClinicalEvaluator
from src.explainability import generate_and_save_gradcam_visuals


def load_config(config_path: str = "config/config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(model_name: str, num_classes: int, config: dict) -> torch.nn.Module:
    model_name = model_name.lower()
    if model_name == "ocunet_cbam":
        return OcuNetCBAM(
            in_channels=3,
            num_classes=num_classes,
            base_filters=config["model"]["ocunet"]["base_filters"],
            dropout_rate=config["model"]["ocunet"]["dropout_rate"],
            use_cbam=config["model"]["ocunet"]["use_cbam"],
        )
    elif model_name == "ocunet_no_cbam":
        return OcuNetCBAM(
            in_channels=3,
            num_classes=num_classes,
            base_filters=config["model"]["ocunet"]["base_filters"],
            dropout_rate=config["model"]["ocunet"]["dropout_rate"],
            use_cbam=False,
        )
    elif model_name in ["densenet121", "efficientnet_b0", "resnet50"]:
        return TransferEyeModel(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=config["model"]["transfer"]["pretrained"],
        )
    elif model_name == "hybrid_ensemble":
        return HybridEnsembleModel(
            num_classes=num_classes,
            pretrained=config["model"]["transfer"]["pretrained"],
        )
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="EyeCap MCPS AI Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["generate_data", "train", "evaluate", "explain", "ablation"],
        help="Execution mode",
    )
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--model", type=str, default=None, help="Override model architecture")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory")

    args = parser.parse_args()
    config = load_config(args.config)

    classes = config["dataset"]["classes"]
    num_classes = len(classes)
    data_dir = args.data_dir or config["dataset"]["sample_data_dir"]

    # 1. Mode: Generate Synthetic Benchmark Data
    if args.mode == "generate_data" or not os.path.exists(data_dir):
        print(f"[*] Generating benchmark sample dataset in: {data_dir}...")
        SyntheticEyeDataGenerator.generate_sample_dataset(
            output_dir=data_dir, samples_per_class=30, classes=tuple(classes)
        )
        print("[+] Dataset generation complete.")
        if args.mode == "generate_data":
            return

    # Load DataLoaders
    batch_size = config["dataset"]["batch_size"]
    train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(
        data_dir=data_dir,
        classes=classes,
        batch_size=batch_size,
        split_ratios=(
            config["dataset"]["split_ratios"]["train"],
            config["dataset"]["split_ratios"]["val"],
            config["dataset"]["split_ratios"]["test"],
        ),
        random_seed=config["project"]["random_seed"],
    )

    arch_name = args.model or config["model"]["architecture"]
    model = build_model(arch_name, num_classes, config)

    # Print model statistics
    if hasattr(model, "count_parameters"):
        params_info = model.count_parameters()
        print(f"[*] Model Architecture: {arch_name.upper()}")
        print(f"    - Total Parameters: {params_info['total_parameters']:,}")
        print(f"    - Model Size: {params_info['model_size_mb']:.2f} MB")

    epochs = args.epochs or config["training"]["epochs"]
    device = config["training"]["device"]

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=classes,
        device=device,
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        loss_type=config["training"]["loss_function"],
        focal_gamma=config["training"]["focal_loss"]["gamma"],
        checkpoint_dir="experiments/checkpoints",
        results_dir=config["evaluation"]["output_dir"],
    )

    # 2. Mode: Train
    if args.mode == "train":
        train_results = trainer.train(
            epochs=epochs,
            patience=config["training"]["early_stopping"]["patience"],
            model_save_name=f"{arch_name}_best.pth",
        )

        # Automatically evaluate on test set post-training
        print("\n[*] Evaluating Best Model on Held-out Test Set...")
        evaluator = ClinicalEvaluator(class_names=classes, output_dir=config["evaluation"]["output_dir"])

        trainer.model.eval()
        test_preds, test_labels, test_probs = [], [], []
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs = imgs.to(trainer.device)
                outputs = trainer.model(imgs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(lbls.numpy())
                test_probs.extend(probs.cpu().numpy())

        test_metrics = evaluator.evaluate(
            y_true=np.array(test_labels),
            y_pred=np.array(test_preds),
            y_probs=np.array(test_probs),
            prefix="test",
        )

        print("\n=======================================================")
        print("          FINAL CLINICAL TEST RESULTS SUMMARY           ")
        print("=======================================================")
        print(f" Accuracy            : {test_metrics['accuracy']*100:.2f}%")
        print(f" Precision (Macro)   : {test_metrics['precision_macro']*100:.2f}%")
        print(f" Recall / Sensitivity: {test_metrics['recall_macro']*100:.2f}%")
        print(f" Specificity (Macro) : {test_metrics['specificity_macro']*100:.2f}%")
        print(f" F1-Score (Macro)    : {test_metrics['f1_macro']*100:.2f}%")
        print(f" Matthews Corr (MCC) : {test_metrics['mcc']:.4f}")
        print(f" ROC-AUC (Macro OVR) : {test_metrics['roc_auc']:.4f}")
        print("=======================================================\n")

    # 3. Mode: Explain (Grad-CAM XAI)
    elif args.mode == "explain":
        # Load best checkpoint if available
        ckpt_path = os.path.join("experiments/checkpoints", f"{arch_name}_best.pth")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=trainer.device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"[+] Loaded weights from {ckpt_path}")

        # Choose target layer for Grad-CAM
        if hasattr(model, "stage4"):
            target_layer = model.stage4[-1] # Final conv block in OcuNet
        elif hasattr(model, "backbone") and hasattr(model.backbone, "features"):
            target_layer = model.backbone.features[-1]
        else:
            target_layer = list(model.modules())[-4]

        # Collect sample images for each class
        sample_batch = []
        for cls_idx, cls_name in enumerate(classes):
            cls_folder = os.path.join(data_dir, cls_name)
            files = [f for f in os.listdir(cls_folder) if f.endswith(('.png', '.jpg'))]
            if files:
                fpath = os.path.join(cls_folder, files[0])
                raw_img = cv2.imread(fpath)
                tensor = train_loader.dataset.preprocessor.process_image(raw_img, return_tensor=True)
                sample_batch.append((tensor, cls_idx, fpath, raw_img))

        print(f"[*] Generating Grad-CAM heatmaps for {len(sample_batch)} classes...")
        generate_and_save_gradcam_visuals(
            model=model.to(trainer.device),
            target_layer=target_layer,
            sample_images=sample_batch,
            class_names=classes,
            output_dir=os.path.join(config["evaluation"]["output_dir"], "xai"),
            device=str(trainer.device),
        )
        print(f"[+] Grad-CAM visualizations saved to {config['evaluation']['output_dir']}/xai/")

    # 4. Mode: Ablation Studies
    elif args.mode == "ablation":
        print("[*] Running Ablation Suite: (1) OcuNet+CBAM vs (2) OcuNet No-CBAM vs (3) DenseNet-121...")
        architectures = ["ocunet_cbam", "ocunet_no_cbam", "densenet121"]
        ablation_summary = {}

        for arch in architectures:
            print(f"\n---> Training Ablation Variant: {arch}")
            m = build_model(arch, num_classes, config)
            t = ModelTrainer(
                model=m,
                train_loader=train_loader,
                val_loader=val_loader,
                class_names=classes,
                device=device,
                learning_rate=config["training"]["learning_rate"],
                loss_type=config["training"]["loss_function"],
                checkpoint_dir="experiments/checkpoints",
                results_dir=config["evaluation"]["output_dir"],
            )
            res = t.train(epochs=args.epochs or 8, patience=4, model_save_name=f"ablation_{arch}.pth")
            ablation_summary[arch] = res["best_val_acc"]

        print("\n=======================================================")
        print("                 ABLATION STUDY RESULTS                ")
        print("=======================================================")
        for arch, score in ablation_summary.items():
            print(f" {arch.ljust(20)} : Best Val Accuracy = {score*100:.2f}%")
        print("=======================================================\n")


if __name__ == "__main__":
    main()
