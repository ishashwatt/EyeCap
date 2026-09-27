"""
Script to download real-world benchmark Eye Disease & Cataract dataset from Kaggle
Dataset: 'gunavenkatdoddi/eye-diseases-classification' or 'drskprabhakar/cataract-dr-normal-glaucoma-fundus-images-dataset'
Organizes images cleanly into data/raw/{Cataract, Diabetic_Retinopathy, Glaucoma, Normal}
"""

import os
import shutil
import glob
import kagglehub


def download_and_organize_dataset(target_dir: str = "data/raw"):
    print("[*] Downloading benchmark Eye Diseases Classification dataset via kagglehub...")
    os.makedirs(target_dir, exist_ok=True)

    try:
        # Download latest version of the standard 4-class eye disease dataset
        dataset_path = kagglehub.dataset_download("gunavenkatdoddi/eye-diseases-classification")
        print(f"[+] Dataset successfully downloaded to cache: {dataset_path}")
    except Exception as e:
        print(f"[!] Primary dataset download encountered: {e}. Trying secondary mirror...")
        try:
            dataset_path = kagglehub.dataset_download("drskprabhakar/cataract-dr-normal-glaucoma-fundus-images-dataset")
            print(f"[+] Secondary dataset successfully downloaded to: {dataset_path}")
        except Exception as e2:
            print(f"[!] Secondary download failed: {e2}")
            return None

    # Map discovered folders to our canonical class names
    class_mapping = {
        "cataract": "Cataract",
        "cataracts": "Cataract",
        "diabetic_retinopathy": "Diabetic_Retinopathy",
        "dr": "Diabetic_Retinopathy",
        "glaucoma": "Glaucoma",
        "normal": "Normal",
    }

    canonical_classes = ["Cataract", "Diabetic_Retinopathy", "Glaucoma", "Normal"]
    for c in canonical_classes:
        os.makedirs(os.path.join(target_dir, c), exist_ok=True)

    copied_counts = {c: 0 for c in canonical_classes}

    # Walk through downloaded folder and copy images
    for root, dirs, files in os.walk(dataset_path):
        folder_name = os.path.basename(root).lower().replace(" ", "_").replace("-", "_")
        for key, canonical_name in class_mapping.items():
            if key in folder_name:
                dest_dir = os.path.join(target_dir, canonical_name)
                for f in files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                        src_file = os.path.join(root, f)
                        dst_file = os.path.join(dest_dir, f"{canonical_name.lower()}_{f}")
                        if not os.path.exists(dst_file):
                            shutil.copy2(src_file, dst_file)
                        copied_counts[canonical_name] += 1
                break

    print("\n=======================================================")
    print("          KAGGLE DATASET INGESTION SUMMARY              ")
    print("=======================================================")
    for cls_name, count in copied_counts.items():
        print(f" {cls_name.ljust(22)} : {count} real images")
    print(f" Total Ingested Images: {sum(copied_counts.values())}")
    print(f" Stored in: {os.path.abspath(target_dir)}")
    print("=======================================================\n")

    return target_dir


if __name__ == "__main__":
    download_and_organize_dataset()
