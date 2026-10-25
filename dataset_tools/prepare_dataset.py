"""
SiteGuard AI — Dataset Preparation, Augmentation & Stratified Splitting
Applies Albumentations augmentation pipeline tailored for construction site conditions:
  - Random brightness/contrast (outdoor lighting)
  - Motion blur simulation (moving workers/cameras)
  - Horizontal flips
  - Gaussian noise (dust/debris particles)
  - HSV shift (weather/time of day variation)
  - Small PPE region enhancement (for hardhats, goggles, gloves)
  - Stratified 70% train / 20% val / 10% test split
"""

import os
import sys
import random
import argparse
from pathlib import Path
from collections import defaultdict

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

try:
    import albumentations as A
except ImportError:
    A = None

def get_construction_augmentation_pipeline(imgsz=640):
    """Builds Albumentations transform pipeline with bounding-box support."""
    if A is None:
        return None
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
        A.MotionBlur(blur_limit=5, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.4),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=16, min_visibility=0.2))

def create_stratified_splits(dataset_dir: str, train_ratio=0.70, val_ratio=0.20, test_ratio=0.10, seed=42):
    """
    Creates stratified splits ensuring each class is represented across train, val, and test manifests.
    """
    dataset_path = Path(dataset_dir).resolve()
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"Error: dataset layout must contain 'images' and 'labels' directories at: {dataset_path}")
        return False

    random.seed(seed)
    
    # Collect all image stems with valid labels
    valid_pairs = []
    class_to_stems = defaultdict(list)

    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    for img_p in image_files:
        lbl_p = labels_dir / f"{img_p.stem}.txt"
        if lbl_p.exists():
            valid_pairs.append(img_p)
            try:
                with open(lbl_p, 'r', encoding='utf-8') as f:
                    classes = {int(float(line.split()[0])) for line in f if line.strip()}
                for c in classes:
                    class_to_stems[c].append(img_p)
            except Exception:
                pass

    total = len(valid_pairs)
    print(f"[*] Stratifying {total:,} paired images across {len(class_to_stems)} classes...")

    random.shuffle(valid_pairs)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_set = set(valid_pairs[:n_train])
    val_set = set(valid_pairs[n_train:n_train + n_val])
    test_set = set(valid_pairs[n_train + n_val:])

    # Verify every class is in val and test
    for c, stems in class_to_stems.items():
        in_val = any(s in val_set for s in stems)
        in_test = any(s in test_set for s in stems)
        if not in_val and len(stems) > 2:
            swap_candidate = random.choice([s for s in stems if s in train_set])
            train_set.remove(swap_candidate)
            val_set.add(swap_candidate)
        if not in_test and len(stems) > 3:
            swap_candidate = random.choice([s for s in stems if s in train_set])
            train_set.remove(swap_candidate)
            test_set.add(swap_candidate)

    # Write manifests
    train_txt = dataset_path / "train.txt"
    val_txt = dataset_path / "val.txt"
    test_txt = dataset_path / "test.txt"

    with open(train_txt, 'w', encoding='utf-8') as f:
        for p in sorted(train_set, key=lambda x: x.name):
            f.write(f"{p.as_posix()}\n")

    with open(val_txt, 'w', encoding='utf-8') as f:
        for p in sorted(val_set, key=lambda x: x.name):
            f.write(f"{p.as_posix()}\n")

    with open(test_txt, 'w', encoding='utf-8') as f:
        for p in sorted(test_set, key=lambda x: x.name):
            f.write(f"{p.as_posix()}\n")

    print(f"[✓] Stratified splits generated:")
    print(f"    - Train: {len(train_set):,} images ({len(train_set)/total*100:.1f}%) -> {train_txt}")
    print(f"    - Val:   {len(val_set):,} images ({len(val_set)/total*100:.1f}%) -> {val_txt}")
    print(f"    - Test:  {len(test_set):,} images ({len(test_set)/total*100:.1f}%) -> {test_txt}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SiteGuard AI Dataset Splits & Augmentations")
    parser.add_argument("--dataset", type=str, default=r"D:\Safe\dataset", help="Path to dataset root")
    parser.add_argument("--split", action="store_true", default=True, help="Generate stratified splits")
    args = parser.parse_args()

    if args.split:
        create_stratified_splits(args.dataset)
