"""
Construction Safety PPE Dataset Downloader & Multi-Source Merger
================================================================
Merges multiple Roboflow and local YOLO-formatted datasets into a single
unified 17-class dataset matching the AI-Based Construction Safety Monitoring spec:

Classes (17 Total):
  0: person (worker anchor)
  1: helmet            / 2: no-helmet
  3: goggles           / 4: no-goggles
  5: earmuff           / 6: no-earmuff
  7: vest              / 8: no-vest
  9: safety-pants      / 10: no-safety-pants
  11: gloves           / 12: no-gloves
  13: boots            / 14: no-boots
  15: harness          / 16: no-harness

Author: Construction CV Safety Team
"""

import os
import re
import sys
import glob
import shutil
import logging
import argparse
from typing import Dict, List, Tuple, Optional
from collections import Counter
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("DatasetMerger")

# Standard 17 Class Definition
TARGET_CLASSES: List[str] = [
    "person",           # 0
    "helmet",           # 1
    "no-helmet",        # 2
    "goggles",          # 3
    "no-goggles",       # 4
    "earmuff",          # 5
    "no-earmuff",       # 6
    "vest",             # 7
    "no-vest",          # 8
    "safety-pants",     # 9
    "no-safety-pants",  # 10
    "gloves",           # 11
    "no-gloves",        # 12
    "boots",            # 13
    "no-boots",         # 14
    "harness",          # 15
    "no-harness"        # 16
]

TARGET_CLASS_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(TARGET_CLASSES)}

# Semantic Class Alias Normalizer (maps diverse annotations across datasets into canonical class names)
CLASS_ALIASES: Dict[str, str] = {
    # 0: person
    "person": "person",
    "worker": "person",
    "human": "person",
    "workers": "person",
    "people": "person",

    # 1: helmet / 2: no-helmet
    "helmet": "helmet",
    "hard-hat": "helmet",
    "hardhat": "helmet",
    "hard_hat": "helmet",
    "safety-helmet": "helmet",
    "safety_helmet": "helmet",
    "no-helmet": "no-helmet",
    "no_helmet": "no-helmet",
    "no-hardhat": "no-helmet",
    "no_hardhat": "no-helmet",
    "head": "no-helmet",  # Head without helmet in many PPE datasets
    "without_helmet": "no-helmet",
    "without-helmet": "no-helmet",

    # 3: goggles / 4: no-goggles
    "goggles": "goggles",
    "goggle": "goggles",
    "glasses": "goggles",
    "safety-glasses": "goggles",
    "safety_glasses": "goggles",
    "eye-protection": "goggles",
    "eye_protection": "goggles",
    "eyewear": "goggles",
    "no-goggles": "no-goggles",
    "no_goggles": "no-goggles",
    "no-glasses": "no-goggles",
    "without-goggles": "no-goggles",
    "face": "no-goggles",  # In datasets where bare face represents no-goggles anchor

    # 5: earmuff / 6: no-earmuff
    "earmuff": "earmuff",
    "earmuffs": "earmuff",
    "ear-muff": "earmuff",
    "ear-muffs": "earmuff",
    "ear_muffs": "earmuff",
    "ear-mufs": "earmuff",
    "hearing-protection": "earmuff",
    "hearing_protection": "earmuff",
    "no-earmuff": "no-earmuff",
    "no_earmuff": "no-earmuff",
    "no-earmuffs": "no-earmuff",
    "without-earmuff": "no-earmuff",
    "ear": "no-earmuff",  # Exposed ear anchor represents missing earmuff

    # 7: vest / 8: no-vest
    "vest": "vest",
    "safety-vest": "vest",
    "safety_vest": "vest",
    "reflective-vest": "vest",
    "hi-vis": "vest",
    "hi-vis-vest": "vest",
    "jacket": "vest",
    "no-vest": "no-vest",
    "no_vest": "no-vest",
    "no-safety-vest": "no-vest",
    "without-vest": "no-vest",
    "without_vest": "no-vest",

    # 9: safety-pants / 10: no-safety-pants
    "safety-pants": "safety-pants",
    "safety_pants": "safety-pants",
    "safety-suit": "safety-pants",
    "reflective-pants": "safety-pants",
    "hi-vis-pants": "safety-pants",
    "protective-pants": "safety-pants",
    "chaps": "safety-pants",
    "no-safety-pants": "no-safety-pants",
    "no_safety_pants": "no-safety-pants",
    "no-pants": "no-safety-pants",
    "regular-pants": "no-safety-pants",

    # 11: gloves / 12: no-gloves
    "gloves": "gloves",
    "glove": "gloves",
    "safety-gloves": "gloves",
    "work-gloves": "gloves",
    "hand-protection": "gloves",
    "no-gloves": "no-gloves",
    "no_gloves": "no-gloves",
    "hands": "no-gloves",  # Bare hands in dataset represents absence of gloves
    "hand": "no-gloves",
    "without-gloves": "no-gloves",

    # 13: boots / 14: no-boots
    "boots": "boots",
    "boot": "boots",
    "shoes": "boots",
    "safety-shoes": "boots",
    "safety-boots": "boots",
    "steel-toe": "boots",
    "safety_boots": "boots",
    "no-boots": "no-boots",
    "no_boots": "no-boots",
    "foot": "no-boots",  # Bare foot or non-safety shoe anchor
    "no-shoes": "no-boots",
    "regular-shoes": "no-boots",

    # 15: harness / 16: no-harness
    "harness": "harness",
    "safety-harness": "harness",
    "safety_harness": "harness",
    "safety-belt": "harness",
    "fall-protection": "harness",
    "fall_arrest": "harness",
    "no-harness": "no-harness",
    "no_harness": "no-harness",
    "without-harness": "no-harness"
}


def normalize_class_name(name: str) -> Optional[str]:
    """Sanitize and look up canonical class name."""
    clean = re.sub(r"[_\s]+", "-", name.strip().lower())
    if clean in CLASS_ALIASES:
        return CLASS_ALIASES[clean]
    # Try underscore variant
    alt = clean.replace("-", "_")
    if alt in CLASS_ALIASES:
        return CLASS_ALIASES[alt]
    return None


def download_roboflow_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    output_dir: str
) -> str:
    """Download YOLOv8 dataset using Roboflow Python SDK."""
    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError(
            "Roboflow SDK not found. Install it via: pip install roboflow"
        )

    logger.info(f"Connecting to Roboflow workspace: {workspace}, project: {project}, v{version}...")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("yolov8", location=output_dir)
    logger.info(f"Successfully downloaded Roboflow dataset to: {dataset.location}")
    return dataset.location


def load_dataset_class_map(data_yaml_path: str) -> Dict[int, int]:
    """
    Parses a source dataset's data.yaml and creates an ID translation map:
    source_class_id -> target_class_id (or -1 if ignored).
    """
    if not os.path.isfile(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}")

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    raw_names = data.get("names", [])
    if isinstance(raw_names, list):
        src_names = {idx: name for idx, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        src_names = {int(k): str(v) for k, v in raw_names.items()}
    else:
        raise ValueError(f"Unexpected names format in {data_yaml_path}: {raw_names}")

    translation_map = {}
    for src_id, src_name in src_names.items():
        canonical = normalize_class_name(src_name)
        if canonical and canonical in TARGET_CLASS_TO_ID:
            target_id = TARGET_CLASS_TO_ID[canonical]
            translation_map[src_id] = target_id
            logger.info(f"  [Map] Source ID {src_id} ('{src_name}') -> Target ID {target_id} ('{canonical}')")
        else:
            translation_map[src_id] = -1
            logger.warning(f"  [Ignore] Source ID {src_id} ('{src_name}') has no target class equivalent; will be skipped.")

    return translation_map


def sanitize_yolo_bbox(line: str) -> Optional[Tuple[int, float, float, float, float]]:
    """Validates and clamps YOLO normalized bounding box format: class_id x_c y_c w h."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cid = int(float(parts[0]))
        xc = max(0.0, min(1.0, float(parts[1])))
        yc = max(0.0, min(1.0, float(parts[2])))
        w = max(0.0, min(1.0, float(parts[3])))
        h = max(0.0, min(1.0, float(parts[4])))
        if w <= 0.001 or h <= 0.001:
            return None
        return cid, xc, yc, w, h
    except (ValueError, IndexError):
        return None


def merge_single_dataset(
    src_dir: str,
    prefix: str,
    dest_dir: str,
    class_map: Dict[int, int],
    stats_counter: Counter
):
    """
    Merges images and labels from a source dataset into the unified destination directory.
    Handles standard directory layouts (e.g., train/images, train/labels or images/train, labels/train).
    """
    for split in ["train", "valid", "val", "test"]:
        target_split = "val" if split in ["valid", "val"] else ("test" if split == "test" else "train")

        # Discover images in various folder conventions
        candidate_img_dirs = [
            os.path.join(src_dir, split, "images"),
            os.path.join(src_dir, "images", split),
            os.path.join(src_dir, split)
        ]

        img_dir = None
        for cd in candidate_img_dirs:
            if os.path.isdir(cd):
                img_dir = cd
                break

        if not img_dir:
            continue

        # Find corresponding labels directory
        candidate_lbl_dirs = [
            os.path.join(src_dir, split, "labels"),
            os.path.join(src_dir, "labels", split),
            img_dir.replace("images", "labels")
        ]
        lbl_dir = None
        for ld in candidate_lbl_dirs:
            if os.path.isdir(ld):
                lbl_dir = ld
                break

        dest_img_split_dir = os.path.join(dest_dir, "images", target_split)
        dest_lbl_split_dir = os.path.join(dest_dir, "labels", target_split)
        os.makedirs(dest_img_split_dir, exist_ok=True)
        os.makedirs(dest_lbl_split_dir, exist_ok=True)

        img_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            img_files.extend(glob.glob(os.path.join(img_dir, ext)))

        logger.info(f"Processing split '{split}' -> '{target_split}' for '{prefix}': found {len(img_files)} images.")

        for img_path in img_files:
            base_stem = os.path.splitext(os.path.basename(img_path))[0]
            ext = os.path.splitext(img_path)[1]
            unique_base = f"{prefix}_{base_stem}"
            dest_img_path = os.path.join(dest_img_split_dir, f"{unique_base}{ext}")

            # Check for matching label file
            src_lbl_path = None
            if lbl_dir:
                cand = os.path.join(lbl_dir, f"{base_stem}.txt")
                if os.path.isfile(cand):
                    src_lbl_path = cand

            valid_annotations = []
            if src_lbl_path and os.path.isfile(src_lbl_path):
                with open(src_lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parsed = sanitize_yolo_bbox(line)
                        if not parsed:
                            continue
                        src_cid, xc, yc, w, h = parsed
                        target_cid = class_map.get(src_cid, -1)
                        if target_cid >= 0:
                            valid_annotations.append(f"{target_cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                            stats_counter[f"{target_split}_{TARGET_CLASSES[target_cid]}"] += 1

            # Only copy image if it has valid annotations or background inclusion is allowed
            if valid_annotations:
                shutil.copy2(img_path, dest_img_path)
                dest_lbl_path = os.path.join(dest_lbl_split_dir, f"{unique_base}.txt")
                with open(dest_lbl_path, "w", encoding="utf-8") as f:
                    f.writelines(valid_annotations)


def generate_unified_dataset(
    dataset_configs: List[Dict],
    output_dir: str
):
    """
    Main orchestration function to download, map, merge datasets, and output final data.yaml.
    """
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    stats = Counter()

    for idx, cfg in enumerate(dataset_configs):
        name = cfg.get("name", f"ds_{idx}")
        logger.info(f"\n[{idx+1}/{len(dataset_configs)}] Processing source dataset: {name}")

        # Check if Roboflow download is required
        if cfg.get("type") == "roboflow":
            download_cache = os.path.join(output_dir, "_downloads", name)
            os.makedirs(download_cache, exist_ok=True)
            local_path = download_roboflow_dataset(
                api_key=cfg["api_key"],
                workspace=cfg["workspace"],
                project=cfg["project"],
                version=cfg["version"],
                output_dir=download_cache
            )
        else:
            local_path = cfg["path"]

        # Locate source data.yaml
        yaml_candidates = [
            os.path.join(local_path, "data.yaml"),
            os.path.join(local_path, "dataset.yaml")
        ]
        yaml_file = None
        for yc in yaml_candidates:
            if os.path.isfile(yc):
                yaml_file = yc
                break

        if not yaml_file:
            logger.error(f"Cannot find data.yaml in {local_path}. Skipping.")
            continue

        # Load ID translation
        class_map = load_dataset_class_map(yaml_file)

        # Merge
        merge_single_dataset(
            src_dir=local_path,
            prefix=name,
            dest_dir=output_dir,
            class_map=class_map,
            stats_counter=stats
        )

    # Generate train.txt and val.txt index files for Ultralytics
    train_imgs = glob.glob(os.path.join(output_dir, "images", "train", "*.*"))
    val_imgs = glob.glob(os.path.join(output_dir, "images", "val", "*.*"))

    train_txt_path = os.path.join(output_dir, "train.txt")
    val_txt_path = os.path.join(output_dir, "val.txt")

    with open(train_txt_path, "w", encoding="utf-8") as f:
        for p in train_imgs:
            f.write(os.path.abspath(p).replace("\\", "/") + "\n")

    with open(val_txt_path, "w", encoding="utf-8") as f:
        for p in val_imgs:
            f.write(os.path.abspath(p).replace("\\", "/") + "\n")

    # Write unified data.yaml
    unified_yaml = {
        "path": os.path.abspath(output_dir).replace("\\", "/"),
        "train": "train.txt",
        "val": "val.txt",
        "test": "val.txt",
        "nc": len(TARGET_CLASSES),
        "names": {i: name for i, name in enumerate(TARGET_CLASSES)}
    }

    final_yaml_path = os.path.join(output_dir, "data.yaml")
    with open(final_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(unified_yaml, f, sort_keys=False)

    logger.info("\n" + "=" * 65)
    logger.info("DATASET MERGE COMPLETE — SUMMARY STATISTICS")
    logger.info("=" * 65)
    logger.info(f"Destination: {output_dir}")
    logger.info(f"Generated YAML: {final_yaml_path}")
    logger.info(f"Train Images: {len(train_imgs)} | Val Images: {len(val_imgs)}")
    logger.info("\nClass Instance Breakdown:")
    for cls_name in TARGET_CLASSES:
        tr_cnt = stats.get(f"train_{cls_name}", 0)
        vl_cnt = stats.get(f"val_{cls_name}", 0)
        logger.info(f"  {cls_name:<16} => Train: {tr_cnt:>5} | Val: {vl_cnt:>4} | Total: {tr_cnt+vl_cnt:>5}")
    logger.info("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple Roboflow and local datasets for Construction PPE")
    parser.add_argument("--dest", type=str, default=r"D:\Construction Safety\dataset", help="Target unified dataset path")
    parser.add_argument("--roboflow_key", type=str, default="", help="Roboflow API key (optional)")
    args = parser.parse_args()

    # Define list of datasets to merge (customize with your Roboflow projects or existing local folder)
    sources = [
        # Source 1: The current existing construction safety dataset
        {
            "name": "local_site_dataset",
            "type": "local",
            "path": r"D:\Construction Safety\dataset"
        }
    ]

    # Example template for adding Roboflow online datasets:
    if args.roboflow_key:
        sources.append({
            "name": "rf_ppe_detection",
            "type": "roboflow",
            "api_key": args.roboflow_key,
            "workspace": "roboflow-universe-projects",
            "project": "construction-site-safety",
            "version": 1
        })

    generate_unified_dataset(sources, args.dest)
