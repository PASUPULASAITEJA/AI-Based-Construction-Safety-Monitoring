"""
Roboflow Multi-Dataset Merger for 17-Class PPE Construction System
Downloads disparate Roboflow datasets and unifies them into a single YOLOv8 dataset.
"""

import os
import shutil
import glob
from pathlib import Path
from roboflow import Roboflow

# Master 17-Class Mapping Dictionary
TARGET_CLASSES = [
    "helmet", "no-helmet",
    "goggles", "no-goggles",
    "earmuff", "no-earmuff",
    "vest", "no-vest",
    "safety-pants", "no-safety-pants",
    "gloves", "no-gloves",
    "boots", "no-boots",
    "harness", "no-harness",
    "person"
]
CLASS_TO_ID = {name: i for i, name in enumerate(TARGET_CLASSES)}

# Source dataset mapping aliases
SOURCE_MAPPINGS = {
    # Dataset 1: Standard PPE (Hardhat / Vest)
    "hard-hat-workers": {
        "helmet": "helmet",
        "head": "no-helmet",
        "no-helmet": "no-helmet",
        "vest": "vest",
        "no-vest": "no-vest",
        "person": "person",
        "worker": "person"
    },
    # Dataset 2: Eye & Face Protection
    "safety-glasses-detection": {
        "glasses": "goggles",
        "goggles": "goggles",
        "safety_glasses": "goggles",
        "no_glasses": "no-goggles",
        "face": "no-goggles",
        "earmuffs": "earmuff",
        "person": "person"
    },
    # Dataset 3: Fall Protection & Harnesses
    "construction-safety-fall-protection": {
        "harness": "harness",
        "safety_harness": "harness",
        "no_harness": "no-harness",
        "protective_pants": "safety-pants",
        "boots": "boots",
        "shoes": "boots",
        "no_boots": "no-boots",
        "gloves": "gloves",
        "no_gloves": "no-gloves",
        "worker": "person"
    }
}

OUTPUT_DATASET_DIR = Path("./datasets/construction_ppe_17class")

def setup_directory_structure():
    for split in ["train", "val", "test"]:
        (OUTPUT_DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def download_roboflow_dataset(api_key: str, workspace: str, project: str, version: int) -> Path:
    print(f"[*] Downloading {workspace}/{project} v{version}...")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("yolov8", location=f"./downloads/{project}")
    return Path(dataset.location)

def remap_and_merge_labels(source_dir: Path, dataset_key: str):
    mapping = SOURCE_MAPPINGS.get(dataset_key, {})
    
    # Read original data.yaml from downloaded dataset to get source class names
    yaml_path = source_dir / "data.yaml"
    src_class_names = []
    if yaml_path.exists():
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            src_class_names = data.get("names", [])
            if isinstance(src_class_names, dict):
                src_class_names = [src_class_names[k] for k in sorted(src_class_names.keys())]

    for split in ["train", "valid", "test"]:
        target_split = "val" if split == "valid" else split
        split_dir = source_dir / split
        if not split_dir.exists():
            continue

        label_files = list((split_dir / "labels").glob("*.txt"))
        print(f"[*] Processing {len(label_files)} annotations from {dataset_key} [{split}]...")

        for lbl_path in label_files:
            img_stem = lbl_path.stem
            # Find corresponding image
            img_candidates = list((split_dir / "images").glob(f"{img_stem}.*"))
            if not img_candidates:
                continue
            src_img_path = img_candidates[0]

            new_annotations = []
            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                src_cid = int(parts[0])
                if src_cid >= len(src_class_names):
                    continue
                src_name = src_class_names[src_cid].lower().replace(" ", "_")
                
                # Check mapping
                target_name = mapping.get(src_name)
                if target_name and target_name in CLASS_TO_ID:
                    target_cid = CLASS_TO_ID[target_name]
                    # Clamp bounding box coords to [0.0, 1.0]
                    coords = [max(0.0, min(1.0, float(x))) for x in parts[1:5]]
                    new_annotations.append(f"{target_cid} " + " ".join(f"{c:.6f}" for c in coords))

            if new_annotations:
                # Target unique filename with prefix
                unique_stem = f"{dataset_key}_{img_stem}"
                out_img = OUTPUT_DATASET_DIR / "images" / target_split / f"{unique_stem}{src_img_path.suffix}"
                out_lbl = OUTPUT_DATASET_DIR / "labels" / target_split / f"{unique_stem}.txt"

                shutil.copy2(src_img_path, out_img)
                with open(out_lbl, "w", encoding="utf-8") as f:
                    f.write("\n".join(new_annotations) + "\n")

def create_master_yaml():
    yaml_content = f"""path: {OUTPUT_DATASET_DIR.resolve()}
train: images/train
val: images/val
test: images/test

nc: 17
names: {TARGET_CLASSES}
"""
    with open(OUTPUT_DATASET_DIR / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"[✓] Created master data.yaml at {OUTPUT_DATASET_DIR / 'data.yaml'}")

if __name__ == "__main__":
    API_KEY = os.environ.get("ROBOFLOW_API_KEY", "YOUR_API_KEY")
    setup_directory_structure()

    # Datasets configuration (customize workspace, project, version)
    datasets_to_pull = [
        ("hard-hat-workers", "roboflow-universe-projects", "hard-hat-workers", 3),
        ("safety-glasses-detection", "eyewear-safety", "safety-glasses-detection", 1),
        ("construction-safety-fall-protection", "site-ppe", "fall-protection-ppe", 2)
    ]

    for key, ws, proj, ver in datasets_to_pull:
        try:
            path = download_roboflow_dataset(API_KEY, ws, proj, ver)
            remap_and_merge_labels(path, key)
        except Exception as e:
            print(f"[!] Warning: Could not download {proj}: {e}")

    create_master_yaml()
