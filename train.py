"""
Stage 2: YOLOv8 Training Script for Construction Safety Monitoring
Trains exclusively on the Construction-PPE dataset.
"""
import os
import sys
import shutil
import argparse
import torch
import yaml
from ultralytics import YOLO
import config

def train_model(
    epochs=15,
    imgsz=640,
    batch=16,
    lr0=0.01,
    workers=0,
    device=None,
    data_yaml=config.DATA_YAML_PATH,
    model_name="yolov8n.pt",
    project="runs/train",
    name="construction_ppe_exp"
):
    print("=" * 60)
    print("AI-BASED CONSTRUCTION SAFETY MONITORING - MODEL TRAINING")
    print("=" * 60)

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"[+] Active Computation Device: {device}")
    print(f"[+] Dataset Configuration: {data_yaml}")
    print(f"[+] Base Model: {model_name}")
    print(f"[+] Training Config: epochs={epochs}, imgsz={imgsz}, batch={batch}, lr0={lr0}")

    # Ensure data.yaml has absolute path for Ultralytics
    with open(data_yaml, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # Set root path in data.yaml to actual dataset folder
    data_cfg["path"] = os.path.abspath(config.DATASET_DIR)
    temp_yaml_path = os.path.join(config.BASE_DIR, "dataset_train.yaml")
    with open(temp_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_cfg, f)

    print(f"[+] Prepared training dataset YAML: {temp_yaml_path}")
    print(f"[+] Class names defined: {data_cfg.get('names')}")

    # Load YOLOv8 model
    model = YOLO(model_name)

    # Train
    results = model.train(
        data=temp_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        device=device,
        workers=workers,
        project=project,
        name=name,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True
    )

    # Check and copy best.pt to model/best.pt
    exp_dir = os.path.join(project, name)
    weights_best = os.path.join(exp_dir, "weights", "best.pt")
    weights_last = os.path.join(exp_dir, "weights", "last.pt")

    if os.path.exists(weights_best):
        target_best = config.MODEL_PATH
        shutil.copy(weights_best, target_best)
        print(f"\n[+] Successfully saved best model to: {target_best}")
    else:
        print("[-] Warning: best.pt was not found in experiment output directory.")

    print("\n" + "=" * 60)
    print(f"[+] Training completed. Experiment artifacts saved in: {exp_dir}")
    print("=" * 60)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Construction-PPE dataset")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda', '0')")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")

    args = parser.parse_args()
    train_model(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr,
        device=args.device,
        workers=args.workers
    )
