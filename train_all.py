#!/usr/bin/env python3
"""
Multi-Model Training Orchestrator
==================================
Trains all 4 comparison architectures (YOLOv8-n, YOLOv8-m, YOLOv8-x, YOLOv9-e)
on the Construction Safety dataset using identical hyperparameters and fixed seed.
Outputs are saved to model/<name>_ppe.pt for downstream comparison and evaluation.

Usage:
  python train_all.py --epochs 25 --batch 16
  python train_all.py --models v8n v8m --epochs 10
"""

import os
import sys
import argparse

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

import config
from train import train_model

TARGET_MODELS = [
    {
        "id": "v8n",
        "display_name": "YOLOv8-n",
        "base_model": "yolov8n.pt",
        "output_weights": os.path.join(config.MODEL_DIR, "yolov8n_ppe.pt"),
        "exp_name": "train_yolov8n_ppe"
    },
    {
        "id": "v8m",
        "display_name": "YOLOv8-m",
        "base_model": "yolov8m.pt",
        "output_weights": os.path.join(config.MODEL_DIR, "yolov8m_ppe.pt"),
        "exp_name": "train_yolov8m_ppe"
    },
    {
        "id": "v8x",
        "display_name": "YOLOv8-x",
        "base_model": "yolov8x.pt",
        "output_weights": os.path.join(config.MODEL_DIR, "yolov8x_ppe.pt"),
        "exp_name": "train_yolov8x_ppe"
    },
    {
        "id": "v9e",
        "display_name": "YOLOv9-e",
        "base_model": "yolov9e.pt",
        "output_weights": os.path.join(config.MODEL_DIR, "yolov9e_ppe.pt"),
        "exp_name": "train_yolov9e_ppe"
    },
]


def main():
    parser = argparse.ArgumentParser(description="Train all YOLO comparison models on Construction-PPE")
    parser.add_argument("--models", nargs="+", default=["v8n", "v8m", "v8x", "v9e"], help="Subset of models to train")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed (default: 42)")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda', '0')")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")

    args = parser.parse_args()

    selected_ids = [m.lower().replace("-", "") for m in args.models]
    queue = [m for m in TARGET_MODELS if m["id"] in selected_ids or m["display_name"].lower() in [x.lower() for x in args.models]]

    if not queue:
        print(f"[-] No matching models found in queue for: {args.models}")
        sys.exit(1)

    print("=" * 70)
    print(f"MULTI-MODEL FINE-TUNING ORCHESTRATION: {len(queue)} MODEL(S)")
    print(f"Hyperparameters: epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, seed={args.seed}")
    print("=" * 70)

    for i, target in enumerate(queue):
        print(f"\n[>>>] [{i+1}/{len(queue)}] Starting fine-tuning for {target['display_name']} ({target['base_model']})...")
        train_model(
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            lr0=args.lr,
            seed=args.seed,
            device=args.device,
            workers=args.workers,
            model_name=target["base_model"],
            output_path=target["output_weights"],
            name=target["exp_name"]
        )
        print(f"[✓] Completed training for {target['display_name']} -> {target['output_weights']}")

    print("\n" + "=" * 70)
    print("All requested comparison models successfully fine-tuned and saved!")
    print("=" * 70)


if __name__ == "__main__":
    main()
