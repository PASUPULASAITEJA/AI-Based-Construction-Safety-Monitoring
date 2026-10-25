"""
SiteGuard AI — Automated YOLOv8 17-Class Model Training & Deployment Pipeline
Trains YOLOv8 on the construction safety PPE dataset (D:/Safe/dataset).
"""
import os
import sys
import shutil
import argparse
import datetime
from pathlib import Path
import torch
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on SiteGuard AI PPE Dataset")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml configuration")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained weights or model architecture")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (reduce if VRAM/RAM constrained)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size resolution (pixels)")
    parser.add_argument("--device", type=str, default="", help="Device: '0', 'cpu', etc. (default auto-detect)")
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader worker processes")
    parser.add_argument("--project", type=str, default="runs/train", help="Save results to project/name")
    parser.add_argument("--name", type=str, default="siteguard_ppe_17class", help="Experiment name")
    parser.add_argument("--deploy", action="store_true", default=True, help="Deploy best.pt to model/best.pt when complete")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 70)
    print("   SITEGUARD AI — YOLOV8 MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    device = args.device
    if not device:
        device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Target Compute Device: {device.upper()}")
    if device != "cpu" and torch.cuda.is_available():
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"[*] VRAM Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("[*] Running on Multi-Threaded CPU Engine")
        # Ensure batch size is sensible for CPU
        if args.batch > 16 and device == "cpu":
            print(f"[*] Adjusting batch size from {args.batch} to 8 for CPU memory optimization.")
            args.batch = 8

    print(f"[*] Config: {args.data}")
    print(f"[*] Base Weights: {args.model}")
    print(f"[*] Epochs: {args.epochs} | Batch: {args.batch} | Image Size: {args.imgsz}")
    
    # 1. Load Pretrained Model
    print("\n[1/3] Loading YOLOv8 Architecture...")
    model = YOLO(args.model)
    
    # 2. Train Model
    print("\n[2/3] Initiating Training Cycle...")
    start_time = datetime.datetime.now()
    
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        save=True,
        val=True,
        patience=15,
        verbose=True
    )
    
    elapsed = datetime.datetime.now() - start_time
    print(f"\n✓ Training Completed in {elapsed}")
    
    # 3. Deployment of Best Weights
    if args.deploy:
        print("\n[3/3] Deploying Best Weights to Production...")
        trained_best = Path(args.project) / args.name / "weights" / "best.pt"
        target_dir = Path("model")
        target_best = target_dir / "best.pt"
        
        if trained_best.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            # Backup previous model if exists
            if target_best.exists():
                archive_dir = target_dir / "archive"
                archive_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = archive_dir / f"best_backup_{timestamp}.pt"
                shutil.copy2(target_best, backup_path)
                print(f"[*] Previous production model archived to: {backup_path}")
                
            shutil.copy2(trained_best, target_best)
            print(f"✓ Deployed new best weights to: {target_best} ({os.path.getsize(target_best) / 1024**2:.2f} MB)")
        else:
            print(f"[!] Warning: Expected best weights file not found at {trained_best}")

    print("\n" + "=" * 70)
    print("   MODEL TRAINING & VERIFICATION PIPELINE FINISHED")
    print("=" * 70)

if __name__ == "__main__":
    main()
