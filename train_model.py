"""
SiteGuard AI — Production YOLOv8 Model Training & Evaluation Pipeline
Comprehensive training script with augmentation, class imbalance warnings,
epoch callbacks, post-training evaluation, confusion matrix generation,
and automatic production model deployment.
"""

import os
import sys
import json
import shutil
import datetime
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Training Hyperparameters & Configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "data": "data.yaml",
    "model": "yolov8m.pt",       # Medium architecture: optimal speed/accuracy trade-off
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,                 # Auto-adjusts down to 8 if GPU memory < 8GB or on CPU
    "patience": 20,              # Early stopping patience
    "device": "0" if torch.cuda.is_available() else "cpu",
    "project": "model/runs",
    "name": "siteguard_v1",
    "exist_ok": True,
    "pretrained": True,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "cos_lr": True,
    "label_smoothing": 0.1,
    "degrees": 10.0,             # Rotation augmentation
    "translate": 0.1,            # Translation
    "scale": 0.5,                # Scale jitter
    "shear": 2.0,                # Shear
    "flipud": 0.1,               # Vertical flip
    "fliplr": 0.5,               # Horizontal flip
    "mosaic": 1.0,               # Mosaic augmentation
    "mixup": 0.15,               # Mixup augmentation
    "copy_paste": 0.1,           # Copy-paste augmentation for rare PPE
    "conf": 0.25,
    "iou": 0.45,
    "save": True,
    "save_period": 10,
    "plots": True,
    "workers": 4 if torch.cuda.is_available() else 2,
}

TRAINING_LOG_PATH = Path("model/training_log.json")
EVAL_RESULTS_PATH = Path("model/evaluation_results.json")
MODEL_INFO_PATH = Path("model/model_info.json")
CONFUSION_MATRIX_PATH = Path("model/confusion_matrix.png")
PRODUCTION_BEST_PATH = Path("model/best.pt")

# ---------------------------------------------------------------------------
# Pre-Training Integrity & Class Imbalance Verification
# ---------------------------------------------------------------------------
def verify_pre_training(data_yaml_path: str):
    print("\n" + "=" * 70)
    print("   [1/5] PRE-TRAINING CLASS BALANCE & DATASET VERIFICATION")
    print("=" * 70)

    yaml_p = Path(data_yaml_path)
    if not yaml_p.exists():
        raise FileNotFoundError(f"Missing dataset configuration: {yaml_p}")

    report_p = Path("dataset_tools/inspection_report.txt")
    if report_p.exists():
        print(f"[*] Found verified inspection report: {report_p}")
        try:
            with open(report_p, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                if "Overall Dataset Imbalance Factor" in line or "Total Bounding Box Instances" in line or "Total Images Found" in line:
                    print(f"    {line.strip()}")
                if "[WARNING]" in line:
                    print(f"    {line.strip()}")
        except Exception as e:
            print(f"[!] Warning reading inspection report: {e}")
    else:
        print("[!] No inspection report found. Run: python dataset_tools/inspect_dataset.py")

    print("[*] Augmentations enabled to address rare PPE classes:")
    print(f"    - Mosaic: {CONFIG['mosaic']} | Mixup: {CONFIG['mixup']} | Copy-Paste: {CONFIG['copy_paste']}")
    print(f"    - Scale:  {CONFIG['scale']}  | Flips: H={CONFIG['fliplr']}, V={CONFIG['flipud']}")

# ---------------------------------------------------------------------------
# Live Training Callback
# ---------------------------------------------------------------------------
epoch_logs = []

def on_fit_epoch_end_callback(trainer):
    """Writes live epoch metrics to model/training_log.json on each completed epoch."""
    try:
        epoch = getattr(trainer, "epoch", 0) + 1
        total_epochs = getattr(trainer, "epochs", CONFIG["epochs"])
        
        # Safely extract metrics
        metrics = getattr(trainer, "metrics", {}) or {}
        fitness = getattr(trainer, "fitness", 0.0)
        
        # Losses
        loss_items = getattr(trainer, "loss_items", None)
        loss_val = float(loss_items[0].item()) if loss_items is not None and len(loss_items) > 0 else 0.0
        
        # Learning rates
        lr = 0.0
        optimizer = getattr(trainer, "optimizer", None)
        if optimizer and hasattr(optimizer, "param_groups") and optimizer.param_groups:
            lr = optimizer.param_groups[0].get("lr", 0.0)

        entry = {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "loss": round(float(loss_val), 5),
            "fitness": round(float(fitness), 5),
            "map50": round(float(metrics.get("metrics/mAP50(B)", 0.0)), 5),
            "map50_95": round(float(metrics.get("metrics/mAP50-95(B)", 0.0)), 5),
            "precision": round(float(metrics.get("metrics/precision(B)", 0.0)), 5),
            "recall": round(float(metrics.get("metrics/recall(B)", 0.0)), 5),
            "learning_rate": round(float(lr), 6),
            "timestamp": datetime.datetime.now().isoformat()
        }
        epoch_logs.append(entry)

        TRAINING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TRAINING_LOG_PATH, 'w', encoding='utf-8') as f:
            json.dump({
                "model_name": CONFIG["model"],
                "dataset": CONFIG["data"],
                "last_updated": datetime.datetime.now().isoformat(),
                "current_epoch": epoch,
                "total_epochs": total_epochs,
                "history": epoch_logs
            }, f, indent=2)

        print(f" [SiteGuard Epoch {epoch}/{total_epochs}] Loss: {entry['loss']} | mAP50: {entry['map50']} | mAP50-95: {entry['map50_95']}")
    except Exception as e:
        print(f"[!] Callback telemetry error: {e}")

# ---------------------------------------------------------------------------
# Post-Training Evaluation & Metrics Archival
# ---------------------------------------------------------------------------
def run_post_training_evaluation(model, best_weights_path: Path):
    print("\n" + "=" * 70)
    print("   [4/5] RUNNING VALIDATION & SAVING EVALUATION METRICS")
    print("=" * 70)

    try:
        # Load best weights
        val_model = YOLO(str(best_weights_path))
        print(f"[*] Validating weights: {best_weights_path}")
        metrics = val_model.val(data=CONFIG["data"], split="val", imgsz=CONFIG["imgsz"], device=CONFIG["device"], verbose=True)

        class_names = val_model.names if hasattr(val_model, "names") else {}
        
        # Per-class metrics
        class_eval = {}
        maps_per_class = getattr(metrics.box, "maps", [])
        p_per_class = getattr(metrics.box, "p", [])
        r_per_class = getattr(metrics.box, "r", [])
        f1_per_class = getattr(metrics.box, "f1", [])

        for idx, cname in class_names.items():
            class_eval[cname] = {
                "class_id": idx,
                "mAP50_95": round(float(maps_per_class[idx]), 4) if idx < len(maps_per_class) else 0.0,
                "precision": round(float(p_per_class[idx]), 4) if idx < len(p_per_class) else 0.0,
                "recall": round(float(r_per_class[idx]), 4) if idx < len(r_per_class) else 0.0,
                "f1": round(float(f1_per_class[idx]), 4) if idx < len(f1_per_class) else 0.0
            }

        eval_summary = {
            "evaluation_date": datetime.datetime.now().isoformat(),
            "model_weights": str(best_weights_path),
            "dataset_config": CONFIG["data"],
            "overall_metrics": {
                "mAP50": round(float(metrics.box.map50), 4),
                "mAP50_95": round(float(metrics.box.map), 4),
                "precision": round(float(metrics.box.mp), 4),
                "recall": round(float(metrics.box.mr), 4),
                "f1_mean": round(float(sum(f1_per_class) / max(1, len(f1_per_class))), 4) if len(f1_per_class) > 0 else 0.0
            },
            "per_class_metrics": class_eval
        }

        EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(EVAL_RESULTS_PATH, 'w', encoding='utf-8') as f:
            json.dump(eval_summary, f, indent=2)
        print(f"[✓] Saved comprehensive evaluation results to: {EVAL_RESULTS_PATH}")

        # Update model_info.json
        model_info = {
            "model_name": "SiteGuard AI YOLOv8 Construction Safety Monitor",
            "architecture": CONFIG["model"],
            "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_size_images": 8099,
            "classes_count": len(class_names),
            "classes": class_names,
            "best_mAP50": eval_summary["overall_metrics"]["mAP50"],
            "best_mAP50_95": eval_summary["overall_metrics"]["mAP50_95"],
            "mean_precision": eval_summary["overall_metrics"]["precision"],
            "mean_recall": eval_summary["overall_metrics"]["recall"],
            "input_resolution": CONFIG["imgsz"],
            "device": CONFIG["device"]
        }
        with open(MODEL_INFO_PATH, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2)
        print(f"[✓] Updated model metadata: {MODEL_INFO_PATH}")

        # Copy confusion matrix if generated by YOLO
        run_dir = Path(CONFIG["project"]) / CONFIG["name"]
        for cm_candidate in ["confusion_matrix.png", "confusion_matrix_normalized.png"]:
            cand_p = run_dir / cm_candidate
            if cand_p.exists():
                shutil.copy2(cand_p, CONFUSION_MATRIX_PATH)
                print(f"[✓] Copied confusion matrix: {CONFUSION_MATRIX_PATH}")
                break

    except Exception as e:
        print(f"[!] Post-training evaluation warning: {e}")

# ---------------------------------------------------------------------------
# Production Deployment
# ---------------------------------------------------------------------------
def deploy_best_weights(trained_best_path: Path):
    print("\n" + "=" * 70)
    print("   [5/5] DEPLOYING BEST WEIGHTS TO PRODUCTION (model/best.pt)")
    print("=" * 70)

    if not trained_best_path.exists():
        print(f"[!] Error: Trained weights not found at: {trained_best_path}")
        return

    PRODUCTION_BEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PRODUCTION_BEST_PATH.exists():
        archive_dir = Path("model/archive")
        archive_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_p = archive_dir / f"best_backup_{ts}.pt"
        shutil.copy2(PRODUCTION_BEST_PATH, backup_p)
        print(f"[*] Archived previous production model to: {backup_p}")

    shutil.copy2(trained_best_path, PRODUCTION_BEST_PATH)
    file_size_mb = os.path.getsize(PRODUCTION_BEST_PATH) / (1024 ** 2)
    print(f"[✓] Production model successfully deployed: {PRODUCTION_BEST_PATH} ({file_size_mb:.2f} MB)")

# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SiteGuard AI YOLOv8 Training & Deployment")
    parser.add_argument("--data", type=str, default=CONFIG["data"], help="Path to data.yaml")
    parser.add_argument("--model", type=str, default=CONFIG["model"], help="YOLOv8 base model (e.g. yolov8m.pt, yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"], help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=CONFIG["batch"], help="Batch size")
    parser.add_argument("--imgsz", type=int, default=CONFIG["imgsz"], help="Image resolution")
    parser.add_argument("--device", type=str, default="", help="Device: '0', 'cpu', etc.")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers (default auto)")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only on model/best.pt")
    args = parser.parse_args()

    # Update config from CLI
    CONFIG["data"] = args.data
    CONFIG["model"] = args.model
    CONFIG["epochs"] = args.epochs
    CONFIG["batch"] = args.batch
    CONFIG["imgsz"] = args.imgsz
    if args.device:
        CONFIG["device"] = args.device

    # Handle device & batch sizing
    if CONFIG["device"] == "cpu" or not torch.cuda.is_available():
        CONFIG["device"] = "cpu"
        if CONFIG["batch"] > 8:
            print(f"[*] CPU detected: Adjusting batch size from {CONFIG['batch']} to 8 for stability.")
            CONFIG["batch"] = 8
        CONFIG["workers"] = 2
    else:
        # Check GPU VRAM
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[*] GPU Detected: {torch.cuda.get_device_name(0)} ({vram_gb:.2f} GB VRAM)")
        if vram_gb < 8.0 and CONFIG["batch"] > 8:
            print(f"[*] VRAM < 8GB ({vram_gb:.2f}GB): Adjusting batch size to 8.")
            CONFIG["batch"] = 8

    if args.workers > 0:
        CONFIG["workers"] = args.workers

    # If eval only requested
    if args.eval_only:
        print("[*] Evaluation-only mode requested.")
        if PRODUCTION_BEST_PATH.exists():
            val_m = YOLO(str(PRODUCTION_BEST_PATH))
            run_post_training_evaluation(val_m, PRODUCTION_BEST_PATH)
        else:
            print(f"[!] Model {PRODUCTION_BEST_PATH} does not exist.")
        return

    # 1. Pre-training verification
    verify_pre_training(CONFIG["data"])

    # 2. Model Initialization
    print("\n" + "=" * 70)
    print(f"   [2/5] INITIALIZING ARCHITECTURE: {CONFIG['model']}")
    print("=" * 70)
    model = YOLO(CONFIG["model"])

    # Attach live epoch callback
    try:
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end_callback)
        print("[✓] Live epoch telemetry callback registered.")
    except Exception as e:
        print(f"[!] Warning attaching callback: {e}")

    # 3. Training Execution
    print("\n" + "=" * 70)
    print("   [3/5] LAUNCHING YOLOV8 TRAINING ENGINE")
    print("=" * 70)
    print(f"[*] Configuration: Data={CONFIG['data']} | Epochs={CONFIG['epochs']} | Batch={CONFIG['batch']} | Device={CONFIG['device']}")
    
    start_t = datetime.datetime.now()
    results = model.train(
        data=CONFIG["data"],
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["imgsz"],
        batch=CONFIG["batch"],
        patience=CONFIG["patience"],
        device=CONFIG["device"],
        project=CONFIG["project"],
        name=CONFIG["name"],
        exist_ok=CONFIG["exist_ok"],
        pretrained=CONFIG["pretrained"],
        optimizer=CONFIG["optimizer"],
        lr0=CONFIG["lr0"],
        lrf=CONFIG["lrf"],
        momentum=CONFIG["momentum"],
        weight_decay=CONFIG["weight_decay"],
        warmup_epochs=CONFIG["warmup_epochs"],
        cos_lr=CONFIG["cos_lr"],
        label_smoothing=CONFIG["label_smoothing"],
        degrees=CONFIG["degrees"],
        translate=CONFIG["translate"],
        scale=CONFIG["scale"],
        shear=CONFIG["shear"],
        flipud=CONFIG["flipud"],
        fliplr=CONFIG["fliplr"],
        mosaic=CONFIG["mosaic"],
        mixup=CONFIG["mixup"],
        copy_paste=CONFIG["copy_paste"],
        conf=CONFIG["conf"],
        iou=CONFIG["iou"],
        save=CONFIG["save"],
        save_period=CONFIG["save_period"],
        plots=CONFIG["plots"],
        workers=CONFIG["workers"],
        verbose=True
    )
    elapsed = datetime.datetime.now() - start_t
    print(f"\n[✓] Training session completed in {elapsed}!")

    # 4. Evaluation
    trained_best = Path(CONFIG["project"]) / CONFIG["name"] / "weights" / "best.pt"
    if trained_best.exists():
        run_post_training_evaluation(model, trained_best)
        # 5. Deploy
        deploy_best_weights(trained_best)
    else:
        print(f"[!] Warning: Best weights not found at {trained_best}")

if __name__ == "__main__":
    main()
