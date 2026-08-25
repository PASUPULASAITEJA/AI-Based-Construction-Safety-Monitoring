"""
Stage 3: Model Evaluation Script for Construction Safety Monitoring
Evaluates trained YOLOv8 model on Construction-PPE test split.
Reports Precision, Recall, mAP@50, mAP@50-95, and F1 score per class.
"""
import os
import sys
import argparse
import json
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import config

def evaluate_model(
    model_path=config.MODEL_PATH,
    data_yaml=config.DATA_YAML_PATH,
    split="test",
    imgsz=640,
    device=None,
    save_dir="outputs/evaluation"
):
    print("=" * 60)
    print("AI-BASED CONSTRUCTION SAFETY MONITORING - MODEL EVALUATION")
    print("=" * 60)

    if device is None:
        device = "cuda:0" if config.DEVICE == "cuda" else "cpu"

    if not os.path.exists(model_path):
        print(f"[-] Model weights not found at: {model_path}")
        print(f"[!] Falling back to standard base YOLOv8 model for initial pipeline testing: {config.FALLBACK_MODEL}")
        model_path = config.FALLBACK_MODEL

    # Prepare temp YAML with absolute paths
    with open(data_yaml, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    data_cfg["path"] = os.path.abspath(config.DATASET_DIR)

    temp_yaml = os.path.join(config.BASE_DIR, "dataset_eval.yaml")
    with open(temp_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_cfg, f)

    os.makedirs(save_dir, exist_ok=True)

    print(f"[+] Evaluating Model: {model_path}")
    print(f"[+] Dataset Split: {split.upper()} from {data_yaml}")
    print(f"[+] Device: {device}")

    model = YOLO(model_path)
    val_results = model.val(
        data=temp_yaml,
        split=split,
        imgsz=imgsz,
        device=device,
        project=save_dir,
        name="test_results",
        exist_ok=True,
        plots=True,
        verbose=True
    )

    # Extract metrics
    metrics = {
        "precision": float(val_results.results_dict.get("metrics/precision(B)", 0.0)),
        "recall": float(val_results.results_dict.get("metrics/recall(B)", 0.0)),
        "mAP50": float(val_results.results_dict.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(val_results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
    }

    # Calculate F1
    p = metrics["precision"]
    r = metrics["recall"]
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    metrics["f1_score"] = float(f1)

    print("\n" + "=" * 60)
    print("CONSTRUCTION-PPE TEST EVALUATION METRICS REPORT")
    print("=" * 60)
    print(f"  • Precision:  {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  • Recall:     {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  • F1-Score:   {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f"  • mAP@50:     {metrics['mAP50']:.4f} ({metrics['mAP50']*100:.2f}%)")
    print(f"  • mAP@50:95:  {metrics['mAP50_95']:.4f} ({metrics['mAP50_95']*100:.2f}%)")
    print("=" * 60)

    # Save metrics JSON
    metrics_file = os.path.join(save_dir, "evaluation_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as jf:
        json.dump(metrics, jf, indent=4)
    print(f"[+] Saved evaluation metrics to: {metrics_file}")

    # Run inference on sample test images
    test_unseen_samples(model, save_dir)
    return metrics

def test_unseen_samples(model, save_dir, sample_count=5):
    test_img_dir = os.path.join(config.DATASET_DIR, "images", "test")
    if not os.path.exists(test_img_dir):
        return

    sample_imgs = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:sample_count]
    sample_out_dir = os.path.join(save_dir, "sample_predictions")
    os.makedirs(sample_out_dir, exist_ok=True)

    for s_img in sample_imgs:
        img_path = os.path.join(test_img_dir, s_img)
        res = model.predict(img_path, conf=0.35, imgsz=640, verbose=False)
        if res and len(res) > 0:
            res_plotted = res[0].plot()
            out_path = os.path.join(sample_out_dir, f"pred_{s_img}")
            cv2.imwrite(out_path, res_plotted)

    print(f"[+] Saved {len(sample_imgs)} sample test predictions to: {sample_out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO on Construction-PPE")
    parser.add_argument("--model", type=str, default=config.MODEL_PATH, help="Path to model weights")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate ('val' or 'test')")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    args = parser.parse_args()

    evaluate_model(model_path=args.model, split=args.split, imgsz=args.imgsz)
