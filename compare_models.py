#!/usr/bin/env python3
"""
CLI Tool: Model Comparison Engine (SH17 Paper Fig. 5 Reproducer)
================================================================
Compares multiple YOLO architectures (YOLOv8-n, YOLOv8-m, YOLOv8-x, YOLOv9-e)
side-by-side on the same construction site images.

Example Usage:
  python compare_models.py --images construction/images/val --models v8n v8m v8x v9e
  python compare_models.py --images dataset/images/test --models v8n v8m --conf 0.30
"""

import os
import sys
import argparse
import glob
from typing import List

# Ensure repository root is on path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

import config
from cv.model_compare import (
    MODEL_REGISTRY,
    MODEL_ALIASES,
    MODEL_METADATA,
    compare_models,
    build_comparison_grid,
    resolve_model_name
)


def collect_image_paths(image_arg: List[str], max_images: int = 7) -> List[str]:
    """Collects up to max_images image paths from directory or explicit file list."""
    collected = []
    for arg in image_arg:
        if os.path.isdir(arg):
            valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
            for ext in valid_exts:
                collected.extend(glob.glob(os.path.join(arg, ext)))
                collected.extend(glob.glob(os.path.join(arg, ext.upper())))
        elif os.path.isfile(arg):
            collected.append(arg)
        else:
            # Check if glob pattern
            matched = glob.glob(arg)
            if matched:
                collected.extend(matched)

    # Deduplicate preserving order
    seen = set()
    unique = []
    for p in collected:
        abs_p = os.path.abspath(p)
        if abs_p not in seen and os.path.exists(abs_p):
            seen.add(abs_p)
            unique.append(abs_p)

    return unique[:max_images]


def print_benchmark_table(stats: dict, skipped_models: list):
    """Prints a clean ASCII/Markdown benchmark table to stdout."""
    print("\n" + "=" * 78)
    print("           SH17 EMPIRICAL MODEL COMPARISON BENCHMARK (FIG. 5)")
    print("=" * 78)
    header = f"{'MODEL':<12} | {'PARAMS (M)':<10} | {'AVG LATENCY (ms)':<17} | {'AVG DETECTIONS':<14} | {'STATUS'}"
    print(header)
    print("-" * 78)

    for m_name, data in stats.items():
        params = f"{data.get('params_m', 0.0):.1f}M"
        lat = f"{data.get('avg_inference_ms', 0.0):.2f} ms"
        dets = f"{data.get('avg_detections', 0.0):.2f}"
        status = "Base (Fallback)" if data.get("is_fallback") else "Fine-Tuned"
        print(f"{m_name:<12} | {params:<10} | {lat:<17} | {dets:<14} | {status}")

    print("=" * 78)

    if skipped_models:
        print("\n[!] Skipped Models:")
        for sm in skipped_models:
            print(f"    - {sm['model']}: {sm['reason']}")

    any_fallback = any(d.get("is_fallback") for d in stats.values())
    if any_fallback:
        print("\n[*] Notice: One or more models used base pretrained weights because fine-tuned weights")
        print("    were not yet generated. To train fine-tuned weights for all 4 models, run:")
        print("        python train_all.py --epochs 25 --batch 16\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-YOLO model comparison and generate SH17 Fig. 5 side-by-side grid."
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Input image directory, glob pattern, or list of file paths (1 to 8 images)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["v8n", "v8m", "v8x", "v9e"],
        help="Models to compare. Available: v8n, v8m, v8x, v9e (or YOLOv8-n, YOLOv8-m, etc.)"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution (default: 640)")
    parser.add_argument("--max-images", type=int, default=7, help="Maximum number of images to compare (default: 7)")
    parser.add_argument("--tile-width", type=int, default=480, help="Tile width in output grid (default: 480)")
    parser.add_argument("--out", type=str, default=os.path.join(config.OUTPUT_FOLDER, "comparison"), help="Output directory")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: skip models whose custom fine-tuned weights are missing instead of falling back to base weights"
    )

    args = parser.parse_args()

    image_paths = collect_image_paths(args.images, max_images=args.max_images)
    if not image_paths:
        print(f"[-] Error: No valid images found matching: {args.images}")
        sys.exit(1)

    print(f"[+] Found {len(image_paths)} image(s) for comparison.")
    print(f"[+] Selected models: {args.models}")
    print(f"[+] Running inference on {config.DEVICE.upper()} (conf={args.conf}, iou={args.iou}, imgsz={args.imgsz})...")

    # Run multi-model comparison
    results = compare_models(
        image_paths=image_paths,
        model_names=args.models,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        allow_fallback=not args.strict,
        style="plain"
    )

    if "error" in results:
        print(f"[-] Error during comparison: {results['error']}")
        sys.exit(1)

    # Render comparison grid
    grid_path, tile_paths = build_comparison_grid(
        comparison_results=results,
        tile_width=args.tile_width,
        output_dir=args.out
    )

    # Print benchmark table
    print_benchmark_table(results.get("model_stats", {}), results.get("skipped_models", []))
    print(f"[+] Comparison grid successfully generated:")
    print(f"    -> {grid_path}\n")


if __name__ == "__main__":
    main()
