"""
CV Module: Model Comparison Engine
===================================
Produces scientific multi-model comparisons replicating Fig. 5 of the SH17 paper (arXiv 2407.04590).
Runs the same input images through multiple YOLO models side-by-side in columns:
(YOLOv8-n | YOLOv8-m | YOLOv8-x | YOLOv9-e), with column headers, row labels (1), (2), (3)...,
clean per-class plain annotations (Fig. 1/6 paper style), and per-model latency/detection benchmarks.
"""

import os
import time
import datetime
import yaml
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

import config
from cv.annotator import plain_annotate, get_class_color, CLASS_COLORS

# Default model registry mapping display name -> fine-tuned weights path
MODEL_REGISTRY: Dict[str, str] = {
    "YOLOv8-n": os.path.join(config.MODEL_DIR, "yolov8n_ppe.pt"),
    "YOLOv8-m": os.path.join(config.MODEL_DIR, "yolov8m_ppe.pt"),
    "YOLOv8-x": os.path.join(config.MODEL_DIR, "yolov8x_ppe.pt"),
    "YOLOv9-e": os.path.join(config.MODEL_DIR, "yolov9e_ppe.pt"),
}

# Base pretrained model fallback mapping
BASE_MODEL_WEIGHTS: Dict[str, str] = {
    "YOLOv8-n": "yolov8n.pt",
    "YOLOv8-m": "yolov8m.pt",
    "YOLOv8-x": "yolov8x.pt",
    "YOLOv9-e": "yolov9e.pt",
}

# Model architectural metadata (parameters in Millions)
MODEL_METADATA: Dict[str, Dict[str, Any]] = {
    "YOLOv8-n": {
        "id": "v8n",
        "name": "YOLOv8-n",
        "params_m": 3.2,
        "arch": "YOLOv8n",
        "description": "YOLOv8 Nano (3.2M params) - Edge & mobile real-time deployment"
    },
    "YOLOv8-m": {
        "id": "v8m",
        "name": "YOLOv8-m",
        "params_m": 25.9,
        "arch": "YOLOv8m",
        "description": "YOLOv8 Medium (25.9M params) - Balanced operational speed & precision"
    },
    "YOLOv8-x": {
        "id": "v8x",
        "name": "YOLOv8-x",
        "params_m": 68.2,
        "arch": "YOLOv8x",
        "description": "YOLOv8 XLarge (68.2M params) - High capacity feature extraction"
    },
    "YOLOv9-e": {
        "id": "v9e",
        "name": "YOLOv9-e",
        "params_m": 57.3,
        "arch": "YOLOv9e",
        "description": "YOLOv9 Extended (57.3M params) - Programmable Gradient Information (PGI)"
    },
}

# Normalized model alias lookup
MODEL_ALIASES: Dict[str, str] = {
    "v8n": "YOLOv8-n",
    "yolov8n": "YOLOv8-n",
    "yolov8-n": "YOLOv8-n",
    "v8m": "YOLOv8-m",
    "yolov8m": "YOLOv8-m",
    "yolov8-m": "YOLOv8-m",
    "v8x": "YOLOv8-x",
    "yolov8x": "YOLOv8-x",
    "yolov8-x": "YOLOv8-x",
    "v9e": "YOLOv9-e",
    "yolov9e": "YOLOv9-e",
    "yolov9-e": "YOLOv9-e",
}

# In-memory cached models and class mappings
_MODEL_CACHE: Dict[str, YOLO] = {}
_CLASS_NAMES_CACHE: Optional[Dict[int, str]] = None


def resolve_model_name(name: str) -> str:
    """Normalizes alias (e.g. 'v8n', 'yolov8n') to canonical display name (e.g. 'YOLOv8-n')."""
    clean = str(name).strip().lower()
    return MODEL_ALIASES.get(clean, name)


def load_dataset_classes(data_yaml_path: str = config.DATA_YAML_PATH) -> Dict[int, str]:
    """Loads standardized ground truth class taxonomy from dataset YAML."""
    global _CLASS_NAMES_CACHE
    if _CLASS_NAMES_CACHE is not None:
        return _CLASS_NAMES_CACHE

    classes = {}
    if os.path.exists(data_yaml_path):
        try:
            with open(data_yaml_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                names = cfg.get("names", {})
                if isinstance(names, list):
                    classes = {i: name for i, name in enumerate(names)}
                elif isinstance(names, dict):
                    classes = {int(k): v for k, v in names.items()}
        except Exception as e:
            print(f"[-] Warning reading {data_yaml_path}: {e}")

    if not classes:
        # Fallback to standard 17 classes
        classes = {i: name for i, name in enumerate(config.CLASSES_17)}

    _CLASS_NAMES_CACHE = classes
    return classes


def get_cached_model(
    model_name: str,
    allow_fallback: bool = True
) -> Tuple[Optional[YOLO], bool, str]:
    """
    Loads and caches a YOLO model instance.
    Returns: (model_instance, is_fallback, status_message)
    """
    canonical = resolve_model_name(model_name)
    if canonical in _MODEL_CACHE:
        return _MODEL_CACHE[canonical], False, "Cached"

    weights_path = MODEL_REGISTRY.get(canonical, canonical)
    is_fallback = False

    # Check if target weights file exists
    if not os.path.exists(weights_path):
        # Check if default best.pt exists and can be used for the requested model
        if canonical == "YOLOv8-m" and os.path.exists(config.MODEL_PATH):
            weights_path = config.MODEL_PATH
            print(f"[ModelCompare] Using default {config.MODEL_PATH} for {canonical}")
        elif allow_fallback:
            fallback_target = BASE_MODEL_WEIGHTS.get(canonical, "yolov8n.pt")
            print(f"[!] Warning: Fine-tuned weights '{weights_path}' for {canonical} not found.")
            print(f"    Falling back to base pretrained model '{fallback_target}' for comparison.")
            weights_path = fallback_target
            is_fallback = True
        else:
            msg = f"Model weights not found for '{canonical}' at {weights_path}. Model skipped."
            print(f"[-] Warning: {msg}")
            return None, False, msg

    try:
        print(f"[ModelCompare] Loading {canonical} from {weights_path} on {config.DEVICE}...")
        model = YOLO(weights_path)
        _MODEL_CACHE[canonical] = model
        return model, is_fallback, "Loaded"
    except Exception as e:
        msg = f"Failed to load model {canonical} ({weights_path}): {e}"
        print(f"[-] Error: {msg}")
        return None, False, msg


def run_model_inference(
    model: YOLO,
    image: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Runs inference on an image and returns standardized detections and latency in milliseconds.
    """
    dataset_classes = load_dataset_classes()
    t0 = time.perf_counter()

    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=config.DEVICE,
        verbose=False
    )
    t1 = time.perf_counter()
    inference_ms = (t1 - t0) * 1000.0

    detections = []
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())

            # Resolve class name: model.names or dataset_classes
            if hasattr(model, "names") and class_id in model.names:
                class_name = model.names[class_id]
            else:
                class_name = dataset_classes.get(class_id, f"class_{class_id}")

            norm_cat = str(class_name).strip().lower().replace("_", "-")
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": round(confidence, 4),
                "class_id": class_id,
                "class_name": class_name,
                "category": norm_cat
            })

    return detections, inference_ms


def compare_models(
    image_paths: List[str],
    model_names: Optional[List[str]] = None,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    allow_fallback: bool = True,
    style: str = "plain"
) -> Dict[str, Any]:
    """
    Runs multi-model comparison across input images.
    Returns annotated frames, detection counts, inference latency, and aggregated benchmarks.
    """
    if model_names is None or len(model_names) == 0:
        model_names = list(MODEL_REGISTRY.keys())

    # Resolve and load requested models
    active_models: Dict[str, YOLO] = {}
    fallback_flags: Dict[str, bool] = {}
    skipped_models: List[Dict[str, str]] = []

    for name in model_names:
        canonical = resolve_model_name(name)
        if canonical in active_models:
            continue
        model_obj, is_fb, msg = get_cached_model(canonical, allow_fallback=allow_fallback)
        if model_obj is not None:
            active_models[canonical] = model_obj
            fallback_flags[canonical] = is_fb
        else:
            skipped_models.append({"model": canonical, "reason": msg})

    loaded_names = list(active_models.keys())
    if not loaded_names:
        return {
            "error": "No valid models could be loaded for comparison.",
            "skipped_models": skipped_models,
            "results_by_image": [],
            "model_stats": {}
        }

    # Initialize stats accumulator
    model_latencies: Dict[str, List[float]] = {m: [] for m in loaded_names}
    model_det_counts: Dict[str, List[int]] = {m: [] for m in loaded_names}
    results_by_image: List[Dict[str, Any]] = []

    for img_idx, img_path in enumerate(image_paths):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[-] Warning: Unable to load image: {img_path}")
            continue

        img_name = os.path.basename(img_path)
        model_outputs: Dict[str, Any] = {}

        for m_name in loaded_names:
            model = active_models[m_name]
            dets, latency_ms = run_model_inference(
                model=model,
                image=frame,
                conf=conf,
                iou=iou,
                imgsz=imgsz
            )

            # Plain paper-style annotation
            annotated = plain_annotate(frame, dets, draw_conf=True)

            model_outputs[m_name] = {
                "annotated_frame": annotated,
                "detections": dets,
                "detection_count": len(dets),
                "inference_ms": round(latency_ms, 2),
                "is_fallback": fallback_flags[m_name]
            }

            model_latencies[m_name].append(latency_ms)
            model_det_counts[m_name].append(len(dets))

        results_by_image.append({
            "image_index": img_idx + 1,
            "image_path": img_path,
            "image_name": img_name,
            "model_outputs": model_outputs
        })

    # Compute overall benchmark statistics
    model_stats: Dict[str, Any] = {}
    for m_name in loaded_names:
        lats = model_latencies[m_name]
        cnts = model_det_counts[m_name]
        meta = MODEL_METADATA.get(m_name, {})
        model_stats[m_name] = {
            "model": m_name,
            "params_m": meta.get("params_m", 0.0),
            "avg_inference_ms": round(float(np.mean(lats)), 2) if lats else 0.0,
            "avg_detections": round(float(np.mean(cnts)), 2) if cnts else 0.0,
            "total_detections": sum(cnts),
            "is_fallback": fallback_flags[m_name]
        }

    return {
        "model_names": loaded_names,
        "skipped_models": skipped_models,
        "results_by_image": results_by_image,
        "model_stats": model_stats
    }


def build_comparison_grid(
    comparison_results: Dict[str, Any],
    tile_width: int = 480,
    output_dir: Optional[str] = None,
    title: Optional[str] = None
) -> Tuple[str, Dict[str, List[str]]]:
    """
    Renders a unified publication-grade comparison grid image (SH17 Fig. 5 style):
    - Top header row with bold model titles (YOLOv8-n, YOLOv8-m, YOLOv8-x, YOLOv9-e).
    - Left column with row index labels (1), (2), (3)...
    - Clean white/light gutters (3px) separating tiles.
    - Uniform aspect ratio tile scaling per row.
    - Bottom caption under each tile: 'N detections · X.X ms'.
    
    Saves:
    1. Grid PNG to outputs/comparison/comparison_<timestamp>.png
    2. Individual tiles to outputs/comparison/<model_name>/tile_<row_idx>.png
    
    Returns: (grid_image_filepath, per_model_tile_filepaths_dict)
    """
    if output_dir is None:
        output_dir = os.path.join(config.OUTPUT_FOLDER, "comparison")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_filename = f"comparison_{timestamp}.png"
    grid_filepath = os.path.join(output_dir, grid_filename)

    model_names: List[str] = comparison_results.get("model_names", [])
    results_by_image: List[Dict[str, Any]] = comparison_results.get("results_by_image", [])

    if not model_names or not results_by_image:
        raise ValueError("Cannot build comparison grid: no models or image results found.")

    # Save individual tiles per model
    tile_paths: Dict[str, List[str]] = {m: [] for m in model_names}
    for m in model_names:
        m_dir = os.path.join(output_dir, m.replace(" ", "_"))
        os.makedirs(m_dir, exist_ok=True)

    for row_info in results_by_image:
        row_idx = row_info["image_index"]
        for m in model_names:
            out_info = row_info["model_outputs"].get(m)
            if out_info:
                tile_fname = f"tile_row{row_idx:02d}_{m.replace(' ', '_')}.png"
                tile_p = os.path.join(output_dir, m.replace(" ", "_"), tile_fname)
                cv2.imwrite(tile_p, out_info["annotated_frame"])
                tile_paths[m].append(tile_p)

    # Grid Dimensions & Layout Constants
    gutter = 3                       # Gutter thickness between columns & rows
    left_margin_w = 54                # Left margin for row numbers (1), (2)...
    header_h = 56                     # Column header height
    caption_h = 28                    # Caption strip below each tile
    bg_color = (248, 250, 252)        # Off-white publication canvas (BGR)
    header_bg = (15, 23, 42)          # Dark slate header bar
    tile_caption_bg = (241, 245, 249) # Subtle light caption bar
    text_dark = (30, 41, 59)          # Dark slate font
    border_color = (203, 213, 225)    # Subtle border color

    num_cols = len(model_names)
    num_rows = len(results_by_image)

    # Precompute row heights based on tile_width scaling
    row_heights = []
    resized_tiles_matrix = []  # shape: (num_rows, num_cols)

    for row_info in results_by_image:
        max_h = 0
        row_tiles = []
        for m in model_names:
            ann_frame = row_info["model_outputs"][m]["annotated_frame"]
            orig_h, orig_w = ann_frame.shape[:2]
            scale = tile_width / float(orig_w)
            scaled_h = int(round(orig_h * scale))
            if scaled_h > max_h:
                max_h = scaled_h
            row_tiles.append((ann_frame, scaled_h, scale))
        row_heights.append(max_h)
        resized_tiles_matrix.append(row_tiles)

    # Calculate total canvas size
    total_grid_w = left_margin_w + (num_cols * tile_width) + ((num_cols + 1) * gutter)
    total_grid_h = header_h + sum(row_heights) + (num_rows * caption_h) + ((num_rows + 1) * gutter)

    # Create master blank canvas
    canvas = np.full((total_grid_h, total_grid_w, 3), bg_color, dtype=np.uint8)

    # 1. Render Column Header Row
    cv2.rectangle(canvas, (0, 0), (total_grid_w, header_h), header_bg, -1)

    # Model names centered in column headers
    for c_idx, m_name in enumerate(model_names):
        col_x = left_margin_w + gutter + c_idx * (tile_width + gutter)
        meta = MODEL_METADATA.get(m_name, {})
        params_str = f"({meta.get('params_m', 0.0)}M)" if meta else ""

        # Title text (e.g. YOLOv8-n)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale_title = 0.65
        th_title = 2
        (tw, th), _ = cv2.getTextSize(m_name, font, scale_title, th_title)
        center_x = col_x + (tile_width - tw) // 2
        cv2.putText(canvas, m_name, (center_x, 26), font, scale_title, (255, 255, 255), th_title, cv2.LINE_AA)

        # Subtitle: parameters
        if params_str:
            scale_sub = 0.38
            (sw, sh), _ = cv2.getTextSize(params_str, font, scale_sub, 1)
            sub_x = col_x + (tile_width - sw) // 2
            cv2.putText(canvas, params_str, (sub_x, 46), font, scale_sub, (200, 215, 235), 1, cv2.LINE_AA)

    # 2. Render Rows (Row Labels + Tiles + Captions)
    current_y = header_h + gutter

    for r_idx, row_info in enumerate(results_by_image):
        r_height = row_heights[r_idx]
        total_cell_h = r_height + caption_h

        # Left margin row label: (1), (2), (3)...
        row_label = f"({r_idx + 1})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (lw, lh), _ = cv2.getTextSize(row_label, font, 0.55, 2)
        label_x = (left_margin_w - lw) // 2
        label_y = current_y + (total_cell_h // 2) + (lh // 2)
        cv2.putText(canvas, row_label, (label_x, label_y), font, 0.55, text_dark, 2, cv2.LINE_AA)

        # Draw each model tile in this row
        for c_idx, m_name in enumerate(model_names):
            col_x = left_margin_w + gutter + c_idx * (tile_width + gutter)
            ann_frame, scaled_h, _ = resized_tiles_matrix[r_idx][c_idx]

            # Resize tile to width while keeping aspect ratio
            resized = cv2.resize(ann_frame, (tile_width, scaled_h), interpolation=cv2.INTER_AREA)

            # Pad vertically if this image is shorter than max row height
            pad_top = (r_height - scaled_h) // 2
            pad_bottom = r_height - scaled_h - pad_top
            if pad_top > 0 or pad_bottom > 0:
                resized = cv2.copyMakeBorder(
                    resized, pad_top, pad_bottom, 0, 0,
                    cv2.BORDER_CONSTANT, value=(240, 240, 242)
                )

            # Place image tile on canvas
            canvas[current_y : current_y + r_height, col_x : col_x + tile_width] = resized

            # Caption Bar below image
            cap_y1 = current_y + r_height
            cap_y2 = cap_y1 + caption_h
            cv2.rectangle(canvas, (col_x, cap_y1), (col_x + tile_width, cap_y2), tile_caption_bg, -1)
            cv2.rectangle(canvas, (col_x, current_y), (col_x + tile_width, cap_y2), border_color, 1)

            out_data = row_info["model_outputs"][m_name]
            dets_cnt = out_data["detection_count"]
            l_ms = out_data["inference_ms"]
            fb_tag = " [Base]" if out_data.get("is_fallback") else ""

            cap_text = f"{dets_cnt} detections · {l_ms:.1f} ms{fb_tag}"
            (cw, ch), _ = cv2.getTextSize(cap_text, font, 0.38, 1)
            cap_x = col_x + (tile_width - cw) // 2
            cv2.putText(canvas, cap_text, (cap_x, cap_y1 + 18), font, 0.38, text_dark, 1, cv2.LINE_AA)

        current_y += total_cell_h + gutter

    # Save the master comparison grid image
    cv2.imwrite(grid_filepath, canvas)
    print(f"[+] Saved SH17 Comparison Grid ({total_grid_w}x{total_grid_h}px) to: {grid_filepath}")

    return grid_filepath, tile_paths
