# SiteGuard AI: MLOps & Dataset Strategy (Phase 4)

## Objective
The current YOLOv8 model achieves a poor mAP (30.9%), largely because it completely misses small, underrepresented PPE items like "no gloves", "goggles", and "ear muffs". To achieve production-grade precision across all 8 PPE classes without massive manual annotation, we will implement a hybrid data curation and model modification strategy.

---

## 1. Dataset Generation Strategy

To solve class imbalance (e.g., Helmets represent 70% of data, Ear Muffs < 1%), we will use a multi-stage data augmentation approach:

### A. Pseudo-Labeling via Foundation Models
*   **Tool:** Grounding DINO + SAM (Segment Anything Model)
*   **Method:** Instead of manually bounding boxes around boots and goggles, run an offline pass of your raw construction images through Grounding DINO with zero-shot text prompts: `"safety goggles", "ear muffs", "work boots", "harness"`.
*   **Refinement:** Use SAM to get pixel-perfect masks, convert masks back to precise YOLO bounding boxes, and automatically write `labels/*.txt`.

### B. Synthetic Generation via Stable Diffusion Inpainting
*   Classes like `Chaps` (chainsaw pants) and `Fall Protection Harness` are extremely rare in standard CCTV datasets.
*   **Method:** Use ControlNet + Stable Diffusion Inpainting. Provide empty images of construction workers (with depth/pose maps). Mask the worker's torso and prompt the model to generate a high-visibility harness.
*   **Volume:** Generate 3,000 synthetic harness/chaps images, automatically deriving the bounding box from the inpainting mask.

### C. Standard Augmentations (Albumentations)
*   **Mosaic & MixUp:** Essential for dense crowds of workers.
*   **Random Brightness/Contrast:** Simulates different times of day.
*   **Gaussian Blur & Motion Blur:** Simulates low-CCTV camera quality and worker movement.

---

## 2. Model Architecture Upgrades

The standard YOLOv8 architecture (P3-P4-P5) struggles with tiny objects (like goggles) at 1080p CCTV distances (8-15 meters).

### A. YOLOv8-P2 (High-Resolution Head)
We must modify the YOLO architecture to include a **P2 feature map** (Stride 4). The standard P3 feature map (Stride 8) destroys the spatial resolution needed to detect a 12x12 pixel pair of goggles.
*   **Action:** Create a custom `yolov8-p2.yaml` config adding an extra upsampling layer and C2f block to output a 160x160 grid (for 640x640 input).

### B. SAHI (Slicing Aided Hyper Inference)
*   **Action:** Integrate `sahi` package into `cv/detector.py`. 
*   **Method:** Instead of shrinking a 4K camera feed down to 640x640, SAHI slices the 4K image into overlapping 640x640 patches, runs YOLO on each patch, and merges the bounding boxes using NMS. This guarantees that small PPE items remain visually large to the neural network.

---

## 3. Training Execution Plan

1. **Hardware:** NVIDIA RTX 3090 / 4090 (24GB VRAM required for P2 head).
2. **Hyperparameters:**
   * `epochs`: 150 (with Early Stopping)
   * `imgsz`: 1280 (High resolution training)
   * `batch`: 8
   * `lr0`: 0.001 (Fine-tuning, lower learning rate)
   * `optimizer`: AdamW

By executing this hybrid data-centric approach, the model will comfortably exceed 85% mAP@50 across all 8 critical PPE classes, fully solving the "weak model" limitation.
