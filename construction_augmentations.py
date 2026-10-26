"""
Construction Site Data Augmentation Pipeline (Albumentations)
============================================================
Designed specifically for rugged, outdoor construction site conditions:
  1. Motion blur & camera vibration (moving machinery, wind-shaken CCTV poles)
  2. Dust, fog, haze, and airborne particle simulation
  3. Low light, dusk/dawn shifts, and harsh outdoor direct sunlight/shadows
  4. Partial occlusion simulation (scaffolding bars, tools, barricades, beams)
  5. Full compatibility with YOLO bounding boxes (format='yolo', normalized [x, y, w, h])

Author: Construction CV Safety Team
"""

import os
import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from typing import List, Tuple, Dict, Any, Optional


class ConstructionDustAndHaze(ImageOnlyTransform):
    """
    Simulates construction site airborne particulate matter (silica dust,
    cement powder blowover, atmospheric haze, and lens smudges).
    """
    def __init__(
        self,
        dust_intensity: Tuple[float, float] = (0.10, 0.35),
        color_tint: Tuple[int, int, int] = (195, 185, 170), # Earthy ochre / cement tint (BGR)
        always_apply: bool = False,
        p: float = 0.5
    ):
        super(ConstructionDustAndHaze, self).__init__(always_apply=always_apply, p=p)
        self.dust_intensity = dust_intensity
        self.color_tint = np.array(color_tint, dtype=np.float32)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w = img.shape[:2]
        intensity = np.random.uniform(self.dust_intensity[0], self.dust_intensity[1])

        # Generate smooth multi-frequency dust cloud noise
        low_res = np.random.uniform(0.7, 1.0, (h // 8, w // 8)).astype(np.float32)
        dust_mask = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_CUBIC)
        dust_mask = cv2.GaussianBlur(dust_mask, (31, 31), 0)
        dust_mask = np.clip(dust_mask, 0.0, 1.0)[:, :, np.newaxis]

        # Blend image towards dust color tint
        blended = img.astype(np.float32) * (1.0 - intensity * dust_mask) + (self.color_tint * intensity * dust_mask)
        return np.clip(blended, 0, 255).astype(np.uint8)


def get_construction_augmentation_pipeline(
    img_size: int = 640,
    p_dust: float = 0.45,
    p_blur: float = 0.50,
    p_lighting: float = 0.60,
    p_occlusion: float = 0.50
) -> A.Compose:
    """
    Constructs an Albumentations pipeline tailored for construction site CV.
    """
    transforms = [
        # Resize preserving aspect ratio or target size
        A.LongestMaxSize(max_size=img_size, p=1.0),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=(114, 114, 114), # YOLO standard neutral grey letterbox fill
            p=1.0
        ),

        # 1. MOTION BLUR & OPTICAL DISTORTION (Crane movements, fast worker motion, CCTV vibration)
        A.OneOf([
            A.MotionBlur(blur_limit=(7, 19), p=0.5),
            A.GaussianBlur(blur_limit=(3, 9), p=0.3),
            A.Defocus(radius=(2, 5), alias_blur=(0.1, 0.3), p=0.2),
        ], p=p_blur),

        # 2. DUST, FOG, ATMOSPHERIC HAZE & LENS IMPERFECTIONS
        A.OneOf([
            ConstructionDustAndHaze(dust_intensity=(0.12, 0.35), p=0.6),
            A.RandomFog(fog_coef_lower=0.15, fog_coef_upper=0.45, alpha_coef=0.1, p=0.4),
            A.Spatter(mean=0.6, std=0.2, gauss_sigma=2, intensity=0.4, p=0.3),
        ], p=p_dust),

        # 3. HARSH OUTDOOR LIGHTING, LOW LIGHT (DUSK/SHADOWS) & HIGH CONTRAST SUNLIGHT
        A.OneOf([
            # Extreme outdoor direct sunlight / high dynamic contrast
            A.RandomBrightnessContrast(
                brightness_limit=(-0.35, 0.40),
                contrast_limit=(-0.30, 0.45),
                p=0.5
            ),
            # Low-light / overcast dusk transition
            A.Sequential([
                A.RandomGamma(gamma_limit=(60, 130), p=0.7),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
            ], p=0.5),
            # Sensor noise in dim conditions
            A.ISONoise(color_shift=(0.02, 0.08), intensity=(0.15, 0.45), p=0.3),
        ], p=p_lighting),

        # 4. PARTIAL OCCLUSION SIMULATION (Scaffolding poles, pipes, wooden planks, barricades)
        A.OneOf([
            # Simulates scaffolding bars & structural obstructions across worker bodies
            A.CoarseDropout(
                max_holes=6,
                min_holes=1,
                max_height=int(img_size * 0.25),
                min_height=int(img_size * 0.04),
                max_width=int(img_size * 0.25),
                min_width=int(img_size * 0.04),
                fill_value=(70, 70, 70), # Dark metal / bar color
                p=0.6
            ),
            # Thin rectangular bars cutting across worker
            A.CoarseDropout(
                max_holes=4,
                min_holes=1,
                max_height=int(img_size * 0.03), # Thin horizontal bar
                min_height=int(img_size * 0.01),
                max_width=int(img_size * 0.70),
                min_width=int(img_size * 0.20),
                fill_value=(180, 150, 40), # Yellow/orange safety barrier tape or timber
                p=0.4
            )
        ], p=p_occlusion),

        # 5. GEOMETRIC ROTATION & HORIZONTAL FLIP
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.10,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
            p=0.5
        ),
    ]

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="yolo", # [x_center, y_center, width, height] normalized in [0, 1]
            label_fields=["category_ids"],
            min_visibility=0.25, # Drop bbox if occlusion hides >75% of it
            min_area=16          # Discard micro-artefact bounding boxes
        )
    )


def augment_image_and_labels(
    image: np.ndarray,
    bboxes: List[List[float]],
    category_ids: List[int],
    pipeline: Optional[A.Compose] = None
) -> Tuple[np.ndarray, List[List[float]], List[int]]:
    """
    Applies the construction site augmentation pipeline to an image and its YOLO bboxes.
    """
    if pipeline is None:
        pipeline = get_construction_augmentation_pipeline()

    transformed = pipeline(
        image=image,
        bboxes=bboxes,
        category_ids=category_ids
    )
    return transformed["image"], transformed["bboxes"], transformed["category_ids"]


if __name__ == "__main__":
    print("[+] Construction Albumentations Augmentation Pipeline loaded successfully.")
    pipe = get_construction_augmentation_pipeline(img_size=640)
    print(f"[+] Total Active Transform Stages: {len(pipe.transforms)}")
    print("[+] Pipeline ready for YOLOv8 data loader integration.")
