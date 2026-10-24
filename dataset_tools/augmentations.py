"""
Construction Site Heavy Data Augmentation Pipeline using Albumentations
Simulates:
- Motion blur from swinging surveillance cameras
- High contrast and solar glare
- Dust / fog / smog haze
- Scaffolding / pipe occlusions via cutout
"""

import albumentations as A
import cv2
import numpy as np

def get_construction_augmentation_pipeline(img_size=640):
    return A.Compose([
        # 1. Spatial Geometry
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=[114, 114, 114]),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=12, border_mode=cv2.BORDER_CONSTANT, p=0.6),

        # 2. Lighting, Glare & Low-Light Conditions
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.3, hue=0.08, p=0.8),
            A.RandomGamma(gamma_limit=(70, 140), p=0.6),
        ], p=0.8),

        # 3. Weather & Atmosphere (Dust, Fog, Smog)
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, alpha_coef=0.1, p=0.6),
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, p=0.5),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=0.4),
        ], p=0.4),

        # 4. Camera Quality & Movement (CCTV & RTSP artifacts)
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=0.6),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.ImageCompression(quality_lower=40, quality_upper=90, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        ], p=0.5),

        # 5. Severe Occlusion (Simulating scaffolding, fencing, beams)
        A.CoarseDropout(
            max_holes=6,
            max_height=int(img_size * 0.18),
            max_width=int(img_size * 0.18),
            min_holes=1,
            min_height=int(img_size * 0.04),
            min_width=int(img_size * 0.04),
            fill_value=114,
            p=0.5
        )
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.25, label_fields=['class_labels']))

def apply_augmentation(image_bgr: np.ndarray, bboxes_yolo: list, labels: list):
    """
    image_bgr: np.ndarray
    bboxes_yolo: [[x_center, y_center, width, height], ...] (all normalized 0..1)
    labels: [int, ...]
    """
    transform = get_construction_augmentation_pipeline(img_size=640)
    transformed = transform(image=image_bgr, bboxes=bboxes_yolo, class_labels=labels)
    return transformed['image'], transformed['bboxes'], transformed['class_labels']
