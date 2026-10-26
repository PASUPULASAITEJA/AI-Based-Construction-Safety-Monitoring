"""
CV Module: Visual Frame Annotator
=================================
Renders high-visibility bounding boxes, 8-category PPE checklist panels,
temporal compliance badges (green check, red X, grey ?), zones, and dashboard HUDs.
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional

# Color Palette (BGR)
COLOR_COMPLIANT = (46, 204, 113)     # Emerald Green
COLOR_WARNING = (243, 156, 18)       # Amber / Orange
COLOR_VIOLATION = (40, 40, 235)      # Crimson Red
COLOR_UNCERTAIN = (140, 150, 160)    # Cool Grey
COLOR_DARK_PANEL = (15, 20, 26)      # Slate Dark Panel
COLOR_TEXT_WHITE = (255, 255, 255)
COLOR_TEXT_MUTED = (160, 175, 190)

# Paper Fig. 1/6 Stable per-class color palette (BGR)
CLASS_COLORS = {
    "person": (235, 130, 40),          # Vibrant Blue
    "worker": (235, 130, 40),
    "helmet": (46, 204, 113),          # Emerald Green
    "no-helmet": (40, 40, 235),        # Crimson Red
    "goggles": (220, 200, 30),         # Teal / Cyan
    "no-goggles": (180, 20, 200),      # Magenta
    "earmuff": (240, 220, 100),        # Light Sky
    "no-earmuff": (40, 60, 180),       # Rust Red
    "vest": (30, 144, 255),            # Amber / Deep Orange
    "no-vest": (20, 20, 220),          # Bright Red
    "safety-pants": (220, 100, 150),   # Purple
    "no-safety-pants": (150, 50, 120), # Dark Violet
    "gloves": (50, 215, 255),          # Gold Yellow
    "no-gloves": (30, 30, 180),        # Dark Crimson
    "boots": (80, 230, 120),           # Lime
    "no-boots": (50, 20, 150),         # Maroon
    "harness": (255, 180, 50),         # Electric Sky Blue
    "no-harness": (60, 40, 210),       # Rose Red
    "none": (120, 120, 120),
}


def get_class_color(class_name: str) -> tuple:
    """Returns a stable BGR color tuple for a given class name."""
    norm = str(class_name).strip().lower().replace("_", "-")
    if norm in CLASS_COLORS:
        return CLASS_COLORS[norm]
    # Fallback deterministic color from string hash
    h = abs(hash(norm))
    b = (h & 0xFF) % 200 + 35
    g = ((h >> 8) & 0xFF) % 200 + 35
    r = ((h >> 16) & 0xFF) % 200 + 35
    return (b, g, r)


def plain_annotate(frame: np.ndarray, detections: List[Dict[str, Any]], draw_conf: bool = True) -> np.ndarray:
    """
    Renders clean, publication-grade annotated outputs (SH17 paper Fig. 1 & Fig. 6 style):
    - Thin bounding box with fixed per-class color.
    - Small filled label tag with class name (and confidence score).
    - No HUD, no checklist panels, no zone polygons.
    """
    out_frame = frame.copy()
    for det in detections:
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        # Ensure within frame bounds
        h_f, w_f = out_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_f - 1, x2), min(h_f - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        cname = det.get("category", det.get("class_name", "object"))
        conf = det.get("confidence", 0.0)
        color = get_class_color(cname)

        # 1. Thin bounding box
        cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # 2. Filled label tag
        label_text = f"{cname} {conf:.2f}" if (draw_conf and conf > 0) else str(cname)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.42
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, font, scale, thickness)

        # Place label above box, or inside if near top boundary
        tag_y1 = max(0, y1 - th - baseline - 4)
        tag_y2 = y1
        if y1 - th - baseline - 4 < 0:
            tag_y1 = y1
            tag_y2 = y1 + th + baseline + 4

        cv2.rectangle(out_frame, (x1, tag_y1), (x1 + tw + 6, tag_y2), color, -1)
        text_y = tag_y2 - baseline - 2 if tag_y1 == y1 else y1 - baseline - 2
        cv2.putText(
            out_frame,
            label_text,
            (x1 + 3, text_y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return out_frame


PPE_CHECKLIST_CONFIG = [
    {"key": "helmet", "abbr": "HLM", "icon": "H"},
    {"key": "goggles", "abbr": "GGL", "icon": "G"},
    {"key": "earmuff", "abbr": "EAR", "icon": "E"},
    {"key": "vest", "abbr": "VST", "icon": "V"},
    {"key": "harness", "abbr": "HAR", "icon": "R"},
    {"key": "safety_pants", "abbr": "PNT", "icon": "P"},
    {"key": "gloves", "abbr": "GLV", "icon": "L"},
    {"key": "boots", "abbr": "BOT", "icon": "B"}
]


class FrameAnnotator:
    def __init__(self):
        pass

    def draw_zones(self, frame: np.ndarray, zones: List[Dict[str, Any]]) -> np.ndarray:
        """Draws semi-transparent polygon zones with styled borders."""
        overlay = frame.copy()
        for z in zones:
            pts = z["points"]
            z_type = z.get("type", "SAFE")
            z_name = z.get("name", "Zone")

            if z_type in ["HAZARD", "CRITICAL"]:
                fill_color = (0, 0, 180)
                border_color = (0, 50, 255)
            elif z_type in ["RESTRICTED", "HIGH"]:
                fill_color = (180, 80, 0)
                border_color = (255, 140, 0)
            else:
                fill_color = (40, 140, 40)
                border_color = (50, 220, 50)

            cv2.fillPoly(overlay, [pts], fill_color)
            cv2.polylines(frame, [pts], isClosed=True, color=border_color, thickness=2)

            p1 = tuple(pts[0][0])
            cv2.putText(frame, f"[{z_type}] {z_name}", (p1[0] + 5, p1[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, border_color, 2, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
        return frame

    def draw_ppe_checklist_panel(
        self,
        frame: np.ndarray,
        panel_x: int,
        panel_y: int,
        worker: Dict[str, Any]
    ) -> None:
        """
        Renders a compact, sleek 8-category PPE checklist panel next to the worker:
        - 8 items: [HLM, GGL, EAR, VST, HAR, PNT, GLV, BOT]
        - Green checkmark (v) = detected
        - Red cross (x) = missing
        - Grey question mark (?) = uncertain
        - Compliance score % badge
        """
        h_frame, w_frame = frame.shape[:2]
        panel_w = 110
        item_h = 16
        header_h = 24
        panel_h = header_h + (len(PPE_CHECKLIST_CONFIG) * item_h) + 6

        # Keep panel inside frame boundaries
        if panel_x + panel_w > w_frame - 5:
            panel_x = max(5, panel_x - panel_w - int(worker["bbox"][2] - worker["bbox"][0]) - 10)
        panel_y = max(5, min(h_frame - panel_h - 5, panel_y))

        # 1. Semi-transparent dark glass background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            COLOR_DARK_PANEL,
            -1
        )
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (50, 65, 80),
            1
        )

        # 2. Header: Compliance Score Badge
        score_val = worker.get("compliance_score", 0.0)
        score_pct = int(round(score_val * 100)) if score_val <= 1.0 else int(score_val)

        if score_pct >= 80:
            badge_color = COLOR_COMPLIANT
        elif score_pct >= 50:
            badge_color = COLOR_WARNING
        else:
            badge_color = COLOR_VIOLATION

        # Header background
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + header_h),
            (25, 33, 44),
            -1
        )
        cv2.putText(
            frame,
            f"PPE {score_pct}%",
            (panel_x + 8, panel_y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            badge_color,
            1,
            cv2.LINE_AA
        )

        # 3. 8 PPE checklist items
        curr_y = panel_y + header_h + 12
        for item in PPE_CHECKLIST_CONFIG:
            key = item["key"]
            abbr = item["abbr"]
            val = worker.get(key, False)

            if val is True or val == "compliant":
                mark = "[v]"
                mark_color = COLOR_COMPLIANT
            elif val == "uncertain" or val == "?":
                mark = "[?]"
                mark_color = COLOR_UNCERTAIN
            else:
                mark = "[x]"
                mark_color = COLOR_VIOLATION

            # Item abbreviation label
            cv2.putText(
                frame,
                abbr,
                (panel_x + 8, curr_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                COLOR_TEXT_MUTED,
                1,
                cv2.LINE_AA
            )
            # Status mark
            cv2.putText(
                frame,
                mark,
                (panel_x + 65, curr_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                mark_color,
                1,
                cv2.LINE_AA
            )
            curr_y += item_h

    def draw_worker(self, frame: np.ndarray, worker: Dict[str, Any], show_panel: bool = True) -> np.ndarray:
        """Renders corner-bracket worker box and triggers adjacent checklist panel if requested."""
        x1, y1, x2, y2 = [int(v) for v in worker["bbox"]]
        label = worker.get("label", worker.get("worker_id", "Worker"))
        score_val = worker.get("compliance_score", 0.0)
        score_pct = int(round(score_val * 100)) if score_val <= 1.0 else int(score_val)
        violations = worker.get("violations", [])

        # Color based on compliance & violations
        if violations or score_pct < 50:
            box_color = COLOR_VIOLATION
        elif score_pct < 80:
            box_color = COLOR_WARNING
        else:
            box_color = COLOR_COMPLIANT

        # Main bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Corner brackets
        c_len = min(18, (x2 - x1) // 4, (y2 - y1) // 4)
        th = 3
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + c_len, y1), box_color, th)
        cv2.line(frame, (x1, y1), (x1, y1 + c_len), box_color, th)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - c_len, y1), box_color, th)
        cv2.line(frame, (x2, y1), (x2, y1 + c_len), box_color, th)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + c_len, y2), box_color, th)
        cv2.line(frame, (x1, y2), (x1, y2 - c_len), box_color, th)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - c_len, y2), box_color, th)
        cv2.line(frame, (x2, y2), (x2, y2 - c_len), box_color, th)

        # Worker Label Header Banner
        header_text = f"{label} ({score_pct}%)"
        text_sz, _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tw, th_sz = text_sz
        by1 = max(0, y1 - th_sz - 10)
        cv2.rectangle(frame, (x1, by1), (x1 + tw + 14, y1), box_color, -1)
        cv2.putText(
            frame,
            header_text,
            (x1 + 6, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_TEXT_WHITE,
            1,
            cv2.LINE_AA
        )

        # Violation banner under box if any
        if violations:
            v_desc = ", ".join([v.get("missing_item", "Violation") for v in violations[:2]])
            cv2.rectangle(frame, (x1, y2), (x1 + tw + 14, y2 + 18), (0, 0, 200), -1)
            cv2.putText(
                frame,
                f"! {v_desc}",
                (x1 + 4, y2 + 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                COLOR_TEXT_WHITE,
                1,
                cv2.LINE_AA
            )

        # Render adjacent 8-item PPE checklist panel next to bounding box if space allows
        if show_panel:
            panel_x = x2 + 6
            panel_y = y1
            self.draw_ppe_checklist_panel(frame, panel_x, panel_y, worker)

        return frame

    def draw_dashboard_hud(
        self,
        frame: np.ndarray,
        fps: float = 0.0,
        worker_count: int = 0,
        violation_count: int = 0
    ) -> np.ndarray:
        """Top-left HUD statistics display."""
        hud_w = 360
        hud_h = 36
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), COLOR_DARK_PANEL, -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        cv2.rectangle(frame, (10, 10), (10 + hud_w, 10 + hud_h), (55, 70, 85), 1)

        fps_str = f"FPS: {fps:.1f} | " if fps > 0 else ""
        hud_text = f"{fps_str}Workers: {worker_count} | Violations: {violation_count}"
        cv2.putText(
            frame,
            hud_text,
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 230, 255),
            1,
            cv2.LINE_AA
        )
        return frame

    def annotate(
        self,
        frame: np.ndarray,
        workers: List[Dict[str, Any]],
        zones: Optional[List[Dict[str, Any]]] = None,
        fps: float = 0.0,
        style: str = "full",
        raw_detections: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        """
        Master annotation pipeline.
        
        Args:
            frame: Input video/image BGR frame
            workers: List of worker dictionaries with spatial association and violations
            zones: Optional geofence polygon zones
            fps: Current system FPS
            style: 'full' (rich HUD + checklist panels + zones) or 'plain' (clean per-class boxes + tags)
            raw_detections: Optional list of raw YOLO detections for plain paper-style rendering
        """
        if style == "plain":
            dets_to_draw = []
            if raw_detections is not None:
                dets_to_draw = raw_detections
            else:
                for w in workers:
                    dets_to_draw.append({
                        "bbox": w["bbox"],
                        "category": "person",
                        "class_name": "person",
                        "confidence": w.get("confidence", 0.90)
                    })
                    for item in PPE_CHECKLIST_CONFIG:
                        k = item["key"]
                        if w.get(k) is True:
                            dets_to_draw.append({
                                "bbox": w["bbox"],
                                "category": k,
                                "class_name": k,
                                "confidence": 0.85
                            })
            return plain_annotate(frame, dets_to_draw)

        out_frame = frame.copy()
        if zones:
            out_frame = self.draw_zones(out_frame, zones)

        active_violations = 0
        for w in workers:
            if w.get("violations"):
                active_violations += len(w["violations"])
            elif w.get("compliance_score", 1.0) < 0.8:
                active_violations += 1

        # Automatically avoid cluttering if multiple workers are present
        show_panels = len(workers) <= 2
        for w in workers:
            out_frame = self.draw_worker(out_frame, w, show_panel=show_panels)

        out_frame = self.draw_dashboard_hud(
            out_frame,
            fps=fps,
            worker_count=len(workers),
            violation_count=active_violations
        )
        return out_frame


def draw_hud_and_detections(frame, workers, zones=None, fps=0.0, style="full", raw_detections=None):
    """Functional wrapper for FrameAnnotator."""
    annotator = FrameAnnotator()
    return annotator.annotate(frame, workers, zones=zones, fps=fps, style=style, raw_detections=raw_detections)


# ==============================================================================
# Research Paper SH17 Visual Representation & Benchmark Renderer (Fig. 1, 5, 6)
# ==============================================================================

SH17_CLASS_COLORS = {
    "person": (72, 30, 225),             # Rose / Red (OSHA Worker anchor)
    "head": (20, 140, 245),              # Warm Orange
    "face": (180, 50, 230),              # Pink / Violet
    "helmet": (240, 200, 0),             # Vivid Cyan / Teal (Hardhat)
    "safety-vest": (230, 110, 20),        # High-Vis Blue
    "glasses": (245, 175, 20),           # Sky Blue
    "face-mask-medical": (180, 210, 30),  # Mint Green
    "face-guard": (180, 170, 30),         # Teal
    "ear": (110, 60, 230),               # Magenta
    "ear-mufs": (210, 90, 150),          # Purple
    "hands": (70, 210, 40),              # Emerald Green
    "gloves": (90, 220, 60),             # Lime / Green
    "shoes": (210, 70, 170),             # Violet / Purple
    "foot": (210, 50, 210),              # Fuchsia
    "tools": (0, 215, 255),              # Bright Yellow
    "safety-suit": (220, 130, 80),       # Indigo
    "medical-suit": (220, 180, 40),      # Cyan
}

def draw_sh17_predictions(frame, detections, model_tag=None, latency_ms=None, show_conf=True):
    """
    Renders bounding boxes and label tags in the exact visual style of:
    - Page 6 Fig. 1 (SH17 Ground Truth Annotations)
    - Page 13 Fig. 5 (Multi-Model Visual Representation: YOLOv8-n, v8-m, v8-x, v9-e)
    - Page 14 Fig. 6 (Pictor-PPE Cross-Domain Inference)
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # Draw model header banner if specified
    if model_tag:
        banner_h = 32
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (12, 16, 22), -1)
        cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)
        cv2.line(out, (0, banner_h), (w, banner_h), (40, 50, 65), 1)

        tag_text = model_tag
        if latency_ms:
            tag_text += f" | Latency: {latency_ms:.1f}ms"
        tag_text += f" | Detections: {len(detections)}"

        cv2.putText(out, tag_text, (12, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)

    # Draw each detection with class-specific color
    for det in detections:
        bbox = det.get("bbox") or det.get("coordinates")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        cls_name = str(det.get("class_name", det.get("class", det.get("name", "object")))).lower()
        conf = det.get("confidence", det.get("conf", 0.0))

        color = SH17_CLASS_COLORS.get(cls_name, (0, 220, 220))

        # 1. Main bounding box (thickness 2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # 2. Corner accent markers (subtle industrial aesthetic)
        c_len = max(6, min(14, (x2 - x1) // 5, (y2 - y1) // 5))
        cv2.line(out, (x1, y1), (x1 + c_len, y1), (255, 255, 255), 2)
        cv2.line(out, (x1, y1), (x1, y1 + c_len), (255, 255, 255), 2)
        cv2.line(out, (x2, y2), (x2 - c_len, y2), (255, 255, 255), 2)
        cv2.line(out, (x2, y2), (x2, y2 - c_len), (255, 255, 255), 2)

        # 3. Label tag above box
        if show_conf and conf > 0:
            label_text = f"{cls_name} {int(conf * 100)}%" if conf <= 1.0 else f"{cls_name} {int(conf)}%"
        else:
            label_text = cls_name

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        ty1 = max(0, y1 - th - 6)
        ty2 = y1
        tx2 = min(w, x1 + tw + 8)

        # Dark background with color bar
        cv2.rectangle(out, (x1, ty1), (tx2, ty2), (15, 18, 24), -1)
        cv2.rectangle(out, (x1, ty1), (x1 + 3, ty2), color, -1)
        cv2.putText(out, label_text, (x1 + 6, ty2 - 3),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return out

