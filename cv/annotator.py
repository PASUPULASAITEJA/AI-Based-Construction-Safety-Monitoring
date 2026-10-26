"""
CV Module: Visual Frame Annotator
Renders high-visibility bounding boxes, PPE status badges, zones, and alert banners over OpenCV frames.
"""
import cv2
import numpy as np

# BGR Color Palette
COLOR_COMPLIANT = (46, 204, 113)    # Emerald Green
COLOR_WARNING = (243, 156, 18)      # Amber / Orange
COLOR_VIOLATION = (40, 40, 235)     # Crimson Red
COLOR_RESTRICTED = (255, 120, 0)    # Azure Blue / Cyan
COLOR_HAZARD = (0, 69, 255)         # Bright Orange-Red
COLOR_TEXT = (255, 255, 255)
COLOR_DARK_BG = (20, 24, 30)

def _violation_field(v, key):
    """Violations are rule dicts ({"type", "severity", ...}); tolerate plain description strings too."""
    if isinstance(v, dict):
        return str(v.get(key, ""))
    return str(v)

class FrameAnnotator:
    def __init__(self):
        pass

    def draw_zones(self, frame, zones):
        """Draws semi-transparent polygon zones."""
        overlay = frame.copy()
        for z in zones:
            pts = z["points"]
            z_type = z["type"]
            z_name = z["name"]

            if z_type == "HAZARD":
                fill_color = (0, 0, 180)
                border_color = (0, 50, 255)
            elif z_type == "RESTRICTED":
                fill_color = (180, 80, 0)
                border_color = (255, 140, 0)
            else:
                fill_color = (40, 140, 40)
                border_color = (50, 220, 50)

            # Fill polygon
            cv2.fillPoly(overlay, [pts], fill_color)
            # Outline
            cv2.polylines(frame, [pts], isClosed=True, color=border_color, thickness=2)

            # Draw zone label at first vertex
            p1 = tuple(pts[0][0])
            cv2.putText(frame, f"[{z_type}] {z_name}", (p1[0] + 5, p1[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 2)

        # Blend overlay
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        return frame

    def draw_worker(self, frame, worker):
        """Draws worker bounding box, associated PPE item boxes, and status badges."""
        x1, y1, x2, y2 = worker["bbox"]
        label = worker.get("label", f"Worker #{worker.get('worker_id', 1)}")
        has_helmet = worker.get("helmet", False)
        has_vest = worker.get("vest", False)
        violations = worker.get("violations", [])
        status = worker.get("status", "COMPLIANT")
        zone_type = worker.get("zone_type", "SAFE")
        associated_items = worker.get("associated_items", {})

        # Determine box color
        if "CRITICAL" in status or any("CRITICAL" in _violation_field(v, "severity") for v in violations):
            box_color = (0, 0, 255)
        elif violations:
            box_color = COLOR_VIOLATION
        elif not has_helmet or not has_vest or zone_type != "SAFE":
            box_color = COLOR_WARNING
        else:
            box_color = COLOR_COMPLIANT

        # 1. Main Worker Bounding Box with corner highlights
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        thickness = 4
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_color, thickness)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), box_color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), box_color, thickness)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), box_color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), box_color, thickness)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), box_color, thickness)

        # 2. Draw Associated PPE item boxes (Helmet / Vest)
        if associated_items:
            h_box = associated_items.get("HEAD_box")
            if h_box and has_helmet:
                hx1, hy1, hx2, hy2 = h_box
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (50, 220, 50), 1)
                cv2.putText(frame, "Helmet", (hx1, max(12, hy1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 220, 50), 1, cv2.LINE_AA)

            v_box = associated_items.get("VISIBILITY_box")
            if v_box and has_vest:
                vx1, vy1, vx2, vy2 = v_box
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 240, 255), 1)
                cv2.putText(frame, "Vest", (vx1, max(12, vy1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 240, 255), 1, cv2.LINE_AA)

        # 3. Worker Label Header
        h_icon = "H:OK" if has_helmet else "H:NO"
        v_icon = "V:OK" if has_vest else "V:NO"
        badge_str = f"{label} | {h_icon} {v_icon}"
        if zone_type != "SAFE":
            badge_str += f" | {zone_type}"

        badge_w = max(180, len(badge_str) * 9)
        badge_h = 24
        by1 = max(0, y1 - badge_h)
        by2 = y1

        cv2.rectangle(frame, (x1, by1), (x1 + badge_w, by2), box_color, -1)
        cv2.putText(frame, badge_str, (x1 + 6, by2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Violation Banner if active
        if violations:
            v_desc = ", ".join(_violation_field(v, "type") for v in violations)
            v_h = 20
            cv2.rectangle(frame, (x1, y2), (x1 + badge_w, y2 + v_h), (0, 0, 200), -1)
            cv2.putText(frame, f"! {v_desc}", (x1 + 4, y2 + v_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def draw_dashboard_hud(self, frame, fps=0, worker_count=0, violation_count=0):
        """Draws a sleek status HUD in the top-left of the stream."""
        h, w, _ = frame.shape
        hud_w = 340
        hud_h = 36
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (15, 20, 26), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (10, 10), (10 + hud_w, 10 + hud_h), (60, 75, 90), 1)

        if fps > 0:
            hud_text = f"FPS: {fps:.1f} | Workers: {worker_count} | Violations: {violation_count}"
        else:
            hud_text = f"SAFETY AUDIT | Workers: {worker_count} | Violations: {violation_count}"

        cv2.putText(frame, hud_text, (20, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 240, 255), 1, cv2.LINE_AA)
        return frame

    def annotate(self, frame, workers, zones=None, fps=0, violations_summary=None):
        """
        Master annotation pipeline.
        """
        out_frame = frame.copy()
        if zones:
            out_frame = self.draw_zones(out_frame, zones)

        active_violations_count = 0
        for w in workers:
            if w.get("violations"):
                active_violations_count += len(w["violations"])
            out_frame = self.draw_worker(out_frame, w)

        out_frame = self.draw_dashboard_hud(
            out_frame,
            fps=fps,
            worker_count=len(workers),
            violation_count=active_violations_count
        )
        return out_frame

# Module-level convenience singleton and helper
_default_annotator = FrameAnnotator()

def draw_hud_and_detections(frame, workers, zones=None, fps=0.0, frame_num=0):
    """Convenience functional wrapper around FrameAnnotator."""
    formatted_zones = []
    if zones:
        for z in zones:
            coords = z.get("coordinates") or z.get("points")
            if coords:
                pts = np.array(coords, np.int32).reshape((-1, 1, 2))
                formatted_zones.append({
                    "points": pts,
                    "type": z.get("zone_type") or z.get("type", "RESTRICTED"),
                    "name": z.get("name", "Zone")
                })
    return _default_annotator.annotate(frame, workers, zones=formatted_zones, fps=fps)

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

