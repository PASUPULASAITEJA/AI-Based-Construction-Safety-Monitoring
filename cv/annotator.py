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
        if "CRITICAL" in status or any("CRITICAL" in str(v.get("severity", "")) for v in violations):
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
            h_box = associated_items.get("helmet_box")
            if h_box and has_helmet:
                hx1, hy1, hx2, hy2 = h_box
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (50, 220, 50), 1)
                cv2.putText(frame, "Helmet", (hx1, max(12, hy1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 220, 50), 1, cv2.LINE_AA)

            v_box = associated_items.get("vest_box")
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
            v_desc = ", ".join([v.get("type", "Violation") for v in violations])
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
