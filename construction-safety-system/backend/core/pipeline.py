"""
Master Safety Vision Pipeline & OpenCV Visual Annotator
Orchestrates Object Detection -> ByteTrack -> Anatomical PPE Association -> Spatial Zones -> Temporal Latching.
Renders high-visibility overlays: bounding boxes, foot anchor dots, PPE badges, and polygon zones.
"""
from typing import List, Dict, Any, Tuple
import json
import time
import cv2
import numpy as np

from core.detector import ObjectDetector
from core.tracker import ByteTrackTracker
from core.association import associate_ppe_to_workers
from core.spatial import evaluate_spatial_containment, compute_foot_anchor
from core.temporal_engine import TemporalSafetyEngine

class SafetyVisionPipeline:
    def __init__(self, detector: ObjectDetector = None):
        self.detector = detector if detector is not None else ObjectDetector()
        self.tracker = ByteTrackTracker(max_age=30, min_hits=2)
        self.temporal_engine = TemporalSafetyEngine()
        self.fps = 0.0
        self.last_process_time = time.time()

    def process_frame(
        self,
        frame: np.ndarray,
        zones: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Executes sequential end-to-end vision pipeline:
        1. YOLO detection
        2. ByteTrack multi-object tracking
        3. Anatomical PPE association
        4. Spatial zone containment evaluation
        5. Temporal hysteresis rule confirmation
        6. Visual HUD overlay annotation
        
        Returns:
            annotated_frame: np.ndarray with drawn visual overlays
            enriched_workers: List of worker states
            new_incidents: List of newly confirmed incidents
            resolved_incidents: List of unlatched recovered incidents
            stats: Telemetry statistics dict
        """
        t0 = time.time()

        # Step 1: Detect workers, helmets, vests
        workers_det, ppe_det = self.detector.detect(frame)

        # Step 2: Track workers across frames (maintain persistent track_id)
        tracked_workers = self.tracker.update(workers_det)

        # Step 3: Anatomical Head/Torso ROI association
        enriched_workers = associate_ppe_to_workers(tracked_workers, ppe_det)

        # Step 4 & 5: Spatial containment & Temporal confirmation per worker
        new_incidents = []
        resolved_incidents = []
        active_violations_map: Dict[int, List[str]] = {}

        for w in enriched_workers:
            # Check spatial zones with bottom-center anchor
            breached_zones = evaluate_spatial_containment(w["bbox"], zones)
            w["breached_zones"] = breached_zones
            w["is_in_zone"] = len(breached_zones) > 0

            # Temporal engine sliding window
            events = self.temporal_engine.process_worker(w, breached_zones, frame)
            
            w_violations = []
            for ev in events:
                if ev.get("is_new"):
                    new_incidents.append(ev)
                if ev.get("is_resolved"):
                    resolved_incidents.append(ev)
                if ev.get("is_active"):
                    w_violations.append(ev["violation_type"])

            w["active_violations"] = w_violations
            active_violations_map[w["track_id"]] = w_violations

        # Prune dead workers
        self.temporal_engine.prune_stale_workers(timeout_seconds=5.0)

        # Measure FPS
        dt = time.time() - t0
        self.fps = 1.0 / dt if dt > 0 else 15.0

        # Step 6: Render visual overlays
        annotated_frame = self.draw_overlays(frame, enriched_workers, zones, active_violations_map)

        # Calculate telemetry summary
        total_workers = len(enriched_workers)
        compliant_workers = sum(1 for w in enriched_workers if len(w.get("active_violations", [])) == 0)
        active_viols_count = sum(len(w.get("active_violations", [])) for w in enriched_workers)
        helmet_viols = sum(1 for w in enriched_workers if "NO_HELMET" in w.get("active_violations", []))
        vest_viols = sum(1 for w in enriched_workers if "NO_VEST" in w.get("active_violations", []))
        zone_viols = sum(1 for w in enriched_workers if "ZONE_INTRUSION" in w.get("active_violations", []))

        stats = {
            "active_workers": total_workers,
            "compliant_workers": compliant_workers,
            "active_violations": active_viols_count,
            "helmet_violations": helmet_viols,
            "vest_violations": vest_viols,
            "zone_intrusions": zone_viols,
            "fps": round(self.fps, 1),
            "compliance_rate": round((compliant_workers / total_workers * 100), 1) if total_workers > 0 else 100.0
        }

        return annotated_frame, enriched_workers, new_incidents, resolved_incidents, stats

    def draw_overlays(
        self,
        frame: np.ndarray,
        workers: List[Dict[str, Any]],
        zones: List[Dict[str, Any]],
        active_violations_map: Dict[int, List[str]]
    ) -> np.ndarray:
        """
        Draws semi-transparent polygon zones, worker bounding boxes, foot anchor dots,
        PPE badges, and status banners.
        """
        out = frame.copy()
        h_frame, w_frame = frame.shape[:2]

        # 1. Draw Semi-Transparent Polygon Zones
        if zones:
            overlay = out.copy()
            for z in zones:
                coords = z.get("coordinates")
                if not coords:
                    continue
                if isinstance(coords, str):
                    try:
                        coords = json.loads(coords)
                    except Exception:
                        continue
                if len(coords) < 3:
                    continue

                pts = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
                sev = z.get("severity", "DANGER").upper()

                if sev == "DANGER":
                    fill_col = (0, 0, 180)     # Red
                    line_col = (0, 0, 255)
                elif sev == "WARNING":
                    fill_col = (0, 140, 255)   # Amber
                    line_col = (0, 180, 255)
                else:
                    fill_col = (40, 140, 40)   # Green
                    line_col = (50, 220, 50)

                cv2.fillPoly(overlay, [pts], fill_col)
                cv2.polylines(out, [pts], isClosed=True, color=line_col, thickness=2)

                # Zone label
                p0 = tuple(coords[0])
                cv2.putText(out, f"[{sev}] {z.get('name', 'ZONE')}", (int(p0[0]), int(p0[1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_col, 2, cv2.LINE_AA)

            # Alpha blend zone fill
            cv2.addWeighted(overlay, 0.28, out, 0.72, 0, out)

        # 2. Draw Worker Tracks & Overlays
        for w in workers:
            x1, y1, x2, y2 = w["bbox"]
            track_id = w["track_id"]
            violations = active_violations_map.get(track_id, [])

            is_compliant = len(violations) == 0
            box_color = (46, 204, 113) if is_compliant else (40, 40, 235)  # Green vs Red

            # Main Bounding Box
            cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)

            # Draw Foot Anchor Dot (P_feet)
            foot_pt = compute_foot_anchor(w["bbox"])
            cv2.circle(out, foot_pt, 4, (0, 240, 255), -1)  # Yellow foot anchor dot
            cv2.circle(out, foot_pt, 6, (0, 0, 0), 1)

            # Draw Associated PPE item boxes
            if w.get("has_helmet") and w.get("helmet_bbox"):
                hx1, hy1, hx2, hy2 = w["helmet_bbox"]
                cv2.rectangle(out, (hx1, hy1), (hx2, hy2), (50, 220, 50), 1)
            if w.get("has_vest") and w.get("vest_bbox"):
                vx1, vy1, vx2, vy2 = w["vest_bbox"]
                cv2.rectangle(out, (vx1, vy1), (vx2, vy2), (0, 220, 255), 1)

            # Header Badge: ID | Helmet | Vest
            h_tag = "H:OK" if w.get("has_helmet") else "H:NO"
            v_tag = "V:OK" if w.get("has_vest") else "V:NO"
            z_tag = " | ZONE" if w.get("is_in_zone") else ""
            badge_text = f"ID:{track_id} | {h_tag} {v_tag}{z_tag}"

            badge_w = max(160, len(badge_text) * 8 + 10)
            badge_h = 22
            by1 = max(0, y1 - badge_h)
            cv2.rectangle(out, (x1, by1), (x1 + badge_w, y1), box_color, -1)
            cv2.putText(out, badge_text, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

            # Violation Banner below box
            if violations:
                banner_text = "! " + ", ".join(violations)
                banner_w = max(160, len(banner_text) * 7 + 8)
                cv2.rectangle(out, (x1, y2), (x1 + banner_w, y2 + 18), (0, 0, 200), -1)
                cv2.putText(out, banner_text, (x1 + 4, y2 + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Top-left HUD Dashboard
        hud_w = 320
        hud_h = 32
        cv2.rectangle(out, (10, 10), (10 + hud_w, 10 + hud_h), (12, 16, 24), -1)
        cv2.rectangle(out, (10, 10), (10 + hud_w, 10 + hud_h), (45, 55, 75), 1)
        hud_info = f"FPS: {self.fps:.1f} | Workers: {len(workers)} | Alerts: {len(new_incidents)}"
        cv2.putText(out, hud_info, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 255), 1, cv2.LINE_AA)

        return out
