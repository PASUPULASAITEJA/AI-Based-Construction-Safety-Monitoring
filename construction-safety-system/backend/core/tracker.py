"""
Multi-Object Tracking (ByteTrack Algorithm)
Maintains persistent integer track_ids across video frames, occlusions, and momentary detection dropouts.
"""
from typing import List, Dict, Any, Tuple
import numpy as np

def compute_iou(boxA: List[int], boxB: List[int]) -> float:
    """Computes standard Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return max(0.0, min(1.0, iou))


class Tracklet:
    def __init__(self, track_id: int, bbox: List[int], confidence: float):
        self.track_id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.state = "ACTIVE"
        # Velocity estimation [vx, vy, vw, vh]
        self.velocity = [0.0, 0.0, 0.0, 0.0]

    def predict(self):
        """Predicts next bounding box position based on constant velocity."""
        self.age += 1
        if self.time_since_update > 0:
            self.hits = max(0, self.hits - 1)
        self.time_since_update += 1

        # Apply velocity smoothing
        x1, y1, x2, y2 = self.bbox
        vx, vy, vw, vh = self.velocity
        w = (x2 - x1) + vw
        h = (y2 - y1) + vh
        x1_new = x1 + vx
        y1_new = y1 + vy
        self.bbox = [int(x1_new), int(y1_new), int(x1_new + w), int(y1_new + h)]
        return self.bbox

    def update(self, bbox: List[int], confidence: float):
        """Updates tracklet state with a newly associated detection."""
        # Estimate velocity
        old_x1, old_y1, old_x2, old_y2 = self.bbox
        new_x1, new_y1, new_x2, new_y2 = bbox

        alpha = 0.6  # velocity smoothing factor
        self.velocity = [
            alpha * (new_x1 - old_x1) + (1 - alpha) * self.velocity[0],
            alpha * (new_y1 - old_y1) + (1 - alpha) * self.velocity[1],
            alpha * ((new_x2 - new_x1) - (old_x2 - old_x1)) + (1 - alpha) * self.velocity[2],
            alpha * ((new_y2 - new_y1) - (old_y2 - old_y1)) + (1 - alpha) * self.velocity[3]
        ]

        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        self.state = "ACTIVE"


class ByteTrackTracker:
    def __init__(self, max_age: int = 30, min_hits: int = 2, iou_threshold: float = 0.30):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[Tracklet] = []
        self.next_id = 1

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ingests worker detections and returns tracklets with persistent track_id.
        """
        # Step 1: Predict positions for existing trackers
        for trk in self.trackers:
            trk.predict()

        # Split detections into high confidence (primary) and low confidence (secondary)
        high_dets = [d for d in detections if d.get("confidence", 0) >= 0.35]
        low_dets = [d for d in detections if d.get("confidence", 0) < 0.35]

        # Stage 1: Associate high confidence detections with existing tracks
        matched_trks, unmatched_trks, unmatched_high_dets = self._match(self.trackers, high_dets, self.iou_threshold)

        for trk_idx, det_idx in matched_trks:
            self.trackers[trk_idx].update(high_dets[det_idx]["bbox"], high_dets[det_idx]["confidence"])

        # Stage 2: Associate remaining tracks with low confidence detections (ByteTrack principle)
        active_unmatched_trks = [self.trackers[i] for i in unmatched_trks if self.trackers[i].time_since_update <= 3]
        matched_low, _, _ = self._match(active_unmatched_trks, low_dets, self.iou_threshold)

        for trk_idx, det_idx in matched_low:
            active_unmatched_trks[trk_idx].update(low_dets[det_idx]["bbox"], low_dets[det_idx]["confidence"])

        # Stage 3: Initiate new tracks for unmatched high confidence detections
        for det_idx in unmatched_high_dets:
            det = high_dets[det_idx]
            new_track = Tracklet(self.next_id, det["bbox"], det["confidence"])
            self.next_id += 1
            self.trackers.append(new_track)

        # Stage 4: Prune dead tracks
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        # Stage 5: Emit validated active tracks
        active_results = []
        for t in self.trackers:
            if t.hits >= self.min_hits or t.time_since_update == 0:
                active_results.append({
                    "track_id": t.track_id,
                    "bbox": t.bbox,
                    "confidence": t.confidence,
                    "class_name": "worker"
                })

        return active_results

    def _match(self, trackers: List[Tracklet], detections: List[Dict[str, Any]], thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy IoU bipartite matching."""
        if len(trackers) == 0:
            return [], [], list(range(len(detections)))
        if len(detections) == 0:
            return [], list(range(len(trackers))), []

        # Cost matrix: IoU
        iou_matrix = np.zeros((len(trackers), len(detections)), dtype=np.float32)
        for t_idx, trk in enumerate(trackers):
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = compute_iou(trk.bbox, det["bbox"])

        matched_indices = []
        unmatched_trackers = set(range(len(trackers)))
        unmatched_detections = set(range(len(detections)))

        while len(unmatched_trackers) > 0 and len(unmatched_detections) > 0:
            best_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            best_iou = iou_matrix[best_idx]
            if best_iou < thresh:
                break
            trk_i, det_j = best_idx
            matched_indices.append((trk_i, det_j))
            unmatched_trackers.discard(trk_i)
            unmatched_detections.discard(det_j)
            iou_matrix[trk_i, :] = -1
            iou_matrix[:, det_j] = -1

        return matched_indices, list(unmatched_trackers), list(unmatched_detections)
