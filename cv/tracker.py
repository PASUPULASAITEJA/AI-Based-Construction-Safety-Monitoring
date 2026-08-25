"""
CV Module: Worker Tracker
Maintains persistent anonymous worker IDs across video frames using IoU & Spatial Association.
"""
import numpy as np

def compute_iou(box1, box2):
    """Computes IoU between box1 [x1, y1, x2, y2] and box2 [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    return inter_area / union_area

class TrackedWorker:
    def __init__(self, track_id, bbox, confidence):
        self.track_id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.hits = 1
        self.time_since_update = 0
        self.history = [bbox]

    def update(self, bbox, confidence):
        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        self.history.append(bbox)
        if len(self.history) > 30:
            self.history.pop(0)

    def mark_missed(self):
        self.time_since_update += 1

class WorkerTracker:
    def __init__(self, max_age=20, min_hits=1, iou_threshold=0.20):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks = {} # track_id -> TrackedWorker

    def reset(self):
        self.next_id = 1
        self.tracks = {}

    def update(self, person_detections):
        """
        Updates worker tracks with new person detections.
        person_detections: list of dicts with 'bbox' and 'confidence'.
        Returns a list of active tracked workers:
        [
            {
                "worker_id": int,
                "label": "Worker #X",
                "bbox": [x1, y1, x2, y2],
                "confidence": float
            }, ...
        ]
        """
        # Age all current tracks
        for t in self.tracks.values():
            t.mark_missed()

        matched_detections = set()
        matched_tracks = set()

        if len(self.tracks) > 0 and len(person_detections) > 0:
            # Build IoU matrix
            track_ids = list(self.tracks.keys())
            iou_matrix = np.zeros((len(track_ids), len(person_detections)), dtype=np.float32)

            for i, tid in enumerate(track_ids):
                t_box = self.tracks[tid].bbox
                for j, det in enumerate(person_detections):
                    iou_matrix[i, j] = compute_iou(t_box, det["bbox"])

            # Greedy bipartite matching
            while True:
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                i, j = max_idx
                max_iou = iou_matrix[i, j]

                if max_iou < self.iou_threshold:
                    break

                tid = track_ids[i]
                if tid not in matched_tracks and j not in matched_detections:
                    self.tracks[tid].update(person_detections[j]["bbox"], person_detections[j]["confidence"])
                    matched_tracks.add(tid)
                    matched_detections.add(j)

                iou_matrix[i, :] = -1
                iou_matrix[:, j] = -1

        # Create new tracks for unmatched detections
        for j, det in enumerate(person_detections):
            if j not in matched_detections:
                new_track = TrackedWorker(self.next_id, det["bbox"], det["confidence"])
                self.tracks[self.next_id] = new_track
                self.next_id += 1

        # Remove dead tracks
        dead_ids = [tid for tid, t in self.tracks.items() if t.time_since_update > self.max_age]
        for tid in dead_ids:
            del self.tracks[tid]

        # Prepare active results
        active_workers = []
        for tid, t in self.tracks.items():
            if t.time_since_update == 0: # Only return workers detected in current frame
                active_workers.append({
                    "worker_id": tid,
                    "label": f"Worker #{tid}",
                    "bbox": t.bbox,
                    "confidence": t.confidence
                })

        return active_workers
