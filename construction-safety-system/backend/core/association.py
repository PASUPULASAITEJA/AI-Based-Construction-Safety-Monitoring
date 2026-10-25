"""
Anatomical PPE Association Heuristic Module
Maps detected PPE items (helmets and safety vests) to individual tracked workers using
anatomically segmented Head and Torso sub-regions and Intersection-over-Item (IoI) thresholds.
"""
from typing import List, Dict, Any, Tuple
import config

def compute_intersection_over_item(box_item: List[int], roi: List[float]) -> float:
    """
    Computes Intersection over Item: Area(B_item ∩ R) / Area(B_item).
    Quantifies what fraction of the PPE bounding box is contained inside the anatomical ROI.
    """
    ix1 = max(float(box_item[0]), roi[0])
    iy1 = max(float(box_item[1]), roi[1])
    ix2 = min(float(box_item[2]), roi[2])
    iy2 = min(float(box_item[3]), roi[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    item_w = max(1.0, float(box_item[2] - box_item[0]))
    item_h = max(1.0, float(box_item[3] - box_item[1]))
    item_area = item_w * item_h

    return inter_area / item_area


def is_centroid_inside_roi(box_item: List[int], roi: List[float]) -> bool:
    """Checks whether the geometric centroid of a PPE item lies inside the anatomical ROI."""
    cx = (box_item[0] + box_item[2]) / 2.0
    cy = (box_item[1] + box_item[3]) / 2.0
    return (roi[0] <= cx <= roi[2]) and (roi[1] <= cy <= roi[3])


def associate_ppe_to_workers(
    workers: List[Dict[str, Any]], 
    ppe_items: List[Dict[str, Any]],
    head_ratio: float = config.HEAD_ROI_HEIGHT_RATIO,
    torso_top_ratio: float = config.TORSO_ROI_TOP_RATIO,
    torso_bottom_ratio: float = config.TORSO_ROI_BOTTOM_RATIO,
    min_ioi: float = config.MIN_IOI_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Evaluates every tracked worker against detected helmets and vests.
    
    ROI Definitions:
        R_head  = [x1, y1, x2, y1 + 0.28 * H]
        R_torso = [x1, y1 + 0.20 * H, x2, y1 + 0.75 * H]
        
    Association Conditions:
        - Item centroid strictly inside the anatomical ROI
        - IoI >= 0.40
        
    Returns:
        List of enriched worker dicts with:
            has_helmet (bool), helmet_conf (float),
            has_vest (bool), vest_conf (float),
            head_roi (List[float]), torso_roi (List[float])
    """
    helmets = [item for item in ppe_items if item["class_name"] == "helmet"]
    vests = [item for item in ppe_items if item["class_name"] == "vest"]

    enriched_workers = []

    for w in workers:
        x1, y1, x2, y2 = w["bbox"]
        H = float(y2 - y1)

        # 1. Compute Anatomical Sub-regions
        R_head = [float(x1), float(y1), float(x2), float(y1 + head_ratio * H)]
        R_torso = [float(x1), float(y1 + torso_top_ratio * H), float(x2), float(y1 + torso_bottom_ratio * H)]

        # 2. Evaluate Helmet Association
        assigned_helmet = None
        best_helmet_score = 0.0

        for h in helmets:
            if is_centroid_inside_roi(h["bbox"], R_head):
                ioi = compute_intersection_over_item(h["bbox"], R_head)
                if ioi >= min_ioi and ioi > best_helmet_score:
                    best_helmet_score = ioi
                    assigned_helmet = h

        # 3. Evaluate Vest Association
        assigned_vest = None
        best_vest_score = 0.0

        for v in vests:
            if is_centroid_inside_roi(v["bbox"], R_torso):
                ioi = compute_intersection_over_item(v["bbox"], R_torso)
                if ioi >= min_ioi and ioi > best_vest_score:
                    best_vest_score = ioi
                    assigned_vest = v

        # Construct enriched worker object
        enriched = dict(w)
        enriched["has_helmet"] = assigned_helmet is not None
        enriched["helmet_conf"] = float(assigned_helmet["confidence"]) if assigned_helmet else 0.0
        enriched["helmet_bbox"] = assigned_helmet["bbox"] if assigned_helmet else None

        enriched["has_vest"] = assigned_vest is not None
        enriched["vest_conf"] = float(assigned_vest["confidence"]) if assigned_vest else 0.0
        enriched["vest_bbox"] = assigned_vest["bbox"] if assigned_vest else None

        enriched["head_roi"] = [int(v) for v in R_head]
        enriched["torso_roi"] = [int(v) for v in R_torso]

        enriched_workers.append(enriched)

    return enriched_workers
