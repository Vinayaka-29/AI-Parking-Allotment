from __future__ import annotations

from typing import Any
import cv2
import numpy as np


class SlotOccupancyEngine:
    def __init__(self, iou_threshold: float = 0.20) -> None:
        self.iou_threshold = iou_threshold

    def compute_slot_status(
        self,
        slot: dict[str, Any],
        detections: list[Any],
        image_shape: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """Determine if a slot is OCCUPIED or AVAILABLE based on vehicle detections."""
        raw_poly = slot.get("polygon", [])
        if not raw_poly or len(raw_poly) < 3:
            return {
                "slot_id": slot["slot_id"],
                "status": slot.get("status", "AVAILABLE"),
                "confidence": 0.95,
                "occupancy_score": 0.0,
            }

        poly = []
        if image_shape:
            h, w = image_shape
            ref_w, ref_h = slot.get("calibration_size") or [None, None]
            for pt in raw_poly:
                px = pt[0] * w if 0 <= pt[0] <= 1.0 else pt[0] * w / ref_w if ref_w else pt[0]
                py = pt[1] * h if 0 <= pt[1] <= 1.0 else pt[1] * h / ref_h if ref_h else pt[1]
                poly.append([px, py])
        else:
            poly = raw_poly

        poly_np = np.asarray(poly, dtype=np.float32)
        poly_area = abs(cv2.contourArea(poly_np))
        if poly_area <= 0:
            return {"slot_id": slot["slot_id"], "status": "UNKNOWN", "confidence": 0.0, "occupancy_score": 0.0}

        best_score = 0.0
        best_conf = 0.0

        for det in detections:
            bbox = getattr(det, "bbox", None) or (det.get("bbox") if isinstance(det, dict) else None)
            if not bbox or len(bbox) < 4:
                continue

            conf = getattr(det, "confidence", 0.0) if hasattr(det, "confidence") else det.get("confidence", 0.0)

            vx1, vy1, vx2, vy2 = bbox
            if vx2 <= vx1 or vy2 <= vy1:
                continue
            vehicle_poly = np.asarray([[vx1, vy1], [vx2, vy1], [vx2, vy2], [vx1, vy2]], dtype=np.float32)
            try:
                intersection_area, _ = cv2.intersectConvexConvex(poly_np, vehicle_poly)
            except cv2.error:
                continue
            overlap_ratio = min(1.0, float(intersection_area) / poly_area)
            score = overlap_ratio * float(conf)
            if score > best_score:
                best_score = score
                best_conf = float(conf)

        is_occupied = best_score >= self.iou_threshold
        status = "OCCUPIED" if is_occupied else "AVAILABLE"
        confidence = round(max(0.75, min(0.99, best_conf if is_occupied else (1.0 - best_score))), 2)

        return {
            "slot_id": slot["slot_id"],
            "status": status,
            "confidence": confidence,
            "occupancy_score": round(best_score, 3),
        }
