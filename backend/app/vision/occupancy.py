from __future__ import annotations

from typing import Any
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

        # Convert slot polygon to absolute pixel coordinates if normalized (0..1)
        poly = []
        if image_shape:
            h, w = image_shape
            for pt in raw_poly:
                px = pt[0] * w if 0 <= pt[0] <= 1.0 else pt[0]
                py = pt[1] * h if 0 <= pt[1] <= 1.0 else pt[1]
                poly.append([px, py])
        else:
            poly = raw_poly

        # Calculate bounding box of the slot polygon: [sx1, sy1, sx2, sy2]
        poly_xs = [p[0] for p in poly]
        poly_ys = [p[1] for p in poly]
        sx1, sy1, sx2, sy2 = min(poly_xs), min(poly_ys), max(poly_xs), max(poly_ys)
        slot_area = max(1.0, (sx2 - sx1) * (sy2 - sy1))

        best_score = 0.0
        best_conf = 0.0

        for det in detections:
            bbox = getattr(det, "bbox", None) or (det.get("bbox") if isinstance(det, dict) else None)
            if not bbox or len(bbox) < 4:
                continue

            conf = getattr(det, "confidence", 0.0) if hasattr(det, "confidence") else det.get("confidence", 0.0)

            # Intersection between slot bounding box and vehicle detection box
            vx1, vy1, vx2, vy2 = bbox
            ix1 = max(sx1, vx1)
            iy1 = max(sy1, vy1)
            ix2 = min(sx2, vx2)
            iy2 = min(sy2, vy2)

            if ix2 > ix1 and iy2 > iy1:
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                vehicle_area = max(1.0, (vx2 - vx1) * (vy2 - vy1))
                # Compute IoU / overlap ratio relative to slot
                overlap_ratio = intersection_area / min(slot_area, vehicle_area)
                score = overlap_ratio * conf
                if score > best_score:
                    best_score = score
                    best_conf = conf

        is_occupied = best_score >= self.iou_threshold
        status = "OCCUPIED" if is_occupied else "AVAILABLE"
        confidence = round(max(0.75, min(0.99, best_conf if is_occupied else (1.0 - best_score))), 2)

        return {
            "slot_id": slot["slot_id"],
            "status": status,
            "confidence": confidence,
            "occupancy_score": round(best_score, 3),
        }
