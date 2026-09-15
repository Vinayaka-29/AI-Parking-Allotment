from __future__ import annotations

from typing import Any


class SlotOccupancyEngine:
    def __init__(self) -> None:
        self.slot_states: dict[str, str] = {}

    def compute_slot_status(self, slot: dict[str, Any], detections: list[dict[str, Any]]) -> dict[str, Any]:
        occupancy_score = 0.0
        for detection in detections:
            bbox = detection.get("bbox") or [0, 0, 0, 0]
            if len(bbox) >= 4:
                overlap = self._calculate_overlap(slot.get("polygon", []), bbox)
                occupancy_score = max(occupancy_score, overlap * (detection.get("confidence", 0.0) or 0.0))

        status = "AVAILABLE"
        confidence = 0.97
        if occupancy_score >= 0.5:
            status = "OCCUPIED"
            confidence = min(0.99, max(0.68, occupancy_score))

        return {"slot_id": slot["slot_id"], "status": status, "confidence": round(confidence, 2)}

    def _calculate_overlap(self, polygon: list[list[float]], bbox: list[float]) -> float:
        if not polygon or len(bbox) < 4:
            return 0.0
        x1, y1, x2, y2 = bbox
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height
        if area <= 0:
            return 0.0
        polygon_points = polygon
        polygon_area = 0.0
        for idx in range(len(polygon_points)):
            x1p, y1p = polygon_points[idx]
            x2p, y2p = polygon_points[(idx + 1) % len(polygon_points)]
            polygon_area += (x1p * y2p - x2p * y1p)
        polygon_area = abs(polygon_area) / 2.0
        return min(1.0, area / max(1.0, polygon_area))
