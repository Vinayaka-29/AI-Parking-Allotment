from __future__ import annotations

from typing import Any


class SlotOccupancyEngine:
    def __init__(self, persistence_frames: int = 2, missing_frames: int = 2) -> None:
        self.slot_states: dict[str, str] = {}
        self._pending: dict[str, tuple[str, int]] = {}
        self.persistence_frames = persistence_frames
        self.missing_frames = missing_frames

    def compute_slot_status(self, slot: dict[str, Any], detections: list[dict[str, Any]]) -> dict[str, Any]:
        occupancy_score = 0.0
        for detection in detections:
            bbox = detection.get("bbox") or [0, 0, 0, 0]
            if len(bbox) >= 4:
                overlap = self._calculate_overlap(slot.get("polygon", []), bbox)
                occupancy_score = max(occupancy_score, overlap * (detection.get("confidence", 0.0) or 0.0))

        observed = "OCCUPIED" if occupancy_score >= 0.5 else "AVAILABLE"
        previous = self.slot_states.get(slot["slot_id"])
        status = previous if previous is not None else ("UNKNOWN" if observed == "OCCUPIED" else "AVAILABLE")
        confidence = min(0.99, max(0.68, occupancy_score)) if observed == "OCCUPIED" else 0.97
        if previous != observed:
            prior_observation, count = self._pending.get(slot["slot_id"], (observed, 0))
            count = count + 1 if prior_observation == observed else 1
            self._pending[slot["slot_id"]] = (observed, count)
            required = self.missing_frames if observed == "AVAILABLE" and previous == "OCCUPIED" else self.persistence_frames
            if count >= required:
                status = observed
                self.slot_states[slot["slot_id"]] = observed
                self._pending.pop(slot["slot_id"], None)
            elif previous == "OCCUPIED" and observed == "AVAILABLE":
                status = "UNKNOWN"
                confidence = 0.45
        else:
            self._pending.pop(slot["slot_id"], None)
            self.slot_states[slot["slot_id"]] = observed

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
        polygon_x = [point[0] for point in polygon]
        polygon_y = [point[1] for point in polygon]
        px1, py1, px2, py2 = min(polygon_x), min(polygon_y), max(polygon_x), max(polygon_y)
        intersection = max(0.0, min(x2, px2) - max(x1, px1)) * max(0.0, min(y2, py2) - max(y1, py1))
        return min(1.0, intersection / area)
