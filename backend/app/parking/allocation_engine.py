from __future__ import annotations

from typing import Any


class AllocationEngine:
    def select_best_slot(self, available_slots: list[dict[str, Any]], vehicle_type: str = "car") -> dict[str, Any]:
        if not available_slots:
            raise ValueError("No available slots")

        def score(slot: dict[str, Any]) -> float:
            distance = float(slot.get("distance_from_entries", 999))
            distance_score = 1 / max(1, distance)
            priority_score = min(1.0, float(slot.get("priority", 0)) / 5)
            compatible = slot.get("compatible_vehicle_types")
            type_score = 1.0 if not compatible or vehicle_type.lower() in {
                str(item).lower() for item in compatible
            } or slot.get("type", "STANDARD").lower() in {vehicle_type.lower(), "standard"} else 0.0
            return 0.55 * distance_score + 0.25 * priority_score + 0.20 * type_score

        best_slot = max(available_slots, key=score)

        return {
            "slot_id": best_slot["slot_id"],
            "score": round(score(best_slot), 2),
            "reasons": ["available", "shortest configured distance", "priority match"],
            "distance": best_slot.get("distance_from_entries", 0),
            "route": ["ENTRY", "LANE_2", best_slot["section_id"], best_slot["slot_id"]],
            "estimated_distance": best_slot.get("distance_from_entries", 0),
            "estimated_route": ["ENTRY", "LANE_2", best_slot["section_id"], best_slot["slot_id"]],
        }
