from __future__ import annotations

from typing import Any


class AllocationEngine:
    def select_best_slot(self, available_slots: list[dict[str, Any]], vehicle_type: str = "car") -> dict[str, Any]:
        if not available_slots:
            raise ValueError("No available slots")

        best_slot = min(
            available_slots,
            key=lambda slot: (
                slot.get("distance_from_entries", 999),
                -slot.get("priority", 0),
            ),
        )

        return {
            "slot_id": best_slot["slot_id"],
            "score": round(1.0 / max(1.0, best_slot.get("distance_from_entries", 1)), 2),
            "reason": "Nearest available slot with priority match",
            "estimated_distance": best_slot.get("distance_from_entries", 0),
            "estimated_route": ["ENTRY", "LANE_2", best_slot["section_id"], best_slot["slot_id"]],
        }
