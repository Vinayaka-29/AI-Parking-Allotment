from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class ParkingStateManager:
    def __init__(self) -> None:
        self.slots: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.allocations: dict[str, str] = {}

    def load_layout(self, config: dict[str, Any]) -> None:
        self.slots = {}
        for raw_slot in config.get("slots", []):
            slot = {
                "slot_id": raw_slot["slot_id"],
                "section_id": raw_slot.get("section_id", "A"),
                "camera_id": raw_slot.get("camera_id", "CAM_01"),
                "polygon": raw_slot.get("polygon", []),
                "status": raw_slot.get("status", "AVAILABLE"),
                "type": raw_slot.get("type", "STANDARD"),
                "priority": raw_slot.get("priority", 1),
                "distance_from_entries": raw_slot.get("distance_from_entries", 0),
                "confidence": 0.97,
            }
            self.slots[slot["slot_id"]] = slot

    def update_slot_status(self, slot_id: str, status: str, confidence: float = 0.95) -> None:
        if slot_id not in self.slots:
            return
        self.slots[slot_id]["status"] = status
        self.slots[slot_id]["confidence"] = confidence
        self.events.append({
            "event_id": f"EVT-{len(self.events) + 1:04d}",
            "slot_id": slot_id,
            "event_type": "SLOT_STATUS_CHANGED",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "confidence": confidence,
        })

    def get_overview(self) -> dict[str, Any]:
        total = len(self.slots)
        available = sum(1 for slot in self.slots.values() if slot["status"] == "AVAILABLE")
        occupied = sum(1 for slot in self.slots.values() if slot["status"] == "OCCUPIED")
        reserved = sum(1 for slot in self.slots.values() if slot["status"] == "RESERVED")
        unknown = sum(1 for slot in self.slots.values() if slot["status"] == "UNKNOWN")
        occupancy = round((occupied / total) * 100, 2) if total else 0.0
        return {
            "total_slots": total,
            "available": available,
            "occupied": occupied,
            "reserved": reserved,
            "unknown": unknown,
            "occupancy_pct": occupancy,
        }

    def get_slots(self) -> list[dict[str, Any]]:
        return list(self.slots.values())

    def get_recent_events(self) -> list[dict[str, Any]]:
        return self.events[-10:]

    def allocate_slot(self, vehicle_id: str, vehicle_type: str = "car") -> dict[str, Any]:
        candidate = None
        for slot in self.slots.values():
            if slot["status"] == "AVAILABLE":
                if candidate is None or slot["distance_from_entries"] < candidate["distance_from_entries"]:
                    candidate = slot
        if candidate is None:
            raise ValueError("Parking Full")

        self.allocations[vehicle_id] = candidate["slot_id"]
        self.update_slot_status(candidate["slot_id"], "RESERVED", 0.92)
        self.events.append({
            "event_id": f"EVT-{len(self.events) + 1:04d}",
            "slot_id": candidate["slot_id"],
            "vehicle_id": vehicle_id,
            "event_type": "VEHICLE_ASSIGNED",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "RESERVED",
            "confidence": 0.92,
        })
        return {
            "vehicle_id": vehicle_id,
            "slot_id": candidate["slot_id"],
            "status": "RESERVED",
            "distance": candidate["distance_from_entries"],
            "reason": "Nearest available slot",
        }
