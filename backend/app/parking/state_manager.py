from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any

from app.parking.allocation_engine import AllocationEngine


class ParkingStateManager:
    def __init__(self) -> None:
        self.slots: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.allocations: dict[str, str] = {}
        self.reservations: dict[str, dict[str, Any]] = {}
        self._lock = RLock()
        self.allocation_engine = AllocationEngine()

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
                "neighboring_slots": raw_slot.get("neighboring_slots", []),
                "reservation_id": None,
                "vehicle_id": None,
            }
            self.slots[slot["slot_id"]] = slot

    def update_slot_status(self, slot_id: str, status: str, confidence: float = 0.95) -> None:
        allowed = {"AVAILABLE", "RESERVED", "OCCUPIED", "TEMPORARILY_OCCUPIED", "UNKNOWN", "OUT_OF_SERVICE"}
        if status not in allowed:
            raise ValueError(f"Invalid slot status: {status}")
        with self._lock:
            if slot_id not in self.slots:
                raise ValueError(f"Unknown slot: {slot_id}")
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
        self.expire_reservations()
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
        with self._lock:
            self.expire_reservations()
            candidates = [slot for slot in self.slots.values() if slot["status"] == "AVAILABLE"]
            recommendation = self.allocation_engine.select_best_slot(candidates, vehicle_type)
            candidate = self.slots[recommendation["slot_id"]]
            reservation_id = f"RES-{len(self.reservations) + 1:04d}"
            expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
            self.allocations[vehicle_id] = candidate["slot_id"]
            self.reservations[reservation_id] = {
                "reservation_id": reservation_id,
                "vehicle_id": vehicle_id,
                "slot_id": candidate["slot_id"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "expires_at": expires_at.isoformat(),
                "status": "ACTIVE",
            }
            candidate["reservation_id"] = reservation_id
            candidate["vehicle_id"] = vehicle_id
            self.update_slot_status(candidate["slot_id"], "RESERVED", 0.92)
        self.events.append({
            "event_id": f"EVT-{len(self.events) + 1:04d}",
            "slot_id": candidate["slot_id"],
            "vehicle_id": vehicle_id,
            "event_type": "VEHICLE_ASSIGNED",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "RESERVED",
            "confidence": 0.92,
            "reservation_id": reservation_id,
        })
        return {
            "vehicle_id": vehicle_id,
            "slot_id": candidate["slot_id"],
            "status": "RESERVED",
            "distance": recommendation["estimated_distance"],
            "reason": recommendation["reason"],
            "score": recommendation["score"],
            "estimated_route": recommendation["estimated_route"],
            "reservation_id": reservation_id,
            "expires_at": expires_at.isoformat(),
        }

    def update_from_occupancy(self, updates: list[dict[str, Any]]) -> None:
        for update in updates:
            slot = self.slots.get(update["slot_id"])
            if slot and slot["status"] == "RESERVED" and update["status"] == "AVAILABLE":
                continue
            self.update_slot_status(update["slot_id"], update["status"], update.get("confidence", 0.0))

    def get_allocations(self) -> list[dict[str, Any]]:
        self.expire_reservations()
        return list(self.reservations.values())

    def get_vehicles(self) -> list[dict[str, Any]]:
        vehicles = []
        for vehicle_id, slot_id in self.allocations.items():
            vehicles.append({"vehicle_id": vehicle_id, "slot_id": slot_id, "status": self.slots[slot_id]["status"]})
        return vehicles

    def mark_vehicle_parked(self, vehicle_id: str, slot_id: str) -> None:
        with self._lock:
            if slot_id not in self.slots or self.slots[slot_id].get("vehicle_id") != vehicle_id:
                raise ValueError("Vehicle is not assigned to this slot")
            reservation_id = self.slots[slot_id].get("reservation_id")
            if reservation_id in self.reservations:
                self.reservations[reservation_id]["status"] = "FULFILLED"
            self.update_slot_status(slot_id, "OCCUPIED", 0.98)
            self.events.append(self._event(slot_id, "VEHICLE_PARKED", vehicle_id, 0.98))

    def release_slot(self, slot_id: str, vehicle_id: str | None = None) -> None:
        with self._lock:
            if slot_id not in self.slots:
                raise ValueError(f"Unknown slot: {slot_id}")
            current_vehicle = self.slots[slot_id].get("vehicle_id")
            if vehicle_id is not None and current_vehicle not in {None, vehicle_id}:
                raise ValueError("Vehicle does not own this slot")
            self.update_slot_status(slot_id, "AVAILABLE", 0.97)
            self.slots[slot_id]["vehicle_id"] = None
            self.slots[slot_id]["reservation_id"] = None
            self.events.append(self._event(slot_id, "SLOT_RELEASED", vehicle_id, 0.97))

    def expire_reservations(self) -> None:
        now = datetime.now(timezone.utc)
        for reservation in list(self.reservations.values()):
            if reservation["status"] == "ACTIVE" and datetime.fromisoformat(reservation["expires_at"]) <= now:
                reservation["status"] = "EXPIRED"
                self.release_slot(reservation["slot_id"], reservation["vehicle_id"])
                self.events.append(self._event(reservation["slot_id"], "RESERVATION_TIMEOUT", reservation["vehicle_id"], 1.0))

    def _event(self, slot_id: str, event_type: str, vehicle_id: str | None, confidence: float) -> dict[str, Any]:
        return {
            "event_id": f"EVT-{len(self.events) + 1:04d}",
            "slot_id": slot_id,
            "vehicle_id": vehicle_id,
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence": confidence,
        }
