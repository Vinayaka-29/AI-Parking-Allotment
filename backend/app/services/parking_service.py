from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.parking.state_manager import ParkingStateManager


class ParkingService:
    def __init__(self) -> None:
        self.state_manager = ParkingStateManager()

    def initialize_demo_state(self) -> None:
        config_path = Path(__file__).resolve().parents[3] / "configs" / "parking_layout.json"
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        self.state_manager.load_layout(config)

    def get_overview(self) -> dict[str, Any]:
        return self.state_manager.get_overview()

    def get_slots(self) -> list[dict[str, Any]]:
        return self.state_manager.get_slots()

    def get_recent_events(self) -> list[dict[str, Any]]:
        return self.state_manager.get_recent_events()

    def get_system_status(self) -> dict[str, Any]:
        return {
            "status": "healthy",
            "camera_count": 1,
            "total_slots": len(self.state_manager.slots),
            "occupied_slots": sum(1 for slot in self.state_manager.slots.values() if slot["status"] == "OCCUPIED"),
            "available_slots": sum(1 for slot in self.state_manager.slots.values() if slot["status"] == "AVAILABLE"),
        }

    def allocate_slot(self, vehicle_id: str, vehicle_type: str = "car") -> dict[str, Any]:
        return self.state_manager.allocate_slot(vehicle_id, vehicle_type)


parking_service = ParkingService()
