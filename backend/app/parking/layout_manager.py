from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


class CameraLayoutManager:
    """Own calibrated parking geometry per camera.

    Demo layouts are loaded only when explicitly requested by a caller.
    """

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self.storage_path = Path(storage_path) if storage_path else None
        self._layouts: dict[str, dict[str, Any]] = {}

    def load_calibrated_layout(self, camera_id: str) -> dict[str, Any] | None:
        if camera_id in self._layouts:
            return deepcopy(self._layouts[camera_id])
        if not self.storage_path or not self.storage_path.exists():
            return None
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            for layout in self._normalise_documents(data):
                if layout["camera_id"] == camera_id:
                    self._layouts[camera_id] = layout
                    return deepcopy(layout)
        except (OSError, ValueError, TypeError):
            return None
        return None

    def save_calibrated_layout(
        self, camera_id: str, slots: list[dict[str, Any]], lot_name: str | None = None
    ) -> dict[str, Any]:
        self._validate_slots(slots)
        previous = self._layouts.get(camera_id)
        version = int(previous.get("layout_version", 0)) + 1 if previous else 1
        layout = {
            "camera_id": camera_id,
            "layout_id": f"{camera_id}_V{version}",
            "layout_version": version,
            "layout_status": "CALIBRATED",
            "lot_name": lot_name or "Calibrated lot",
            "slots": deepcopy(slots),
        }
        for slot in layout["slots"]:
            slot.setdefault("camera_id", camera_id)
        self._layouts[camera_id] = layout
        self._persist()
        return deepcopy(layout)

    def get_layout(self, camera_id: str) -> dict[str, Any] | None:
        layout = self._layouts.get(camera_id)
        return deepcopy(layout) if layout else None

    def clear(self, camera_id: str) -> None:
        self._layouts.pop(camera_id, None)
        self._persist()

    @staticmethod
    def _normalise_documents(data: Any) -> list[dict[str, Any]]:
        documents = data if isinstance(data, list) else [data]
        layouts = []
        for document in documents:
            if not isinstance(document, dict) or not document.get("slots"):
                continue
            camera_id = document.get("camera_id", "CAM_01")
            layouts.append({
                "camera_id": camera_id,
                "layout_id": document.get("layout_id", f"{camera_id}_V1"),
                "layout_version": document.get("layout_version", 1),
                "layout_status": document.get("layout_status", "CALIBRATED"),
                "lot_name": document.get("parking_lot", document.get("lot_name")),
                "slots": document["slots"],
            })
        return layouts

    @staticmethod
    def _validate_slots(slots: list[dict[str, Any]]) -> None:
        seen: set[str] = set()
        for slot in slots:
            slot_id = slot.get("slot_id")
            polygon = slot.get("polygon")
            if not slot_id or slot_id in seen:
                raise ValueError("Every parking slot must have a unique slot_id")
            if not isinstance(polygon, list) or len(polygon) < 3:
                raise ValueError(f"Slot {slot_id} must have a polygon with at least 3 points")
            if any(not isinstance(point, (list, tuple)) or len(point) < 2 for point in polygon):
                raise ValueError(f"Slot {slot_id} contains an invalid polygon point")
            seen.add(slot_id)

    def _persist(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(
            json.dumps(list(self._layouts.values()), indent=2), encoding="utf-8"
        )