from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class VehicleDetection:
    tracking_id: int | None = None
    class_name: str = "car"
    confidence: float = 0.0
    bbox: list[float] | None = None
    camera_id: str = "CAM_01"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class VehicleDetector:
    """Abstract detector interface for a future YOLO implementation."""

    def __init__(self, camera_id: str = "CAM_01") -> None:
        self.camera_id = camera_id

    def detect(self, frame: Any) -> list[VehicleDetection]:
        # Placeholder implementation used for the first milestone.
        return []
