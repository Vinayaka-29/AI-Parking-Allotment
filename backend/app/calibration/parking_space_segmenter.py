"""Optional parking-space segmentation model adapter.

The adapter is intentionally weight-file driven. It never downloads weights
and never treats the vehicle detector as a parking-space segmentation model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class ParkingSpaceSegmenter:
    """Run an optional Ultralytics segmentation model when verified weights exist."""

    def __init__(self, weights_path: str | None = None) -> None:
        self.weights_path = Path(weights_path).expanduser() if weights_path else None
        self._model: Any | None = None
        self.reason = "No parking-space segmentation weights configured"
        if self.weights_path and self.weights_path.is_file():
            try:
                from ultralytics import YOLO
                self._model = YOLO(str(self.weights_path))
                self.reason = ""
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                self.reason = f"Segmentation model unavailable: {exc}"
        elif self.weights_path:
            self.reason = f"Segmentation weights not found: {self.weights_path}"

    @property
    def available(self) -> bool:
        return self._model is not None

    def predict(self, frame: np.ndarray) -> list[list[list[float]]]:
        """Return normalized masks as polygons, or an explicit empty result."""
        if not self.available:
            return []
        results = self._model.predict(frame, verbose=False)
        polygons: list[list[list[float]]] = []
        height, width = frame.shape[:2]
        for result in results:
            masks = getattr(result, "masks", None)
            if masks is None:
                continue
            for polygon in masks.xy:
                if len(polygon) >= 3:
                    polygons.append(
                        [[round(float(x) / width, 6), round(float(y) / height, 6)] for x, y in polygon]
                    )
        return polygons
