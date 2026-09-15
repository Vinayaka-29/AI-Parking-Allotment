from __future__ import annotations

from typing import Any


class CentroidTracker:
    """Small dependency-free tracker suitable for the prototype and tests."""

    def __init__(self, max_missing: int = 10, max_distance: float = 80.0) -> None:
        self.max_missing = max_missing
        self.max_distance = max_distance
        self.next_id = 1
        self.tracks: dict[int, dict[str, Any]] = {}

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unmatched = set(self.tracks)
        updated: list[dict[str, Any]] = []
        for detection in detections:
            bbox = detection.get("bbox") or [0, 0, 0, 0]
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            track_id = self._match(center, unmatched)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
            unmatched.discard(track_id)
            item = dict(detection)
            item["tracking_id"] = track_id
            item["_center"] = center
            self.tracks[track_id] = {"center": center, "missing": 0}
            updated.append(item)

        for track_id in unmatched:
            self.tracks[track_id]["missing"] += 1
            if self.tracks[track_id]["missing"] > self.max_missing:
                del self.tracks[track_id]
        return updated

    def _match(self, center: tuple[float, float], candidates: set[int]) -> int | None:
        best_id = None
        best_distance = self.max_distance
        for track_id in candidates:
            previous = self.tracks[track_id]["center"]
            distance = ((center[0] - previous[0]) ** 2 + (center[1] - previous[1]) ** 2) ** 0.5
            if distance < best_distance:
                best_id, best_distance = track_id, distance
        return best_id
