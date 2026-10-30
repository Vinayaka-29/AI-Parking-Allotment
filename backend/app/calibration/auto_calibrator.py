from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app.vision.parking_space_proposals import propose_parking_spaces


def _iou(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, (ax2 - ax1) * (ay2 - ay1)) + max(0.0, (bx2 - bx1) * (by2 - by1)) - intersection
    return intersection / union if union else 0.0


@dataclass
class _Track:
    bbox: list[float]
    hits: int = 1
    cycles_seen: int = 1
    history: list[list[float]] = field(default_factory=list)


class AutoCalibrator:
    """Infers parking bays from stationary vehicle evidence without new YOLO passes."""

    def __init__(self, total_cycles: int = 40, min_parked: int = 3) -> None:
        self.total_cycles = max(1, total_cycles)
        self.min_parked = max(1, min_parked)
        self.reset()

    def reset(self) -> None:
        self.cycles_done = 0
        self.tracks: list[_Track] = []
        self.reference_frame: np.ndarray | None = None
        self.proposal: dict[str, Any] | None = None
        self.message = "Waiting for live frames"
        self.status = "IDLE"

    def update(self, detections: list[Any], frame: np.ndarray) -> dict[str, Any] | None:
        if self.cycles_done == 0:
            self.reference_frame = frame.copy()
            self.status = "CALIBRATING"
        self.cycles_done = min(self.total_cycles, self.cycles_done + 1)
        boxes = [list(map(float, self._bbox(det))) for det in detections if self._bbox(det)]
        matched: set[int] = set()
        for track in self.tracks:
            best_index, best_score = None, 0.0
            for index, box in enumerate(boxes):
                if index in matched:
                    continue
                score = _iou(track.bbox, box)
                if score > best_score:
                    best_index, best_score = index, score
            track.cycles_seen += 1
            if best_index is not None and best_score >= 0.6:
                track.bbox = boxes[best_index]
                track.hits += 1
                track.history.append(boxes[best_index])
                matched.add(best_index)
        for index, box in enumerate(boxes):
            if index not in matched:
                self.tracks.append(_Track(box, history=[box]))
        if self.cycles_done < self.total_cycles:
            self.message = f"CALIBRATING {self.cycles_done}/{self.total_cycles}"
            return None
        self.proposal = self._build_proposal(frame)
        self.status = self.proposal["status"]
        self.message = self.proposal["message"]
        return self.proposal

    @staticmethod
    def _bbox(detection: Any) -> list[float] | None:
        value = getattr(detection, "bbox", None)
        if value is None and isinstance(detection, dict):
            value = detection.get("bbox")
        return value if value and len(value) >= 4 else None

    def _build_proposal(self, frame: np.ndarray) -> dict[str, Any]:
        height, width = frame.shape[:2]
        parked = [
            track for track in self.tracks
            if track.hits / max(track.cycles_seen, 1) >= 0.70
            and self._is_stationary(track, width, height)
        ]
        if len(parked) < self.min_parked:
            return {
                "status": "NEEDS_MORE_DATA",
                "message": f"Only {len(parked)} stationary vehicles found; need at least {self.min_parked}.",
                "polygons": [],
                "quality": {"number_of_rows": 0, "parked_candidates": len(parked), "inferred_slots": 0, "coverage": 0.0},
                "image_width": width,
                "image_height": height,
            }

        rows: list[list[_Track]] = []
        row_tolerance = max(30.0, height * 0.08)
        for track in sorted(parked, key=lambda item: (item.bbox[1] + item.bbox[3], item.bbox[0])):
            center_y = (track.bbox[1] + track.bbox[3]) / 2.0
            row = next((candidate for candidate in rows if abs(candidate[0].bbox[1] + candidate[0].bbox[3] - 2 * center_y) <= row_tolerance), None)
            if row is None:
                rows.append([track])
            else:
                row.append(track)

        polygons: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows):
            row.sort(key=lambda item: (item.bbox[0] + item.bbox[2]) / 2.0)
            widths = [item.bbox[2] - item.bbox[0] for item in row]
            heights = [item.bbox[3] - item.bbox[1] for item in row]
            pitch_values = [
                (right.bbox[0] + right.bbox[2]) / 2.0 - (left.bbox[0] + left.bbox[2]) / 2.0
                for left, right in zip(row, row[1:])
            ]
            pitch = float(np.median([value for value in pitch_values if value > np.median(widths) * 0.7])) if any(
                value > np.median(widths) * 0.7 for value in pitch_values
            ) else float(np.median(widths) * 1.25)
            width_px, height_px = float(np.median(widths) * 1.25), float(np.median(heights) * 1.35)
            centers = [(item.bbox[0] + item.bbox[2]) / 2.0 for item in row]
            start = max(width_px / 2.0, min(centers))
            end = min(width - width_px / 2.0, max(centers))
            count = max(1, int(round((end - start) / pitch)) + 1)
            for slot_index in range(count):
                center_x = start + slot_index * pitch
                nearest = min(row, key=lambda item: abs(((item.bbox[0] + item.bbox[2]) / 2.0) - center_x))
                is_parked = abs(((nearest.bbox[0] + nearest.bbox[2]) / 2.0) - center_x) <= pitch * 0.35
                center_y = (nearest.bbox[1] + nearest.bbox[3]) / 2.0
                x1, x2 = max(0.0, center_x - width_px / 2), min(width, center_x + width_px / 2)
                y1, y2 = max(0.0, center_y - height_px / 2), min(height, center_y + height_px / 2)
                polygons.append({
                    "slot_id": f"{chr(65 + row_index)}{slot_index + 1:02d}",
                    "section_id": chr(65 + row_index),
                    "polygon": [[round(x1 / width, 6), round(y1 / height, 6)],
                                [round(x2 / width, 6), round(y1 / height, 6)],
                                [round(x2 / width, 6), round(y2 / height, 6)],
                                [round(x1 / width, 6), round(y2 / height, 6)]],
                    "type": "STANDARD",
                    "priority": 1,
                    "distance_from_entries": round((slot_index + 1) * 3 + row_index * 10, 2),
                    "confidence": 0.9 if is_parked else 0.65,
                    "evidence": "parked" if is_parked else "inferred",
                })

        line_proposals = propose_parking_spaces(frame).proposals
        for candidate in line_proposals:
            if not any(self._polygon_iou(candidate["polygon"], slot["polygon"]) >= 0.4 for slot in polygons):
                candidate = dict(candidate)
                candidate["confidence"] = min(candidate.get("confidence", 0.35), 0.5)
                candidate["evidence"] = "line-only"
                candidate["slot_id"] = f"L{len(polygons) + 1:02d}"
                candidate["section_id"] = "L"
                polygons.append(candidate)
        coverage = min(1.0, len(polygons) / max(1, len(parked)))
        return {
            "status": "PROPOSED",
            "message": f"Proposed {len(polygons)} slots across {len(rows)} rows.",
            "polygons": polygons,
            "quality": {
                "number_of_rows": len(rows),
                "parked_candidates": len(parked),
                "inferred_slots": sum(item["evidence"] == "inferred" for item in polygons),
                "coverage": round(coverage, 3),
            },
            "image_width": width,
            "image_height": height,
        }

    @staticmethod
    def _is_stationary(track: _Track, width: int, height: int) -> bool:
        centers = np.asarray([[(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0] for box in track.history])
        if len(centers) < 2:
            return True
        spread = np.max(np.abs(centers - np.median(centers, axis=0)), axis=0)
        return bool(spread[0] <= max(5.0, width * 0.01) and spread[1] <= max(5.0, height * 0.01))

    @staticmethod
    def _polygon_iou(first: list[list[float]], second: list[list[float]]) -> float:
        first_np, second_np = np.asarray(first, dtype=np.float32), np.asarray(second, dtype=np.float32)
        first_area, second_area = abs(cv2.contourArea(first_np)), abs(cv2.contourArea(second_np))
        if not first_area or not second_area:
            return 0.0
        intersection, _ = cv2.intersectConvexConvex(first_np, second_np)
        return float(intersection) / (first_area + second_area - float(intersection))
