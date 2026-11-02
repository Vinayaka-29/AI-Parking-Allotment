from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app.vision.parking_space_proposals import infer_parking_spaces, propose_parking_spaces


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
    """Infers parking bays and perspective layout from vehicle evidence and markings."""

    def __init__(self, total_cycles: int = 3, min_parked: int = 2) -> None:
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

    def calibrate_instant(self, frame: np.ndarray, detections: list[Any]) -> dict[str, Any]:
        """One-shot instant perspective calibration from a single frame or uploaded image."""
        self.reference_frame = frame.copy()
        height, width = frame.shape[:2]
        result = infer_parking_spaces(frame, detections)

        if not result.proposals:
            proposal = {
                "status": "NEEDS_MORE_DATA",
                "message": "Could not identify clear vehicle rows or stall lines in this scene.",
                "polygons": [],
                "quality": {"number_of_rows": 0, "parked_candidates": len(detections), "inferred_slots": 0, "coverage": 0.0},
                "image_width": width,
                "image_height": height,
            }
        else:
            sections = set(slot.get("section_id", "A") for slot in result.proposals)
            proposal = {
                "status": "PROPOSED",
                "message": f"Proposed {len(result.proposals)} perspective-calibrated bays across {len(sections)} sections.",
                "polygons": result.proposals,
                "quality": {
                    "number_of_rows": len(sections),
                    "parked_candidates": len(detections),
                    "inferred_slots": sum(slot.get("evidence") == "inferred-empty" for slot in result.proposals),
                    "coverage": 1.0,
                    "requires_review": False,
                },
                "image_width": width,
                "image_height": height,
            }

        self.proposal = proposal
        self.status = proposal["status"]
        self.message = proposal["message"]
        self.cycles_done = self.total_cycles
        return self.proposal

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
            if best_index is not None and best_score >= 0.5:
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

        # Build proposal using tracks
        self.proposal = self._build_proposal(frame, detections)
        self.status = self.proposal["status"]
        self.message = self.proposal["message"]
        return self.proposal

    @staticmethod
    def _bbox(detection: Any) -> list[float] | None:
        value = getattr(detection, "bbox", None)
        if value is None and isinstance(detection, dict):
            value = detection.get("bbox")
        return value if value and len(value) >= 4 else None

    def _build_proposal(self, frame: np.ndarray, current_detections: list[Any] | None = None) -> dict[str, Any]:
        height, width = frame.shape[:2]
        # Use stationary tracks + current detections
        det_candidates = current_detections if current_detections is not None else [
            {"bbox": track.bbox, "confidence": 0.85, "class_name": "car"}
            for track in self.tracks if track.hits >= 1
        ]
        result = infer_parking_spaces(frame, det_candidates)
        if result.proposals:
            sections = set(slot.get("section_id", "A") for slot in result.proposals)
            return {
                "status": "PROPOSED",
                "message": f"Proposed {len(result.proposals)} perspective slots across {len(sections)} rows.",
                "polygons": result.proposals,
                "quality": {
                    "number_of_rows": len(sections),
                    "parked_candidates": len(self.tracks),
                    "inferred_slots": sum(slot.get("evidence") == "inferred-empty" for slot in result.proposals),
                    "coverage": 1.0,
                },
                "image_width": width,
                "image_height": height,
            }

        return {
            "status": "NEEDS_MORE_DATA",
            "message": f"Insufficient spatial patterns found to calibrate bays.",
            "polygons": [],
            "quality": {"number_of_rows": 0, "parked_candidates": len(self.tracks), "inferred_slots": 0, "coverage": 0.0},
            "image_width": width,
            "image_height": height,
        }
