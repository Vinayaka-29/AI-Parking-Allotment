"""OpenCV parking-space proposals from visible geometric markings.

This module is deliberately independent from vehicle detection and occupancy
state. Proposals are unverified until a caller reviews and saves them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class ParkingSpaceProposalResult:
    status: str
    image_width: int
    image_height: int
    proposals: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "image": {"width": self.image_width, "height": self.image_height},
            "proposals": self.proposals,
        }


def infer_vehicle_slots(
    frame: np.ndarray,
    detections: list[Any],
    *,
    min_detections_per_row: int = 2,
) -> list[dict[str, Any]]:
    """Infer rectangular bays from vehicle centers and the gaps between them.

    This is deliberately conservative: a row needs at least two vehicles so
    the observed spacing provides evidence for an unmarked bay.
    """
    if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
        return []
    height, width = frame.shape[:2]
    boxes: list[list[float]] = []
    for detection in detections:
        bbox = getattr(detection, "bbox", None)
        if bbox is None and isinstance(detection, dict):
            bbox = detection.get("bbox")
        if bbox is None or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = map(float, bbox[:4])
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
    if len(boxes) < min_detections_per_row:
        return []

    median_height = float(np.median([box[3] - box[1] for box in boxes]))
    row_tolerance = max(24.0, median_height * 1.5)
    rows: list[list[list[float]]] = []
    for box in sorted(boxes, key=lambda item: (item[1] + item[3], item[0])):
        center_y = (box[1] + box[3]) / 2.0
        row = next(
            (candidate for candidate in rows
             if abs((candidate[0][1] + candidate[0][3]) / 2.0 - center_y) <= row_tolerance),
            None,
        )
        if row is None:
            rows.append([box])
        else:
            row.append(box)

    proposals: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if len(row) < min_detections_per_row:
            continue
        row.sort(key=lambda item: (item[0] + item[2]) / 2.0)
        centers = np.asarray([(box[0] + box[2]) / 2.0 for box in row], dtype=np.float32)
        widths = np.asarray([box[2] - box[0] for box in row], dtype=np.float32)
        heights = np.asarray([box[3] - box[1] for box in row], dtype=np.float32)
        gaps = np.diff(centers)
        valid_gaps = gaps[gaps >= max(float(np.median(widths)) * 1.2, 8.0)]
        if not len(valid_gaps):
            continue
        pitch = float(np.median(valid_gaps))
        bay_width = float(np.clip(np.median(widths) * 1.35, pitch * 0.55, pitch * 0.95))
        bay_height = float(np.median(heights) * 1.45)
        y_center = float(np.median([(box[1] + box[3]) / 2.0 for box in row]))
        start = max(bay_width / 2.0, float(centers[0]))
        end = min(width - bay_width / 2.0, float(centers[-1]))
        count = max(1, int(round((end - start) / pitch)) + 1)
        for slot_index in range(count):
            center_x = start + slot_index * pitch
            nearest_distance = float(np.min(np.abs(centers - center_x)))
            occupied = nearest_distance <= pitch * 0.35
            x1 = max(0.0, center_x - bay_width / 2.0)
            x2 = min(float(width), center_x + bay_width / 2.0)
            y1 = max(0.0, y_center - bay_height / 2.0)
            y2 = min(float(height), y_center + bay_height / 2.0)
            proposals.append({
                "slot_id": f"{chr(65 + row_index)}{slot_index + 1:02d}",
                "section_id": chr(65 + row_index),
                "polygon": [[round(x1 / width, 6), round(y1 / height, 6)],
                            [round(x2 / width, 6), round(y1 / height, 6)],
                            [round(x2 / width, 6), round(y2 / height, 6)],
                            [round(x1 / width, 6), round(y2 / height, 6)]],
                "type": "STANDARD",
                "priority": 1,
                "distance_from_entries": round((slot_index + 1) * 3 + row_index * 10, 2),
                "confidence": 0.78 if occupied else 0.66,
                "evidence": "vehicle-row" if occupied else "vehicle-gap",
                "status": "NEEDS_REVIEW",
            })
    return proposals


def infer_parking_spaces(frame: np.ndarray, detections: list[Any]) -> ParkingSpaceProposalResult:
    """Combine line evidence and vehicle-row geometry for both input modes."""
    line_result = propose_parking_spaces(frame)
    vehicle_proposals = infer_vehicle_slots(frame, detections)
    combined = list(vehicle_proposals)
    for candidate in line_result.proposals:
        if not any(_polygon_iou(candidate["polygon"], slot["polygon"]) >= 0.4 for slot in combined):
            line_candidate = dict(candidate)
            line_candidate["slot_id"] = f"L{len(combined) + 1:02d}"
            line_candidate["section_id"] = "L"
            line_candidate["evidence"] = "line-only"
            combined.append(line_candidate)
    return ParkingSpaceProposalResult(
        "PROPOSALS_FOUND" if combined else "NO_PROPOSALS",
        line_result.image_width,
        line_result.image_height,
        combined,
    )


def propose_parking_spaces(
    frame: np.ndarray,
    *,
    min_area_ratio: float = 0.002,
    max_area_ratio: float = 0.35,
    duplicate_iou: float = 0.85,
) -> ParkingSpaceProposalResult:
    """Propose quadrilateral bays supported by bright closed line geometry.

    The detector does not inspect YOLO detections and never fills gaps with a
    generated grid. A frame with insufficient geometric evidence returns no
    proposals.
    """
    if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
        return ParkingSpaceProposalResult("NO_PROPOSALS", 0, 0, [])

    height, width = frame.shape[:2]
    if width < 1 or height < 1:
        return ParkingSpaceProposalResult("NO_PROPOSALS", width, height, [])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bright = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), dtype=np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(bright, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, Any]] = []
    image_area = float(width * height)

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        polygon = cv2.approxPolyDP(contour, 0.04 * perimeter, True).reshape(-1, 2)
        area = abs(float(cv2.contourArea(polygon)))
        area_ratio = area / image_area
        if len(polygon) != 4 or area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        normalized = [[round(float(x) / width, 6), round(float(y) / height, 6)] for x, y in polygon]
        validation = validate_polygon(normalized)
        if not validation["valid"]:
            continue

        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
        support = float(cv2.mean(gray, mask=mask)[0]) / 255.0
        confidence = round(min(0.99, max(0.05, 0.55 * support + 0.45 * validation["geometry_score"])), 3)
        candidates.append({
            "polygon": normalized,
            "confidence": confidence,
            "source": "opencv_bright_geometry",
            "evidence": {
                "line_support": round(support, 3),
                "geometry_score": validation["geometry_score"],
                "area_ratio": round(area_ratio, 6),
            },
            "status": "NEEDS_REVIEW",
        })

    proposals = suppress_duplicate_proposals(candidates, duplicate_iou)
    status = "PROPOSALS_FOUND" if proposals else "NO_PROPOSALS"
    return ParkingSpaceProposalResult(status, width, height, proposals)


def validate_polygon(polygon: list[list[float]], *, min_area: float = 0.0001) -> dict[str, Any]:
    """Validate a normalized polygon and return a simple geometry score."""
    if len(polygon) < 3:
        return {"valid": False, "geometry_score": 0.0}
    points = np.asarray(polygon, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        return {"valid": False, "geometry_score": 0.0}
    if np.any(points < 0.0) or np.any(points > 1.0):
        return {"valid": False, "geometry_score": 0.0}
    area = abs(float(cv2.contourArea(points)))
    if area < min_area or not cv2.isContourConvex(points):
        return {"valid": False, "geometry_score": 0.0}
    perimeter = cv2.arcLength(points, True)
    if perimeter <= 0:
        return {"valid": False, "geometry_score": 0.0}
    compactness = min(1.0, 4.0 * np.pi * area / (perimeter * perimeter))
    return {"valid": True, "geometry_score": round(float(max(0.0, compactness)), 3)}


def suppress_duplicate_proposals(
    proposals: list[dict[str, Any]], iou_threshold: float = 0.85
) -> list[dict[str, Any]]:
    """Keep the highest-confidence proposal for heavily overlapping polygons."""
    kept: list[dict[str, Any]] = []
    for proposal in sorted(proposals, key=lambda item: item.get("confidence", 0.0), reverse=True):
        if any(_polygon_iou(proposal["polygon"], other["polygon"]) >= iou_threshold for other in kept):
            continue
        kept.append(proposal)
    return kept


def _polygon_iou(first: list[list[float]], second: list[list[float]]) -> float:
    first_points = np.asarray(first, dtype=np.float32)
    second_points = np.asarray(second, dtype=np.float32)
    first_area = abs(float(cv2.contourArea(first_points)))
    second_area = abs(float(cv2.contourArea(second_points)))
    if first_area <= 0 or second_area <= 0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(first_points, second_points)
    union = first_area + second_area - float(intersection)
    return float(intersection) / union if union > 0 else 0.0
