"""OpenCV & geometric parking-space proposals with perspective and row inference.

Detects vehicle rows, estimates perspective vanishing geometry, extracts line dividers,
interpolates empty bays between parked vehicles, and produces perspective-aligned
quadrilateral slot polygons.
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


def _extract_slanted_lines(frame: np.ndarray) -> list[tuple[float, float, float]]:
    """Extract stall divider line segments and their perspective slants (dx_per_dy)."""
    if frame is None or frame.size == 0:
        return []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 60, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([15, 80, 150]), np.array([35, 255, 255]))
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

    edges = cv2.Canny(combined_mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=20, maxLineGap=12)

    slanted_lines: list[tuple[float, float, float]] = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length >= 20.0:
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if 45.0 <= angle <= 135.0:
                    top_x, top_y = (x1, y1) if y1 < y2 else (x2, y2)
                    bot_x, bot_y = (x2, y2) if y1 < y2 else (x1, y1)
                    dy = bot_y - top_y
                    if dy >= 15.0:
                        dx_per_dy = (top_x - bot_x) / dy
                        slanted_lines.append((float(bot_x), float(bot_y), float(dx_per_dy)))
    return slanted_lines


def _get_perspective_slant(
    x: float,
    y: float,
    width: int,
    height: int,
    slanted_lines: list[tuple[float, float, float]],
) -> float:
    """Compute how much X shifts toward top per 1px dy at (x, y) under perspective."""
    if slanted_lines:
        weights = [1.0 / ((np.hypot(x - lx, y - ly) + 1.0) ** 2) for lx, ly, _ in slanted_lines]
        total_w = sum(weights)
        if total_w > 0:
            return float(sum(s * w for (_, _, s), w in zip(slanted_lines, weights)) / total_w)

    vx, vy = width * 0.52, -height * 0.55
    dy = y - vy
    return float((vx - x) / dy) if dy > 0 else 0.0


def infer_parking_spaces(
    frame: np.ndarray,
    detections: list[Any],
    *,
    min_detections_per_row: int = 2,
) -> ParkingSpaceProposalResult:
    """Infers parking bays, perspective geometry, and empty spaces from vehicles and lines.

    Works even when parking lines are missing by using vehicle footprint lattice grid
    and vanishing perspective modeling.
    """
    if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
        return ParkingSpaceProposalResult("NO_PROPOSALS", 0, 0, [])

    height, width = frame.shape[:2]
    if width < 1 or height < 1:
        return ParkingSpaceProposalResult("NO_PROPOSALS", width, height, [])

    boxes: list[dict[str, Any]] = []
    for detection in detections:
        bbox = getattr(detection, "bbox", None)
        if bbox is None and isinstance(detection, dict):
            bbox = detection.get("bbox")
        if bbox is None or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = map(float, bbox[:4])
        if x2 > x1 and y2 > y1:
            conf = getattr(detection, "confidence", 0.85) if hasattr(detection, "confidence") else detection.get("confidence", 0.85)
            class_name = getattr(detection, "class_name", "car") if hasattr(detection, "class_name") else detection.get("class_name", "car")
            boxes.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf or 0.85),
                "class_name": str(class_name or "car"),
            })

    slanted_lines = _extract_slanted_lines(frame)

    if len(boxes) < min_detections_per_row:
        return propose_parking_spaces(frame)

    # 1. Perspective Row Clustering
    boxes.sort(key=lambda b: b["bbox"][3])
    rows: list[list[dict[str, Any]]] = []

    for b in boxes:
        cy = b["bbox"][3]
        matched_row: list[dict[str, Any]] | None = None
        best_diff = 1e9

        for r in rows:
            row_y = float(np.median([item["bbox"][3] for item in r]))
            row_h = float(np.median([item["bbox"][3] - item["bbox"][1] for item in r]))
            tolerance = max(16.0, row_h * 0.35)
            diff = abs(cy - row_y)
            if diff <= tolerance and diff < best_diff:
                best_diff = diff
                matched_row = r

        if matched_row is not None:
            matched_row.append(b)
        else:
            rows.append([b])

    valid_rows = [r for r in rows if len(r) >= min_detections_per_row]
    valid_rows.sort(key=lambda r: float(np.median([item["bbox"][3] for item in r])))

    proposals: list[dict[str, Any]] = []
    row_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for row_idx, r in enumerate(valid_rows):
        sec = row_letters[row_idx % len(row_letters)]
        r.sort(key=lambda b: (b["bbox"][0] + b["bbox"][2]) / 2.0)

        widths = [b["bbox"][2] - b["bbox"][0] for b in r]
        heights = [b["bbox"][3] - b["bbox"][1] for b in r]
        centers = [(b["bbox"][0] + b["bbox"][2]) / 2.0 for b in r]
        bottoms = [b["bbox"][3] for b in r]

        med_w = float(np.median(widths))
        med_h = float(np.median(heights))
        med_bot = float(np.median(bottoms))

        stall_w = med_w * 1.15
        stall_h = med_h * 1.25
        y2 = med_bot
        y1 = max(0.0, y2 - stall_h)
        dy = y2 - y1

        gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        normal_gaps = [g for g in gaps if 0.75 * med_w <= g <= 1.55 * med_w]
        pitch = float(np.median(normal_gaps)) if normal_gaps else float(med_w * 1.18)

        slot_centers: list[tuple[float, bool, dict[str, Any] | None]] = []

        for i, b in enumerate(r):
            cx = (b["bbox"][0] + b["bbox"][2]) / 2.0
            slot_centers.append((cx, True, b))

            if i < len(r) - 1:
                next_cx = (r[i + 1]["bbox"][0] + r[i + 1]["bbox"][2]) / 2.0
                gap = next_cx - cx
                if gap >= 1.6 * pitch:
                    num_empty = int(round(gap / pitch)) - 1
                    step = gap / (num_empty + 1)
                    for s in range(1, num_empty + 1):
                        empty_cx = cx + s * step
                        slot_centers.append((empty_cx, False, None))

        slot_centers.sort(key=lambda item: item[0])

        for slot_idx, (scx, is_occ, veh) in enumerate(slot_centers, start=1):
            s = _get_perspective_slant(scx, y2, width, height, slanted_lines)
            dx = s * dy

            p1 = [round(max(0.0, min(float(width), scx - stall_w / 2.0 + dx)) / width, 6), round(y1 / height, 6)]
            p2 = [round(max(0.0, min(float(width), scx + stall_w / 2.0 + dx)) / width, 6), round(y1 / height, 6)]
            p3 = [round(max(0.0, min(float(width), scx + stall_w / 2.0)) / width, 6), round(y2 / height, 6)]
            p4 = [round(max(0.0, min(float(width), scx - stall_w / 2.0)) / width, 6), round(y2 / height, 6)]

            slot_id = f"{sec}{slot_idx:02d}"
            confidence = round(float(veh["confidence"]) if veh else 0.88, 2)
            evidence = "vehicle-row" if is_occ else "vehicle-gap"

            proposals.append({
                "slot_id": slot_id,
                "section_id": sec,
                "polygon": [p1, p2, p3, p4],
                "type": "STANDARD",
                "priority": 1,
                "distance_from_entries": round(float((slot_idx * 3) + (row_idx * 10)), 2),
                "confidence": confidence,
                "evidence": evidence,
                "status": "NEEDS_REVIEW",
            })

    status = "PROPOSALS_FOUND" if proposals else "NO_PROPOSALS"
    return ParkingSpaceProposalResult(status, width, height, proposals)


def infer_vehicle_slots(
    frame: np.ndarray,
    detections: list[Any],
    *,
    min_detections_per_row: int = 2,
) -> list[dict[str, Any]]:
    """Legacy alias returning just the proposal list for backwards compatibility."""
    res = infer_parking_spaces(frame, detections, min_detections_per_row=min_detections_per_row)
    return res.proposals


def propose_parking_spaces(
    frame: np.ndarray,
    *,
    min_area_ratio: float = 0.001,
    max_area_ratio: float = 0.35,
    duplicate_iou: float = 0.85,
) -> ParkingSpaceProposalResult:
    """Propose parking slots when vehicles are absent by extracting lane lines."""
    if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
        return ParkingSpaceProposalResult("NO_PROPOSALS", 0, 0, [])

    height, width = frame.shape[:2]
    if width < 1 or height < 1:
        return ParkingSpaceProposalResult("NO_PROPOSALS", width, height, [])

    slanted_lines = _extract_slanted_lines(frame)
    proposals: list[dict[str, Any]] = []

    # If no Hough lines from color masks, try basic edge/contour detection on gray
    if len(slanted_lines) < 2:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = float(width * height)
        for c in contours:
            peri = cv2.arcLength(c, True)
            if peri > 30:
                approx = cv2.approxPolyDP(c, 0.04 * peri, True).reshape(-1, 2)
                area = abs(float(cv2.contourArea(approx)))
                if len(approx) == 4 and min_area_ratio <= area / image_area <= max_area_ratio:
                    normalized = [[round(float(x) / width, 6), round(float(y) / height, 6)] for x, y in approx]
                    if validate_polygon(normalized)["valid"]:
                        proposals.append({
                            "slot_id": f"L{len(proposals) + 1:02d}",
                            "section_id": "L",
                            "polygon": normalized,
                            "type": "STANDARD",
                            "priority": 1,
                            "distance_from_entries": (len(proposals) + 1) * 3,
                            "confidence": 0.82,
                            "evidence": "line-only",
                            "status": "NEEDS_REVIEW",
                        })
        if proposals:
            kept = suppress_duplicate_proposals(proposals, duplicate_iou)
            return ParkingSpaceProposalResult("PROPOSALS_FOUND", width, height, kept)

    # Group lines by approximate horizontal bottom location and height
    slanted_lines.sort(key=lambda item: item[0])

    for i in range(len(slanted_lines) - 1):
        x1_bot, y1_bot, s1 = slanted_lines[i]
        x2_bot, y2_bot, s2 = slanted_lines[i + 1]
        span = x2_bot - x1_bot
        if 25.0 <= span <= 220.0:
            y_bot = max(y1_bot, y2_bot)
            dy = max(60.0, float(height * 0.25))
            y_top = max(0.0, y_bot - dy)
            s_avg = (s1 + s2) / 2.0
            dx = s_avg * dy

            p1 = [round(max(0.0, x1_bot + dx) / width, 6), round(y_top / height, 6)]
            p2 = [round(min(float(width), x2_bot + dx) / width, 6), round(y_top / height, 6)]
            p3 = [round(min(float(width), x2_bot) / width, 6), round(y_bot / height, 6)]
            p4 = [round(max(0.0, x1_bot) / width, 6), round(y_bot / height, 6)]

            proposals.append({
                "slot_id": f"L{len(proposals) + 1:02d}",
                "section_id": "L",
                "polygon": [p1, p2, p3, p4],
                "type": "STANDARD",
                "priority": 1,
                "distance_from_entries": (len(proposals) + 1) * 3,
                "confidence": 0.85,
                "evidence": "line-only",
                "status": "NEEDS_REVIEW",
            })

    kept = suppress_duplicate_proposals(proposals, duplicate_iou)
    return ParkingSpaceProposalResult(
        "PROPOSALS_FOUND" if kept else "NO_PROPOSALS",
        width,
        height,
        kept,
    )


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
