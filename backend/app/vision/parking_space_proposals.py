"""OpenCV & geometric parking-space proposals with perspective and row inference.

Architecture: DIVIDER-FIRST
1. Extract candidate line segments with full metadata
2. Classify as STALL_DIVIDER vs BOUNDARY_LINE based on length, regularity, spacing
3. Pair dividers into parking bays (left+right divider = one slot)
4. Use vehicles only for occupancy classification within line-defined bays
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


@dataclass
class LineSegment:
    """A detected line segment with full geometric metadata."""
    x1: float
    y1: float
    x2: float
    y2: float
    length: float
    angle_deg: float
    dx_per_dy: float  # perspective slant
    is_divider: bool = False  # classified later

    @property
    def bottom(self) -> tuple[float, float]:
        return (self.x2, self.y2) if self.y2 > self.y1 else (self.x1, self.y1)

    @property
    def top(self) -> tuple[float, float]:
        return (self.x1, self.y1) if self.y1 < self.y2 else (self.x2, self.y2)

    @property
    def mid_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def mid_x(self) -> float:
        return (self.x1 + self.x2) / 2.0


def _extract_line_segments(frame: np.ndarray) -> list[LineSegment]:
    """Extract all candidate line segments with full metadata from white/yellow masks."""
    if frame is None or frame.size == 0:
        return []

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 60, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([15, 80, 150]), np.array([35, 255, 255]))
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Morphological cleanup to connect broken line segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(combined_mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=15, maxLineGap=10)

    segments: list[LineSegment] = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < 15.0:
                continue
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            # Keep roughly vertical lines (parking dividers and boundaries)
            if not (30.0 <= angle <= 150.0):
                continue

            # Ensure consistent orientation: top -> bottom
            if y1 < y2:
                top_x, top_y, bot_x, bot_y = x1, y1, x2, y2
            else:
                top_x, top_y, bot_x, bot_y = x2, y2, x1, y1

            dy = bot_y - top_y
            if dy < 10.0:
                continue
            dx_per_dy = (top_x - bot_x) / dy

            segments.append(LineSegment(
                x1=top_x, y1=top_y, x2=bot_x, y2=bot_y,
                length=length, angle_deg=angle, dx_per_dy=dx_per_dy
            ))
    return segments


def _classify_dividers(segments: list[LineSegment], img_height: int, img_width: int) -> list[LineSegment]:
    """
    Classify segments as STALL_DIVIDER vs BOUNDARY_LINE.

    Heuristics for stall dividers:
    - Short to medium length (15-120px typically, not full image height)
    - Regular horizontal spacing with neighbors
    - Not touching image borders (usually)
    - Similar slant to nearby segments (consistent perspective)

    Heuristics for boundary lines:
    - Very long (span large fraction of image height)
    - Touch or near image left/right borders
    - Isolated (no regular neighbors)
    """
    if not segments:
        return []

    # Sort by bottom x-coordinate for spatial analysis
    segments.sort(key=lambda s: s.bottom[0])

    # Compute spacing regularity for each segment
    for i, seg in enumerate(segments):
        left_gap = seg.bottom[0] - segments[i-1].bottom[0] if i > 0 else float('inf')
        right_gap = segments[i+1].bottom[0] - seg.bottom[0] if i < len(segments)-1 else float('inf')
        min_gap = min(left_gap, right_gap)

        # Heuristics (RELAXED for real-world CCTV)
        is_short = seg.length < img_height * 0.6        # was 0.4 - allow longer faded lines
        is_regular_spacing = 10.0 < min_gap < 250.0     # was 15-150 - wider range for perspective
        not_on_border = 10.0 < seg.bottom[0] < img_width - 10.0  # was 20px - allow closer to edge
        consistent_slant = True

        # Check slant consistency with neighbors (more tolerant)
        if i > 0:
            consistent_slant &= abs(seg.dx_per_dy - segments[i-1].dx_per_dy) < 0.5  # was 0.3
        if i < len(segments)-1:
            consistent_slant &= abs(seg.dx_per_dy - segments[i+1].dx_per_dy) < 0.5

        # Strong divider candidate - require at least 2 of 4 criteria
        score = sum([is_short, is_regular_spacing, not_on_border, consistent_slant])
        if score >= 2:
            seg.is_divider = True

    return segments


def _pair_dividers_into_bays(dividers: list[LineSegment], img_height: int, img_width: int) -> list[dict[str, Any]]:
    """
    Pair adjacent dividers into parking bays.
    Each bay = left divider + right divider.
    Returns list of bay dicts with polygon vertices.
    """
    if len(dividers) < 2:
        return []

    bays = []
    for i in range(len(dividers) - 1):
        left = dividers[i]
        right = dividers[i + 1]

        span = right.bottom[0] - left.bottom[0]
        # RELAXED bay width constraints for perspective
        if not (15.0 <= span <= 300.0):  # was 20-200
            continue

        # Both dividers should be at similar vertical positions (same row)
        y_diff = abs(left.mid_y - right.mid_y)
        if y_diff > img_height * 0.25:  # was 0.15 - allow more perspective variation
            continue

        # Perspective-aware polygon construction
        # Bottom edge: follows divider bottoms
        y_bot = max(left.bottom[1], right.bottom[1])
        # Top edge: project up using average slant
        dy = max(50.0, img_height * 0.2)
        y_top = max(0.0, y_bot - dy)
        s_avg = (left.dx_per_dy + right.dx_per_dy) / 2.0
        dx = s_avg * dy

        # Normalized coordinates [0, 1]
        p1 = [max(0.0, min(1.0, (left.bottom[0] + dx) / img_width)), y_top / img_height]
        p2 = [max(0.0, min(1.0, (right.bottom[0] + dx) / img_width)), y_top / img_height]
        p3 = [max(0.0, min(1.0, right.bottom[0] / img_width)), y_bot / img_height]
        p4 = [max(0.0, min(1.0, left.bottom[0] / img_width)), y_bot / img_height]

        bays.append({
            "left_divider": left,
            "right_divider": right,
            "polygon": [p1, p2, p3, p4],
            "span": span,
            "mid_x": (left.bottom[0] + right.bottom[0]) / 2.0,
            "mid_y": y_bot,
        })

    return bays


def _assign_vehicles_to_bays(bays: list[dict], boxes: list[dict]) -> list[dict]:
    """
    For each bay, determine if occupied by checking vehicle centers.
    Updates bay dict with occupancy info.
    """
    for bay in bays:
        poly_pts = np.array([
            [p[0] * 1920, p[1] * 1080] for p in bay["polygon"]  # approximate denorm for containment test
        ], dtype=np.float32)

        occupied = False
        vehicle_id = None
        best_conf = 0.0

        for box in boxes:
            x1, y1, x2, y2 = box["bbox"]
            cx = (x1 + x2) / 2.0
            cy = y2  # bottom center of vehicle

            # Quick bbox check first
            if cv2.pointPolygonTest(poly_pts, (cx, cy), False) >= 0:
                occupied = True
                vehicle_id = box.get("class_name", "car")
                best_conf = max(best_conf, box.get("confidence", 0.85))

        bay["occupied"] = occupied
        bay["vehicle_id"] = vehicle_id
        bay["confidence"] = round(best_conf if occupied else 0.88, 2)

    return bays


def infer_parking_spaces(
    frame: np.ndarray,
    detections: list[Any],
    *,
    min_detections_per_row: int = 2,
) -> ParkingSpaceProposalResult:
    """
    DIVIDER-FIRST pipeline:
    1. Extract all line segments
    2. Classify stall dividers vs boundary lines
    3. Pair dividers into bays (geometric ground truth)
    4. Use vehicles only for occupancy classification
    """
    if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
        return ParkingSpaceProposalResult("NO_PROPOSALS", 0, 0, [])

    height, width = frame.shape[:2]
    if width < 1 or height < 1:
        return ParkingSpaceProposalResult("NO_PROPOSALS", width, height, [])

    # Parse detections
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

    # 1. Extract line segments with full metadata
    segments = _extract_line_segments(frame)

    # 2. Classify dividers
    segments = _classify_dividers(segments, height, width)
    dividers = [s for s in segments if s.is_divider]

    # 3. Pair dividers into bays - this IS the slot geometry
    bays = _pair_dividers_into_bays(dividers, height, width)

    if not bays:
        # No divider pairs found - fallback to line-only contour detection
        return propose_parking_spaces(frame)

    # 4. Assign vehicles to bays for occupancy
    bays = _assign_vehicles_to_bays(bays, boxes)

    # 5. Build proposals
    proposals = []
    # Group bays by vertical position (rows)
    bays.sort(key=lambda b: b["mid_y"])

    row_groups: list[list[dict]] = []
    for bay in bays:
        matched = False
        for row in row_groups:
            row_y = np.median([b["mid_y"] for b in row])
            if abs(bay["mid_y"] - row_y) < height * 0.08:
                row.append(bay)
                matched = True
                break
        if not matched:
            row_groups.append([bay])

    row_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for row_idx, row in enumerate(row_groups):
        row.sort(key=lambda b: b["mid_x"])
        sec = row_letters[row_idx % len(row_letters)]

        for slot_idx, bay in enumerate(row, start=1):
            slot_id = f"{sec}{slot_idx:02d}"
            evidence = "vehicle-row" if bay["occupied"] else "line-pair"

            proposals.append({
                "slot_id": slot_id,
                "section_id": sec,
                "polygon": bay["polygon"],
                "type": "STANDARD",
                "priority": 1,
                "distance_from_entries": round(float((slot_idx * 3) + (row_idx * 10)), 2),
                "confidence": bay["confidence"],
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
    """
    Fallback when no vehicles detected: propose slots from line pairs only.
    Uses the same divider-first logic but without vehicle occupancy.
    """
    if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
        return ParkingSpaceProposalResult("NO_PROPOSALS", 0, 0, [])

    height, width = frame.shape[:2]
    if width < 1 or height < 1:
        return ParkingSpaceProposalResult("NO_PROPOSALS", width, height, [])

    segments = _extract_line_segments(frame)
    segments = _classify_dividers(segments, height, width)
    dividers = [s for s in segments if s.is_divider]
    bays = _pair_dividers_into_bays(dividers, height, width)

    if not bays:
        # Last resort: contour-based rectangles (very conservative)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = float(width * height)
        proposals = []
        for c in contours:
            peri = cv2.arcLength(c, True)
            if peri > 50:
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
                            "confidence": 0.75,
                            "evidence": "contour-fallback",
                            "status": "NEEDS_REVIEW",
                        })
        if proposals:
            kept = suppress_duplicate_proposals(proposals, duplicate_iou)
            return ParkingSpaceProposalResult("PROPOSALS_FOUND", width, height, kept)
        return ParkingSpaceProposalResult("NO_PROPOSALS", width, height, [])

    # Build proposals from bays (no vehicles = all available)
    proposals = []
    bays.sort(key=lambda b: b["mid_y"])
    row_groups = []
    for bay in bays:
        matched = False
        for row in row_groups:
            row_y = np.median([b["mid_y"] for b in row])
            if abs(bay["mid_y"] - row_y) < height * 0.08:
                row.append(bay)
                matched = True
                break
        if not matched:
            row_groups.append([bay])

    row_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for row_idx, row in enumerate(row_groups):
        row.sort(key=lambda b: b["mid_x"])
        sec = row_letters[row_idx % len(row_letters)]
        for slot_idx, bay in enumerate(row, start=1):
            slot_id = f"{sec}{slot_idx:02d}"
            proposals.append({
                "slot_id": slot_id,
                "section_id": sec,
                "polygon": bay["polygon"],
                "type": "STANDARD",
                "priority": 1,
                "distance_from_entries": round(float((slot_idx * 3) + (row_idx * 10)), 2),
                "confidence": 0.85,
                "evidence": "line-pair",
                "status": "NEEDS_REVIEW",
            })

    kept = suppress_duplicate_proposals(proposals, duplicate_iou)
    return ParkingSpaceProposalResult(
        "PROPOSALS_FOUND" if kept else "NO_PROPOSALS",
        width, height, kept
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