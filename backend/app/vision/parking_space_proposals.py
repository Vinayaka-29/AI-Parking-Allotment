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
    orientation: str = "vertical"
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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 135]), np.array([180, 90, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([15, 80, 150]), np.array([35, 255, 255]))
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -3
    )
    # Adaptive thresholding is useful for faded paint, but unrestricted use
    # turns reflections, windows, cracks, and crosswalk stripes into dividers.
    # Keep it only where it agrees with a paint-colored/bright region.
    adaptive = cv2.bitwise_and(adaptive, cv2.dilate(white_mask, np.ones((5, 5), np.uint8)))
    combined_mask = cv2.bitwise_or(cv2.bitwise_or(white_mask, yellow_mask), adaptive)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(combined_mask, 60, 170)
    height, width = frame.shape[:2]
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(18, int(min(width, height) * 0.04)),
        minLineLength=max(18, int(height * 0.08)),
        maxLineGap=max(8, int(height * 0.025)),
    )

    segments: list[LineSegment] = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = np.asarray(line).reshape(-1)[:4]
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < 15.0:
                continue
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            # Keep roughly vertical lines (parking dividers and boundaries)
            vertical_fraction = abs(y2 - y1) / max(length, 1.0)
            horizontal_fraction = abs(x2 - x1) / max(length, 1.0)
            if max(vertical_fraction, horizontal_fraction) < 0.82:
                continue
            if vertical_fraction >= horizontal_fraction:
                orientation = "vertical"
                if y1 < y2:
                    top_x, top_y, bot_x, bot_y = x1, y1, x2, y2
                else:
                    top_x, top_y, bot_x, bot_y = x2, y2, x1, y1
                dy = bot_y - top_y
                if dy < 10.0:
                    continue
                dx_per_dy = (top_x - bot_x) / dy
            else:
                orientation = "horizontal"
                if x1 < x2:
                    top_x, top_y, bot_x, bot_y = x1, y1, x2, y2
                else:
                    top_x, top_y, bot_x, bot_y = x2, y2, x1, y1
                dx_per_dy = 0.0

            segments.append(LineSegment(
                x1=top_x, y1=top_y, x2=bot_x, y2=bot_y,
                length=length, angle_deg=angle, dx_per_dy=dx_per_dy,
                orientation=orientation,
            ))
    return _deduplicate_segments(segments)


def _deduplicate_segments(segments: list[LineSegment]) -> list[LineSegment]:
    """Collapse Hough duplicates from both edges of one painted marking."""
    kept: list[LineSegment] = []
    for candidate in sorted(segments, key=lambda item: item.length, reverse=True):
        duplicate = False
        for existing in kept:
            angle_delta = abs(candidate.angle_deg - existing.angle_deg)
            angle_delta = min(angle_delta, 180.0 - angle_delta)
            if angle_delta > 8.0:
                continue
            distance = abs(candidate.mid_x - existing.mid_x)
            y_distance = abs(candidate.mid_y - existing.mid_y)
            if distance < 0.025 * max(1, abs(candidate.x2 - candidate.x1) + abs(existing.x2 - existing.x1)) and y_distance < 0.12 * max(candidate.length, existing.length):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


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

    vertical = [segment for segment in segments if segment.orientation == "vertical"]
    horizontal = [segment for segment in segments if segment.orientation == "horizontal"]
    classified: list[LineSegment] = []
    for group in (vertical, horizontal):
        group.sort(key=lambda s: s.bottom[0] if s.orientation == "vertical" else s.mid_y)
        classified.extend(group)
    segments = classified

    # Compute spacing regularity for each segment
    for i, seg in enumerate(segments):
        same_orientation = [
            item for item in segments
            if item.orientation == seg.orientation
        ]
        position = (
            seg.bottom[0] if seg.orientation == "vertical" else seg.mid_y
        )
        positions = [
            item.bottom[0] if item.orientation == "vertical" else item.mid_y
            for item in same_orientation
        ]
        current_index = positions.index(position)
        left_gap = position - positions[current_index - 1] if current_index > 0 else float("inf")
        right_gap = positions[current_index + 1] - position if current_index < len(positions) - 1 else float("inf")
        min_gap = min(left_gap, right_gap)

        is_reasonable_length = img_height * 0.06 <= seg.length <= img_height * 0.75
        max_gap = 0.45 * (img_width if seg.orientation == "vertical" else img_height)
        is_regular_spacing = 0.02 * min(img_width, img_height) < min_gap < max_gap
        border_position = seg.bottom[0] if seg.orientation == "vertical" else seg.mid_y
        border_limit = img_width if seg.orientation == "vertical" else img_height
        not_on_border = 0.01 * border_limit < border_position < border_limit * 0.99
        consistent_slant = True

        # Check slant consistency with neighbors (more tolerant)
        neighbors = [
            item for item in same_orientation
            if item is not seg
        ]
        if neighbors:
            consistent_slant = any(
                abs(seg.angle_deg - neighbor.angle_deg) < 15.0
                for neighbor in neighbors
            )

        score = sum([is_reasonable_length, is_regular_spacing, not_on_border, consistent_slant])
        if score >= 3:
            seg.is_divider = True

    return segments


def _pair_dividers_into_bays(dividers: list[LineSegment], img_height: int, img_width: int) -> list[dict[str, Any]]:
    """
    Pair adjacent dividers into parking bays.
    Each bay = left divider + right divider.
    Returns list of bay dicts with polygon vertices.
    """
    dividers = [divider for divider in dividers if divider.orientation == "vertical"]
    if len(dividers) < 2:
        return []

    def x_at_y(segment: LineSegment, y: float) -> float:
        top_x, top_y = segment.top
        bottom_x, bottom_y = segment.bottom
        if abs(bottom_y - top_y) < 1e-6:
            return float(top_x)
        ratio = (y - top_y) / (bottom_y - top_y)
        return float(top_x + ratio * (bottom_x - top_x))

    bays = []
    for i in range(len(dividers) - 1):
        left = dividers[i]
        right = dividers[i + 1]
        angle_delta = abs(left.angle_deg - right.angle_deg)
        angle_delta = min(angle_delta, 180.0 - angle_delta)
        if angle_delta > 12.0:
            continue
        y_top = max(left.top[1], right.top[1])
        y_bottom = min(left.bottom[1], right.bottom[1])
        shared_height = y_bottom - y_top
        if shared_height < img_height * 0.08:
            continue
        left_top = x_at_y(left, y_top)
        right_top = x_at_y(right, y_top)
        left_bottom = x_at_y(left, y_bottom)
        right_bottom = x_at_y(right, y_bottom)
        top_width = right_top - left_top
        bottom_width = right_bottom - left_bottom
        if top_width <= 0 or bottom_width <= 0:
            continue
        if not (0.015 * img_width <= bottom_width <= 0.45 * img_width):
            continue
        if not (0.45 <= top_width / bottom_width <= 2.2):
            continue
        polygon_px = np.asarray(
            [[left_top, y_top], [right_top, y_top],
             [right_bottom, y_bottom], [left_bottom, y_bottom]],
            dtype=np.float32,
        )
        area = abs(float(cv2.contourArea(polygon_px)))
        if area < img_width * img_height * 0.00015:
            continue
        bbox_area = max(1.0, float((right_bottom - left_top) * (y_bottom - y_top)))
        if area / bbox_area < 0.45:
            continue
        p1 = [round(float(left_top / img_width), 6), round(float(y_top / img_height), 6)]
        p2 = [round(float(right_top / img_width), 6), round(float(y_top / img_height), 6)]
        p3 = [round(float(right_bottom / img_width), 6), round(float(y_bottom / img_height), 6)]
        p4 = [round(float(left_bottom / img_width), 6), round(float(y_bottom / img_height), 6)]

        bays.append({
            "left_divider": left,
            "right_divider": right,
            "polygon": [p1, p2, p3, p4],
            "span": bottom_width,
            "mid_x": (left_bottom + right_bottom) / 2.0,
            "mid_y": (y_top + y_bottom) / 2.0,
        })

    return bays


def _pair_horizontal_dividers_into_bays(
    dividers: list[LineSegment],
    img_height: int,
    img_width: int,
) -> list[dict[str, Any]]:
    """Pair long horizontal boundaries for parallel/curb-side layouts."""
    lines = [line for line in dividers if line.orientation == "horizontal"]
    lines.sort(key=lambda line: line.mid_y)
    bays: list[dict[str, Any]] = []

    def y_at_x(line: LineSegment, x: float) -> float:
        if abs(line.x2 - line.x1) < 1e-6:
            return float(line.y1)
        ratio = (x - line.x1) / (line.x2 - line.x1)
        return float(line.y1 + ratio * (line.y2 - line.y1))

    for upper, lower in zip(lines, lines[1:]):
        angle_delta = abs(upper.angle_deg - lower.angle_deg)
        angle_delta = min(angle_delta, 180.0 - angle_delta)
        if angle_delta > 12.0:
            continue
        x_start = max(min(upper.x1, upper.x2), min(lower.x1, lower.x2))
        x_end = min(max(upper.x1, upper.x2), max(lower.x1, lower.x2))
        shared_width = x_end - x_start
        if shared_width < img_width * 0.08:
            continue
        upper_start, lower_start = y_at_x(upper, x_start), y_at_x(lower, x_start)
        upper_end, lower_end = y_at_x(upper, x_end), y_at_x(lower, x_end)
        left_height = lower_start - upper_start
        right_height = lower_end - upper_end
        if left_height <= 0 or right_height <= 0:
            continue
        if not (0.45 <= right_height / left_height <= 2.2):
            continue
        polygon_px = np.asarray(
            [[x_start, upper_start], [x_end, upper_end],
             [x_end, lower_end], [x_start, lower_start]],
            dtype=np.float32,
        )
        area = abs(float(cv2.contourArea(polygon_px)))
        if area < img_width * img_height * 0.00015:
            continue
        bbox_area = max(1.0, shared_width * max(left_height, right_height))
        if area / bbox_area < 0.45:
            continue
        bays.append({
            "left_divider": upper,
            "right_divider": lower,
            "polygon": [
                [round(float(x_start / img_width), 6), round(float(upper_start / img_height), 6)],
                [round(float(x_end / img_width), 6), round(float(upper_end / img_height), 6)],
                [round(float(x_end / img_width), 6), round(float(lower_end / img_height), 6)],
                [round(float(x_start / img_width), 6), round(float(lower_start / img_height), 6)],
            ],
            "span": shared_width,
            "mid_x": (x_start + x_end) / 2.0,
            "mid_y": (upper_start + upper_end + lower_start + lower_end) / 4.0,
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


def _vehicle_lattice_bays(
    boxes: list[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    """Create conservative bay polygons from repeated parked-vehicle positions."""
    if len(boxes) < 2:
        return []
    rows: list[list[dict[str, Any]]] = []
    for box in sorted(boxes, key=lambda item: (item["bbox"][3], item["bbox"][0])):
        bottom_y = box["bbox"][3]
        row = next(
            (candidate for candidate in rows
             if abs(bottom_y - np.median([item["bbox"][3] for item in candidate])) < image_height * 0.12),
            None,
        )
        if row is None:
            rows.append([box])
        else:
            row.append(box)

    proposals: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if len(row) < 2:
            continue
        row.sort(key=lambda item: item["bbox"][0])
        widths = [item["bbox"][2] - item["bbox"][0] for item in row]
        heights = [item["bbox"][3] - item["bbox"][1] for item in row]
        centers = [((item["bbox"][0] + item["bbox"][2]) / 2.0) for item in row]
        gaps = np.diff(centers)
        pitch = float(np.median(gaps)) if len(gaps) else float(np.median(widths) * 1.8)
        pitch = max(pitch, float(np.median(widths) * 1.8), 20.0)
        bay_width = max(pitch * 0.8, float(np.median(widths) * 1.8), 20.0)
        bay_height = max(float(np.median(heights) * 1.8), 30.0)
        row_bottom = float(np.median([item["bbox"][3] for item in row]))
        section = chr(ord("A") + row_index % 26)
        first_center = centers[0]
        last_center = centers[-1]
        lattice_centers = np.arange(first_center, last_center + pitch * 0.5, pitch)
        for slot_index, center_x in enumerate(lattice_centers, start=1):
            nearest = min(row, key=lambda item: abs(
                ((item["bbox"][0] + item["bbox"][2]) / 2.0) - center_x
            ))
            nearest_center = (nearest["bbox"][0] + nearest["bbox"][2]) / 2.0
            is_vehicle_slot = abs(nearest_center - center_x) <= pitch * 0.35
            bottom_y = min(float(image_height), row_bottom + bay_height * 0.35)
            top_y = max(0.0, bottom_y - bay_height)
            left = max(0.0, center_x - bay_width / 2.0)
            right = min(float(image_width), center_x + bay_width / 2.0)
            if not is_vehicle_slot:
                nearest = {"confidence": 0.0, "bbox": [center_x, 0, center_x, 0]}
            proposals.append({
                "slot_id": f"{section}{slot_index:02d}",
                "section_id": section,
                "polygon": [
                    [round(left / image_width, 6), round(top_y / image_height, 6)],
                    [round(right / image_width, 6), round(top_y / image_height, 6)],
                    [round(right / image_width, 6), round(bottom_y / image_height, 6)],
                    [round(left / image_width, 6), round(bottom_y / image_height, 6)],
                ],
                "type": "STANDARD",
                "priority": 1,
                "distance_from_entries": round(float(slot_index * 3 + row_index * 10), 2),
                "confidence": round(min(0.75, max(0.45, float(nearest.get("confidence", 0.6)))), 2),
                "evidence": "vehicle-lattice" if is_vehicle_slot else "vehicle-gap",
                "status": "NEEDS_REVIEW",
            })
    return proposals


def infer_parking_spaces(
    frame: np.ndarray,
    detections: list[Any],
    *,
    min_detections_per_row: int = 2,
    allow_vehicle_geometry: bool = False,
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
        bays = _pair_horizontal_dividers_into_bays(dividers, height, width)

    if not bays:
        if allow_vehicle_geometry:
            lattice = _vehicle_lattice_bays(boxes, width, height)
            if lattice:
                return ParkingSpaceProposalResult("PROPOSALS_FOUND", width, height, lattice)
        # Contour rectangles are too easily produced by vehicles, road
        # markings, and compression artifacts. Keep them out of the automatic
        # inference path; callers may explicitly use propose_parking_spaces()
        # for diagnostic/manual workflows.
        return ParkingSpaceProposalResult("NO_PROPOSALS", width, height, [])

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
            # A pedestrian crossing is a repeated-marking field, not a row of
            # bays. Reject it before IDs are assigned so it cannot reach the
            # profile store as a plausible-looking layout.
            if _has_repeated_marking_pattern(frame, bay["polygon"]):
                continue
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
    res = infer_parking_spaces(
        frame,
        detections,
        min_detections_per_row=min_detections_per_row,
        allow_vehicle_geometry=True,
    )
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
        bays = _pair_horizontal_dividers_into_bays(dividers, height, width)

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
    bounds = {
        "min_x": float(points[:, 0].min()),
        "max_x": float(points[:, 0].max()),
        "min_y": float(points[:, 1].min()),
        "max_y": float(points[:, 1].max()),
    }
    return {
        "valid": True,
        "geometry_score": round(float(max(0.0, compactness)), 3),
        "area": area,
        "width": bounds["max_x"] - bounds["min_x"],
        "height": bounds["max_y"] - bounds["min_y"],
        "bounds": bounds,
    }


def validate_proposals(
    proposals: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    *,
    min_slots: int = 2,
    max_overlap: float = 0.10,
    frame: np.ndarray | None = None,
) -> dict[str, Any]:
    """Validate a complete layout before it can be persisted as confirmed.

    Coordinates are normalized. Vehicle-only proposals are deliberately never
    eligible for automatic confirmation because detections do not describe
    parking-space boundaries.
    """
    issues: list[str] = []
    valid: list[dict[str, Any]] = []
    if image_width <= 0 or image_height <= 0:
        return {"confidence": "LOW", "eligible": False, "issues": ["Invalid image dimensions"], "valid_count": 0}
    if len(proposals) < min_slots:
        issues.append(f"Only {len(proposals)} candidate bays were found; at least {min_slots} are required")

    areas: list[float] = []
    widths: list[float] = []
    heights: list[float] = []
    centers: list[tuple[float, float]] = []
    visual_scores: list[float] = []
    for index, proposal in enumerate(proposals):
        geometry = validate_polygon(proposal.get("polygon", []), min_area=0.00015)
        if not geometry["valid"]:
            issues.append(f"Slot {index + 1} has invalid geometry")
            continue
        if proposal.get("evidence") in {"vehicle-lattice", "vehicle-gap"}:
            issues.append("Vehicle positions are supporting evidence only; painted or structural boundaries are required")
        areas.append(float(geometry["area"]))
        widths.append(float(geometry["width"]))
        heights.append(float(geometry["height"]))
        centers.append((
            (geometry["bounds"]["min_x"] + geometry["bounds"]["max_x"]) / 2,
            (geometry["bounds"]["min_y"] + geometry["bounds"]["max_y"]) / 2,
        ))
        if frame is not None:
            visual_scores.append(_polygon_visual_support(frame, proposal["polygon"]))
        valid.append(proposal)

    if areas:
        median_area = float(np.median(areas))
        median_width = float(np.median(widths))
        median_height = float(np.median(heights))
        if median_area <= 0 or median_width <= 0 or median_height <= 0:
            issues.append("Candidate bay dimensions are degenerate")
        if any(area < median_area * 0.25 or area > median_area * 4.0 for area in areas):
            issues.append("Candidate bay areas are inconsistent")
        if any(width < median_width * 0.35 or width > median_width * 3.0 for width in widths):
            issues.append("Candidate bay widths are inconsistent")
        if any(height < median_height * 0.35 or height > median_height * 3.0 for height in heights):
            issues.append("Candidate bay heights are inconsistent")

    for first_index, first in enumerate(valid):
        for second in valid[first_index + 1:]:
            overlap = _polygon_iou(first["polygon"], second["polygon"])
            if overlap > max_overlap:
                issues.append("Candidate bays overlap beyond the allowed threshold")
                break
        if issues and issues[-1] == "Candidate bays overlap beyond the allowed threshold":
            break

    rows: dict[str, list[float]] = {}
    for proposal, (_, center_y) in zip(valid, centers):
        rows.setdefault(str(proposal.get("section_id", "A")), []).append(center_y)
    if rows and len(rows) > 1:
        row_centers = [float(np.median(values)) for values in rows.values()]
        if max(row_centers) - min(row_centers) < 0.02:
            issues.append("Rows are not vertically separated")

    low_confidence = any(float(item.get("confidence", 0.0)) < 0.65 for item in valid)
    if low_confidence:
        issues.append("One or more candidate bays have low geometric confidence")
    if frame is not None and visual_scores:
        weak_visual = sum(score < 0.08 for score in visual_scores)
        if weak_visual:
            issues.append(
                f"{weak_visual} candidate bays lack sufficient painted-line or edge evidence"
            )

    if not valid or issues:
        confidence = "LOW" if any("required" in issue or "invalid" in issue or "Vehicle" in issue for issue in issues) else "MEDIUM"
        return {
            "confidence": confidence,
            "eligible": False,
            "issues": list(dict.fromkeys(issues)),
            "valid_count": len(valid),
            "coverage": round(sum(areas) if areas else 0.0, 4),
        }

    return {
        "confidence": "HIGH",
        "eligible": True,
        "issues": [],
        "valid_count": len(valid),
        "coverage": round(sum(areas), 4),
    }


def _polygon_visual_support(frame: np.ndarray, polygon: list[list[float]]) -> float:
    """Measure edge/marking support along a normalized candidate polygon."""
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        return 0.0
    height, width = frame.shape[:2]
    points = np.asarray(
        [[int(round(x * width)), int(round(y * height))] for x, y in polygon],
        dtype=np.int32,
    )
    if len(points) < 3:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if frame.ndim == 3 else None
    edges = cv2.Canny(gray, 40, 130)
    if hsv is not None:
        bright = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([180, 110, 255]))
        evidence = cv2.bitwise_or(edges, bright)
    else:
        evidence = edges
    evidence = cv2.dilate(evidence, np.ones((3, 3), dtype=np.uint8))
    samples = 0
    supported = 0
    for start, end in zip(points, np.roll(points, -1, axis=0)):
        length = max(2, int(np.linalg.norm(end - start)))
        for ratio in np.linspace(0.05, 0.95, min(80, length)):
            x = int(round(start[0] + ratio * (end[0] - start[0])))
            y = int(round(start[1] + ratio * (end[1] - start[1])))
            if 0 <= x < width and 0 <= y < height:
                samples += 1
                supported += int(evidence[y, x] > 0)
    return supported / samples if samples else 0.0


def _has_repeated_marking_pattern(frame: np.ndarray, polygon: list[list[float]]) -> bool:
    """Detect dense repeated stripes around a candidate (typically a crossing)."""
    height, width = frame.shape[:2]
    points = np.asarray(
        [[int(round(x * width)), int(round(y * height))] for x, y in polygon],
        dtype=np.int32,
    )
    if len(points) < 3:
        return False
    x_min, y_min = np.maximum(points.min(axis=0) - [40, 40], [0, 0])
    x_max, y_max = np.minimum(points.max(axis=0) + [40, 40], [width, height])
    crop = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    edges = cv2.Canny(gray, 70, 180)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(10, min(crop.shape[:2]) // 12),
        minLineLength=max(12, min(crop.shape[:2]) // 4),
        maxLineGap=6,
    )
    if lines is None or len(lines) < 6:
        return False
    angles = []
    for raw in lines:
        x1, y1, x2, y2 = map(float, np.asarray(raw).reshape(-1)[:4])
        angles.append(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
    angles = np.asarray(angles)
    for angle in angles:
        delta = np.abs(((angles - angle + 90.0) % 180.0) - 90.0)
        if int(np.count_nonzero(delta < 8.0)) >= 6:
            return True
    return False


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