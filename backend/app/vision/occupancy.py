from __future__ import annotations

from typing import Any
import cv2
import numpy as np


class SlotOccupancyEngine:
    def __init__(self, iou_threshold: float = 0.30, min_confidence: float = 0.25) -> None:
        self.iou_threshold = iou_threshold
        self.min_confidence = min_confidence

    def compute_slot_status(
        self,
        slot: dict[str, Any],
        detections: list[Any],
        image_shape: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """Determine occupancy from a vehicle anchor point or area overlap."""
        raw_poly = slot.get("polygon", [])
        if not raw_poly or len(raw_poly) < 3:
            return {
                "slot_id": slot["slot_id"],
                "status": "UNKNOWN",
                "confidence": 0.0,
                "occupancy_score": 0.0,
            }

        poly = []
        if image_shape:
            h, w = image_shape
            ref_w, ref_h = slot.get("calibration_size") or [None, None]
            for pt in raw_poly:
                px = pt[0] * w if 0 <= pt[0] <= 1.0 else pt[0] * w / ref_w if ref_w else pt[0]
                py = pt[1] * h if 0 <= pt[1] <= 1.0 else pt[1] * h / ref_h if ref_h else pt[1]
                poly.append([px, py])
        else:
            poly = raw_poly

        poly_np = np.asarray(poly, dtype=np.float32)
        poly_area = abs(cv2.contourArea(poly_np))
        if poly_area <= 0:
            return {"slot_id": slot["slot_id"], "status": "UNKNOWN", "confidence": 0.0, "occupancy_score": 0.0}

        best_score = 0.0
        best_conf = 0.0
        best_detection = None

        for det in detections:
            bbox = getattr(det, "bbox", None) or (det.get("bbox") if isinstance(det, dict) else None)
            if not bbox or len(bbox) < 4:
                continue

            conf = getattr(det, "confidence", 0.0) if hasattr(det, "confidence") else det.get("confidence", 0.0)

            conf = float(conf)
            if conf < self.min_confidence:
                continue
            vx1, vy1, vx2, vy2 = bbox
            if vx2 <= vx1 or vy2 <= vy1:
                continue
            vehicle_poly = np.asarray([[vx1, vy1], [vx2, vy1], [vx2, vy2], [vx1, vy2]], dtype=np.float32)
            try:
                intersection_area, _ = cv2.intersectConvexConvex(poly_np, vehicle_poly)
            except cv2.error:
                continue
            bbox_area = float((vx2 - vx1) * (vy2 - vy1))
            overlap_ratio = min(1.0, float(intersection_area) / min(poly_area, bbox_area)) if bbox_area else 0.0
            anchor_x = (vx1 + vx2) / 2.0
            anchor_y = vy2 - ((vy2 - vy1) * 0.15)
            anchor_inside = cv2.pointPolygonTest(poly_np, (anchor_x, anchor_y), False) >= 0
            score = max(overlap_ratio, 1.0 if anchor_inside else 0.0)
            if score > best_score:
                best_score = score
                best_conf = conf
                best_detection = det

        is_occupied = best_score >= self.iou_threshold
        status = "OCCUPIED" if is_occupied else "AVAILABLE"
        confidence = round(best_conf if is_occupied else max(0.0, 1.0 - best_score), 2)

        return {
            "slot_id": slot["slot_id"],
            "status": status,
            "confidence": confidence,
            "occupancy_score": round(best_score, 3),
            "vehicle_id": self._vehicle_id(best_detection) if best_detection is not None else None,
        }

    def assign_slots(
        self,
        slots: list[dict[str, Any]],
        detections: list[Any],
        image_shape: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        """Assign vehicles to slots globally so one vehicle cannot fill two slots."""
        candidates: list[tuple[float, int, int, dict[str, Any]]] = []
        for slot_index, slot in enumerate(slots):
            poly = self._slot_polygon(slot.get("polygon", []), image_shape)
            if len(poly) < 3:
                continue
            for detection_index, detection in enumerate(detections):
                match = self._match_score(poly, detection)
                if match["score"] >= self.iou_threshold:
                    candidates.append((match["score"], slot_index, detection_index, match))

        assignments: dict[int, tuple[int, dict[str, Any]]] = {}
        assigned_detections: set[int] = set()
        for _, slot_index, detection_index, match in sorted(candidates, reverse=True):
            if slot_index in assignments or detection_index in assigned_detections:
                continue
            assignments[slot_index] = (detection_index, match)
            assigned_detections.add(detection_index)

        results = []
        for slot_index, slot in enumerate(slots):
            if slot_index in assignments:
                detection_index, match = assignments[slot_index]
                results.append({
                    "slot_id": slot["slot_id"],
                    "status": "OCCUPIED",
                    "confidence": match["confidence"],
                    "occupancy_score": match["score"],
                    "vehicle_id": self._vehicle_id(detections[detection_index]),
                })
            elif len(self._slot_polygon(slot.get("polygon", []), image_shape)) >= 3:
                results.append({
                    "slot_id": slot["slot_id"],
                    "status": "AVAILABLE",
                    "confidence": 1.0,
                    "occupancy_score": 0.0,
                    "vehicle_id": None,
                })
            else:
                results.append({
                    "slot_id": slot["slot_id"],
                    "status": "UNKNOWN",
                    "confidence": 0.0,
                    "occupancy_score": 0.0,
                    "vehicle_id": None,
                })
        return results

    def _best_match(self, polygon: list[list[float]], detections: list[Any]) -> dict[str, Any]:
        matches = [self._match_score(polygon, detection) for detection in detections]
        matches = [match for match in matches if match["score"] >= self.iou_threshold]
        if not matches:
            return {"status": "AVAILABLE", "confidence": 1.0, "occupancy_score": 0.0}
        match = max(matches, key=lambda item: item["score"])
        return {"status": "OCCUPIED", "occupancy_score": match["score"], **match}

    def _match_score(self, polygon: list[list[float]], detection: Any) -> dict[str, Any]:
        bbox = getattr(detection, "bbox", None) or (detection.get("bbox") if isinstance(detection, dict) else None)
        if not bbox or len(bbox) < 4:
            return {"score": 0.0, "confidence": 0.0}
        slot_contour = np.asarray(polygon, dtype=np.float32)
        vehicle_contour = np.asarray(
            [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]],
            dtype=np.float32,
        )
        slot_area = max(1.0, cv2.contourArea(slot_contour))
        vehicle_area = max(1.0, cv2.contourArea(vehicle_contour))
        intersection_area, _ = cv2.intersectConvexConvex(slot_contour, vehicle_contour)
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        center_inside = cv2.pointPolygonTest(slot_contour, center, False) >= 0
        overlap = float(intersection_area) / min(slot_area, vehicle_area)
        confidence = self._confidence(detection)
        score = overlap * confidence
        if center_inside:
            score = max(score, 0.5 * confidence)
        return {"score": round(score, 3), "confidence": round(confidence, 3)}

    @staticmethod
    def _slot_polygon(raw_poly: list[Any], image_shape: tuple[int, int] | None) -> list[list[float]]:
        if not raw_poly or len(raw_poly) < 3:
            return []
        if not image_shape:
            return [[float(point[0]), float(point[1])] for point in raw_poly]
        height, width = image_shape
        return [[float(point[0]) * width if 0 <= point[0] <= 1 else float(point[0]),
                 float(point[1]) * height if 0 <= point[1] <= 1 else float(point[1])] for point in raw_poly]

    @staticmethod
    def _confidence(detection: Any) -> float:
        value = getattr(detection, "confidence", None)
        if value is None and isinstance(detection, dict):
            value = detection.get("confidence", 0.0)
        return max(0.0, min(1.0, float(value or 0.0)))

    @staticmethod
    def _vehicle_id(detection: Any) -> Any:
        value = getattr(detection, "tracking_id", None)
        if value is None and isinstance(detection, dict):
            value = detection.get("tracking_id")
        return value
