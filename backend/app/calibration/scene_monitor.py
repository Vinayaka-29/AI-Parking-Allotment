from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.calibration.profile_store import build_fingerprint


class SceneMonitor:
    def __init__(self, small_translation_ratio: float = 0.08) -> None:
        self.small_translation_ratio = small_translation_ratio

    def evaluate(self, reference: np.ndarray, frame: np.ndarray, polygons: list[dict[str, Any]]) -> dict[str, Any]:
        if reference is None or frame is None:
            return {"status": "NEEDS_RECALIBRATION", "reason": "reference or live frame unavailable"}
        if reference.shape[:2] == frame.shape[:2] and np.array_equal(reference, frame):
            return {"status": "OK", "inlier_ratio": 1.0, "polygons": polygons}
        reference_points, reference_descriptors = build_fingerprint(reference)
        frame_points, frame_descriptors = build_fingerprint(frame)
        if reference_descriptors is None or frame_descriptors is None:
            return {"status": "NEEDS_RECALIBRATION", "reason": "insufficient visual features", "inlier_ratio": 0.0}
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(reference_descriptors, frame_descriptors, k=2)
        good = [first for first, second in pairs if first.distance < 0.75 * second.distance]
        if len(good) < 8:
            return {"status": "NEEDS_RECALIBRATION", "reason": "insufficient matches", "inlier_ratio": 0.0}
        source = np.float32([reference_points[m.queryIdx].pt for m in good])
        target = np.float32([frame_points[m.trainIdx].pt for m in good])
        homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
        if homography is None or mask is None:
            return {"status": "NEEDS_RECALIBRATION", "reason": "homography failed", "inlier_ratio": 0.0}
        ratio = float(mask.ravel().mean())
        height, width = reference.shape[:2]
        translation = float(np.linalg.norm(homography[:2, 2])) / max(width, height)
        if ratio < 0.25 or translation >= self.small_translation_ratio:
            return {"status": "NEEDS_RECALIBRATION", "reason": "scene transform is too large", "inlier_ratio": round(ratio, 3)}
        warped = []
        for slot in polygons:
            points = np.float32([[x * width, y * height] for x, y in slot["polygon"]]).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(points, homography).reshape(-1, 2)
            updated = dict(slot)
            updated["polygon"] = [[round(float(x / width), 6), round(float(y / height), 6)] for x, y in transformed]
            warped.append(updated)
        return {"status": "WARPED" if translation > 0.001 else "OK", "inlier_ratio": round(ratio, 3), "polygons": warped}
