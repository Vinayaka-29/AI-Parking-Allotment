from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.vision.camera_worker import CameraWorker
from app.vision.detector import DetectorInferenceError, VehicleDetection, VehicleDetector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectionSnapshot:
    jpeg_bytes: bytes | None
    detections: tuple[VehicleDetection, ...]
    frame_id: int | None
    inference_ms: float | None
    detector_status: str
    last_error: str | None
    updated_at: float | None


class DetectionWorker:
    """Runs inference independently from camera acquisition and HTTP clients."""

    def __init__(
        self,
        camera_worker: CameraWorker,
        detector: VehicleDetector,
        state_manager: Any,
        *,
        interval_s: float = 1.0,
        conf_threshold: float = 0.25,
        tiles: str = "3x2",
        tile_overlap: float = 0.20,
        full_frame: bool = False,
        cycle_callback: Any | None = None,
    ) -> None:
        self.camera_worker = camera_worker
        self.detector = detector
        self.state_manager = state_manager
        self.interval_s = max(0.05, interval_s)
        self.conf_threshold = conf_threshold
        self.tile_columns, self.tile_rows = self._parse_tiles(tiles)
        self.tile_overlap = min(0.5, max(0.0, tile_overlap))
        self.full_frame = full_frame
        self.cycle_callback = cycle_callback
        self._lock = threading.Lock()
        self._snapshot = DetectionSnapshot(None, (), None, None, "OFFLINE", None, None)
        self._last_frame_id: int | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _parse_tiles(value: str) -> tuple[int, int]:
        try:
            columns, rows = (int(part) for part in value.lower().split("x", 1))
            if columns < 1 or rows < 1:
                raise ValueError
            return columns, rows
        except (ValueError, TypeError):
            raise ValueError("DETECT_TILES must use the CxR format, for example 3x2")

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, name="detection-worker", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=10)
        with self._lock:
            self._thread = None

    def latest(self) -> DetectionSnapshot:
        with self._lock:
            return self._snapshot

    def diagnostics(self) -> dict[str, object]:
        snapshot = self.latest()
        return {
            "detector_status": snapshot.detector_status,
            "last_inference_ms": snapshot.inference_ms,
            "last_detection_count": len(snapshot.detections),
            "last_error": snapshot.last_error,
            "detect_model": self.detector.model_name,
            "tiles": f"{self.tile_columns}x{self.tile_rows}",
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            camera_frame = self.camera_worker.get_latest()
            if camera_frame is None or camera_frame.is_stale:
                self._mark_unknown("camera frame is unavailable or stale", "FAILED" if camera_frame else "OFFLINE")
                self._stop_event.wait(self.interval_s)
                continue
            if camera_frame.frame_id == self._last_frame_id:
                self._stop_event.wait(self.interval_s)
                continue

            started = time.perf_counter()
            try:
                detections = self.detect_frame(camera_frame.frame)
                inference_ms = round((time.perf_counter() - started) * 1000, 1)
                if self.cycle_callback is not None:
                    self.cycle_callback(detections, camera_frame.frame)
                self.state_manager.process_vision_detections(
                    detections,
                    image_shape=camera_frame.frame.shape[:2],
                )
                annotated = self.detector.annotate_frame(camera_frame.frame, detections)
                success, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if not success:
                    raise DetectorInferenceError("failed to encode annotated frame")
                with self._lock:
                    self._snapshot = DetectionSnapshot(
                        encoded.tobytes(),
                        tuple(detections),
                        camera_frame.frame_id,
                        inference_ms,
                        "READY",
                        None,
                        time.time(),
                    )
                self._last_frame_id = camera_frame.frame_id
            except Exception as exc:
                self._mark_unknown(str(exc), "FAILED", inference_ms=round((time.perf_counter() - started) * 1000, 1))
                self._last_frame_id = camera_frame.frame_id
            self._stop_event.wait(self.interval_s)

    def _mark_unknown(self, error: str, status: str, inference_ms: float | None = None) -> None:
        self.state_manager.mark_camera_unknown()
        with self._lock:
            previous = self._snapshot
            self._snapshot = DetectionSnapshot(
                previous.jpeg_bytes,
                (),
                previous.frame_id,
                inference_ms if inference_ms is not None else previous.inference_ms,
                status,
                error,
                previous.updated_at,
            )

    def detect_frame(self, frame: np.ndarray) -> list[VehicleDetection]:
        height, width = frame.shape[:2]
        detections: list[VehicleDetection] = []
        if self.full_frame:
            detections.extend(self.detector.detect(frame, conf_threshold=self.conf_threshold))
        if width >= 1280:
            for x1, y1, x2, y2 in self._tile_bounds(width, height):
                tile = frame[y1:y2, x1:x2]
                for detection in self.detector.detect(tile, conf_threshold=self.conf_threshold):
                    detection.bbox = [
                        detection.bbox[0] + x1,
                        detection.bbox[1] + y1,
                        detection.bbox[2] + x1,
                        detection.bbox[3] + y1,
                    ]
                    detections.append(detection)
        else:
            detections.extend(self.detector.detect(frame, conf_threshold=self.conf_threshold))
        return self._nms(detections)

    def _tile_bounds(self, width: int, height: int) -> list[tuple[int, int, int, int]]:
        tile_width = min(width, int(np.ceil((width / self.tile_columns) * (1.0 + self.tile_overlap))))
        tile_height = min(height, int(np.ceil((height / self.tile_rows) * (1.0 + self.tile_overlap))))
        positions_x = [
            round(index * (width - tile_width) / max(self.tile_columns - 1, 1))
            for index in range(self.tile_columns)
        ]
        positions_y = [
            round(index * (height - tile_height) / max(self.tile_rows - 1, 1))
            for index in range(self.tile_rows)
        ]
        bounds: list[tuple[int, int, int, int]] = []
        for y in positions_y:
            for x in positions_x:
                x2 = min(width, x + tile_width)
                y2 = min(height, y + tile_height)
                bounds.append((x, y, x2, y2))
        return bounds

    @staticmethod
    def _nms(detections: list[VehicleDetection]) -> list[VehicleDetection]:
        if len(detections) < 2:
            return detections
        boxes = [[int(d.bbox[0]), int(d.bbox[1]), int(d.bbox[2] - d.bbox[0]), int(d.bbox[3] - d.bbox[1])] for d in detections]
        indices = cv2.dnn.NMSBoxes(boxes, [d.confidence for d in detections], 0.0, 0.45)
        flat = [int(index) for index in np.asarray(indices).reshape(-1)] if len(indices) else []
        return [detections[index] for index in flat]
