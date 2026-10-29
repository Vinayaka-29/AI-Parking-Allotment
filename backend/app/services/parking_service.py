from __future__ import annotations

import io
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

import cv2
import numpy as np
from PIL import Image

from app.vision.occupancy import SlotOccupancyEngine
from app.parking.state_manager import ParkingStateManager
from app.vision.detector import VehicleDetector
from app.vision.camera_sources import SnapshotCameraSource, OpenCVCameraSource, VideoFrame
from app.core.config import settings

logger = logging.getLogger(__name__)

# ─── Camera metadata ───────────────────────────────────────────────────────────
ASAHI_CAMERA_META = {
    "camera_id": "CAM_ASAHI_JP",
    "name": "Asahi Parking Lot — Japan (Live CCTV)",
    "location": "Asahi, Chiba, Japan",
    "source_url": "http://124.155.121.218:3000/webcapture.jpg?command=snap&channel=1",
    "insecam_ref": "http://www.insecam.org/en/view/945438/",
    "manufacturer": "Hi3516",
    "resolution": "1920x1080",
    "fps": 3,
}


class ParkingService:
    def __init__(self) -> None:
        self.state_manager = ParkingStateManager()
        self.detector = VehicleDetector(model_name="yolov8n.pt", camera_id="CAM_ASAHI_JP")
        self.camera_active: bool = False
        self._cam_source: Optional[SnapshotCameraSource | OpenCVCameraSource] = None
        self.last_frame_annotated_b64: str | None = None
        self.camera_fps: float = 0.0
        self.camera_source_name: str = "STANDBY"
        self.camera_status: str = "OFFLINE"
        self.last_frame_timestamp: str | None = None
        self._settings = settings

    # ─── Layout loading ────────────────────────────────────────────────────────

    def initialize_demo_state(self) -> None:
        """Load parking layout config — prefers the Asahi calibrated layout."""
        base = Path(__file__).resolve().parents[3] / "configs"

        # Try the layout named in settings first (e.g. parking_layout_asahi.json)
        layout_name = getattr(self._settings, "parking_layout", "asahi")
        candidate = base / f"parking_layout_{layout_name}.json"
        fallback = base / "parking_layout.json"

        config_path = candidate if candidate.exists() else fallback
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            self.state_manager.load_layout(config)
            logger.info("Loaded parking layout: %s (%d slots)", config_path.name, len(config.get("slots", [])))
        else:
            logger.warning("No parking layout config found at %s or %s", candidate, fallback)

    # ─── Camera initialisation ─────────────────────────────────────────────────

    def _initialize_camera_source(self) -> bool:
        """Initialise the camera source based on settings."""
        try:
            self._release_cam_source()

            source = self._settings.camera_source.upper()
            url = self._settings.camera_url

            if source == "SNAPSHOT":
                # Pull-mode JPEG endpoint — works with Hi3516 cameras like the Asahi lot
                self._cam_source = SnapshotCameraSource(
                    url=url,
                    camera_id="CAM_ASAHI_JP",
                )
                self.camera_source_name = f"SNAPSHOT: Asahi Parking Lot, Japan"
                logger.info("Snapshot camera source initialised: %s", url)
                return True

            elif source == "WEBCAM":
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    logger.error("Webcam not available")
                    return False
                # Wrap in a simple OpenCVCameraSource-compatible object
                self._cam_source = _CVCapWrapper(cap, "WEBCAM")
                self.camera_source_name = "WEBCAM"
                return True

            elif source in ("URL", "RTSP"):
                cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    logger.error("Cannot open stream: %s", url)
                    return False
                self._cam_source = _CVCapWrapper(cap, f"URL: {url}")
                self.camera_source_name = f"URL: {url}"
                return True

            elif source == "FILE":
                if not url or not Path(url).exists():
                    logger.error("Camera file not found: %s", url)
                    return False
                cap = cv2.VideoCapture(url)
                self._cam_source = _CVCapWrapper(cap, f"FILE: {url}")
                self.camera_source_name = f"FILE: {url}"
                return True

            else:  # SAMPLE or unknown
                self._cam_source = None
                self.camera_source_name = "SAMPLE / SIMULATED"
                return True

        except Exception as exc:
            logger.error("Camera init error: %s", exc)
            self._cam_source = None
            return False

    def _release_cam_source(self) -> None:
        if self._cam_source is not None:
            try:
                self._cam_source.release()
            except Exception:
                pass
            self._cam_source = None

    # ─── Legacy compatibility property ────────────────────────────────────────
    @property
    def camera_cap(self) -> None:
        return None

    def release_camera_source(self) -> None:
        self._release_cam_source()
        self.camera_source_name = "STANDBY"
        logger.info("Camera source released")

    # ─── Image upload analysis ─────────────────────────────────────────────────

    def process_image_upload(self, file_bytes: bytes) -> dict[str, Any]:
        """Process an uploaded image file with YOLOv8 and update slot occupancy."""
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_np = np.array(image)
        frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w = frame_bgr.shape[:2]

        start_time = time.time()
        detections = self.detector.detect(frame_bgr, conf_threshold=0.15)
        inference_time_ms = round((time.time() - start_time) * 1000, 1)

        self.state_manager.process_vision_detections(detections, image_shape=(h, w))

        annotated_bgr = self.detector.annotate_frame(frame_bgr, detections, slots=[])
        annotated_b64 = self.detector.encode_base64_jpeg(annotated_bgr)
        self.last_frame_annotated_b64 = annotated_b64

        return {
            "overview": self.state_manager.get_overview(),
            "slots": self.state_manager.get_slots(),
            "detections": [d.to_dict() for d in detections],
            "total_detected_vehicles": len(detections),
            "inference_time_ms": inference_time_ms,
            "annotated_image": annotated_b64,
            "timestamp": time.time(),
        }

    # ─── Live camera stream ────────────────────────────────────────────────────

    def generate_camera_stream(self) -> Generator[bytes, None, None]:
        """Yield MJPEG frames from the live camera feed with real-time YOLO overlays."""
        if self._settings.camera_enabled and self._cam_source is None:
            if not self._initialize_camera_source():
                self._cam_source = None
                self.camera_source_name = "SAMPLE / SIMULATED (FALLBACK)"

        live_mode = self.camera_active and self._cam_source is not None
        while True:
            start_t = time.time()
            frame: Optional[np.ndarray] = None

            if live_mode:
                try:
                    vf = self._cam_source.read()  # type: ignore[union-attr]
                    if vf is not None and vf.frame is not None and not vf.is_cached:
                        frame = vf.frame
                        self.last_frame_timestamp = vf.timestamp
                        self.camera_status = "LIVE"
                    elif vf is not None and vf.frame is not None:
                        self.camera_status = "STALE"
                        self.state_manager.mark_camera_unknown()
                    else:
                        self.camera_status = "OFFLINE"
                        self.state_manager.mark_camera_unknown()
                        logger.warning("No frame from camera source, using simulated frame")
                except Exception as exc:
                    self.camera_status = "OFFLINE"
                    self.state_manager.mark_camera_unknown()
                    logger.error("Camera read error: %s", exc)

            if frame is None:
                frame = self._generate_camera_unavailable_frame()
                live_mode_label = False
            else:
                live_mode_label = live_mode

            # ── YOLO detection ──────────────────────────────────────────────
            # Lower confidence for the 1920x1080 overhead camera; cars are
            # small in the top portion of the frame.
            conf_thresh = 0.12 if live_mode_label else 0.30
            detections = self.detector.detect(frame, conf_threshold=conf_thresh) if live_mode_label else []

            # Never use synthetic fallback frames as occupancy evidence.
            if self.camera_active and live_mode_label:
                self.state_manager.process_vision_detections(
                    detections, image_shape=frame.shape[:2]
                )

            annotated = self.detector.annotate_frame(
                frame, detections, self.state_manager.get_slots()
            )

            # ── Draw calibrated slot polygons ───────────────────────────────
            if live_mode_label:
                annotated = self._draw_slot_overlays(annotated)

            # ── Camera meta overlay ─────────────────────────────────────────
            elapsed = time.time() - start_t
            self.camera_fps = round(1.0 / elapsed, 1) if elapsed > 0 else 0.0
            status_text = self.camera_status if self.camera_active else "STANDBY"

            fh, fw = annotated.shape[:2]
            cv2.putText(
                annotated,
                f"FPS: {self.camera_fps} | {status_text} | {self.camera_source_name}",
                (15, fh - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            success, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not success:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

            # ── Rate-limit to target FPS ────────────────────────────────────
            target = 1.0 / max(self._settings.camera_fps, 1)
            elapsed = time.time() - start_t
            if elapsed < target:
                time.sleep(target - elapsed)

    # ─── Slot polygon overlay ──────────────────────────────────────────────────

    def _draw_slot_overlays(self, frame: np.ndarray) -> np.ndarray:
        """Draw calibrated parking slot polygons on the live frame."""
        h, w = frame.shape[:2]
        for slot in self.state_manager.get_slots():
            poly = slot.get("polygon", [])
            if not poly or len(poly) < 3:
                continue
            ref_w, ref_h = slot.get("calibration_size") or [None, None]
            scaled = [[p[0] * w if 0 <= p[0] <= 1 else p[0] * w / ref_w if ref_w else p[0],
                       p[1] * h if 0 <= p[1] <= 1 else p[1] * h / ref_h if ref_h else p[1]] for p in poly]
            pts = np.array(scaled, dtype=np.int32)
            status = slot.get("status", "AVAILABLE")
            if status == "OCCUPIED":
                color = (0, 60, 220)   # Red-ish (BGR)
                fill_alpha = 0.35
            elif status == "RESERVED":
                color = (0, 165, 255)  # Orange (BGR)
                fill_alpha = 0.30
            else:
                color = (0, 200, 60)   # Green (BGR)
                fill_alpha = 0.20

            # Blend each slot independently so its color does not leak to neighbors.
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)
            # Border
            cv2.polylines(frame, [pts], True, color, 2)
            # Label
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.putText(frame, slot["slot_id"], (cx - 14, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    # ─── Simulated frame fallback ──────────────────────────────────────────────

    def _generate_simulated_frame(self, frame_idx: int) -> np.ndarray:
        """Synthetic high-tech parking scene for demo/fallback."""
        w, h = 640, 360
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (26, 32, 44)

        cv2.rectangle(img, (50, 40), (590, 280), (35, 45, 60), -1)
        cv2.line(img, (60, 150), (580, 150), (100, 110, 130), 2, cv2.LINE_AA)

        for x in range(100, 550, 90):
            cv2.line(img, (x, 50), (x, 130), (70, 80, 95), 1)
            cv2.line(img, (x, 170), (x, 250), (70, 80, 95), 1)

        scan_x = int((frame_idx * 4) % w)
        cv2.line(img, (scan_x, 40), (scan_x, 280), (0, 255, 157), 1)

        for slot in self.state_manager.get_slots():
            if slot["status"] == "OCCUPIED":
                raw_poly = slot.get("polygon", [])
                if raw_poly and len(raw_poly) >= 4:
                    xs = [p[0] for p in raw_poly]
                    ys = [p[1] for p in raw_poly]
                    # Scale down to simulated frame size
                    cx = int(np.mean(xs) * w / 1920)
                    cy = int(np.mean(ys) * h / 1080)
                    cv2.rectangle(img, (cx - 30, cy - 18), (cx + 30, cy + 18), (180, 50, 60), -1)
                    cv2.rectangle(img, (cx - 18, cy - 12), (cx + 18, cy + 12), (220, 220, 240), -1)
                    cv2.circle(img, (cx + 28, cy - 10), 3, (0, 255, 255), -1)
                    cv2.circle(img, (cx + 28, cy + 10), 3, (0, 255, 255), -1)

        return img

    def _generate_camera_unavailable_frame(self) -> np.ndarray:
        """Never present generated graphics as if they were CCTV footage."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:] = (12, 18, 28)
        cv2.rectangle(frame, (28, 28), (612, 332), (30, 44, 60), 1)
        title = "LIVE CAMERA SOURCE UNAVAILABLE"
        detail = "The camera snapshot endpoint did not return an image."
        cv2.putText(frame, title, (54, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, detail, (54, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (190, 205, 220), 1, cv2.LINE_AA)
        return frame

    # ─── Camera toggle ─────────────────────────────────────────────────────────

    def toggle_camera(self, active: bool | None = None) -> dict[str, Any]:
        if active is not None:
            self.camera_active = active
        else:
            self.camera_active = not self.camera_active

        if self.camera_active:
            if self._settings.camera_source.upper() not in ("SAMPLE",) and self._settings.camera_enabled:
                if not self._initialize_camera_source():
                    self.camera_source_name = "SAMPLE / SIMULATED (INIT FAILED)"
                    logger.warning("Camera init failed, falling back to sample mode")
            else:
                self.camera_source_name = "SAMPLE / SIMULATED"
            self.camera_status = "CONNECTING"
        else:
            self.release_camera_source()
            self.camera_source_name = "STANDBY"
            self.camera_status = "OFFLINE"

        return {
            "camera_active": self.camera_active,
            "status": "RUNNING" if self.camera_active else "STOPPED",
            "source": self.camera_source_name,
            "camera_status": self.camera_status,
            "last_frame_timestamp": self.last_frame_timestamp,
            "camera_meta": ASAHI_CAMERA_META if self.camera_active else {},
        }

    # ─── Accessors ─────────────────────────────────────────────────────────────

    def get_overview(self) -> dict[str, Any]:
        return self.state_manager.get_overview()

    def get_slots(self) -> list[dict[str, Any]]:
        return self.state_manager.get_slots()

    def get_recent_events(self) -> list[dict[str, Any]]:
        return self.state_manager.get_recent_events()

    def simulate_random_event(self) -> dict[str, Any]:
        return self.state_manager.simulate_random_change()

    def reset_slots(self) -> dict[str, Any]:
        return self.state_manager.reset_all_slots()

    def allocate_slot(self, vehicle_id: str, vehicle_type: str = "car") -> dict[str, Any]:
        return self.state_manager.allocate_slot(vehicle_id, vehicle_type)

    def get_system_status(self) -> dict[str, Any]:
        overview = self.state_manager.get_overview()
        return {
            "status": "healthy",
            "camera_active": self.camera_active,
            "camera_fps": self.camera_fps,
            "camera_source": self.camera_source_name,
            "camera_status": self.camera_status,
            "last_frame_timestamp": self.last_frame_timestamp,
            "camera_meta": ASAHI_CAMERA_META,
            "model": "YOLOv8-Nano (COCO Pre-trained)",
            "camera_count": 1,
            "total_slots": overview["total_slots"],
            "occupied_slots": overview["occupied"],
            "available_slots": overview["available"],
            "occupancy_pct": overview["occupancy_pct"],
        }

    def get_sections(self) -> list[dict[str, Any]]:
        sections: dict[str, dict[str, Any]] = {}
        for slot in self.state_manager.get_slots():
            sid = slot.get("section_id", "A")
            if sid not in sections:
                sections[sid] = {"section_id": sid, "slots": [], "occupied": 0, "available": 0}
            sections[sid]["slots"].append(slot["slot_id"])
            if slot["status"] == "OCCUPIED":
                sections[sid]["occupied"] += 1
            else:
                sections[sid]["available"] += 1
        return list(sections.values())


# ─── Helper wrapper ────────────────────────────────────────────────────────────

class _CVCapWrapper:
    """Thin wrapper to give cv2.VideoCapture the same .read()/.release() interface
    as our custom camera sources."""

    def __init__(self, cap: cv2.VideoCapture, name: str) -> None:  # type: ignore[name-defined]
        self._cap = cap
        self.name = name

    def read(self):  # type: ignore[no-untyped-def]
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return VideoFrame(
            frame=frame,
            timestamp=datetime.now(timezone.utc).isoformat(),
            camera_id=self.name,
            frame_id=0,
        )

    def release(self) -> None:
        self._cap.release()


parking_service = ParkingService()
