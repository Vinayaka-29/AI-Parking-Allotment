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
from app.vision.camera_sources import SnapshotCameraSource, OpenCVCameraSource, VideoFrame, CameraSource
from app.vision.camera_worker import CameraWorker
from app.vision.detection_worker import DetectionWorker
from app.core.config import settings
from app.cameras import active_camera, camera_metadata
from app.calibration.auto_calibrator import AutoCalibrator
from app.calibration.profile_store import CalibrationProfileStore, build_fingerprint
from app.calibration.scene_monitor import SceneMonitor

logger = logging.getLogger(__name__)

# ─── Camera metadata ───────────────────────────────────────────────────────────
ASAHI_CAMERA_META = camera_metadata(active_camera())


class ParkingService:
    def __init__(self) -> None:
        self._settings = settings
        self.camera = active_camera()
        self.profile_store = CalibrationProfileStore()
        self.scene_monitor = SceneMonitor()
        self.auto_calibrator = AutoCalibrator(settings.calib_cycles, settings.calib_min_parked)
        self.calibration_status = "NEEDS_CALIBRATION"
        self.calibration_message = "No confirmed profile loaded"
        self._last_calibration_frame: np.ndarray | None = None
        self._last_scene_check = 0.0
        self.state_manager = ParkingStateManager()
        self.state_manager.occupancy_votes_required = self._settings.occupancy_votes
        self.detector = VehicleDetector(model_name=settings.detect_model, camera_id=self.camera.camera_id)
        self.camera_active: bool = False
        self._cam_source: Optional[SnapshotCameraSource | OpenCVCameraSource] = None
        self.last_frame_annotated_b64: str | None = None
        self.camera_fps: float = 0.0
        self.camera_source_name: str = "STANDBY"
        self.camera_status: str = "OFFLINE"
        self.last_frame_timestamp: str | None = None
        self._camera_worker: CameraWorker | None = None
        self._detection_worker: DetectionWorker | None = None
        self._stream_placeholder: bytes | None = None

    # ─── Layout loading ────────────────────────────────────────────────────────

    def initialize_demo_state(self) -> None:
        """Migrate/load only the confirmed profile belonging to the active camera."""
        base = Path(__file__).resolve().parents[3] / "configs"
        candidate = base / "parking_layout_asahi.json"
        if self.camera.camera_id == "CAM_ASAHI_JP":
            profile = self.profile_store.migrate_layout(self.camera.camera_id, self.camera.url, candidate)
        else:
            profile = self.profile_store.load_validated(self.camera.camera_id, self.camera.url)
        self._apply_profile(profile)

    def _apply_profile(self, profile: dict[str, Any] | None) -> None:
        if not profile or profile.get("status") != "CONFIRMED":
            self.calibration_status = "NEEDS_CALIBRATION"
            self.calibration_message = "No confirmed calibration profile for this camera URL"
            self.state_manager.load_layout({"slots": []})
            return
        if profile.get("camera_url_hash") != self.profile_store.url_hash(self.camera.url):
            self.calibration_status = "NEEDS_CALIBRATION"
            self.calibration_message = "Calibration profile URL does not match the active camera"
            self.state_manager.load_layout({"slots": []})
            return
        config = {
            "image_width": profile.get("frame_width"),
            "image_height": profile.get("frame_height"),
            "slots": [
                {**slot, "camera_id": self.camera.camera_id,
                 "calibration_size": [profile.get("frame_width"), profile.get("frame_height")]}
                for slot in profile.get("polygons", [])
            ],
        }
        self.state_manager.load_layout(config)
        self.calibration_status = "CONFIRMED"
        self.calibration_message = "Confirmed calibration profile loaded"

    # ─── Camera initialisation ─────────────────────────────────────────────────

    def _create_camera_source(self) -> CameraSource:
        source = self._settings.camera_source.upper()
        url = self._settings.camera_url
        if source == "SNAPSHOT":
            return SnapshotCameraSource(url=url, camera_id=self.camera.camera_id)
        if source == "WEBCAM":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError("Webcam not available")
            return _CVCapWrapper(cap, "WEBCAM")
        if source in ("URL", "RTSP"):
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"Cannot open stream: {url}")
            return _CVCapWrapper(cap, f"URL: {url}")
        if source == "FILE":
            if not url or not Path(url).exists():
                raise RuntimeError(f"Camera file not found: {url}")
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"Cannot open camera file: {url}")
            return _CVCapWrapper(cap, f"FILE: {url}")
        raise RuntimeError(f"Unsupported camera source: {source}")

    def _start_workers(self) -> None:
        if self._camera_worker and self._detection_worker:
            self._camera_worker.start()
            self._detection_worker.start()
            return
        source = self._settings.camera_source.upper()
        if source == "SAMPLE" or not self._settings.camera_enabled:
            self.camera_source_name = "SAMPLE / SIMULATED"
            return
        self._camera_worker = CameraWorker(
            self._create_camera_source,
            fetch_interval_s=self._settings.camera_fetch_interval_s,
            stale_after_s=self._settings.camera_stale_after_s,
            camera_id="CAM_ASAHI_JP",
        )
        self._detection_worker = DetectionWorker(
            self._camera_worker,
            self.detector,
            self.state_manager,
            interval_s=self._settings.detect_interval_s,
            conf_threshold=self._settings.detect_conf,
            tiles=self._settings.detect_tiles,
            tile_overlap=self._settings.detect_tile_overlap,
            full_frame=self._settings.detect_fullframe,
            cycle_callback=self._handle_detection_cycle,
        )
        self.camera_source_name = f"{source}: {self.camera.name}"
        self._camera_worker.start()
        self._detection_worker.start()

    def _stop_workers(self) -> None:
        if self._detection_worker:
            self._detection_worker.stop()
        if self._camera_worker:
            self._camera_worker.stop()
        self._detection_worker = None
        self._camera_worker = None

    def _initialize_camera_source(self) -> bool:
        """Validate settings and prepare the worker-owned camera source."""
        try:
            self._create_camera_source().release()
            return True
        except Exception as exc:
            logger.error("Camera init error: %s", exc)
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
        self._stop_workers()
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
        """Yield the shared latest annotated frame without doing camera work."""
        while True:
            if self.camera_active and self._detection_worker:
                snapshot = self._detection_worker.latest()
                payload = snapshot.jpeg_bytes
                self.camera_status = self._camera_worker.diagnostics()["camera_status"] if self._camera_worker else "OFFLINE"
            else:
                payload = None
                self.camera_status = "OFFLINE"
            if payload is None:
                payload = self._placeholder_bytes()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
            time.sleep(1.0 / max(self._settings.camera_fps, 1))

    def _placeholder_bytes(self) -> bytes:
        if self._stream_placeholder is None:
            success, buffer = cv2.imencode(".jpg", self._generate_camera_unavailable_frame())
            if success:
                self._stream_placeholder = buffer.tobytes()
        return self._stream_placeholder or b""

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
            profile = self.profile_store.load_validated(self.camera.camera_id, self.camera.url)
            self._apply_profile(profile)
            self._start_workers()
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
            **self._diagnostics(),
        }

    def _diagnostics(self) -> dict[str, Any]:
        camera = self._camera_worker.diagnostics() if self._camera_worker else {
            "camera_status": "OFFLINE", "frame_age_s": None, "last_error": None
        }
        detector = self._detection_worker.diagnostics() if self._detection_worker else {
            "detector_status": "OFFLINE", "last_inference_ms": None,
            "last_detection_count": 0, "last_error": None,
            "detect_model": self.detector.model_name, "tiles": self._settings.detect_tiles,
        }
        return {**camera, **detector, "calibration_status": self.calibration_status,
                "calibration_message": self.calibration_message, "camera_id": self.camera.camera_id}

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
            "camera_meta": camera_metadata(self.camera),
            "model": "YOLOv8-Nano (COCO Pre-trained)",
            "camera_count": 1,
            "total_slots": overview["total_slots"],
            "occupied_slots": overview["occupied"],
            "available_slots": overview["available"],
            "occupancy_pct": overview["occupancy_pct"],
            **self._diagnostics(),
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

    def _handle_detection_cycle(self, detections: list[Any], frame: np.ndarray) -> None:
        self._last_calibration_frame = frame.copy()
        if self.calibration_status == "CONFIRMED" and not self.profile_store.reference_path(self.camera.camera_id).exists():
            profile = self.profile_store.load(self.camera.camera_id)
            if profile:
                self.profile_store.save(profile, reference=frame, fingerprint=build_fingerprint(frame))
        self._check_scene(frame)
        if self.calibration_status in {"NEEDS_CALIBRATION", "NEEDS_RECALIBRATION", "CALIBRATING"}:
            proposal = self.auto_calibrator.update(detections, frame)
            self.calibration_status = self.auto_calibrator.status
            self.calibration_message = self.auto_calibrator.message
            if proposal and proposal.get("status") == "PROPOSED":
                self._save_proposal(proposal, frame)

    def _check_scene(self, frame: np.ndarray) -> None:
        now = time.time()
        if now - self._last_scene_check < self._settings.scene_check_interval_s:
            return
        self._last_scene_check = now
        profile = self.profile_store.load(self.camera.camera_id)
        if not profile or profile.get("status") != "CONFIRMED":
            return
        reference = cv2.imread(str(self.profile_store.reference_path(self.camera.camera_id)))
        if reference is None:
            return
        result = self.scene_monitor.evaluate(reference, frame, profile.get("polygons", []))
        if result["status"] == "NEEDS_RECALIBRATION":
            self.calibration_status = "NEEDS_RECALIBRATION"
            self.calibration_message = result.get("reason", "Camera scene changed")
            self.state_manager.mark_camera_unknown()
            self.auto_calibrator.reset()
        elif result["status"] == "WARPED":
            profile["polygons"] = result["polygons"]
            profile["status"] = "CONFIRMED"
            self.profile_store.save(profile, reference=frame, fingerprint=build_fingerprint(frame))
            self._apply_profile(profile)

    def start_calibration(self) -> None:
        self.auto_calibrator.reset()
        self.calibration_status = "CALIBRATING"
        self.calibration_message = "Waiting for live frames"
        self.state_manager.mark_camera_unknown()

    def _save_proposal(self, proposal: dict[str, Any], frame: np.ndarray) -> None:
        self.profile_store.save({
            "camera_id": self.camera.camera_id,
            "camera_url_hash": self.profile_store.url_hash(self.camera.url),
            "frame_width": frame.shape[1],
            "frame_height": frame.shape[0],
            "polygons": proposal["polygons"],
            "status": "PROPOSED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "confirmed_at": None,
            "source": "auto",
            "scene_fingerprint": None,
            "quality": proposal.get("quality", {}),
            "version": 0,
        }, reference=frame, fingerprint=build_fingerprint(frame))

    def calibration_status_payload(self) -> dict[str, Any]:
        return {
            "status": self.calibration_status,
            "progress": {"cycles_done": self.auto_calibrator.cycles_done, "cycles_total": self.auto_calibrator.total_cycles},
            "parked_candidates": self.auto_calibrator.proposal.get("quality", {}).get("parked_candidates", 0)
                if self.auto_calibrator.proposal else 0,
            "message": self.calibration_message,
            "camera_id": self.camera.camera_id,
        }

    def calibration_proposal(self) -> dict[str, Any]:
        profile = self.profile_store.load(self.camera.camera_id)
        proposal = profile if profile and profile.get("status") == "PROPOSED" else None
        result = {"status": "NO_PROPOSAL", "polygons": [], "quality": {}}
        if proposal:
            result.update({"status": "PROPOSED", "polygons": proposal.get("polygons", []), "quality": proposal.get("quality", {})})
            reference = cv2.imread(str(self.profile_store.reference_path(self.camera.camera_id)))
            if reference is not None:
                overlay = reference.copy()
                for slot in result["polygons"]:
                    points = np.int32([[x * overlay.shape[1], y * overlay.shape[0]] for x, y in slot["polygon"]])
                    cv2.polylines(overlay, [points], True, (0, 255, 170), 2)
                    cv2.putText(overlay, slot["slot_id"], tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 170), 2)
                ok, encoded = cv2.imencode(".jpg", overlay)
                if ok:
                    import base64
                    result["reference_jpeg"] = base64.b64encode(encoded).decode("ascii")
        return result

    def confirm_calibration(self, slots: list[dict[str, Any]]) -> dict[str, Any]:
        if not slots:
            raise ValueError("At least one slot is required")
        for slot in slots:
            polygon = slot.get("polygon", [])
            if len(polygon) < 4 or any(len(point) != 2 or any(float(value) < 0 or float(value) > 1 for value in point) for point in polygon):
                raise ValueError(f"Invalid normalized polygon for {slot.get('slot_id', 'unknown')}")
        frame = self._last_calibration_frame
        width = frame.shape[1] if frame is not None else 1920
        height = frame.shape[0] if frame is not None else 1080
        now = datetime.now(timezone.utc).isoformat()
        profile = self.profile_store.save({
            "camera_id": self.camera.camera_id,
            "camera_url_hash": self.profile_store.url_hash(self.camera.url),
            "frame_width": width,
            "frame_height": height,
            "polygons": slots,
            "status": "CONFIRMED",
            "created_at": now,
            "confirmed_at": now,
            "source": "manual",
            "scene_fingerprint": None,
            "version": 0,
        }, reference=frame, fingerprint=build_fingerprint(frame) if frame is not None else None)
        self._apply_profile(profile)
        return {"status": "CONFIRMED", "slots_count": len(slots), "camera_id": self.camera.camera_id}

    def reset_calibration(self) -> dict[str, Any]:
        self.profile_store.delete(self.camera.camera_id)
        self.start_calibration()
        return self.calibration_status_payload()

    def persist_manual_layout(self, slots: list[dict[str, Any]], width: int | None = None, height: int | None = None) -> dict[str, Any]:
        frame = self._last_calibration_frame
        image_width = width or (frame.shape[1] if frame is not None else 1920)
        image_height = height or (frame.shape[0] if frame is not None else 1080)
        normalized = []
        for index, raw in enumerate(slots):
            points = raw.get("polygon", [])
            if any(float(point[0]) > 1 or float(point[1]) > 1 for point in points):
                points = [[round(float(x) / image_width, 8), round(float(y) / image_height, 8)] for x, y in points]
            normalized.append({**raw, "slot_id": raw.get("slot_id") or raw.get("id") or f"A{index + 1:02d}", "polygon": points})
        result = self.confirm_calibration(normalized)
        self.state_manager.load_layout({
            "image_width": image_width,
            "image_height": image_height,
            "slots": [{**slot, "calibration_size": [image_width, image_height]} for slot in normalized],
        })
        return result


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
