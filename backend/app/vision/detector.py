from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Vehicle classes in COCO: 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class VehicleDetection:
    tracking_id: int | None = None
    class_name: str = "car"
    confidence: float = 0.0
    bbox: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # [x1, y1, x2, y2]
    camera_id: str = "CAM_01"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracking_id": self.tracking_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox": [round(coord, 1) for coord in self.bbox],
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
        }


class VehicleDetector:
    """YOLOv8-powered real-time vehicle detector."""

    def __init__(self, model_name: str = "yolov8n.pt", camera_id: str = "CAM_01") -> None:
        self.camera_id = camera_id
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            logger.info("Successfully loaded YOLO model: %s", self.model_name)
        except Exception as exc:
            logger.warning("Could not load YOLO model (%s). Will use simulated detector: %s", self.model_name, exc)
            self.model = None

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.25) -> list[VehicleDetection]:
        """Detect vehicles in an image frame (BGR format)."""
        if frame is None or frame.size == 0:
            return []

        if self.model is None:
            self._load_model()

        detections: list[VehicleDetection] = []

        if self.model is not None:
            try:
                results = self.model.predict(source=frame, conf=conf_threshold, verbose=False)
                if results and len(results) > 0:
                    first_result = results[0]
                    boxes = first_result.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls_id = int(box.cls[0].item()) if hasattr(box.cls[0], "item") else int(box.cls[0])
                            # Check if detected class is a vehicle
                            if cls_id in VEHICLE_CLASS_IDS:
                                conf = float(box.conf[0].item()) if hasattr(box.conf[0], "item") else float(box.conf[0])
                                xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else list(box.xyxy[0])
                                class_name = VEHICLE_CLASS_IDS.get(cls_id, "car")
                                detections.append(
                                    VehicleDetection(
                                        class_name=class_name,
                                        confidence=conf,
                                        bbox=xyxy,
                                        camera_id=self.camera_id,
                                    )
                                )
            except Exception as exc:
                logger.error("YOLO detection error: %s", exc)

        return detections

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: list[VehicleDetection],
        slots: list[dict[str, Any]],
    ) -> np.ndarray:
        """Draw high-tech bounding boxes and glowing parking slots onto frame."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Draw parking slots first
        for slot in slots:
            raw_poly = slot.get("polygon", [])
            if not raw_poly or len(raw_poly) < 3:
                continue

            # Scale polygon if coordinates are relative (0..1) or absolute
            poly_pts = []
            for pt in raw_poly:
                px = int(pt[0] * w) if 0 <= pt[0] <= 1.0 else int(pt[0])
                py = int(pt[1] * h) if 0 <= pt[1] <= 1.0 else int(pt[1])
                poly_pts.append([px, py])

            pts_np = np.array([poly_pts], dtype=np.int32)
            status = slot.get("status", "AVAILABLE")
            slot_id = slot.get("slot_id", "SLOT")

            # Colors: Green for AVAILABLE (0, 255, 157), Red for OCCUPIED (95, 42, 255 BGR)
            if status == "AVAILABLE":
                color = (157, 255, 0)       # BGR: Neon emerald green
                fill_color = (100, 200, 0)
            elif status == "RESERVED":
                color = (0, 183, 255)       # BGR: Cyber amber
                fill_color = (0, 140, 200)
            else:
                color = (95, 42, 255)       # BGR: Cyber Crimson Red
                fill_color = (60, 20, 200)

            # Draw transparent slot fill overlay
            overlay = annotated.copy()
            cv2.fillPoly(overlay, pts_np, fill_color)
            cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

            # Draw glowing boundary
            cv2.polylines(annotated, pts_np, isClosed=True, color=color, thickness=2)

            # Slot label
            cx = int(np.mean([p[0] for p in poly_pts]))
            cy = int(np.mean([p[1] for p in poly_pts]))
            label = f"{slot_id}: {status[:3]}"
            cv2.putText(
                annotated,
                label,
                (cx - 30, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Draw detected vehicle bounding boxes
        for det in detections:
            bbox = det.bbox
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            # Cyan box for detected vehicle
            box_color = (255, 240, 0)  # BGR Cyan
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # Tag label background
            tag = f"AI {det.class_name.upper()} {int(det.confidence * 100)}%"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + tw + 6, max(20, y1)), box_color, -1)
            cv2.putText(
                annotated,
                tag,
                (x1 + 3, max(14, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # Futuristic HUD Header banner
        cv2.putText(
            annotated,
            f"AI-PARK VISION ENGINE | ACTIVE DETECTIONS: {len(detections)}",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 157),
            2,
            cv2.LINE_AA,
        )

        return annotated

    def encode_base64_jpeg(self, image: np.ndarray) -> str:
        """Encode OpenCV image to base64 JPEG data URL string."""
        success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            return ""
        encoded = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
