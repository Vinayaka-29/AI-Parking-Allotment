"""Explicit Asahi CCTV benchmark.

This script contacts a public camera and is intentionally not a pytest test.
Invoke it directly when an external-camera benchmark is authorized.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.vision.camera_sources import SnapshotCameraSource
from app.vision.detector import VehicleDetector
from app.vision.occupancy import SlotOccupancyEngine
from app.services.parking_service import ASAHI_CAMERA_META

CCTV_URL = "http://124.155.121.218:3000/webcapture.jpg?command=snap&channel=1"
LAYOUT_PATH = Path(__file__).parent.parent / "configs" / "parking_layout_asahi.json"


def main() -> int:
    print("=" * 60)
    print("  AI-PARK: Asahi CCTV Integration Benchmark")
    print("=" * 60)

    with LAYOUT_PATH.open(encoding="utf-8") as handle:
        layout = json.load(handle)
    slots = layout["slots"]
    print(f"[OK] Loaded layout: {layout['parking_lot']}")
    print(f"     {len(slots)} calibrated slots, image size: {layout['image_width']}x{layout['image_height']}")

    source = SnapshotCameraSource(url=CCTV_URL, camera_id="CAM_ASAHI_JP")
    try:
        print("\n[>] Fetching a fresh frame from CCTV...")
        started = time.time()
        video_frame = source.read()
        fetch_ms = round((time.time() - started) * 1000)
        if video_frame is None or video_frame.frame is None or video_frame.is_cached:
            print("[X] Failed to fetch a fresh frame.")
            return 1

        frame = video_frame.frame
        height, width = frame.shape[:2]
        print(f"[OK] Frame fetched: {width}x{height} in {fetch_ms}ms")

        print("\n[>] Running YOLOv8-Nano detection...")
        detector = VehicleDetector(model_name="yolov8n.pt", camera_id="CAM_ASAHI_JP")
        started = time.time()
        detections = detector.detect(frame, conf_threshold=0.12)
        inference_ms = round((time.time() - started) * 1000)
        print(f"[OK] Detected {len(detections)} vehicles in {inference_ms}ms")

        engine = SlotOccupancyEngine(iou_threshold=0.15)
        occupied = 0
        for slot in slots:
            result = engine.compute_slot_status(slot, detections, image_shape=(height, width))
            if result["status"] == "OCCUPIED":
                occupied += 1
            print(
                f"    {slot['slot_id']:5s} [{result['status']:9s}] "
                f"score={result['occupancy_score']:.3f} conf={result['confidence']:.2f}"
            )

        available = len(slots) - occupied
        occupancy_pct = round(occupied / len(slots) * 100, 1) if slots else 0
        print(f"\n{'=' * 60}")
        print(f"  SUMMARY: {occupied} OCCUPIED / {available} AVAILABLE / {len(slots)} TOTAL")
        print(f"  Occupancy: {occupancy_pct}%")
        print(f"  Camera: {ASAHI_CAMERA_META['name']}")
        print(f"  Location: {ASAHI_CAMERA_META['location']}")
        print("=" * 60)
        return 0
    finally:
        source.release()


if __name__ == "__main__":
    raise SystemExit(main())
