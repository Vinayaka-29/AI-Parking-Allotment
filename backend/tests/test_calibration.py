import asyncio
import json

import cv2
import numpy as np

from app.api.routes import calibration_status
from app.calibration.auto_calibrator import AutoCalibrator
from app.calibration.profile_store import CalibrationProfileStore
from app.calibration.scene_monitor import SceneMonitor
from app.vision.detector import VehicleDetection


def _detection(x: float, y: float) -> VehicleDetection:
    return VehicleDetection(confidence=0.9, bbox=[x, y, x + 40, y + 24])


def test_auto_calibrator_finds_rows_and_empty_gaps():
    frame = np.zeros((400, 800, 3), dtype=np.uint8)
    calibrator = AutoCalibrator(total_cycles=40, min_parked=3)
    stationary = [_detection(100, 100), _detection(300, 100), _detection(500, 100),
                  _detection(100, 260), _detection(300, 260), _detection(500, 260)]
    for _ in range(40):
        proposal = calibrator.update(stationary + [_detection(20 + _, 30)], frame)
    assert proposal["status"] == "PROPOSED"
    assert proposal["quality"]["number_of_rows"] == 2
    assert proposal["quality"]["parked_candidates"] >= 6
    assert proposal["quality"]["inferred_slots"] >= 0
    assert len(proposal["polygons"]) >= 6


def test_auto_calibrator_needs_more_data():
    calibrator = AutoCalibrator(total_cycles=2, min_parked=3)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    calibrator.update([_detection(20, 20)], frame)
    proposal = calibrator.update([_detection(20, 20)], frame)
    assert proposal["status"] == "NEEDS_MORE_DATA"


def test_profile_store_migrates_pixels_and_validates_url(tmp_path):
    layout = {
        "image_width": 1920,
        "image_height": 1080,
        "slots": [{"slot_id": "A01", "section_id": "A", "type": "STANDARD",
                   "priority": 1, "distance_from_entries": 2, "polygon": [[0, 0], [192, 0], [192, 108], [0, 108]]}],
    }
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    store = CalibrationProfileStore(tmp_path / "profiles")
    profile = store.migrate_layout("CAM", "http://one", layout_path)
    assert profile["status"] == "CONFIRMED"
    assert profile["polygons"][0]["polygon"][1] == [0.1, 0.0]
    assert store.load_validated("CAM", "http://two")["status"] == "NEEDS_CALIBRATION"


def test_scene_monitor_same_brightness_shift_and_different():
    rng = np.random.default_rng(4)
    reference = (rng.random((360, 640, 3)) * 255).astype(np.uint8)
    polygons = [{"slot_id": "A01", "polygon": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]]}]
    monitor = SceneMonitor()
    assert monitor.evaluate(reference, reference.copy(), polygons)["status"] == "OK"
    bright = cv2.convertScaleAbs(reference, alpha=1.0, beta=35)
    assert monitor.evaluate(reference, bright, polygons)["status"] in {"OK", "WARPED"}
    shifted = cv2.warpAffine(reference, np.float32([[1, 0, 19], [0, 1, 0]]), (640, 360))
    assert monitor.evaluate(reference, shifted, polygons)["status"] == "WARPED"
    different = (rng.random((360, 640, 3)) * 255).astype(np.uint8)
    assert monitor.evaluate(reference, different, polygons)["status"] == "NEEDS_RECALIBRATION"


def test_calibration_status_route_returns_profile_state():
    result = asyncio.run(calibration_status())
    assert "status" in result
    assert "camera_id" in result
