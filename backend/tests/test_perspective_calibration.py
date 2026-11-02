import time
import cv2
import numpy as np
from pathlib import Path
from app.services.parking_service import parking_service
from app.vision.detector import VehicleDetector
from app.vision.parking_space_proposals import infer_parking_spaces


def test_user_image_analysis_rejects_unverified_vehicle_lattice():
    """Actual footage must not become a false confirmed layout."""
    img_path = Path(r"C:\Users\Tilak M K\.gemini\antigravity\brain\77c68d6e-95d3-44f6-84f4-6d8275ec551d\.user_uploaded\media_1793613209606_7503d367.jpg")
    assert img_path.exists(), "User test image not found"

    with open(img_path, "rb") as f:
        file_bytes = f.read()

    # Measure total execution time
    t0 = time.time()
    result = parking_service.process_image_upload(file_bytes)
    total_time = time.time() - t0

    # 1. Total time must be well under 10 seconds
    assert total_time <= 10.0, f"Analysis took {total_time:.2f}s, exceeding 10s budget"

    # 2. Must detect parked vehicles
    assert result["total_detected_vehicles"] >= 20, "Should detect vehicles in the lot"

    # Vehicle detections are useful, but they are not parking-space boundaries.
    slots = result["slots"]
    assert slots == []

    # No reliable boundary evidence means calibration remains unconfirmed.
    calib_res = parking_service.auto_calibrate_image(file_bytes)
    assert calib_res["status"] == "NO_PROPOSAL"
    assert calib_res["polygons"] == []
