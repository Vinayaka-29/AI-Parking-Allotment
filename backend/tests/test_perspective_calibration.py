import time
import cv2
import numpy as np
from pathlib import Path
from app.services.parking_service import parking_service
from app.vision.detector import VehicleDetector
from app.vision.parking_space_proposals import infer_parking_spaces


def test_user_image_analysis_within_budget():
    """Verify all 7 user requirements on the uploaded parking lot image."""
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

    # 3. Must reconstruct bays including occupied and empty spaces
    slots = result["slots"]
    assert len(slots) >= 30, f"Expected at least 30 slots, got {len(slots)}"

    occupied = [s for s in slots if s["status"] == "OCCUPIED"]
    available = [s for s in slots if s["status"] == "AVAILABLE"]

    # 4. Must identify BOTH parked and empty parking spaces
    assert len(occupied) >= 15, "Should identify occupied slots"
    assert len(available) >= 10, "Should identify empty/available slots"

    # 5. Check foreground row (contains the 4 prominent empty spaces in the center)
    fg_section = slots[-1]["section_id"]
    fg_slots = [s for s in slots if s["section_id"] == fg_section]
    fg_available = [s for s in fg_slots if s["status"] == "AVAILABLE"]
    assert len(fg_available) >= 3, "Should accurately identify the central empty bays in foreground"

    # 6. Must produce valid perspective polygons (4 vertices normalized between 0 and 1)
    for s in slots:
        poly = s["polygon"]
        assert len(poly) == 4, f"Slot {s['slot_id']} must have 4 quadrilateral vertices"
        for pt in poly:
            assert 0.0 <= pt[0] <= 1.0, f"X coord {pt[0]} out of bounds in {s['slot_id']}"
            assert 0.0 <= pt[1] <= 1.0, f"Y coord {pt[1]} out of bounds in {s['slot_id']}"

    # 7. One-shot auto-calibration produces reusable profile
    calib_res = parking_service.auto_calibrate_image(file_bytes)
    assert calib_res["status"] == "PROPOSED"
    assert len(calib_res["polygons"]) >= 30
    assert "reference_jpeg" in calib_res

