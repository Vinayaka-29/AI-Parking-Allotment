import numpy as np
from app.vision.occupancy import SlotOccupancyEngine
from app.vision.detector import VehicleDetection, VehicleDetector


def test_occupancy_engine_detection_overlap():
    engine = SlotOccupancyEngine(iou_threshold=0.20)
    slot = {
        "slot_id": "A01",
        "polygon": [[100, 50], [200, 50], [200, 150], [100, 150]],
        "status": "AVAILABLE",
    }
    
    # Overlapping vehicle detection bbox
    det = VehicleDetection(
        class_name="car",
        confidence=0.95,
        bbox=[110, 60, 190, 140],
    )
    
    res = engine.compute_slot_status(slot, [det])
    assert res["status"] == "OCCUPIED"
    assert res["confidence"] >= 0.80

    # Non-overlapping vehicle
    det_outside = VehicleDetection(
        class_name="car",
        confidence=0.95,
        bbox=[400, 400, 500, 500],
    )
    res_empty = engine.compute_slot_status(slot, [det_outside])
    assert res_empty["status"] == "AVAILABLE"


def test_detector_annotation_runs_smoothly():
    detector = VehicleDetector(model_name="yolov8n.pt")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    slots = [
        {"slot_id": "A01", "polygon": [[50, 50], [150, 50], [150, 150], [50, 150]], "status": "AVAILABLE"}
    ]
    annotated = detector.annotate_frame(dummy_frame, [], slots)
    assert annotated.shape == (480, 640, 3)
    b64 = detector.encode_base64_jpeg(annotated)
    assert b64.startswith("data:image/jpeg;base64,")
