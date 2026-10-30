import time

import numpy as np

from app.parking.state_manager import ParkingStateManager
from app.vision.camera_worker import CameraWorker
from app.vision.detection_worker import DetectionWorker
from app.vision.detector import VehicleDetection
from app.vision.occupancy import SlotOccupancyEngine


def test_a01_bbox_anchor_marks_slot_occupied_without_confidence_multiplication():
    engine = SlotOccupancyEngine()
    result = engine.compute_slot_status(
        {
            "slot_id": "A01",
            "polygon": [[845, 60], [931, 60], [931, 109], [845, 109]],
        },
        [VehicleDetection(confidence=0.30, bbox=[880.4, 69.2, 915.6, 101.9])],
        image_shape=(1080, 1920),
    )
    assert result["status"] == "OCCUPIED"
    assert result["occupancy_score"] >= 1.0
    assert result["confidence"] == 0.3


def test_state_manager_requires_three_agreeing_votes():
    state = ParkingStateManager()
    state.load_layout({"slots": [{"slot_id": "A01", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}]})
    detection = VehicleDetection(confidence=0.9, bbox=[1, 1, 9, 9])
    for _ in range(2):
        state.process_vision_detections([detection], image_shape=(10, 10))
        assert state.slots["A01"]["status"] == "AVAILABLE"
    state.process_vision_detections([detection], image_shape=(10, 10))
    assert state.slots["A01"]["status"] == "OCCUPIED"


def test_unknown_is_immediate_for_camera_failure():
    state = ParkingStateManager()
    state.load_layout({"slots": [{"slot_id": "A01", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}]})
    state.mark_camera_unknown()
    assert state.slots["A01"]["status"] == "UNKNOWN"


def test_tile_bounds_are_configured_grid_and_coordinates_merge():
    class FakeDetector:
        model_name = "fake"

        def detect(self, frame, conf_threshold):
            height, width = frame.shape[:2]
            return [VehicleDetection(confidence=0.8, bbox=[0, 0, width, height])]

    camera = object()
    worker = DetectionWorker.__new__(DetectionWorker)
    worker.tile_columns = 3
    worker.tile_rows = 2
    worker.tile_overlap = 0.2
    worker.full_frame = False
    worker.conf_threshold = 0.25
    assert len(worker._tile_bounds(1920, 1080)) == 6

    worker.detector = FakeDetector()
    result = worker.detect_frame(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert result
    assert all(0 <= value <= limit for detection in result for value, limit in zip(detection.bbox, [1920, 1080, 1920, 1080]))


def test_camera_worker_retries_after_failure():
    class FakeFrame:
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        timestamp = "now"
        frame_id = 1
        is_cached = False

    class FakeSource:
        attempts = 0

        def read(self):
            FakeSource.attempts += 1
            if FakeSource.attempts == 1:
                raise RuntimeError("temporary failure")
            return FakeFrame()

        def release(self):
            return None

    worker = CameraWorker(lambda: FakeSource(), fetch_interval_s=0.01, stale_after_s=1)
    worker._backoff_s = 0.01
    worker.start()
    deadline = time.time() + 1
    while time.time() < deadline and worker.get_latest() is None:
        time.sleep(0.01)
    worker.stop()
    assert worker.get_latest() is not None
    assert FakeSource.attempts >= 2
