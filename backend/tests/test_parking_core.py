from app.parking.state_manager import ParkingStateManager


def test_load_layout_and_overview():
    state = ParkingStateManager()
    state.load_layout({
        "slots": [
            {"slot_id": "A01", "section_id": "A", "camera_id": "CAM_01", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "status": "AVAILABLE"},
            {"slot_id": "A02", "section_id": "A", "camera_id": "CAM_01", "polygon": [[11, 0], [21, 0], [21, 10], [11, 10]], "status": "AVAILABLE"},
        ]
    })

    overview = state.get_overview()
    assert overview["total_slots"] == 2
    assert overview["available"] == 2


def test_allocate_slot_returns_reservation():
    state = ParkingStateManager()
    state.load_layout({
        "slots": [
            {"slot_id": "A01", "section_id": "A", "camera_id": "CAM_01", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "status": "AVAILABLE", "distance_from_entries": 5},
            {"slot_id": "A02", "section_id": "A", "camera_id": "CAM_01", "polygon": [[11, 0], [21, 0], [21, 10], [11, 10]], "status": "AVAILABLE", "distance_from_entries": 7},
        ]
    })

    result = state.allocate_slot("vehicle-1", "car")
    assert result["slot_id"] == "A01"
    assert result["status"] == "RESERVED"
    assert state.slots["A01"]["status"] == "RESERVED"


def test_vision_preserves_reservation_and_reports_conflict():
    state = ParkingStateManager()
    state.load_layout({
        "slots": [
            {"slot_id": "A01", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "status": "AVAILABLE"},
        ]
    })
    state.allocate_slot("vehicle-1")

    detection = type("Detection", (), {
        "bbox": [1, 1, 9, 9],
        "confidence": 0.95,
    })()
    for _ in range(3):
        state.process_vision_detections([detection], image_shape=(10, 10))

    slot = state.slots["A01"]
    assert slot["reservation_status"] == "RESERVED"
    assert slot["physical_status"] == "OCCUPIED"
    assert slot["status"] == "CONFLICT"


def test_unknown_camera_state_does_not_make_slot_available():
    state = ParkingStateManager()
    state.load_layout({
        "slots": [
            {"slot_id": "A01", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "status": "AVAILABLE"},
        ]
    })
    state.mark_camera_unknown()
    assert state.slots["A01"]["status"] == "UNKNOWN"
