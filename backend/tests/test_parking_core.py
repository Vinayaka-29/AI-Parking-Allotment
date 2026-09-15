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
