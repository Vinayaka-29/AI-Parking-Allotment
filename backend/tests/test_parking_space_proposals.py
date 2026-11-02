import cv2
import numpy as np

from app.vision.parking_space_proposals import (
    infer_parking_spaces,
    propose_parking_spaces,
    suppress_duplicate_proposals,
    validate_polygon,
    validate_proposals,
)
from app.vision.detector import VehicleDetection


def test_empty_image_returns_no_proposals():
    result = propose_parking_spaces(np.zeros((240, 320, 3), dtype=np.uint8))
    assert result.status == "NO_PROPOSALS"
    assert result.proposals == []
    assert (result.image_width, result.image_height) == (320, 240)


def test_synthetic_perspective_lines_produce_variable_bay_count():
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    bays = [
        np.array([[60, 80], [180, 65], [190, 210], [70, 230]], dtype=np.int32),
        np.array([[205, 62], [325, 50], [340, 195], [215, 210]], dtype=np.int32),
        np.array([[370, 45], [500, 35], [525, 175], [385, 190]], dtype=np.int32),
    ]
    for bay in bays:
        cv2.polylines(image, [bay], True, (255, 255, 255), 5)

    result = propose_parking_spaces(image)

    assert result.status == "PROPOSALS_FOUND"
    assert len(result.proposals) >= 2
    assert len(result.proposals) != 16
    for proposal in result.proposals:
        assert proposal["status"] == "NEEDS_REVIEW"
        assert all(0.0 <= value <= 1.0 for point in proposal["polygon"] for value in point)


def test_invalid_polygons_are_rejected():
    assert not validate_polygon([[0.1, 0.1], [0.2, 0.2]])["valid"]
    assert not validate_polygon([[-0.1, 0.1], [0.2, 0.1], [0.2, 0.2]])["valid"]
    assert not validate_polygon([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])["valid"]


def test_duplicate_proposals_keep_highest_confidence():
    polygon = [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]]
    proposals = [
        {"polygon": polygon, "confidence": 0.6},
        {"polygon": polygon, "confidence": 0.9},
    ]
    result = suppress_duplicate_proposals(proposals)
    assert len(result) == 1
    assert result[0]["confidence"] == 0.9


def test_overlapping_layout_is_not_eligible_for_confirmation():
    proposals = [
        {"polygon": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.5], [0.1, 0.5]], "confidence": 0.9, "evidence": "line-pair"},
        {"polygon": [[0.3, 0.1], [0.6, 0.1], [0.6, 0.5], [0.3, 0.5]], "confidence": 0.9, "evidence": "line-pair"},
    ]
    result = validate_proposals(proposals, 1000, 600)
    assert result["eligible"] is False
    assert result["confidence"] != "HIGH"


def test_vehicle_lattice_is_not_eligible_for_confirmation():
    proposals = [
        {"polygon": [[0.1, 0.2], [0.2, 0.2], [0.2, 0.5], [0.1, 0.5]], "confidence": 0.7, "evidence": "vehicle-lattice"},
        {"polygon": [[0.3, 0.2], [0.4, 0.2], [0.4, 0.5], [0.3, 0.5]], "confidence": 0.7, "evidence": "vehicle-gap"},
    ]
    result = validate_proposals(proposals, 1000, 600)
    assert result["eligible"] is False
    assert any("Vehicle positions" in issue for issue in result["issues"])


def test_proposals_report_original_dimensions_and_normalized_coordinates():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (100, 80), (255, 255, 255), 4)
    result = propose_parking_spaces(image)
    assert (result.image_width, result.image_height) == (200, 100)
    for proposal in result.proposals:
        assert all(0.0 <= coordinate <= 1.0 for point in proposal["polygon"] for coordinate in point)


def test_vehicle_geometry_is_not_used_for_automatic_calibration():
    image = np.zeros((400, 800, 3), dtype=np.uint8)
    detections = [
        VehicleDetection(confidence=0.9, bbox=[80, 120, 160, 220]),
        VehicleDetection(confidence=0.9, bbox=[560, 120, 640, 220]),
        VehicleDetection(confidence=0.9, bbox=[720, 120, 800, 220]),
    ]

    result = infer_parking_spaces(image, detections)

    assert not any(
        proposal.get("evidence") in {"vehicle-lattice", "vehicle-gap"}
        for proposal in result.proposals
    )
