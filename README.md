# AI-PARK

AI-PARK is a modular real-time parking intelligence platform designed for a miniature cardboard parking lot and scalable to larger camera-based systems.

## Milestone delivered

This repository implements the first working milestone described in the project brief:

- phone camera / simulated video input
- parking layout configuration
- vehicle detection abstraction with YOLO-ready interface
- slot occupancy engine
- state management for AVAILABLE / OCCUPIED / RESERVED / UNKNOWN
- FastAPI backend with WebSocket live updates
- React dashboard for the live parking map and event stream
- database-ready models and schema layer

## Architecture

- Backend: FastAPI + SQLAlchemy-ready models
- Vision: detection and occupancy modules
- Parking logic: state manager and allocation engine
- Frontend: React + Vite dashboard

## Automatic CCTV calibration

When a camera starts without a confirmed profile, the backend analyzes new
frames, extracts parking-marking geometry with OpenCV, groups boundaries into
perspective-aware polygons, and validates the complete layout before saving it.
Repeated/stale frames are not counted as independent calibration evidence.

Vehicle detections are used for occupancy and supporting calibration evidence,
but vehicle positions alone cannot confirm parking boundaries. Layout validation
rejects invalid polygons, excessive overlap, inconsistent dimensions, and
vehicle-only proposals. Rejected proposals remain available through the
calibration API for the manual editor fallback.

An optional `ParkingSpaceSegmenter` adapter is available under
`backend/app/calibration/parking_space_segmenter.py`. It only loads an
explicitly supplied, locally existing segmentation weight file; it never
downloads weights and does not use the vehicle detector as a segmentation
model. Without verified parking-space weights, OpenCV remains the automatic
calibration engine.

## Run backend

```bash
cd backend
pip install -r ../requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Run frontend

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

The commands above install dependencies in the active system/user environments; a
virtual environment is not required.

## Validation and external benchmark

Run the deterministic backend tests without contacting a camera:

```bash
cd backend
PYTHONPATH=. python -m pytest -q
```

The Asahi CCTV check is an explicit external benchmark and is not collected by
pytest. Run it only when access to that public camera is authorized:

```bash
cd backend
PYTHONPATH=. python test_asahi_pipeline.py
```

Camera URLs, source types, and allowed frontend origins are configured through
environment variables. Do not commit credentials or private camera URLs.

## API

- GET /api/health
- GET /api/overview
- GET /api/slots
- GET /api/events
- POST /api/allocate
- WS /ws

## Notes

The detection layer is intentionally modular so it can be replaced with a YOLO model later without changing the parking intelligence layer.
The project is maintained on the `main` branch.
