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
