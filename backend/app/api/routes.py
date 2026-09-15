from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.parking_service import parking_service

router = APIRouter()


class AllocateRequest(BaseModel):
    vehicle_id: str
    vehicle_type: str = "car"


class CameraToggleRequest(BaseModel):
    active: Optional[bool] = None


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "ai-park", "vision_model": "YOLOv8-Nano"}


@router.get("/overview")
async def overview() -> dict:
    return parking_service.get_overview()


@router.get("/slots")
async def slots() -> dict:
    return {"slots": parking_service.get_slots()}


@router.get("/events")
async def events() -> dict:
    return {"events": parking_service.get_recent_events()}


@router.post("/allocate")
async def allocate(payload: AllocateRequest) -> dict:
    try:
        result = parking_service.allocate_slot(payload.vehicle_id, payload.vehicle_type)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/system")
async def system() -> dict:
    return parking_service.get_system_status()


@router.post("/detect/image")
async def detect_image(file: UploadFile = File(...)) -> dict:
    """Run real-time YOLOv8 vehicle detection & slot occupancy check on uploaded image."""
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty image file received")
        result = parking_service.process_image_upload(content)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image detection failed: {str(exc)}") from exc


@router.post("/camera/toggle")
async def toggle_camera(payload: CameraToggleRequest = CameraToggleRequest()) -> dict:
    """Start or stop the camera video feed & continuous AI scanning."""
    return parking_service.toggle_camera(active=payload.active)


@router.get("/camera/feed")
async def camera_feed():
    """Live MJPEG video stream with real-time YOLO detections and slot overlays."""
    return StreamingResponse(
        parking_service.generate_camera_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/simulate/event")
async def simulate_event() -> dict:
    """Trigger a simulated vehicle arrival or departure."""
    return parking_service.simulate_random_event()


@router.post("/simulate/reset")
async def simulate_reset() -> dict:
    """Reset all parking bays to available."""
    return parking_service.reset_slots()
