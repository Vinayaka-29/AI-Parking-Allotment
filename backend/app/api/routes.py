from __future__ import annotations

import json
from typing import Any, Optional
from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.parking_service import parking_service
from app.cameras import camera_metadata

router = APIRouter()
connected_clients: set[WebSocket] = set()
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


class AllocateRequest(BaseModel):
    vehicle_id: str
    vehicle_type: str = "car"
    preferred_slot: Optional[str] = None


class CameraToggleRequest(BaseModel):
    active: Optional[bool] = None


class LayoutCalibrationRequest(BaseModel):
    slots: list[dict[str, Any]]
    lot_name: Optional[str] = "Default Custom Lot"
    image_width: Optional[int] = None
    image_height: Optional[int] = None


class CalibrationConfirmRequest(BaseModel):
    slots: list[dict[str, Any]]


async def broadcast(message: dict) -> None:
    """Broadcast JSON message to all connected WebSocket clients."""
    dead_clients = set()
    for ws in connected_clients:
        try:
            await ws.send_text(json.dumps(message))
        except Exception:
            dead_clients.add(ws)
    connected_clients.difference_update(dead_clients)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time WebSocket endpoint for slot status changes and occupancy events."""
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        # Send initial status snapshot upon connection
        await websocket.send_text(json.dumps({
            "type": "INITIAL_SNAPSHOT",
            "overview": parking_service.get_overview(),
            "slots": parking_service.get_slots(),
        }))
        while True:
            # Keep connection alive; handle potential client heartbeats
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "PING":
                    await websocket.send_text(json.dumps({"type": "PONG"}))
            except Exception:
                pass
    except (WebSocketDisconnect, Exception):
        connected_clients.discard(websocket)


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "ai-park", "vision_model": "YOLOv8-Nano"}


@router.get("/overview")
async def overview() -> dict:
    return parking_service.get_overview()


@router.get("/slots")
async def slots() -> dict:
    return {"slots": parking_service.get_slots()}


@router.get("/sections")
async def sections() -> dict:
    return {"sections": parking_service.get_sections()}


@router.get("/allocations")
async def allocations() -> dict:
    return {"allocations": parking_service.state_manager.get_allocations()}


@router.get("/vehicles")
async def vehicles() -> dict:
    return {"vehicles": parking_service.state_manager.get_vehicles()}


@router.get("/cameras")
async def cameras() -> dict:
    camera = parking_service.camera
    return {"cameras": [{"camera_id": camera.camera_id, "name": camera.name, "status": "ONLINE",
                          "calibration_status": parking_service.calibration_status}]}


@router.get("/camera/info")
async def camera_info() -> dict:
    """Return metadata for the live CCTV camera source."""
    return {
        "camera": parking_service.camera and {
            **camera_metadata(parking_service.camera),
            "insecam_ref": "http://www.insecam.org/en/view/945438/" if parking_service.camera.camera_id == "CAM_ASAHI_JP" else None,
        },
        "active": parking_service.camera_active,
        "fps": parking_service.camera_fps,
        "source_name": parking_service.camera_source_name,
        "status": parking_service.camera_status,
        "last_frame_timestamp": parking_service.last_frame_timestamp,
        **parking_service._diagnostics(),
    }


@router.get("/events")
async def events() -> dict:
    return {"events": parking_service.get_recent_events()}



@router.post("/allocate")
async def allocate(payload: AllocateRequest) -> dict:
    try:
        if not payload.vehicle_id.strip():
            raise HTTPException(status_code=422, detail="vehicle_id must not be empty")
        # If preferred slot requested and available, assign it
        if payload.preferred_slot and payload.preferred_slot not in parking_service.state_manager.slots:
            raise HTTPException(status_code=400, detail="Unknown preferred slot")
        if payload.preferred_slot and payload.preferred_slot in parking_service.state_manager.slots:
            slot = parking_service.state_manager.slots[payload.preferred_slot]
            if slot["status"] == "AVAILABLE":
                parking_service.state_manager.update_slot_status(
                    payload.preferred_slot, "RESERVED", 0.95, payload.vehicle_id
                )
                result = {
                    "ticket_id": f"TKT-{hash(payload.vehicle_id) % 90000 + 10000}",
                    "vehicle_id": payload.vehicle_id,
                    "vehicle_type": payload.vehicle_type,
                    "slot_id": payload.preferred_slot,
                    "section_id": slot.get("section_id", "A"),
                    "status": "RESERVED",
                    "distance": slot.get("distance_from_entries", 10),
                    "issued_at": parking_service.state_manager.last_analysis_time or "NOW",
                }
            else:
                result = parking_service.allocate_slot(payload.vehicle_id, payload.vehicle_type)
        else:
            result = parking_service.allocate_slot(payload.vehicle_id, payload.vehicle_type)

        await broadcast({
            "type": "SLOT_STATUS_CHANGED",
            "slot_id": result["slot_id"],
            "status": "RESERVED",
            "confidence": 0.95,
        })
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/layout/calibrate")
async def calibrate_layout(payload: LayoutCalibrationRequest) -> dict:
    """Calibrate or update parking layout geometry."""
    try:
        config = {
            "parking_lot": payload.lot_name or "Custom Calibrated Lot",
            "slots": payload.slots,
        }
        result = parking_service.persist_manual_layout(
            payload.slots, payload.image_width, payload.image_height
        )
        await broadcast({
            "type": "LAYOUT_CALIBRATED",
            "slots_count": len(payload.slots),
            "overview": parking_service.get_overview(),
        })
        return {"status": "success", "slots_count": len(payload.slots), **result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to calibrate layout: {str(exc)}") from exc


@router.get("/system")
async def system() -> dict:
    return parking_service.get_system_status()


@router.post("/calibration/start")
async def calibration_start() -> dict:
    parking_service.start_calibration()
    return parking_service.calibration_status_payload()


@router.get("/calibration/status")
async def calibration_status() -> dict:
    return parking_service.calibration_status_payload()


@router.get("/calibration/proposal")
async def calibration_proposal() -> dict:
    return parking_service.calibration_proposal()


@router.post("/calibration/confirm")
async def calibration_confirm(payload: CalibrationConfirmRequest) -> dict:
    try:
        result = parking_service.confirm_calibration(payload.slots)
        await broadcast({"type": "LAYOUT_CALIBRATED", "slots_count": result["slots_count"],
                         "overview": parking_service.get_overview()})
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/calibration/reset")
async def calibration_reset() -> dict:
    return parking_service.reset_calibration()


@router.post("/detect/image")
async def detect_image(file: UploadFile = File(...)) -> dict:
    """Run real-time YOLOv8 vehicle detection & slot occupancy check on uploaded image."""
    try:
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=415, detail="Only JPEG, PNG, and WebP images are supported")
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty image file received")
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image file exceeds the 10 MB limit")
        result = parking_service.process_image_upload(content)
        result["parking_spaces_detected"] = len(result.get("slots", [])) > 0
        await broadcast({
            "type": "DETECTION_COMPLETE",
            "overview": result.get("overview"),
            "inference_time_ms": result.get("inference_time_ms"),
        })
        return result
    except HTTPException:
        raise
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Image detection failed: {str(exc)}") from exc


@router.post("/camera/toggle")
async def toggle_camera(payload: CameraToggleRequest = CameraToggleRequest()) -> dict:
    """Start or stop the camera video feed & continuous AI scanning."""
    res = parking_service.toggle_camera(active=payload.active)
    await broadcast({
        "type": "CAMERA_STATUS_CHANGED",
        "active": res["camera_active"],
    })
    return res


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
    res = parking_service.simulate_random_event()
    await broadcast({
        "type": "SIMULATED_EVENT",
        "overview": res,
    })
    return res


@router.post("/simulate/reset")
async def simulate_reset() -> dict:
    """Reset all parking bays to available."""
    res = parking_service.reset_slots()
    await broadcast({
        "type": "RESET_COMPLETE",
        "overview": res,
    })
    return res
