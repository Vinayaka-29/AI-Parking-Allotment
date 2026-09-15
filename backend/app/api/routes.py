from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.services.parking_service import parking_service

router = APIRouter()
connected_clients: set[WebSocket] = set()


class AllocateRequest(BaseModel):
    vehicle_id: str
    vehicle_type: str = "car"


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "ai-park"}


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
    camera_ids = sorted({slot["camera_id"] for slot in parking_service.state_manager.slots.values()})
    return {"cameras": [{"camera_id": camera_id, "status": "ONLINE"} for camera_id in camera_ids]}


@router.get("/events")
async def events() -> dict:
    return {"events": parking_service.get_recent_events()}


@router.post("/allocate")
async def allocate(payload: AllocateRequest) -> dict:
    try:
        result = parking_service.allocate_slot(payload.vehicle_id, payload.vehicle_type)
        await broadcast({"event": "VEHICLE_ASSIGNED", **result})
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/system")
async def system() -> dict:
    return parking_service.get_system_status()


@router.post("/occupancy")
async def occupancy(payload: list[dict]) -> dict:
    updates = parking_service.update_occupancy(payload)
    await broadcast({"event": "OCCUPANCY_UPDATED", "updates": updates})
    return {"updates": updates}


@router.websocket("/ws")
async def websocket_updates(websocket: WebSocket) -> None:
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        await websocket.send_json({"event": "SNAPSHOT", "overview": parking_service.get_overview(), "slots": parking_service.get_slots()})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.discard(websocket)


async def broadcast(payload: dict) -> None:
    disconnected = []
    for client in connected_clients:
        try:
            await client.send_json(payload)
        except Exception:
            disconnected.append(client)
    for client in disconnected:
        connected_clients.discard(client)
