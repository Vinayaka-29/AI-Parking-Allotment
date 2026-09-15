from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.parking_service import parking_service

router = APIRouter()


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
