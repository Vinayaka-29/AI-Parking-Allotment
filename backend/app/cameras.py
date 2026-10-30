from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    name: str
    source_type: str
    url: str
    location: str = "Unknown"
    resolution: str = "unknown"
    fps: int = 3


def active_camera() -> CameraConfig:
    camera_id = settings.camera_id
    name = "Asahi Parking Lot - Japan (Live CCTV)" if camera_id == "CAM_ASAHI_JP" else f"Camera {camera_id}"
    return CameraConfig(
        camera_id=camera_id,
        name=name,
        source_type=settings.camera_source.upper(),
        url=settings.camera_url,
        location="Asahi, Chiba, Japan" if camera_id == "CAM_ASAHI_JP" else "Configured CCTV",
        resolution="1920x1080" if camera_id == "CAM_ASAHI_JP" else "unknown",
        fps=settings.camera_fps,
    )


def camera_metadata(camera: CameraConfig) -> dict[str, object]:
    return {
        "camera_id": camera.camera_id,
        "name": camera.name,
        "location": camera.location,
        "source_url": camera.url,
        "source_type": camera.source_type,
        "resolution": camera.resolution,
        "fps": camera.fps,
    }
