from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    app_name: str = "AI-PARK"
    app_version: str = "0.1.0"
    env_file: str = ".env"

    # Camera Configuration
    # Options: WEBCAM, FILE, URL, SNAPSHOT, SAMPLE
    # SNAPSHOT = JPEG pull-mode camera (fetches a new image at camera_fps intervals)
    # URL       = MJPEG stream (continuous multipart stream)
    camera_source: str = "SNAPSHOT"
    camera_url: str = "http://124.155.121.218:3000/webcapture.jpg?command=snap&channel=1"
    camera_fps: int = 3          # Processing FPS for live mode
    camera_enabled: bool = True  # Master switch for live camera
    camera_stale_after_s: float = 10.0
    camera_fetch_interval_s: float = 1.0
    detect_model: str = "yolov8s.pt"
    detect_conf: float = 0.25
    detect_interval_s: float = 1.0
    detect_tiles: str = "3x2"
    detect_tile_overlap: float = 0.20
    detect_fullframe: bool = False
    occupancy_votes: int = 3

    # Parking layout selection (without extension).
    # The backend looks for configs/parking_layout_{parking_layout}.json first,
    # falling back to configs/parking_layout.json.
    parking_layout: str = "asahi"

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[3] / ".env"),
        extra="ignore"
    )


settings = Settings()
