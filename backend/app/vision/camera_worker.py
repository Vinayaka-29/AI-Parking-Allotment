from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

from app.vision.camera_sources import CameraSource, VideoFrame

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraFrame:
    frame: object
    timestamp: str
    received_at: float
    frame_id: int
    is_stale: bool = False


class CameraWorker:
    """Owns one camera source and publishes only the newest frame."""

    def __init__(
        self,
        source_factory: Callable[[], CameraSource],
        *,
        fetch_interval_s: float = 1.0,
        stale_after_s: float = 10.0,
        camera_id: str = "CAM_01",
    ) -> None:
        self.source_factory = source_factory
        self.fetch_interval_s = max(0.05, fetch_interval_s)
        self.stale_after_s = max(0.1, stale_after_s)
        self.camera_id = camera_id
        self._lock = threading.Lock()
        self._source: CameraSource | None = None
        self._latest: CameraFrame | None = None
        self._last_error: str | None = None
        self._last_success_at: float | None = None
        self._status = "OFFLINE"
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._backoff_s = 1.0

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name=f"camera-worker-{self.camera_id}",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=5)
        with self._lock:
            self._close_source_locked()
            self._thread = None
            self._status = "OFFLINE"

    def get_latest(self) -> CameraFrame | None:
        with self._lock:
            latest = self._latest
            if latest is None:
                return None
            age = time.time() - latest.received_at
            return CameraFrame(
                frame=latest.frame,
                timestamp=latest.timestamp,
                received_at=latest.received_at,
                frame_id=latest.frame_id,
                is_stale=age > self.stale_after_s,
            )

    def diagnostics(self) -> dict[str, object]:
        with self._lock:
            age = None if self._last_success_at is None else max(0.0, time.time() - self._last_success_at)
            status = "OFFLINE" if age is None else "STALE" if age > self.stale_after_s else self._status
            return {
                "camera_status": status,
                "frame_age_s": round(age, 3) if age is not None else None,
                "last_error": self._last_error,
            }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                source = self._get_source()
                video_frame = source.read()
                if video_frame is None or video_frame.frame is None or video_frame.is_cached:
                    raise RuntimeError("camera source returned no fresh frame")
                now = time.time()
                with self._lock:
                    self._latest = CameraFrame(
                        frame=video_frame.frame,
                        timestamp=video_frame.timestamp,
                        received_at=now,
                        frame_id=video_frame.frame_id,
                    )
                    self._last_success_at = now
                    self._last_error = None
                    self._status = "LIVE"
                    self._backoff_s = 1.0
                self._stop_event.wait(self.fetch_interval_s)
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                    self._status = "STALE" if self._last_success_at else "OFFLINE"
                    self._close_source_locked()
                    delay = self._backoff_s
                    self._backoff_s = min(15.0, self._backoff_s * 2.0)
                logger.warning("Camera worker %s failed: %s; retrying in %.1fs", self.camera_id, exc, delay)
                self._stop_event.wait(delay)

    def _get_source(self) -> CameraSource:
        with self._lock:
            if self._source is None:
                self._source = self.source_factory()
            return self._source

    def _close_source_locked(self) -> None:
        source = self._source
        self._source = None
        if source is not None:
            try:
                source.release()
            except Exception:
                logger.exception("Failed to release camera source %s", self.camera_id)
