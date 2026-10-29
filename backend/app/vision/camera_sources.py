from __future__ import annotations

import logging
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoFrame:
    frame: Any
    timestamp: str
    camera_id: str
    frame_id: int
    is_cached: bool = False
    received_at: str = ""


class CameraSource:
    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self._frame_id = 0

    def read(self) -> VideoFrame | None:
        raise NotImplementedError

    def release(self) -> None:
        return None

    def _wrap(
        self,
        frame: Any,
        *,
        timestamp: str | None = None,
        is_cached: bool = False,
        frame_id: int | None = None,
    ) -> VideoFrame:
        self._frame_id += 1
        received_at = datetime.now(timezone.utc).isoformat()
        return VideoFrame(
            frame=frame,
            timestamp=timestamp or received_at,
            camera_id=self.camera_id,
            frame_id=frame_id or self._frame_id,
            is_cached=is_cached,
            received_at=received_at,
        )


class OpenCVCameraSource(CameraSource):
    """OpenCV-backed source for webcams, files, phone streams, and RTSP/MJPEG URLs."""

    def __init__(self, source: str | int, camera_id: str = "CAM_01") -> None:
        super().__init__(camera_id)
        self.capture = cv2.VideoCapture(source)

    def read(self) -> VideoFrame | None:
        ok, frame = self.capture.read()
        return self._wrap(frame) if ok else None

    def release(self) -> None:
        self.capture.release()


class SnapshotCameraSource(CameraSource):
    """
    Pull-mode JPEG snapshot camera source.

    Works with cameras that expose a single JPEG endpoint (e.g. Hi3516 cameras
    served via insecam).  The URL is fetched at each read() call.

    Camera: Asahi Parking Lot, Japan — insecam.org/en/view/945438/
    Endpoint: http://124.155.121.218:3000/webcapture.jpg?command=snap&channel=1
    """

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "image/jpeg,image/*,*/*",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    def __init__(self, url: str, camera_id: str = "CAM_ASAHI_JP", timeout: int = 5) -> None:
        super().__init__(camera_id)
        self.url = url
        self.timeout = timeout
        self._last_frame: np.ndarray | None = None
        self._last_timestamp: str | None = None
        logger.info("SnapshotCameraSource initialised: %s", url)

    def read(self) -> VideoFrame | None:
        try:
            req = urllib.request.Request(self.url, headers=self.HEADERS)
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = response.read()

            if not data:
                logger.warning("SnapshotCameraSource: empty response from %s", self.url)
                return self._cached_frame()

            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is None:
                logger.warning("SnapshotCameraSource: could not decode JPEG from %s", self.url)
                return self._cached_frame()

            self._last_frame = frame
            self._last_timestamp = datetime.now(timezone.utc).isoformat()
            return self._wrap(frame, timestamp=self._last_timestamp)

        except (OSError, TimeoutError, urllib.error.URLError, cv2.error) as exc:
            logger.error("SnapshotCameraSource read error: %s", exc)
            return self._cached_frame()

    def _cached_frame(self) -> VideoFrame | None:
        if self._last_frame is None or self._last_timestamp is None:
            return None
        return self._wrap(
            self._last_frame,
            timestamp=self._last_timestamp,
            is_cached=True,
            frame_id=self._frame_id,
        )

    def release(self) -> None:
        self._last_frame = None
        self._last_timestamp = None


class VideoFileSource(OpenCVCameraSource):
    def __init__(self, path: str, camera_id: str = "CAM_01") -> None:
        super().__init__(path, camera_id)


class WebcamSource(OpenCVCameraSource):
    def __init__(self, index: int = 0, camera_id: str = "CAM_01") -> None:
        super().__init__(index, camera_id)


class PhoneCameraSource(OpenCVCameraSource):
    def __init__(self, url: str, camera_id: str = "CAM_01") -> None:
        super().__init__(url, camera_id)


class RTSPCameraSource(OpenCVCameraSource):
    def __init__(self, url: str, camera_id: str = "CAM_01") -> None:
        super().__init__(url, camera_id)
