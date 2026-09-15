from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import cv2


@dataclass
class VideoFrame:
    frame: Any
    timestamp: str
    camera_id: str
    frame_id: int


class CameraSource:
    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self._frame_id = 0

    def read(self) -> VideoFrame | None:
        raise NotImplementedError

    def release(self) -> None:
        return None

    def _wrap(self, frame: Any) -> VideoFrame:
        self._frame_id += 1
        return VideoFrame(
            frame=frame,
            timestamp=datetime.now(timezone.utc).isoformat(),
            camera_id=self.camera_id,
            frame_id=self._frame_id,
        )


class OpenCVCameraSource(CameraSource):
    """OpenCV-backed source for webcams, files, phone streams, and RTSP URLs."""

    def __init__(self, source: str | int, camera_id: str = "CAM_01") -> None:
        super().__init__(camera_id)
        self.capture = cv2.VideoCapture(source)

    def read(self) -> VideoFrame | None:
        ok, frame = self.capture.read()
        return self._wrap(frame) if ok else None

    def release(self) -> None:
        self.capture.release()


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
