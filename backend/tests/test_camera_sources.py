from unittest.mock import patch

import numpy as np

from app.vision.camera_sources import SnapshotCameraSource


def test_snapshot_source_marks_cached_frames_without_refreshing_timestamp():
    source = SnapshotCameraSource("http://camera.test/snapshot")
    response = type("Response", (), {
        "__enter__": lambda self: self,
        "__exit__": lambda self, exc_type, exc, tb: None,
        "read": lambda self: b"not-a-jpeg",
    })()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with patch("app.vision.camera_sources.urllib.request.urlopen", return_value=response):
        with patch("app.vision.camera_sources.cv2.imdecode", side_effect=[frame, None]):
            first = source.read()
            cached = source.read()

    assert first is not None and not first.is_cached
    assert cached is not None and cached.is_cached
    assert cached.timestamp == first.timestamp
    assert cached.frame_id == first.frame_id
