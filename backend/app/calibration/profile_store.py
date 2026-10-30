from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class CalibrationProfileStore:
    """Atomic, per-camera storage for normalized calibration profiles."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parents[2] / "data" / "calibrations"
        self._lock = threading.RLock()

    @staticmethod
    def url_hash(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def _directory(self, camera_id: str) -> Path:
        return self.root / camera_id

    def profile_path(self, camera_id: str) -> Path:
        return self._directory(camera_id) / "profile.json"

    def reference_path(self, camera_id: str) -> Path:
        return self._directory(camera_id) / "reference.jpg"

    def fingerprint_path(self, camera_id: str) -> Path:
        return self._directory(camera_id) / "scene.npz"

    def load(self, camera_id: str) -> dict[str, Any] | None:
        with self._lock:
            path = self.profile_path(camera_id)
            if not path.exists():
                return None
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

    def load_validated(self, camera_id: str, camera_url: str) -> dict[str, Any] | None:
        profile = self.load(camera_id)
        if profile is None:
            return None
        if profile.get("camera_url_hash") != self.url_hash(camera_url):
            profile = dict(profile)
            profile["status"] = "NEEDS_CALIBRATION"
        return profile

    def save(
        self,
        profile: dict[str, Any],
        *,
        reference: np.ndarray | None = None,
        fingerprint: tuple[list[cv2.KeyPoint], np.ndarray | None] | None = None,
    ) -> dict[str, Any]:
        camera_id = str(profile["camera_id"])
        directory = self._directory(camera_id)
        with self._lock:
            directory.mkdir(parents=True, exist_ok=True)
            profile = dict(profile)
            profile["version"] = int(profile.get("version", 0)) + 1
            temp_path = directory / "profile.json.tmp"
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(profile, handle, indent=2)
            os.replace(temp_path, self.profile_path(camera_id))
            if reference is not None:
                temp_reference = directory / "reference.jpg.tmp.jpg"
                success, encoded = cv2.imencode(".jpg", reference)
                if not success:
                    raise OSError("Unable to write calibration reference image")
                temp_reference.write_bytes(encoded.tobytes())
                os.replace(temp_reference, self.reference_path(camera_id))
            if fingerprint is not None:
                keypoints, descriptors = fingerprint
                points = np.asarray([kp.pt for kp in keypoints], dtype=np.float32)
                temp_npz = directory / "scene.npz.tmp"
                np.savez_compressed(temp_npz, keypoints=points, descriptors=descriptors)
                actual_temp = Path(str(temp_npz) + ".npz") if not temp_npz.exists() else temp_npz
                os.replace(actual_temp, self.fingerprint_path(camera_id))
            return profile

    def delete(self, camera_id: str) -> None:
        with self._lock:
            directory = self._directory(camera_id)
            for path in (self.profile_path(camera_id), self.reference_path(camera_id), self.fingerprint_path(camera_id)):
                path.unlink(missing_ok=True)
            try:
                directory.rmdir()
            except OSError:
                pass

    def migrate_layout(self, camera_id: str, camera_url: str, layout_path: Path) -> dict[str, Any] | None:
        if self.load(camera_id) is not None or not layout_path.exists():
            return self.load(camera_id)
        with layout_path.open("r", encoding="utf-8") as handle:
            layout = json.load(handle)
        width = int(layout.get("image_width") or 0)
        height = int(layout.get("image_height") or 0)
        if width <= 0 or height <= 0:
            raise ValueError("Migrated layout must define image_width and image_height")
        slots = []
        for raw in layout.get("slots", []):
            polygon = [[round(float(x) / width, 8), round(float(y) / height, 8)] for x, y in raw["polygon"]]
            slots.append({key: raw[key] for key in ("slot_id", "section_id", "type", "priority", "distance_from_entries") if key in raw}
                         | {"polygon": polygon})
        now = datetime.now(timezone.utc).isoformat()
        return self.save({
            "camera_id": camera_id,
            "camera_url_hash": self.url_hash(camera_url),
            "frame_width": width,
            "frame_height": height,
            "polygons": slots,
            "status": "CONFIRMED",
            "created_at": now,
            "confirmed_at": now,
            "source": "manual",
            "scene_fingerprint": None,
            "version": 0,
            "quality": {"migrated": True},
        })


def build_fingerprint(frame: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    orb = cv2.ORB_create(nfeatures=1200)
    return orb.detectAndCompute(gray, None)
