from __future__ import annotations

import platform
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class CameraSettings:
    index: int
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    show_settings: bool = False


@dataclass(frozen=True, slots=True)
class CameraSnapshot:
    connected: bool
    width: int
    height: int
    fps: float
    recording_active: bool
    recording: bool
    frames_written: int
    error: str | None


@dataclass(slots=True)
class _RecordingRequest:
    path: Path
    start_event: threading.Event
    armed_event: threading.Event


def capture_backend() -> int:
    """Prefer DirectShow on Windows; use OpenCV's default elsewhere for development."""
    return cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY


def discover_camera_indices(max_index: int = 10) -> list[int]:
    """Probe camera indices. A device counts only after a frame can be read."""
    found: list[int] = []
    backend = capture_backend()
    for index in range(max_index):
        capture = cv2.VideoCapture(index, backend)
        try:
            if not capture.isOpened():
                continue
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = capture.read()
            if ok and frame is not None:
                found.append(index)
        finally:
            capture.release()
    return found


class CameraWorker:
    """Owns one VideoCapture and optional VideoWriter on a dedicated thread."""

    def __init__(self, settings: CameraSettings) -> None:
        self.settings = settings
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._connected = False
        self._actual_width = 0
        self._actual_height = 0
        self._actual_fps = settings.fps
        self._error: str | None = None
        self._request: _RecordingRequest | None = None
        self._stop_recording = False
        self._recording = False
        self._frames_written = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f"camera-{self.settings.index}",
            daemon=True,
        )
        self._thread.start()

    def arm_recording(self, path: Path, start_event: threading.Event) -> threading.Event:
        armed_event = threading.Event()
        with self._lock:
            if self._request is not None or self._recording:
                raise RuntimeError(f"Camera {self.settings.index} is already armed")
            self._error = None
            self._frames_written = 0
            self._stop_recording = False
            self._request = _RecordingRequest(path, start_event, armed_event)
        return armed_event

    def request_stop_recording(self) -> None:
        with self._lock:
            self._stop_recording = True

    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def snapshot(self) -> CameraSnapshot:
        with self._lock:
            return CameraSnapshot(
                connected=self._connected,
                width=self._actual_width,
                height=self._actual_height,
                fps=self._actual_fps,
                recording_active=self._request is not None or self._recording,
                recording=self._recording,
                frames_written=self._frames_written,
                error=self._error,
            )

    def open_settings(self) -> None:
        """Open DirectShow camera properties dialog if supported and active."""
        with self._lock:
            if self._capture is not None and self._capture.isOpened():
                self._capture.set(cv2.CAP_PROP_SETTINGS, 1)

    def close(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        self.request_stop_recording()
        if self._thread:
            self._thread.join(timeout=timeout)

    def _open_capture(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.settings.index, capture_backend())
        if not capture.isOpened():
            raise RuntimeError("カメラを開けませんでした")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.height)
        capture.set(cv2.CAP_PROP_FPS, self.settings.fps)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.settings.show_settings and platform.system() == "Windows":
            capture.set(cv2.CAP_PROP_SETTINGS, 1)
        return capture

    def _run(self) -> None:
        writer: cv2.VideoWriter | None = None
        try:
            self._capture = self._open_capture()
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            with self._lock:
                self._connected = True
                if 1 <= fps <= 240:
                    self._actual_fps = fps

            while not self._stop_event.is_set():
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    with self._lock:
                        self._error = "映像の取得に失敗しました"
                        stop_recording = self._stop_recording
                    if stop_recording:
                        if writer is not None:
                            writer.release()
                            writer = None
                        with self._lock:
                            self._recording = False
                            self._request = None
                            self._stop_recording = False
                    time.sleep(0.05)
                    continue

                height, width = frame.shape[:2]
                with self._lock:
                    self._frame = frame
                    self._actual_width = width
                    self._actual_height = height
                    request = self._request
                    stop_recording = self._stop_recording

                if request is not None and writer is None and not stop_recording:
                    request.path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(request.path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        self._actual_fps,
                        (width, height),
                    )
                    if writer.isOpened():
                        request.armed_event.set()
                    else:
                        writer.release()
                        writer = None
                        with self._lock:
                            self._error = "録画ファイルを作成できませんでした"
                            self._request = None
                        request.armed_event.set()

                if writer is not None and request is not None and request.start_event.is_set():
                    writer.write(frame)
                    with self._lock:
                        self._recording = True
                        self._frames_written += 1

                if writer is not None and stop_recording:
                    writer.release()
                    writer = None
                    with self._lock:
                        self._recording = False
                        self._request = None
                        self._stop_recording = False

                if writer is None and stop_recording:
                    with self._lock:
                        self._request = None
                        self._recording = False
                        self._stop_recording = False
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
                request = self._request
            if request:
                request.armed_event.set()
        finally:
            if writer is not None:
                writer.release()
            if self._capture is not None:
                self._capture.release()
            with self._lock:
                self._connected = False
                self._recording = False
                self._request = None

