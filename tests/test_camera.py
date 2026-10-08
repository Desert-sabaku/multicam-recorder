from unittest.mock import MagicMock, patch

import cv2
import pytest

from multicam_recorder.camera import (
    ROTATION_OPTIONS,
    CameraSettings,
    CameraWorker,
    apply_rotation,
)


def test_camera_settings_defaults():
    settings = CameraSettings(index=1)
    assert settings.index == 1
    assert settings.width == 1280
    assert settings.height == 720
    assert settings.fps == 30.0
    assert settings.show_settings is False
    assert settings.rotation == "0°"


def test_apply_rotation():
    import numpy as np

    # Create an asymmetric 10x20 image: (height=10, width=20, channels=3)
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    frame[0, 0] = [1, 2, 3]  # top-left marker

    # 0° (no-op)
    rot0 = apply_rotation(frame, "0°")
    assert rot0.shape == (10, 20, 3)
    assert np.array_equal(rot0[0, 0], [1, 2, 3])

    # 90° clockwise
    rot90 = apply_rotation(frame, "90°")
    assert rot90.shape == (20, 10, 3)
    # top-left moves to top-right (row 0, col 9)
    assert np.array_equal(rot90[0, 9], [1, 2, 3])

    # 180°
    rot180 = apply_rotation(frame, "180°")
    assert rot180.shape == (10, 20, 3)
    # top-left moves to bottom-right (row 9, col 19)
    assert np.array_equal(rot180[9, 19], [1, 2, 3])

    # 270° (counter-clockwise 90°)
    rot270 = apply_rotation(frame, "270°")
    assert rot270.shape == (20, 10, 3)
    # top-left moves to bottom-left (row 19, col 0)
    assert np.array_equal(rot270[19, 0], [1, 2, 3])

    # 左右反転 (flip horizontal)
    flip_h = apply_rotation(frame, "左右反転")
    assert flip_h.shape == (10, 20, 3)
    # top-left moves to top-right (row 0, col 19)
    assert np.array_equal(flip_h[0, 19], [1, 2, 3])

    # 上下反転 (flip vertical)
    flip_v = apply_rotation(frame, "上下反転")
    assert flip_v.shape == (10, 20, 3)
    # top-left moves to bottom-left (row 9, col 0)
    assert np.array_equal(flip_v[9, 0], [1, 2, 3])


def test_camera_worker_rotation():
    import threading
    from pathlib import Path

    settings = CameraSettings(index=0, rotation="90°")
    worker = CameraWorker(settings)
    assert worker.rotation == "90°"
    assert worker.snapshot().rotation == "90°"

    worker.set_rotation("180°")
    assert worker.rotation == "180°"
    assert worker.snapshot().rotation == "180°"

    # When armed/recording, set_rotation should be ignored
    start_event = threading.Event()
    worker.arm_recording(Path("dummy.mp4"), start_event)
    worker.set_rotation("270°")
    assert worker.rotation == "180°"


def test_open_capture_order_and_mjpg():
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    calls = []

    def fake_set(prop_id, val):
        calls.append((prop_id, val))
        return True

    mock_cap.set.side_effect = fake_set

    settings = CameraSettings(index=0, width=1920, height=1080, fps=30.0, show_settings=True)
    worker = CameraWorker(settings)

    with patch("cv2.VideoCapture", return_value=mock_cap), patch("platform.system", return_value="Windows"):
        capture = worker._open_capture()
        assert capture is mock_cap

    mjpg_fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    # Verify that WIDTH, HEIGHT, FPS are set, and FOURCC (MJPG) is set after them
    prop_ids = [c[0] for c in calls]
    assert cv2.CAP_PROP_FRAME_WIDTH in prop_ids
    assert cv2.CAP_PROP_FRAME_HEIGHT in prop_ids
    assert cv2.CAP_PROP_FPS in prop_ids
    assert cv2.CAP_PROP_FOURCC in prop_ids

    # Find last occurrence of CAP_PROP_FOURCC
    last_fourcc_idx = max(i for i, c in enumerate(calls) if c[0] == cv2.CAP_PROP_FOURCC)
    fps_idx = max(i for i, c in enumerate(calls) if c[0] == cv2.CAP_PROP_FPS)
    assert last_fourcc_idx > fps_idx
    assert calls[last_fourcc_idx][1] == mjpg_fourcc

    # Verify CAP_PROP_SETTINGS was called
    assert (cv2.CAP_PROP_SETTINGS, 1) in calls
