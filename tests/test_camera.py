from unittest.mock import MagicMock, patch

import cv2
import pytest

from multicam_recorder.camera import CameraSettings, CameraWorker


def test_camera_settings_defaults():
    settings = CameraSettings(index=1)
    assert settings.index == 1
    assert settings.width == 1280
    assert settings.height == 720
    assert settings.fps == 30.0
    assert settings.show_settings is False


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
