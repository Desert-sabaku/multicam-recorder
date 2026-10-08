import json
from datetime import datetime

from multicam_recorder.session import CameraResult, make_session_directory, write_metadata


def test_session_directory_is_unique(tmp_path):
    now = datetime(2026, 9, 27, 12, 34, 56)
    first = make_session_directory(tmp_path, now)
    second = make_session_directory(tmp_path, now)

    assert first.name == "20260927_123456"
    assert second.name == "20260927_123456_2"


def test_write_metadata_includes_rotation(tmp_path):
    started_at = datetime(2026, 9, 27, 12, 0, 0)
    ended_at = datetime(2026, 9, 27, 12, 1, 0)
    results = [
        CameraResult(
            camera_index=0,
            file="camera_00.mp4",
            width=720,
            height=1280,
            fps=30.0,
            frames_written=1800,
            error=None,
            rotation="90°",
        )
    ]
    write_metadata(tmp_path, started_at, ended_at, results)
    payload = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))
    assert payload["cameras"][0]["rotation"] == "90°"
    assert payload["cameras"][0]["width"] == 720
    assert payload["cameras"][0]["height"] == 1280

