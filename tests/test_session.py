from datetime import datetime

from multicam_recorder.session import make_session_directory


def test_session_directory_is_unique(tmp_path):
    now = datetime(2026, 9, 27, 12, 34, 56)
    first = make_session_directory(tmp_path, now)
    second = make_session_directory(tmp_path, now)

    assert first.name == "20260927_123456"
    assert second.name == "20260927_123456_2"

