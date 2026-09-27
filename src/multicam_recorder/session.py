from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CameraResult:
    camera_index: int
    file: str
    width: int
    height: int
    fps: float
    frames_written: int
    error: str | None


def make_session_directory(root: Path, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    candidate = root / timestamp
    counter = 2
    while candidate.exists():
        candidate = root / f"{timestamp}_{counter}"
        counter += 1
    candidate.mkdir(parents=True)
    return candidate


def write_metadata(
    directory: Path,
    started_at: datetime,
    ended_at: datetime,
    results: list[CameraResult],
) -> None:
    payload = {
        "started_at": started_at.astimezone().isoformat(timespec="milliseconds"),
        "ended_at": ended_at.astimezone().isoformat(timespec="milliseconds"),
        "duration_seconds": round((ended_at - started_at).total_seconds(), 3),
        "cameras": [asdict(result) for result in results],
    }
    (directory / "session.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

