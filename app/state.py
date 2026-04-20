from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_DJI_RE = re.compile(
    r"DJI_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})_(\d+)"
)


def parse_dji_datetime(filename: str) -> Optional[datetime]:
    m = _DJI_RE.search(filename)
    if m is None:
        return None
    parts = [int(g) for g in m.groups()[:6]]
    return datetime(*parts, tzinfo=timezone.utc)


def dji_datetime_str(dt: Optional[datetime]) -> str:
    if dt is None:
        return "unknown"
    return dt.strftime("%Y%m%d_%H%M%S")


@dataclass
class VideoTrack:
    camera_index: int = 0
    camera_label: str = ""
    segment_paths: list[str] = field(default_factory=list)
    concatenated_path: str = ""
    trimstart_path: str = ""
    final_output_path: str = ""
    fps: float = 0.0
    frame_count: int = 0
    width: int = 0
    height: int = 0
    codec: str = ""
    duration_sec: float = 0.0
    start_datetime: Optional[str] = None
    sync_offset_sec: float = 0.0

    def parsed_start_datetime(self) -> Optional[datetime]:
        if self.start_datetime is None:
            return None
        return datetime.fromisoformat(self.start_datetime)

    def set_start_datetime(self, dt: Optional[datetime]) -> None:
        self.start_datetime = dt.isoformat() if dt else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VideoTrack:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class LabelInterval:
    label: str = ""
    start_sec: float = 0.0
    end_sec: float = 0.0
    color: str = "#4488cc"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LabelInterval:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ProjectState:
    project_path: str = ""
    output_directory: str = ""
    num_cameras: int = 2
    tracks: list[VideoTrack] = field(default_factory=list)
    signal_paths: list[str] = field(default_factory=list)
    signal_mode: str = "separate"
    labels_library: list[str] = field(default_factory=lambda: [
        "Baseline", "Resting", "Resistance exercise", "Cardio exercise"
    ])
    intervals: list[LabelInterval] = field(default_factory=list)
    active_mode: int = 1
    mode1_complete: bool = False
    mode2_complete: bool = False
    ffmpeg_path: str = ""
    keep_temp_files: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tracks"] = [t.to_dict() for t in self.tracks]
        d["intervals"] = [i.to_dict() for i in self.intervals]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProjectState:
        tracks = [VideoTrack.from_dict(t) for t in d.pop("tracks", [])]
        intervals = [LabelInterval.from_dict(i) for i in d.pop("intervals", [])]
        valid = {k for k in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in valid}
        state = cls(**filtered)
        state.tracks = tracks
        state.intervals = intervals
        return state

    def save(self, path) -> None:
        p = Path(path)
        self.project_path = str(p)
        d = self.to_dict()
        base = p.parent
        d = _relativise_paths(d, base)
        p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "ProjectState":
        p = Path(path)
        raw = json.loads(p.read_text(encoding="utf-8"))
        base = p.parent
        raw = _resolve_paths(raw, base)
        state = cls.from_dict(raw)
        state.project_path = str(p)
        return state


_PATH_KEYS = {
    "project_path", "output_directory", "concatenated_path",
    "trimstart_path", "final_output_path", "ffmpeg_path",
}
_PATH_LIST_KEYS = {"segment_paths", "signal_paths"}


def _try_relative(p: str, base: Path) -> str:
    if not p:
        return p
    try:
        return str(Path(p).relative_to(base))
    except ValueError:
        return p


def _try_resolve(p: str, base: Path) -> str:
    if not p:
        return p
    candidate = base / p
    if candidate.exists():
        return str(candidate.resolve())
    return p


def _relativise_paths(d, base):
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            if k in _PATH_KEYS and isinstance(v, str):
                out[k] = _try_relative(v, base)
            elif k in _PATH_LIST_KEYS and isinstance(v, list):
                out[k] = [_try_relative(s, base) for s in v]
            else:
                out[k] = _relativise_paths(v, base)
        return out
    if isinstance(d, list):
        return [_relativise_paths(item, base) for item in d]
    return d


def _resolve_paths(d, base):
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            if k in _PATH_KEYS and isinstance(v, str):
                out[k] = _try_resolve(v, base)
            elif k in _PATH_LIST_KEYS and isinstance(v, list):
                out[k] = [_try_resolve(s, base) for s in v]
            else:
                out[k] = _resolve_paths(v, base)
        return out
    if isinstance(d, list):
        return [_resolve_paths(item, base) for item in d]
    return d
