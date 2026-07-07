from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

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
    time_correction_mode: str = "none"
    true_start_datetime: Optional[str] = None
    time_correction_offset_sec: float = 0.0

    def parsed_start_datetime(self) -> Optional[datetime]:
        if self.start_datetime is None:
            return None
        return datetime.fromisoformat(self.start_datetime)

    def parsed_true_start_datetime(self) -> Optional[datetime]:
        if self.true_start_datetime is None:
            return None
        return datetime.fromisoformat(self.true_start_datetime)

    def corrected_start_datetime(self) -> Optional[datetime]:
        base = self.parsed_start_datetime()
        if base is None:
            return None
        return base + timedelta(seconds=self.time_correction_offset_sec)

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
    include_signal_average: bool = False
    labels_library: list[str] = field(default_factory=lambda: [
        "Baseline", "Resting", "Resistance exercise", "Cardio exercise"
    ])
    intervals: list[LabelInterval] = field(default_factory=list)
    active_mode: int = 1
    mode1_complete: bool = False
    mode2_complete: bool = False
    ffmpeg_path: str = ""
    keep_temp_files: bool = False
    synced_signal_path: str = ""
    mosaic_preset: str = "balanced"
    time_coherence_tolerance_sec: float = 1.0
    last_signal_anchor_datetime: Optional[str] = None
    last_time_coherence_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tracks"] = [t.to_dict() for t in self.tracks]
        d["intervals"] = [i.to_dict() for i in self.intervals]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProjectState:
        tracks = [VideoTrack.from_dict(t) for t in d.pop("tracks", [])]
        intervals = [LabelInterval.from_dict(i) for i in d.pop("intervals", [])]
        # Backward compat: old "signal_mode" → new "include_signal_average"
        old_mode = d.pop("signal_mode", None)
        if old_mode is not None and "include_signal_average" not in d:
            d["include_signal_average"] = (old_mode == "average")
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
    "synced_signal_path",
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


# ---------------------------------------------------------------------------
# Metadata sidecar (.vrt-meta.json)
# ---------------------------------------------------------------------------

def generate_sidecar(state: ProjectState) -> str:
    """Write a ``_meta.json`` sidecar alongside the final videos.

    Returns the path of the generated file.
    """
    out_dir = state.output_directory
    if not out_dir:
        raise ValueError("No output directory set in project state.")

    project_name = Path(state.project_path).stem if state.project_path else "project"
    sidecar_path = str(Path(out_dir) / f"{project_name}_meta.json")

    cameras = []
    for t in state.tracks:
        corrected = t.corrected_start_datetime()
        cameras.append({
            "label": t.camera_label,
            "final_video_path": Path(t.final_output_path).name if t.final_output_path else "",
            "fps": t.fps,
            "frame_count": t.frame_count,
            "duration_sec": t.duration_sec,
            "dimensions": [t.width, t.height],
            "codec": t.codec,
            "start_datetime_utc": t.start_datetime or "",
            "sync_offset_sec": t.sync_offset_sec,
            "time_correction_mode": t.time_correction_mode,
            "time_correction_offset_sec": t.time_correction_offset_sec,
            "true_start_datetime_utc": t.true_start_datetime,
            "corrected_start_datetime_utc": corrected.isoformat() if corrected else None,
        })

    common_dur = min((t.duration_sec for t in state.tracks), default=0.0)

    synced_name = Path(state.synced_signal_path).name if state.synced_signal_path else None
    sig_range = [0.0, common_dur] if state.synced_signal_path else None
    anchor_dt, warnings = compute_signal_anchor(
        state.tracks, state.time_coherence_tolerance_sec
    )
    anchor_iso = state.last_signal_anchor_datetime or (anchor_dt.isoformat() if anchor_dt else None)
    coherence_warnings = state.last_time_coherence_warnings or warnings

    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cameras": cameras,
        "common_duration_sec": common_dur,
        "signal_paths": [Path(p).name for p in state.signal_paths],
        "synced_signal_path": synced_name,
        "signal_time_range_sec": sig_range,
        "signal_anchor_datetime_utc": anchor_iso,
        "time_coherence_tolerance_sec": state.time_coherence_tolerance_sec,
        "time_coherence_warnings": coherence_warnings,
    }

    Path(sidecar_path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Wrote sidecar: %s", sidecar_path)
    return sidecar_path


def load_sidecar(path: str, state: ProjectState) -> None:
    """Populate *state*.tracks from a ``_meta.json`` sidecar file."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    base_dir = p.parent

    state.tracks.clear()
    for i, cam in enumerate(raw.get("cameras", [])):
        video_file = cam.get("final_video_path", "")
        abs_video = str(base_dir / video_file) if video_file else ""
        dims = cam.get("dimensions", [0, 0])
        track = VideoTrack(
            camera_index=i,
            camera_label=cam.get("label", f"Camera {i+1}"),
            final_output_path=abs_video,
            fps=cam.get("fps", 0.0),
            frame_count=cam.get("frame_count", 0),
            width=dims[0] if len(dims) > 0 else 0,
            height=dims[1] if len(dims) > 1 else 0,
            codec=cam.get("codec", ""),
            duration_sec=cam.get("duration_sec", 0.0),
            start_datetime=cam.get("start_datetime_utc") or None,
            sync_offset_sec=cam.get("sync_offset_sec", 0.0),
            time_correction_mode=cam.get("time_correction_mode", "none"),
            true_start_datetime=cam.get("true_start_datetime_utc") or None,
            time_correction_offset_sec=cam.get("time_correction_offset_sec", 0.0),
        )
        state.tracks.append(track)

    state.num_cameras = len(state.tracks)

    # Load signal paths relative to sidecar directory
    sig_names = raw.get("signal_paths", [])
    for name in sig_names:
        if name:
            abs_sig = str(base_dir / name)
            if abs_sig not in state.signal_paths:
                state.signal_paths.append(abs_sig)

    state.output_directory = str(base_dir)
    state.mode1_complete = True
    state.time_coherence_tolerance_sec = raw.get(
        "time_coherence_tolerance_sec", state.time_coherence_tolerance_sec
    )
    state.last_signal_anchor_datetime = raw.get("signal_anchor_datetime_utc")
    state.last_time_coherence_warnings = raw.get("time_coherence_warnings", [])

    # Load synced signal path if present
    synced_name = raw.get("synced_signal_path")
    if synced_name:
        abs_synced = str(base_dir / synced_name)
        if Path(abs_synced).exists():
            state.synced_signal_path = abs_synced
            state.mode2_complete = True

    log.info("Loaded sidecar with %d camera(s) from %s", len(state.tracks), path)


def populate_tracks_from_videos(
    video_paths: list[str],
    state: ProjectState,
    probe_func=None,
) -> None:
    """Populate *state*.tracks by probing raw video files directly.

    *probe_func* defaults to :func:`ffmpeg_utils.probe_video`.
    """
    if probe_func is None:
        from .ffmpeg_utils import probe_video, find_ffprobe
        ffprobe = find_ffprobe()
        probe_func = lambda p: probe_video(p, ffprobe)  # noqa: E731

    state.tracks.clear()
    for i, vp in enumerate(video_paths):
        info = probe_func(vp)
        track = VideoTrack(
            camera_index=i,
            camera_label=Path(vp).stem,
            final_output_path=vp,
            fps=info.get("fps", 0.0),
            frame_count=info.get("frame_count", 0),
            width=info.get("width", 0),
            height=info.get("height", 0),
            codec=info.get("codec", ""),
            duration_sec=info.get("duration", 0.0),
        )
        dt = parse_dji_datetime(Path(vp).stem)
        track.set_start_datetime(dt)
        state.tracks.append(track)
    state.num_cameras = len(state.tracks)
    state.mode1_complete = True
    log.info("Populated %d track(s) from video files", len(state.tracks))


# ---------------------------------------------------------------------------
# Signal wall-clock anchor helpers
# ---------------------------------------------------------------------------

def track_effective_start_datetime(track: VideoTrack) -> Optional[datetime]:
    """Return the start datetime to use for signal alignment for one track."""
    if track.time_correction_mode != "none":
        return track.corrected_start_datetime()
    return track.parsed_start_datetime()


def compute_signal_anchor(
    tracks: list[VideoTrack],
    tolerance_sec: float = 1.0,
) -> tuple[Optional[datetime], list[str]]:
    """Return the common wall-clock signal anchor and coherence warnings.

    The returned anchor is the absolute datetime corresponding to video time 0.
    Corrections never alter video-to-video alignment; they only change the signal
    clipping/alignment window.
    """
    if not tracks:
        return None, []

    corrected: list[tuple[int, VideoTrack, datetime]] = []
    for idx, track in enumerate(tracks):
        if track.time_correction_mode == "none":
            continue
        effective = track_effective_start_datetime(track)
        if effective is not None:
            corrected.append((idx, track, effective))

    if not corrected:
        return tracks[0].parsed_start_datetime(), []

    # Current final outputs and direct-load previews both use video time 0 as
    # the common timeline start. Do not reinterpret sync_offset_sec here.
    implied = corrected
    timestamps = [dt.timestamp() for _, _, dt in implied]
    spread = max(timestamps) - min(timestamps) if timestamps else 0.0

    anchor = None
    for idx, _track, dt in implied:
        if idx == 0:
            anchor = dt
            break
    if anchor is None:
        anchor = implied[0][2]

    warnings: list[str] = []
    if spread > tolerance_sec:
        earliest_ts = min(timestamps)
        latest_ts = max(timestamps)
        names = []
        for idx, track, dt in implied:
            ts = dt.timestamp()
            if abs(ts - earliest_ts) <= 1e-6 or abs(ts - latest_ts) <= 1e-6:
                names.append(track.camera_label or f"Camera {idx + 1}")
        warnings.append(
            "Camera time corrections disagree by "
            f"{spread:.3f}s, exceeding the {tolerance_sec:.3f}s tolerance. "
            "Check: " + ", ".join(names) + "."
        )

    return anchor, warnings


def format_time_coherence_warnings(warnings: list[str]) -> str:
    return "\n".join(f"- {w}" for w in warnings)
