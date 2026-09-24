from __future__ import annotations

import csv
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
    reference_video_time_sec: float = 0.0
    video_reference_datetime: Optional[str] = None
    time_correction_offset_sec: float = 0.0
    limits_common_duration: bool = True

    def parsed_start_datetime(self) -> Optional[datetime]:
        if self.start_datetime is None:
            return None
        return datetime.fromisoformat(self.start_datetime)

    def parsed_true_start_datetime(self) -> Optional[datetime]:
        if self.true_start_datetime is None:
            return None
        return datetime.fromisoformat(self.true_start_datetime)

    def parsed_video_reference_datetime(self) -> Optional[datetime]:
        if self.video_reference_datetime is None:
            return None
        return datetime.fromisoformat(self.video_reference_datetime)

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
    folder: str = ""

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
    hr_signal_paths: list[str] = field(default_factory=list)
    ecg_signal_paths: list[str] = field(default_factory=list)
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
    synced_hr_path: str = ""
    synced_ecg_path: str = ""
    synced_hr_paths: list[str] = field(default_factory=list)
    synced_ecg_paths: list[str] = field(default_factory=list)
    legacy_synced_paths: list[str] = field(default_factory=list)
    segments_manifest_path: str = ""
    mosaic_preset: str = "balanced"
    archive_removed_files: bool = False
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
        d = dict(d)
        tracks = [VideoTrack.from_dict(t) for t in d.pop("tracks", [])]
        intervals = [LabelInterval.from_dict(i) for i in d.pop("intervals", [])]
        # Backward compat: old "signal_mode" → new "include_signal_average"
        old_mode = d.pop("signal_mode", None)
        if old_mode is not None and "include_signal_average" not in d:
            d["include_signal_average"] = (old_mode == "average")
        # Projects created before ECG support stored HR in generic signal fields.
        if "hr_signal_paths" not in d:
            d["hr_signal_paths"] = list(d.get("signal_paths", []))
        if "synced_hr_path" not in d:
            d["synced_hr_path"] = d.get("synced_signal_path", "")
        if "synced_hr_paths" not in d:
            legacy_hr = d.get("synced_hr_path") or d.get("synced_signal_path")
            d["synced_hr_paths"] = [legacy_hr] if legacy_hr else []
        if "synced_ecg_paths" not in d:
            legacy_ecg = d.get("synced_ecg_path")
            d["synced_ecg_paths"] = [legacy_ecg] if legacy_ecg else []
        if "legacy_synced_paths" not in d:
            legacy_paths = [
                d.get("synced_signal_path"),
                d.get("synced_hr_path"),
                d.get("synced_ecg_path"),
            ]
            d["legacy_synced_paths"] = list(dict.fromkeys(
                path for path in legacy_paths if path
            ))
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
    "synced_signal_path", "synced_hr_path", "synced_ecg_path",
    "segments_manifest_path", "folder",
}
_PATH_LIST_KEYS = {
    "segment_paths", "signal_paths", "hr_signal_paths", "ecg_signal_paths",
    "synced_hr_paths", "synced_ecg_paths", "legacy_synced_paths",
}


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


def load_labelled_segments_manifest(path: str, state: ProjectState) -> list[LabelInterval]:
    """Load labelled intervals and their existing folders into shared state."""
    manifest_path = Path(path)
    intervals = []
    with open(manifest_path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            label = (row.get("label") or "").strip()
            start_sec = float(row["start_sec"])
            end_sec = float(row["end_sec"])
            folder = (row.get("folder") or "").strip()
            if not label:
                raise ValueError("A segment has no label.")
            if start_sec >= end_sec:
                raise ValueError(f"Segment '{label}' must end after it starts.")
            if not folder:
                raise ValueError(f"Segment '{label}' has no folder.")
            intervals.append(LabelInterval(
                label=label,
                start_sec=start_sec,
                end_sec=end_sec,
                folder=str(manifest_path.parent / folder),
            ))
    if not intervals:
        raise ValueError("The manifest contains no segments.")
    intervals.sort(key=lambda interval: interval.start_sec)
    state.intervals = intervals
    state.segments_manifest_path = str(manifest_path)
    for interval in intervals:
        if interval.label not in state.labels_library:
            state.labels_library.append(interval.label)
    return intervals


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
            "reference_video_time_sec": t.reference_video_time_sec,
            "video_reference_datetime_utc": t.video_reference_datetime,
            "corrected_start_datetime_utc": corrected.isoformat() if corrected else None,
            "limits_common_duration": t.limits_common_duration,
        })

    common_dur = video_timeline_duration_sec(state.tracks)

    hr_synced_names = [Path(path).name for path in state.synced_hr_paths]
    ecg_synced_names = [Path(path).name for path in state.synced_ecg_paths]
    anchor_dt, warnings = compute_signal_anchor(
        state.tracks, state.time_coherence_tolerance_sec
    )
    anchor_iso = state.last_signal_anchor_datetime or (anchor_dt.isoformat() if anchor_dt else None)
    coherence_warnings = state.last_time_coherence_warnings or warnings

    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cameras": cameras,
        "common_duration_sec": common_dur,
        "hr_signal_paths": [Path(p).name for p in state.hr_signal_paths],
        "ecg_signal_paths": [Path(p).name for p in state.ecg_signal_paths],
        "synced_hr_paths": hr_synced_names,
        "synced_ecg_paths": ecg_synced_names,
        "hr_time_range_sec": [0.0, common_dur] if hr_synced_names else None,
        "ecg_time_range_sec": [0.0, common_dur] if ecg_synced_names else None,
        "signal_paths": [Path(p).name for p in state.hr_signal_paths],
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
    state.signal_paths.clear()
    state.hr_signal_paths.clear()
    state.ecg_signal_paths.clear()
    state.synced_signal_path = ""
    state.synced_hr_path = ""
    state.synced_ecg_path = ""
    state.synced_hr_paths.clear()
    state.synced_ecg_paths.clear()
    state.legacy_synced_paths.clear()
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
            reference_video_time_sec=cam.get("reference_video_time_sec", 0.0),
            video_reference_datetime=cam.get("video_reference_datetime_utc") or None,
            time_correction_offset_sec=cam.get("time_correction_offset_sec", 0.0),
            limits_common_duration=cam.get("limits_common_duration", True),
        )
        state.tracks.append(track)

    state.num_cameras = len(state.tracks)

    # Load signal paths relative to sidecar directory
    hr_names = raw.get("hr_signal_paths", raw.get("signal_paths", []))
    for name in hr_names:
        if name:
            abs_sig = str(base_dir / name)
            if abs_sig not in state.hr_signal_paths:
                state.hr_signal_paths.append(abs_sig)
    ecg_names = raw.get("ecg_signal_paths", [])
    for name in ecg_names:
        if name:
            abs_sig = str(base_dir / name)
            if abs_sig not in state.ecg_signal_paths:
                state.ecg_signal_paths.append(abs_sig)
    state.signal_paths = list(state.hr_signal_paths)
    legacy_synced_names = list(dict.fromkeys(
        name for name in (
            raw.get("synced_signal_path"),
            raw.get("synced_hr_path"),
            raw.get("synced_ecg_path"),
        ) if name
    ))
    state.legacy_synced_paths = [str(base_dir / name) for name in legacy_synced_names]
    hr_synced_names = raw.get("synced_hr_paths")
    if hr_synced_names is None:
        legacy_hr = raw.get("synced_hr_path", raw.get("synced_signal_path"))
        hr_synced_names = [legacy_hr] if legacy_hr else []
    state.synced_hr_paths = [str(base_dir / name) for name in hr_synced_names if name]
    ecg_synced_names = raw.get("synced_ecg_paths")
    if ecg_synced_names is None:
        legacy_ecg = raw.get("synced_ecg_path")
        ecg_synced_names = [legacy_ecg] if legacy_ecg else []
    state.synced_ecg_paths = [str(base_dir / name) for name in ecg_synced_names if name]

    state.output_directory = str(base_dir)
    state.mode1_complete = True
    state.time_coherence_tolerance_sec = raw.get(
        "time_coherence_tolerance_sec", state.time_coherence_tolerance_sec
    )
    state.last_signal_anchor_datetime = raw.get("signal_anchor_datetime_utc")
    state.last_time_coherence_warnings = raw.get("time_coherence_warnings", [])

    state.mode2_complete = bool(
        any(Path(path).exists() for path in state.synced_hr_paths)
        or any(Path(path).exists() for path in state.synced_ecg_paths)
    )

    log.info("Loaded sidecar with %d camera(s) from %s", len(state.tracks), path)


def video_timeline_duration_sec(tracks: list[VideoTrack]) -> float:
    """Return the duration used for shared signal and timeline views.

    Cameras marked as not limiting size are ignored when at least one camera is
    still allowed to limit the shared duration. If every camera is non-limiting,
    use the longest available duration so no camera shortens the session.
    """
    positive_durations = [t.duration_sec for t in tracks if t.duration_sec > 0]
    if not positive_durations:
        return 0.0

    limiting_durations = [
        t.duration_sec for t in tracks
        if t.duration_sec > 0 and t.limits_common_duration
    ]
    if limiting_durations:
        return min(limiting_durations)
    return max(positive_durations)


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
