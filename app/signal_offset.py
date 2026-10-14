"""Apply a video/signal offset to a project and its exported labelled segments.

The offset corrects the signal anchor (the absolute time of video time 0):
``anchor = camera anchor - offset``. A positive offset therefore places the
signals later on the video timeline.

Applying an offset:

* updates the project state (offset and last signal anchor);
* re-slices the HR, ECG and synthetic PPG files of every exported segment with
  the corrected anchor and rewrites the timing keys of its ``meta.json``
  (``signal_anchor_datetime_utc`` + ``start_sec`` is what dataset loaders use);
* updates the project metadata sidecar in place;
* appends the change to ``<output>/ppg_alignment/alignment.json``.

Videos are never modified.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from .segment_signals import (
    OFFSET_KEY,
    OFFSET_NOTE_KEY,
    SYNTHETIC_ARTIFACT_FIELDS,
    export_signal_slice,
    export_synthetic_artifact_slices,
    segment_timing_fields,
    synthetic_artifact_sources,
)
from .signals import read_synced_signal_csvs
from .state import ProjectState, effective_signal_anchor

log = logging.getLogger(__name__)

ALIGNMENT_DIRECTORY = "ppg_alignment"
ALIGNMENT_RECORD = "alignment.json"


def alignment_directory(state: ProjectState) -> Optional[Path]:
    if state.output_directory:
        return Path(state.output_directory) / ALIGNMENT_DIRECTORY
    if state.segments_manifest_path:
        return Path(state.segments_manifest_path).parent.parent / ALIGNMENT_DIRECTORY
    return None


def read_segment_meta(folder: str | Path) -> dict:
    path = Path(folder) / "meta.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _utc(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def segment_reference_start(meta: dict, interval, state: ProjectState) -> Optional[datetime]:
    """Absolute time of a segment's first frame *without* any video/signal offset.

    Signals plotted at ``timestamp - reference start + offset`` show what the
    segment would contain with that offset applied.
    """
    start = _utc(meta.get("segment_start_datetime_utc"))
    if start is not None:
        return start + timedelta(seconds=float(meta.get(OFFSET_KEY) or 0.0))
    anchor = _utc(meta.get("signal_anchor_datetime_utc"))
    if anchor is not None:
        return anchor + timedelta(seconds=interval.start_sec + float(meta.get(OFFSET_KEY) or 0.0))
    anchor = effective_signal_anchor(state)
    if anchor is None:
        return None
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    return anchor + timedelta(seconds=interval.start_sec + state.video_signal_offset_sec)


def segment_applied_offset(meta: dict, state: ProjectState) -> float:
    """Offset the segment's exported files were cut with."""
    if OFFSET_KEY in meta:
        return float(meta.get(OFFSET_KEY) or 0.0)
    if meta.get("segment_start_datetime_utc") or meta.get("signal_anchor_datetime_utc"):
        return 0.0  # exported before offsets existed
    return state.video_signal_offset_sec


@dataclass
class ApplyReport:
    offset_sec: float
    segments_updated: int = 0
    resliced: list[str] = field(default_factory=list)
    sidecar_updated: bool = False
    record_path: str = ""
    warnings: list[str] = field(default_factory=list)


def _load_synced(paths: list[str]) -> Optional[pd.DataFrame]:
    existing = [path for path in paths if path and Path(path).exists()]
    if not existing:
        return None
    try:
        return read_synced_signal_csvs(existing)
    except Exception as exc:  # keep going with the other signals
        log.warning("Could not read synchronized signals %s: %s", existing, exc)
        return None


def apply_signal_offset(
    state: ProjectState,
    offset_sec: float,
    note: str = "",
    estimate: Optional[dict] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> ApplyReport:
    """Make *offset_sec* the project's video/signal offset everywhere (see module doc).

    Every slice is cut again from the project-level signals, so the result does not
    depend on earlier applications: applying 0 s restores the camera timing exactly.
    """
    offset_sec = round(float(offset_sec), 4)
    report = ApplyReport(offset_sec=offset_sec)
    previous_offset = state.video_signal_offset_sec

    hr = _load_synced(state.synced_hr_paths)
    ecg = _load_synced(state.synced_ecg_paths)
    synthetic_sources = synthetic_artifact_sources(state)
    report.resliced = [
        name for name, available in (("HR", hr is not None), ("ECG", ecg is not None),
                                     ("synthetic PPG", bool(synthetic_sources)))
        if available
    ]

    segments = [interval for interval in state.intervals if interval.folder]
    for number, interval in enumerate(segments, start=1):
        folder = Path(interval.folder)
        meta_path = folder / "meta.json"
        if not meta_path.exists():
            report.warnings.append(f"No meta.json in {folder}")
            continue
        meta = read_segment_meta(folder)
        reference_start = segment_reference_start(meta, interval, state)
        if reference_start is None:
            report.warnings.append(f"No time reference for {folder.name}")
            continue
        anchor = reference_start - timedelta(seconds=interval.start_sec + offset_sec)
        if hr is not None:
            meta["hr_files"] = export_signal_slice(hr, anchor, interval, folder, "heart_rate_bpm")
        if ecg is not None:
            meta["ecg_files"] = export_signal_slice(ecg, anchor, interval, folder, "ecg_waveform")
        if synthetic_sources:
            artifacts = export_synthetic_artifact_slices(synthetic_sources, anchor, interval, folder)
            for field_name in SYNTHETIC_ARTIFACT_FIELDS:
                meta.pop(field_name, None)
            meta.update(artifacts)
            ppg_path = artifacts.get("synthetic_ppg_path")
            meta["ppg_files"] = [ppg_path] if ppg_path else []
        meta.update(segment_timing_fields(anchor, interval, offset_sec, note))
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        report.segments_updated += 1
        if progress is not None:
            progress(number, len(segments))

    delta = offset_sec - previous_offset
    if state.last_signal_anchor_datetime:
        anchor = _utc(state.last_signal_anchor_datetime)
        if anchor is not None:
            state.last_signal_anchor_datetime = (anchor - timedelta(seconds=delta)).isoformat()
    state.video_signal_offset_sec = offset_sec
    state.video_signal_offset_note = note

    try:
        report.sidecar_updated = _update_sidecar(state.metadata_path, offset_sec, note)
    except (OSError, ValueError) as exc:
        report.warnings.append(f"Metadata sidecar not updated: {exc}")
    record_directory = alignment_directory(state)
    if record_directory is not None:
        report.record_path = str(write_alignment_record(record_directory, offset_sec, note, estimate))
    return report


def _update_sidecar(path: str, offset_sec: float, note: str) -> bool:
    if not path or not Path(path).exists():
        return False
    sidecar = Path(path)
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    delta = offset_sec - float(data.get(OFFSET_KEY) or 0.0)
    anchor = _utc(data.get("signal_anchor_datetime_utc"))
    if anchor is not None:
        data["signal_anchor_datetime_utc"] = (anchor - timedelta(seconds=delta)).isoformat()
    data[OFFSET_KEY] = offset_sec
    data[OFFSET_NOTE_KEY] = note
    sidecar.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def write_alignment_record(directory: Path, offset_sec: float, note: str,
                           estimate: Optional[dict] = None) -> Path:
    """Current offset plus the history of applied offsets (importable by other projects)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ALIGNMENT_RECORD
    try:
        record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        record = {}
    entry = {
        OFFSET_KEY: offset_sec,
        OFFSET_NOTE_KEY: note,
        "applied_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estimate": estimate or {},
    }
    history = list(record.get("history", []))
    history.append(entry)
    record = {OFFSET_KEY: offset_sec, OFFSET_NOTE_KEY: note, "history": history}
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_offset_file(path: str | Path) -> tuple[float, str]:
    """Read an offset from an alignment record, a metadata sidecar or a segment
    ``meta.json``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or OFFSET_KEY not in data:
        raise ValueError(
            f"{Path(path).name} has no '{OFFSET_KEY}'. Choose an alignment.json, a metadata "
            "sidecar or a segment meta.json where an offset was applied."
        )
    return float(data[OFFSET_KEY] or 0.0), str(data.get(OFFSET_NOTE_KEY) or "")
