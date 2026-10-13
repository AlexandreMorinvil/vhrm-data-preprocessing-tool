"""Signal slices and timing metadata of exported labelled segments.

Shared by the Labelling export and the PPG alignment step, which re-slices the
signals of existing segments after the video/signal offset changes.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from .signals import write_synced_signal_csvs

log = logging.getLogger(__name__)

SYNTHETIC_ARTIFACT_FIELDS = (
    "synthetic_ppg_path",
    "synthetic_ppg_hr_path",
    "synthetic_ppg_hrv_path",
    "synthetic_rr_path",
)
OFFSET_KEY = "video_signal_offset_sec"
OFFSET_NOTE_KEY = "video_signal_offset_note"


def export_signal_slice(df: Optional[pd.DataFrame], anchor: Optional[datetime], interval,
                        destination: Path, primary_column: str) -> list[str]:
    """Write the part of *df* inside *interval* as one CSV per sensor; return file names."""
    if df is None or df.empty or anchor is None:
        return []
    start = pd.Timestamp(anchor + timedelta(seconds=interval.start_sec))
    end = pd.Timestamp(anchor + timedelta(seconds=interval.end_sec))
    sub = df[(df["timestamp_utc"] >= start) & (df["timestamp_utc"] <= end)]
    if sub.empty:
        return []
    return [
        Path(path).name
        for path in write_synced_signal_csvs(sub, destination, primary_column)
    ]


def synthetic_artifact_sources(state) -> dict[str, Path]:
    """Project-level synthetic PPG artifacts that exist on disk."""
    return {
        field_name: Path(path)
        for field_name in SYNTHETIC_ARTIFACT_FIELDS
        if (path := getattr(state, field_name, ""))
        if Path(path).exists()
    }


def export_synthetic_artifact_slices(sources: dict[str, Path], anchor: datetime, interval,
                                     segment_directory: Path) -> dict[str, str]:
    """Slice every synthetic artifact into ``<segment>/synthetic_ppg/`` (recreated)."""
    if not sources:
        return {}
    destination = segment_directory / "synthetic_ppg"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    start = pd.Timestamp(anchor + timedelta(seconds=interval.start_sec))
    end = pd.Timestamp(anchor + timedelta(seconds=interval.end_sec))
    exported: dict[str, str] = {}
    for field_name, source in sources.items():
        frame = pd.read_csv(source)
        if "timestamp_utc" not in frame.columns:
            log.warning("Synthetic artifact has no timestamp_utc column: %s", source)
            continue
        timestamps = pd.to_datetime(
            frame["timestamp_utc"], format="mixed", utc=True, errors="coerce"
        )
        sliced = frame[(timestamps >= start) & (timestamps <= end)]
        if sliced.empty:
            continue
        relative_path = Path("synthetic_ppg") / source.name
        sliced.to_csv(segment_directory / relative_path, index=False)
        exported[field_name] = relative_path.as_posix()
    return exported


def segment_timing_fields(anchor: Optional[datetime], interval, offset_sec: float,
                          offset_note: str = "") -> dict:
    """Timing keys of a segment's ``meta.json``.

    *anchor* is the absolute time of video time 0 with the offset already
    applied, so ``signal_anchor_datetime_utc + start_sec`` is the signal time of
    the segment's first frame (this is what dataset loaders use).
    """
    return {
        "signal_anchor_datetime_utc": anchor.isoformat() if anchor else None,
        "segment_start_datetime_utc": (
            (anchor + timedelta(seconds=interval.start_sec)).isoformat() if anchor else None
        ),
        "segment_end_datetime_utc": (
            (anchor + timedelta(seconds=interval.end_sec)).isoformat() if anchor else None
        ),
        OFFSET_KEY: float(offset_sec),
        OFFSET_NOTE_KEY: offset_note,
    }
