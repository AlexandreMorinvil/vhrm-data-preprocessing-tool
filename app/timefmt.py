"""Shared time formatting and parsing helpers for video-time displays."""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from typing import Optional


def format_hms(sec: float) -> str:
    """Format seconds as ``H:MM:SS`` (or ``M:SS`` below one hour)."""
    sec = max(0.0, float(sec))
    total = int(sec)
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def format_hms_ms(sec: float) -> str:
    """Format seconds as ``HH:MM:SS.mmm``."""
    ms_total = int(round(max(0.0, float(sec)) * 1000))
    hours, ms_total = divmod(ms_total, 3_600_000)
    minutes, ms_total = divmod(ms_total, 60_000)
    seconds, millis = divmod(ms_total, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def format_axis_time(sec: float, step: float) -> str:
    """Format a tick label with precision adapted to the tick *step*."""
    negative = sec < 0
    sec = abs(sec)
    if step >= 1.0 - 1e-9:
        text = format_hms(sec)
    else:
        decimals = 1 if step >= 0.1 - 1e-9 else (2 if step >= 0.01 - 1e-9 else 3)
        whole = int(sec)
        frac = sec - whole
        rounded = round(frac, decimals)
        if rounded >= 1.0:
            whole += 1
            rounded = 0.0
        text = f"{format_hms(whole)}.{int(round(rounded * 10 ** decimals)):0{decimals}d}"
    return f"-{text}" if negative else text


def format_clock(dt: Optional[datetime], with_ms: bool = True) -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    text = dt.astimezone(timezone.utc).strftime("%H:%M:%S.%f")
    return (text[:-3] if with_ms else text[:-7]) + " UTC"


def clock_at(anchor: Optional[datetime], sec: float) -> Optional[datetime]:
    if anchor is None:
        return None
    return anchor + timedelta(seconds=sec)


_FRAME_RE = re.compile(r"^\s*[fF#]\s*(\d+)\s*$")


def parse_time_text(text: str, fps: float = 0.0) -> Optional[float]:
    """Parse ``H:MM:SS.mmm``, ``MM:SS``, plain seconds or ``f1234`` (frame).

    Returns seconds, or ``None`` when the text cannot be parsed.
    """
    text = (text or "").strip().replace(",", ".")
    if not text:
        return None
    frame_match = _FRAME_RE.match(text)
    if frame_match:
        if fps <= 0:
            return None
        return int(frame_match.group(1)) / fps
    parts = text.split(":")
    if len(parts) > 3:
        return None
    try:
        values = [float(part) for part in parts]
    except ValueError:
        return None
    if any(v < 0 or math.isnan(v) for v in values):
        return None
    seconds = 0.0
    for value in values:
        seconds = seconds * 60 + value
    return seconds


def nice_time_step(min_step_sec: float) -> float:
    """Return a 'nice' tick step (in seconds) that is at least *min_step_sec*."""
    candidates = (
        0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
        1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200,
    )
    for step in candidates:
        if step >= min_step_sec:
            return float(step)
    return float(math.ceil(min_step_sec / 3600) * 3600)
