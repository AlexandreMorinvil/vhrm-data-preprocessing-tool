"""Static exports of visualizations: figures, label overviews, CSV, snapshots.

Rendering uses matplotlib with a clean light "publication" style so exported
figures are consistent regardless of the on-screen theme. Series colours follow
the same fixed categorical order as the interactive plots.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

from .timefmt import format_axis_time, format_hms_ms, nice_time_step

log = logging.getLogger(__name__)

SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
TEXT_PRIMARY = "#1a1a1a"
TEXT_SECONDARY = "#52514e"
GRID = "#d9d8d4"
CURSOR = "#d00000"
FIGURE_FORMATS = ("png", "svg", "pdf", "tiff")
_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_filename(value: str, fallback: str = "figure") -> str:
    name = _SAFE_RE.sub("_", value.strip()).strip("._")
    return name or fallback


@dataclass
class FigureOptions:
    x_range: tuple[float, float]
    panel_indices: Optional[list[int]] = None
    include_label_track: bool = True
    include_shading: bool = True
    include_cursor: bool = False
    include_legend: bool = True
    title: str = ""
    width_in: float = 10.0
    height_in: float = 5.0
    dpi: int = 300
    font_size: float = 9.0
    time_axis_mode: str = "video"
    labels_only: bool = False


@dataclass
class SnapshotFrame:
    label: str
    image_bgr: np.ndarray


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _time_formatter(mode: str, time_zero, span: float):
    step = nice_time_step(max(span, 1e-3) / 8.0)
    if mode == "seconds":
        return None, None
    if mode == "clock" and time_zero is not None:
        def fmt(value, _pos):
            stamp = time_zero + pd.Timedelta(seconds=float(value))
            text = stamp.strftime("%H:%M:%S")
            if step < 1.0:
                text += f".{int(stamp.microsecond / 1000):03d}"
            return text
        return MultipleLocator(step), FuncFormatter(fmt)
    return MultipleLocator(step), FuncFormatter(lambda value, _pos: format_axis_time(value, step))


def _style_axes(ax, font_size: float) -> None:
    ax.tick_params(labelsize=font_size * 0.9, colors=TEXT_SECONDARY, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9a9994")
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def _visible_intervals(intervals, lo: float, hi: float):
    return [iv for iv in intervals if iv[1] > lo and iv[0] < hi]


def _draw_label_track(ax, intervals, lo: float, hi: float, font_size: float) -> None:
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#9a9994")
    renderer_width_in = ax.figure.get_figwidth() * 0.85 if ax.figure else 8.0
    span = max(hi - lo, 1e-9)
    for start, end, color, label in _visible_intervals(intervals, lo, hi):
        s, e = max(start, lo), min(end, hi)
        ax.broken_barh([(s, e - s)], (0.08, 0.84), facecolors=color, edgecolor="white", linewidth=1.0)
        width_in = (e - s) / span * renderer_width_in
        if width_in > 0.12 * max(1, len(label)) * font_size / 9.0 * 0.55:
            ax.text((s + e) / 2, 0.5, label, ha="center", va="center", fontsize=font_size * 0.85,
                    color="white", clip_on=True)
    ax.set_xlim(lo, hi)


def _label_legend_handles(intervals, lo: float, hi: float):
    seen: dict[str, str] = {}
    for _s, _e, color, label in _visible_intervals(intervals, lo, hi):
        seen.setdefault(label, color)
    return [Patch(facecolor=color, edgecolor="none", label=label) for label, color in seen.items()]


# ---------------------------------------------------------------------------
# signal figure
# ---------------------------------------------------------------------------

def render_signal_figure(snapshot, options: FigureOptions) -> Figure:
    """Render a publication-style figure from a ``PlotSnapshot``."""
    lo, hi = options.x_range
    if hi <= lo:
        raise ValueError("The time range is empty.")
    panels = [] if options.labels_only else [
        panel for index, panel in enumerate(snapshot.panels)
        if options.panel_indices is None or index in options.panel_indices
    ]
    label_track = (options.include_label_track or options.labels_only) and bool(snapshot.intervals)
    if not panels and not label_track:
        raise ValueError("Nothing to draw: select at least one panel or the label track.")

    fig = Figure(figsize=(options.width_in, options.height_in), dpi=options.dpi,
                 facecolor="white", layout="constrained")
    fig.get_layout_engine().set(h_pad=0.02, hspace=0.02)
    FigureCanvasAgg(fig)
    ratios = ([0.55] if label_track else []) + [3.0] * len(panels)
    if options.labels_only:
        ratios = [1.0]
    axes = fig.subplots(len(ratios), 1, sharex=True, squeeze=False,
                        gridspec_kw={"height_ratios": ratios})[:, 0]
    locator, formatter = _time_formatter(options.time_axis_mode, snapshot.time_zero, hi - lo)
    ax_iter = iter(axes)
    if label_track:
        track_ax = next(ax_iter)
        _draw_label_track(track_ax, snapshot.intervals, lo, hi, options.font_size)
        track_ax.set_ylabel("Labels", fontsize=options.font_size, color=TEXT_SECONDARY, rotation=0,
                            ha="right", va="center")
    for panel in panels:
        ax = next(ax_iter)
        _style_axes(ax, options.font_size)
        drawn = 0
        for series in panel.series:
            if not series.visible or series.x.size == 0:
                continue
            lo_i = int(np.searchsorted(series.x, lo, side="left"))
            hi_i = int(np.searchsorted(series.x, hi, side="right"))
            lo_i, hi_i = max(0, lo_i - 1), min(series.x.size, hi_i + 1)
            ax.plot(series.x[lo_i:hi_i], series.y[lo_i:hi_i],
                    color=SERIES_COLORS[series.color_index % len(SERIES_COLORS)],
                    linewidth=1.6 if series.dashed else 1.0,
                    linestyle="--" if series.dashed else "-", label=series.name)
            drawn += 1
        if options.include_shading:
            for start, end, color, _label in _visible_intervals(snapshot.intervals, lo, hi):
                ax.axvspan(max(start, lo), min(end, hi), color=color, alpha=0.15, linewidth=0, zorder=0)
        if options.include_cursor and snapshot.cursor_sec is not None and lo <= snapshot.cursor_sec <= hi:
            ax.axvline(snapshot.cursor_sec, color=CURSOR, linewidth=1.2)
        if panel.y_range is not None:
            ax.set_ylim(*panel.y_range)
        ax.set_ylabel(panel.y_label or "Value", fontsize=options.font_size, color=TEXT_PRIMARY)
        if panel.title and len(panels) > 1:
            ax.set_title(panel.title, fontsize=options.font_size, color=TEXT_PRIMARY, loc="left", pad=3)
        if options.include_legend and drawn > 1:
            ax.legend(fontsize=options.font_size * 0.85, loc="upper right", frameon=False)
    bottom = axes[-1]
    bottom.set_xlim(lo, hi)
    if locator is not None:
        bottom.xaxis.set_major_locator(locator)
        bottom.xaxis.set_major_formatter(formatter)
    xlabel = {"clock": "Clock time (UTC)", "seconds": "Time (s from video start)"}.get(
        options.time_axis_mode, "Time from video start (h:mm:ss)")
    bottom.set_xlabel(xlabel, fontsize=options.font_size, color=TEXT_PRIMARY)
    if label_track and options.include_legend:
        handles = _label_legend_handles(snapshot.intervals, lo, hi)
        if handles:
            fig.legend(handles=handles, loc="outside lower center", ncol=min(8, len(handles)),
                       fontsize=options.font_size * 0.85, frameon=False)
    if options.title:
        fig.suptitle(options.title, fontsize=options.font_size * 1.2, color=TEXT_PRIMARY)
    return fig


def save_figure(fig: Figure, path: str | Path, dpi: Optional[int] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi or fig.dpi, facecolor="white")
    return path


def export_interval_figures(snapshot, intervals, out_dir: str | Path, options: FigureOptions,
                            fmt: str = "png", padding_sec: float = 0.0, progress=None) -> list[Path]:
    """Export one figure per interval ``(index, label, start, end)``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    total = len(intervals)
    for count, (index, label, start, end) in enumerate(intervals, start=1):
        lo = max(snapshot.data_range[0], start - padding_sec)
        hi = min(snapshot.data_range[1], end + padding_sec)
        if hi <= lo:
            continue
        per = FigureOptions(**{**options.__dict__, "x_range": (lo, hi),
                               "title": options.title or f"{label} ({format_hms_ms(start)} – {format_hms_ms(end)})"})
        fig = render_signal_figure(snapshot, per)
        written.append(save_figure(fig, out_dir / f"{index:04d}_{safe_filename(label)}.{fmt}"))
        if progress is not None:
            progress(count, total)
    return written


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def export_plot_csv(snapshot, path: str | Path, x_range: tuple[float, float],
                    visible_only: bool = True) -> int:
    """Write the plotted samples within *x_range* (one column per series)."""
    lo, hi = x_range
    frames = []
    for panel in snapshot.panels:
        for series in panel.series:
            if visible_only and not series.visible:
                continue
            mask = (series.x >= lo) & (series.x <= hi)
            if not mask.any():
                continue
            frames.append(pd.Series(series.y[mask], index=np.round(series.x[mask], 6), name=series.name)
                          .groupby(level=0).first())
    if not frames:
        raise ValueError("No visible samples in the selected range.")
    table = pd.concat(frames, axis=1).sort_index()
    table.index.name = "video_time_sec"
    table = table.reset_index()
    if snapshot.time_zero is not None:
        stamps = snapshot.time_zero + pd.to_timedelta(table["video_time_sec"], unit="s")
        table.insert(1, "timestamp_utc", stamps.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
    labels = np.full(len(table), "", dtype=object)
    for start, end, _color, label in snapshot.intervals:
        inside = (table["video_time_sec"] >= start) & (table["video_time_sec"] <= end)
        labels[inside.to_numpy()] = label
    table.insert(2 if snapshot.time_zero is not None else 1, "label", labels)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return len(table)


# ---------------------------------------------------------------------------
# Composite snapshot
# ---------------------------------------------------------------------------

def render_composite_snapshot(frames: list[SnapshotFrame], snapshot, cursor_sec: float,
                              window_sec: float = 30.0, title: str = "",
                              width_in: float = 14.0, dpi: int = 150) -> Figure:
    """Camera frames at the playhead above the signals around the playhead."""
    import cv2

    cams = [f for f in frames if f.image_bgr is not None]
    panels = [p for p in (snapshot.panels if snapshot is not None else []) if any(s.visible for s in p.series)]
    has_track = snapshot is not None and bool(snapshot.intervals)
    if cams:
        aspect = np.median([f.image_bgr.shape[1] / f.image_bgr.shape[0] for f in cams])
        cam_h = width_in / len(cams) / aspect
        cam_h = min(cam_h, 7.0)
    else:
        cam_h = 0.0
    sig_h = 1.8 * len(panels) + (0.5 if has_track else 0.0)
    height = cam_h + sig_h + 0.6 + (0.3 if title else 0.0)
    fig = Figure(figsize=(width_in, max(2.0, height)), dpi=dpi, facecolor="white", layout="constrained")
    FigureCanvasAgg(fig)
    rows = (1 if cams else 0) + (1 if has_track else 0) + len(panels)
    ratios = ([cam_h] if cams else []) + ([0.45] if has_track else []) + [1.8] * len(panels)
    grid = fig.add_gridspec(rows, max(1, len(cams)), height_ratios=ratios)
    row = 0
    if cams:
        for col, frame in enumerate(cams):
            ax = fig.add_subplot(grid[0, col])
            ax.imshow(cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2RGB))
            ax.set_title(frame.label, fontsize=9, color=TEXT_PRIMARY)
            ax.axis("off")
        row = 1
    lo = max(snapshot.data_range[0], cursor_sec - window_sec / 2) if snapshot else 0.0
    hi = min(snapshot.data_range[1], cursor_sec + window_sec / 2) if snapshot else 1.0
    if snapshot is not None and hi - lo < window_sec:
        if lo <= snapshot.data_range[0]:
            hi = min(snapshot.data_range[1], lo + window_sec)
        else:
            lo = max(snapshot.data_range[0], hi - window_sec)
    shared = None
    if has_track:
        ax = fig.add_subplot(grid[row, :])
        _draw_label_track(ax, snapshot.intervals, lo, hi, 8)
        ax.axvline(cursor_sec, color=CURSOR, linewidth=1.5)
        ax.tick_params(bottom=False, labelbottom=False)
        ax.set_ylabel("Labels", fontsize=8, color=TEXT_SECONDARY, rotation=0, ha="right", va="center")
        shared = ax
        row += 1
    locator, formatter = _time_formatter("video", None, hi - lo)
    for index, panel in enumerate(panels):
        ax = fig.add_subplot(grid[row, :], sharex=shared)
        shared = shared or ax
        _style_axes(ax, 8)
        drawn = 0
        for series in panel.series:
            if not series.visible:
                continue
            mask = (series.x >= lo) & (series.x <= hi)
            ax.plot(series.x[mask], series.y[mask], linewidth=1.0, label=series.name,
                    color=SERIES_COLORS[series.color_index % len(SERIES_COLORS)])
            drawn += 1
        for start, end, color, _label in _visible_intervals(snapshot.intervals, lo, hi):
            ax.axvspan(max(start, lo), min(end, hi), color=color, alpha=0.15, linewidth=0, zorder=0)
        ax.axvline(cursor_sec, color=CURSOR, linewidth=1.5)
        ax.set_ylabel(panel.y_label or "Value", fontsize=8)
        if drawn > 1:
            ax.legend(fontsize=7, loc="upper right", frameon=False)
        ax.set_xlim(lo, hi)
        if index < len(panels) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            ax.set_xlabel("Time from video start (h:mm:ss)", fontsize=8)
        row += 1
    heading = title or f"Playhead {format_hms_ms(cursor_sec)}"
    fig.suptitle(heading, fontsize=10, color=TEXT_PRIMARY)
    return fig
