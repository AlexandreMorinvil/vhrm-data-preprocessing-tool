"""Mosaic video export: composites all camera angles + signal graph into one MP4.

Layout:
  ┌──────────────────────────────────────────┐
  │ [Frame: N]                [Time: HH:MM…] │  top bar overlay
  ├─────────────────────┬────────────────────┤
  │   Camera 1          │   Camera 2         │  camera grid
  ├─────────────────────┼────────────────────┤
  │   Camera 3          │   (black if <4)    │
  ├─────────────────────┴────────────────────┤
  │   Signal plot with moving cursor         │  signal strip
  └──────────────────────────────────────────┘
"""
from __future__ import annotations

import logging
import math
import subprocess
import threading
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .ffmpeg_utils import find_ffmpeg
from .face_privacy import FaceAnonymizer
from .signals import is_aux_signal_column
from .visual_export import SERIES_COLORS

log = logging.getLogger(__name__)

MOSAIC_PRESET_NAMES = ["Speed", "Balanced", "Quality"]

_MOSAIC_PRESETS: dict[str, dict[str, object]] = {
    # Smaller frame + faster encoder for quick review exports
    "speed": {
        "max_cell_w": 640,
        "max_cell_h": 360,
        "signal_ratio": 0.24,
        "x264_preset": "ultrafast",
        "crf": 28,
    },
    # Existing default behavior
    "balanced": {
        "max_cell_w": 960,
        "max_cell_h": 540,
        "signal_ratio": 0.30,
        "x264_preset": "fast",
        "crf": 20,
    },
    # Better compression quality (slower encode)
    "quality": {
        "max_cell_w": 960,
        "max_cell_h": 540,
        "signal_ratio": 0.30,
        "x264_preset": "medium",
        "crf": 17,
    },
}


def normalise_mosaic_preset(name: str) -> str:
    """Normalise user-facing preset text to internal key."""
    key = (name or "").strip().lower()
    if key in _MOSAIC_PRESETS:
        return key
    return "balanced"

# ---------------------------------------------------------------------------
# Grid layout helpers
# ---------------------------------------------------------------------------

def _grid_dims(n_cameras: int) -> tuple[int, int]:
    """Return (rows, cols) for the camera grid."""
    if n_cameras <= 1:
        return 1, 1
    if n_cameras == 2:
        return 1, 2
    # 3–4 → 2×2,  5–6 → 2×3
    cols = math.ceil(math.sqrt(n_cameras))
    rows = math.ceil(n_cameras / cols)
    return rows, cols


def _cell_size(n_cameras: int, max_cell_w: int = 960, max_cell_h: int = 540
               ) -> tuple[int, int, int, int]:
    """Return (cell_w, cell_h, grid_w, grid_h) for the camera grid.

    Each cell is at most *max_cell_w × max_cell_h*; the grid is
    ``cols * cell_w × rows * cell_h``.
    """
    rows, cols = _grid_dims(n_cameras)
    cell_w, cell_h = max_cell_w, max_cell_h
    grid_w = cols * cell_w
    grid_h = rows * cell_h
    return cell_w, cell_h, grid_w, grid_h


def _fit_rect(src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int, int, int]:
    """Return fitted rectangle (x_off, y_off, w, h) preserving aspect ratio."""
    if src_w <= 0 or src_h <= 0:
        return 0, 0, 0, 0
    scale = min(dst_w / src_w, dst_h / src_h)
    fit_w = max(1, int(src_w * scale))
    fit_h = max(1, int(src_h * scale))
    x_off = (dst_w - fit_w) // 2
    y_off = (dst_h - fit_h) // 2
    return x_off, y_off, fit_w, fit_h


# ---------------------------------------------------------------------------
# Signal plot renderer (matplotlib agg → numpy)
# ---------------------------------------------------------------------------

def _build_signal_strip_background(
    signal_df: Optional[pd.DataFrame],
    video_duration_sec: float,
    strip_w: int,
    strip_h: int,
    time_zero: Optional[pd.Timestamp] = None,
) -> tuple[np.ndarray, int, int]:
    """Render static signal background once and return (image, plot_x0, plot_x1).

    *time_zero* is the absolute time of the first output frame; when omitted
    the first signal sample is used (legacy behaviour).
    """

    dpi = 100
    fig = Figure(figsize=(strip_w / dpi, strip_h / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    if signal_df is not None and not signal_df.empty:
        # Determine time axis (relative seconds from video start)
        if "timestamp_utc" in signal_df.columns:
            t0 = time_zero if time_zero is not None else signal_df["timestamp_utc"].iloc[0]
            rel_sec = (signal_df["timestamp_utc"] - t0).dt.total_seconds()
        else:
            rel_sec = np.linspace(0, video_duration_sec, len(signal_df))

        # Detect format
        if "sensor_id" in signal_df.columns:
            for index, sid in enumerate(signal_df["sensor_id"].unique()):
                sub = signal_df[signal_df["sensor_id"] == sid]
                t0_sub = time_zero if time_zero is not None else sub["timestamp_utc"].iloc[0]
                rs = (sub["timestamp_utc"] - t0_sub).dt.total_seconds()
                style = ("--" if sid == "averaged" else "-")
                ax.plot(rs, sub["value"], label=str(sid), linewidth=0.8, linestyle=style,
                        color=SERIES_COLORS[index % len(SERIES_COLORS)])
        else:
            value_cols = [
                c for c in signal_df.columns
                if c != "timestamp_utc" and not is_aux_signal_column(str(c))
            ]
            for index, col in enumerate(value_cols):
                style = ("--" if col == "averaged" else "-")
                values = pd.to_numeric(signal_df[col], errors="coerce")
                valid = values.notna()
                ax.plot(rel_sec[valid], values[valid], label=col, linewidth=0.8, linestyle=style,
                        color=SERIES_COLORS[index % len(SERIES_COLORS)])

        if len(signal_df.columns) > 2:
            ax.legend(fontsize=7, loc="upper right")

    ax.set_xlim(0, max(video_duration_sec, 0.1))
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Value", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.4)

    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(int(fig.get_figheight() * dpi), int(fig.get_figwidth() * dpi), 4)
    bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)

    # Ensure exact target size
    if bgr.shape[1] != strip_w or bgr.shape[0] != strip_h:
        bgr = cv2.resize(bgr, (strip_w, strip_h), interpolation=cv2.INTER_AREA)

    # Compute where x=0 and x=video_duration map in pixel coordinates.
    px0 = int(ax.transData.transform((0.0, 0.0))[0])
    px1 = int(ax.transData.transform((max(video_duration_sec, 0.1), 0.0))[0])
    px0 = max(0, min(strip_w - 1, px0))
    px1 = max(px0 + 1, min(strip_w - 1, px1))

    return bgr, px0, px1


def _signal_cursor_x(current_sec: float, duration_sec: float, x0: int, x1: int) -> int:
    """Map time in seconds to the plotted x pixel range."""
    if duration_sec <= 0:
        return x0
    t = max(0.0, min(current_sec, duration_sec))
    frac = t / duration_sec
    return int(round(x0 + frac * (x1 - x0)))


# ---------------------------------------------------------------------------
# Text overlay
# ---------------------------------------------------------------------------

def _overlay_text(frame: np.ndarray, frame_no: int, time_sec: float) -> None:
    """Draw frame counter + time in the top-right area with a dark background."""
    h, w = frame.shape[:2]
    time_str = _format_overlay_time(time_sec)
    txt_frame = f"Frame: {frame_no}"
    txt_time = f"Time: {time_str}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    color = (255, 255, 255)
    bg_color = (0, 0, 0)

    (tw1, th1), _ = cv2.getTextSize(txt_frame, font, scale, thickness)
    (tw2, th2), _ = cv2.getTextSize(txt_time, font, scale, thickness)

    pad = 6
    box_w = max(tw1, tw2) + 2 * pad
    box_h = th1 + th2 + 3 * pad
    x0 = w - box_w - 8
    y0 = 8

    # Semi-transparent dark background
    overlay = frame[y0:y0 + box_h, x0:x0 + box_w].copy()
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), bg_color, -1)
    alpha = 0.6
    frame[y0:y0 + box_h, x0:x0 + box_w] = cv2.addWeighted(
        frame[y0:y0 + box_h, x0:x0 + box_w], alpha, overlay, 1 - alpha, 0
    )

    cv2.putText(frame, txt_frame, (x0 + pad, y0 + pad + th1),
                font, scale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame, txt_time, (x0 + pad, y0 + 2 * pad + th1 + th2),
                font, scale, color, thickness, cv2.LINE_AA)


def _format_overlay_time(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


# ---------------------------------------------------------------------------
# Main export function (runs in a worker thread)
# ---------------------------------------------------------------------------

def export_mosaic(
    video_paths: list[str],
    camera_labels: list[str],
    fps: float,
    total_frames: int,
    video_duration_sec: float,
    output_path: str,
    signal_df: Optional[pd.DataFrame] = None,
    start_sec: float = 0.0,
    end_sec: float = 0.0,
    quality_preset: str = "balanced",
    ffmpeg_path: str = "",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    blur_faces: bool = False,
    signal_time_zero: Optional[pd.Timestamp] = None,
    audio_path: str = "",
) -> str:
    """Render a mosaic video frame-by-frame and pipe to FFmpeg.

    Parameters
    ----------
    video_paths : list[str]
        Paths to the camera video files.
    camera_labels : list[str]
        Display labels for each camera.
    fps : float
        Output frame rate.
    total_frames : int
        Total frames to render.
    video_duration_sec : float
        Duration of the video for the signal plot x-axis.
    output_path : str
        Destination MP4 file.
    signal_df : Optional[pd.DataFrame]
        Signal data (wide or long format with ``timestamp_utc``).
    start_sec : float
        Start offset for trimming cameras (0 for full video).
    end_sec : float
        End offset (0 means use full video).
    ffmpeg_path : str
        Path to ffmpeg binary.
    progress_callback : callable(current_frame, total_frames)
        Called after each frame.
    cancel_check : callable() → bool
        Return True to abort.
    signal_time_zero : pd.Timestamp, optional
        Absolute time of video time 0 (the signal anchor). When omitted, the
        first signal sample is treated as video time 0 (legacy behaviour).
    audio_path : str, optional
        Camera video whose audio track is added to the mosaic (same range).

    Returns
    -------
    str  —  path of the written file.
    """
    ffmpeg = ffmpeg_path or find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")

    n_cams = len(video_paths)
    if n_cams == 0:
        raise ValueError("No video paths provided")

    preset_key = normalise_mosaic_preset(quality_preset)
    preset = _MOSAIC_PRESETS[preset_key]

    # --- Compute layout ---
    cell_w, cell_h, grid_w, grid_h = _cell_size(
        n_cams,
        max_cell_w=int(preset["max_cell_w"]),
        max_cell_h=int(preset["max_cell_h"]),
    )
    strip_h = int(grid_h * float(preset["signal_ratio"]))
    out_w = grid_w
    out_h = grid_h + strip_h
    # Make dimensions even (required by H.264)
    out_w = out_w if out_w % 2 == 0 else out_w + 1
    out_h = out_h if out_h % 2 == 0 else out_h + 1
    strip_h = out_h - grid_h

    rows, cols = _grid_dims(n_cams)

    # --- Open camera captures ---
    caps: list[Optional[cv2.VideoCapture]] = []
    for vp in video_paths:
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            log.warning("Cannot open video: %s", vp)
            caps.append(None)
        else:
            if start_sec > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
            caps.append(cap)

    # --- Clip the signal to the requested range ---
    sig_clip = None
    strip_zero = None
    if signal_time_zero is not None:
        zero = pd.Timestamp(signal_time_zero)
        zero = zero.tz_localize("UTC") if zero.tzinfo is None else zero.tz_convert("UTC")
        strip_zero = zero + pd.Timedelta(seconds=max(0.0, start_sec))
    if signal_df is not None and not signal_df.empty:
        sig_clip = signal_df.copy()
        if "timestamp_utc" in sig_clip.columns and strip_zero is not None:
            _rel = (sig_clip["timestamp_utc"] - strip_zero).dt.total_seconds()
            sig_clip = sig_clip[(_rel >= 0) & (_rel <= video_duration_sec)].copy()
        elif "timestamp_utc" in sig_clip.columns:
            t0 = sig_clip["timestamp_utc"].iloc[0]
            # Compute relative seconds for the whole signal
            _rel = (sig_clip["timestamp_utc"] - t0).dt.total_seconds()
            if start_sec > 0 or (end_sec > 0 and end_sec < _rel.iloc[-1]):
                mask = _rel >= start_sec
                if end_sec > 0:
                    mask = mask & (_rel <= end_sec)
                sig_clip = sig_clip[mask].copy()

    # --- Start FFmpeg encoder ---
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{out_w}x{out_h}",
        "-r", str(fps),
        "-i", "-",
    ]
    if audio_path:
        cmd += ["-ss", f"{max(0.0, start_sec):.6f}", "-t", f"{total_frames / fps:.6f}",
                "-i", audio_path, "-map", "0:v:0", "-map", "1:a:0?"]
    cmd += [
        "-c:v", "libx264",
        "-preset", str(preset["x264_preset"]),
        "-crf", str(preset["crf"]),
        "-pix_fmt", "yuv420p",
    ]
    if audio_path:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += [output_path]
    log.info("Mosaic FFmpeg: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    stderr_lines: list[str] = [""]

    def _drain():
        try:
            stderr_lines[0] = proc.stderr.read().decode(errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass

    drain_t = threading.Thread(target=_drain, daemon=True)
    drain_t.start()

    # Precompute fitted placement for each camera to avoid repeated math.
    cam_layouts: list[tuple[int, int, int, int]] = []
    for cap in caps:
        if cap is None:
            cam_layouts.append((0, 0, 0, 0))
            continue
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cam_layouts.append(_fit_rect(src_w, src_h, cell_w, cell_h))

    # Render the expensive matplotlib plot only once.
    signal_bg, signal_x0, signal_x1 = _build_signal_strip_background(
        sig_clip, video_duration_sec, out_w, strip_h, time_zero=strip_zero
    )
    anonymizers = [FaceAnonymizer(fps=fps) for _ in caps] if blur_faces else []

    # Pre-allocate destination buffers to reduce per-frame allocations.
    grid_frame = np.zeros((grid_h, out_w, 3), dtype=np.uint8)
    composite = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    try:
        for frame_no in range(total_frames):
            if cancel_check and cancel_check():
                proc.stdin.close()  # type: ignore[union-attr]
                proc.kill()
                proc.wait()
                drain_t.join(timeout=3)
                raise RuntimeError("Cancelled by user")

            current_sec = frame_no / fps

            # --- Build camera grid (in-place) ---
            grid_frame.fill(0)
            cam_idx = 0
            for r in range(rows):
                for c in range(cols):
                    x0 = c * cell_w
                    y0 = r * cell_h
                    cell_view = grid_frame[y0:y0 + cell_h, x0:x0 + cell_w]

                    if cam_idx < n_cams and caps[cam_idx] is not None:
                        ret, raw = caps[cam_idx].read()
                        if ret:
                            if blur_faces:
                                raw = anonymizers[cam_idx].process(raw, current_sec)
                            lx, ly, lw, lh = cam_layouts[cam_idx]
                            if lw > 0 and lh > 0:
                                resized = cv2.resize(raw, (lw, lh), interpolation=cv2.INTER_AREA)
                                cell_view[ly:ly + lh, lx:lx + lw] = resized
                            # Draw camera label at top-left
                            if cam_idx < len(camera_labels):
                                lbl = camera_labels[cam_idx]
                                cv2.putText(cell_view, lbl, (8, 24),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                            (255, 255, 255), 1, cv2.LINE_AA)

                    cam_idx += 1

            # --- Build signal strip (copy static + draw moving cursor) ---
            sig_strip = signal_bg.copy()
            cursor_x = _signal_cursor_x(current_sec, video_duration_sec, signal_x0, signal_x1)
            cv2.line(sig_strip, (cursor_x, 0), (cursor_x, strip_h - 1), (0, 0, 255), 2)

            # --- Composite in-place ---
            composite[:grid_h, :out_w] = grid_frame
            composite[grid_h:out_h, :out_w] = sig_strip

            # --- Text overlay ---
            _overlay_text(composite, frame_no, current_sec)

            # --- Pipe to FFmpeg ---
            proc.stdin.write(composite.tobytes())  # type: ignore[union-attr]

            if progress_callback:
                progress_callback(frame_no + 1, total_frames)

    except BrokenPipeError:
        log.error("FFmpeg pipe broke early")
    finally:
        for cap in caps:
            if cap is not None:
                cap.release()
        try:
            proc.stdin.close()  # type: ignore[union-attr]
        except Exception:
            pass
        proc.wait()
        drain_t.join(timeout=5)

    if proc.returncode != 0:
        err = stderr_lines[0]
        log.error("FFmpeg mosaic failed: %s", err)
        raise RuntimeError(f"FFmpeg mosaic error:\n{err}")

    return output_path


# ---------------------------------------------------------------------------
# QThread worker
# ---------------------------------------------------------------------------

class MosaicWorker(QThread):
    """Async worker for mosaic export with progress and cancel support."""
    progress = pyqtSignal(int, int)       # current_frame, total_frames
    finished = pyqtSignal(bool, str)      # success, message
    log_message = pyqtSignal(str)

    def __init__(
        self,
        video_paths: list[str],
        camera_labels: list[str],
        fps: float,
        total_frames: int,
        video_duration_sec: float,
        output_path: str,
        signal_df: Optional[pd.DataFrame] = None,
        start_sec: float = 0.0,
        end_sec: float = 0.0,
        quality_preset: str = "balanced",
        ffmpeg_path: str = "",
        blur_faces: bool = False,
        signal_time_zero=None,
        audio_path: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._video_paths = video_paths
        self._camera_labels = camera_labels
        self._fps = fps
        self._total_frames = total_frames
        self._video_duration_sec = video_duration_sec
        self._output_path = output_path
        self._signal_df = signal_df
        self._start_sec = start_sec
        self._end_sec = end_sec
        self._quality_preset = quality_preset
        self._ffmpeg_path = ffmpeg_path
        self._blur_faces = blur_faces
        self._signal_time_zero = signal_time_zero
        self._audio_path = audio_path
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def run(self):
        try:
            self.log_message.emit(f"Starting mosaic export → {self._output_path}")
            export_mosaic(
                video_paths=self._video_paths,
                camera_labels=self._camera_labels,
                fps=self._fps,
                total_frames=self._total_frames,
                video_duration_sec=self._video_duration_sec,
                output_path=self._output_path,
                signal_df=self._signal_df,
                start_sec=self._start_sec,
                end_sec=self._end_sec,
                quality_preset=self._quality_preset,
                ffmpeg_path=self._ffmpeg_path,
                progress_callback=self._on_progress,
                cancel_check=lambda: self._cancelled,
                blur_faces=self._blur_faces,
                signal_time_zero=self._signal_time_zero,
                audio_path=self._audio_path,
            )
            if self._cancelled:
                self.finished.emit(False, "Cancelled.")
            else:
                self.finished.emit(True, self._output_path)
        except Exception as exc:
            self.finished.emit(False, str(exc))

    def _on_progress(self, current: int, total: int):
        self.progress.emit(current, total)
