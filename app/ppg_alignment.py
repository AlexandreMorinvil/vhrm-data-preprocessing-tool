"""Video/signal alignment from an algorithmic rPPG estimate.

Pipeline (one labelled-segment camera clip at a time):

1. FFmpeg decodes the clip at ``fs`` frames per second, scaled down so face
   detection and colour averaging stay fast (hardware decoding when available).
2. YuNet finds the face every few frames (optionally every frame); the box is
   smoothed between detections (optionally without lag, for head motion). The
   region of interest (ROI) is the inner part of the face box, restricted to
   skin-coloured pixels.
3. The mean R, G, B of the ROI per frame is cached; the ROI itself is written
   to a small H.264 video so it can be played next to the cameras.
4. An rPPG method (see :mod:`app.rppg`) turns the colour trace into a pulse
   waveform, which is cross-correlated with the synthetic PPG over a range of
   offsets.

Offset convention: a positive offset places the signals *later* on the video
timeline. The correlation for offset ``L`` compares the pulse at video time
``t`` with the reference at ``t - L``.
"""
from __future__ import annotations

import bisect
import json
import logging
import os
import re
import subprocess
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from scipy import signal as sp_signal

from .face_privacy import FacePrivacySettings, detect_faces_scored
from .rppg import PULSE_BAND_HZ, bandpass, dominant_rate_bpm, standardise

log = logging.getLogger(__name__)

CACHE_VERSION = 1
ROI_VIDEO_SIZE = (150, 200)  # width, height of the ROI preview video
_NO_FACE_GREY = 40
# Centred smoothing: box position averaged over 3 frames, size (median) over 15.
_CENTRED_POSITION_FRAMES = 3
_CENTRED_SIZE_FRAMES = 15
# Settings left out of the cache signature while off, so traces cached before they
# existed stay valid.
_OPTIONAL_SETTINGS = ("detect_every_frame", "centred_smoothing")


@dataclass
class FaceTraceSettings:
    fs: float = 30.0
    analysis_size: int = 960          # long side of the decoded frames (pixels)
    detect_every: int = 10            # run face detection every N frames
    score_threshold: float = 0.6
    box_smoothing: float = 0.35       # weight of a new detection in the smoothed box
    roi_width_fraction: float = 0.8   # keep the central part of the face box (drops background)
    skin_mask: bool = True
    hardware_decoding: bool = True
    # Options for head motion (off: the original tracking). See _BoxTracker.
    detect_every_frame: bool = False  # detection in every frame instead of every detect_every
    centred_smoothing: bool = False   # box interpolated between detections, smoothed without lag

    @classmethod
    def from_dict(cls, data: dict) -> "FaceTraceSettings":
        names = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in (data or {}).items() if key in names})


@dataclass
class FaceTrace:
    """Per-frame mean colour of the face ROI of one camera clip."""

    fs: float
    rgb: np.ndarray                 # (n, 3) mean R, G, B; NaN where no face was available
    detected: np.ndarray            # (n,) True where the face was detected near this frame
    boxes: np.ndarray               # (n, 4) ROI box x, y, w, h in analysed-frame pixels
    frame_size: tuple[int, int]     # analysed frame width, height
    video_path: str = ""
    settings: FaceTraceSettings = field(default_factory=FaceTraceSettings)

    @property
    def times(self) -> np.ndarray:
        return np.arange(len(self.rgb)) / self.fs

    @property
    def face_coverage(self) -> float:
        return float(np.mean(self.detected)) if len(self.detected) else 0.0


class Cancelled(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Face ROI
# ---------------------------------------------------------------------------

def _iou(a, b) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def choose_face(detections, previous: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Keep following the previous face; otherwise take the largest one."""
    if not detections:
        return None
    boxes = [np.asarray(d[:4], dtype=np.float64) for d in detections]
    if previous is not None:
        overlaps = [_iou(box, previous) for box in boxes]
        best = int(np.argmax(overlaps))
        if overlaps[best] > 0.1:
            return boxes[best]
    return max(boxes, key=lambda box: box[2] * box[3])


def roi_from_face(face: np.ndarray, width_fraction: float, frame_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Inner part of a face box, clipped to the frame."""
    x, y, w, h = face
    margin = w * (1.0 - width_fraction) / 2.0
    x0 = int(round(max(0.0, x + margin)))
    y0 = int(round(max(0.0, y)))
    x1 = int(round(min(frame_size[0], x + w - margin)))
    y1 = int(round(min(frame_size[1], y + h)))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def skin_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """Chai & Ngan YCrCb skin rule (boolean mask)."""
    ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[..., 1], ycrcb[..., 2]
    return (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)


def roi_mean_rgb(crop_bgr: np.ndarray, use_skin_mask: bool) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Mean R, G, B of the ROI (skin pixels when enough are found) and the mask used."""
    if crop_bgr.size == 0:
        return np.full(3, np.nan), None
    mask = None
    if use_skin_mask:
        mask = skin_mask(crop_bgr)
        if mask.mean() < 0.15:  # lighting outside the skin rule: use the whole ROI
            mask = None
    pixels = crop_bgr[mask] if mask is not None else crop_bgr.reshape(-1, 3)
    mean_bgr = pixels.reshape(-1, 3).mean(axis=0)
    return mean_bgr[::-1].astype(np.float64), mask


def roi_preview_frame(crop_bgr: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """ROI resized for the preview video; pixels outside the skin mask are dimmed."""
    width, height = ROI_VIDEO_SIZE
    if crop_bgr.size == 0:
        return np.full((height, width, 3), _NO_FACE_GREY, np.uint8)
    shown = crop_bgr
    if mask is not None:
        shown = crop_bgr.copy()
        shown[~mask] = (shown[~mask] * 0.3).astype(np.uint8)
    return cv2.resize(shown, (width, height), interpolation=cv2.INTER_AREA)


class _BoxTracker:
    """Face box of every frame, from face detections run on some of the frames.

    Original tracking: a detection every ``detect_every`` frames, blended into the
    previous box (``box_smoothing``). Cheap and steady, but the box trails a head
    that moves quickly (e.g. during squats) and partly leaves the face.

    ``detect_every_frame`` runs the detection in every frame. ``centred_smoothing``
    interpolates the box between detections and smooths it over a window centred on
    each frame, so it does not lag behind the head; a frame is then released a few
    frames later, once the detections it depends on are known.

    :meth:`push` and :meth:`flush` return ``(frame, box or None, detected)`` in frame order.
    """

    def __init__(self, settings: FaceTraceSettings, detect: Callable[[np.ndarray], list]):
        self._settings = settings
        self._detect = detect
        self._interval = 1 if settings.detect_every_frame else max(1, settings.detect_every)
        self._hold_frames = int(2.0 * settings.fs)  # a face not re-detected for 2 s is flagged
        self._frames_since_detection = 10 ** 9
        self._face: Optional[np.ndarray] = None  # original: smoothed box; centred: last detection
        # Centred smoothing: successful detections and the frames waiting for later ones.
        self._indices: list[int] = []
        self._boxes: list[np.ndarray] = []
        self._pending: deque = deque()
        self._latest = -1
        self._lookahead = self._interval + _CENTRED_SIZE_FRAMES // 2

    def push(self, index: int, frame: np.ndarray) -> list[tuple[np.ndarray, Optional[np.ndarray], bool]]:
        if self._face is None or index % self._interval == 0:
            found = choose_face(self._detect(frame), self._face)
            if found is not None:
                if self._settings.centred_smoothing:
                    self._face = found
                    self._indices.append(index)
                    self._boxes.append(found)
                else:
                    smoothing = self._settings.box_smoothing
                    self._face = found if self._face is None else (1.0 - smoothing) * self._face + smoothing * found
                self._frames_since_detection = 0
        self._frames_since_detection += 1
        detected = self._frames_since_detection <= self._hold_frames
        if self._face is None:
            return [(frame, None, False)]
        if not self._settings.centred_smoothing:
            return [(frame, self._face, detected)]
        self._latest = index
        self._pending.append((index, frame, detected))
        return self._release(index - self._lookahead)

    def flush(self) -> list[tuple[np.ndarray, Optional[np.ndarray], bool]]:
        return self._release(self._latest)

    def _release(self, last: int) -> list[tuple[np.ndarray, Optional[np.ndarray], bool]]:
        released = []
        while self._pending and self._pending[0][0] <= last:
            index, frame, detected = self._pending.popleft()
            released.append((frame, self._centred_box(index), detected))
        return released

    def _centred_box(self, index: int) -> np.ndarray:
        half = _CENTRED_SIZE_FRAMES // 2
        frames = np.arange(max(0, index - half), min(self._latest, index + half) + 1)
        # Detections around the window (one more on each side to interpolate from).
        lo = max(0, bisect.bisect_left(self._indices, frames[0]) - 1)
        hi = bisect.bisect_right(self._indices, frames[-1]) + 1
        indices = np.asarray(self._indices[lo:hi], dtype=np.float64)
        boxes = np.asarray(self._boxes[lo:hi], dtype=np.float64)
        centres = boxes[:, :2] + boxes[:, 2:] / 2.0
        track = np.column_stack([np.interp(frames, indices, centres[:, 0]),
                                 np.interp(frames, indices, centres[:, 1]),
                                 np.interp(frames, indices, boxes[:, 2]),
                                 np.interp(frames, indices, boxes[:, 3])])
        near = np.abs(frames - index) <= _CENTRED_POSITION_FRAMES // 2
        centre = track[near, :2].mean(axis=0)
        size = np.median(track[:, 2:], axis=0)
        return np.concatenate([centre - size / 2.0, size])


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def display_geometry(video_path: str) -> tuple[int, int, float]:
    """Displayed width, height (rotation applied) and an approximate duration."""
    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read a frame from {video_path}")
        height, width = frame.shape[:2]
    finally:
        capture.release()
    return width, height, count / fps if fps > 0 else 0.0


def analysis_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    scale = min(1.0, long_side / max(width, height))
    return max(2, int(round(width * scale / 2)) * 2), max(2, int(round(height * scale / 2)) * 2)


def _drain(pipe, chunks: list[bytes]) -> threading.Thread:
    thread = threading.Thread(target=lambda: chunks.append(pipe.read() if pipe else b""), daemon=True)
    thread.start()
    return thread


def _decode_frames(video_path: str, ffmpeg: str, fs: float, size: tuple[int, int], hardware: bool):
    """Yield BGR frames at *fs* (frame k shows video time k / fs)."""
    width, height = size
    command = [ffmpeg, "-v", "error", "-nostdin"]
    if hardware:
        command += ["-hwaccel", "auto"]
    command += [
        "-i", video_path, "-an", "-sn", "-dn",
        "-vf", f"fps={fs:g}:round=near,scale={width}:{height}:flags=area",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    errors: list[bytes] = []
    drain = _drain(process.stderr, errors)
    frame_bytes = width * height * 3
    reached_end = False
    try:
        while True:
            data = process.stdout.read(frame_bytes)
            if len(data) < frame_bytes:
                reached_end = True
                break
            yield np.frombuffer(data, np.uint8).reshape(height, width, 3)
    finally:
        if reached_end:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        if process.poll() is None:
            process.kill()
        process.wait()
        drain.join(timeout=5)
    if process.returncode not in (0, None):
        message = b"".join(errors).decode(errors="replace").strip()
        if message:
            raise RuntimeError(f"FFmpeg decoding failed: {message}")


class _RoiVideoWriter:
    """Writes the ROI preview video through FFmpeg (H.264, keyframe every 0.5 s)."""

    def __init__(self, path: str, ffmpeg: str, fs: float):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._temporary = str(Path(path).with_suffix(".partial.mp4"))
        width, height = ROI_VIDEO_SIZE
        command = [
            ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fs:g}", "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-g", str(max(1, int(fs // 2))), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            self._temporary,
        ]
        self._process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._errors: list[bytes] = []
        self._drain = _drain(self._process.stderr, self._errors)

    def write(self, frame: np.ndarray) -> None:
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError):
            pass

    def close(self, keep: bool) -> None:
        try:
            self._process.stdin.close()
        except OSError:
            pass
        if not keep:
            self._process.kill()
        self._process.wait()
        self._drain.join(timeout=5)
        if keep and self._process.returncode == 0:
            try:
                os.replace(self._temporary, self.path)
            except OSError as exc:  # e.g. the previous ROI video is open in a player
                log.warning("Could not replace %s: %s", self.path, exc)
                Path(self._temporary).unlink(missing_ok=True)
        else:
            if keep:
                log.warning("ROI video encoding failed: %s",
                            b"".join(self._errors).decode(errors="replace"))
            Path(self._temporary).unlink(missing_ok=True)


def extract_face_trace(
    video_path: str,
    settings: Optional[FaceTraceSettings] = None,
    *,
    ffmpeg: str,
    roi_video_path: str = "",
    progress: Optional[Callable[[int, int], None]] = None,
    cancel: Optional[Callable[[], bool]] = None,
) -> FaceTrace:
    """Track the face in *video_path* and return the mean colour of its ROI per frame."""
    settings = settings or FaceTraceSettings()
    if not ffmpeg:
        raise RuntimeError("FFmpeg not found.")
    width, height, duration = display_geometry(video_path)
    size = analysis_size(width, height, settings.analysis_size)
    expected = max(1, int(duration * settings.fs))
    detector_settings = FacePrivacySettings(detection_quality="fast",
                                            score_threshold=settings.score_threshold)
    writer = _RoiVideoWriter(roi_video_path, ffmpeg, settings.fs) if roi_video_path else None

    colours: list[np.ndarray] = []
    detected: list[bool] = []
    boxes: list[tuple[float, float, float, float]] = []
    tracker = _BoxTracker(settings, lambda frame: detect_faces_scored(frame, detector_settings))

    def record(frame: np.ndarray, face: Optional[np.ndarray], face_detected: bool) -> None:
        if face is None:
            colours.append(np.full(3, np.nan))
            detected.append(False)
            boxes.append((np.nan,) * 4)
            if writer:
                writer.write(roi_preview_frame(np.empty((0, 0, 3), np.uint8), None))
            return
        x, y, w, h = roi_from_face(face, settings.roi_width_fraction, size)
        crop = frame[y:y + h, x:x + w]
        colour, mask = roi_mean_rgb(crop, settings.skin_mask)
        colours.append(colour)
        detected.append(face_detected)
        boxes.append((x, y, w, h))
        if writer:
            writer.write(roi_preview_frame(crop, mask))

    def frames():
        if settings.hardware_decoding:
            produced = False
            try:
                for frame in _decode_frames(video_path, ffmpeg, settings.fs, size, True):
                    produced = True
                    yield frame
            except RuntimeError:
                if produced:
                    raise
            if produced:
                return
            log.info("Hardware decoding unavailable for %s; using software decoding", video_path)
        yield from _decode_frames(video_path, ffmpeg, settings.fs, size, False)

    finished = False
    try:
        for index, frame in enumerate(frames()):
            if cancel is not None and cancel():
                raise Cancelled("Cancelled by user")
            for released in tracker.push(index, frame):
                record(*released)
            if progress is not None and index % 15 == 0:
                progress(index + 1, max(expected, index + 1))
        for released in tracker.flush():
            record(*released)
        finished = True
    finally:
        if writer:
            writer.close(keep=finished)
    if not colours:
        raise RuntimeError(f"No frames decoded from {video_path}")
    if progress is not None:
        progress(len(colours), len(colours))
    return FaceTrace(
        fs=settings.fs,
        rgb=np.asarray(colours, dtype=np.float64),
        detected=np.asarray(detected, dtype=bool),
        boxes=np.asarray(boxes, dtype=np.float64),
        frame_size=size,
        video_path=video_path,
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def cache_key(segment_folder: str, camera_label: str) -> str:
    text = f"{Path(segment_folder).name}__{camera_label}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def _signature(video_path: str, settings: FaceTraceSettings) -> dict:
    stat = os.stat(video_path)
    values = asdict(settings)
    for name in _OPTIONAL_SETTINGS:
        if not values[name]:
            del values[name]
    return {
        "version": CACHE_VERSION,
        "video": Path(video_path).name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "settings": values,
    }


def cache_paths(cache_dir: str | Path, key: str) -> tuple[Path, Path]:
    directory = Path(cache_dir)
    return directory / f"{key}.npz", directory / f"{key}_roi.mp4"


def save_trace(cache_dir: str | Path, key: str, trace: FaceTrace) -> Path:
    trace_path, _roi_path = cache_paths(cache_dir, key)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    signature = json.dumps(_signature(trace.video_path, trace.settings))
    temporary = trace_path.with_suffix(".partial.npz")
    np.savez_compressed(
        temporary, rgb=trace.rgb, detected=trace.detected, boxes=trace.boxes,
        frame_size=np.asarray(trace.frame_size), signature=np.asarray(signature),
    )
    os.replace(temporary, trace_path)
    return trace_path


def load_trace(cache_dir: str | Path, key: str, video_path: str,
               settings: FaceTraceSettings) -> Optional[FaceTrace]:
    """Return the cached trace when it was computed from the same video and settings."""
    trace_path, _roi_path = cache_paths(cache_dir, key)
    if not trace_path.exists() or not Path(video_path).exists():
        return None
    try:
        with np.load(trace_path, allow_pickle=False) as data:
            stored = json.loads(str(data["signature"]))
            if stored != json.loads(json.dumps(_signature(video_path, settings))):
                return None
            return FaceTrace(
                fs=settings.fs,
                rgb=data["rgb"],
                detected=data["detected"],
                boxes=data["boxes"],
                frame_size=tuple(int(v) for v in data["frame_size"]),
                video_path=video_path,
                settings=settings,
            )
    except (OSError, ValueError, KeyError) as exc:
        log.warning("Ignoring unreadable alignment cache %s: %s", trace_path, exc)
        return None


# ---------------------------------------------------------------------------
# Reference signal and lag search
# ---------------------------------------------------------------------------

def prepare_reference(times_sec: np.ndarray, values: np.ndarray,
                      band_hz: tuple[float, float] = PULSE_BAND_HZ) -> tuple[np.ndarray, np.ndarray]:
    """Band-pass (zero phase) and standardise a timestamped reference waveform."""
    times_sec = np.asarray(times_sec, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(times_sec, kind="stable")
    times_sec, values = times_sec[order], values[order]
    finite = np.isfinite(times_sec) & np.isfinite(values)
    times_sec, values = times_sec[finite], values[finite]
    if len(times_sec) < 16:
        return times_sec, standardise(values)
    step = float(np.median(np.diff(times_sec)))
    if step <= 0:
        return times_sec, standardise(values)
    # Resample on a regular grid so the filter sees a uniform rate.
    grid = np.arange(times_sec[0], times_sec[-1] + step / 2, step)
    regular = np.interp(grid, times_sec, values)
    filtered = standardise(bandpass(regular, 1.0 / step, band_hz))
    return grid, filtered


@dataclass
class LagScan:
    lags: np.ndarray
    correlation: np.ndarray
    peaks: np.ndarray  # indices of local maxima

    @classmethod
    def from_curve(cls, lags: np.ndarray, correlation: np.ndarray) -> "LagScan":
        peaks, _ = sp_signal.find_peaks(np.nan_to_num(correlation, nan=-2.0))
        peaks = np.asarray([p for p in peaks if np.isfinite(correlation[p])], dtype=int)
        return cls(lags=lags, correlation=correlation, peaks=peaks)

    @property
    def valid(self) -> bool:
        return bool(np.isfinite(self.correlation).any())

    def refined(self, index: int) -> float:
        """Lag of sample *index*, refined by a parabola through its neighbours."""
        lag = float(self.lags[index])
        if 0 < index < len(self.lags) - 1:
            left, centre, right = self.correlation[index - 1:index + 2]
            curvature = left - 2.0 * centre + right
            if np.isfinite([left, centre, right]).all() and curvature < 0:
                shift = 0.5 * (left - right) / curvature
                lag += float(np.clip(shift, -0.5, 0.5)) * float(self.lags[1] - self.lags[0])
        return round(lag, 4)

    @property
    def peak_lags(self) -> np.ndarray:
        return np.asarray([self.refined(int(index)) for index in self.peaks], dtype=np.float64)

    def best(self) -> Optional[float]:
        if not self.valid:
            return None
        return self.refined(int(np.nanargmax(self.correlation)))

    def at(self, lag: float) -> float:
        if not self.valid:
            return float("nan")
        return float(np.interp(lag, self.lags, self.correlation, left=np.nan, right=np.nan))

    def step(self, current: float, direction: int) -> Optional[float]:
        """Nearest local maximum strictly after (+1) or before (-1) *current*."""
        if not len(self.peaks):
            return None
        candidates = self.peak_lags
        tolerance = 1e-3 + 0.5 * (self.lags[1] - self.lags[0] if len(self.lags) > 1 else 0.0)
        if direction > 0:
            after = candidates[candidates > current + tolerance]
            return float(after.min()) if len(after) else None
        before = candidates[candidates < current - tolerance]
        return float(before.max()) if len(before) else None

    def peak_ratio(self, min_separation: float = 0.25) -> float:
        """Best correlation divided by the best other peak (> min_separation away)."""
        best = self.best()
        if best is None:
            return float("nan")
        best_value = float(np.nanmax(self.correlation))
        others = [
            self.correlation[index] for index in self.peaks
            if abs(self.lags[index] - best) > min_separation and np.isfinite(self.correlation[index])
        ]
        second = max(others) if others else float("nan")
        return best_value / second if np.isfinite(second) and second > 0 else float("inf")


def scan_lags(pulse_times: np.ndarray, pulse: np.ndarray,
              reference_times: np.ndarray, reference: np.ndarray,
              lag_min: float, lag_max: float, step: float = 0.02,
              window: Optional[tuple[float, float]] = None,
              min_coverage: float = 0.8) -> LagScan:
    """Pearson correlation of pulse(t) with reference(t - lag) for each lag."""
    pulse_times = np.asarray(pulse_times, dtype=np.float64)
    pulse = np.asarray(pulse, dtype=np.float64)
    keep = np.isfinite(pulse)
    if window is not None:
        keep &= (pulse_times >= window[0]) & (pulse_times <= window[1])
    times, values = pulse_times[keep], pulse[keep]
    lag_min, lag_max = sorted((float(lag_min), float(lag_max)))
    lags = np.arange(lag_min, lag_max + step / 2, step)
    correlation = np.full(len(lags), np.nan)
    if len(times) >= 8 and len(reference_times) >= 2:
        for index, lag in enumerate(lags):
            shifted = np.interp(times - lag, reference_times, reference, left=np.nan, right=np.nan)
            valid = np.isfinite(shifted)
            if valid.sum() < max(8, min_coverage * len(times)):
                continue
            a = values[valid] - values[valid].mean()
            b = shifted[valid] - shifted[valid].mean()
            denominator = np.sqrt((a * a).sum() * (b * b).sum())
            if denominator > 0:
                correlation[index] = float((a * b).sum() / denominator)
    return LagScan.from_curve(lags, correlation)


# ---------------------------------------------------------------------------
# Pooling windows and segments
# ---------------------------------------------------------------------------

@dataclass
class CorrelationSum:
    """Correlation curves of several analysis windows, averaged per lag.

    The pulse is nearly periodic, so a single window has a peak every heartbeat
    with almost the same height. Windows (and segments) at different heart
    rates put those neighbouring peaks at different offsets, while the true
    offset is common to all of them: averaging makes it stand out.
    """

    lags: np.ndarray
    total: np.ndarray       # sum of the accepted windows' correlation per lag
    count: np.ndarray       # accepted windows with a correlation at each lag
    accepted: int = 0
    windows: int = 0

    @classmethod
    def empty(cls, lags: np.ndarray) -> "CorrelationSum":
        return cls(lags=lags, total=np.zeros(len(lags)), count=np.zeros(len(lags), dtype=int))

    def add_curve(self, correlation: np.ndarray) -> None:
        finite = np.isfinite(correlation)
        self.total[finite] += correlation[finite]
        self.count[finite] += 1

    def add(self, other: "CorrelationSum") -> None:
        if len(other.lags) != len(self.lags) or not np.allclose(other.lags, self.lags):
            raise ValueError("Correlation sums computed on different offset grids")
        self.total += other.total
        self.count += other.count
        self.accepted += other.accepted
        self.windows += other.windows

    def scan(self) -> LagScan:
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(self.count > 0, self.total / np.maximum(self.count, 1), np.nan)
        return LagScan.from_curve(self.lags, mean)


@dataclass
class WindowCorrelation:
    """Correlation curve of one analysis window, with what tells whether to trust it."""

    start: float
    end: float
    face_coverage: float     # fraction of the window's frames where the face was found
    video_hr: float          # dominant rate of the rPPG (bpm)
    reference_hr: float      # dominant rate of the reference over the same time (bpm)
    correlation: np.ndarray  # per lag; NaN where the reference does not cover the window

    def reliable(self, max_hr_difference_bpm: Optional[float] = 3.0,
                 min_face_coverage: float = 0.9) -> bool:
        """Face found in most frames and, unless *max_hr_difference_bpm* is None,
        rPPG and reference heart rates within that difference."""
        if self.face_coverage < min_face_coverage or not np.isfinite(self.correlation).any():
            return False
        if max_hr_difference_bpm is None:
            return True
        return bool(np.isfinite(self.video_hr) and np.isfinite(self.reference_hr)
                    and abs(self.video_hr - self.reference_hr) <= max_hr_difference_bpm)


def window_correlations(pulse_times: np.ndarray, pulse: np.ndarray, fs: float,
                        reference_times: np.ndarray, reference: np.ndarray,
                        lag_min: float, lag_max: float, step: float = 0.02, *,
                        window_sec: float = 30.0, hop_sec: float = 15.0,
                        detected: Optional[np.ndarray] = None) -> tuple[np.ndarray, list[WindowCorrelation]]:
    """Offset grid and the correlation of every analysis window of one segment.

    The segment is cut into *window_sec* windows every *hop_sec*. Every window
    is scanned, so the reliability criteria (:meth:`WindowCorrelation.reliable`)
    can be changed afterwards without scanning again. The search range is a few
    seconds, much shorter than heart-rate changes, so the reference heart rate is
    measured over the same time without shifting it.
    """
    pulse_times = np.asarray(pulse_times, dtype=np.float64)
    pulse = np.asarray(pulse, dtype=np.float64)
    reference_times = np.asarray(reference_times, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    lag_min, lag_max = sorted((float(lag_min), float(lag_max)))
    lags = np.arange(lag_min, lag_max + step / 2, step)
    windows: list[WindowCorrelation] = []
    if len(pulse_times) < 2 or len(reference_times) < 2:
        return lags, windows
    reference_rate = 1.0 / float(np.median(np.diff(reference_times)))
    start, end = float(pulse_times[0]), float(pulse_times[-1]) + 1.0 / fs  # the last frame lasts 1/fs
    span = min(window_sec, end - start)
    for window_start in np.arange(start, end - span + 1e-6, hop_sec):
        window = (float(window_start), float(window_start + span))
        in_window = (pulse_times >= window[0]) & (pulse_times <= window[1])
        in_reference = (reference_times >= window[0]) & (reference_times <= window[1])
        scan = scan_lags(pulse_times, pulse, reference_times, reference, lag_min, lag_max,
                         step=step, window=window)
        windows.append(WindowCorrelation(
            start=window[0], end=window[1],
            face_coverage=float(np.mean(np.asarray(detected)[in_window])) if detected is not None else 1.0,
            video_hr=dominant_rate_bpm(pulse[in_window], fs),
            reference_hr=dominant_rate_bpm(reference[in_reference], reference_rate),
            correlation=scan.correlation,
        ))
    return lags, windows


def pool_windows(lags: np.ndarray, windows: list[WindowCorrelation], *,
                 max_hr_difference_bpm: Optional[float] = 3.0,
                 min_face_coverage: float = 0.9) -> CorrelationSum:
    """Average of the reliable windows (see :meth:`WindowCorrelation.reliable`)."""
    result = CorrelationSum.empty(lags)
    for window in windows:
        result.windows += 1
        if window.reliable(max_hr_difference_bpm, min_face_coverage):
            result.add_curve(window.correlation)
            result.accepted += 1
    return result


def windowed_correlation(pulse_times: np.ndarray, pulse: np.ndarray, fs: float,
                         reference_times: np.ndarray, reference: np.ndarray,
                         lag_min: float, lag_max: float, step: float = 0.02, *,
                         window_sec: float = 30.0, hop_sec: float = 15.0,
                         max_hr_difference_bpm: Optional[float] = 3.0,
                         detected: Optional[np.ndarray] = None,
                         min_face_coverage: float = 0.9) -> CorrelationSum:
    """Average correlation of the reliable analysis windows of one segment.

    A window is used only when the face was found in most of its frames and the
    heart rate of the rPPG matches that of the reference over the same time
    (they differ when motion or lighting dominate the rPPG).
    """
    lags, windows = window_correlations(pulse_times, pulse, fs, reference_times, reference,
                                        lag_min, lag_max, step, window_sec=window_sec,
                                        hop_sec=hop_sec, detected=detected)
    return pool_windows(lags, windows, max_hr_difference_bpm=max_hr_difference_bpm,
                        min_face_coverage=min_face_coverage)
