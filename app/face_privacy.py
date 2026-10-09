"""Face anonymisation for previews, captured frames, and exported videos.

Detection uses the OpenCV Zoo YuNet model. Compared with a single detection
pass on a small image, this module:

* detects on a higher-resolution image (small, distant faces are kept),
  optionally adding overlapping full-resolution tiles ("thorough" quality);
* uses a privacy-oriented (lower) confidence threshold;
* tracks faces over time in videos, so a face that is missed for a few
  frames (head turned, motion blur) stays masked instead of flickering;
* masks an expanded elliptical region with feathered edges, using a smooth
  irreversible blur by default (pixelation and solid fill are available).
"""
from __future__ import annotations

import logging
import math
import subprocess
import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

_FACE_MARGIN_RATIO = 0.30
_YUNET_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "face_detection_yunet_2023mar.onnx"
)
_detector_local = threading.local()

BLUR_STYLES = ("blur", "pixelate", "solid")
DETECTION_QUALITIES = ("fast", "balanced", "thorough")
_QUALITY_MAX_DIMENSION = {"fast": 960, "balanced": 1920, "thorough": 1920}
_TILE_SIZE = 1280
_TILE_OVERLAP = 0.25


@dataclass
class FacePrivacySettings:
    """User-tunable anonymisation parameters."""

    style: str = "blur"
    strength: int = 7
    detection_quality: str = "balanced"
    score_threshold: float = 0.35
    margin_ratio: float = _FACE_MARGIN_RATIO
    persistence_sec: float = 0.6

    def normalised(self) -> "FacePrivacySettings":
        return FacePrivacySettings(
            style=self.style if self.style in BLUR_STYLES else "blur",
            strength=int(min(10, max(1, self.strength))),
            detection_quality=(
                self.detection_quality
                if self.detection_quality in DETECTION_QUALITIES else "balanced"
            ),
            score_threshold=float(min(0.95, max(0.05, self.score_threshold))),
            margin_ratio=float(min(1.5, max(0.0, self.margin_ratio))),
            persistence_sec=float(min(5.0, max(0.0, self.persistence_sec))),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FacePrivacySettings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in names}).normalised()


_settings_lock = threading.Lock()
_current_settings = FacePrivacySettings()


def get_privacy_settings() -> FacePrivacySettings:
    with _settings_lock:
        return FacePrivacySettings(**asdict(_current_settings))


def set_privacy_settings(settings: FacePrivacySettings) -> None:
    global _current_settings
    with _settings_lock:
        _current_settings = settings.normalised()


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _yunet_detector() -> cv2.FaceDetectorYN:
    detector = getattr(_detector_local, "yunet", None)
    if detector is None:
        if not _YUNET_MODEL_PATH.exists():
            raise RuntimeError(f"Face privacy model not found: {_YUNET_MODEL_PATH}")
        detector = cv2.FaceDetectorYN.create(
            str(_YUNET_MODEL_PATH),
            "",
            (320, 320),
            0.5,
            0.3,
            5000,
            cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU,
        )
        _detector_local.yunet = detector
    return detector


def _detect_on_image(
    image: np.ndarray, score_threshold: float
) -> list[tuple[float, float, float, float, float]]:
    if image.size == 0 or image.shape[0] < 16 or image.shape[1] < 16:
        return []
    detector = _yunet_detector()
    detector.setScoreThreshold(float(score_threshold))
    detector.setInputSize((image.shape[1], image.shape[0]))
    _status, faces = detector.detect(image)
    if faces is None:
        return []
    return [
        (float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[14]))
        for f in faces
    ]


def _detect_scaled(
    frame: np.ndarray, max_dimension: int, score_threshold: float,
    offset: tuple[int, int] = (0, 0),
) -> list[tuple[float, float, float, float, float]]:
    height, width = frame.shape[:2]
    scale = min(1.0, max_dimension / max(width, height))
    if scale < 1.0:
        sample = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        sample = frame
    inverse = 1.0 / scale
    ox, oy = offset
    return [
        (x * inverse + ox, y * inverse + oy, w * inverse, h * inverse, score)
        for x, y, w, h, score in _detect_on_image(sample, score_threshold)
    ]


def _tile_origins(length: int, tile: int) -> list[int]:
    if length <= tile:
        return [0]
    step = max(1, int(tile * (1.0 - _TILE_OVERLAP)))
    origins = list(range(0, length - tile, step))
    origins.append(length - tile)
    return sorted(set(origins))


def _non_max_suppression(
    boxes: list[tuple[float, float, float, float, float]], iou_threshold: float = 0.3
) -> list[tuple[float, float, float, float, float]]:
    if len(boxes) <= 1:
        return boxes
    rects = [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in boxes]
    scores = [b[4] for b in boxes]
    keep = cv2.dnn.NMSBoxes(rects, scores, 0.0, iou_threshold)
    return [boxes[int(i)] for i in np.array(keep).flatten()]


def detect_faces_scored(
    frame: np.ndarray, settings: Optional[FacePrivacySettings] = None
) -> list[tuple[float, float, float, float, float]]:
    """Return ``(x, y, w, h, score)`` face boxes in frame coordinates."""
    settings = (settings or get_privacy_settings()).normalised()
    if frame is None or frame.size == 0:
        return []
    max_dimension = _QUALITY_MAX_DIMENSION[settings.detection_quality]
    boxes = _detect_scaled(frame, max_dimension, settings.score_threshold)
    if settings.detection_quality == "thorough":
        height, width = frame.shape[:2]
        if max(width, height) > _TILE_SIZE:
            for y0 in _tile_origins(height, _TILE_SIZE):
                for x0 in _tile_origins(width, _TILE_SIZE):
                    tile = frame[y0:y0 + _TILE_SIZE, x0:x0 + _TILE_SIZE]
                    boxes.extend(_detect_scaled(
                        tile, _TILE_SIZE, settings.score_threshold, offset=(x0, y0)
                    ))
        boxes = _non_max_suppression(boxes)
    return boxes


def _detect_faces(
    frame: np.ndarray, settings: Optional[FacePrivacySettings] = None
) -> list[tuple[int, int, int, int]]:
    return [
        (int(round(x)), int(round(y)), int(round(w)), int(round(h)))
        for x, y, w, h, _score in detect_faces_scored(frame, settings)
    ]


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def _expanded_region(
    face: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
    margin_ratio: float = _FACE_MARGIN_RATIO,
) -> tuple[int, int, int, int]:
    """Return ``(x0, y0, x1, y1)`` of the face box grown by *margin_ratio*.

    Extra room is added above the box so hair and forehead are covered.
    """
    x, y, width, height = face[:4]
    margin_x = width * margin_ratio
    margin_top = height * margin_ratio * 1.3
    margin_bottom = height * margin_ratio
    return (
        max(0, int(round(x - margin_x))),
        max(0, int(round(y - margin_top))),
        min(frame_width, int(round(x + width + margin_x))),
        min(frame_height, int(round(y + height + margin_bottom))),
    )


def _ellipse_mask(width: int, height: int, feather: bool = True) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    cv2.ellipse(
        mask,
        (width // 2, height // 2),
        (max(1, int(width * 0.47)), max(1, int(height * 0.47))),
        0, 0, 360, 1.0, -1,
    )
    if feather:
        sigma = max(1.0, min(width, height) * 0.05)
        mask = cv2.GaussianBlur(mask, (0, 0), sigma)
        # Keep the core fully opaque; feathering only softens the outer edge.
        inner = np.zeros_like(mask)
        cv2.ellipse(
            inner,
            (width // 2, height // 2),
            (max(1, int(width * 0.40)), max(1, int(height * 0.40))),
            0, 0, 360, 1.0, -1,
        )
        mask = np.maximum(mask, inner)
    return mask[..., None]


def _masked_region(region: np.ndarray, style: str, strength: int) -> np.ndarray:
    """Return an anonymised replacement for *region* (same shape)."""
    height, width = region.shape[:2]
    if style == "solid":
        return np.full_like(region, 48)
    if style == "pixelate":
        blocks = max(3, 15 - strength)
        small = cv2.resize(
            region,
            (max(1, min(blocks, width)), max(1, min(round(blocks * height / max(width, 1)), height))),
            interpolation=cv2.INTER_AREA,
        )
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    cells = max(2, 12 - strength)
    small = cv2.resize(
        region,
        (max(1, min(cells, width)), max(1, min(round(cells * height / max(width, 1)), height))),
        interpolation=cv2.INTER_AREA,
    )
    smooth = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    sigma = max(1.0, min(width, height) / 6.0)
    return cv2.GaussianBlur(smooth, (0, 0), sigma)


def apply_privacy_masks(
    frame: np.ndarray,
    regions: list[tuple[int, int, int, int]],
    settings: Optional[FacePrivacySettings] = None,
) -> np.ndarray:
    """Return a copy of *frame* with every ``(x0, y0, x1, y1)`` region masked."""
    settings = (settings or get_privacy_settings()).normalised()
    result = frame.copy()
    for x0, y0, x1, y1 in regions:
        roi = result[y0:y1, x0:x1]
        if roi.size == 0 or roi.shape[0] < 2 or roi.shape[1] < 2:
            continue
        replacement = _masked_region(roi, settings.style, settings.strength)
        if settings.style == "pixelate":
            result[y0:y1, x0:x1] = replacement
            continue
        mask = _ellipse_mask(roi.shape[1], roi.shape[0])
        blended = roi.astype(np.float32) * (1.0 - mask) + replacement.astype(np.float32) * mask
        result[y0:y1, x0:x1] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    return result


def anonymize_faces(
    frame: np.ndarray, settings: Optional[FacePrivacySettings] = None
) -> np.ndarray:
    """Return a copy of a BGR frame with detected faces anonymised."""
    if frame.size == 0:
        return frame.copy()
    settings = (settings or get_privacy_settings()).normalised()
    height, width = frame.shape[:2]
    regions = [
        _expanded_region(face, width, height, settings.margin_ratio)
        for face in _detect_faces(frame, settings)
    ]
    return apply_privacy_masks(frame, regions, settings)


# ---------------------------------------------------------------------------
# Temporal tracking for video
# ---------------------------------------------------------------------------

def _iou(a, b) -> float:
    ax0, ay0, aw, ah = a[:4]
    bx0, by0, bw, bh = b[:4]
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class _Track:
    __slots__ = ("box", "last_seen", "velocity")

    def __init__(self, box, t):
        self.box = tuple(float(v) for v in box[:4])
        self.last_seen = t
        self.velocity = (0.0, 0.0)


class FaceAnonymizer:
    """Stateful anonymiser for consecutive frames of one video.

    Faces stay masked for ``persistence_sec`` after their last detection; the
    mask follows the last observed motion and grows while the face is unseen.
    """

    def __init__(self, settings: Optional[FacePrivacySettings] = None, fps: float = 30.0):
        self.settings = (settings or get_privacy_settings()).normalised()
        self.fps = fps if fps and fps > 0 else 30.0
        self._tracks: list[_Track] = []
        self._frame_index = 0

    def reset(self) -> None:
        self._tracks.clear()

    def update(self, frame: np.ndarray, t_sec: Optional[float] = None) -> list[tuple[float, float, float, float]]:
        """Detect faces in *frame* and return the boxes that must be masked."""
        if t_sec is None:
            t_sec = self._frame_index / self.fps
        self._frame_index += 1
        detections = detect_faces_scored(frame, self.settings)
        return self.update_with_detections(detections, t_sec)

    def update_with_detections(self, detections, t_sec: float) -> list[tuple[float, float, float, float]]:
        persistence = self.settings.persistence_sec
        unmatched = list(range(len(self._tracks)))
        for det in detections:
            best, best_score = None, 0.0
            dcx, dcy = det[0] + det[2] / 2, det[1] + det[3] / 2
            for idx in unmatched:
                track = self._tracks[idx]
                score = _iou(track.box, det)
                if score <= 0.0:
                    tcx = track.box[0] + track.box[2] / 2
                    tcy = track.box[1] + track.box[3] / 2
                    distance = math.hypot(dcx - tcx, dcy - tcy)
                    if distance < 0.75 * max(track.box[2], track.box[3], det[2], det[3]):
                        score = 0.01
                if score > best_score:
                    best, best_score = idx, score
            if best is None:
                self._tracks.append(_Track(det, t_sec))
                continue
            track = self._tracks[best]
            dt = max(1e-3, t_sec - track.last_seen)
            old_cx = track.box[0] + track.box[2] / 2
            old_cy = track.box[1] + track.box[3] / 2
            vx, vy = (dcx - old_cx) / dt, (dcy - old_cy) / dt
            track.velocity = (0.5 * track.velocity[0] + 0.5 * vx, 0.5 * track.velocity[1] + 0.5 * vy)
            track.box = tuple(float(v) for v in det[:4])
            track.last_seen = t_sec
            unmatched.remove(best)

        self._tracks = [
            track for track in self._tracks if t_sec - track.last_seen <= persistence
        ]
        boxes = []
        for track in self._tracks:
            age = max(0.0, t_sec - track.last_seen)
            x, y, w, h = track.box
            if age > 0:
                x += track.velocity[0] * age
                y += track.velocity[1] * age
                grow = 1.0 + (0.5 * age / persistence if persistence > 0 else 0.0)
                cx, cy = x + w / 2, y + h / 2
                w, h = w * grow, h * grow
                x, y = cx - w / 2, cy - h / 2
            boxes.append((x, y, w, h))
        return boxes

    def regions(self, frame: np.ndarray, t_sec: Optional[float] = None) -> list[tuple[int, int, int, int]]:
        height, width = frame.shape[:2]
        return [
            _expanded_region(box, width, height, self.settings.margin_ratio)
            for box in self.update(frame, t_sec)
        ]

    def process(self, frame: np.ndarray, t_sec: Optional[float] = None) -> np.ndarray:
        if frame.size == 0:
            return frame.copy()
        return apply_privacy_masks(frame, self.regions(frame, t_sec), self.settings)


# ---------------------------------------------------------------------------
# Video export
# ---------------------------------------------------------------------------

def export_anonymized_video_segment(
    source_path: str,
    output_path: str,
    start_sec: float,
    duration_sec: float,
    ffmpeg: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> None:
    """Encode an anonymised video segment while copying its source audio."""
    capture = cv2.VideoCapture(source_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid video dimensions: {source_path}")

    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_sec) * 1000.0)
    frame_limit = max(0, int(round(duration_sec * fps)))
    command = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-ss", str(max(0.0, start_sec)),
        "-t", str(max(0.0, duration_sec)),
        "-i", source_path,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-shortest",
        output_path,
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    stderr_chunks: list[bytes] = []
    drain = threading.Thread(
        target=lambda: stderr_chunks.append(process.stderr.read() if process.stderr else b""),
        daemon=True,
    )
    drain.start()
    anonymizer = FaceAnonymizer(fps=fps)
    cancelled = False
    try:
        for index in range(frame_limit):
            if cancel_check is not None and cancel_check():
                cancelled = True
                break
            ok, frame = capture.read()
            if not ok:
                break
            process.stdin.write(anonymizer.process(frame, index / fps).tobytes())  # type: ignore[union-attr]
            if progress_callback is not None and (index % 15 == 0 or index == frame_limit - 1):
                progress_callback(index + 1, frame_limit)
    except BrokenPipeError:
        pass
    finally:
        capture.release()
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if cancelled:
            process.kill()
        process.wait()
        drain.join(timeout=5)

    if cancelled:
        Path(output_path).unlink(missing_ok=True)
        raise RuntimeError("Cancelled by user")
    if process.returncode != 0:
        stderr = b"".join(stderr_chunks).decode(errors="replace")
        raise RuntimeError(f"FFmpeg anonymized export failed:\n{stderr}")
