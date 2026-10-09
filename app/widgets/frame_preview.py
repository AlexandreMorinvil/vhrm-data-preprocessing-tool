"""Synchronized multi-camera video player.

Playback uses Qt Multimedia (FFmpeg backend, hardware decoding, GPU
rendering) with one ``QMediaPlayer`` per camera. One camera is the master
clock (the one providing audio); the others are kept in sync by small playback
rate corrections, or a re-seek when they drift too far. Seeking is
frame-accurate.

When Qt Multimedia is unavailable (for example PyQt6 < 6.8 on Windows, whose
wheels do not ship the FFmpeg libraries), the player falls back to still-frame
previews decoded with OpenCV, as in earlier versions of the application.
"""
from __future__ import annotations

import logging
import math
import os
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QEvent, QPoint, QRect, QSettings, QSize, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QGuiApplication, QImage, QKeySequence, QPainter, QPixmap, QRasterWindow, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QSizePolicy,
    QSlider,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..face_privacy import (
    FaceAnonymizer,
    _expanded_region,
    _masked_region,
    anonymize_faces,
    apply_privacy_masks,
    get_privacy_settings,
)
from ..timefmt import clock_at, format_clock, format_hms_ms, parse_time_text

try:  # Qt Multimedia is optional at import time (see module docstring).
    from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoFrame
    from PyQt6.QtMultimediaWidgets import QVideoWidget
    _QT_MULTIMEDIA_IMPORTED = True
except ImportError:  # pragma: no cover - depends on the installed wheel
    _QT_MULTIMEDIA_IMPORTED = False

log = logging.getLogger(__name__)

_SETTINGS_ORG = "VideoResearchTool"
_SETTINGS_APP = "VRT"
_HEADER_HEIGHT = 22
_TICK_MS = 33
_SYNC_EVERY_TICKS = 8
_HARD_RESYNC_MS = 250
_SOFT_RESYNC_MS = 25
_PRIVACY_INTERVAL_MS = 120
SPEEDS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
LAYOUTS = ("auto", "row", "grid", "focus")

_BACKEND_AVAILABLE: Optional[bool] = None


def multimedia_backend_available() -> bool:
    """Return True when Qt Multimedia can play video on this installation."""
    global _BACKEND_AVAILABLE
    if _BACKEND_AVAILABLE is None:
        if not _QT_MULTIMEDIA_IMPORTED or os.environ.get("VRT_DISABLE_QT_MULTIMEDIA"):
            _BACKEND_AVAILABLE = False
        else:
            probe = QMediaPlayer()
            _BACKEND_AVAILABLE = bool(probe.isAvailable())
            probe.deleteLater()
            if not _BACKEND_AVAILABLE:
                log.warning(
                    "Qt Multimedia backend unavailable; falling back to OpenCV still "
                    "previews. Install PyQt6>=6.8 for smooth playback with audio."
                )
    return _BACKEND_AVAILABLE


def frame_to_ms(frame_no: int, fps: float) -> int:
    """Position (ms) that displays *frame_no* (first ms at or after its start)."""
    if fps <= 0:
        return 0
    return int(math.ceil(frame_no * 1000.0 / fps - 1e-6))


def ms_to_frame(ms: float, fps: float) -> int:
    if fps <= 0:
        return 0
    return int(math.floor(ms * fps / 1000.0 + 1e-6))


def _settings() -> QSettings:
    return QSettings(_SETTINGS_ORG, _SETTINGS_APP)


def qimage_to_bgr(image: QImage) -> np.ndarray:
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    width, height = image.width(), image.height()
    stride = image.bytesPerLine()
    ptr = image.constBits()
    ptr.setsize(stride * height)
    array = np.frombuffer(ptr, dtype=np.uint8).reshape(height, stride)[:, : width * 3]
    return cv2.cvtColor(array.reshape(height, width, 3), cv2.COLOR_RGB2BGR)


def bgr_to_qimage(array: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(np.ascontiguousarray(array), cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()


def _probe_with_cv2(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return {}
        return {
            "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Privacy preview (detection on a worker thread, native mask widgets on top)
# ---------------------------------------------------------------------------

class _MaskWindow(QRasterWindow):
    """Privacy mask drawn as a child window of Qt's GPU video window.

    Being a child of the native video window, it is composited above the
    video surface and follows it (including into the pop-out window).
    """

    def __init__(self, parent_window):
        super().__init__(parent_window)
        self.setFlag(Qt.WindowType.WindowTransparentForInput, True)
        self._pixmap: Optional[QPixmap] = None

    def set_patch(self, image: Optional[QImage]) -> None:
        self._pixmap = QPixmap.fromImage(image) if image is not None else None
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        rect = QRect(0, 0, self.width(), self.height())
        if self._pixmap is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawPixmap(rect, self._pixmap)
        else:
            painter.fillRect(rect, QColor(48, 48, 48))
        painter.end()


class _PrivacyWorker(QThread):
    """Detect faces on sampled frames and produce masked patches."""

    regions_ready = pyqtSignal(int, int, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = threading.Condition()
        self._jobs: dict[int, tuple] = {}
        self._stop = False
        self._anonymizers: dict[int, FaceAnonymizer] = {}
        self._last_t: dict[int, float] = {}

    def submit(self, camera: int, frame, t_sec: float, generation: int, playing: bool) -> None:
        with self._lock:
            self._jobs[camera] = (frame, t_sec, generation, playing)
            self._lock.notify()

    def busy_with(self, camera: int) -> bool:
        with self._lock:
            return camera in self._jobs

    def reset_tracking(self) -> None:
        with self._lock:
            self._anonymizers.clear()
            self._last_t.clear()

    def stop(self) -> None:
        with self._lock:
            self._stop = True
            self._lock.notify()
        self.wait(3000)

    def run(self) -> None:
        while True:
            with self._lock:
                while not self._jobs and not self._stop:
                    self._lock.wait()
                if self._stop:
                    return
                camera = next(iter(self._jobs))
                frame, t_sec, generation, playing = self._jobs.pop(camera)
            try:
                payload = self._process(camera, frame, t_sec, playing)
            except Exception:  # pragma: no cover - defensive, keeps the thread alive
                log.exception("Privacy preview detection failed")
                continue
            if payload is not None:
                self.regions_ready.emit(camera, generation, payload)

    def _process(self, camera: int, frame, t_sec: float, playing: bool):
        image = frame.toImage() if not isinstance(frame, QImage) else frame
        if image.isNull():
            return None
        settings = get_privacy_settings()
        bgr = qimage_to_bgr(image)
        height, width = bgr.shape[:2]
        if settings.detection_quality != "thorough" and max(width, height) > 1920:
            scale = 1920 / max(width, height)
            bgr = cv2.resize(bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            height, width = bgr.shape[:2]
        anonymizer = self._anonymizers.get(camera)
        if anonymizer is None or anonymizer.settings != settings.normalised():
            anonymizer = FaceAnonymizer(settings)
            self._anonymizers[camera] = anonymizer
        last = self._last_t.get(camera)
        if last is not None and (t_sec < last - 0.05 or t_sec - last > 1.5):
            anonymizer.reset()
        self._last_t[camera] = t_sec
        margin = settings.margin_ratio + (0.25 if playing else 0.0)
        regions = [
            _expanded_region(box, width, height, margin)
            for box in anonymizer.update(bgr, t_sec)
        ]
        if playing:
            # The patch is shown for a few frames while the video moves on, so blur
            # the whole box: no sharp, stale surroundings around the face.
            patches = [
                bgr_to_qimage(_masked_region(bgr[y0:y1, x0:x1], settings.style, settings.strength))
                for x0, y0, x1, y1 in regions
            ]
        else:
            masked = apply_privacy_masks(bgr, regions, settings)
            patches = [bgr_to_qimage(masked[y0:y1, x0:x1]) for x0, y0, x1, y1 in regions]
        normalised = [
            (x0 / width, y0 / height, x1 / width, y1 / height) for x0, y0, x1, y1 in regions
        ]
        return normalised, patches


# ---------------------------------------------------------------------------
# One camera
# ---------------------------------------------------------------------------

class FramePreview(QFrame):
    """One camera tile: title bar plus a GPU video surface (or OpenCV still)."""

    clicked = pyqtSignal()
    focus_requested = pyqtSignal()
    audio_requested = pyqtSignal()

    def __init__(self, label: str = "Camera", face_blur_enabled: bool = False,
                 use_qt: Optional[bool] = None, parent=None):
        super().__init__(parent)
        self._use_qt = multimedia_backend_available() if use_qt is None else use_qt
        self._label_text = label
        self._video_path = ""
        self._fps = 30.0
        self._frame_count = 0
        self._frame_size = QSize(0, 0)
        self._position_ms = 0
        self._face_blur_enabled = face_blur_enabled
        self._active = False
        self._player = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._still: Optional[QPixmap] = None
        self._masks: list[_MaskWindow] = []
        self._mask_parent = None
        self._mask_rects: list[tuple[float, float, float, float]] = []
        self._is_audio_source = False

        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)

        header = QHBoxLayout()
        header.setContentsMargins(4, 0, 4, 0)
        self._label = QLabel(label)
        self._label.setStyleSheet("font-weight: bold;")
        header.addWidget(self._label)
        header.addStretch()
        self._info_label = QLabel("No video loaded")
        self._info_label.setStyleSheet("color: #8a8a8a; font-size: 11px;")
        header.addWidget(self._info_label)
        header_widget = QWidget()
        header_widget.setFixedHeight(_HEADER_HEIGHT)
        header_widget.setLayout(header)
        layout.addWidget(header_widget)

        if self._use_qt:
            self._display = QVideoWidget()
            self._display.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        else:
            self._display = QLabel()
            self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setMinimumSize(QSize(120, 120))
        self._display.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._display.setStyleSheet("background: #111;")
        self._display.installEventFilter(self)
        layout.addWidget(self._display, 1)

    # -- compatibility properties -------------------------------------------------
    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def current_frame(self) -> int:
        return ms_to_frame(self._position_ms, self._fps)

    @property
    def video_path(self) -> str:
        return self._video_path

    @property
    def face_blur_enabled(self) -> bool:
        return self._face_blur_enabled

    @property
    def label_text(self) -> str:
        return self._label_text

    @property
    def duration_ms(self) -> int:
        if self._player is not None and self._player.duration() > 0:
            return int(self._player.duration())
        return int(self._frame_count * 1000.0 / self._fps) if self._fps > 0 else 0

    @property
    def aspect_ratio(self) -> float:
        if self._frame_size.width() > 0 and self._frame_size.height() > 0:
            return self._frame_size.width() / self._frame_size.height()
        return 16 / 9

    @property
    def media_player(self):
        return self._player

    @property
    def display_widget(self) -> QWidget:
        return self._display

    @property
    def is_active(self) -> bool:
        return self._active

    # -- loading -------------------------------------------------------------------
    def load_video(self, path: str) -> bool:
        self.release()
        if not path or not Path(path).exists():
            self._info_label.setText("Missing video")
            return False
        info = _probe_with_cv2(path)
        if not info:
            log.warning("Cannot open video: %s", path)
            self._info_label.setText("Cannot open video")
            return False
        self._video_path = path
        self._fps = info["fps"]
        self._frame_count = info["frame_count"]
        self._frame_size = QSize(info["width"], info["height"])
        self._position_ms = 0
        self._info_label.setText(f"{self._fps:.2f} fps")
        self.setToolTip(
            f"{Path(path).name}\n{self._frame_count} frames | {self._fps:.3f} fps | "
            f"{info['width']}x{info['height']}"
        )
        return True

    def release(self) -> None:
        self.deactivate()
        self._video_path = ""
        self._frame_count = 0
        self._position_ms = 0
        self._still = None
        self._drop_masks()

    def activate(self) -> None:
        """Open decoders for the loaded file (only while visible)."""
        if self._active or not self._video_path:
            return
        self._active = True
        if self._use_qt:
            if self._player is None:
                self._player = QMediaPlayer(self)
                self._player.setVideoOutput(self._display)
                self._player.errorOccurred.connect(self._on_player_error)
                self._player.mediaStatusChanged.connect(self._on_media_status)
            self._player.setSource(QUrl.fromLocalFile(self._video_path))
            self._player.pause()
        else:
            self._cap = cv2.VideoCapture(self._video_path)
            self._render_still()

    def deactivate(self) -> None:
        if not self._active:
            return
        self._active = False
        self._drop_masks()
        if self._player is not None:
            self._player.stop()
            self._player.setSource(QUrl())
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _on_media_status(self, status) -> None:
        if status == QMediaPlayer.MediaStatus.LoadedMedia and self._player is not None:
            self._player.setPosition(int(self._position_ms))
            if self._player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                self._player.pause()

    def _on_player_error(self, _error, message: str) -> None:
        log.warning("Playback error for %s: %s", self._video_path, message)
        self._info_label.setText("Playback error")

    # -- positioning ---------------------------------------------------------------
    def set_position_ms(self, ms: float) -> None:
        self._position_ms = max(0, int(ms))
        if not self._active:
            return
        if self._player is not None:
            self._player.setPosition(self._position_ms)
        else:
            self._render_still()

    def note_position_ms(self, ms: float) -> None:
        """Record the playhead without seeking (used while playing)."""
        self._position_ms = max(0, int(ms))

    def seek_frame(self, frame_no: int) -> None:
        if self._frame_count > 0:
            frame_no = max(0, min(frame_no, self._frame_count - 1))
        self.set_position_ms(frame_to_ms(frame_no, self._fps))

    def seek_normalised(self, ratio: float) -> None:
        if self._frame_count > 0:
            self.seek_frame(int(ratio * (self._frame_count - 1)))

    def set_audio_source(self, enabled: bool) -> None:
        self._is_audio_source = enabled
        self._label.setText(f"{self._label_text}  \U0001F50A" if enabled else self._label_text)

    def set_label(self, text: str) -> None:
        self._label_text = text
        self.set_audio_source(self._is_audio_source)

    # -- privacy -------------------------------------------------------------------
    def set_face_blur_enabled(self, enabled: bool) -> None:
        if self._face_blur_enabled == enabled:
            return
        self._face_blur_enabled = enabled
        if not enabled:
            self.clear_masks()
        if not self._use_qt:
            self._render_still()

    def clear_masks(self) -> None:
        self._mask_rects = []
        for mask in self._masks:
            mask.hide()

    def _drop_masks(self) -> None:
        for mask in self._masks:
            mask.hide()
            mask.destroy()
            mask.deleteLater()
        self._masks = []
        self._mask_parent = None

    def _video_window(self):
        """Return Qt's internal native video window for this tile."""
        if not self._use_qt:
            return None
        container = next(
            (w for w in self._display.findChildren(QWidget)
             if w.metaObject().className() == "QWindowContainer"),
            None,
        )
        top = self._display.window().windowHandle()
        if container is None or top is None:
            return None
        origin = container.mapTo(container.window(), QPoint(0, 0))
        target = QRect(origin, container.size())
        candidates = [
            w for w in QGuiApplication.allWindows()
            if w.metaObject().className() == "QVideoWindow" and w.parent() == top
        ]
        for window in candidates:
            if window.geometry() == target:
                return window
        return None

    def set_privacy_masks(self, rects, patches) -> None:
        if not self._face_blur_enabled:
            self.clear_masks()
            return
        parent = self._video_window()
        if parent is None:
            return
        if parent is not self._mask_parent:
            self._drop_masks()
            self._mask_parent = parent
        self._mask_rects = list(rects)
        while len(self._masks) < len(rects):
            self._masks.append(_MaskWindow(parent))
        for index, mask in enumerate(self._masks):
            if index < len(rects):
                mask.set_patch(patches[index] if index < len(patches) else None)
            else:
                mask.hide()
        self._layout_masks()

    def _video_rect(self) -> QRect:
        area = self._display.rect()
        aspect = self.aspect_ratio
        if area.width() <= 0 or area.height() <= 0:
            return area
        if area.width() / area.height() > aspect:
            width = int(area.height() * aspect)
            return QRect((area.width() - width) // 2, 0, width, area.height())
        height = int(area.width() / aspect)
        return QRect(0, (area.height() - height) // 2, area.width(), height)

    def _layout_masks(self) -> None:
        rect = self._video_rect()
        for index, mask in enumerate(self._masks):
            if index >= len(self._mask_rects):
                mask.hide()
                continue
            x0, y0, x1, y1 = self._mask_rects[index]
            mask.setGeometry(
                rect.x() + int(x0 * rect.width()),
                rect.y() + int(y0 * rect.height()),
                max(2, int((x1 - x0) * rect.width())),
                max(2, int((y1 - y0) * rect.height())),
            )
            mask.show()
            mask.raise_()

    # -- OpenCV fallback rendering -------------------------------------------------
    def read_exact_frame(self, frame_no: Optional[int] = None) -> Optional[np.ndarray]:
        """Decode one exact frame with OpenCV (used for captures and fallback)."""
        if not self._video_path:
            return None
        frame_no = self.current_frame if frame_no is None else frame_no
        cap = cv2.VideoCapture(self._video_path)
        try:
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(frame_no, max(0, self._frame_count - 1))))
            ok, frame = cap.read()
        finally:
            cap.release()
        return frame if ok else None

    def _render_still(self) -> None:
        if self._use_qt or self._cap is None:
            return
        frame_no = max(0, min(self.current_frame, max(0, self._frame_count - 1)))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = self._cap.read()
        if not ok:
            return
        if self._face_blur_enabled:
            frame = anonymize_faces(frame)
        self._still = QPixmap.fromImage(bgr_to_qimage(frame))
        self._scale_still()

    def _scale_still(self) -> None:
        if self._still is None or self._use_qt:
            return
        self._display.setPixmap(self._still.scaled(
            self._display.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    # -- events --------------------------------------------------------------------
    def eventFilter(self, obj, event):
        if obj is self._display:
            if event.type() == QEvent.Type.Resize:
                self._layout_masks()
                self._scale_still()
            elif event.type() == QEvent.Type.MouseButtonDblClick:
                self.focus_requested.emit()
                return True
            elif event.type() == QEvent.Type.MouseButtonPress:
                self.clicked.emit()
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.focus_requested.emit()
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        audio = menu.addAction("Play audio from this camera")
        focus = menu.addAction("Focus / unfocus this camera")
        chosen = menu.exec(event.globalPos())
        if chosen is audio:
            self.audio_requested.emit()
        elif chosen is focus:
            self.focus_requested.emit()

    def sizeHint(self) -> QSize:
        return QSize(320, 480)


# ---------------------------------------------------------------------------
# Pop-out window
# ---------------------------------------------------------------------------

class _PopoutWindow(QWidget):
    closed = pyqtSignal()

    def __init__(self, title: str):
        super().__init__(None, Qt.WindowType.Window)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        geometry = _settings().value("player/popout_geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            self.resize(1200, 800)

    def closeEvent(self, event):
        _settings().setValue("player/popout_geometry", self.saveGeometry())
        self.closed.emit()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Multi-camera player
# ---------------------------------------------------------------------------

class MultiCameraPlayer(QWidget):
    """Grid of synchronized camera views with a shared transport bar."""

    frame_changed = pyqtSignal(int)
    position_changed = pyqtSignal(float)
    playing_changed = pyqtSignal(bool)
    export_frames_requested = pyqtSignal()
    snapshot_requested = pyqtSignal()

    def __init__(self, face_blur_enabled: bool = False, parent=None):
        super().__init__(parent)
        self._use_qt = multimedia_backend_available()
        self._previews: list[FramePreview] = []
        self._playing = False
        self._face_blur_enabled = face_blur_enabled
        self._total_frames = 0
        self._position_ms = 0.0
        self._pending_seek_ms: Optional[float] = None
        self._seek_cooldown = QTimer(self)
        self._seek_cooldown.setSingleShot(True)
        self._seek_cooldown.setInterval(40)
        self._seek_cooldown.timeout.connect(self._flush_pending_seek)
        self._tick = QTimer(self)
        self._tick.setInterval(_TICK_MS)
        self._tick.timeout.connect(self._on_tick)
        self._tick_count = 0
        self._rate = 1.0
        self._clock_anchor = None
        self._layout_mode = str(_settings().value("player/layout", "auto"))
        self._focus_index = 0
        self._columns = 0
        self._visible_media = False
        self._want_popout = False
        self._popout: Optional[_PopoutWindow] = None
        self._shortcut_hosts: list[QWidget] = []
        self._audio_index = int(_settings().value("player/audio_camera", 0, type=int))
        self._audio_output = None
        self._privacy_worker: Optional[_PrivacyWorker] = None
        self._privacy_generation = 0
        self._latest_frames: dict[int, object] = {}
        self._privacy_timer = QTimer(self)
        self._privacy_timer.setInterval(_PRIVACY_INTERVAL_MS)
        self._privacy_timer.timeout.connect(self._submit_privacy_frames)
        self._privacy_debounce = QTimer(self)
        self._privacy_debounce.setSingleShot(True)
        self._privacy_debounce.setInterval(60)
        self._privacy_debounce.timeout.connect(self._submit_privacy_frames)
        self._sink_slots: list[tuple[object, object]] = []
        if self._use_qt:
            self._audio_output = QAudioOutput(self)
            self._audio_output.setVolume(float(_settings().value("player/volume", 0.8, type=float)))
            self._audio_output.setMuted(bool(_settings().value("player/muted", False, type=bool)))

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        self._video_slot = QVBoxLayout()
        self._video_slot.setContentsMargins(0, 0, 0, 0)
        root.addLayout(self._video_slot, 1)
        self._video_area = QWidget()
        self._video_area.setMinimumHeight(140)
        self._grid = QGridLayout(self._video_area)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(4)
        self._video_area.installEventFilter(self)
        self._video_slot.addWidget(self._video_area)
        self._placeholder = QLabel(
            "Cameras are shown in a separate window.\nClick “Dock” to bring them back."
        )
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #8a8a8a;")
        self._placeholder.hide()
        self._video_slot.addWidget(self._placeholder)
        self._empty_label = QLabel("No camera video loaded")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #8a8a8a;")
        self._grid.addWidget(self._empty_label, 0, 0)
        if not self._use_qt:
            self._empty_label.setText(
                "No camera video loaded\n(Install PyQt6 ≥ 6.8 for smooth playback with audio.)"
            )

        root.addWidget(self._build_transport())
        self._update_label()

    # ------------------------------------------------------------------ UI
    def _tool(self, icon: QStyle.StandardPixmap | None, text: str, tip: str, slot) -> QToolButton:
        button = QToolButton()
        if icon is not None:
            button.setIcon(self.style().standardIcon(icon))
        else:
            button.setText(text)
        button.setToolTip(tip)
        button.setAutoRaise(True)
        button.clicked.connect(slot)
        return button

    def _build_transport(self) -> QWidget:
        bar = QWidget()
        outer = QVBoxLayout(bar)
        outer.setContentsMargins(4, 0, 4, 2)
        outer.setSpacing(0)

        slider_row = QHBoxLayout()
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)
        self._slider.setToolTip("Scrub all cameras")
        slider_row.addWidget(self._slider, 1)
        outer.addLayout(slider_row)

        row = QHBoxLayout()
        row.setSpacing(2)
        sp = QStyle.StandardPixmap
        row.addWidget(self._tool(sp.SP_MediaSkipBackward, "", "Go to start (Home)", lambda: self.seek_seconds(0.0)))
        row.addWidget(self._tool(sp.SP_MediaSeekBackward, "", "Back 1 s (Shift+Left)", lambda: self.step_seconds(-1.0)))
        row.addWidget(self._tool(None, "◀|", "Previous frame (Left)", lambda: self.step_frames(-1)))
        self._play_btn = self._tool(sp.SP_MediaPlay, "", "Play / pause (Space)", self._toggle_play)
        self._play_btn.setIconSize(QSize(22, 22))
        row.addWidget(self._play_btn)
        row.addWidget(self._tool(None, "|▶", "Next frame (Right)", lambda: self.step_frames(1)))
        row.addWidget(self._tool(sp.SP_MediaSeekForward, "", "Forward 1 s (Shift+Right)", lambda: self.step_seconds(1.0)))
        row.addWidget(self._tool(sp.SP_MediaSkipForward, "", "Go to end (End)", self._seek_end))

        self._time_label = QLabel()
        self._time_label.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 13px;")
        self._time_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._time_label.setMinimumWidth(420)
        row.addSpacing(6)
        row.addWidget(self._time_label)

        self._goto_edit = QLineEdit()
        self._goto_edit.setPlaceholderText("Go to…")
        self._goto_edit.setToolTip("Jump to a time (h:mm:ss.mmm, mm:ss, seconds) or a frame (f1234)")
        self._goto_edit.setMaximumWidth(110)
        self._goto_edit.returnPressed.connect(self._on_goto)
        row.addWidget(self._goto_edit)

        row.addStretch(1)

        self._speed_combo = QComboBox()
        for speed in SPEEDS:
            self._speed_combo.addItem(f"{speed:g}×", speed)
        self._speed_combo.setCurrentIndex(SPEEDS.index(1.0))
        self._speed_combo.setToolTip("Playback speed ([ slower, ] faster)")
        self._speed_combo.currentIndexChanged.connect(lambda _i: self.set_rate(self._speed_combo.currentData()))
        row.addWidget(self._speed_combo)

        self._audio_combo = QComboBox()
        self._audio_combo.setToolTip("Camera providing the sound")
        self._audio_combo.setMinimumWidth(150)
        self._audio_combo.currentIndexChanged.connect(self._on_audio_combo)
        row.addWidget(self._audio_combo)
        self._mute_btn = self._tool(sp.SP_MediaVolume, "", "Mute / unmute (M)", self.toggle_mute)
        row.addWidget(self._mute_btn)
        self._volume = QSlider(Qt.Orientation.Horizontal)
        self._volume.setRange(0, 100)
        self._volume.setFixedWidth(70)
        self._volume.setToolTip("Volume")
        if self._audio_output is not None:
            self._volume.setValue(int(self._audio_output.volume() * 100))
        self._volume.valueChanged.connect(self._on_volume)
        row.addWidget(self._volume)
        self._update_mute_icon()
        if not self._use_qt:
            for widget in (self._audio_combo, self._mute_btn, self._volume):
                widget.setEnabled(False)
                widget.setToolTip("Audio requires PyQt6 ≥ 6.8 (Qt Multimedia FFmpeg backend)")

        self._layout_btn = QToolButton()
        self._layout_btn.setText("Layout")
        self._layout_btn.setToolTip("Camera arrangement (double-click a camera to focus it)")
        self._layout_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._layout_btn.setAutoRaise(True)
        layout_menu = QMenu(self._layout_btn)
        for key, text in (("auto", "Auto (largest videos)"), ("row", "Single row"),
                          ("grid", "Two columns"), ("focus", "Focus one camera")):
            action = layout_menu.addAction(text)
            action.triggered.connect(lambda _c=False, k=key: self.set_layout_mode(k))
        self._layout_btn.setMenu(layout_menu)
        row.addWidget(self._layout_btn)

        self._popout_btn = self._tool(sp.SP_TitleBarNormalButton, "", "Show cameras in a separate window (for a second screen)", self.toggle_popout)
        row.addWidget(self._popout_btn)

        self._capture_btn = QToolButton()
        self._capture_btn.setText("Capture")
        self._capture_btn.setToolTip("Export the current frames")
        self._capture_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._capture_btn.setAutoRaise(True)
        capture_menu = QMenu(self._capture_btn)
        capture_menu.addAction("Export camera frames (PNG)…", self.export_frames_requested.emit)
        capture_menu.addAction("Export composite snapshot (cameras + signals)…", self.snapshot_requested.emit)
        self._capture_btn.setMenu(capture_menu)
        row.addWidget(self._capture_btn)
        # Backwards-compatible name used by older code.
        self._export_frames_btn = self._capture_btn
        outer.addLayout(row)
        self._extra_widget = QWidget()
        self._extra_row = QHBoxLayout(self._extra_widget)
        self._extra_row.setContentsMargins(0, 2, 0, 0)
        self._extra_row.setSpacing(6)
        self._extra_row.addStretch(1)
        self._extra_widget.hide()
        outer.addWidget(self._extra_widget)
        return bar

    def add_transport_widget(self, widget: QWidget) -> None:
        """Add a mode-specific control on a row below the transport bar."""
        self._extra_row.insertWidget(self._extra_row.count() - 1, widget)
        self._extra_widget.show()

    # ------------------------------------------------------------------ cameras
    def set_cameras(self, labels: list[str]) -> None:
        self.stop()
        for preview in self._previews:
            preview.release()
            preview.setParent(None)
            preview.deleteLater()
        self._disconnect_privacy_sinks()
        self._previews.clear()
        self._latest_frames.clear()
        for index, label in enumerate(labels):
            preview = FramePreview(label=label, face_blur_enabled=self._face_blur_enabled, use_qt=self._use_qt)
            preview.focus_requested.connect(lambda i=index: self._toggle_focus(i))
            preview.audio_requested.connect(lambda i=index: self.set_audio_camera(i))
            self._previews.append(preview)
        self._columns = 0
        self._total_frames = 0
        self._slider.setMaximum(0)
        self._rebuild_audio_combo()
        self._connect_privacy_sinks()
        self._relayout(force=True)
        self._update_label()

    def load_videos(self, paths: list[str]) -> None:
        for index, path in enumerate(paths):
            if index < len(self._previews):
                self._previews[index].load_video(path)
        self._total_frames = max((p.frame_count for p in self._previews), default=0)
        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, self._total_frames - 1))
        self._slider.blockSignals(False)
        self._position_ms = 0.0
        self._apply_audio_routing()
        self._relayout(force=True)
        if self._media_should_be_active():
            self._activate_media()
        self._update_label()

    @property
    def previews(self) -> list[FramePreview]:
        return self._previews

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def current_frame(self) -> int:
        return ms_to_frame(self._position_ms, self.get_fps())

    @property
    def current_sec(self) -> float:
        return self.current_frame / self.get_fps() if not self._playing else self._position_ms / 1000.0

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def slider(self) -> QSlider:
        return self._slider

    @property
    def duration_sec(self) -> float:
        fps = self.get_fps()
        return self._total_frames / fps if fps > 0 else 0.0

    def get_fps(self) -> float:
        for preview in self._previews:
            if preview.fps > 0 and preview.video_path:
                return preview.fps
        return 30.0

    def set_clock_anchor(self, anchor) -> None:
        """Show the wall-clock time at the playhead (anchor = video time 0)."""
        self._clock_anchor = anchor
        self._update_label()

    # ------------------------------------------------------------------ layout
    def set_layout_mode(self, mode: str) -> None:
        if mode not in LAYOUTS:
            return
        self._layout_mode = mode
        _settings().setValue("player/layout", mode)
        self._relayout(force=True)

    def _toggle_focus(self, index: int) -> None:
        if self._layout_mode == "focus" and self._focus_index == index:
            self.set_layout_mode("auto")
        else:
            self._focus_index = index
            self.set_layout_mode("focus")

    def _loaded_previews(self) -> list[FramePreview]:
        return [p for p in self._previews if p.video_path] or list(self._previews)

    def _best_columns(self) -> int:
        count = len(self._previews)
        if count <= 1:
            return 1
        aspects = sorted(p.aspect_ratio for p in self._loaded_previews())
        aspect = aspects[len(aspects) // 2]
        width = max(1, self._video_area.width())
        height = max(1, self._video_area.height())
        best_cols, best_area = 1, -1.0
        for cols in range(1, count + 1):
            rows = math.ceil(count / cols)
            cell_w = width / cols
            cell_h = height / rows - _HEADER_HEIGHT
            if cell_h <= 0:
                continue
            video_w = min(cell_w, cell_h * aspect)
            area = video_w * (video_w / aspect)
            if area > best_area * 1.02:
                best_cols, best_area = cols, area
        return best_cols

    def _relayout(self, force: bool = False) -> None:
        count = len(self._previews)
        if count == 0:
            self._empty_label.show()
            return
        self._empty_label.hide()
        if self._layout_mode == "row":
            cols = count
        elif self._layout_mode == "grid":
            cols = min(2, count)
        elif self._layout_mode == "focus":
            cols = -1
        else:
            cols = self._best_columns()
        if not force and cols == self._columns:
            return
        self._columns = cols
        for preview in self._previews:
            self._grid.removeWidget(preview)
        for r in range(self._grid.rowCount()):
            self._grid.setRowStretch(r, 0)
        for c in range(self._grid.columnCount()):
            self._grid.setColumnStretch(c, 0)
        if cols == -1:
            focus = min(self._focus_index, count - 1)
            others = [i for i in range(count) if i != focus]
            rows = max(1, len(others))
            self._grid.addWidget(self._previews[focus], 0, 0, rows, 1)
            self._grid.setColumnStretch(0, 4)
            for row, index in enumerate(others):
                self._grid.addWidget(self._previews[index], row, 1)
                self._grid.setRowStretch(row, 1)
            self._grid.setColumnStretch(1, 1)
        else:
            for index, preview in enumerate(self._previews):
                row, col = divmod(index, cols)
                self._grid.addWidget(preview, row, col)
            for c in range(cols):
                self._grid.setColumnStretch(c, 1)
            for r in range(math.ceil(count / cols)):
                self._grid.setRowStretch(r, 1)
        for preview in self._previews:
            preview.show()

    def eventFilter(self, obj, event):
        if obj is self._video_area and event.type() == QEvent.Type.Resize:
            if self._layout_mode == "auto":
                self._relayout()
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------ pop-out
    @property
    def is_popped_out(self) -> bool:
        return self._popout is not None

    def toggle_popout(self) -> None:
        if self._popout is None:
            self._want_popout = True
            self._pop_out()
        else:
            self._want_popout = False
            self._dock()

    def _pop_out(self) -> None:
        if self._popout is not None:
            return
        title = self.window().windowTitle() or "Video Research Tool"
        self._popout = _PopoutWindow(f"{title} — cameras")
        self._video_slot.removeWidget(self._video_area)
        self._popout.layout().addWidget(self._video_area)
        self._popout.closed.connect(self._on_popout_closed)
        self.install_shortcuts(self._popout, Qt.ShortcutContext.WindowShortcut)
        self._placeholder.show()
        self._popout_btn.setToolTip("Dock the cameras back into this window")
        self._popout.show()
        self._relayout(force=True)

    def _dock(self) -> None:
        if self._popout is None:
            return
        window = self._popout
        self._popout = None
        window.closed.disconnect(self._on_popout_closed)
        self._video_slot.insertWidget(0, self._video_area)
        self._placeholder.hide()
        self._popout_btn.setToolTip("Show cameras in a separate window (for a second screen)")
        window.close()
        window.deleteLater()
        self._relayout(force=True)

    def _on_popout_closed(self) -> None:
        self._want_popout = False
        if self._popout is not None:
            window = self._popout
            self._popout = None
            self._video_slot.insertWidget(0, self._video_area)
            self._placeholder.hide()
            window.deleteLater()
            self._relayout(force=True)

    # ------------------------------------------------------------------ visibility
    def _media_should_be_active(self) -> bool:
        return self._visible_media

    def _activate_media(self) -> None:
        for preview in self._previews:
            preview.note_position_ms(self._position_ms)
            preview.activate()
        self._apply_audio_routing()
        self._apply_rate()
        self._connect_privacy_sinks()

    def _deactivate_media(self) -> None:
        self.stop()
        for preview in self._previews:
            preview.deactivate()

    def showEvent(self, event):
        super().showEvent(event)
        if not event.spontaneous() and not self._visible_media:
            self._visible_media = True
            self._activate_media()
            if self._want_popout and self._popout is None:
                QTimer.singleShot(0, self._pop_out)

    def hideEvent(self, event):
        super().hideEvent(event)
        if not event.spontaneous() and self._visible_media and not self.isVisible():
            self._visible_media = False
            if self._popout is not None:
                want = self._want_popout
                self._dock()
                self._want_popout = want
            self._deactivate_media()

    # ------------------------------------------------------------------ transport
    def _master(self) -> Optional[FramePreview]:
        active = [p for p in self._previews if p.is_active and p.media_player is not None]
        if not active:
            return None
        if 0 <= self._audio_index < len(self._previews) and self._previews[self._audio_index] in active:
            return self._previews[self._audio_index]
        return active[0]

    def play(self) -> None:
        if self._playing or not self._previews:
            return
        if self.current_frame >= self._total_frames - 1 and self._total_frames > 0:
            self.seek_frame(0)
        self._playing = True
        self._flush_pending_seek()
        if self._use_qt:
            for preview in self._previews:
                if preview.media_player is not None and preview.is_active:
                    if preview.current_frame < max(0, preview.frame_count - 1):
                        preview.media_player.play()
        self._tick_count = 0
        self._tick.setInterval(
            _TICK_MS if self._use_qt
            else max(10, int(1000 / (self.get_fps() * max(self._rate, 0.01))))
        )
        self._tick.start()
        if self._face_blur_enabled and self._use_qt:
            self._privacy_timer.start()
        self._play_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self.playing_changed.emit(True)

    def pause(self) -> None:
        if not self._playing:
            return
        self._playing = False
        self._tick.stop()
        self._privacy_timer.stop()
        master = self._master()
        if master is not None:
            self._position_ms = float(master.media_player.position())
        for preview in self._previews:
            if preview.media_player is not None:
                preview.media_player.pause()
        # Snap every camera to the same exact frame.
        self._apply_seek(frame_to_ms(self.current_frame, self.get_fps()))
        self._play_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.playing_changed.emit(False)
        self._emit_position()

    def stop(self) -> None:
        """Pause playback (kept for compatibility with the previous player)."""
        self.pause()

    def toggle_play(self) -> None:
        self._toggle_play()

    def _toggle_play(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()

    def set_rate(self, rate: float) -> None:
        self._rate = float(rate or 1.0)
        index = self._speed_combo.findData(self._rate)
        if index >= 0 and index != self._speed_combo.currentIndex():
            self._speed_combo.blockSignals(True)
            self._speed_combo.setCurrentIndex(index)
            self._speed_combo.blockSignals(False)
        self._apply_rate()

    def change_speed(self, direction: int) -> None:
        index = max(0, min(len(SPEEDS) - 1, self._speed_combo.currentIndex() + direction))
        self._speed_combo.setCurrentIndex(index)

    def _apply_rate(self) -> None:
        for preview in self._previews:
            if preview.media_player is not None:
                preview.media_player.setPlaybackRate(self._rate)
        if not self._use_qt and self._playing:
            self._tick.setInterval(max(10, int(1000 / (self.get_fps() * self._rate))))

    def seek_frame(self, frame_no: int) -> None:
        frame_no = max(0, min(int(frame_no), max(0, self._total_frames - 1)))
        self._request_seek(frame_to_ms(frame_no, self.get_fps()))

    def seek_seconds(self, sec: float) -> None:
        self.seek_frame(int(math.floor(max(0.0, sec) * self.get_fps() + 1e-6)))

    def seek_normalised(self, ratio: float) -> None:
        self.seek_frame(int(ratio * max(0, self._total_frames - 1)))

    def step_frames(self, count: int) -> None:
        if self._playing:
            self.pause()
        self.seek_frame(self.current_frame + count)

    def step_seconds(self, sec: float) -> None:
        base = self._position_ms / 1000.0 if self._playing else self.current_frame / self.get_fps()
        self.seek_seconds(base + sec)

    def _seek_end(self) -> None:
        self.seek_frame(self._total_frames - 1)

    def _on_goto(self) -> None:
        sec = parse_time_text(self._goto_edit.text(), self.get_fps())
        if sec is None:
            self._goto_edit.setStyleSheet("border: 1px solid #c33;")
            return
        self._goto_edit.setStyleSheet("")
        self._goto_edit.clear()
        self.seek_seconds(sec)

    def _request_seek(self, ms: float) -> None:
        self._position_ms = float(ms)
        self._update_slider()
        self._update_label()
        self._emit_position()
        if self._seek_cooldown.isActive():
            self._pending_seek_ms = ms
            return
        self._apply_seek(ms)
        self._seek_cooldown.start()

    def _flush_pending_seek(self) -> None:
        if self._pending_seek_ms is not None:
            ms = self._pending_seek_ms
            self._pending_seek_ms = None
            self._apply_seek(ms)
            self._seek_cooldown.start()

    def _apply_seek(self, ms: float) -> None:
        self._privacy_generation += 1
        for preview in self._previews:
            target = ms
            if preview.frame_count > 0:
                target = min(ms, frame_to_ms(preview.frame_count - 1, preview.fps))
            preview.set_position_ms(target)

    def _on_slider(self, value: int) -> None:
        if self._playing:
            self.pause()
        self.seek_frame(value)

    def _on_tick(self) -> None:
        fps = self.get_fps()
        if not self._use_qt:
            next_frame = self.current_frame + 1
            if next_frame >= self._total_frames:
                self.pause()
                return
            self._position_ms = frame_to_ms(next_frame, fps)
            for preview in self._previews:
                preview.set_position_ms(self._position_ms)
            self._update_slider()
            self._update_label()
            self._emit_position()
            return
        master = self._master()
        if master is None:
            self.pause()
            return
        player = master.media_player
        self._position_ms = float(player.position())
        for preview in self._previews:
            if preview.media_player is not None:
                preview.note_position_ms(preview.media_player.position())
        self._tick_count += 1
        if self._tick_count % _SYNC_EVERY_TICKS == 0:
            self._resync(master)
        if player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia or (
            self._total_frames > 0 and self.current_frame >= self._total_frames - 1
        ):
            self.pause()
            return
        self._update_slider()
        self._update_label()
        self._emit_position()

    def _resync(self, master: FramePreview) -> None:
        master_ms = master.media_player.position()
        for preview in self._previews:
            player = preview.media_player
            if preview is master or player is None or not preview.is_active:
                continue
            end_ms = frame_to_ms(max(0, preview.frame_count - 1), preview.fps)
            if master_ms >= end_ms:
                continue
            drift = player.position() - master_ms
            if player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                player.setPosition(master_ms)
                player.play()
            elif abs(drift) > _HARD_RESYNC_MS:
                player.setPosition(master_ms + 40)
                player.setPlaybackRate(self._rate)
            elif abs(drift) > _SOFT_RESYNC_MS:
                correction = max(-0.08, min(0.08, -drift / 1000.0))
                player.setPlaybackRate(self._rate * (1.0 + correction))
            else:
                player.setPlaybackRate(self._rate)

    def _emit_position(self) -> None:
        frame = self.current_frame
        self.frame_changed.emit(frame)
        sec = self._position_ms / 1000.0 if self._playing else frame / self.get_fps()
        self.position_changed.emit(sec)

    def _update_slider(self) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(self.current_frame)
        self._slider.blockSignals(False)

    def _update_label(self) -> None:
        fps = self.get_fps()
        frame = self.current_frame
        sec = frame / fps if fps > 0 else 0.0
        total_sec = self._total_frames / fps if fps > 0 else 0.0
        text = f"{format_hms_ms(sec)} / {format_hms_ms(total_sec)}   f {frame:,} / {self._total_frames:,}"
        clock = format_clock(clock_at(self._clock_anchor, sec))
        if clock:
            text += f"   {clock}"
        self._time_label.setText(text)

    # ------------------------------------------------------------------ audio
    def _rebuild_audio_combo(self) -> None:
        self._audio_combo.blockSignals(True)
        self._audio_combo.clear()
        self._audio_combo.addItem("\U0001F507 No sound", -1)
        for index, preview in enumerate(self._previews):
            self._audio_combo.addItem(f"\U0001F50A {preview.label_text}", index)
        if self._audio_index >= len(self._previews):
            self._audio_index = 0
        position = self._audio_combo.findData(self._audio_index)
        self._audio_combo.setCurrentIndex(position if position >= 0 else 0)
        self._audio_combo.blockSignals(False)
        self._apply_audio_routing()

    def _on_audio_combo(self, _index: int) -> None:
        self.set_audio_camera(self._audio_combo.currentData())

    def set_audio_camera(self, index: int) -> None:
        self._audio_index = int(index) if index is not None else -1
        _settings().setValue("player/audio_camera", self._audio_index)
        position = self._audio_combo.findData(self._audio_index)
        if position >= 0 and position != self._audio_combo.currentIndex():
            self._audio_combo.blockSignals(True)
            self._audio_combo.setCurrentIndex(position)
            self._audio_combo.blockSignals(False)
        self._apply_audio_routing()

    @property
    def audio_camera(self) -> int:
        return self._audio_index

    def _apply_audio_routing(self) -> None:
        for index, preview in enumerate(self._previews):
            enabled = index == self._audio_index
            preview.set_audio_source(enabled and self._use_qt)
            player = preview.media_player
            if player is not None:
                player.setAudioOutput(self._audio_output if enabled else None)

    def toggle_mute(self) -> None:
        if self._audio_output is None:
            return
        self._audio_output.setMuted(not self._audio_output.isMuted())
        _settings().setValue("player/muted", self._audio_output.isMuted())
        self._update_mute_icon()

    def _on_volume(self, value: int) -> None:
        if self._audio_output is None:
            return
        self._audio_output.setVolume(value / 100.0)
        _settings().setValue("player/volume", value / 100.0)
        if value > 0 and self._audio_output.isMuted():
            self.toggle_mute()

    def _update_mute_icon(self) -> None:
        muted = self._audio_output is not None and self._audio_output.isMuted()
        icon = QStyle.StandardPixmap.SP_MediaVolumeMuted if muted else QStyle.StandardPixmap.SP_MediaVolume
        self._mute_btn.setIcon(self.style().standardIcon(icon))

    # ------------------------------------------------------------------ privacy
    def set_face_blur_enabled(self, enabled: bool) -> None:
        self._face_blur_enabled = enabled
        for preview in self._previews:
            preview.set_face_blur_enabled(enabled)
        if self._privacy_worker is not None:
            self._privacy_worker.reset_tracking()
        self._connect_privacy_sinks()
        if enabled and self._use_qt:
            if self._playing:
                self._privacy_timer.start()
            else:
                QTimer.singleShot(50, self._submit_privacy_frames)
        else:
            self._privacy_timer.stop()

    def refresh_privacy(self) -> None:
        """Re-run privacy detection after settings changed."""
        if self._privacy_worker is not None:
            self._privacy_worker.reset_tracking()
        for preview in self._previews:
            if not self._use_qt:
                preview._render_still()
        if self._face_blur_enabled:
            QTimer.singleShot(50, self._submit_privacy_frames)

    def _disconnect_privacy_sinks(self) -> None:
        # Only remove our own slots: the sink also drives Qt's internal rendering.
        for sink, slot in self._sink_slots:
            try:
                sink.videoFrameChanged.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._sink_slots.clear()

    def _connect_privacy_sinks(self) -> None:
        self._disconnect_privacy_sinks()
        if not self._use_qt or not self._face_blur_enabled:
            return
        for index, preview in enumerate(self._previews):
            sink = preview.display_widget.videoSink()
            slot = lambda frame, i=index: self._on_sink_frame(i, frame)  # noqa: E731
            sink.videoFrameChanged.connect(slot)
            self._sink_slots.append((sink, slot))

    def _on_sink_frame(self, index: int, frame) -> None:
        self._latest_frames[index] = frame
        if not self._playing and not self._privacy_debounce.isActive():
            self._privacy_debounce.start()

    def _ensure_privacy_worker(self) -> _PrivacyWorker:
        if self._privacy_worker is None:
            # No QObject parent: the thread is stopped explicitly in shutdown().
            self._privacy_worker = _PrivacyWorker()
            self._privacy_worker.regions_ready.connect(self._on_privacy_regions)
            self._privacy_worker.start()
            app = QApplication.instance()
            if app is not None:
                app.aboutToQuit.connect(self.shutdown)
        return self._privacy_worker

    def _submit_privacy_frames(self) -> None:
        if not self._face_blur_enabled or not self._use_qt:
            return
        worker = self._ensure_privacy_worker()
        for index, frame in list(self._latest_frames.items()):
            if frame is None or not frame.isValid() or worker.busy_with(index):
                continue
            t_sec = frame.startTime() / 1e6 if frame.startTime() >= 0 else self._position_ms / 1000.0
            worker.submit(index, QVideoFrame(frame), t_sec, self._privacy_generation, self._playing)

    def _on_privacy_regions(self, camera: int, _generation: int, payload) -> None:
        if not self._face_blur_enabled or camera >= len(self._previews):
            return
        rects, patches = payload
        self._previews[camera].set_privacy_masks(rects, patches)

    # ------------------------------------------------------------------ shortcuts
    def install_shortcuts(self, host: QWidget,
                          context: Qt.ShortcutContext = Qt.ShortcutContext.WidgetWithChildrenShortcut) -> None:
        """Install transport keyboard shortcuts on *host* (a mode or window)."""
        bindings = (
            ("Space", self._toggle_play),
            ("K", self._toggle_play),
            ("Left", lambda: self.step_frames(-1)),
            ("Right", lambda: self.step_frames(1)),
            ("Shift+Left", lambda: self.step_seconds(-1.0)),
            ("Shift+Right", lambda: self.step_seconds(1.0)),
            ("Ctrl+Left", lambda: self.step_seconds(-10.0)),
            ("Ctrl+Right", lambda: self.step_seconds(10.0)),
            ("J", lambda: self.step_seconds(-5.0)),
            ("L", lambda: self.step_seconds(5.0)),
            ("Home", lambda: self.seek_seconds(0.0)),
            ("End", self._seek_end),
            ("M", self.toggle_mute),
            ("[", lambda: self.change_speed(-1)),
            ("]", lambda: self.change_speed(1)),
        )
        for key, slot in bindings:
            shortcut = QShortcut(QKeySequence(key), host)
            shortcut.setContext(context)
            shortcut.activated.connect(slot)
        self._shortcut_hosts.append(host)

    def shutdown(self) -> None:
        """Stop playback, background detection and close the pop-out window."""
        self.stop()
        self._privacy_timer.stop()
        self._disconnect_privacy_sinks()
        if self._privacy_worker is not None:
            self._privacy_worker.stop()
            self._privacy_worker = None
        if self._popout is not None:
            self._dock()
        for preview in self._previews:
            preview.deactivate()


SHORTCUT_HELP = (
    ("Space / K", "Play / pause"),
    ("Left / Right", "Previous / next frame"),
    ("Shift+Left / Shift+Right", "Back / forward 1 s"),
    ("Ctrl+Left / Ctrl+Right", "Back / forward 10 s"),
    ("J / L", "Back / forward 5 s"),
    ("Home / End", "Go to start / end"),
    ("[ / ]", "Slower / faster playback"),
    ("M", "Mute / unmute"),
    ("Double-click a camera", "Focus that camera (double-click again to restore)"),
)
