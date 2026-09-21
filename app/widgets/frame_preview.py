from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QSlider,
    QPushButton, QVBoxLayout, QWidget,
)

log = logging.getLogger(__name__)

_DISPLAY_MIN_SIZE = QSize(240, 135)
_DISPLAY_MAX_HEIGHT = 360
_DISPLAY_RESIZE_STEP = 48
_PREVIEW_SIZE_HINT = QSize(480, 350)
_PREVIEW_COLUMNS = 2


class FramePreview(QWidget):
    clicked = pyqtSignal()

    def __init__(self, label: str = "Camera", parent=None):
        super().__init__(parent)
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 30.0
        self._frame_count: int = 0
        self._current_frame: int = 0
        self._video_path: str = ""
        self._source_pixmap: Optional[QPixmap] = None
        self._scaled_size_bucket: Optional[tuple[int, int]] = None

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._label = QLabel(label)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._label)

        self._display = QLabel()
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._display.setMinimumSize(_DISPLAY_MIN_SIZE)
        self._display.setMaximumHeight(_DISPLAY_MAX_HEIGHT)
        self._display.setStyleSheet("background: #222; border: 1px solid #555;")
        layout.addWidget(self._display)

        self._info_label = QLabel("No video loaded")
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._info_label)

    def load_video(self, path: str) -> bool:
        self.release()
        if not path or not Path(path).exists():
            return False
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            log.warning("Cannot open video: %s", path)
            return False
        self._cap = cap
        self._video_path = path
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._current_frame = 0
        self._show_frame(0)
        self._info_label.setText(
            f"{Path(path).name}  |  {self._frame_count} frames  |  {self._fps:.1f} fps"
        )
        return True

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._frame_count = 0
        self._current_frame = 0
        self._source_pixmap = None
        self._scaled_size_bucket = None

    def seek_frame(self, frame_no: int) -> None:
        if self._cap is None:
            return
        frame_no = max(0, min(frame_no, self._frame_count - 1))
        self._show_frame(frame_no)

    def seek_normalised(self, ratio: float) -> None:
        if self._frame_count <= 0:
            return
        frame_no = int(ratio * (self._frame_count - 1))
        self.seek_frame(frame_no)

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def current_frame(self) -> int:
        return self._current_frame

    @property
    def video_path(self) -> str:
        return self._video_path

    def _show_frame(self, frame_no: int) -> None:
        if self._cap is None:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = self._cap.read()
        if not ret:
            return
        self._current_frame = frame_no
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._source_pixmap = QPixmap.fromImage(qimg.copy())
        self._render_cached_frame(force=True)

    def _display_size_bucket(self) -> tuple[int, int]:
        size = self._display.size()
        width = max(1, size.width())
        height = max(1, min(size.height(), _DISPLAY_MAX_HEIGHT))
        bucket_width = max(
            _DISPLAY_MIN_SIZE.width(),
            (width // _DISPLAY_RESIZE_STEP) * _DISPLAY_RESIZE_STEP,
        )
        bucket_height = max(
            _DISPLAY_MIN_SIZE.height(),
            (height // _DISPLAY_RESIZE_STEP) * _DISPLAY_RESIZE_STEP,
        )
        return bucket_width, bucket_height

    def _render_cached_frame(self, force: bool = False) -> None:
        if self._source_pixmap is None:
            return
        bucket = self._display_size_bucket()
        if not force and bucket == self._scaled_size_bucket:
            return
        self._scaled_size_bucket = bucket
        pixmap = self._source_pixmap.scaled(
            QSize(*bucket),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._display.setPixmap(pixmap)

    def sizeHint(self) -> QSize:
        return _PREVIEW_SIZE_HINT

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_cached_frame()


class MultiCameraPlayer(QWidget):
    frame_changed = pyqtSignal(int)
    export_frames_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._previews: list[FramePreview] = []
        self._playing = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)
        self._total_frames = 0

        self._vlayout = QVBoxLayout(self)
        self._preview_layout = QGridLayout()
        self._vlayout.addLayout(self._preview_layout)

        ctrl_row = QHBoxLayout()
        self._play_btn = QPushButton("\u25b6 Play")
        self._play_btn.setFixedWidth(100)
        self._play_btn.clicked.connect(self._toggle_play)
        ctrl_row.addWidget(self._play_btn)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)
        ctrl_row.addWidget(self._slider)

        self._frame_label = QLabel("0 / 0")
        self._frame_label.setFixedWidth(140)
        ctrl_row.addWidget(self._frame_label)

        self._export_frames_btn = QPushButton("Export frames")
        self._export_frames_btn.setToolTip("Export the current frame from each loaded camera")
        self._export_frames_btn.clicked.connect(self.export_frames_requested.emit)
        ctrl_row.addWidget(self._export_frames_btn)

        self._vlayout.addLayout(ctrl_row)

    def set_cameras(self, labels: list[str]) -> None:
        self.stop()
        for pw in self._previews:
            pw.release()
            pw.setParent(None)
            pw.deleteLater()
        self._previews.clear()
        for index, lbl in enumerate(labels):
            pw = FramePreview(label=lbl)
            row, column = divmod(index, _PREVIEW_COLUMNS)
            self._preview_layout.addWidget(pw, row, column)
            self._previews.append(pw)

    def load_videos(self, paths: list[str]) -> None:
        for i, p in enumerate(paths):
            if i < len(self._previews):
                self._previews[i].load_video(p)
        self._total_frames = max((pw.frame_count for pw in self._previews), default=0)
        self._slider.setMaximum(max(0, self._total_frames - 1))
        self._update_label()

    def seek_frame(self, frame_no: int) -> None:
        for pw in self._previews:
            pw.seek_frame(frame_no)
        self._slider.blockSignals(True)
        self._slider.setValue(frame_no)
        self._slider.blockSignals(False)
        self._update_label()
        self.frame_changed.emit(frame_no)

    def seek_normalised(self, ratio: float) -> None:
        frame_no = int(ratio * max(0, self._total_frames - 1))
        self.seek_frame(frame_no)

    def stop(self) -> None:
        self._playing = False
        self._timer.stop()
        self._play_btn.setText("\u25b6 Play")

    @property
    def previews(self) -> list[FramePreview]:
        return self._previews

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def current_frame(self) -> int:
        return self._slider.value()

    @property
    def slider(self) -> QSlider:
        return self._slider

    def get_fps(self) -> float:
        for pw in self._previews:
            if pw.fps > 0:
                return pw.fps
        return 30.0

    def _toggle_play(self) -> None:
        if self._playing:
            self.stop()
        else:
            self._playing = True
            self._play_btn.setText("\u23f8 Pause")
            fps = self.get_fps()
            self._timer.start(int(1000 / fps))

    def _advance_frame(self) -> None:
        cur = self._slider.value()
        if cur >= self._total_frames - 1:
            self.stop()
            return
        self._slider.setValue(cur + 1)

    def _on_slider(self, value: int) -> None:
        for pw in self._previews:
            pw.seek_frame(value)
        self._update_label()
        self.frame_changed.emit(value)

    def _update_label(self) -> None:
        cur = self._slider.value()
        fps = self.get_fps()
        secs = cur / fps if fps > 0 else 0
        m, s = divmod(secs, 60)
        self._frame_label.setText(f"{cur} / {self._total_frames}  ({int(m):02d}:{s:05.2f})")
