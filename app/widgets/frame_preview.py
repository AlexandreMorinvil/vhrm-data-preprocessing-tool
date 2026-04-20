from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QSizePolicy, QSlider,
    QPushButton, QVBoxLayout, QWidget,
)

log = logging.getLogger(__name__)


class FramePreview(QWidget):
    clicked = pyqtSignal()

    def __init__(self, label: str = "Camera", parent=None):
        super().__init__(parent)
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 30.0
        self._frame_count: int = 0
        self._current_frame: int = 0
        self._video_path: str = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._label = QLabel(label)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._label)

        self._display = QLabel()
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._display.setMinimumSize(320, 180)
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
        pixmap = QPixmap.fromImage(qimg).scaled(
            self._display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._display.setPixmap(pixmap)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._cap is not None:
            self._show_frame(self._current_frame)


class MultiCameraPlayer(QWidget):
    frame_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._previews: list[FramePreview] = []
        self._playing = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)
        self._total_frames = 0

        self._vlayout = QVBoxLayout(self)
        self._preview_layout = QHBoxLayout()
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

        self._vlayout.addLayout(ctrl_row)

    def set_cameras(self, labels: list[str]) -> None:
        self.stop()
        for pw in self._previews:
            pw.release()
            pw.setParent(None)
            pw.deleteLater()
        self._previews.clear()
        for lbl in labels:
            pw = FramePreview(label=lbl)
            self._preview_layout.addWidget(pw)
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
