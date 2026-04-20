from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import QInputDialog, QMenu, QWidget

log = logging.getLogger(__name__)

_COLOURS = [
    "#4488cc", "#cc4444", "#44aa44", "#cc8844",
    "#8844cc", "#44cccc", "#cc44aa", "#88cc44",
    "#886644", "#448888", "#aa44cc", "#cccc44",
]


def colour_for_label(label: str, library: list[str]) -> str:
    try:
        idx = library.index(label)
    except ValueError:
        idx = hash(label)
    return _COLOURS[idx % len(_COLOURS)]


class IntervalItem:
    __slots__ = ("label", "start_sec", "end_sec", "color")

    def __init__(self, label, start_sec, end_sec, color="#4488cc"):
        self.label = label
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.color = color


class TimelineWidget(QWidget):
    interval_created = pyqtSignal(float, float)
    interval_selected = pyqtSignal(int)
    interval_deleted = pyqtSignal(int)
    playhead_moved = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(50)
        self.setMouseTracking(True)

        self._duration_sec: float = 0.0
        self._intervals: list[IntervalItem] = []
        self._playhead_sec: float = 0.0
        self._drag_start: Optional[float] = None
        self._drag_current: Optional[float] = None
        self._selected_idx: int = -1

    def set_duration(self, duration_sec):
        self._duration_sec = max(0.0, duration_sec)
        self.update()

    def set_intervals(self, intervals):
        self._intervals = list(intervals)
        self._selected_idx = -1
        self.update()

    def set_playhead(self, sec):
        self._playhead_sec = sec
        self.update()

    @property
    def intervals(self):
        return self._intervals

    def _sec_to_x(self, sec):
        if self._duration_sec <= 0:
            return 0.0
        return (sec / self._duration_sec) * self.width()

    def _x_to_sec(self, x):
        if self.width() <= 0:
            return 0.0
        return (x / self.width()) * self._duration_sec

    def _interval_at(self, x):
        sec = self._x_to_sec(x)
        for i, iv in enumerate(self._intervals):
            if iv.start_sec <= sec <= iv.end_sec:
                return i
        return -1

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#2a2a2a"))

        font = QFont("Segoe UI", 8)
        p.setFont(font)
        for i, iv in enumerate(self._intervals):
            x0 = self._sec_to_x(iv.start_sec)
            x1 = self._sec_to_x(iv.end_sec)
            color = QColor(iv.color)
            if i == self._selected_idx:
                color = color.lighter(140)
            p.fillRect(QRectF(x0, 0, x1 - x0, h), color)
            p.setPen(QPen(QColor("#ffffff")))
            rect = QRectF(x0 + 2, 0, x1 - x0 - 4, h)
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, iv.label)

        if self._drag_start is not None and self._drag_current is not None:
            x0 = self._sec_to_x(min(self._drag_start, self._drag_current))
            x1 = self._sec_to_x(max(self._drag_start, self._drag_current))
            p.fillRect(QRectF(x0, 0, x1 - x0, h), QColor(255, 255, 255, 40))

        px = self._sec_to_x(self._playhead_sec)
        p.setPen(QPen(QColor("red"), 2))
        p.drawLine(int(px), 0, int(px), h)
        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            idx = self._interval_at(event.position().x())
            if idx >= 0:
                self._selected_idx = idx
                self.interval_selected.emit(idx)
                self.update()
            else:
                self._drag_start = self._x_to_sec(event.position().x())
                self._drag_current = self._drag_start
        elif event.button() == Qt.MouseButton.RightButton:
            idx = self._interval_at(event.position().x())
            if idx >= 0:
                self._show_context_menu(event, idx)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None:
            self._drag_current = self._x_to_sec(event.position().x())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._drag_start is not None and self._drag_current is not None:
                s = min(self._drag_start, self._drag_current)
                e = max(self._drag_start, self._drag_current)
                if e - s > 0.5:
                    self.interval_created.emit(s, e)
                else:
                    self.playhead_moved.emit(self._x_to_sec(event.position().x()))
            self._drag_start = None
            self._drag_current = None
            self.update()

    def _show_context_menu(self, event, idx):
        menu = QMenu(self)
        relabel_action = menu.addAction("Relabel")
        delete_action = menu.addAction("Delete")
        chosen = menu.exec(event.globalPosition().toPoint())
        if chosen == relabel_action:
            new_label, ok = QInputDialog.getText(
                self, "Relabel interval", "New label:",
                text=self._intervals[idx].label,
            )
            if ok and new_label:
                self._intervals[idx].label = new_label
                self.update()
        elif chosen == delete_action:
            self.interval_deleted.emit(idx)
