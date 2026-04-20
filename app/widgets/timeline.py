from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import QInputDialog, QMenu, QToolTip, QWidget

log = logging.getLogger(__name__)

_COLOURS = [
    "#4488cc", "#cc4444", "#44aa44", "#cc8844",
    "#8844cc", "#44cccc", "#cc44aa", "#88cc44",
    "#886644", "#448888", "#aa44cc", "#cccc44",
]

_AXIS_HEIGHT = 18  # pixels reserved for the time-axis labels


def colour_for_label(label: str, library: list[str]) -> str:
    try:
        idx = library.index(label)
    except ValueError:
        idx = hash(label)
    return _COLOURS[idx % len(_COLOURS)]


def _format_time(sec: float) -> str:
    """Format seconds as HH:MM:SS (or MM:SS if < 1 hour)."""
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _format_time_ms(sec: float) -> str:
    """Format seconds as HH:MM:SS.mmm for tooltips."""
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


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
    interval_relabelled = pyqtSignal(int, str)
    subdivide_requested = pyqtSignal(int)
    playhead_moved = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(50 + _AXIS_HEIGHT)
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

    @property
    def selected_index(self) -> int:
        return self._selected_idx

    @selected_index.setter
    def selected_index(self, idx: int):
        self._selected_idx = idx
        self.update()

    def _bar_height(self) -> int:
        return self.height() - _AXIS_HEIGHT

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

    # ------------------------------------------------------------------
    # Tick interval helpers
    # ------------------------------------------------------------------
    def _tick_step(self) -> float:
        """Choose a tick step in seconds appropriate for the current width."""
        if self._duration_sec <= 0:
            return 30.0
        px_per_sec = self.width() / self._duration_sec
        # We want at least ~60 px between ticks
        min_gap_sec = 60.0 / max(px_per_sec, 0.001)
        candidates = [5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600]
        for c in candidates:
            if c >= min_gap_sec:
                return float(c)
        return 3600.0

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        bar_h = self._bar_height()

        # ---- background ----
        p.fillRect(0, 0, w, self.height(), QColor("#2a2a2a"))

        # ---- intervals ----
        font = QFont("Segoe UI", 8)
        p.setFont(font)
        for i, iv in enumerate(self._intervals):
            x0 = self._sec_to_x(iv.start_sec)
            x1 = self._sec_to_x(iv.end_sec)
            color = QColor(iv.color)
            if i == self._selected_idx:
                color = color.lighter(140)
            p.fillRect(QRectF(x0, 0, x1 - x0, bar_h), color)
            p.setPen(QPen(QColor("#ffffff")))
            rect = QRectF(x0 + 2, 0, x1 - x0 - 4, bar_h)
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, iv.label)

        # ---- drag preview ----
        if self._drag_start is not None and self._drag_current is not None:
            x0 = self._sec_to_x(min(self._drag_start, self._drag_current))
            x1 = self._sec_to_x(max(self._drag_start, self._drag_current))
            p.fillRect(QRectF(x0, 0, x1 - x0, bar_h), QColor(255, 255, 255, 40))

        # ---- playhead ----
        px = self._sec_to_x(self._playhead_sec)
        p.setPen(QPen(QColor("red"), 2))
        p.drawLine(int(px), 0, int(px), bar_h)

        # ---- time axis ----
        if self._duration_sec > 0:
            axis_y = bar_h
            p.setPen(QPen(QColor("#888888"), 1))
            p.drawLine(0, axis_y, w, axis_y)

            tick_font = QFont("Segoe UI", 7)
            p.setFont(tick_font)
            step = self._tick_step()
            t = 0.0
            while t <= self._duration_sec:
                tx = self._sec_to_x(t)
                p.setPen(QPen(QColor("#888888"), 1))
                p.drawLine(int(tx), axis_y, int(tx), axis_y + 4)
                p.setPen(QPen(QColor("#aaaaaa")))
                label = _format_time(t)
                p.drawText(QRectF(tx - 30, axis_y + 3, 60, _AXIS_HEIGHT - 3),
                           Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                           label)
                t += step

        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            idx = self._interval_at(event.position().x())
            if idx >= 0:
                self._selected_idx = idx
                self.interval_selected.emit(idx)
                self.update()
            else:
                self._selected_idx = -1
                self.interval_selected.emit(-1)
                self._drag_start = self._x_to_sec(event.position().x())
                self._drag_current = self._drag_start
                self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            idx = self._interval_at(event.position().x())
            if idx >= 0:
                self._show_context_menu(event, idx)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_start is not None:
            self._drag_current = self._x_to_sec(event.position().x())
            self.update()
        # Hover tooltip
        if self._duration_sec > 0:
            sec = self._x_to_sec(event.position().x())
            sec = max(0.0, min(sec, self._duration_sec))
            QToolTip.showText(event.globalPosition().toPoint(),
                              _format_time_ms(sec), self)

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
        subdivide_action = menu.addAction("Subdivide …")
        menu.addSeparator()
        delete_action = menu.addAction("Delete")
        chosen = menu.exec(event.globalPosition().toPoint())
        if chosen == relabel_action:
            new_label, ok = QInputDialog.getText(
                self, "Relabel interval", "New label:",
                text=self._intervals[idx].label,
            )
            if ok and new_label:
                self._intervals[idx].label = new_label
                self.interval_relabelled.emit(idx, new_label)
                self.update()
        elif chosen == subdivide_action:
            self.subdivide_requested.emit(idx)
        elif chosen == delete_action:
            self.interval_deleted.emit(idx)
