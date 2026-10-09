"""Interactive, zoomable label timeline.

Layout (top to bottom): a time ruler (click/drag to scrub), the interval lane
(drag on empty space to create, drag edges to resize, drag a selected interval
to move it), and an overview strip of the whole session (click/drag to move
the visible window).

Mouse: wheel = zoom around the cursor, Shift+wheel or middle-drag = pan.
Edges snap to the playhead and to neighbouring intervals (hold Shift to
disable snapping). Intervals never overlap while being edited.
"""
from __future__ import annotations

import logging
import zlib
from typing import Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QFont, QFontMetrics, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import QInputDialog, QMenu, QToolTip, QWidget

from .. import theme
from ..timefmt import format_axis_time, format_hms, format_hms_ms, nice_time_step

log = logging.getLogger(__name__)

_COLOURS = [
    "#4488cc", "#cc4444", "#44aa44", "#cc8844",
    "#8844cc", "#44cccc", "#cc44aa", "#88cc44",
    "#886644", "#448888", "#aa44cc", "#cccc44",
]

_RULER_HEIGHT = 18
_OVERVIEW_HEIGHT = 12
_EDGE_PX = 5
_SNAP_PX = 7
_DRAG_START_PX = 4
_MIN_INTERVAL_SEC = 0.5
_MIN_VIEW_SEC = 0.2


def colour_for_label(label: str, library: list[str]) -> str:
    try:
        idx = library.index(label)
    except ValueError:
        idx = zlib.crc32(label.encode("utf-8"))
    return _COLOURS[idx % len(_COLOURS)]


def _format_time(sec: float) -> str:
    """Format seconds as H:MM:SS (or M:SS if < 1 hour)."""
    return format_hms(sec)


def _format_time_ms(sec: float) -> str:
    """Format seconds as HH:MM:SS.mmm for tooltips."""
    return format_hms_ms(sec)


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
    interval_resized = pyqtSignal(int, float, float)   # idx, new_start, new_end (resize or move)
    interval_activated = pyqtSignal(int)               # double-click
    subdivide_requested = pyqtSignal(int)
    mosaic_requested = pyqtSignal(int)
    figure_requested = pyqtSignal(int)
    split_requested = pyqtSignal(int, float)
    playhead_moved = pyqtSignal(float)
    view_range_changed = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(_RULER_HEIGHT + 36 + _OVERVIEW_HEIGHT)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self._duration_sec: float = 0.0
        self._view_start: float = 0.0
        self._view_end: float = 0.0
        self._intervals: list[IntervalItem] = []
        self._playhead_sec: float = 0.0
        self._selected_idx: int = -1
        self._label_choices: list[str] = []
        self._editable = True
        self._follow_playhead = True
        self._margin_left = 0
        self._margin_right = 0

        # Drag state
        self._drag_mode: str = ""          # create|resize|move|scrub|pan|overview|press-interval
        self._drag_idx: int = -1
        self._drag_side: str = ""
        self._press_x: float = 0.0
        self._press_sec: float = 0.0
        self._orig_start: float = 0.0
        self._orig_end: float = 0.0
        self._pan_origin: tuple[float, float] = (0.0, 0.0)
        self._drag_start: Optional[float] = None
        self._drag_current: Optional[float] = None
        self._snap_sec: Optional[float] = None

        # Pending label (for "Start label / End label" workflow)
        self._pending_start_sec: Optional[float] = None
        self._pending_color: str = "#ffffff"

        theme.notifier.changed.connect(lambda _p: self.update())

    # ------------------------------------------------------------------ data
    def set_duration(self, duration_sec):
        self._duration_sec = max(0.0, float(duration_sec))
        self._view_start = 0.0
        self._view_end = self._duration_sec
        self.update()

    @property
    def duration(self) -> float:
        return self._duration_sec

    def set_intervals(self, intervals):
        self._intervals = list(intervals)
        if self._selected_idx >= len(self._intervals):
            self._selected_idx = -1
        self.update()

    def set_playhead(self, sec):
        self._playhead_sec = float(sec)
        if (
            self._follow_playhead
            and self._drag_mode == ""
            and self._view_span() < self._duration_sec - 1e-6
            and not (self._view_start <= self._playhead_sec <= self._view_end)
        ):
            span = self._view_span()
            self._set_view(self._playhead_sec - 0.1 * span, self._playhead_sec + 0.9 * span, emit=True)
        self.update()

    @property
    def playhead_sec(self) -> float:
        return self._playhead_sec

    def set_label_choices(self, labels: list[str]) -> None:
        self._label_choices = list(labels)

    def set_editable(self, editable: bool) -> None:
        self._editable = editable

    def set_follow_playhead(self, enabled: bool) -> None:
        self._follow_playhead = enabled

    def set_content_margins(self, left: int, right: int) -> None:
        """Align the time axis with an adjacent plot's data area."""
        left, right = max(0, int(left)), max(0, int(right))
        if (left, right) != (self._margin_left, self._margin_right):
            self._margin_left, self._margin_right = left, right
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

    # ------------------------------------------------------------------ view
    def view_range(self) -> tuple[float, float]:
        return self._view_start, self._view_end

    def set_view_range(self, start: float, end: float, emit: bool = False) -> None:
        self._set_view(start, end, emit=emit)

    def zoom_to_fit(self) -> None:
        self._set_view(0.0, self._duration_sec, emit=True)

    def zoom_to(self, start: float, end: float, padding: float = 0.05) -> None:
        span = max(_MIN_VIEW_SEC, end - start)
        self._set_view(start - span * padding, end + span * padding, emit=True)

    def _view_span(self) -> float:
        return max(1e-9, self._view_end - self._view_start)

    def _set_view(self, start: float, end: float, emit: bool) -> None:
        if self._duration_sec <= 0:
            return
        span = min(max(_MIN_VIEW_SEC, end - start), self._duration_sec)
        start = max(0.0, min(start, self._duration_sec - span))
        new = (start, start + span)
        if abs(new[0] - self._view_start) < 1e-9 and abs(new[1] - self._view_end) < 1e-9:
            return
        self._view_start, self._view_end = new
        self.update()
        if emit:
            self.view_range_changed.emit(*new)

    # ------------------------------------------------------------------ pending
    def set_pending_start(self, sec: Optional[float], color: str = "#ffffff"):
        """Mark *sec* as the pending label start. Pass *None* to clear."""
        self._pending_start_sec = sec
        self._pending_color = color
        self.update()

    @property
    def pending_start(self) -> Optional[float]:
        return self._pending_start_sec

    # ------------------------------------------------------------------ geometry
    def _lane_top(self) -> int:
        return _RULER_HEIGHT

    def _lane_bottom(self) -> int:
        return self.height() - _OVERVIEW_HEIGHT - 2

    def _bar_height(self) -> int:
        return self._lane_bottom() - self._lane_top()

    def _content_width(self) -> float:
        return max(1.0, self.width() - self._margin_left - self._margin_right)

    def _sec_to_x(self, sec):
        if self._duration_sec <= 0:
            return float(self._margin_left)
        return self._margin_left + (sec - self._view_start) / self._view_span() * self._content_width()

    def _x_to_sec(self, x):
        if self._duration_sec <= 0:
            return 0.0
        return self._view_start + (x - self._margin_left) / self._content_width() * self._view_span()

    def _overview_x(self, sec: float) -> float:
        if self._duration_sec <= 0:
            return float(self._margin_left)
        return self._margin_left + sec / self._duration_sec * self._content_width()

    def _overview_sec(self, x: float) -> float:
        return max(0.0, min(self._duration_sec, (x - self._margin_left) / self._content_width() * self._duration_sec))

    def _interval_at(self, x):
        sec = self._x_to_sec(x)
        for i, iv in enumerate(self._intervals):
            if iv.start_sec <= sec <= iv.end_sec:
                return i
        return -1

    def _edge_at(self, x) -> tuple[int, str]:
        """Return ``(interval_index, 'start'|'end')`` near *x*, else ``(-1, '')``."""
        best = (-1, "", _EDGE_PX + 1.0)
        for i, iv in enumerate(self._intervals):
            for side, sec in (("start", iv.start_sec), ("end", iv.end_sec)):
                distance = abs(x - self._sec_to_x(sec))
                if distance <= _EDGE_PX and distance < best[2]:
                    best = (i, side, distance)
        return best[0], best[1]

    def _neighbour_limits(self, idx: int) -> tuple[float, float]:
        iv = self._intervals[idx]
        lower, upper = 0.0, self._duration_sec
        for j, other in enumerate(self._intervals):
            if j == idx:
                continue
            if other.end_sec <= iv.start_sec + 1e-9:
                lower = max(lower, other.end_sec)
            elif other.start_sec >= iv.end_sec - 1e-9:
                upper = min(upper, other.start_sec)
        return lower, upper

    def _snap(self, sec: float, exclude_idx: int = -1, modifiers=None) -> float:
        self._snap_sec = None
        if modifiers is not None and modifiers & Qt.KeyboardModifier.ShiftModifier:
            return sec
        targets = [self._playhead_sec]
        if self._pending_start_sec is not None:
            targets.append(self._pending_start_sec)
        for j, iv in enumerate(self._intervals):
            if j != exclude_idx:
                targets.extend((iv.start_sec, iv.end_sec))
        tolerance = _SNAP_PX / self._content_width() * self._view_span()
        best = min(targets, key=lambda t: abs(t - sec), default=None)
        if best is not None and abs(best - sec) <= tolerance:
            self._snap_sec = best
            return best
        return sec

    def _clamp_sec(self, sec: float) -> float:
        return max(0.0, min(sec, self._duration_sec))

    # ------------------------------------------------------------------ painting
    def paintEvent(self, event):
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        top, bottom = self._lane_top(), self._lane_bottom()
        bar_h = bottom - top

        p.fillRect(0, 0, w, self.height(), QColor(pal.panel))
        p.fillRect(QRectF(self._margin_left, top, self._content_width(), bar_h), QColor(pal.lane))

        self._paint_ruler(p, pal)
        p.save()
        p.setClipRect(QRectF(self._margin_left, 0, self._content_width(), self.height()))

        font = QFont(self.font())
        font.setPointSize(8)
        p.setFont(font)
        metrics = QFontMetrics(font)
        for i, iv in enumerate(self._intervals):
            x0 = self._sec_to_x(iv.start_sec)
            x1 = self._sec_to_x(iv.end_sec)
            if x1 < self._margin_left - 2 or x0 > w - self._margin_right + 2:
                continue
            rect = QRectF(x0, top + 2, max(1.0, x1 - x0), bar_h - 4)
            color = QColor(iv.color)
            if i == self._selected_idx:
                color = color.lighter(135)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            p.drawRoundedRect(rect, 3, 3)
            if i == self._selected_idx:
                p.setPen(QPen(QColor("#ffffff"), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 3, 3)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor("#ffffff"))
                p.drawRect(QRectF(x0, top + bar_h / 2 - 6, 3, 12))
                p.drawRect(QRectF(x1 - 3, top + bar_h / 2 - 6, 3, 12))
            if rect.width() > 14:
                p.setPen(QPen(QColor("#ffffff")))
                visible = rect.intersected(QRectF(self._margin_left, 0, self._content_width(), self.height()))
                text = metrics.elidedText(iv.label, Qt.TextElideMode.ElideRight, int(visible.width()) - 6)
                p.drawText(visible.adjusted(3, 0, -3, 0), Qt.AlignmentFlag.AlignCenter, text)

        if self._drag_mode == "create" and self._drag_start is not None and self._drag_current is not None:
            x0 = self._sec_to_x(min(self._drag_start, self._drag_current))
            x1 = self._sec_to_x(max(self._drag_start, self._drag_current))
            p.fillRect(QRectF(x0, top, x1 - x0, bar_h), QColor(255, 255, 255, 50))
            p.setPen(QPen(QColor(255, 255, 255, 160), 1, Qt.PenStyle.DashLine))
            p.drawRect(QRectF(x0, top, x1 - x0, bar_h))

        if self._pending_start_sec is not None:
            pend_x = self._sec_to_x(self._pending_start_sec)
            head_x = self._sec_to_x(self._playhead_sec)
            pc = QColor(self._pending_color)
            pc.setAlpha(60)
            p.fillRect(QRectF(min(pend_x, head_x), top, abs(head_x - pend_x), bar_h), pc)
            p.setPen(QPen(QColor(self._pending_color), 1.5, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(pend_x, 0), QPointF(pend_x, bottom))

        if self._snap_sec is not None and self._drag_mode in ("create", "resize", "move"):
            sx = self._sec_to_x(self._snap_sec)
            p.setPen(QPen(QColor("#ffd54a"), 1, Qt.PenStyle.DotLine))
            p.drawLine(QPointF(sx, top), QPointF(sx, bottom))

        px = self._sec_to_x(self._playhead_sec)
        p.setPen(QPen(QColor(pal.playhead), 2))
        p.drawLine(QPointF(px, 0), QPointF(px, bottom))
        p.setBrush(QColor(pal.playhead))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon([QPointF(px - 5, 0), QPointF(px + 5, 0), QPointF(px, 7)])
        p.restore()

        self._paint_overview(p, pal)
        p.end()

    def _paint_ruler(self, p: QPainter, pal) -> None:
        if self._duration_sec <= 0:
            return
        font = QFont(self.font())
        font.setPointSize(7)
        p.setFont(font)
        px_per_sec = self._content_width() / self._view_span()
        step = nice_time_step(70.0 / max(px_per_sec, 1e-9))
        minor = step / 5 if step >= 0.005 else step
        first = int(self._view_start // minor)
        last = int(self._view_end // minor) + 1
        for k in range(first, last + 1):
            t = k * minor
            if t < self._view_start - 1e-9 or t > self._view_end + 1e-9:
                continue
            x = self._sec_to_x(t)
            major = abs(t / step - round(t / step)) < 1e-6
            p.setPen(QPen(QColor(pal.axis if major else pal.grid), 1))
            p.drawLine(QPointF(x, _RULER_HEIGHT - (7 if major else 3)), QPointF(x, _RULER_HEIGHT))
            if major:
                p.setPen(QPen(QColor(pal.muted_text)))
                p.drawText(QRectF(x + 2, 0, 90, _RULER_HEIGHT - 4),
                           Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                           format_axis_time(t, step))
        p.setPen(QPen(QColor(pal.grid), 1))
        p.drawLine(QPointF(self._margin_left, _RULER_HEIGHT), QPointF(self.width() - self._margin_right, _RULER_HEIGHT))

    def _paint_overview(self, p: QPainter, pal) -> None:
        if self._duration_sec <= 0:
            return
        y = self.height() - _OVERVIEW_HEIGHT
        rect = QRectF(self._margin_left, y, self._content_width(), _OVERVIEW_HEIGHT - 1)
        p.fillRect(rect, QColor(pal.lane))
        for iv in self._intervals:
            x0 = self._overview_x(iv.start_sec)
            x1 = self._overview_x(iv.end_sec)
            color = QColor(iv.color)
            color.setAlpha(170)
            p.fillRect(QRectF(x0, y + 2, max(1.0, x1 - x0), _OVERVIEW_HEIGHT - 5), color)
        vx0 = self._overview_x(self._view_start)
        vx1 = self._overview_x(self._view_end)
        p.setPen(QPen(QColor(pal.text), 1))
        p.setBrush(QColor(255, 255, 255, 25))
        p.drawRect(QRectF(vx0, y, max(3.0, vx1 - vx0), _OVERVIEW_HEIGHT - 1))
        px = self._overview_x(self._playhead_sec)
        p.setPen(QPen(QColor(pal.playhead), 1))
        p.drawLine(QPointF(px, y), QPointF(px, y + _OVERVIEW_HEIGHT))

    # ------------------------------------------------------------------ mouse
    def _zone(self, y: float) -> str:
        if y < _RULER_HEIGHT:
            return "ruler"
        if y >= self.height() - _OVERVIEW_HEIGHT:
            return "overview"
        return "lane"

    def wheelEvent(self, event):
        if self._duration_sec <= 0:
            return
        delta = event.angleDelta()
        x = event.position().x()
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier or abs(delta.x()) > abs(delta.y()):
            amount = delta.x() if abs(delta.x()) > abs(delta.y()) else delta.y()
            shift = -amount / 120.0 * 0.15 * self._view_span()
            self._set_view(self._view_start + shift, self._view_end + shift, emit=True)
        else:
            factor = 0.8 ** (delta.y() / 120.0)
            anchor = self._x_to_sec(x)
            start = anchor - (anchor - self._view_start) * factor
            end = anchor + (self._view_end - anchor) * factor
            self._set_view(start, end, emit=True)
        event.accept()

    def mousePressEvent(self, event):
        x, y = event.position().x(), event.position().y()
        self._press_x = x
        zone = self._zone(y)
        if event.button() == Qt.MouseButton.MiddleButton:
            self._drag_mode = "pan"
            self._pan_origin = (self._view_start, self._view_end)
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            return
        if event.button() == Qt.MouseButton.RightButton:
            idx = self._interval_at(x) if zone == "lane" else -1
            self._show_context_menu(event, idx)
            return
        if event.button() != Qt.MouseButton.LeftButton or self._duration_sec <= 0:
            return
        if zone == "ruler":
            self._drag_mode = "scrub"
            self.playhead_moved.emit(self._clamp_sec(self._x_to_sec(x)))
            return
        if zone == "overview":
            self._drag_mode = "overview"
            self._center_view_on(self._overview_sec(x))
            return
        if self._editable:
            edge_idx, edge_side = self._edge_at(x)
            if edge_idx >= 0:
                self._drag_mode = "resize"
                self._drag_idx = edge_idx
                self._drag_side = edge_side
                iv = self._intervals[edge_idx]
                self._orig_start, self._orig_end = iv.start_sec, iv.end_sec
                return
        idx = self._interval_at(x)
        if idx >= 0:
            already_selected = idx == self._selected_idx
            self._selected_idx = idx
            self.interval_selected.emit(idx)
            self._drag_mode = "press-interval" if (already_selected and self._editable) else "select"
            self._drag_idx = idx
            self._press_sec = self._x_to_sec(x)
            iv = self._intervals[idx]
            self._orig_start, self._orig_end = iv.start_sec, iv.end_sec
            self.update()
            return
        self._selected_idx = -1
        self.interval_selected.emit(-1)
        self._drag_mode = "create" if self._editable else "click"
        sec = self._clamp_sec(self._x_to_sec(x))
        self._drag_start = self._snap(sec, modifiers=event.modifiers())
        self._drag_current = self._drag_start
        self.update()

    def _center_view_on(self, sec: float) -> None:
        span = self._view_span()
        self._set_view(sec - span / 2, sec + span / 2, emit=True)

    def mouseMoveEvent(self, event: QMouseEvent):
        x = event.position().x()
        mode = self._drag_mode
        if mode == "pan":
            dx = (x - self._press_x) / self._content_width() * (self._pan_origin[1] - self._pan_origin[0])
            self._set_view(self._pan_origin[0] - dx, self._pan_origin[1] - dx, emit=True)
            return
        if mode == "scrub":
            self.playhead_moved.emit(self._clamp_sec(self._x_to_sec(x)))
            return
        if mode == "overview":
            self._center_view_on(self._overview_sec(x))
            return
        if mode == "resize":
            iv = self._intervals[self._drag_idx]
            lower, upper = self._neighbour_limits(self._drag_idx)
            sec = self._snap(self._clamp_sec(self._x_to_sec(x)), self._drag_idx, event.modifiers())
            if self._drag_side == "start":
                iv.start_sec = max(lower, min(sec, iv.end_sec - _MIN_INTERVAL_SEC))
            else:
                iv.end_sec = min(upper, max(sec, iv.start_sec + _MIN_INTERVAL_SEC))
            self.update()
            self._show_hover_tip(event, f"{format_hms_ms(iv.start_sec)} → {format_hms_ms(iv.end_sec)}")
            return
        if mode == "press-interval" and abs(x - self._press_x) >= _DRAG_START_PX:
            mode = self._drag_mode = "move"
        if mode == "move":
            iv = self._intervals[self._drag_idx]
            iv.start_sec, iv.end_sec = self._orig_start, self._orig_end
            lower, upper = self._neighbour_limits(self._drag_idx)
            length = self._orig_end - self._orig_start
            start = self._orig_start + (self._x_to_sec(x) - self._press_sec)
            snapped_start = self._snap(start, self._drag_idx, event.modifiers())
            if self._snap_sec is None:
                snapped_end = self._snap(start + length, self._drag_idx, event.modifiers())
                start = snapped_end - length if self._snap_sec is not None else start
            else:
                start = snapped_start
            start = max(lower, min(start, upper - length))
            iv.start_sec, iv.end_sec = start, start + length
            self.update()
            self._show_hover_tip(event, f"{format_hms_ms(iv.start_sec)} → {format_hms_ms(iv.end_sec)}")
            return
        if mode == "create":
            self._drag_current = self._snap(self._clamp_sec(self._x_to_sec(x)), modifiers=event.modifiers())
            self.update()
            if self._drag_start is not None:
                self._show_hover_tip(
                    event,
                    f"{format_hms_ms(min(self._drag_start, self._drag_current))} → "
                    f"{format_hms_ms(max(self._drag_start, self._drag_current))}",
                )
            return

        # Hover feedback
        zone = self._zone(event.position().y())
        cursor = Qt.CursorShape.ArrowCursor
        if zone == "ruler":
            cursor = Qt.CursorShape.IBeamCursor
        elif zone == "lane" and self._editable and self._edge_at(x)[0] >= 0:
            cursor = Qt.CursorShape.SizeHorCursor
        elif zone == "lane" and self._interval_at(x) == self._selected_idx >= 0 and self._editable:
            cursor = Qt.CursorShape.SizeAllCursor
        self.setCursor(QCursor(cursor))
        if self._duration_sec > 0:
            sec = self._clamp_sec(self._x_to_sec(x) if zone != "overview" else self._overview_sec(x))
            idx = self._interval_at(x) if zone == "lane" else -1
            if idx >= 0:
                iv = self._intervals[idx]
                text = (f"{iv.label}\n{format_hms_ms(iv.start_sec)} → {format_hms_ms(iv.end_sec)}"
                        f"\nDuration {format_hms_ms(iv.end_sec - iv.start_sec)}")
            else:
                text = format_hms_ms(sec)
            self._show_hover_tip(event, text)

    def _show_hover_tip(self, event, text: str) -> None:
        QToolTip.showText(event.globalPosition().toPoint(), text, self)

    def mouseReleaseEvent(self, event):
        mode = self._drag_mode
        self._drag_mode = ""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if mode in ("resize", "move"):
            iv = self._intervals[self._drag_idx]
            changed = (abs(iv.start_sec - self._orig_start) > 1e-6
                       or abs(iv.end_sec - self._orig_end) > 1e-6)
            if changed:
                self.interval_resized.emit(self._drag_idx, round(iv.start_sec, 3), round(iv.end_sec, 3))
        elif mode == "create":
            if self._drag_start is not None and self._drag_current is not None:
                s = min(self._drag_start, self._drag_current)
                e = max(self._drag_start, self._drag_current)
                if e - s > _MIN_INTERVAL_SEC and abs(event.position().x() - self._press_x) >= _DRAG_START_PX:
                    self.interval_created.emit(s, e)
                else:
                    self.playhead_moved.emit(self._clamp_sec(self._x_to_sec(event.position().x())))
        elif mode == "click":
            self.playhead_moved.emit(self._clamp_sec(self._x_to_sec(event.position().x())))
        self._drag_start = None
        self._drag_current = None
        self._drag_idx = -1
        self._snap_sec = None
        self.update()

    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x = event.position().x()
        if self._zone(event.position().y()) == "lane":
            idx = self._interval_at(x)
            if idx >= 0:
                self._selected_idx = idx
                self.interval_selected.emit(idx)
                self.interval_activated.emit(idx)
                self.playhead_moved.emit(self._intervals[idx].start_sec)
                self.update()
                return
        self.playhead_moved.emit(self._clamp_sec(self._x_to_sec(x)))

    # ------------------------------------------------------------------ menu
    def _show_context_menu(self, event, idx):
        menu = QMenu(self)
        if idx < 0:
            fit = menu.addAction("Zoom to fit")
            sel = menu.addAction("Zoom to selected interval")
            sel.setEnabled(0 <= self._selected_idx < len(self._intervals))
            chosen = menu.exec(event.globalPosition().toPoint())
            if chosen is fit:
                self.zoom_to_fit()
            elif chosen is sel:
                iv = self._intervals[self._selected_idx]
                self.zoom_to(iv.start_sec, iv.end_sec)
            return

        self._selected_idx = idx
        self.interval_selected.emit(idx)
        self.update()
        iv = self._intervals[idx]
        actions = {}
        if self._editable:
            relabel_menu = menu.addMenu("Set label")
            for label in self._label_choices:
                action = relabel_menu.addAction(label)
                action.setCheckable(True)
                action.setChecked(label == iv.label)
                actions[action] = ("label", label)
            if self._label_choices:
                relabel_menu.addSeparator()
            actions[relabel_menu.addAction("Other…")] = ("relabel", None)
        actions[menu.addAction("Go to start")] = ("seek", iv.start_sec)
        actions[menu.addAction("Go to end")] = ("seek", iv.end_sec)
        actions[menu.addAction("Zoom to interval")] = ("zoom", None)
        if self._editable:
            menu.addSeparator()
            inside = iv.start_sec < self._playhead_sec < iv.end_sec
            a = menu.addAction("Set start to playhead")
            a.setEnabled(self._playhead_sec < iv.end_sec - _MIN_INTERVAL_SEC)
            actions[a] = ("set-start", None)
            a = menu.addAction("Set end to playhead")
            a.setEnabled(self._playhead_sec > iv.start_sec + _MIN_INTERVAL_SEC)
            actions[a] = ("set-end", None)
            a = menu.addAction("Split at playhead")
            a.setEnabled(inside)
            actions[a] = ("split", None)
            actions[menu.addAction("Subdivide …")] = ("subdivide", None)
        menu.addSeparator()
        actions[menu.addAction("Export figure …")] = ("figure", None)
        actions[menu.addAction("Export mosaic …")] = ("mosaic", None)
        if self._editable:
            menu.addSeparator()
            actions[menu.addAction("Delete")] = ("delete", None)
        chosen = menu.exec(event.globalPosition().toPoint())
        if chosen is None or chosen not in actions:
            return
        kind, value = actions[chosen]
        if kind == "label":
            iv.label = value
            self.interval_relabelled.emit(idx, value)
        elif kind == "relabel":
            new_label, ok = QInputDialog.getText(self, "Relabel interval", "New label:", text=iv.label)
            if ok and new_label:
                iv.label = new_label
                self.interval_relabelled.emit(idx, new_label)
        elif kind == "seek":
            self.playhead_moved.emit(value)
        elif kind == "zoom":
            self.zoom_to(iv.start_sec, iv.end_sec)
        elif kind == "set-start":
            lower, _upper = self._neighbour_limits(idx)
            self.interval_resized.emit(idx, round(max(lower, self._playhead_sec), 3), iv.end_sec)
        elif kind == "set-end":
            _lower, upper = self._neighbour_limits(idx)
            self.interval_resized.emit(idx, iv.start_sec, round(min(upper, self._playhead_sec), 3))
        elif kind == "split":
            self.split_requested.emit(idx, self._playhead_sec)
        elif kind == "subdivide":
            self.subdivide_requested.emit(idx)
        elif kind == "figure":
            self.figure_requested.emit(idx)
        elif kind == "mosaic":
            self.mosaic_requested.emit(idx)
        elif kind == "delete":
            self.interval_deleted.emit(idx)
        self.update()
