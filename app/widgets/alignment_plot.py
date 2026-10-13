"""Plots of the PPG alignment step (pyqtgraph).

Top panel: the rPPG estimated from the video and the synthetic PPG shifted by
the current offset, on the segment's video timeline. The shaded band is the
analysis window used for the correlation; drag its edges (or their round
handles, which point outward) to change it.

Bottom panel: correlation between the two signals for every offset of the
search range. Drag the vertical marker or click a peak to choose an offset.

Mouse (toolbar toggle above the graphs):
- **Zoom** mode (default): drag to select the region to focus on, as in the other
  graphs; a mostly horizontal drag sets only the time range, a taller box also
  zooms the y-axis. **Undo zoom** steps back, **Full range** shows everything.
- **Shift PPG** mode: dragging the top graph shifts the synthetic PPG.
- In both modes Shift+drag shifts the PPG, Ctrl+drag pans, the wheel zooms time
  and a click on the top graph moves the playhead.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QButtonGroup,
    QGraphicsRectItem,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from .signal_plot import VIEW_HISTORY_LIMIT, ZOOM_MIN_PIXELS, ZOOM_Y_MIN_PIXELS, TimeAxisItem, series_color

MODE_ZOOM = "zoom"
MODE_SHIFT = "shift"


class _AlignmentViewBox(pg.ViewBox):
    """Drag = zoom selection or PPG shift (see module doc); click = seek on the signal plot."""

    def __init__(self, owner: "AlignmentPlot", signal_view: bool):
        super().__init__()
        self._owner = owner
        self._signal_view = signal_view
        self._drag = ""
        self._zoom_box: Optional[QGraphicsRectItem] = None
        self.setMouseEnabled(x=True, y=False)

    def mouseClickEvent(self, ev):
        if self._signal_view and ev.button() == Qt.MouseButton.LeftButton:
            ev.accept()
            self._owner.seek_requested.emit(float(self.mapSceneToView(ev.scenePos()).x()))
        else:
            super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        if ev.isStart():
            self._drag = self._owner.drag_action(ev, axis, self._signal_view)
            if self._drag == MODE_SHIFT:
                self._owner.begin_shift()
        if not self._drag:
            super().mouseDragEvent(ev, axis)  # pan
            return
        ev.accept()
        down, now = ev.buttonDownScenePos(), ev.scenePos()
        start, current = self.mapSceneToView(down), self.mapSceneToView(now)
        if self._drag == MODE_SHIFT:
            self._owner.shift_by(float(current.x() - start.x()), finished=ev.isFinish())
        else:
            low, high = sorted((float(start.x()), float(current.x())))
            use_y = abs(now.y() - down.y()) >= ZOOM_Y_MIN_PIXELS
            if use_y:
                y_low, y_high = sorted((float(start.y()), float(current.y())))
            else:
                y_low, y_high = (float(v) for v in self.viewRange()[1])
            self._draw_zoom_box(ev, low, high, y_low, y_high)
            if ev.isFinish() and abs(now.x() - down.x()) >= ZOOM_MIN_PIXELS:
                self._owner.zoom_view(self, (low, high), (y_low, y_high) if use_y else None)
        if ev.isFinish():
            self._drag = ""

    def _draw_zoom_box(self, ev, low: float, high: float, y_low: float, y_high: float) -> None:
        if ev.isStart():
            accent = theme.current().accent
            fill = QColor(accent)
            fill.setAlpha(50)
            self._zoom_box = QGraphicsRectItem()
            self._zoom_box.setPen(pg.mkPen(accent, width=1, style=Qt.PenStyle.DashLine))
            self._zoom_box.setBrush(pg.mkBrush(fill))
            self._zoom_box.setZValue(100)
            self.addItem(self._zoom_box, ignoreBounds=True)
        if self._zoom_box is not None:
            self._zoom_box.setRect(QRectF(low, y_low, high - low, y_high - y_low))
        if ev.isFinish() and self._zoom_box is not None:
            self.removeItem(self._zoom_box)
            self._zoom_box = None


class _WindowEdge(pg.InfiniteLine):
    """Edge of the analysis window: a wide grab band and a half-disc handle pointing outward."""

    HANDLE_RADIUS = 9  # pixels; also the half-width of the grab band around the line
    MIN_WIDTH_SEC = 0.5

    def __init__(self, outward: int):
        super().__init__(pos=0.0, angle=90, movable=True)
        self._outward = outward  # -1: start edge (handle on the left), +1: end edge
        # pyqtgraph sizes the line's clickable band from its marker size.
        self._maxMarkerSize = self.HANDLE_RADIUS
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setToolTip("Drag to change the analysis window")
        self.setZValue(40)

    def paint(self, p, *args):
        super().paint(p, *args)
        view = self.getViewBox()
        if view is None:
            return
        (x_low, x_high), (y_low, y_high) = view.viewRange()
        middle = (y_low + y_high) / 2.0
        transform = p.transform()
        centre = transform.map(self.mapFromView(QPointF(self.value(), middle)))
        outward = transform.map(self.mapFromView(
            QPointF(self.value() + self._outward * (x_high - x_low) * 1e-3, middle)))
        radius = self.HANDLE_RADIUS
        colour = self.currentPen.color()
        p.save()
        p.resetTransform()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(pg.mkPen(colour, width=1))
        p.setBrush(pg.mkBrush(colour))
        start_angle = -90 if outward.x() >= centre.x() else 90
        p.drawPie(QRectF(centre.x() - radius, centre.y() - radius, 2 * radius, 2 * radius),
                  start_angle * 16, 180 * 16)
        p.restore()


class AlignmentPlot(QWidget):
    seek_requested = pyqtSignal(float)
    offset_changed = pyqtSignal(float, bool)   # offset, finished (mouse released)
    window_changed = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._offset = 0.0
        self._drag_origin = 0.0
        self._duration = 0.0
        self._reference_t = np.empty(0)
        self._reference = np.empty(0)
        self._lag_extent: Optional[tuple[float, float]] = None
        self._history: list[tuple[pg.ViewBox, list[float], list[float], bool]] = []
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(self._build_toolbar())
        self._graphics = pg.GraphicsLayoutWidget()
        self._graphics.ci.setContentsMargins(2, 2, 6, 2)
        self._graphics.ci.setSpacing(6)
        layout.addWidget(self._graphics, 1)

        self._signal_plot = self._graphics.addPlot(
            row=0, col=0, viewBox=_AlignmentViewBox(self, signal_view=True),
            axisItems={"bottom": TimeAxisItem(orientation="bottom")},
        )
        self._signal_plot.setLabel("left", "Normalised")
        self._signal_plot.setLabel("bottom", "Time in segment")
        self._signal_plot.getViewBox().setAutoVisible(y=True)
        self._legend = self._signal_plot.addLegend(offset=(-8, 4))
        self._pulse_curve = self._signal_plot.plot(name="rPPG from video")
        self._reference_curve = self._signal_plot.plot(name="Synthetic PPG (shifted)")
        for curve in (self._pulse_curve, self._reference_curve):
            curve.setDownsampling(auto=True, method="peak")
            curve.setClipToView(True)
        # Analysis window: shaded band (not interactive) between two draggable edges.
        self._window_fill = pg.LinearRegionItem((0.0, 1.0), movable=False)
        self._window_fill.setZValue(-10)
        self._signal_plot.addItem(self._window_fill, ignoreBounds=True)
        self._window_start = _WindowEdge(outward=-1)
        self._window_end = _WindowEdge(outward=1)
        self._window_end.setValue(1.0)
        for edge in (self._window_start, self._window_end):
            edge.sigPositionChanged.connect(self._on_edge_moved)
            edge.sigPositionChangeFinished.connect(self._on_window_changed)
            self._signal_plot.addItem(edge, ignoreBounds=True)
        self._cursor = pg.InfiniteLine(pos=0.0, angle=90, movable=True)
        self._cursor.setZValue(50)
        self._cursor.sigDragged.connect(lambda line: self.seek_requested.emit(float(line.value())))
        self._signal_plot.addItem(self._cursor, ignoreBounds=True)

        self._lag_plot = self._graphics.addPlot(row=1, col=0, viewBox=_AlignmentViewBox(self, signal_view=False))
        self._lag_plot.setLabel("left", "Correlation")
        self._lag_plot.setLabel("bottom", "Offset (s) — positive: signals later")
        self._lag_plot.getViewBox().setAutoVisible(y=True)
        self._lag_curve = self._lag_plot.plot()
        self._peaks = pg.ScatterPlotItem(size=9, symbol="o")
        self._peaks.setToolTip("Click a peak to use that offset")
        self._peaks.sigClicked.connect(self._on_peak_clicked)
        self._lag_plot.addItem(self._peaks)
        self._zero_line = pg.InfiniteLine(pos=0.0, angle=0, movable=False)
        self._lag_plot.addItem(self._zero_line, ignoreBounds=True)
        self._applied_line = pg.InfiniteLine(pos=0.0, angle=90, movable=False,
                                             label="applied", labelOpts={"position": 0.92})
        self._applied_line.setVisible(False)
        self._lag_plot.addItem(self._applied_line, ignoreBounds=True)
        self._offset_line = pg.InfiniteLine(pos=0.0, angle=90, movable=True,
                                            label="{value:+.3f} s", labelOpts={"position": 0.08})
        self._offset_line.setZValue(50)
        self._offset_line.setToolTip("Drag to change the offset")
        self._offset_line.sigDragged.connect(
            lambda line: self._set_offset_from_mouse(float(line.value()), False))
        self._offset_line.sigPositionChangeFinished.connect(
            lambda line: self._set_offset_from_mouse(float(line.value()), True))
        self._lag_plot.addItem(self._offset_line, ignoreBounds=True)

        self._graphics.ci.layout.setRowStretchFactor(0, 3)
        self._graphics.ci.layout.setRowStretchFactor(1, 2)
        theme.notifier.changed.connect(lambda _palette: self._apply_theme())
        self._apply_theme()

    def _build_toolbar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(QLabel("Mouse drag:"))
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._zoom_btn = QToolButton()
        self._zoom_btn.setText("Zoom")
        self._zoom_btn.setToolTip(
            "Drag to select the region to focus on (a taller box also zooms the y-axis)."
        )
        self._shift_btn = QToolButton()
        self._shift_btn.setText("Shift PPG")
        self._shift_btn.setToolTip("Drag the top graph to shift the synthetic PPG (changes the offset).")
        for button in (self._zoom_btn, self._shift_btn):
            button.setCheckable(True)
            self._mode_group.addButton(button)
            row.addWidget(button)
        self._zoom_btn.setChecked(True)
        self._undo_btn = QToolButton()
        self._undo_btn.setText("Undo zoom")
        self._undo_btn.setToolTip("Return to the previous focus region")
        self._undo_btn.setEnabled(False)
        self._undo_btn.clicked.connect(self.undo_zoom)
        row.addWidget(self._undo_btn)
        full_btn = QToolButton()
        full_btn.setText("Full range")
        full_btn.setToolTip("Show the whole segment and every offset, with automatic y-scaling")
        full_btn.clicked.connect(self.reset_zoom)
        row.addWidget(full_btn)
        hint = QLabel("Shift+drag: shift PPG · Ctrl+drag: pan · wheel: zoom time")
        hint.setStyleSheet("color: #8a8a8a;")
        row.addWidget(hint)
        row.addStretch(1)
        return row

    # ------------------------------------------------------------------ mouse mode
    @property
    def drag_mode(self) -> str:
        return MODE_SHIFT if self._shift_btn.isChecked() else MODE_ZOOM

    def set_drag_mode(self, mode: str) -> None:
        (self._shift_btn if mode == MODE_SHIFT else self._zoom_btn).setChecked(True)

    def drag_action(self, ev, axis, signal_view: bool) -> str:
        """What a left-button drag does: shift, zoom, or '' (pan)."""
        if ev.button() != Qt.MouseButton.LeftButton or axis is not None:
            return ""
        modifiers = ev.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            return ""
        can_shift = signal_view and len(self._reference) > 0
        if can_shift and (modifiers & Qt.KeyboardModifier.ShiftModifier or self.drag_mode == MODE_SHIFT):
            return MODE_SHIFT
        return MODE_ZOOM

    # ------------------------------------------------------------------ zoom
    def zoom_view(self, view: pg.ViewBox, x_range: tuple[float, float],
                  y_range: Optional[tuple[float, float]]) -> None:
        x_now, y_now = view.viewRange()
        auto_y = bool(view.autoRangeEnabled()[1])
        self._history.append((view, list(x_now), list(y_now), auto_y))
        del self._history[:-VIEW_HISTORY_LIMIT]
        view.setXRange(*x_range, padding=0)
        if y_range is not None:
            view.disableAutoRange(axis=pg.ViewBox.YAxis)
            view.setYRange(*y_range, padding=0)
        self._undo_btn.setEnabled(True)

    def undo_zoom(self) -> None:
        if not self._history:
            return
        view, x_range, y_range, auto_y = self._history.pop()
        view.setXRange(*x_range, padding=0)
        if auto_y:
            view.enableAutoRange(axis=pg.ViewBox.YAxis)
        else:
            view.setYRange(*y_range, padding=0)
        self._undo_btn.setEnabled(bool(self._history))

    def reset_zoom(self) -> None:
        self._history.clear()
        self._undo_btn.setEnabled(False)
        self.show_full_range()
        if self._lag_extent is not None:
            self._lag_plot.setXRange(*self._lag_extent, padding=0.02)
        self._lag_plot.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)

    def show_full_range(self) -> None:
        # The margin leaves room for the window handles at the segment's ends.
        self._signal_plot.setXRange(0.0, max(1.0, self._duration), padding=0.025)
        self._signal_plot.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)

    # ------------------------------------------------------------------ data
    def clear(self) -> None:
        self._duration = 0.0
        self._pulse_curve.setData([], [])
        self._reference_t = np.empty(0)
        self._reference = np.empty(0)
        self._reference_curve.setData([], [])
        self.set_scan(None)

    def set_segment_duration(self, duration_sec: float) -> None:
        self._duration = max(0.0, float(duration_sec))
        view = self._signal_plot.getViewBox()
        view.setLimits(xMin=-5.0, xMax=self._duration + 5.0, minXRange=0.5)
        self._history.clear()
        self._undo_btn.setEnabled(False)
        self._update_window_bounds()
        self.show_full_range()

    def set_pulse(self, times: np.ndarray, values: np.ndarray) -> None:
        self._pulse_curve.setData(np.asarray(times), np.asarray(values))

    def set_reference(self, times: np.ndarray, values: np.ndarray) -> None:
        """Reference samples at segment time *without* offset (the plot adds it)."""
        self._reference_t = np.asarray(times, dtype=np.float64)
        self._reference = np.asarray(values, dtype=np.float64)
        self._redraw_reference()

    def set_scan(self, scan) -> None:
        if scan is None or not scan.valid:
            self._lag_curve.setData([], [])
            self._peaks.setData([], [])
            return
        self._lag_curve.setData(scan.lags, scan.correlation)
        lags = scan.peak_lags
        values = np.interp(lags, scan.lags, scan.correlation)
        keep = values > 0
        self._peaks.setData(lags[keep], values[keep])
        extent = (float(scan.lags[0]), float(scan.lags[-1]))
        if extent != self._lag_extent:  # keep the user's zoom while only the window changes
            self._lag_extent = extent
            self._lag_plot.setXRange(*extent, padding=0.02)
            self._lag_plot.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)

    # ------------------------------------------------------------------ offset
    @property
    def offset(self) -> float:
        return self._offset

    def set_offset(self, offset: float) -> None:
        """Programmatic change (no signal emitted)."""
        self._offset = float(offset)
        self._updating = True
        self._offset_line.setValue(self._offset)
        self._updating = False
        self._redraw_reference()

    def set_applied_offset(self, offset: Optional[float]) -> None:
        self._applied_line.setVisible(offset is not None)
        if offset is not None:
            self._applied_line.setValue(float(offset))

    def begin_shift(self) -> None:
        self._drag_origin = self._offset

    def shift_by(self, delta: float, finished: bool) -> None:
        self._set_offset_from_mouse(self._drag_origin + delta, finished)

    def _set_offset_from_mouse(self, offset: float, finished: bool) -> None:
        if self._updating:
            return
        self.set_offset(round(offset, 3))
        self.offset_changed.emit(self._offset, finished)

    def _on_peak_clicked(self, _item, points, *_args) -> None:
        if len(points):
            self._set_offset_from_mouse(float(points[0].pos().x()), True)

    def _redraw_reference(self) -> None:
        if len(self._reference_t):
            self._reference_curve.setData(self._reference_t + self._offset, self._reference)
        else:
            self._reference_curve.setData([], [])

    # ------------------------------------------------------------------ window / cursor
    def window(self) -> tuple[float, float]:
        return float(self._window_start.value()), float(self._window_end.value())

    def set_window(self, start: float, end: float) -> None:
        for edge, value in ((self._window_start, start), (self._window_end, end)):
            edge.blockSignals(True)
            edge.setBounds((None, None))  # the old limits would clamp the new position
            edge.setValue(float(value))
            edge.blockSignals(False)
        self._update_window_bounds()

    def _on_edge_moved(self, *_args) -> None:
        self._update_window_bounds()

    def _update_window_bounds(self) -> None:
        """Shade the window and keep the edges inside the segment and in order."""
        start, end = self.window()
        self._window_fill.setRegion((start, end))
        upper = self._duration if self._duration else None
        gap = _WindowEdge.MIN_WIDTH_SEC
        self._window_start.setBounds((0.0, end - gap))
        self._window_end.setBounds((start + gap, upper))

    def window_to_view(self) -> None:
        low, high = self._signal_plot.getViewBox().viewRange()[0]
        low, high = max(0.0, low), min(self._duration or high, high)
        if high > low:
            self.set_window(low, high)
            self.window_changed.emit(low, high)

    def zoom_to_window(self) -> None:
        low, high = self.window()
        self.zoom_view(self._signal_plot.getViewBox(), (low, high), None)

    def set_cursor(self, sec: float) -> None:
        self._cursor.setValue(float(sec))

    def _on_window_changed(self, *_args) -> None:
        low, high = self.window()
        low, high = max(0.0, low), min(self._duration, high) if self._duration else high
        self.window_changed.emit(low, high)

    # ------------------------------------------------------------------ theme
    def _apply_theme(self) -> None:
        palette = theme.current()
        self._graphics.setBackground(palette.background)
        for plot in (self._signal_plot, self._lag_plot):
            for name in ("left", "bottom"):
                axis = plot.getAxis(name)
                axis.setPen(pg.mkPen(palette.axis))
                axis.setTextPen(pg.mkPen(palette.text))
            plot.showGrid(x=True, y=True, alpha=0.18)
            plot.getAxis("left").setWidth(62)
        self._legend.setLabelTextColor(palette.text)
        self._legend.setBrush(pg.mkBrush(palette.panel + "cc"))
        self._pulse_curve.setPen(pg.mkPen(series_color(0), width=1.3))
        self._reference_curve.setPen(pg.mkPen(series_color(1), width=1.3))
        accent = QColor(palette.accent)
        accent.setAlpha(28)
        self._window_fill.setBrush(pg.mkBrush(accent))
        for line in self._window_fill.lines:
            line.setPen(pg.mkPen(None))
            line.setHoverPen(pg.mkPen(None))
        for edge in (self._window_start, self._window_end):
            edge.setPen(pg.mkPen(palette.accent, width=1.5, style=Qt.PenStyle.DashLine))
            edge.setHoverPen(pg.mkPen(palette.accent, width=3))
        self._cursor.setPen(pg.mkPen(palette.playhead, width=2))
        self._cursor.setHoverPen(pg.mkPen(palette.playhead, width=4))
        self._lag_curve.setPen(pg.mkPen(series_color(2), width=1.3))
        self._peaks.setBrush(pg.mkBrush(series_color(2)))
        self._peaks.setPen(pg.mkPen(palette.text))
        self._zero_line.setPen(pg.mkPen(palette.grid))
        self._applied_line.setPen(pg.mkPen(palette.muted_text, width=1, style=Qt.PenStyle.DashLine))
        self._applied_line.label.setColor(palette.muted_text)
        self._offset_line.setPen(pg.mkPen(series_color(1), width=2))
        self._offset_line.setHoverPen(pg.mkPen(series_color(1), width=4))
        self._offset_line.label.setColor(palette.text)
