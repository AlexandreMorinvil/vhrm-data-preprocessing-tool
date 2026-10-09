"""Interactive signal plot (pyqtgraph) with stacked, time-linked panels.

Mouse: wheel = zoom time, drag = pan time, click = move the playhead,
Ctrl+drag = zoom to the dragged range, Shift+drag = draw a new label interval
(when enabled). The red cursor can be dragged to scrub the videos.

Each panel has its own y-axis (never two scales on one axis). The x-axis is
the video timeline in seconds; with a time reference (the signal anchor),
samples are placed at ``timestamp - anchor`` so the plot matches exactly what
is exported for each labelled interval.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from ..signals import is_aux_signal_column
from ..timefmt import format_axis_time, format_hms_ms, nice_time_step

log = logging.getLogger(__name__)

pg.setConfigOptions(antialias=False, useOpenGL=False)

SERIES_COLORS_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SERIES_COLORS_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767")
TIME_AXIS_MODES = ("video", "seconds", "clock")


def series_color(index: int, dark: Optional[bool] = None) -> str:
    dark = theme.current().name == "dark" if dark is None else dark
    palette = SERIES_COLORS_DARK if dark else SERIES_COLORS_LIGHT
    return palette[index % len(palette)]


# ---------------------------------------------------------------------------
# Snapshot used by figure / CSV export
# ---------------------------------------------------------------------------

@dataclass
class SeriesData:
    name: str
    x: np.ndarray
    y: np.ndarray
    visible: bool = True
    color_index: int = 0
    dashed: bool = False


@dataclass
class PanelData:
    title: str
    y_label: str
    series: list[SeriesData] = field(default_factory=list)
    y_range: Optional[tuple[float, float]] = None


@dataclass
class PlotSnapshot:
    panels: list[PanelData]
    intervals: list[tuple[float, float, str, str]]
    cursor_sec: Optional[float]
    view: tuple[float, float]
    data_range: tuple[float, float]
    time_zero: Optional[pd.Timestamp]
    time_axis_mode: str = "video"


# ---------------------------------------------------------------------------
# pyqtgraph building blocks
# ---------------------------------------------------------------------------

class TimeAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = "video"
        self.time_zero: Optional[pd.Timestamp] = None

    def tickSpacing(self, minVal, maxVal, size):
        if self.mode == "seconds" or size <= 0 or maxVal <= minVal:
            return super().tickSpacing(minVal, maxVal, size)
        step = nice_time_step((maxVal - minVal) * 95.0 / size)
        return [(step, 0), (step / 5.0, 0)]

    def tickStrings(self, values, scale, spacing):
        if self.mode == "seconds":
            return super().tickStrings(values, scale, spacing)
        if self.mode == "clock" and self.time_zero is not None:
            out = []
            for value in values:
                stamp = self.time_zero + pd.Timedelta(seconds=float(value))
                text = stamp.strftime("%H:%M:%S")
                if spacing < 1.0:
                    text += f".{int(stamp.microsecond / 1000):03d}"[: 2 if spacing >= 0.1 else 4]
                out.append(text)
            return out
        return [format_axis_time(float(v), spacing) for v in values]


class _SignalViewBox(pg.ViewBox):
    """ViewBox with click-to-seek, Ctrl-drag zoom and Shift-drag interval drawing."""

    def __init__(self, owner: "SignalPlot", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._owner = owner
        self._band: Optional[pg.LinearRegionItem] = None
        self._band_mode = ""
        self.setMouseEnabled(x=True, y=False)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            ev.accept()
            self._owner.seek_requested.emit(float(self.mapSceneToView(ev.scenePos()).x()))
        else:
            super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        modifiers = ev.modifiers()
        wants_band = ev.button() == Qt.MouseButton.LeftButton and (
            modifiers & Qt.KeyboardModifier.ControlModifier
            or (modifiers & Qt.KeyboardModifier.ShiftModifier and self._owner.interval_drawing_enabled)
        )
        if not wants_band and self._band is None:
            super().mouseDragEvent(ev, axis)
            return
        ev.accept()
        start = float(self.mapSceneToView(ev.buttonDownScenePos()).x())
        current = float(self.mapSceneToView(ev.scenePos()).x())
        lo, hi = sorted((start, current))
        if ev.isStart():
            self._band_mode = "zoom" if modifiers & Qt.KeyboardModifier.ControlModifier else "interval"
            color = QColor(theme.current().accent if self._band_mode == "zoom" else "#ffd54a")
            color.setAlpha(60)
            self._band = pg.LinearRegionItem((lo, hi), movable=False, brush=pg.mkBrush(color))
            self.addItem(self._band, ignoreBounds=True)
        if self._band is not None:
            self._band.setRegion((lo, hi))
        if ev.isFinish():
            if self._band is not None:
                self.removeItem(self._band)
                self._band = None
            if self._band_mode == "zoom":
                self._owner._on_range_selected(lo, hi)
            elif hi - lo > 0:
                self._owner.interval_drawn.emit(lo, hi)
            self._band_mode = ""


class _Panel:
    def __init__(self, title: str, y_label: str, df: pd.DataFrame):
        self.title = title
        self.y_label = y_label
        self.df = df
        self.series: list[tuple[str, np.ndarray, np.ndarray]] = []
        self.plot: Optional[pg.PlotItem] = None
        self.lines: list[pg.PlotDataItem] = []
        self.cursor: Optional[pg.InfiniteLine] = None
        self.regions: list[pg.LinearRegionItem] = []
        self.auto_min = True
        self.auto_max = True
        self.y_min = 0.0
        self.y_max = 1.0


class _LineHandle:
    """Small adapter exposing a matplotlib-like API on a pyqtgraph curve."""

    def __init__(self, item: pg.PlotDataItem, name: str):
        self.item = item
        self.name = name

    def get_label(self) -> str:
        return self.name

    def get_visible(self) -> bool:
        return self.item.isVisible()

    def get_xdata(self):
        return self.item.xData

    def get_ydata(self):
        return self.item.yData


# ---------------------------------------------------------------------------
# The widget
# ---------------------------------------------------------------------------

class SignalPlot(QWidget):
    seek_requested = pyqtSignal(float)
    interval_drawn = pyqtSignal(float, float)
    view_range_changed = pyqtSignal(float, float)
    plot_area_changed = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.interval_drawing_enabled = False
        self._panels: list[_Panel] = []
        self._active_panel = 0
        self._cursor_sec: Optional[float] = None
        self._video_start_sec = 0.0
        self._video_duration_sec = 0.0
        self._time_zero: Optional[pd.Timestamp] = None
        self._intervals: list[tuple[float, float, str, str]] = []
        self._signal_visibility: dict[str, bool] = {}
        self._signal_checkboxes: dict[str, QCheckBox] = {}
        self._data_start_sec = 0.0
        self._data_end_sec = 0.0
        self._window_start_sec = 0.0
        self._updating_window_controls = False
        self._updating_y_controls = False
        self._syncing_view = False
        self._syncing_x = False
        self.figure_export_context = None  # callable returning extra export context

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        controls = QHBoxLayout()
        controls.setSpacing(4)
        self._back_btn = QToolButton()
        self._back_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowLeft))
        self._back_btn.setToolTip("Move the visible time window backward")
        self._back_btn.clicked.connect(lambda: self._pan_window(-1))
        controls.addWidget(self._back_btn)
        self._forward_btn = QToolButton()
        self._forward_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight))
        self._forward_btn.setToolTip("Move the visible time window forward")
        self._forward_btn.clicked.connect(lambda: self._pan_window(1))
        controls.addWidget(self._forward_btn)
        controls.addWidget(QLabel("Window:"))
        self._window_duration = QDoubleSpinBox()
        self._window_duration.setDecimals(2)
        self._window_duration.setRange(0.01, 86_400.0)
        self._window_duration.setSuffix(" s")
        self._window_duration.setToolTip("Duration of the visible time window")
        self._window_duration.valueChanged.connect(self._on_window_duration_changed)
        controls.addWidget(self._window_duration)
        self._window_slider = QSlider(Qt.Orientation.Horizontal)
        self._window_slider.setRange(0, 10_000)
        self._window_slider.setToolTip("Move the visible time window")
        self._window_slider.valueChanged.connect(self._on_window_position_changed)
        controls.addWidget(self._window_slider, 1)
        self._range_label = QLabel("Full range")
        self._range_label.setMinimumWidth(150)
        controls.addWidget(self._range_label)
        self._full_range_btn = QPushButton("Full range")
        self._full_range_btn.setToolTip("Show the complete signal")
        self._full_range_btn.clicked.connect(self._show_full_range)
        controls.addWidget(self._full_range_btn)
        self._axis_mode = QComboBox()
        self._axis_mode.addItem("Video time", "video")
        self._axis_mode.addItem("Seconds", "seconds")
        self._axis_mode.addItem("Clock (UTC)", "clock")
        self._axis_mode.setToolTip("Time axis format")
        self._axis_mode.currentIndexChanged.connect(self._on_axis_mode)
        controls.addWidget(self._axis_mode)
        self._export_btn = QToolButton()
        self._export_btn.setText("Export")
        self._export_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        export_menu = QMenu(self._export_btn)
        export_menu.addAction("Figure (PNG / SVG / PDF)…", self.open_figure_export)
        export_menu.addAction("Plotted data (CSV)…", self.export_csv_dialog)
        self._export_btn.setMenu(export_menu)
        controls.addWidget(self._export_btn)
        layout.addLayout(controls)

        y_controls = QHBoxLayout()
        y_controls.setSpacing(4)
        y_controls.addWidget(QLabel("Y scale:"))
        self._y_panel = QComboBox()
        self._y_panel.setToolTip("Panel whose y-axis the controls apply to")
        self._y_panel.currentIndexChanged.connect(self._on_y_panel_changed)
        y_controls.addWidget(self._y_panel)
        self._auto_y_min = QCheckBox("Auto min")
        self._auto_y_min.setChecked(True)
        self._auto_y_min.toggled.connect(self._on_y_controls_changed)
        y_controls.addWidget(self._auto_y_min)
        self._y_min = QDoubleSpinBox()
        self._configure_y_spin(self._y_min)
        self._y_min.valueChanged.connect(self._on_y_controls_changed)
        y_controls.addWidget(self._y_min)
        self._zero_y_min_btn = QPushButton("Min 0")
        self._zero_y_min_btn.setToolTip("Set the displayed Y-axis minimum to zero")
        self._zero_y_min_btn.clicked.connect(self._set_zero_y_min)
        y_controls.addWidget(self._zero_y_min_btn)
        self._auto_y_max = QCheckBox("Auto max")
        self._auto_y_max.setChecked(True)
        self._auto_y_max.toggled.connect(self._on_y_controls_changed)
        y_controls.addWidget(self._auto_y_max)
        self._y_max = QDoubleSpinBox()
        self._configure_y_spin(self._y_max)
        self._y_max.setValue(1.0)
        self._y_max.valueChanged.connect(self._on_y_controls_changed)
        y_controls.addWidget(self._y_max)
        self._hr_y_max_btn = QPushButton("Max 220")
        self._hr_y_max_btn.setToolTip(
            "Set a 220 BPM display ceiling. This is not a universal physiological maximum."
        )
        self._hr_y_max_btn.clicked.connect(self._set_hr_y_max)
        y_controls.addWidget(self._hr_y_max_btn)
        y_controls.addSpacing(12)
        self._signal_controls_widget = QWidget()
        self._signal_controls = QHBoxLayout(self._signal_controls_widget)
        self._signal_controls.setContentsMargins(0, 0, 0, 0)
        y_controls.addWidget(self._signal_controls_widget)
        y_controls.addStretch()
        layout.addLayout(y_controls)

        self._graphics = pg.GraphicsLayoutWidget()
        self._graphics.ci.setContentsMargins(2, 2, 6, 2)
        self._graphics.ci.setSpacing(2)
        layout.addWidget(self._graphics, 1)

        self._readout = QLabel(" ")
        self._readout.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 11px;")
        self._readout.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._readout)

        self._y_refresh = QTimer(self)
        self._y_refresh.setSingleShot(True)
        self._y_refresh.setInterval(15)
        self._y_refresh.timeout.connect(self._apply_y_scale)
        self._mouse_proxy = pg.SignalProxy(self._graphics.scene().sigMouseMoved, rateLimit=30, slot=self._on_mouse_moved)
        theme.notifier.changed.connect(lambda _p: self._redraw())

        self._set_window_controls_enabled(False)
        self._update_y_control_states()
        self._build_empty()

    @staticmethod
    def _configure_y_spin(spin: QDoubleSpinBox) -> None:
        spin.setDecimals(3)
        spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
        spin.setSingleStep(1.0)
        spin.setMinimumWidth(95)

    # ------------------------------------------------------------------ public API
    def set_time_zero(self, time_zero) -> None:
        """Set the absolute time of video time 0 (``None``: first sample)."""
        if time_zero is None:
            self._time_zero = None
        else:
            stamp = pd.Timestamp(time_zero)
            self._time_zero = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")

    def set_data(self, df, video_start_sec=0.0, video_duration_sec=0.0, title: str = "", y_label: str = "Value"):
        self.set_panels([(title, df, y_label)], video_start_sec=video_start_sec,
                        video_duration_sec=video_duration_sec)

    def set_panels(self, panels, video_start_sec: float = 0.0, video_duration_sec: float = 0.0) -> None:
        """Show stacked panels: ``[(title, dataframe, y_label), ...]``."""
        previous = {p.title: p for p in self._panels}
        self._panels = []
        for title, df, y_label in panels:
            if df is None or df.empty:
                continue
            panel = _Panel(title, y_label, df)
            old = previous.get(title)
            if old is not None:
                panel.auto_min, panel.auto_max = old.auto_min, old.auto_max
                panel.y_min, panel.y_max = old.y_min, old.y_max
            self._panels.append(panel)
        self._video_start_sec = float(video_start_sec)
        self._video_duration_sec = float(video_duration_sec)
        for panel in self._panels:
            panel.series = self._series_columns(panel.df)
        self._active_panel = min(self._active_panel, max(0, len(self._panels) - 1))
        keep_view = (self._data_end_sec > self._data_start_sec
                     and self._visible_duration() < self._full_duration() - 1e-6)
        old_window = (self._window_start_sec, self._visible_duration())
        self._reset_window_bounds()
        self._redraw()
        if keep_view:
            self._window_start_sec = old_window[0]
            self._updating_window_controls = True
            self._window_duration.setValue(old_window[1])
            self._updating_window_controls = False
            self._apply_time_window()

    @property
    def panel_count(self) -> int:
        return len(self._panels)

    def clear(self) -> None:
        self._panels = []
        self._data_start_sec = 0.0
        self._data_end_sec = 0.0
        self._window_start_sec = 0.0
        self._set_window_controls_enabled(False)
        self._range_label.setText("Full range")
        self._sync_signal_controls([])
        self._redraw()

    def set_intervals(self, intervals) -> None:
        self._intervals = []
        for interval in intervals:
            start_sec = getattr(interval, "start_sec", None)
            end_sec = getattr(interval, "end_sec", None)
            if start_sec is None or end_sec is None:
                continue
            start_sec, end_sec = float(start_sec), float(end_sec)
            if end_sec <= start_sec:
                continue
            self._intervals.append((
                start_sec, end_sec,
                str(getattr(interval, "color", "#4488cc")),
                str(getattr(interval, "label", "")),
            ))
        self._draw_intervals()

    def set_cursor(self, time_sec: float) -> None:
        self._cursor_sec = float(time_sec)
        for panel in self._panels:
            if panel.cursor is not None:
                panel.cursor.blockSignals(True)
                panel.cursor.setValue(self._cursor_sec)
                panel.cursor.blockSignals(False)
                if not panel.cursor.isVisible():
                    panel.cursor.setVisible(True)

    def set_view_range(self, start: float, end: float) -> None:
        """Programmatic x-range change (does not re-emit ``view_range_changed``)."""
        if not self._panels or end <= start:
            return
        self._window_start_sec = start
        self._updating_window_controls = True
        self._window_duration.setValue(min(end - start, self._full_duration()))
        self._updating_window_controls = False
        self._syncing_view = True
        try:
            self._apply_time_window()
        finally:
            self._syncing_view = False

    def x_range(self) -> tuple[float, float]:
        if not self._panels or self._panels[0].plot is None:
            return (0.0, 1.0)
        lo, hi = self._panels[0].plot.getViewBox().viewRange()[0]
        return float(lo), float(hi)

    def y_range(self, panel: int = 0) -> tuple[float, float]:
        if panel >= len(self._panels) or self._panels[panel].plot is None:
            return (0.0, 1.0)
        lo, hi = self._panels[panel].plot.getViewBox().viewRange()[1]
        return float(lo), float(hi)

    def legend_labels(self, panel: int = 0) -> list[str]:
        if panel >= len(self._panels) or self._panels[panel].plot is None:
            return []
        legend = self._panels[panel].plot.legend
        if legend is None:
            return []
        return [label.text for _sample, label in legend.items]

    @property
    def _signal_lines(self) -> list[_LineHandle]:
        return [
            _LineHandle(line, line.name())
            for panel in self._panels for line in panel.lines
        ]

    def plot_area_margins(self) -> tuple[int, int]:
        """Left/right widget-pixel margins of the data area (for alignment)."""
        if not self._panels or self._panels[0].plot is None:
            return 0, 0
        vb = self._panels[0].plot.getViewBox()
        rect = vb.mapRectToScene(vb.boundingRect())
        view_rect = self._graphics.mapFromScene(rect).boundingRect()
        left = self._graphics.mapTo(self, view_rect.topLeft()).x()
        right = self.width() - (left + view_rect.width())
        return int(left), int(right)

    # ------------------------------------------------------------------ drawing
    def _build_empty(self) -> None:
        self._graphics.clear()
        pal = theme.current()
        self._graphics.setBackground(pal.background)
        plot = self._graphics.addPlot(
            viewBox=_SignalViewBox(self),
            axisItems={"bottom": TimeAxisItem(orientation="bottom")},
        )
        self._style_plot(plot)
        plot.setLabel("left", "Value")
        plot.setLabel("bottom", "Time (from video start)")
        text = pg.TextItem("No signal loaded", color=pal.muted_text, anchor=(0.5, 0.5))
        plot.addItem(text)
        text.setPos(0.5, 0.5)
        plot.setXRange(0, 1, padding=0)
        plot.setYRange(0, 1, padding=0)

    def _style_plot(self, plot: pg.PlotItem) -> None:
        pal = theme.current()
        for name in ("left", "bottom"):
            axis = plot.getAxis(name)
            axis.setPen(pg.mkPen(pal.axis))
            axis.setTextPen(pg.mkPen(pal.text))
        plot.showGrid(x=True, y=True, alpha=0.18)
        plot.getAxis("left").setWidth(62)

    def _series_columns(self, df: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
        out = []
        zero = self._time_zero
        if "sensor_id" in df.columns:
            for sid in df["sensor_id"].unique():
                sub = df[df["sensor_id"] == sid]
                t0 = zero if zero is not None else sub["timestamp_utc"].iloc[0]
                rel = (sub["timestamp_utc"] - t0).dt.total_seconds().to_numpy() + self._video_start_sec
                out.append((str(sid), rel, pd.to_numeric(sub["value"], errors="coerce").to_numpy(float)))
            return out
        value_cols = [
            c for c in df.columns
            if c != "timestamp_utc" and not is_aux_signal_column(str(c))
        ]
        t0 = zero if zero is not None else df["timestamp_utc"].iloc[0]
        rel_all = (df["timestamp_utc"] - t0).dt.total_seconds().to_numpy() + self._video_start_sec
        for col in value_cols:
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            valid = np.isfinite(values) & np.isfinite(rel_all)
            if not valid.any():
                continue
            out.append((str(col), rel_all[valid], values[valid]))
        return out

    def _redraw(self) -> None:
        if not self._panels:
            self._build_empty()
            self._sync_signal_controls([])
            self._sync_y_panel_combo()
            return
        pal = theme.current()
        self._graphics.clear()
        self._graphics.setBackground(pal.background)
        names: list[str] = []
        color_index = 0
        first_plot: Optional[pg.PlotItem] = None
        axis_mode = self._axis_mode.currentData() or "video"
        for index, panel in enumerate(self._panels):
            axis = TimeAxisItem(orientation="bottom")
            axis.mode = axis_mode
            axis.time_zero = self._clock_zero()
            plot = self._graphics.addPlot(row=index, col=0, viewBox=_SignalViewBox(self), axisItems={"bottom": axis})
            self._style_plot(plot)
            panel.plot = plot
            panel.lines = []
            if panel.title:
                plot.setTitle(panel.title, color=pal.text, size="9pt")
            plot.setLabel("left", panel.y_label)
            if index == len(self._panels) - 1:
                plot.setLabel("bottom", "Time (from video start)" if axis_mode != "clock" else "Clock time (UTC)")
            else:
                plot.getAxis("bottom").setStyle(showValues=False)
            if first_plot is None:
                first_plot = plot
            series = panel.series
            if len(series) > 1:
                plot.addLegend(offset=(-8, 4), labelTextColor=pal.text, brush=pg.mkBrush(pal.panel + "cc"))
            for name, x, y in series:
                dashed = name == "averaged"
                pen = pg.mkPen(series_color(color_index), width=1.2 if not dashed else 1.6,
                               style=Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
                line = plot.plot(x, y, pen=pen, name=name)
                line.setDownsampling(auto=True, method="peak")
                line.setClipToView(True)
                line.setVisible(self._signal_visibility.get(name, True))
                panel.lines.append(line)
                names.append(name)
                color_index += 1
            panel.cursor = pg.InfiniteLine(
                pos=self._cursor_sec if self._cursor_sec is not None else self._data_start_sec,
                angle=90, movable=True, pen=pg.mkPen(pal.playhead, width=2),
                hoverPen=pg.mkPen(pal.playhead, width=4),
            )
            panel.cursor.setVisible(self._cursor_sec is not None)
            panel.cursor.setZValue(50)
            panel.cursor.sigDragged.connect(self._on_cursor_dragged)
            plot.addItem(panel.cursor, ignoreBounds=True)
            vb = plot.getViewBox()
            vb.disableAutoRange()
            vb.setLimits(xMin=self._data_start_sec, xMax=self._data_end_sec,
                         minXRange=0.01)
            vb.sigXRangeChanged.connect(lambda _vb, rng, i=index: self._on_view_x_changed(i, rng))
            if index == 0:
                vb.sigResized.connect(self._emit_plot_area)
        self._sync_signal_controls(names)
        self._sync_y_panel_combo()
        self._update_legends()
        self._draw_intervals()
        self._apply_time_window(redraw=False)
        QTimer.singleShot(0, self._emit_plot_area)

    def _clock_zero(self) -> Optional[pd.Timestamp]:
        if self._time_zero is not None:
            return self._time_zero - pd.Timedelta(seconds=self._video_start_sec)
        for panel in self._panels:
            if "timestamp_utc" in panel.df.columns and not panel.df.empty:
                return panel.df["timestamp_utc"].iloc[0] - pd.Timedelta(seconds=self._video_start_sec)
        return None

    def _draw_intervals(self) -> None:
        for panel in self._panels:
            if panel.plot is None:
                continue
            for region in panel.regions:
                panel.plot.removeItem(region)
            panel.regions = []
            for start, end, color, label in self._intervals:
                fill = QColor(color)
                fill.setAlpha(46)
                region = pg.LinearRegionItem((start, end), movable=False, brush=pg.mkBrush(fill),
                                             pen=pg.mkPen(None))
                region.setZValue(-10)
                region.setToolTip(label)
                for line in region.lines:
                    line.setPen(pg.mkPen(None))
                panel.plot.addItem(region, ignoreBounds=True)
                panel.regions.append(region)

    def _emit_plot_area(self, *_args) -> None:
        left, right = self.plot_area_margins()
        self.plot_area_changed.emit(left, right)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._emit_plot_area)

    def _on_axis_mode(self, _index: int) -> None:
        self._redraw()

    # ------------------------------------------------------------------ signal toggles
    def _sync_signal_controls(self, signal_names: list[str]) -> None:
        while self._signal_controls.count():
            item = self._signal_controls.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._signal_checkboxes = {}
        self._signal_visibility = {
            name: self._signal_visibility.get(name, True) for name in signal_names
        } | {k: v for k, v in self._signal_visibility.items() if k not in signal_names}
        if signal_names:
            self._signal_controls.addWidget(QLabel("Signals:"))
        for index, name in enumerate(signal_names):
            checkbox = QCheckBox(name)
            checkbox.setChecked(self._signal_visibility.get(name, True))
            checkbox.setStyleSheet(f"QCheckBox {{ color: {series_color(index)}; font-weight: bold; }}")
            checkbox.toggled.connect(
                lambda checked, signal_name=name: self._set_signal_visible(signal_name, checked)
            )
            self._signal_checkboxes[name] = checkbox
            self._signal_controls.addWidget(checkbox)
        self._signal_controls_widget.setVisible(bool(signal_names))

    def _set_signal_visible(self, signal_name: str, visible: bool) -> None:
        self._signal_visibility[signal_name] = visible
        for panel in self._panels:
            for line in panel.lines:
                if line.name() == signal_name:
                    line.setVisible(visible)
        self._update_legends()
        self._apply_y_scale()

    def _update_legends(self) -> None:
        for panel in self._panels:
            legend = panel.plot.legend if panel.plot is not None else None
            if legend is None:
                continue
            legend.clear()
            for line in panel.lines:
                if line.isVisible():
                    legend.addItem(line, line.name())
            legend.setVisible(len(legend.items) > 0)

    # ------------------------------------------------------------------ y scale
    def _sync_y_panel_combo(self) -> None:
        self._y_panel.blockSignals(True)
        self._y_panel.clear()
        for panel in self._panels:
            self._y_panel.addItem(panel.title or panel.y_label or "Signal")
        self._y_panel.setCurrentIndex(self._active_panel if self._panels else -1)
        self._y_panel.setVisible(len(self._panels) > 1)
        self._y_panel.blockSignals(False)
        self._load_y_controls()

    def _on_y_panel_changed(self, index: int) -> None:
        if index < 0:
            return
        self._active_panel = index
        self._load_y_controls()

    def _load_y_controls(self) -> None:
        if not self._panels:
            return
        panel = self._panels[self._active_panel]
        self._updating_y_controls = True
        self._auto_y_min.setChecked(panel.auto_min)
        self._auto_y_max.setChecked(panel.auto_max)
        self._y_min.setValue(panel.y_min)
        self._y_max.setValue(panel.y_max)
        self._updating_y_controls = False
        self._update_y_control_states()

    def _on_y_controls_changed(self, *_args) -> None:
        if self._updating_y_controls:
            return
        if self._panels:
            panel = self._panels[self._active_panel]
            panel.auto_min = self._auto_y_min.isChecked()
            panel.auto_max = self._auto_y_max.isChecked()
            panel.y_min = self._y_min.value()
            panel.y_max = self._y_max.value()
        self._update_y_control_states()
        self._apply_y_scale()

    def _update_y_control_states(self) -> None:
        self._y_min.setEnabled(not self._auto_y_min.isChecked())
        self._y_max.setEnabled(not self._auto_y_max.isChecked())

    def _set_zero_y_min(self) -> None:
        self._auto_y_min.setChecked(False)
        self._y_min.setValue(0.0)
        self._apply_y_scale()

    def _set_hr_y_max(self) -> None:
        self._auto_y_max.setChecked(False)
        self._y_max.setValue(220.0)
        self._apply_y_scale()

    def _visible_auto_y_bounds(self, panel: _Panel) -> Optional[tuple[float, float]]:
        window_start = self._window_start_sec
        window_end = window_start + self._visible_duration()
        mins, maxs = [], []
        for line in panel.lines:
            if not line.isVisible() or line.xData is None:
                continue
            x, y = line.xData, line.yData
            lo = int(np.searchsorted(x, window_start, side="left"))
            hi = int(np.searchsorted(x, window_end, side="right"))
            segment = y[lo:hi]
            segment = segment[np.isfinite(segment)]
            if segment.size:
                mins.append(float(segment.min()))
                maxs.append(float(segment.max()))
        if not mins:
            return None
        value_min, value_max = min(mins), max(maxs)
        span = value_max - value_min
        margin = span * 0.05 if span > 0 else max(abs(value_min) * 0.05, 0.5)
        return value_min - margin, value_max + margin

    def _apply_y_scale(self) -> None:
        for index, panel in enumerate(self._panels):
            if panel.plot is None:
                continue
            bounds = self._visible_auto_y_bounds(panel)
            if bounds is None:
                continue
            auto_min, auto_max = bounds
            if panel.auto_min:
                panel.y_min = auto_min
            if panel.auto_max:
                panel.y_max = auto_max
            y_min = auto_min if panel.auto_min else panel.y_min
            y_max = auto_max if panel.auto_max else panel.y_max
            if y_max <= y_min:
                y_max = y_min + max(abs(y_min) * 0.01, 0.001)
            panel.plot.getViewBox().setYRange(y_min, y_max, padding=0)
            if index == self._active_panel:
                self._updating_y_controls = True
                if panel.auto_min:
                    self._y_min.setValue(auto_min)
                if panel.auto_max:
                    self._y_max.setValue(auto_max)
                self._updating_y_controls = False

    # ------------------------------------------------------------------ time window
    def _reset_window_bounds(self) -> None:
        if not self._panels:
            self._set_window_controls_enabled(False)
            return
        starts, ends = [], []
        for panel in self._panels:
            for _name, x, _y in panel.series:
                if len(x):
                    starts.append(float(np.nanmin(x)))
                    ends.append(float(np.nanmax(x)))
        data_start = self._video_start_sec
        signal_end = max(ends) if ends else data_start
        if self._time_zero is None and starts:
            signal_duration = signal_end - min(starts)
            full_duration = max(float(self._video_duration_sec), signal_duration, 0.01)
        else:
            data_start = min([data_start, *starts]) if starts else data_start
            full_duration = max(self._video_start_sec + self._video_duration_sec, signal_end) - data_start
            full_duration = max(full_duration, 0.01)
        self._data_start_sec = float(data_start)
        self._data_end_sec = self._data_start_sec + full_duration
        self._window_start_sec = self._data_start_sec
        self._updating_window_controls = True
        self._window_duration.setMaximum(full_duration)
        self._window_duration.setValue(full_duration)
        self._window_slider.setValue(0)
        self._updating_window_controls = False
        self._set_window_controls_enabled(True)
        self._update_window_controls()

    def _set_window_controls_enabled(self, enabled: bool) -> None:
        self._window_duration.setEnabled(enabled)
        self._window_slider.setEnabled(enabled)
        self._full_range_btn.setEnabled(enabled)
        self._back_btn.setEnabled(False)
        self._forward_btn.setEnabled(False)

    def _on_window_duration_changed(self, duration_sec: float) -> None:
        if self._updating_window_controls:
            return
        self._window_start_sec = min(self._window_start_sec, self._data_end_sec - duration_sec)
        self._apply_time_window()

    def _on_window_position_changed(self, position: int) -> None:
        if self._updating_window_controls:
            return
        travel = max(0.0, self._full_duration() - self._visible_duration())
        self._window_start_sec = self._data_start_sec + travel * position / 10_000
        self._apply_time_window()

    def _pan_window(self, direction: int) -> None:
        self._window_start_sec += direction * self._visible_duration() * 0.8
        self._apply_time_window()

    def _show_full_range(self) -> None:
        self._window_start_sec = self._data_start_sec
        self._updating_window_controls = True
        self._window_duration.setValue(self._full_duration())
        self._updating_window_controls = False
        self._apply_time_window()

    def _on_range_selected(self, start_sec: float, end_sec: float) -> None:
        start_sec = max(self._data_start_sec, min(start_sec, self._data_end_sec))
        end_sec = max(self._data_start_sec, min(end_sec, self._data_end_sec))
        if end_sec - start_sec < 0.01:
            return
        self._window_start_sec = start_sec
        self._updating_window_controls = True
        self._window_duration.setValue(end_sec - start_sec)
        self._updating_window_controls = False
        self._apply_time_window()

    def _full_duration(self) -> float:
        return max(0.01, self._data_end_sec - self._data_start_sec)

    def _visible_duration(self) -> float:
        return min(self._window_duration.value(), self._full_duration())

    def _apply_time_window(self, redraw: bool = True) -> None:
        if not self._panels:
            return
        duration = self._visible_duration()
        latest_start = self._data_end_sec - duration
        self._window_start_sec = max(self._data_start_sec, min(self._window_start_sec, latest_start))
        self._syncing_x = True
        try:
            for panel in self._panels:
                if panel.plot is not None:
                    panel.plot.getViewBox().setXRange(
                        self._window_start_sec, self._window_start_sec + duration, padding=0)
        finally:
            self._syncing_x = False
        self._apply_y_scale()
        self._update_window_controls()
        if not self._syncing_view:
            self.view_range_changed.emit(self._window_start_sec, self._window_start_sec + duration)

    def _on_view_x_changed(self, source: int, x_range) -> None:
        if self._syncing_x:
            return
        lo, hi = float(x_range[0]), float(x_range[1])
        if hi <= lo:
            return
        self._syncing_x = True
        try:
            for index, panel in enumerate(self._panels):
                if index != source and panel.plot is not None:
                    panel.plot.getViewBox().setXRange(lo, hi, padding=0)
        finally:
            self._syncing_x = False
        self._window_start_sec = lo
        self._updating_window_controls = True
        self._window_duration.setValue(min(hi - lo, self._full_duration()))
        self._updating_window_controls = False
        self._update_window_controls()
        self._y_refresh.start()
        if not self._syncing_view:
            self.view_range_changed.emit(lo, hi)

    def _update_window_controls(self) -> None:
        duration = self._visible_duration()
        end_sec = self._window_start_sec + duration
        travel = max(0.0, self._full_duration() - duration)
        position = (
            round((self._window_start_sec - self._data_start_sec) / travel * 10_000)
            if travel > 0 else 0
        )
        tolerance = 1e-6
        self._updating_window_controls = True
        self._window_slider.setValue(position)
        self._updating_window_controls = False
        self._window_slider.setEnabled(travel > tolerance)
        self._back_btn.setEnabled(self._window_start_sec > self._data_start_sec + tolerance)
        self._forward_btn.setEnabled(end_sec < self._data_end_sec - tolerance)
        self._full_range_btn.setEnabled(duration < self._full_duration() - tolerance)
        self._range_label.setText(f"{format_hms_ms(self._window_start_sec)} – {format_hms_ms(end_sec)}")

    # ------------------------------------------------------------------ interaction
    def _on_cursor_dragged(self, line: pg.InfiniteLine) -> None:
        self.seek_requested.emit(float(line.value()))

    def _on_mouse_moved(self, event) -> None:
        pos = event[0]
        for panel in self._panels:
            if panel.plot is None or not panel.plot.sceneBoundingRect().contains(pos):
                continue
            x = float(panel.plot.getViewBox().mapSceneToView(pos).x())
            parts = [format_hms_ms(max(0.0, x))]
            for line in panel.lines:
                if not line.isVisible() or line.xData is None or len(line.xData) == 0:
                    continue
                i = int(np.clip(np.searchsorted(line.xData, x), 1, len(line.xData) - 1))
                if abs(line.xData[i - 1] - x) < abs(line.xData[i] - x):
                    i -= 1
                parts.append(f"{line.name()}: {line.yData[i]:.4g}")
            labels = [label for start, end, _c, label in self._intervals if start <= x <= end]
            if labels:
                parts.append(f"[{labels[0]}]")
            self._readout.setText("   ".join(parts))
            return

    # ------------------------------------------------------------------ export
    def snapshot(self) -> PlotSnapshot:
        panels = []
        index = 0
        for panel in self._panels:
            series = []
            for line in panel.lines:
                series.append(SeriesData(
                    name=line.name(), x=np.asarray(line.xData), y=np.asarray(line.yData),
                    visible=line.isVisible(), color_index=index, dashed=line.name() == "averaged",
                ))
                index += 1
            y_range = None
            if not (panel.auto_min and panel.auto_max):
                y_range = (panel.y_min, panel.y_max)
            panels.append(PanelData(panel.title, panel.y_label, series, y_range))
        return PlotSnapshot(
            panels=panels,
            intervals=list(self._intervals),
            cursor_sec=self._cursor_sec,
            view=(self._window_start_sec, self._window_start_sec + self._visible_duration()),
            data_range=(self._data_start_sec, self._data_end_sec),
            time_zero=self._clock_zero(),
            time_axis_mode=self._axis_mode.currentData() or "video",
        )

    def open_figure_export(self, initial_range: Optional[tuple[float, float]] = None) -> None:
        if not self._panels:
            QMessageBox.information(self, "Export figure", "There is no signal to export.")
            return
        from .figure_export_dialog import FigureExportDialog

        context = self.figure_export_context() if callable(self.figure_export_context) else {}
        dialog = FigureExportDialog(self.snapshot(), context=context, initial_range=initial_range, parent=self)
        dialog.exec()

    def export_csv_dialog(self) -> None:
        if not self._panels:
            QMessageBox.information(self, "Export data", "There is no signal to export.")
            return
        from ..visual_export import export_plot_csv

        context = self.figure_export_context() if callable(self.figure_export_context) else {}
        default_dir = context.get("output_directory", "") if context else ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export plotted data", str(default_dir) + "/plotted_data.csv" if default_dir else "plotted_data.csv",
            "CSV (*.csv)",
        )
        if not path:
            return
        snapshot = self.snapshot()
        try:
            rows = export_plot_csv(snapshot, path, snapshot.view)
        except Exception as exc:
            QMessageBox.critical(self, "Export data", f"Export failed:\n{exc}")
            return
        QMessageBox.information(self, "Export data", f"Wrote {rows:,} rows to\n{path}")
