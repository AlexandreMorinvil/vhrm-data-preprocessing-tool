from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..signals import is_aux_signal_column

log = logging.getLogger(__name__)


class SignalPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._fig = Figure(figsize=(8, 2.5), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        controls = QHBoxLayout()
        self._back_btn = QToolButton()
        self._back_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowLeft))
        self._back_btn.setToolTip("Move the selected time window backward")
        self._back_btn.clicked.connect(lambda: self._pan_window(-1))
        controls.addWidget(self._back_btn)

        self._forward_btn = QToolButton()
        self._forward_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight))
        self._forward_btn.setToolTip("Move the selected time window forward")
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
        self._window_slider.setToolTip("Move the selected time window")
        self._window_slider.valueChanged.connect(self._on_window_position_changed)
        controls.addWidget(self._window_slider, 1)

        self._range_label = QLabel("Full range")
        self._range_label.setMinimumWidth(125)
        controls.addWidget(self._range_label)

        self._full_range_btn = QPushButton("Full range")
        self._full_range_btn.setToolTip("Show the complete signal")
        self._full_range_btn.clicked.connect(self._show_full_range)
        controls.addWidget(self._full_range_btn)
        layout.addLayout(controls)

        y_controls = QHBoxLayout()
        y_controls.addWidget(QLabel("Y scale:"))
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
        y_controls.addStretch()
        layout.addLayout(y_controls)

        self._signal_controls_widget = QWidget()
        self._signal_controls = QHBoxLayout(self._signal_controls_widget)
        self._signal_controls.setContentsMargins(0, 0, 0, 0)
        self._signal_controls.addWidget(QLabel("Displayed signals:"))
        self._signal_controls.addStretch()
        layout.addWidget(self._signal_controls_widget)
        layout.addWidget(self._canvas)

        self._cursor_line = None
        self._cursor_sec: Optional[float] = None
        self._df: Optional[pd.DataFrame] = None
        self._video_start_sec: float = 0.0
        self._video_duration_sec: float = 0.0
        self._sensor_ids: list[str] = []
        self._signal_lines = []
        self._signal_checkboxes: dict[str, QCheckBox] = {}
        self._signal_visibility: dict[str, bool] = {}
        self._intervals: list[tuple[float, float, str]] = []
        self._data_start_sec: float = 0.0
        self._data_end_sec: float = 0.0
        self._window_start_sec: float = 0.0
        self._updating_window_controls = False
        self._updating_y_controls = False
        self._range_selector = SpanSelector(
            self._ax,
            self._on_range_selected,
            "horizontal",
            useblit=True,
            props={"alpha": 0.25, "facecolor": "#0078d4"},
            minspan=0.01,
        )
        self._set_window_controls_enabled(False)
        self._update_y_control_states()

    @staticmethod
    def _configure_y_spin(spin: QDoubleSpinBox) -> None:
        spin.setDecimals(3)
        spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
        spin.setSingleStep(1.0)
        spin.setMinimumWidth(105)

    def set_data(self, df, video_start_sec=0.0, video_duration_sec=0.0):
        self._df = df
        self._video_start_sec = video_start_sec
        self._video_duration_sec = video_duration_sec
        self._reset_window_bounds()
        self._redraw()

    def clear(self) -> None:
        self._df = None
        self._ax.clear()
        self._ax.set_xlabel("Time (s from video start)")
        self._ax.set_ylabel("Value")
        self._cursor_line = None
        self._cursor_sec = None
        self._data_start_sec = 0.0
        self._data_end_sec = 0.0
        self._window_start_sec = 0.0
        self._set_window_controls_enabled(False)
        self._range_label.setText("Full range")
        self._sync_signal_controls([])
        self._canvas.draw_idle()

    def set_intervals(self, intervals) -> None:
        self._intervals = []
        for interval in intervals:
            start_sec = getattr(interval, "start_sec", None)
            end_sec = getattr(interval, "end_sec", None)
            color = getattr(interval, "color", "#4488cc")
            if start_sec is None or end_sec is None:
                continue
            start_sec = float(start_sec)
            end_sec = float(end_sec)
            if end_sec <= start_sec:
                continue
            self._intervals.append((start_sec, end_sec, str(color)))
        self._redraw()

    def set_cursor(self, time_sec: float) -> None:
        self._cursor_sec = time_sec
        if self._cursor_line is not None:
            self._cursor_line.set_xdata([time_sec, time_sec])
        else:
            self._cursor_line = self._ax.axvline(x=time_sec, color="red", linewidth=1.2)
        self._canvas.draw_idle()

    def _on_y_controls_changed(self, *_args) -> None:
        if self._updating_y_controls:
            return
        self._update_y_control_states()
        self._apply_y_scale()
        self._canvas.draw_idle()

    def _update_y_control_states(self) -> None:
        self._y_min.setEnabled(not self._auto_y_min.isChecked())
        self._y_max.setEnabled(not self._auto_y_max.isChecked())

    def _set_zero_y_min(self) -> None:
        self._auto_y_min.setChecked(False)
        self._y_min.setValue(0.0)
        self._apply_y_scale()
        self._canvas.draw_idle()

    def _set_hr_y_max(self) -> None:
        self._auto_y_max.setChecked(False)
        self._y_max.setValue(220.0)
        self._apply_y_scale()
        self._canvas.draw_idle()

    def _sync_signal_controls(self, signal_names: list[str]) -> None:
        while self._signal_controls.count():
            item = self._signal_controls.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._signal_checkboxes = {}
        self._signal_visibility = {
            name: self._signal_visibility.get(name, True) for name in signal_names
        }
        self._signal_controls.addWidget(QLabel("Displayed signals:"))
        for name in signal_names:
            checkbox = QCheckBox(name)
            checkbox.setChecked(self._signal_visibility[name])
            checkbox.toggled.connect(
                lambda checked, signal_name=name: self._set_signal_visible(
                    signal_name, checked
                )
            )
            self._signal_checkboxes[name] = checkbox
            self._signal_controls.addWidget(checkbox)
        self._signal_controls.addStretch()
        self._signal_controls_widget.setVisible(bool(signal_names))

    def _set_signal_visible(self, signal_name: str, visible: bool) -> None:
        self._signal_visibility[signal_name] = visible
        for line in self._signal_lines:
            if line.get_label() == signal_name:
                line.set_visible(visible)
        self._update_legend()
        self._apply_y_scale()
        self._canvas.draw_idle()

    def _update_legend(self) -> None:
        existing_legend = self._ax.get_legend()
        if existing_legend is not None:
            existing_legend.remove()

        signal_line_ids = {id(line) for line in self._signal_lines}
        handles, labels = self._ax.get_legend_handles_labels()
        unique_handles = []
        unique_labels = []
        seen = set()
        for handle, label in zip(handles, labels):
            if id(handle) in signal_line_ids and not handle.get_visible():
                continue
            if not label or label.startswith("_") or label in seen:
                continue
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)
        if unique_handles and (len(self._sensor_ids) > 1 or len(unique_handles) > 1):
            self._ax.legend(unique_handles, unique_labels, fontsize=8)

    def _visible_auto_y_bounds(self) -> Optional[tuple[float, float]]:
        visible_values = []
        window_start = self._window_start_sec
        window_end = window_start + self._visible_duration()
        for line in self._signal_lines:
            if not line.get_visible():
                continue
            x_values = np.asarray(line.get_xdata(), dtype=float)
            y_values = np.asarray(line.get_ydata(), dtype=float)
            visible = (
                np.isfinite(x_values)
                & np.isfinite(y_values)
                & (x_values >= window_start)
                & (x_values <= window_end)
            )
            if visible.any():
                visible_values.append(y_values[visible])
        if not visible_values:
            return None

        values = np.concatenate(visible_values)
        value_min = float(values.min())
        value_max = float(values.max())
        span = value_max - value_min
        margin = span * 0.05 if span > 0 else max(abs(value_min) * 0.05, 0.5)
        return value_min - margin, value_max + margin

    def _apply_y_scale(self) -> None:
        auto_bounds = self._visible_auto_y_bounds()
        if auto_bounds is None:
            return
        auto_min, auto_max = auto_bounds

        self._updating_y_controls = True
        if self._auto_y_min.isChecked():
            self._y_min.setValue(auto_min)
        if self._auto_y_max.isChecked():
            self._y_max.setValue(auto_max)
        self._updating_y_controls = False

        y_min = auto_min if self._auto_y_min.isChecked() else self._y_min.value()
        y_max = auto_max if self._auto_y_max.isChecked() else self._y_max.value()
        if y_max <= y_min:
            y_max = y_min + max(abs(y_min) * 0.01, 0.001)
        self._ax.set_ylim(y_min, y_max)

    def _reset_window_bounds(self) -> None:
        if self._df is None or self._df.empty:
            self._set_window_controls_enabled(False)
            return
        timestamps = self._df["timestamp_utc"]
        signal_duration = (timestamps.max() - timestamps.min()).total_seconds()
        full_duration = max(float(self._video_duration_sec), signal_duration, 0.01)
        self._data_start_sec = float(self._video_start_sec)
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
        self._window_start_sec = min(
            self._window_start_sec,
            self._data_end_sec - duration_sec,
        )
        self._apply_time_window()

    def _on_window_position_changed(self, position: int) -> None:
        if self._updating_window_controls:
            return
        travel = max(0.0, self._full_duration() - self._visible_duration())
        self._window_start_sec = self._data_start_sec + travel * position / 10_000
        self._apply_time_window()

    def _pan_window(self, direction: int) -> None:
        step = self._visible_duration() * 0.8
        self._window_start_sec += direction * step
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
        if self._df is None or self._df.empty:
            return
        duration = self._visible_duration()
        latest_start = self._data_end_sec - duration
        self._window_start_sec = max(
            self._data_start_sec,
            min(self._window_start_sec, latest_start),
        )
        self._ax.set_xlim(self._window_start_sec, self._window_start_sec + duration)
        self._apply_y_scale()
        self._update_window_controls()
        if redraw:
            self._canvas.draw_idle()

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
        self._range_label.setText(f"{self._window_start_sec:.2f} - {end_sec:.2f} s")

    def _draw_interval_highlights(self) -> None:
        for start_sec, end_sec, color in self._intervals:
            self._ax.axvspan(
                start_sec,
                end_sec,
                color=color,
                alpha=0.18,
                linewidth=0,
                zorder=0,
            )

    def _redraw(self) -> None:
        self._ax.clear()
        self._signal_lines = []
        if self._df is None or self._df.empty:
            self._ax.set_xlabel("Time (s from video start)")
            self._ax.set_ylabel("Value")
            self._canvas.draw_idle()
            return

        self._draw_interval_highlights()

        # Detect format: wide (no sensor_id column) vs legacy long
        if "sensor_id" in self._df.columns:
            # Legacy long format
            self._sensor_ids = list(self._df["sensor_id"].unique())
            for sid in self._sensor_ids:
                sub = self._df[self._df["sensor_id"] == sid]
                t0 = sub["timestamp_utc"].iloc[0]
                rel_sec = (sub["timestamp_utc"] - t0).dt.total_seconds() + self._video_start_sec
                line, = self._ax.plot(rel_sec, sub["value"], label=str(sid), linewidth=0.8)
                self._signal_lines.append(line)
        else:
            # Wide format: each column except timestamp_utc is a series
            value_cols = [
                c for c in self._df.columns
                if c != "timestamp_utc" and not is_aux_signal_column(str(c))
            ]
            self._sensor_ids = value_cols
            t0 = self._df["timestamp_utc"].iloc[0]
            for col in value_cols:
                valid = self._df[col].notna()
                if not valid.any():
                    continue
                rel_sec = (
                    self._df.loc[valid, "timestamp_utc"] - t0
                ).dt.total_seconds() + self._video_start_sec
                values = self._df.loc[valid, col]
                if col == "averaged":
                    line, = self._ax.plot(rel_sec, values, label=col,
                                          linewidth=1.2, linestyle="--")
                else:
                    line, = self._ax.plot(rel_sec, values, label=col, linewidth=0.8)
                self._signal_lines.append(line)

        signal_names = [str(line.get_label()) for line in self._signal_lines]
        self._sync_signal_controls(signal_names)
        for line in self._signal_lines:
            line.set_visible(self._signal_visibility.get(str(line.get_label()), True))

        self._ax.set_xlabel("Time (s from video start)")
        self._ax.set_ylabel("Value")
        self._update_legend()
        self._ax.grid(True, alpha=0.3)
        self._apply_time_window(redraw=False)
        self._fig.tight_layout()
        self._cursor_line = None
        if self._cursor_sec is not None:
            self._cursor_line = self._ax.axvline(
                x=self._cursor_sec,
                color="red",
                linewidth=1.2,
            )
        self._canvas.draw_idle()
