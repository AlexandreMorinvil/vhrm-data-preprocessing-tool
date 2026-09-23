from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QWidget

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
        layout.addWidget(self._canvas)

        self._cursor_line = None
        self._cursor_sec: Optional[float] = None
        self._df: Optional[pd.DataFrame] = None
        self._video_start_sec: float = 0.0
        self._video_duration_sec: float = 0.0
        self._sensor_ids: list[str] = []
        self._intervals: list[tuple[float, float, str]] = []

    def set_data(self, df, video_start_sec=0.0, video_duration_sec=0.0):
        self._df = df
        self._video_start_sec = video_start_sec
        self._video_duration_sec = video_duration_sec
        self._redraw()

    def clear(self) -> None:
        self._df = None
        self._ax.clear()
        self._ax.set_xlabel("Time (s from video start)")
        self._ax.set_ylabel("Value")
        self._cursor_line = None
        self._cursor_sec = None
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
                self._ax.plot(rel_sec, sub["value"], label=str(sid), linewidth=0.8)
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
                    self._ax.plot(rel_sec, values, label=col,
                                  linewidth=1.2, linestyle="--")
                else:
                    self._ax.plot(rel_sec, values, label=col, linewidth=0.8)

        self._ax.set_xlabel("Time (s from video start)")
        self._ax.set_ylabel("Value")
        if len(self._sensor_ids) > 1:
            self._ax.legend(fontsize=8)
        self._ax.grid(True, alpha=0.3)
        self._fig.tight_layout()
        self._cursor_line = None
        if self._cursor_sec is not None:
            self._cursor_line = self._ax.axvline(
                x=self._cursor_sec,
                color="red",
                linewidth=1.2,
            )
        self._canvas.draw_idle()
