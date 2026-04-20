from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QWidget

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
        self._df: Optional[pd.DataFrame] = None
        self._video_start_sec: float = 0.0
        self._video_duration_sec: float = 0.0
        self._sensor_ids: list[str] = []

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
        self._canvas.draw_idle()

    def set_cursor(self, time_sec: float) -> None:
        if self._cursor_line is not None:
            self._cursor_line.set_xdata([time_sec, time_sec])
        else:
            self._cursor_line = self._ax.axvline(x=time_sec, color="red", linewidth=1.2)
        self._canvas.draw_idle()

    def _redraw(self) -> None:
        self._ax.clear()
        if self._df is None or self._df.empty:
            self._canvas.draw_idle()
            return

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
            value_cols = [c for c in self._df.columns if c != "timestamp_utc"]
            self._sensor_ids = value_cols
            t0 = self._df["timestamp_utc"].iloc[0]
            rel_sec = (self._df["timestamp_utc"] - t0).dt.total_seconds() + self._video_start_sec
            for col in value_cols:
                if col == "averaged":
                    self._ax.plot(rel_sec, self._df[col], label=col,
                                  linewidth=1.2, linestyle="--")
                else:
                    self._ax.plot(rel_sec, self._df[col], label=col, linewidth=0.8)

        self._ax.set_xlabel("Time (s from video start)")
        self._ax.set_ylabel("Value")
        if len(self._sensor_ids) > 1:
            self._ax.legend(fontsize=8)
        self._ax.grid(True, alpha=0.3)
        self._fig.tight_layout()
        self._cursor_line = None
        self._canvas.draw_idle()
