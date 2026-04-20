from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..state import ProjectState
from ..widgets.frame_preview import MultiCameraPlayer
from ..widgets.signal_plot import SignalPlot
from ..widgets.timeline import IntervalItem, TimelineWidget

log = logging.getLogger(__name__)


class _SegmentInfo:
    __slots__ = ("index", "label", "start_sec", "end_sec", "duration_sec", "folder", "base_dir")

    def __init__(self, row: dict, base_dir: Path):
        self.index = int(row.get("index", 0))
        self.label = row.get("label", "")
        self.start_sec = float(row.get("start_sec", 0))
        self.end_sec = float(row.get("end_sec", 0))
        self.duration_sec = float(row.get("duration_sec", 0))
        self.folder = row.get("folder", "")
        self.base_dir = base_dir

    @property
    def dir_path(self) -> Path:
        return self.base_dir / self.folder


class Mode4Widget(QWidget):
    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state
        self._segments: list[_SegmentInfo] = []
        self._current_seg: Optional[_SegmentInfo] = None

        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)

        self._load_btn = QPushButton("Load manifest CSV …")
        self._load_btn.clicked.connect(self._load_manifest)
        ll.addWidget(self._load_btn)

        self._seg_list = QListWidget()
        self._seg_list.currentRowChanged.connect(self._on_segment_selected)
        ll.addWidget(self._seg_list)

        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        ll.addWidget(self._info_label)
        ll.addStretch()
        splitter.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)

        self._timeline = TimelineWidget()
        self._timeline.setFixedHeight(50)
        rl.addWidget(self._timeline)

        self._player = MultiCameraPlayer()
        rl.addWidget(self._player)

        self._plot = SignalPlot()
        rl.addWidget(self._plot)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        self._timeline.playhead_moved.connect(self._on_playhead)
        self._player.frame_changed.connect(self._on_frame_changed)

    def _load_manifest(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select manifest.csv", "", "CSV (*.csv);;All files (*)",
        )
        if not path:
            return
        base_dir = Path(path).parent
        self._segments.clear()
        self._seg_list.clear()

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seg = _SegmentInfo(row, base_dir)
                self._segments.append(seg)
                item = QListWidgetItem(f"[{seg.index}] {seg.label}  ({seg.duration_sec:.1f}s)")
                self._seg_list.addItem(item)

        total_dur = sum(s.end_sec - s.start_sec for s in self._segments)
        self._timeline.set_duration(total_dur)
        items = []
        offset = 0.0
        for seg in self._segments:
            dur = seg.end_sec - seg.start_sec
            items.append(IntervalItem(seg.label, offset, offset + dur, "#4488cc"))
            offset += dur
        self._timeline.set_intervals(items)
        self._info_label.setText(f"Loaded {len(self._segments)} segments from {Path(path).name}")

    def _on_segment_selected(self, row: int):
        if row < 0 or row >= len(self._segments):
            return
        seg = self._segments[row]
        self._current_seg = seg

        seg_dir = seg.dir_path
        if not seg_dir.exists():
            self._info_label.setText(f"Folder not found: {seg_dir}")
            return

        video_files = sorted(seg_dir.glob("cam*.*"))
        if video_files:
            labels = [f.stem for f in video_files]
            self._player.set_cameras(labels)
            self._player.load_videos([str(f) for f in video_files])
        else:
            self._player.set_cameras([])

        signal_csv = seg_dir / "signal.csv"
        if signal_csv.exists():
            try:
                df = pd.read_csv(signal_csv, parse_dates=["timestamp_utc"])
                if "sensor_id" not in df.columns:
                    df["sensor_id"] = "signal"
                self._plot.set_data(df, video_duration_sec=seg.duration_sec)
            except Exception as exc:
                log.error("Error loading signal for segment %d: %s", seg.index, exc)
                self._plot.clear()
        else:
            self._plot.clear()

        meta_json = seg_dir / "meta.json"
        info_lines = [
            f"Segment: {seg.index}",
            f"Label: {seg.label}",
            f"Duration: {seg.duration_sec:.2f}s",
            f"Original time: {seg.start_sec:.2f} – {seg.end_sec:.2f}s",
        ]
        if meta_json.exists():
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
            for k, v in meta.items():
                if k not in ("index", "label", "start_sec", "end_sec", "duration_sec", "folder"):
                    info_lines.append(f"{k}: {v}")
        self._info_label.setText("\n".join(info_lines))

    def _on_playhead(self, sec: float):
        cumulative = 0.0
        for i, seg in enumerate(self._segments):
            dur = seg.end_sec - seg.start_sec
            if cumulative + dur >= sec:
                if self._current_seg is not seg:
                    self._seg_list.setCurrentRow(i)
                local_sec = sec - cumulative
                fps = self._player.get_fps()
                self._player.seek_frame(int(local_sec * fps))
                return
            cumulative += dur

    def _on_frame_changed(self, frame_no: int):
        fps = self._player.get_fps()
        sec = frame_no / fps if fps > 0 else 0
        if self._current_seg:
            cumulative = 0.0
            for seg in self._segments:
                if seg is self._current_seg:
                    break
                cumulative += seg.end_sec - seg.start_sec
            self._timeline.set_playhead(cumulative + sec)
        self._plot.set_cursor(sec)
