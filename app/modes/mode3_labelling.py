from __future__ import annotations

import csv
import json
import logging
import os
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..ffmpeg_utils import trim_video, find_ffmpeg
from ..signals import load_signal
from ..state import LabelInterval, ProjectState, load_sidecar, populate_tracks_from_videos
from ..widgets.frame_preview import MultiCameraPlayer
from ..widgets.signal_plot import SignalPlot
from ..widgets.timeline import IntervalItem, TimelineWidget, colour_for_label

log = logging.getLogger(__name__)

_META_FILTER = "Metadata sidecar (*.json);;All files (*)"
_VIDEO_FILTER = "Videos (*.mp4 *.mov *.lrf *.avi *.mkv);;All files (*)"
_SIGNAL_FILTER = "CSV / signal files (*.csv *.tsv *.txt);;All files (*)"


class Mode3Widget(QWidget):
    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state
        self._merged_df: Optional[pd.DataFrame] = None

        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)

        # --- Load videos section (allows skipping Mode 1/2) ---
        load_grp = QGroupBox("Load videos")
        load_lay = QVBoxLayout(load_grp)
        meta_btn = QPushButton("Load from metadata file …")
        meta_btn.clicked.connect(self._load_from_meta)
        load_lay.addWidget(meta_btn)
        vids_btn = QPushButton("Select video files directly …")
        vids_btn.clicked.connect(self._load_from_videos)
        load_lay.addWidget(vids_btn)
        sig_btn = QPushButton("Add signal file …")
        sig_btn.clicked.connect(self._add_signal_inline)
        load_lay.addWidget(sig_btn)
        self._load_status = QLabel("")
        self._load_status.setWordWrap(True)
        load_lay.addWidget(self._load_status)
        ll.addWidget(load_grp)

        lib_grp = QGroupBox("Label library")
        lib_lay = QVBoxLayout(lib_grp)
        self._lib_list = QListWidget()
        lib_lay.addWidget(self._lib_list)
        lbtn_row = QHBoxLayout()
        add_lb = QPushButton("Add label")
        add_lb.clicked.connect(self._add_label)
        lbtn_row.addWidget(add_lb)
        rm_lb = QPushButton("Remove")
        rm_lb.clicked.connect(self._remove_label)
        lbtn_row.addWidget(rm_lb)
        lib_lay.addLayout(lbtn_row)
        ll.addWidget(lib_grp)

        self._label_combo = QComboBox()
        self._label_combo.setEditable(True)
        ll.addWidget(QLabel("Current label:"))
        ll.addWidget(self._label_combo)

        self._keep_unlabelled = QCheckBox("Keep unlabelled segments")
        self._keep_unlabelled.setChecked(False)
        ll.addWidget(self._keep_unlabelled)

        self._export_btn = QPushButton("Export labelled segments")
        self._export_btn.setStyleSheet("font-weight:bold; padding:8px;")
        self._export_btn.clicked.connect(self._export)
        ll.addWidget(self._export_btn)

        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        ll.addWidget(self._progress)

        self._status = QLabel("")
        ll.addWidget(self._status)
        ll.addStretch()
        splitter.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)

        self._player = MultiCameraPlayer()
        rl.addWidget(self._player)

        self._plot = SignalPlot()
        rl.addWidget(self._plot)

        self._timeline = TimelineWidget()
        self._timeline.setFixedHeight(60)
        rl.addWidget(self._timeline)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        self._timeline.interval_created.connect(self._on_interval_created)
        self._timeline.interval_deleted.connect(self._on_interval_deleted)
        self._timeline.playhead_moved.connect(self._on_playhead)
        self._player.frame_changed.connect(self._on_frame_changed)

        self._restore_from_state()

    def _restore_from_state(self):
        for lbl in self.state.labels_library:
            self._lib_list.addItem(lbl)
            self._label_combo.addItem(lbl)

        if self.state.tracks:
            labels = [t.camera_label for t in self.state.tracks]
            self._player.set_cameras(labels)
            paths = [t.final_output_path for t in self.state.tracks]
            valid = [p for p in paths if p and Path(p).exists()]
            if valid:
                self._player.load_videos(valid)

            dur = self.state.tracks[0].duration_sec if self.state.tracks else 0
            self._timeline.set_duration(dur)

        items = []
        for iv in self.state.intervals:
            items.append(IntervalItem(iv.label, iv.start_sec, iv.end_sec, iv.color))
        self._timeline.set_intervals(items)

        self._load_signals()

    def _load_signals(self):
        if not self.state.signal_paths:
            return
        dfs = []
        for p in self.state.signal_paths:
            df = load_signal(p)
            if df is not None:
                dfs.append(df)
        if dfs:
            merged = pd.concat(dfs, ignore_index=True)

            # Clip to video time range if tracks are available
            if self.state.tracks:
                t0_track = self.state.tracks[0]
                dt_start = t0_track.parsed_start_datetime()
                dur = t0_track.duration_sec
                if dt_start is not None:
                    from datetime import timedelta as _td
                    dt_end = dt_start + _td(seconds=dur)
                    start_ts = pd.Timestamp(dt_start)
                    end_ts = pd.Timestamp(dt_end)
                    merged = merged[
                        (merged["timestamp_utc"] >= start_ts)
                        & (merged["timestamp_utc"] <= end_ts)
                    ]

            if merged.empty:
                return

            # Pivot to wide format
            wide = merged.pivot_table(
                index="timestamp_utc", columns="sensor_id", values="value", aggfunc="first"
            )
            wide.columns.name = None
            wide = wide.sort_index().ffill().bfill()

            if self.state.include_signal_average:
                wide["averaged"] = wide.mean(axis=1)

            self._merged_df = wide.reset_index()
            dur = self.state.tracks[0].duration_sec if self.state.tracks else 0
            self._plot.set_data(self._merged_df, video_duration_sec=dur)

    def _load_synced_signal(self) -> bool:
        """Load pre-synced signal CSV from state. Returns True if loaded."""
        sp = self.state.synced_signal_path
        if not sp or not Path(sp).exists():
            return False
        try:
            df = pd.read_csv(sp)
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
            self._merged_df = df
            dur = self.state.tracks[0].duration_sec if self.state.tracks else 0
            self._plot.set_data(self._merged_df, video_duration_sec=dur)
            return True
        except Exception as exc:
            log.warning("Could not load synced signal CSV: %s", exc)
            return False

    def _load_from_meta(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select metadata sidecar", "", _META_FILTER,
        )
        if not path:
            return
        try:
            load_sidecar(path, self.state)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load metadata:\n{exc}")
            return
        self._refresh_from_tracks()
        if not self._load_synced_signal():
            self._load_signals()
        n = len(self.state.tracks)
        self._load_status.setText(f"Loaded {n} camera(s) from sidecar.")

    def _load_from_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select video files", "", _VIDEO_FILTER,
        )
        if not files:
            return
        try:
            populate_tracks_from_videos(files, self.state)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to probe videos:\n{exc}")
            return
        self._refresh_from_tracks()
        n = len(self.state.tracks)
        self._load_status.setText(f"Loaded {n} video(s) directly.")

    def _add_signal_inline(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select signal files", "", _SIGNAL_FILTER,
        )
        if not files:
            return
        for p in files:
            if p not in self.state.signal_paths:
                self.state.signal_paths.append(p)
        self._load_signals()
        self._load_status.setText(
            f"Loaded {len(self.state.signal_paths)} signal file(s)."
        )

    def _refresh_from_tracks(self):
        """Reload player and timeline from the current state.tracks."""
        if self.state.tracks:
            labels = [t.camera_label for t in self.state.tracks]
            self._player.set_cameras(labels)
            paths = [t.final_output_path for t in self.state.tracks]
            valid = [p for p in paths if p and Path(p).exists()]
            if valid:
                self._player.load_videos(valid)
            dur = self.state.tracks[0].duration_sec if self.state.tracks else 0
            self._timeline.set_duration(dur)

    def _add_label(self):
        text, ok = QInputDialog.getText(self, "New label", "Label name:")
        if ok and text:
            self._lib_list.addItem(text)
            self._label_combo.addItem(text)
            if text not in self.state.labels_library:
                self.state.labels_library.append(text)

    def _remove_label(self):
        for item in self._lib_list.selectedItems():
            row = self._lib_list.row(item)
            self._lib_list.takeItem(row)
            idx = self._label_combo.findText(item.text())
            if idx >= 0:
                self._label_combo.removeItem(idx)
            if item.text() in self.state.labels_library:
                self.state.labels_library.remove(item.text())

    def _on_interval_created(self, start_sec, end_sec):
        label = self._label_combo.currentText().strip()
        if not label:
            label = "Unlabelled"
        color = colour_for_label(label, self.state.labels_library)
        iv = LabelInterval(label=label, start_sec=start_sec, end_sec=end_sec, color=color)
        self.state.intervals.append(iv)
        items = [
            IntervalItem(i.label, i.start_sec, i.end_sec, i.color)
            for i in self.state.intervals
        ]
        self._timeline.set_intervals(items)

    def _on_interval_deleted(self, idx):
        if 0 <= idx < len(self.state.intervals):
            self.state.intervals.pop(idx)
            items = [
                IntervalItem(i.label, i.start_sec, i.end_sec, i.color)
                for i in self.state.intervals
            ]
            self._timeline.set_intervals(items)

    def _on_playhead(self, sec):
        if not self.state.tracks:
            return
        fps = self.state.tracks[0].fps or 30.0
        frame = int(sec * fps)
        self._player.seek_frame(frame)

    def _on_frame_changed(self, frame_no: int):
        if not self.state.tracks:
            return
        fps = self.state.tracks[0].fps or 30.0
        sec = frame_no / fps
        self._timeline.set_playhead(sec)
        self._plot.set_cursor(sec)

    def _export(self):
        out_dir = self.state.output_directory
        if not out_dir:
            QMessageBox.warning(self, "Warning", "No output directory set.")
            return

        intervals = self.state.intervals
        if not intervals:
            QMessageBox.information(self, "Info", "No labelled intervals to export.")
            return

        if not self._keep_unlabelled.isChecked():
            intervals = [iv for iv in intervals if iv.label != "Unlabelled"]

        ffmpeg = self.state.ffmpeg_path or find_ffmpeg()
        if not ffmpeg:
            QMessageBox.critical(self, "Error", "FFmpeg not found.")
            return

        segments_dir = Path(out_dir) / "labelled_segments"
        segments_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows = []
        total = len(intervals)

        for idx, iv in enumerate(intervals):
            self._progress.setValue(int((idx / total) * 100))
            seg_name = f"{idx:04d}_{iv.label.replace(' ', '_')}"
            seg_dir = segments_dir / seg_name
            seg_dir.mkdir(parents=True, exist_ok=True)

            for ti, track in enumerate(self.state.tracks):
                src = track.final_output_path
                if not src or not Path(src).exists():
                    continue
                dst = str(seg_dir / f"cam{ti+1}_{track.camera_label}.mp4")
                duration = iv.end_sec - iv.start_sec
                try:
                    trim_video(
                        src, dst,
                        start_sec=iv.start_sec,
                        duration_sec=duration,
                        ffmpeg=ffmpeg,
                    )
                except Exception as exc:
                    log.error("Trim failed for %s: %s", dst, exc)

            if self._merged_df is not None and not self._merged_df.empty:
                try:
                    t0 = self.state.tracks[0]
                    dt_start = t0.parsed_start_datetime()
                    if dt_start is not None:
                        seg_start = dt_start + timedelta(seconds=iv.start_sec)
                        seg_end = dt_start + timedelta(seconds=iv.end_sec)
                        start_ts = pd.Timestamp(seg_start)
                        end_ts = pd.Timestamp(seg_end)
                        sub = self._merged_df[
                            (self._merged_df["timestamp_utc"] >= start_ts)
                            & (self._merged_df["timestamp_utc"] <= end_ts)
                        ]
                        if not sub.empty:
                            sub.to_csv(seg_dir / "signal.csv", index=False)
                except Exception as exc:
                    log.error("Signal export error for segment %d: %s", idx, exc)

            meta = {
                "index": idx,
                "label": iv.label,
                "start_sec": iv.start_sec,
                "end_sec": iv.end_sec,
                "duration_sec": iv.end_sec - iv.start_sec,
                "folder": seg_name,
            }
            (seg_dir / "meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            manifest_rows.append(meta)

        manifest_path = segments_dir / "manifest.csv"
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "label", "start_sec", "end_sec", "duration_sec", "folder"])
            writer.writeheader()
            writer.writerows(manifest_rows)

        self._progress.setValue(100)
        self._status.setText(f"Exported {len(manifest_rows)} segments to {segments_dir}")
        log.info("Exported %d segments to %s", len(manifest_rows), segments_dir)
