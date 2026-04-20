from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
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
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..signals import get_loaders, load_signal
from ..state import ProjectState, generate_sidecar, load_sidecar, populate_tracks_from_videos
from ..widgets.frame_preview import MultiCameraPlayer
from ..widgets.signal_plot import SignalPlot

log = logging.getLogger(__name__)

_SIGNAL_FILTER = "CSV / signal files (*.csv *.tsv *.txt);;All files (*)"
_META_FILTER = "Metadata sidecar (*.json);;All files (*)"
_VIDEO_FILTER = "Videos (*.mp4 *.mov *.lrf *.avi *.mkv);;All files (*)"


class Mode2Widget(QWidget):
    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state
        self._signal_dfs: list[pd.DataFrame] = []
        self._merged_df: Optional[pd.DataFrame] = None

        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)

        # --- Load videos section (allows skipping Mode 1) ---
        load_grp = QGroupBox("Load videos")
        load_lay = QVBoxLayout(load_grp)
        meta_btn = QPushButton("Load from metadata file …")
        meta_btn.clicked.connect(self._load_from_meta)
        load_lay.addWidget(meta_btn)
        vids_btn = QPushButton("Select video files directly …")
        vids_btn.clicked.connect(self._load_from_videos)
        load_lay.addWidget(vids_btn)
        self._load_status = QLabel("")
        self._load_status.setWordWrap(True)
        load_lay.addWidget(self._load_status)
        ll.addWidget(load_grp)

        loader_names = [type(l).__name__ for l in get_loaders()]
        info = QLabel(f"Available loaders: {', '.join(loader_names) or 'none'}")
        info.setWordWrap(True)
        ll.addWidget(info)

        self._file_list = QListWidget()
        ll.addWidget(self._file_list)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add signal file …")
        add_btn.clicked.connect(self._add_signal)
        btn_row.addWidget(add_btn)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_signal)
        btn_row.addWidget(remove_btn)
        ll.addLayout(btn_row)

        mode_grp = QGroupBox("Dual-sensor handling")
        mode_lay = QVBoxLayout(mode_grp)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Keep separate", "Average values"])
        self._mode_combo.setCurrentText(
            "Average values" if state.signal_mode == "average" else "Keep separate"
        )
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_lay.addWidget(self._mode_combo)
        ll.addWidget(mode_grp)

        self._load_btn = QPushButton("Load && synchronise signals")
        self._load_btn.setStyleSheet("font-weight:bold; padding:8px;")
        self._load_btn.clicked.connect(self._load_and_sync)
        ll.addWidget(self._load_btn)

        self._skip_btn = QPushButton("Skip (no signal)")
        self._skip_btn.clicked.connect(self._skip)
        ll.addWidget(self._skip_btn)

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
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        self._player.frame_changed.connect(self._on_frame_changed)

        self._restore_from_state()

    def _restore_from_state(self):
        for p in self.state.signal_paths:
            item = QListWidgetItem(Path(p).name)
            item.setData(Qt.ItemDataRole.UserRole, p)
            item.setToolTip(p)
            self._file_list.addItem(item)

        if self.state.tracks:
            labels = [t.camera_label for t in self.state.tracks]
            self._player.set_cameras(labels)
            paths = [t.final_output_path for t in self.state.tracks]
            valid = [p for p in paths if p and Path(p).exists()]
            if valid:
                self._player.load_videos(valid)

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
        self._refresh_player()
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
        self._refresh_player()
        n = len(self.state.tracks)
        self._load_status.setText(f"Loaded {n} video(s) directly.")

    def _refresh_player(self):
        """Reload the player/plot from the current state.tracks."""
        if self.state.tracks:
            labels = [t.camera_label for t in self.state.tracks]
            self._player.set_cameras(labels)
            paths = [t.final_output_path for t in self.state.tracks]
            valid = [p for p in paths if p and Path(p).exists()]
            if valid:
                self._player.load_videos(valid)

    def _add_signal(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select signal files", "", _SIGNAL_FILTER,
        )
        for p in files:
            item = QListWidgetItem(Path(p).name)
            item.setData(Qt.ItemDataRole.UserRole, p)
            item.setToolTip(p)
            self._file_list.addItem(item)

    def _remove_signal(self):
        for item in self._file_list.selectedItems():
            self._file_list.takeItem(self._file_list.row(item))

    def _on_mode_changed(self, text):
        self.state.signal_mode = "average" if "Average" in text else "separate"

    def _load_and_sync(self):
        paths = []
        for i in range(self._file_list.count()):
            paths.append(self._file_list.item(i).data(Qt.ItemDataRole.UserRole))
        if not paths:
            QMessageBox.information(self, "Info", "No signal files selected.")
            return

        self.state.signal_paths = paths
        self._signal_dfs.clear()
        for p in paths:
            df = load_signal(p)
            if df is not None:
                self._signal_dfs.append(df)
            else:
                self._status.setText(f"Warning: could not load {Path(p).name}")
                log.warning("Failed to load signal: %s", p)

        if not self._signal_dfs:
            QMessageBox.warning(self, "Warning", "No signals could be loaded.")
            return

        merged = pd.concat(self._signal_dfs, ignore_index=True)

        if self.state.tracks:
            t0_track = self.state.tracks[0]
            dt_start = t0_track.parsed_start_datetime()
            dur = t0_track.duration_sec
            if dt_start is not None:
                from datetime import timedelta
                dt_end = dt_start + timedelta(seconds=dur)
                start_ts = pd.Timestamp(dt_start)
                end_ts = pd.Timestamp(dt_end)
                merged = merged[
                    (merged["timestamp_utc"] >= start_ts)
                    & (merged["timestamp_utc"] <= end_ts)
                ]

        if self.state.signal_mode == "average":
            merged = (
                merged.groupby("timestamp_utc", as_index=False)
                .agg({"value": "mean", "sensor_id": "first"})
            )
            merged["sensor_id"] = "averaged"

        self._merged_df = merged.sort_values("timestamp_utc").reset_index(drop=True)

        video_dur = 0.0
        if self.state.tracks:
            video_dur = self.state.tracks[0].duration_sec
        self._plot.set_data(self._merged_df, video_duration_sec=video_dur)

        # Export clipped signal CSV
        csv_name = "signal_synced_averaged.csv" if self.state.signal_mode == "average" else "signal_synced.csv"
        out_dir = self.state.output_directory
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            csv_path = str(Path(out_dir) / csv_name)
            self._merged_df.to_csv(csv_path, index=False)
            self.state.synced_signal_path = csv_path
            log.info("Exported synced signal: %s", csv_path)

            # Re-generate sidecar with signal info
            try:
                generate_sidecar(self.state)
            except Exception as exc:
                log.warning("Could not update sidecar: %s", exc)

        status_parts = [f"Loaded {len(self._merged_df)} samples from {len(self._signal_dfs)} file(s)"]
        if self.state.synced_signal_path:
            status_parts.append(f"Exported: {csv_name}")
        self._status.setText(" — ".join(status_parts))
        self.state.mode2_complete = True

    def _skip(self):
        self.state.mode2_complete = True
        self._status.setText("Skipped signal synchronisation.")

    def _on_frame_changed(self, frame_no: int):
        if not self.state.tracks:
            return
        fps = self.state.tracks[0].fps or 30.0
        time_sec = frame_no / fps
        self._plot.set_cursor(time_sec)
