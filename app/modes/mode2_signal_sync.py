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
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
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

from ..mosaic_export import MosaicWorker
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

        self._avg_checkbox = QCheckBox("Include average column")
        self._avg_checkbox.setChecked(state.include_signal_average)
        self._avg_checkbox.toggled.connect(self._on_avg_toggled)
        ll.addWidget(self._avg_checkbox)

        self._load_btn = QPushButton("Load && synchronise signals")
        self._load_btn.setStyleSheet("font-weight:bold; padding:8px;")
        self._load_btn.clicked.connect(self._load_and_sync)
        ll.addWidget(self._load_btn)

        self._skip_btn = QPushButton("Skip (no signal)")
        self._skip_btn.clicked.connect(self._skip)
        ll.addWidget(self._skip_btn)

        self._status = QLabel("")
        ll.addWidget(self._status)

        # --- Mosaic export ---
        mosaic_grp = QGroupBox("Mosaic video export")
        mosaic_lay = QVBoxLayout(mosaic_grp)
        self._mosaic_btn = QPushButton("Export mosaic video \u2026")
        self._mosaic_btn.setStyleSheet("font-weight:bold; padding:6px;")
        self._mosaic_btn.clicked.connect(self._export_mosaic)
        mosaic_lay.addWidget(self._mosaic_btn)
        self._mosaic_cancel_btn = QPushButton("Cancel")
        self._mosaic_cancel_btn.setEnabled(False)
        self._mosaic_cancel_btn.clicked.connect(self._cancel_mosaic)
        mosaic_lay.addWidget(self._mosaic_cancel_btn)
        self._mosaic_progress = QProgressBar()
        self._mosaic_progress.setTextVisible(True)
        mosaic_lay.addWidget(self._mosaic_progress)
        self._mosaic_status = QLabel("")
        mosaic_lay.addWidget(self._mosaic_status)
        ll.addWidget(mosaic_grp)
        self._mosaic_worker: MosaicWorker | None = None

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
        self._load_synced_signal()
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

    def _load_synced_signal(self):
        """Load synced signal CSV from state if available."""
        sp = self.state.synced_signal_path
        if not sp or not Path(sp).exists():
            return
        try:
            df = pd.read_csv(sp)
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
            self._merged_df = df
            video_dur = self.state.tracks[0].duration_sec if self.state.tracks else 0.0
            self._plot.set_data(self._merged_df, video_duration_sec=video_dur)
            self._status.setText(f"Signal loaded from {Path(sp).name}")
        except Exception as exc:
            log.warning("Could not load synced signal CSV: %s", exc)

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

    def _on_avg_toggled(self, checked):
        self.state.include_signal_average = checked

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

        # Track which sensors exist before clipping
        all_sensors = set(merged["sensor_id"].unique())

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

                # Warn about sensors that were completely filtered out
                surviving_sensors = set(merged["sensor_id"].unique())
                lost = all_sensors - surviving_sensors
                if lost:
                    QMessageBox.warning(
                        self, "Sensors outside video range",
                        f"The following sensor(s) had no data within the video "
                        f"time range and were excluded:\n\n"
                        + "\n".join(f"  • {s}" for s in sorted(lost))
                    )

        if merged.empty:
            QMessageBox.warning(self, "Warning", "No signal data within the video time range.")
            return

        # Pivot from long to wide format
        wide = merged.pivot_table(
            index="timestamp_utc", columns="sensor_id", values="value", aggfunc="first"
        )
        wide.columns.name = None  # remove the "sensor_id" label from columns
        wide = wide.sort_index().ffill().bfill()

        if self.state.include_signal_average:
            wide["averaged"] = wide.mean(axis=1)

        wide = wide.reset_index()

        self._merged_df = wide

        video_dur = 0.0
        if self.state.tracks:
            video_dur = self.state.tracks[0].duration_sec
        self._plot.set_data(self._merged_df, video_duration_sec=video_dur)

        # Export clipped signal CSV
        csv_name = "signal_synced.csv"
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

    # ------------------------------------------------------------------
    # Mosaic export
    # ------------------------------------------------------------------

    def _export_mosaic(self):
        if not self.state.tracks:
            QMessageBox.warning(self, "Warning", "No videos loaded.")
            return

        paths = [t.final_output_path for t in self.state.tracks]
        valid = [p for p in paths if p and Path(p).exists()]
        if not valid:
            QMessageBox.warning(self, "Warning", "No valid video files found.")
            return

        out_dir = self.state.output_directory
        if not out_dir:
            out_dir = str(Path(valid[0]).parent)

        default_name = str(Path(out_dir) / "mosaic_full.mp4")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save mosaic video", default_name,
            "MP4 video (*.mp4);;All files (*)",
        )
        if not path:
            return

        t0 = self.state.tracks[0]
        fps = t0.fps or 30.0
        total_frames = t0.frame_count or int(t0.duration_sec * fps)
        labels = [t.camera_label for t in self.state.tracks]
        duration = t0.duration_sec

        self._mosaic_worker = MosaicWorker(
            video_paths=valid,
            camera_labels=labels,
            fps=fps,
            total_frames=total_frames,
            video_duration_sec=duration,
            output_path=path,
            signal_df=self._merged_df,
            ffmpeg_path=self.state.ffmpeg_path,
        )
        self._mosaic_worker.progress.connect(self._on_mosaic_progress)
        self._mosaic_worker.finished.connect(self._on_mosaic_finished)
        self._mosaic_btn.setEnabled(False)
        self._mosaic_cancel_btn.setEnabled(True)
        self._mosaic_progress.setValue(0)
        self._mosaic_status.setText("Exporting\u2026")
        self._mosaic_worker.start()

    def _cancel_mosaic(self):
        if self._mosaic_worker:
            self._mosaic_worker.cancel()

    def _on_mosaic_progress(self, current: int, total: int):
        if total > 0:
            self._mosaic_progress.setValue(int(current * 100 / total))
        self._mosaic_status.setText(f"Frame {current} / {total}")

    def _on_mosaic_finished(self, success: bool, msg: str):
        self._mosaic_btn.setEnabled(True)
        self._mosaic_cancel_btn.setEnabled(False)
        if success:
            self._mosaic_progress.setValue(100)
            self._mosaic_status.setText(f"Saved: {Path(msg).name}")
            QMessageBox.information(self, "Mosaic export", f"Mosaic video saved:\n{msg}")
        else:
            self._mosaic_status.setText(f"Failed: {msg}")
            if "Cancelled" not in msg:
                QMessageBox.critical(self, "Error", f"Mosaic export failed:\n{msg}")
        self._mosaic_worker = None

    def _on_frame_changed(self, frame_no: int):
        if not self.state.tracks:
            return
        fps = self.state.tracks[0].fps or 30.0
        time_sec = frame_no / fps
        self._plot.set_cursor(time_sec)
