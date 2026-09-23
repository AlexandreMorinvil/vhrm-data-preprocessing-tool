from __future__ import annotations

import csv
import json
import logging
import math
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from PyQt6.QtCore import Qt, QTime
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from ..ffmpeg_utils import trim_video, find_ffmpeg
from ..mosaic_export import MOSAIC_PRESET_NAMES, MosaicWorker, normalise_mosaic_preset
from ..signals import (
    load_signal_files,
    read_synced_signal_csvs,
    signal_file_type_name,
    signal_long_to_wide,
    write_synced_signal_csvs,
)
from ..widgets.layout import configure_main_splitter
from ..state import (
    LabelInterval,
    ProjectState,
    compute_signal_anchor,
    format_time_coherence_warnings,
    load_labelled_segments_manifest,
    load_sidecar,
    populate_tracks_from_videos,
    video_timeline_duration_sec,
)
from ..widgets.frame_preview import MultiCameraPlayer
from ..widgets.signal_plot import SignalPlot
from ..widgets.subdivide_dialog import SubdivideDialog
from ..widgets.timeline import IntervalItem, TimelineWidget, colour_for_label

log = logging.getLogger(__name__)

_META_FILTER = "Metadata sidecar (*.json);;All files (*)"
_VIDEO_FILTER = "Videos (*.mp4 *.mov *.lrf *.avi *.mkv);;All files (*)"
_SIGNAL_FILTER = "CSV / signal files (*.csv *.tsv *.txt);;All files (*)"


def _secs_to_qtime(sec: float) -> QTime:
    ms = int(round(sec * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return QTime(h, m, s, ms)


def _qtime_to_secs(t: QTime) -> float:
    return t.hour() * 3600 + t.minute() * 60 + t.second() + t.msec() / 1000.0


def _format_duration(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _format_dt(dt: Optional[datetime]) -> str:
    if dt is None:
        return "unknown"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="milliseconds")


def _subdivide_interval(iv: LabelInterval, mode: str, value: float, color: str) -> list[LabelInterval]:
    """Return a list of sub-intervals from *iv*.

    *mode* is ``'count'`` or ``'duration'``.
    """
    dur = iv.end_sec - iv.start_sec
    if mode == "count":
        n = max(2, int(value))
        seg_dur = dur / n
    else:
        seg_dur = max(0.5, value)
        n = math.ceil(dur / seg_dur)

    result: list[LabelInterval] = []
    for i in range(n):
        s = iv.start_sec + i * seg_dur
        e = min(iv.start_sec + (i + 1) * seg_dur, iv.end_sec)
        if e - s < 0.01:
            break
        sub = LabelInterval(
            label=f"{iv.label}_part{i + 1}",
            start_sec=round(s, 3),
            end_sec=round(e, 3),
            color=color,
        )
        result.append(sub)
    return result


class Mode3Widget(QWidget):
    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state
        self._hr_merged_df: Optional[pd.DataFrame] = None
        self._ecg_merged_df: Optional[pd.DataFrame] = None
        self._has_exported: bool = False
        self._mosaic_worker: MosaicWorker | None = None

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
        manifest_btn = QPushButton("Import labelled segments manifest …")
        manifest_btn.clicked.connect(self._import_segments_manifest)
        load_lay.addWidget(manifest_btn)
        vids_btn = QPushButton("Select video files directly …")
        vids_btn.clicked.connect(self._load_from_videos)
        load_lay.addWidget(vids_btn)
        hr_btn = QPushButton("Add HR file …")
        hr_btn.clicked.connect(lambda: self._add_signal_inline("HR"))
        load_lay.addWidget(hr_btn)
        ecg_btn = QPushButton("Add ECG file …")
        ecg_btn.clicked.connect(lambda: self._add_signal_inline("ECG"))
        load_lay.addWidget(ecg_btn)
        self._load_status = QLabel("")
        self._load_status.setWordWrap(True)
        load_lay.addWidget(self._load_status)
        ll.addWidget(load_grp)

        time_grp = QGroupBox("Camera time correction")
        time_lay = QVBoxLayout(time_grp)
        self._time_summary_grid = QGridLayout()
        time_lay.addLayout(self._time_summary_grid)
        ll.addWidget(time_grp)

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

        # --- Start / End label at playhead ---
        ph_row = QHBoxLayout()
        self._start_label_btn = QPushButton("Start label ▶")
        self._start_label_btn.setCheckable(True)
        self._start_label_btn.setToolTip("Mark current playhead position as label start")
        self._start_label_btn.clicked.connect(self._start_label_at_playhead)
        ph_row.addWidget(self._start_label_btn)
        self._end_label_btn = QPushButton("End label ■")
        self._end_label_btn.setToolTip("Mark current playhead position as label end")
        self._end_label_btn.clicked.connect(self._end_label_at_playhead)
        ph_row.addWidget(self._end_label_btn)
        ll.addLayout(ph_row)
        self._pending_start: Optional[float] = None

        # --- Add interval manually ---
        add_grp = QGroupBox("Add interval manually")
        add_lay = QFormLayout(add_grp)
        self._add_start = QTimeEdit()
        self._add_start.setDisplayFormat("HH:mm:ss.zzz")
        add_lay.addRow("Start:", self._add_start)
        self._add_end = QTimeEdit()
        self._add_end.setDisplayFormat("HH:mm:ss.zzz")
        self._add_end.setTime(QTime(0, 0, 30, 0))
        add_lay.addRow("End:", self._add_end)
        self._add_btn = QPushButton("Add interval")
        self._add_btn.clicked.connect(self._add_interval_manual)
        add_lay.addRow(self._add_btn)
        ll.addWidget(add_grp)

        self._keep_unlabelled = QCheckBox("Keep unlabelled segments")
        self._keep_unlabelled.setChecked(False)
        ll.addWidget(self._keep_unlabelled)

        # --- Subdivide all ---
        self._subdivide_all_btn = QPushButton("Subdivide all intervals …")
        self._subdivide_all_btn.clicked.connect(self._subdivide_all)
        ll.addWidget(self._subdivide_all_btn)

        self._export_btn = QPushButton("Export labelled segments")
        self._export_btn.setStyleSheet("font-weight:bold; padding:8px;")
        self._export_btn.clicked.connect(self._export)
        ll.addWidget(self._export_btn)
        self._export_ecg_btn = QPushButton("Export ECG to existing segments")
        self._export_ecg_btn.clicked.connect(self._export_ecg_to_existing_segments)
        ll.addWidget(self._export_ecg_btn)
        self._export_hr_btn = QPushButton("Export HR to existing segments")
        self._export_hr_btn.clicked.connect(self._export_hr_to_existing_segments)
        ll.addWidget(self._export_hr_btn)

        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        ll.addWidget(self._progress)

        self._status = QLabel("")
        ll.addWidget(self._status)
        ll.addStretch()
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)

        self._right_content = QWidget()
        rl = QVBoxLayout(self._right_content)
        rl.setContentsMargins(4, 4, 4, 4)
        rl.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        self._player = MultiCameraPlayer()
        self._player.setMinimumHeight(300)
        rl.addWidget(self._player)

        self._plot_type = QComboBox()
        self._plot_type.addItems(["Heart rate", "ECG"])
        self._plot_type.currentIndexChanged.connect(self._show_selected_plot)
        rl.addWidget(self._plot_type)
        self._plot = SignalPlot()
        self._plot.setFixedHeight(420)
        rl.addWidget(self._plot)

        self._timeline = TimelineWidget()
        self._timeline.setFixedHeight(78)
        rl.addWidget(self._timeline)

        # --- Interval editor panel ---
        self._editor_grp = QGroupBox("Selected interval")
        ed_lay = QFormLayout(self._editor_grp)
        self._ed_label = QComboBox()
        self._ed_label.setEditable(True)
        ed_lay.addRow("Label:", self._ed_label)
        self._ed_start = QTimeEdit()
        self._ed_start.setDisplayFormat("HH:mm:ss.zzz")
        ed_lay.addRow("Start:", self._ed_start)
        self._ed_end = QTimeEdit()
        self._ed_end.setDisplayFormat("HH:mm:ss.zzz")
        ed_lay.addRow("End:", self._ed_end)
        self._ed_duration = QLabel("—")
        ed_lay.addRow("Duration:", self._ed_duration)
        self._ed_apply = QPushButton("Apply")
        self._ed_apply.clicked.connect(self._apply_editor)
        ed_lay.addRow(self._ed_apply)
        rl.addWidget(self._editor_grp)
        self._editor_grp.setVisible(False)

        self._right_scroll = QScrollArea()
        self._right_scroll.setWidgetResizable(True)
        self._right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._right_scroll.verticalScrollBar().setSingleStep(40)
        self._right_scroll.verticalScrollBar().setPageStep(280)
        self._right_scroll.setStyleSheet("QScrollBar:vertical { width: 18px; }")
        self._right_scroll.setWidget(self._right_content)
        splitter.addWidget(self._right_scroll)

        configure_main_splitter(splitter, left_scroll, self._right_scroll)

        self._timeline.interval_created.connect(self._on_interval_created)
        self._timeline.interval_deleted.connect(self._on_interval_deleted)
        self._timeline.interval_selected.connect(self._on_interval_selected)
        self._timeline.interval_relabelled.connect(self._on_interval_relabelled)
        self._timeline.interval_resized.connect(self._on_interval_resized)
        self._timeline.subdivide_requested.connect(self._on_subdivide_single)
        self._timeline.mosaic_requested.connect(self._on_mosaic_interval)
        self._timeline.playhead_moved.connect(self._on_playhead)
        self._player.frame_changed.connect(self._on_frame_changed)

        self._ed_start.timeChanged.connect(self._update_editor_duration)
        self._ed_end.timeChanged.connect(self._update_editor_duration)

        self.refresh_from_state()

    def refresh_from_state(self):
        self._lib_list.clear()
        self._label_combo.clear()
        for lbl in self.state.labels_library:
            self._lib_list.addItem(lbl)
            self._label_combo.addItem(lbl)
        for interval in self.state.intervals:
            interval.color = colour_for_label(interval.label, self.state.labels_library)

        self._refresh_from_tracks()
        self._sync_timeline()
        self._hr_merged_df = None
        self._ecg_merged_df = None
        if not self._load_synced_signal():
            self._load_signals()
        self._refresh_time_summary()

    def _load_signals(self):
        paths = self.state.hr_signal_paths or self.state.signal_paths
        if not paths:
            return
        dfs, _display_type_by_path, failed_paths = load_signal_files(paths)
        for p in failed_paths:
            log.warning("Failed to load signal: %s", p)
        if dfs:
            merged = pd.concat(dfs, ignore_index=True)

            # Clip to video time range if tracks are available
            if self.state.tracks:
                dt_start, coherence_warnings = compute_signal_anchor(
                    self.state.tracks,
                    self.state.time_coherence_tolerance_sec,
                )
                if coherence_warnings:
                    QMessageBox.warning(
                        self,
                        "Camera time correction warning",
                        format_time_coherence_warnings(coherence_warnings),
                    )
                self.state.last_signal_anchor_datetime = dt_start.isoformat() if dt_start else None
                self.state.last_time_coherence_warnings = coherence_warnings
                dur = video_timeline_duration_sec(self.state.tracks)
                if dt_start is not None:
                    dt_end = dt_start + timedelta(seconds=dur)
                    start_ts = pd.Timestamp(dt_start)
                    end_ts = pd.Timestamp(dt_end)
                    merged = merged[
                        (merged["timestamp_utc"] >= start_ts)
                        & (merged["timestamp_utc"] <= end_ts)
                    ]

            if merged.empty:
                return

            self._hr_merged_df = signal_long_to_wide(
                merged,
                include_average=False,
            )
            dur = video_timeline_duration_sec(self.state.tracks)
            self._show_selected_plot()

    def _load_synced_signal(self) -> bool:
        """Load pre-synced signal CSV from state. Returns True if loaded."""
        hr_paths = [path for path in self.state.synced_hr_paths if Path(path).exists()]
        ecg_paths = [path for path in self.state.synced_ecg_paths if Path(path).exists()]
        if not hr_paths and not ecg_paths:
            return False
        loaded = False
        if hr_paths:
            try:
                self._hr_merged_df = read_synced_signal_csvs(hr_paths)
                loaded = True
            except Exception as exc:
                log.warning("Could not load synchronized HR CSVs %s: %s", hr_paths, exc)
        if ecg_paths:
            try:
                self._ecg_merged_df = read_synced_signal_csvs(ecg_paths)
                loaded = True
            except Exception as exc:
                log.warning("Could not load synchronized ECG CSVs %s: %s", ecg_paths, exc)
        if loaded:
            self._show_selected_plot()
        return loaded

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
        self.refresh_from_state()
        n = len(self.state.tracks)
        self._load_status.setText(f"Loaded {n} camera(s) from sidecar.")

    def _import_segments_manifest(self):
        default_path = self.state.segments_manifest_path or str(
            Path(self.state.output_directory) / "labelled_segments" / "manifest.csv"
        )
        path, _ = QFileDialog.getOpenFileName(
            self, "Select labelled-segments manifest", default_path,
            "CSV (*.csv);;All files (*)",
        )
        if not path:
            return
        try:
            intervals = load_labelled_segments_manifest(path, self.state)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            QMessageBox.critical(self, "Invalid manifest", f"Failed to import labelled segments:\n{exc}")
            return
        self._lib_list.clear()
        self._label_combo.clear()
        for label in self.state.labels_library:
            self._lib_list.addItem(label)
            self._label_combo.addItem(label)
        for interval in intervals:
            interval.color = colour_for_label(interval.label, self.state.labels_library)
        self._sync_timeline()
        self._editor_grp.setVisible(False)
        self._load_status.setText(f"Imported {len(intervals)} labelled segment(s) from {Path(path).name}.")

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
        self._refresh_time_summary()
        n = len(self.state.tracks)
        self._load_status.setText(f"Loaded {n} video(s) directly.")

    def _add_signal_inline(self, signal_kind: str):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select signal files", "", _SIGNAL_FILTER,
        )
        if not files:
            return
        target = self.state.ecg_signal_paths if signal_kind == "ECG" else self.state.hr_signal_paths
        for p in files:
            is_ecg = signal_file_type_name(p) == "ECG waveform"
            if is_ecg != (signal_kind == "ECG"):
                QMessageBox.warning(self, "Wrong signal type", f"{Path(p).name} is not an {signal_kind} file.")
                continue
            if p not in target:
                target.append(p)
        self.state.signal_paths = list(self.state.hr_signal_paths)
        if signal_kind == "ECG":
            self._ecg_merged_df = self._load_and_clip_paths(self.state.ecg_signal_paths, False)
        else:
            self._load_signals()
        self._show_selected_plot()
        type_names = [signal_file_type_name(p) for p in files]
        self._load_status.setText(
            f"Loaded {len(target)} {signal_kind} file(s): {', '.join(type_names)}."
        )

    def _load_and_clip_paths(self, paths: list[str], include_average: bool) -> Optional[pd.DataFrame]:
        if not paths:
            return None
        dfs, _types, failed_paths = load_signal_files(paths)
        for path in failed_paths:
            log.warning("Failed to load signal: %s", path)
        if not dfs:
            return None
        merged = pd.concat(dfs, ignore_index=True)
        anchor = self._signal_anchor()
        if anchor is not None and self.state.tracks:
            end = pd.Timestamp(anchor + timedelta(seconds=video_timeline_duration_sec(self.state.tracks)))
            merged = merged[(merged["timestamp_utc"] >= pd.Timestamp(anchor)) & (merged["timestamp_utc"] <= end)]
        if merged.empty:
            return None
        return signal_long_to_wide(merged, include_average=include_average)

    def _signal_anchor(self) -> Optional[datetime]:
        if self.state.last_signal_anchor_datetime:
            try:
                return datetime.fromisoformat(self.state.last_signal_anchor_datetime)
            except ValueError:
                pass
        anchor, _warnings = compute_signal_anchor(self.state.tracks, self.state.time_coherence_tolerance_sec)
        return anchor

    def _show_selected_plot(self):
        df = self._hr_merged_df if self._plot_type.currentIndex() == 0 else self._ecg_merged_df
        if df is None:
            self._plot.clear()
            self._plot.set_intervals(self.state.intervals)
            return
        self._plot.set_data(df, video_duration_sec=video_timeline_duration_sec(self.state.tracks))
        self._plot.set_intervals(self.state.intervals)

    def _refresh_from_tracks(self):
        """Reload player and timeline from the current state.tracks."""
        labels = [t.camera_label for t in self.state.tracks]
        self._player.set_cameras(labels)
        camera_rows = max(1, (len(labels) + 1) // 2)
        player_height = camera_rows * 190 + 70
        self._player.setMinimumHeight(player_height)
        self._right_content.setMinimumHeight(player_height + self._plot.height() + 390)
        paths = [t.final_output_path for t in self.state.tracks]
        valid = [p for p in paths if p and Path(p).exists()]
        if valid:
            self._player.load_videos(valid)
        self._timeline.set_duration(video_timeline_duration_sec(self.state.tracks))

    def _clear_time_summary(self):
        while self._time_summary_grid.count():
            item = self._time_summary_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _refresh_time_summary(self):
        self._clear_time_summary()
        headers = ["Camera", "Mode", "Filename start", "Reference point", "Offset (s)", "Corrected start"]
        for col, text in enumerate(headers):
            label = QLabel(text)
            label.setStyleSheet("font-weight:bold;")
            self._time_summary_grid.addWidget(label, 0, col)
        if not self.state.tracks:
            self._time_summary_grid.addWidget(QLabel("No videos loaded."), 1, 0, 1, 6)
            return
        for row, track in enumerate(self.state.tracks, start=1):
            ref_dt = track.parsed_video_reference_datetime()
            reference_point = ""
            if track.time_correction_mode == "reference_video_time":
                reference_point = (
                    f"video {track.reference_video_time_sec:.3f}s -> "
                    f"{_format_dt(ref_dt)}"
                )
            values = [
                track.camera_label or f"Camera {row}",
                track.time_correction_mode,
                _format_dt(track.parsed_start_datetime()),
                reference_point,
                f"{track.time_correction_offset_sec:+.3f}",
                _format_dt(track.corrected_start_datetime()),
            ]
            for col, value in enumerate(values):
                label = QLabel(value)
                label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                self._time_summary_grid.addWidget(label, row, col)

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

        # Overlap validation
        if self._overlaps_existing(start_sec, end_sec):
            QMessageBox.warning(self, "Overlap",
                                "This interval overlaps with an existing one.")
            return

        color = colour_for_label(label, self.state.labels_library)
        iv = LabelInterval(label=label, start_sec=start_sec, end_sec=end_sec, color=color)
        self.state.intervals.append(iv)
        self._sync_timeline()

    def _on_interval_deleted(self, idx):
        if 0 <= idx < len(self.state.intervals):
            self.state.intervals.pop(idx)
            self._sync_timeline()
            self._editor_grp.setVisible(False)

    def _on_interval_selected(self, idx):
        if idx < 0 or idx >= len(self.state.intervals):
            self._editor_grp.setVisible(False)
            return
        iv = self.state.intervals[idx]
        self._populate_editor(iv)
        self._editor_grp.setVisible(True)

    def _on_interval_relabelled(self, idx, new_label):
        if 0 <= idx < len(self.state.intervals):
            self.state.intervals[idx].label = new_label
            color = colour_for_label(new_label, self.state.labels_library)
            self.state.intervals[idx].color = color
            self._sync_timeline()

    def _on_interval_resized(self, idx, new_start, new_end):
        """Handle edge-drag resize from the timeline widget."""
        if idx < 0 or idx >= len(self.state.intervals):
            return
        # Overlap check (excluding self)
        if self._overlaps_existing(new_start, new_end, exclude_idx=idx):
            # Revert: re-sync timeline with unchanged state
            self._sync_timeline()
            return
        iv = self.state.intervals[idx]
        iv.start_sec = new_start
        iv.end_sec = new_end
        self._sync_timeline()
        # Refresh the editor panel if this interval is selected
        if self._timeline.selected_index == idx:
            self._populate_editor(iv)

    # ------------------------------------------------------------------
    # Start / End label at playhead
    # ------------------------------------------------------------------

    def _start_label_at_playhead(self):
        sec = self._timeline._playhead_sec
        self._pending_start = sec
        self._start_label_btn.setChecked(True)
        label = self._label_combo.currentText().strip() or "Unlabelled"
        color = colour_for_label(label, self.state.labels_library)
        self._timeline.set_pending_start(sec, color)
        self._status.setText(f"Label start marked at {_format_duration(sec)}. "
                             "Play/seek to the end, then click 'End label'.")

    def _end_label_at_playhead(self):
        if self._pending_start is None:
            QMessageBox.information(self, "Info",
                                    "Click 'Start label' first to mark the start position.")
            return
        end_sec = self._timeline._playhead_sec
        start_sec = self._pending_start

        # Allow either order
        if start_sec > end_sec:
            start_sec, end_sec = end_sec, start_sec

        start_sec = round(start_sec, 3)
        end_sec = round(end_sec, 3)

        if end_sec - start_sec < 0.5:
            QMessageBox.warning(self, "Too short",
                                "The interval is shorter than 0.5 seconds.")
            return

        if self._overlaps_existing(start_sec, end_sec):
            QMessageBox.warning(self, "Overlap",
                                "This interval overlaps with an existing one.")
            return

        label = self._label_combo.currentText().strip() or "Unlabelled"
        color = colour_for_label(label, self.state.labels_library)
        iv = LabelInterval(label=label, start_sec=start_sec,
                           end_sec=end_sec, color=color)
        self.state.intervals.append(iv)
        self._sync_timeline()

        # Clear pending state
        self._pending_start = None
        self._start_label_btn.setChecked(False)
        self._timeline.set_pending_start(None)
        self._status.setText(
            f"Created '{label}' ({_format_duration(end_sec - start_sec)})."
        )

    # ------------------------------------------------------------------
    # Interval editor panel
    # ------------------------------------------------------------------

    def _populate_editor(self, iv: LabelInterval):
        self._ed_label.blockSignals(True)
        self._ed_label.clear()
        for lbl in self.state.labels_library:
            self._ed_label.addItem(lbl)
        idx = self._ed_label.findText(iv.label)
        if idx >= 0:
            self._ed_label.setCurrentIndex(idx)
        else:
            self._ed_label.setCurrentText(iv.label)
        self._ed_label.blockSignals(False)

        self._ed_start.blockSignals(True)
        self._ed_start.setTime(_secs_to_qtime(iv.start_sec))
        self._ed_start.blockSignals(False)

        self._ed_end.blockSignals(True)
        self._ed_end.setTime(_secs_to_qtime(iv.end_sec))
        self._ed_end.blockSignals(False)

        self._update_editor_duration()

    def _update_editor_duration(self):
        s = _qtime_to_secs(self._ed_start.time())
        e = _qtime_to_secs(self._ed_end.time())
        dur = max(0.0, e - s)
        self._ed_duration.setText(_format_duration(dur))

    def _apply_editor(self):
        idx = self._timeline.selected_index
        if idx < 0 or idx >= len(self.state.intervals):
            return

        new_label = self._ed_label.currentText().strip()
        new_start = round(_qtime_to_secs(self._ed_start.time()), 3)
        new_end = round(_qtime_to_secs(self._ed_end.time()), 3)

        if new_start >= new_end:
            QMessageBox.warning(self, "Invalid", "Start must be before end.")
            return

        if self._overlaps_existing(new_start, new_end, exclude_idx=idx):
            QMessageBox.warning(self, "Overlap",
                                "This interval overlaps with another existing interval.")
            return

        iv = self.state.intervals[idx]
        iv.label = new_label or iv.label
        iv.start_sec = new_start
        iv.end_sec = new_end
        iv.color = colour_for_label(iv.label, self.state.labels_library)
        self._sync_timeline()
        self._timeline.selected_index = idx
        self._status.setText(f"Updated interval #{idx}.")

    # ------------------------------------------------------------------
    # Manual interval creation
    # ------------------------------------------------------------------

    def _add_interval_manual(self):
        label = self._label_combo.currentText().strip()
        if not label:
            label = "Unlabelled"

        start = round(_qtime_to_secs(self._add_start.time()), 3)
        end = round(_qtime_to_secs(self._add_end.time()), 3)

        if start >= end:
            QMessageBox.warning(self, "Invalid", "Start must be before end.")
            return

        if self._overlaps_existing(start, end):
            QMessageBox.warning(self, "Overlap",
                                "This interval overlaps with an existing one.")
            return

        color = colour_for_label(label, self.state.labels_library)
        iv = LabelInterval(label=label, start_sec=start, end_sec=end, color=color)
        self.state.intervals.append(iv)
        self._sync_timeline()
        self._status.setText(f"Added interval '{label}' ({_format_duration(end - start)}).")

    # ------------------------------------------------------------------
    # Overlap validation
    # ------------------------------------------------------------------

    def _overlaps_existing(self, start: float, end: float, exclude_idx: int = -1) -> bool:
        for i, other in enumerate(self.state.intervals):
            if i == exclude_idx:
                continue
            if start < other.end_sec and end > other.start_sec:
                return True
        return False

    # ------------------------------------------------------------------
    # Subdivision
    # ------------------------------------------------------------------

    def _on_subdivide_single(self, idx):
        if idx < 0 or idx >= len(self.state.intervals):
            return
        iv = self.state.intervals[idx]
        dlg = SubdivideDialog(iv.label, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        mode, value = dlg.result_values()
        self._do_subdivide([idx], mode, value)

    def _subdivide_all(self):
        if not self.state.intervals:
            QMessageBox.information(self, "Info", "No intervals to subdivide.")
            return
        dlg = SubdivideDialog("all intervals", parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        mode, value = dlg.result_values()
        indices = list(range(len(self.state.intervals)))
        self._do_subdivide(indices, mode, value)

    def _do_subdivide(self, indices: list[int], mode: str, value: float):
        """Replace intervals at *indices* with sub-intervals."""
        # Process in reverse so index shifting doesn't matter
        new_intervals = list(self.state.intervals)
        affected_labels: list[str] = []
        for idx in sorted(indices, reverse=True):
            iv = new_intervals[idx]
            color = colour_for_label(iv.label, self.state.labels_library)
            subs = _subdivide_interval(iv, mode, value, color)
            if len(subs) < 2:
                continue
            affected_labels.append(iv.label)
            new_intervals[idx:idx + 1] = subs

        self.state.intervals = new_intervals
        self._sync_timeline()
        self._editor_grp.setVisible(False)

        n_new = sum(1 for _ in new_intervals)
        self._status.setText(
            f"Subdivided {len(affected_labels)} interval(s) → {n_new} total."
        )

        # Auto re-export if a previous export exists
        if self._has_exported:
            reply = QMessageBox.question(
                self, "Re-export?",
                "Export has been run previously. Re-export all segments now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._export()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sync_timeline(self):
        """Push current state.intervals to the timeline and signal plot."""
        items = [
            IntervalItem(i.label, i.start_sec, i.end_sec, i.color)
            for i in self.state.intervals
        ]
        self._timeline.set_intervals(items)
        self._plot.set_intervals(self.state.intervals)

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

        intervals = list(self.state.intervals)
        if not intervals:
            QMessageBox.information(self, "Info", "No labelled intervals to export.")
            return

        if not self._keep_unlabelled.isChecked():
            intervals = [iv for iv in intervals if iv.label != "Unlabelled"]

        # Sort by start time for consistent ordering
        intervals.sort(key=lambda iv: iv.start_sec)

        ffmpeg = self.state.ffmpeg_path or find_ffmpeg()
        if not ffmpeg:
            QMessageBox.critical(self, "Error", "FFmpeg not found.")
            return

        segments_dir = Path(out_dir) / "labelled_segments"
        segments_dir.mkdir(parents=True, exist_ok=True)

        signal_anchor = None
        coherence_warnings: list[str] = []
        if self.state.last_signal_anchor_datetime:
            try:
                signal_anchor = datetime.fromisoformat(self.state.last_signal_anchor_datetime)
                coherence_warnings = list(self.state.last_time_coherence_warnings)
            except ValueError:
                signal_anchor = None
        if signal_anchor is None:
            signal_anchor, coherence_warnings = compute_signal_anchor(
                self.state.tracks,
                self.state.time_coherence_tolerance_sec,
            )
        if coherence_warnings:
            QMessageBox.warning(
                self,
                "Camera time correction warning",
                format_time_coherence_warnings(coherence_warnings),
            )

        manifest_rows = []
        total = len(intervals)

        for idx, iv in enumerate(intervals):
            self._progress.setValue(int((idx / total) * 100))
            seg_name = f"{idx:04d}_{iv.label.replace(' ', '_')}"
            seg_dir = segments_dir / seg_name
            seg_dir.mkdir(parents=True, exist_ok=True)
            iv.folder = str(seg_dir)

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

            hr_files = self._export_signal_slice(
                self._hr_merged_df, signal_anchor, iv, seg_dir, "heart_rate_bpm"
            )
            ecg_files = self._export_signal_slice(
                self._ecg_merged_df, signal_anchor, iv, seg_dir, "ecg_waveform"
            )

            # Build enriched meta.json (aligned with project_meta.json)
            cameras_meta = []
            for ti, track in enumerate(self.state.tracks):
                cam_file = f"cam{ti+1}_{track.camera_label}.mp4"
                corrected_start = track.corrected_start_datetime()
                cameras_meta.append({
                    "label": track.camera_label,
                    "video_file": cam_file,
                    "fps": track.fps,
                    "dimensions": [track.width, track.height],
                    "codec": track.codec,
                    "time_correction_mode": track.time_correction_mode,
                    "time_correction_offset_sec": track.time_correction_offset_sec,
                    "true_start_datetime_utc": track.true_start_datetime,
                    "reference_video_time_sec": track.reference_video_time_sec,
                    "video_reference_datetime_utc": track.video_reference_datetime,
                    "corrected_start_datetime_utc": corrected_start.isoformat() if corrected_start else None,
                })

            synced_hr_rel = self._relative_signal_paths(self.state.synced_hr_paths, seg_dir)
            synced_ecg_rel = self._relative_signal_paths(self.state.synced_ecg_paths, seg_dir)

            meta = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "index": idx,
                "label": iv.label,
                "start_sec": iv.start_sec,
                "end_sec": iv.end_sec,
                "duration_sec": iv.end_sec - iv.start_sec,
                "folder": seg_name,
                "cameras": cameras_meta,
                "hr_files": hr_files,
                "ecg_files": ecg_files,
                "synced_hr_paths": synced_hr_rel,
                "synced_ecg_paths": synced_ecg_rel,
                "signal_anchor_datetime_utc": signal_anchor.isoformat() if signal_anchor else None,
                "segment_start_datetime_utc": (
                    (signal_anchor + timedelta(seconds=iv.start_sec)).isoformat()
                    if signal_anchor else None
                ),
                "segment_end_datetime_utc": (
                    (signal_anchor + timedelta(seconds=iv.end_sec)).isoformat()
                    if signal_anchor else None
                ),
                "time_coherence_tolerance_sec": self.state.time_coherence_tolerance_sec,
                "time_coherence_warnings": coherence_warnings,
            }
            (seg_dir / "meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            manifest_rows.append({
                "index": idx,
                "label": iv.label,
                "start_sec": iv.start_sec,
                "end_sec": iv.end_sec,
                "duration_sec": iv.end_sec - iv.start_sec,
                "folder": seg_name,
            })

        manifest_path = segments_dir / "manifest.csv"
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "label", "start_sec", "end_sec", "duration_sec", "folder"])
            writer.writeheader()
            writer.writerows(manifest_rows)
        self.state.segments_manifest_path = str(manifest_path)

        self._progress.setValue(100)
        self._has_exported = True
        self._status.setText(f"Exported {len(manifest_rows)} segments to {segments_dir}")
        log.info("Exported %d segments to %s", len(manifest_rows), segments_dir)

    @staticmethod
    def _relative_signal_path(path: str, seg_dir: Path) -> Optional[str]:
        if not path or not Path(path).exists():
            return None
        try:
            return os.path.relpath(path, seg_dir)
        except ValueError:
            return path

    @classmethod
    def _relative_signal_paths(cls, paths: list[str], seg_dir: Path) -> list[str]:
        return [
            relative for path in paths
            if (relative := cls._relative_signal_path(path, seg_dir)) is not None
        ]

    @staticmethod
    def _export_signal_slice(df, anchor, interval, destination: Path, primary_column: str) -> list[str]:
        if df is None or df.empty or anchor is None:
            return []
        start = pd.Timestamp(anchor + timedelta(seconds=interval.start_sec))
        end = pd.Timestamp(anchor + timedelta(seconds=interval.end_sec))
        sub = df[(df["timestamp_utc"] >= start) & (df["timestamp_utc"] <= end)]
        if sub.empty:
            return []
        return [
            Path(path).name
            for path in write_synced_signal_csvs(sub, destination, primary_column)
        ]

    def _export_ecg_to_existing_segments(self):
        self._export_signal_to_existing_segments(
            "ECG", "ecg", "synced_ecg_paths", "ecg_waveform"
        )

    def _export_hr_to_existing_segments(self):
        self._export_signal_to_existing_segments(
            "HR", "hr", "synced_hr_paths", "heart_rate_bpm"
        )

    def _export_signal_to_existing_segments(
        self,
        display_name: str,
        file_stem: str,
        state_paths_name: str,
        primary_column: str,
    ):
        df = self._ecg_merged_df if file_stem == "ecg" else self._hr_merged_df
        if df is None and not self._load_synced_signal():
            QMessageBox.warning(self, f"{display_name} unavailable", f"Load a project metadata file with a synchronized {display_name} CSV first.")
            return
        df = self._ecg_merged_df if file_stem == "ecg" else self._hr_merged_df
        if df is None:
            QMessageBox.warning(self, f"{display_name} unavailable", f"No synchronized {display_name} CSV is available.")
            return
        intervals = [interval for interval in self.state.intervals if interval.folder]
        if not intervals:
            QMessageBox.information(
                self, "No existing segments",
                "Import a labelled-segments manifest or export labelled segments first.",
            )
            return
        anchor = self._signal_anchor()
        if anchor is None:
            QMessageBox.warning(self, f"{display_name} unavailable", "Video time anchor is unavailable.")
            return
        count = 0
        for interval in intervals:
            seg_dir = Path(interval.folder)
            if not seg_dir.exists():
                continue
            file_names = self._export_signal_slice(
                df, anchor, interval, seg_dir, primary_column
            )
            if not file_names:
                continue
            meta_path = seg_dir / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            meta[f"{file_stem}_files"] = file_names
            meta[state_paths_name] = self._relative_signal_paths(
                getattr(self.state, state_paths_name), seg_dir
            )
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            count += 1
        self._status.setText(f"Exported {display_name} to {count} existing segment(s).")

    # ------------------------------------------------------------------
    # Mosaic export for selected interval
    # ------------------------------------------------------------------

    def _on_mosaic_interval(self, idx: int):
        if idx < 0 or idx >= len(self.state.intervals):
            return
        iv = self.state.intervals[idx]

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

        safe_label = iv.label.replace(" ", "_")
        default_name = str(Path(out_dir) / f"mosaic_{idx:04d}_{safe_label}.mp4")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save mosaic video", default_name,
            "MP4 video (*.mp4);;All files (*)",
        )
        if not path:
            return

        preset_choice, ok = QInputDialog.getItem(
            self,
            "Mosaic export preset",
            "Speed vs quality:",
            MOSAIC_PRESET_NAMES,
            ({"speed": 0, "balanced": 1, "quality": 2}.get(
                normalise_mosaic_preset(self.state.mosaic_preset), 1
            )),
            False,
        )
        if not ok:
            return
        preset_key = normalise_mosaic_preset(preset_choice)
        self.state.mosaic_preset = preset_key

        t0 = self.state.tracks[0]
        fps = t0.fps or 30.0
        duration = iv.end_sec - iv.start_sec
        total_frames = int(duration * fps)
        labels = [t.camera_label for t in self.state.tracks]

        self._mosaic_worker = MosaicWorker(
            video_paths=valid,
            camera_labels=labels,
            fps=fps,
            total_frames=total_frames,
            video_duration_sec=duration,
            output_path=path,
            signal_df=self._hr_merged_df,
            start_sec=iv.start_sec,
            end_sec=iv.end_sec,
            quality_preset=preset_key,
            ffmpeg_path=self.state.ffmpeg_path,
        )
        self._mosaic_worker.progress.connect(self._on_mosaic_progress)
        self._mosaic_worker.finished.connect(self._on_mosaic_finished)
        self._export_btn.setEnabled(False)
        self._progress.setValue(0)
        self._status.setText(f"Exporting mosaic for '{iv.label}' ({preset_choice})\u2026")
        self._mosaic_worker.start()

    def _on_mosaic_progress(self, current: int, total: int):
        if total > 0:
            self._progress.setValue(int(current * 100 / total))
        self._status.setText(f"Mosaic: frame {current} / {total}")

    def _on_mosaic_finished(self, success: bool, msg: str):
        self._export_btn.setEnabled(True)
        if success:
            self._progress.setValue(100)
            self._status.setText(f"Mosaic saved: {Path(msg).name}")
            QMessageBox.information(self, "Mosaic export", f"Mosaic video saved:\n{msg}")
        else:
            self._status.setText(f"Mosaic failed: {msg}")
            if "Cancelled" not in msg:
                QMessageBox.critical(self, "Error", f"Mosaic export failed:\n{msg}")
        self._mosaic_worker = None
