from __future__ import annotations

import csv
import json
import logging
import math
import os
import shutil
from datetime import timedelta
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
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from ..ffmpeg_utils import trim_video, find_ffmpeg
from ..signals import load_signal
from ..state import LabelInterval, ProjectState, load_sidecar, populate_tracks_from_videos
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
        self._merged_df: Optional[pd.DataFrame] = None
        self._has_exported: bool = False

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

        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        self._timeline.interval_created.connect(self._on_interval_created)
        self._timeline.interval_deleted.connect(self._on_interval_deleted)
        self._timeline.interval_selected.connect(self._on_interval_selected)
        self._timeline.interval_relabelled.connect(self._on_interval_relabelled)
        self._timeline.interval_resized.connect(self._on_interval_resized)
        self._timeline.subdivide_requested.connect(self._on_subdivide_single)
        self._timeline.playhead_moved.connect(self._on_playhead)
        self._player.frame_changed.connect(self._on_frame_changed)

        self._ed_start.timeChanged.connect(self._update_editor_duration)
        self._ed_end.timeChanged.connect(self._update_editor_duration)

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
        """Push current state.intervals to the timeline widget."""
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
        self._has_exported = True
        self._status.setText(f"Exported {len(manifest_rows)} segments to {segments_dir}")
        log.info("Exported %d segments to %s", len(manifest_rows), segments_dir)
