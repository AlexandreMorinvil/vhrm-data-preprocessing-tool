from __future__ import annotations

import copy
import csv
import json
import logging
import math
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from PyQt6.QtCore import QSettings, Qt, QTime
from PyQt6.QtGui import QBrush, QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from ..file_cleanup import cleanup_obsolete_paths
from ..face_privacy import export_anonymized_video_segment
from ..ffmpeg_utils import FFmpegWorker, trim_video, find_ffmpeg
from ..mosaic_export import MosaicWorker
from ..signals import (
    load_signal_files,
    read_synced_signal_csvs,
    signal_long_to_wide,
    write_synced_signal_csvs,
)
from ..state import (
    LabelInterval,
    ProjectState,
    compute_signal_anchor,
    effective_signal_anchor,
    format_time_coherence_warnings,
    load_labelled_segments_manifest,
    load_sidecar,
    video_timeline_duration_sec,
)
from ..timefmt import format_hms_ms
from ..widgets.frame_preview import MultiCameraPlayer
from ..widgets.layout import ModeWorkspace
from ..widgets.signal_plot import SignalPlot
from ..widgets.subdivide_dialog import SubdivideDialog
from ..widgets.timeline import IntervalItem, TimelineWidget, colour_for_label
from .common import ask_mosaic_options, export_frames_interactive, export_snapshot_interactive

log = logging.getLogger(__name__)

_META_FILTER = "Metadata sidecar (*.json);;All files (*)"
_SYNTHETIC_ARTIFACT_FIELDS = (
    "synthetic_ppg_path",
    "synthetic_ppg_hr_path",
    "synthetic_ppg_hrv_path",
    "synthetic_rr_path",
)
_UNDO_LIMIT = 100


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
        self._ppg_merged_df: Optional[pd.DataFrame] = None
        self._has_exported: bool = False
        self._mosaic_worker: MosaicWorker | None = None
        self._export_worker: FFmpegWorker | None = None
        self._pending_start: Optional[float] = None
        self._selected: Optional[LabelInterval] = None
        self._undo: list[tuple[list[LabelInterval], list[str]]] = []
        self._redo: list[tuple[list[LabelInterval], list[str]]] = []
        self._figure_prefer_selected = False
        self._settings = QSettings("VideoResearchTool", "VRT")

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self._workspace = ModeWorkspace("labelling", self)
        root.addWidget(self._workspace)
        ws = self._workspace

        # ---------------- Project data ----------------
        load_box = QWidget()
        load_lay = QVBoxLayout(load_box)
        load_lay.setContentsMargins(0, 0, 0, 0)
        meta_btn = QPushButton("Load from metadata file …")
        meta_btn.clicked.connect(self._load_from_meta)
        load_lay.addWidget(meta_btn)
        manifest_btn = QPushButton("Import labelled segments manifest …")
        manifest_btn.clicked.connect(self._import_segments_manifest)
        load_lay.addWidget(manifest_btn)
        self._load_status = QLabel("")
        self._load_status.setWordWrap(True)
        load_lay.addWidget(self._load_status)
        ws.add_section("Project data", load_box)

        # ---------------- Labels ----------------
        lib_box = QWidget()
        lib_lay = QVBoxLayout(lib_box)
        lib_lay.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("Keys 1–9 pick a label. I = start, O = end (creates the interval).")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8a8a8a;")
        lib_lay.addWidget(hint)
        self._lib_list = QListWidget()
        self._lib_list.setMaximumHeight(170)
        self._lib_list.currentRowChanged.connect(self._on_library_row)
        lib_lay.addWidget(self._lib_list)
        lbtn_row = QHBoxLayout()
        add_lb = QPushButton("Add label")
        add_lb.clicked.connect(self._add_label)
        lbtn_row.addWidget(add_lb)
        rename_lb = QPushButton("Rename")
        rename_lb.clicked.connect(self._rename_label)
        lbtn_row.addWidget(rename_lb)
        rm_lb = QPushButton("Remove")
        rm_lb.clicked.connect(self._remove_label)
        lbtn_row.addWidget(rm_lb)
        lib_lay.addLayout(lbtn_row)
        ws.add_section("Label library", lib_box)

        # ---------------- Intervals ----------------
        iv_box = QWidget()
        iv_lay = QVBoxLayout(iv_box)
        iv_lay.setContentsMargins(0, 0, 0, 0)
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["#", "Label", "Start", "End", "Duration"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.setMinimumHeight(160)
        self._table.itemSelectionChanged.connect(self._on_table_selection)
        self._table.cellDoubleClicked.connect(self._on_table_double_click)
        iv_lay.addWidget(self._table)
        undo_row = QHBoxLayout()
        self._undo_btn = QPushButton("Undo")
        self._undo_btn.setToolTip("Undo the last interval change (Ctrl+Z)")
        self._undo_btn.clicked.connect(self._undo_action)
        self._redo_btn = QPushButton("Redo")
        self._redo_btn.setToolTip("Redo (Ctrl+Y)")
        self._redo_btn.clicked.connect(self._redo_action)
        undo_row.addWidget(self._undo_btn)
        undo_row.addWidget(self._redo_btn)
        self._subdivide_all_btn = QPushButton("Subdivide all …")
        self._subdivide_all_btn.clicked.connect(self._subdivide_all)
        undo_row.addWidget(self._subdivide_all_btn)
        iv_lay.addLayout(undo_row)

        # Selected interval editor
        self._editor_grp = QWidget()
        ed_lay = QFormLayout(self._editor_grp)
        ed_lay.setContentsMargins(0, 4, 0, 0)
        self._ed_label = QComboBox()
        self._ed_label.setEditable(True)
        ed_lay.addRow("Label:", self._ed_label)
        start_row = QHBoxLayout()
        self._ed_start = QTimeEdit()
        self._ed_start.setDisplayFormat("HH:mm:ss.zzz")
        start_row.addWidget(self._ed_start, 1)
        start_here = QPushButton("← playhead")
        start_here.setToolTip("Set the start to the current playhead")
        start_here.clicked.connect(lambda: self._ed_start.setTime(_secs_to_qtime(self._timeline.playhead_sec)))
        start_row.addWidget(start_here)
        ed_lay.addRow("Start:", start_row)
        end_row = QHBoxLayout()
        self._ed_end = QTimeEdit()
        self._ed_end.setDisplayFormat("HH:mm:ss.zzz")
        end_row.addWidget(self._ed_end, 1)
        end_here = QPushButton("← playhead")
        end_here.setToolTip("Set the end to the current playhead")
        end_here.clicked.connect(lambda: self._ed_end.setTime(_secs_to_qtime(self._timeline.playhead_sec)))
        end_row.addWidget(end_here)
        ed_lay.addRow("End:", end_row)
        self._ed_duration = QLabel("—")
        ed_lay.addRow("Duration:", self._ed_duration)
        ed_buttons = QHBoxLayout()
        self._ed_apply = QPushButton("Apply")
        self._ed_apply.clicked.connect(self._apply_editor)
        ed_buttons.addWidget(self._ed_apply)
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self._delete_selected)
        ed_buttons.addWidget(delete_btn)
        ed_lay.addRow(ed_buttons)
        iv_lay.addWidget(self._editor_grp)
        self._editor_grp.setVisible(False)
        ws.add_section("Intervals", iv_box)

        # Manual interval creation
        add_box = QWidget()
        add_lay = QFormLayout(add_box)
        add_lay.setContentsMargins(0, 0, 0, 0)
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
        ws.add_section("Add interval manually", add_box, expanded=False)

        # ---------------- Export ----------------
        exp_box = QWidget()
        exp_lay = QVBoxLayout(exp_box)
        exp_lay.setContentsMargins(0, 0, 0, 0)
        self._keep_unlabelled = QCheckBox("Keep unlabelled segments")
        self._keep_unlabelled.setChecked(False)
        exp_lay.addWidget(self._keep_unlabelled)
        self._remove_legacy_csv_cb = QCheckBox("Remove legacy combined signal.csv files after export")
        self._remove_legacy_csv_cb.setToolTip(
            "Removes an old combined signal.csv only after replacement per-sensor "
            "heart-rate files are exported successfully."
        )
        exp_lay.addWidget(self._remove_legacy_csv_cb)
        self._export_btn = QPushButton("Export labelled segments")
        self._export_btn.setStyleSheet("font-weight:bold; padding:6px;")
        self._export_btn.setToolTip(
            "Cut every labelled interval losslessly (stream copy) from each camera, "
            "plus the synchronized signals. With face blurring enabled the videos are re-encoded."
        )
        self._export_btn.clicked.connect(self._export)
        exp_lay.addWidget(self._export_btn)
        self._export_signals_btn = QPushButton("Export all signals to existing segments")
        self._export_signals_btn.clicked.connect(self._export_signals_to_existing_segments)
        exp_lay.addWidget(self._export_signals_btn)
        prog_row = QHBoxLayout()
        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        prog_row.addWidget(self._progress, 1)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel_jobs)
        prog_row.addWidget(self._cancel_btn)
        exp_lay.addLayout(prog_row)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        exp_lay.addWidget(self._status)
        ws.add_section("Export", exp_box)

        time_box = QWidget()
        self._time_summary_grid = QGridLayout(time_box)
        self._time_summary_grid.setContentsMargins(0, 0, 0, 0)
        ws.add_section("Camera time correction", time_box, expanded=False)
        ws.finish_left()

        # ---------------- Work area ----------------
        self._player = MultiCameraPlayer(face_blur_enabled=state.blur_faces)
        self._label_combo = QComboBox()
        self._label_combo.setEditable(True)
        self._label_combo.setMinimumWidth(150)
        self._label_combo.setToolTip("Label for new intervals (keys 1–9)")
        self._start_label_btn = QPushButton("Start label ▶ (I)")
        self._start_label_btn.setCheckable(True)
        self._start_label_btn.setToolTip("Mark current playhead position as label start (I)")
        self._start_label_btn.clicked.connect(self._start_label_at_playhead)
        self._end_label_btn = QPushButton("End label ■ (O)")
        self._end_label_btn.setToolTip("Mark current playhead position as label end (O)")
        self._end_label_btn.clicked.connect(self._end_label_at_playhead)
        label_caption = QLabel("New interval label:")
        key_hint = QLabel("Keys: 1–9 label • I/O start/end • S split • R relabel • N/P next/prev • Del delete • Ctrl+Z undo")
        key_hint.setStyleSheet("color: #8a8a8a;")
        for widget in (label_caption, self._label_combo, self._start_label_btn, self._end_label_btn, key_hint):
            self._player.add_transport_widget(widget)
        ws.add_work(self._player, 520)

        signal_area = QWidget()
        sig_lay = QVBoxLayout(signal_area)
        sig_lay.setContentsMargins(0, 0, 0, 0)
        sig_lay.setSpacing(2)
        toggles = QHBoxLayout()
        toggles.addWidget(QLabel("Show:"))
        self._show_hr = QCheckBox("Heart rate")
        self._show_ecg = QCheckBox("ECG")
        self._show_ppg = QCheckBox("Synthetic PPG")
        for key, box, default in (("hr", self._show_hr, True), ("ecg", self._show_ecg, True),
                                  ("ppg", self._show_ppg, False)):
            box.setChecked(self._settings.value(f"labelling/show_{key}", default, type=bool))
            box.toggled.connect(lambda checked, k=key: (
                self._settings.setValue(f"labelling/show_{k}", checked), self._show_selected_plot()))
            toggles.addWidget(box)
        toggles.addSpacing(16)
        self._follow_cb = QCheckBox("Follow playhead")
        self._follow_cb.setChecked(True)
        self._follow_cb.setToolTip("Scroll the timeline and graphs with the playhead when zoomed in")
        self._follow_cb.toggled.connect(lambda checked: self._timeline.set_follow_playhead(checked))
        toggles.addWidget(self._follow_cb)
        tip = QLabel("Wheel: zoom • drag: pan • click: seek • Ctrl+drag: zoom to range • Shift+drag: new interval")
        tip.setStyleSheet("color: #8a8a8a;")
        toggles.addStretch()
        toggles.addWidget(tip)
        sig_lay.addLayout(toggles)
        self._plot = SignalPlot()
        self._plot.interval_drawing_enabled = True
        self._plot.figure_export_context = self._figure_context
        sig_lay.addWidget(self._plot, 1)
        ws.add_work(signal_area, 330)

        self._timeline = TimelineWidget()
        self._timeline.setMinimumHeight(84)
        ws.add_work(self._timeline, 96)

        # ---------------- Connections ----------------
        self._timeline.interval_created.connect(self._on_interval_created)
        self._timeline.interval_deleted.connect(self._on_interval_deleted)
        self._timeline.interval_selected.connect(self._on_interval_selected)
        self._timeline.interval_relabelled.connect(self._on_interval_relabelled)
        self._timeline.interval_resized.connect(self._on_interval_resized)
        self._timeline.subdivide_requested.connect(self._on_subdivide_single)
        self._timeline.mosaic_requested.connect(self._on_mosaic_interval)
        self._timeline.figure_requested.connect(self._on_figure_interval)
        self._timeline.split_requested.connect(self._split_interval)
        self._timeline.playhead_moved.connect(self._on_playhead)
        self._timeline.view_range_changed.connect(self._plot.set_view_range)
        self._plot.view_range_changed.connect(lambda s, e: self._timeline.set_view_range(s, e))
        self._plot.plot_area_changed.connect(self._timeline.set_content_margins)
        self._plot.seek_requested.connect(self._on_playhead)
        self._plot.interval_drawn.connect(self._on_interval_created)
        self._player.frame_changed.connect(self._on_frame_changed)
        self._player.position_changed.connect(self._on_position_changed)
        self._player.export_frames_requested.connect(self._export_current_frames)
        self._player.snapshot_requested.connect(self._export_snapshot)
        self._ed_start.timeChanged.connect(self._update_editor_duration)
        self._ed_end.timeChanged.connect(self._update_editor_duration)

        self._player.install_shortcuts(self)
        for key, slot in (
            ("I", self._start_label_at_playhead),
            ("O", self._end_label_at_playhead),
            ("Escape", self._cancel_pending),
            ("Delete", self._delete_selected),
            ("Ctrl+Z", self._undo_action),
            ("Ctrl+Y", self._redo_action),
            ("Ctrl+Shift+Z", self._redo_action),
            ("S", self._split_at_playhead),
            ("R", self._relabel_selected_with_current),
            ("N", lambda: self._jump_interval(1)),
            ("P", lambda: self._jump_interval(-1)),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)
        for number in range(1, 10):
            shortcut = QShortcut(QKeySequence(str(number)), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda n=number: self._choose_label(n - 1))

        self._update_undo_buttons()
        self.refresh_from_state()

    # ------------------------------------------------------------------
    # State refresh
    # ------------------------------------------------------------------

    def refresh_from_state(self):
        self._refresh_label_lists()
        for interval in self.state.intervals:
            interval.color = colour_for_label(interval.label, self.state.labels_library)

        self._refresh_from_tracks()
        self._sync_timeline()
        self._hr_merged_df = None
        self._ecg_merged_df = None
        self._ppg_merged_df = None
        if not self._load_synced_signal():
            self._load_signals()
        self._refresh_time_summary()

    def _refresh_label_lists(self) -> None:
        current = self._label_combo.currentText()
        self._lib_list.blockSignals(True)
        self._lib_list.clear()
        self._label_combo.clear()
        for index, lbl in enumerate(self.state.labels_library):
            prefix = f"{index + 1}  " if index < 9 else "    "
            self._lib_list.addItem(f"{prefix}{lbl}")
            self._label_combo.addItem(lbl)
        self._lib_list.blockSignals(False)
        if current:
            idx = self._label_combo.findText(current)
            if idx >= 0:
                self._label_combo.setCurrentIndex(idx)
        self._timeline.set_label_choices(self.state.labels_library)

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
            self._show_selected_plot()

    def _load_synced_signal(self) -> bool:
        """Load pre-synced signal CSV from state. Returns True if loaded."""
        hr_paths = [path for path in self.state.synced_hr_paths if Path(path).exists()]
        ecg_paths = [path for path in self.state.synced_ecg_paths if Path(path).exists()]
        ppg_paths = (
            [self.state.synthetic_ppg_path]
            if self.state.synthetic_ppg_path and Path(self.state.synthetic_ppg_path).exists()
            else []
        )
        if not hr_paths and not ecg_paths and not ppg_paths:
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
        if ppg_paths:
            try:
                self._ppg_merged_df = read_synced_signal_csvs(ppg_paths)
                loaded = True
            except Exception as exc:
                log.warning("Could not load synthetic PPG CSV %s: %s", ppg_paths, exc)
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
        self._push_undo()
        try:
            intervals = load_labelled_segments_manifest(path, self.state)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            self._undo.pop()
            QMessageBox.critical(self, "Invalid manifest", f"Failed to import labelled segments:\n{exc}")
            return
        self._refresh_label_lists()
        for interval in intervals:
            interval.color = colour_for_label(interval.label, self.state.labels_library)
        self._select(None)
        self._sync_timeline()
        self._load_status.setText(f"Imported {len(intervals)} labelled segment(s) from {Path(path).name}.")

    def _signal_anchor(self) -> Optional[datetime]:
        if self.state.last_signal_anchor_datetime:
            try:
                return datetime.fromisoformat(self.state.last_signal_anchor_datetime)
            except ValueError:
                pass
        anchor, _warnings = compute_signal_anchor(self.state.tracks, self.state.time_coherence_tolerance_sec)
        return anchor

    def _show_selected_plot(self):
        panels = []
        if self._show_hr.isChecked():
            panels.append(("Heart rate", self._hr_merged_df, "BPM"))
        if self._show_ecg.isChecked():
            panels.append(("ECG", self._ecg_merged_df, "ECG"))
        if self._show_ppg.isChecked():
            panels.append(("Synthetic PPG", self._ppg_merged_df, "PPG (a.u.)"))
        panels = [p for p in panels if p[1] is not None and not p[1].empty]
        self._plot.set_time_zero(effective_signal_anchor(self.state) if self.state.tracks else None)
        if not panels:
            self._plot.clear()
        else:
            self._plot.set_panels(panels, video_duration_sec=video_timeline_duration_sec(self.state.tracks))
        self._plot.set_intervals(self.state.intervals)
        self._plot.set_cursor(self._timeline.playhead_sec)

    def _refresh_from_tracks(self):
        """Reload player and timeline from the current state.tracks."""
        labels = [t.camera_label for t in self.state.tracks]
        self._player.set_cameras(labels)
        paths = [t.final_output_path for t in self.state.tracks]
        valid = [p for p in paths if p and Path(p).exists()]
        if valid:
            self._player.load_videos(valid)
        self._player.set_clock_anchor(effective_signal_anchor(self.state) if self.state.tracks else None)
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

    # ------------------------------------------------------------------
    # Label library
    # ------------------------------------------------------------------

    def _add_label(self):
        text, ok = QInputDialog.getText(self, "New label", "Label name:")
        text = text.strip() if ok else ""
        if text and text not in self.state.labels_library:
            self.state.labels_library.append(text)
            self._refresh_label_lists()
            self._label_combo.setCurrentText(text)

    def _rename_label(self):
        row = self._lib_list.currentRow()
        if row < 0 or row >= len(self.state.labels_library):
            QMessageBox.information(self, "Rename label", "Select a label in the library first.")
            return
        old = self.state.labels_library[row]
        new, ok = QInputDialog.getText(self, "Rename label", f"New name for “{old}”:", text=old)
        new = new.strip() if ok else ""
        if not new or new == old:
            return
        if new in self.state.labels_library:
            QMessageBox.warning(self, "Rename label", f"“{new}” already exists.")
            return
        self._push_undo()
        self.state.labels_library[row] = new
        renamed = 0
        for interval in self.state.intervals:
            if interval.label == old:
                interval.label = new
                renamed += 1
        self._refresh_label_lists()
        self._sync_timeline()
        self._status.setText(f"Renamed “{old}” to “{new}” ({renamed} interval(s) updated).")

    def _remove_label(self):
        row = self._lib_list.currentRow()
        if row < 0 or row >= len(self.state.labels_library):
            return
        self.state.labels_library.pop(row)
        self._refresh_label_lists()

    def _on_library_row(self, row: int) -> None:
        if 0 <= row < self._label_combo.count():
            self._label_combo.setCurrentIndex(row)

    def _choose_label(self, index: int) -> None:
        if 0 <= index < len(self.state.labels_library):
            self._label_combo.setCurrentIndex(index)
            self._lib_list.blockSignals(True)
            self._lib_list.setCurrentRow(index)
            self._lib_list.blockSignals(False)
            self._status.setText(f"Current label: {self.state.labels_library[index]}")
            if self._pending_start is not None:
                self._timeline.set_pending_start(
                    self._pending_start, colour_for_label(self._current_label(), self.state.labels_library))

    def _current_label(self) -> str:
        return self._label_combo.currentText().strip() or "Unlabelled"

    # ------------------------------------------------------------------
    # Undo / redo
    # ------------------------------------------------------------------

    def _snapshot_state(self):
        return copy.deepcopy(self.state.intervals), list(self.state.labels_library)

    def _push_undo(self) -> None:
        self._undo.append(self._snapshot_state())
        if len(self._undo) > _UNDO_LIMIT:
            self._undo.pop(0)
        self._redo.clear()
        self._update_undo_buttons()

    def _restore(self, snapshot) -> None:
        intervals, library = snapshot
        self.state.intervals = intervals
        self.state.labels_library = library
        self._select(None)
        self._refresh_label_lists()
        self._sync_timeline()

    def _undo_action(self) -> None:
        if not self._undo:
            return
        self._redo.append(self._snapshot_state())
        self._restore(self._undo.pop())
        self._update_undo_buttons()
        self._status.setText("Undone.")

    def _redo_action(self) -> None:
        if not self._redo:
            return
        self._undo.append(self._snapshot_state())
        self._restore(self._redo.pop())
        self._update_undo_buttons()
        self._status.setText("Redone.")

    def _update_undo_buttons(self) -> None:
        self._undo_btn.setEnabled(bool(self._undo))
        self._redo_btn.setEnabled(bool(self._redo))

    # ------------------------------------------------------------------
    # Interval changes
    # ------------------------------------------------------------------

    def _on_interval_created(self, start_sec, end_sec):
        label = self._current_label()
        start_sec, end_sec = round(min(start_sec, end_sec), 3), round(max(start_sec, end_sec), 3)
        if end_sec - start_sec < 0.5:
            return

        # Overlap validation
        if self._overlaps_existing(start_sec, end_sec):
            QMessageBox.warning(self, "Overlap",
                                "This interval overlaps with an existing one.")
            return

        self._push_undo()
        color = colour_for_label(label, self.state.labels_library)
        iv = LabelInterval(label=label, start_sec=start_sec, end_sec=end_sec, color=color)
        self.state.intervals.append(iv)
        self._sync_timeline()
        self._select(iv)
        self._status.setText(f"Created '{label}' ({_format_duration(end_sec - start_sec)}).")

    def _on_interval_deleted(self, idx):
        if 0 <= idx < len(self.state.intervals):
            self._push_undo()
            self.state.intervals.pop(idx)
            self._select(None)
            self._sync_timeline()

    def _delete_selected(self):
        if self._selected is not None and self._selected in self.state.intervals:
            self._on_interval_deleted(self.state.intervals.index(self._selected))

    def _on_interval_selected(self, idx):
        self._select(self.state.intervals[idx] if 0 <= idx < len(self.state.intervals) else None)

    def _on_interval_relabelled(self, idx, new_label):
        if 0 <= idx < len(self.state.intervals):
            self._push_undo()
            self.state.intervals[idx].label = new_label
            color = colour_for_label(new_label, self.state.labels_library)
            self.state.intervals[idx].color = color
            self._sync_timeline()
            self._select(self.state.intervals[idx])

    def _relabel_selected_with_current(self):
        if self._selected is not None and self._selected in self.state.intervals:
            self._on_interval_relabelled(self.state.intervals.index(self._selected), self._current_label())

    def _on_interval_resized(self, idx, new_start, new_end):
        """Handle edge-drag resize or move from the timeline widget."""
        if idx < 0 or idx >= len(self.state.intervals):
            return
        # Overlap check (excluding self)
        if new_end - new_start < 0.5 or self._overlaps_existing(new_start, new_end, exclude_idx=idx):
            # Revert: re-sync timeline with unchanged state
            self._sync_timeline()
            return
        # The timeline edits its own item copies, so state still holds the old values.
        self._push_undo()
        iv = self.state.intervals[idx]
        iv.start_sec = new_start
        iv.end_sec = new_end
        self._sync_timeline()
        self._select(iv)

    def _split_at_playhead(self):
        sec = self._timeline.playhead_sec
        for idx, iv in enumerate(self.state.intervals):
            if iv.start_sec + 0.5 <= sec <= iv.end_sec - 0.5:
                self._split_interval(idx, sec)
                return
        self._status.setText("Place the playhead inside an interval (at least 0.5 s from its edges) to split it.")

    def _split_interval(self, idx: int, sec: float):
        if idx < 0 or idx >= len(self.state.intervals):
            return
        iv = self.state.intervals[idx]
        sec = round(sec, 3)
        if not (iv.start_sec + 0.5 <= sec <= iv.end_sec - 0.5):
            QMessageBox.information(self, "Split", "Both parts must be at least 0.5 s long.")
            return
        self._push_undo()
        second = LabelInterval(label=iv.label, start_sec=sec, end_sec=iv.end_sec, color=iv.color)
        iv.end_sec = sec
        self.state.intervals.insert(idx + 1, second)
        self._sync_timeline()
        self._select(second)
        self._status.setText(f"Split '{iv.label}' at {format_hms_ms(sec)}.")

    def _jump_interval(self, direction: int):
        if not self.state.intervals:
            return
        now = self._timeline.playhead_sec
        ordered = sorted(self.state.intervals, key=lambda iv: iv.start_sec)
        if direction > 0:
            target = next((iv for iv in ordered if iv.start_sec > now + 1e-3), None)
        else:
            target = next((iv for iv in reversed(ordered) if iv.start_sec < now - 1e-3), None)
        if target is not None:
            self._select(target)
            self._on_playhead(target.start_sec)

    # ------------------------------------------------------------------
    # Start / End label at playhead
    # ------------------------------------------------------------------

    def _start_label_at_playhead(self):
        sec = self._timeline.playhead_sec
        self._pending_start = sec
        self._start_label_btn.setChecked(True)
        label = self._current_label()
        color = colour_for_label(label, self.state.labels_library)
        self._timeline.set_pending_start(sec, color)
        self._status.setText(f"Label start marked at {_format_duration(sec)}. "
                             "Play/seek to the end, then click 'End label' (O).")

    def _cancel_pending(self):
        if self._pending_start is None:
            return
        self._pending_start = None
        self._start_label_btn.setChecked(False)
        self._timeline.set_pending_start(None)
        self._status.setText("Pending label start cleared.")

    def _end_label_at_playhead(self):
        if self._pending_start is None:
            QMessageBox.information(self, "Info",
                                    "Click 'Start label' (I) first to mark the start position.")
            self._start_label_btn.setChecked(False)
            return
        end_sec = self._timeline.playhead_sec
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

        self._push_undo()
        label = self._current_label()
        color = colour_for_label(label, self.state.labels_library)
        iv = LabelInterval(label=label, start_sec=start_sec,
                           end_sec=end_sec, color=color)
        self.state.intervals.append(iv)
        self._sync_timeline()
        self._select(iv)

        # Clear pending state
        self._pending_start = None
        self._start_label_btn.setChecked(False)
        self._timeline.set_pending_start(None)
        self._status.setText(
            f"Created '{label}' ({_format_duration(end_sec - start_sec)})."
        )

    # ------------------------------------------------------------------
    # Selection, table and editor
    # ------------------------------------------------------------------

    def _select(self, interval: Optional[LabelInterval]) -> None:
        self._selected = interval if interval in self.state.intervals else None
        idx = self.state.intervals.index(self._selected) if self._selected is not None else -1
        self._timeline.selected_index = idx
        self._table.blockSignals(True)
        self._table.clearSelection()
        if idx >= 0:
            self._table.selectRow(idx)
            self._table.scrollToItem(self._table.item(idx, 0))
        self._table.blockSignals(False)
        if self._selected is None:
            self._editor_grp.setVisible(False)
        else:
            self._populate_editor(self._selected)
            self._editor_grp.setVisible(True)

    def _on_table_selection(self):
        rows = self._table.selectionModel().selectedRows()
        if rows:
            row = rows[0].row()
            if 0 <= row < len(self.state.intervals):
                self._select(self.state.intervals[row])

    def _on_table_double_click(self, row: int, _col: int):
        if 0 <= row < len(self.state.intervals):
            self._select(self.state.intervals[row])
            self._on_playhead(self.state.intervals[row].start_sec)

    def _refresh_table(self) -> None:
        self._table.blockSignals(True)
        self._table.setRowCount(len(self.state.intervals))
        for row, iv in enumerate(self.state.intervals):
            values = [str(row), iv.label, format_hms_ms(iv.start_sec), format_hms_ms(iv.end_sec),
                      format_hms_ms(iv.end_sec - iv.start_sec)]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 1:
                    item.setForeground(QBrush(QColor("#ffffff")))
                    item.setBackground(QBrush(QColor(iv.color)))
                self._table.setItem(row, col, item)
        self._table.blockSignals(False)

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
        if self._selected is None or self._selected not in self.state.intervals:
            return
        idx = self.state.intervals.index(self._selected)

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

        self._push_undo()
        iv = self.state.intervals[idx]
        iv.label = new_label or iv.label
        iv.start_sec = new_start
        iv.end_sec = new_end
        iv.color = colour_for_label(iv.label, self.state.labels_library)
        self._sync_timeline()
        self._select(iv)
        self._status.setText(f"Updated interval #{self.state.intervals.index(iv)}.")

    # ------------------------------------------------------------------
    # Manual interval creation
    # ------------------------------------------------------------------

    def _add_interval_manual(self):
        label = self._current_label()

        start = round(_qtime_to_secs(self._add_start.time()), 3)
        end = round(_qtime_to_secs(self._add_end.time()), 3)

        if start >= end:
            QMessageBox.warning(self, "Invalid", "Start must be before end.")
            return

        if self._overlaps_existing(start, end):
            QMessageBox.warning(self, "Overlap",
                                "This interval overlaps with an existing one.")
            return

        self._push_undo()
        color = colour_for_label(label, self.state.labels_library)
        iv = LabelInterval(label=label, start_sec=start, end_sec=end, color=color)
        self.state.intervals.append(iv)
        self._sync_timeline()
        self._select(iv)
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
        self._push_undo()
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
        self._select(None)
        self._sync_timeline()

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
        """Push current state.intervals to the timeline, table and signal plot."""
        selected = self._selected
        self.state.intervals.sort(key=lambda iv: iv.start_sec)
        for iv in self.state.intervals:
            iv.color = colour_for_label(iv.label, self.state.labels_library)
        items = [
            IntervalItem(i.label, i.start_sec, i.end_sec, i.color)
            for i in self.state.intervals
        ]
        self._timeline.set_intervals(items)
        self._timeline.set_label_choices(self.state.labels_library)
        self._plot.set_intervals(self.state.intervals)
        self._refresh_table()
        if selected is not None and selected in self.state.intervals:
            self._timeline.selected_index = self.state.intervals.index(selected)
        else:
            self._timeline.selected_index = -1

    def _on_playhead(self, sec):
        if not self.state.tracks:
            self._timeline.set_playhead(sec)
            self._plot.set_cursor(sec)
            return
        self._player.seek_seconds(sec)

    def _on_frame_changed(self, frame_no: int):
        """Kept for compatibility; positions arrive through ``position_changed``."""

    def _on_position_changed(self, sec: float):
        self._timeline.set_playhead(sec)
        self._plot.set_cursor(sec)

    def _figure_context(self) -> dict:
        ordered = [(i, iv.label, iv.start_sec, iv.end_sec) for i, iv in enumerate(self.state.intervals)]
        selected = None
        if self._selected is not None and self._selected in self.state.intervals:
            idx = self.state.intervals.index(self._selected)
            selected = (idx, self._selected.label, self._selected.start_sec, self._selected.end_sec)
        prefer = self._figure_prefer_selected
        self._figure_prefer_selected = False
        return {
            "output_directory": str(Path(self.state.output_directory) / "figures") if self.state.output_directory else "",
            "intervals": ordered,
            "selected_interval": selected,
            "prefer_selected": prefer,
        }

    def _on_figure_interval(self, idx: int):
        if 0 <= idx < len(self.state.intervals):
            self._select(self.state.intervals[idx])
            self._figure_prefer_selected = True
            self._plot.open_figure_export()

    def _export_current_frames(self):
        message = export_frames_interactive(self, self._player, self.state.output_directory, "labelling")
        if message:
            self._status.setText(message)

    def _export_snapshot(self):
        message = export_snapshot_interactive(self, self._player, self._plot, self.state.output_directory, "labelling")
        if message:
            self._status.setText(message)

    def _set_busy(self, busy: bool) -> None:
        self._export_btn.setEnabled(not busy)
        self._export_signals_btn.setEnabled(not busy)
        self._cancel_btn.setEnabled(busy)

    def _cancel_jobs(self):
        if self._export_worker is not None:
            self._export_worker.cancel()
            self._status.setText("Cancelling export…")
        if self._mosaic_worker is not None:
            self._mosaic_worker.cancel()

    # ------------------------------------------------------------------
    # Labelled segment export (runs in a background thread)
    # ------------------------------------------------------------------

    def _export(self):
        if self._export_worker is not None:
            return
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

        self._player.stop()
        worker = FFmpegWorker(
            self._export_job,
            intervals=intervals,
            segments_dir=segments_dir,
            ffmpeg=ffmpeg,
            signal_anchor=signal_anchor,
            coherence_warnings=coherence_warnings,
            remove_legacy=self._remove_legacy_csv_cb.isChecked(),
            blur_faces=self.state.blur_faces,
            out_dir=out_dir,
        )
        worker.progress.connect(self._on_export_progress)
        worker.finished.connect(self._on_export_finished)
        self._export_worker = worker
        self._export_result: dict = {}
        self._set_busy(True)
        self._progress.setValue(0)
        self._status.setText(f"Exporting {len(intervals)} segment(s)…")
        worker.start()

    def _on_export_progress(self, value: int, message: str):
        self._progress.setValue(value)
        self._status.setText(message)

    def _on_export_finished(self, ok: bool, message: str):
        self._export_worker = None
        self._set_busy(False)
        result = getattr(self, "_export_result", {})
        if ok and result:
            self.state.segments_manifest_path = result["manifest_path"]
            self._progress.setValue(100)
            self._has_exported = True
            status = f"Exported {result['count']} segments to {result['segments_dir']}"
            if result.get("cleaned"):
                action = "archived" if self.state.archive_removed_files else "removed"
                status += f"; legacy CSVs {action}: {result['cleaned']}"
            if result.get("failures"):
                status += f"\nWarning: {len(result['failures'])} camera clip(s) failed (see log)."
            self._status.setText(status)
            log.info("Exported %d segments to %s", result["count"], result["segments_dir"])
        else:
            self._status.setText(f"Export stopped: {message}")
            if "Cancelled" not in message:
                QMessageBox.critical(self, "Export labelled segments", f"Export failed:\n{message}")

    def _export_camera_clips(self, iv: LabelInterval, seg_dir: Path, ffmpeg: str, blur_faces: bool,
                             worker, report) -> list[str]:
        """Export every camera clip for one interval (cameras in parallel)."""
        jobs = []
        for ti, track in enumerate(self.state.tracks):
            src = track.final_output_path
            if not src or not Path(src).exists():
                continue
            dst = str(seg_dir / f"cam{ti+1}_{track.camera_label}.mp4")
            jobs.append((ti, src, dst))
        duration = iv.end_sec - iv.start_sec
        fractions = {ti: 0.0 for ti, _s, _d in jobs}
        lock = threading.Lock()

        def run(job):
            ti, src, dst = job

            def on_progress(done, total, ti=ti):
                with lock:
                    fractions[ti] = done / total if total else 1.0
                    report(sum(fractions.values()) / max(1, len(fractions)))

            try:
                if blur_faces:
                    export_anonymized_video_segment(
                        src,
                        dst,
                        start_sec=iv.start_sec,
                        duration_sec=duration,
                        ffmpeg=ffmpeg,
                        progress_callback=on_progress,
                        cancel_check=lambda: worker.is_cancelled,
                    )
                else:
                    trim_video(
                        src, dst,
                        start_sec=iv.start_sec,
                        duration_sec=duration,
                        ffmpeg=ffmpeg,
                    )
                    on_progress(1, 1)
            except Exception as exc:
                if worker.is_cancelled:
                    return None
                log.error("Trim failed for %s: %s", dst, exc)
                return dst
            return None

        if not jobs:
            return []
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            return [failure for failure in pool.map(run, jobs) if failure]

    def _export_job(self, *, intervals, segments_dir, ffmpeg, signal_anchor, coherence_warnings,
                    remove_legacy, blur_faces, out_dir, worker):
        manifest_rows = []
        total = len(intervals)
        cleaned_legacy_count = 0
        failures: list[str] = []

        for idx, iv in enumerate(intervals):
            if worker.is_cancelled:
                return
            seg_name = f"{idx:04d}_{iv.label.replace(' ', '_')}"

            def report(fraction, idx=idx, seg_name=seg_name):
                worker.progress.emit(int(((idx + fraction) / total) * 100),
                                     f"Segment {idx + 1}/{total}: {seg_name}")

            report(0.0)
            seg_dir = segments_dir / seg_name
            seg_dir.mkdir(parents=True, exist_ok=True)
            iv.folder = str(seg_dir)
            legacy_csv_paths = self._legacy_segment_csv_paths(seg_dir)

            failures.extend(self._export_camera_clips(iv, seg_dir, ffmpeg, blur_faces, worker, report))
            if worker.is_cancelled:
                return

            signal_meta = self._export_all_signal_slices(signal_anchor, iv, seg_dir)
            hr_files = signal_meta["hr_files"]
            ecg_files = signal_meta["ecg_files"]
            ppg_files = signal_meta["ppg_files"]
            if remove_legacy and hr_files:
                try:
                    cleaned_legacy_count += cleanup_obsolete_paths(
                        legacy_csv_paths,
                        out_dir,
                        self.state.archive_removed_files,
                    )
                except OSError as exc:
                    log.warning("Could not clean up legacy segment CSV files: %s", exc)

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
                "ppg_files": ppg_files,
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
            meta.update(signal_meta["synthetic_artifacts"])
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
        self._export_result = {
            "manifest_path": str(manifest_path),
            "count": len(manifest_rows),
            "segments_dir": str(segments_dir),
            "cleaned": cleaned_legacy_count,
            "failures": failures,
        }
        worker.progress.emit(100, "Writing manifest…")

    @staticmethod
    def _legacy_segment_csv_paths(seg_dir: Path) -> list[Path]:
        candidates = [seg_dir / "signal.csv"]
        meta_path = seg_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                legacy_name = meta.get("signal_file")
                if legacy_name:
                    candidate = (seg_dir / legacy_name).resolve()
                    candidate.relative_to(seg_dir.resolve())
                    candidates.append(candidate)
            except (json.JSONDecodeError, OSError, ValueError):
                pass
        return list(dict.fromkeys(candidates))

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

    def _export_synthetic_artifact_slices(
        self, anchor: datetime, interval: LabelInterval, segment_directory: Path
    ) -> dict[str, str]:
        sources = {
            field_name: Path(path)
            for field_name in _SYNTHETIC_ARTIFACT_FIELDS
            if (path := getattr(self.state, field_name))
            if path and Path(path).exists()
        }
        if not sources:
            return {}

        destination = segment_directory / "synthetic_ppg"
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        start = pd.Timestamp(anchor + timedelta(seconds=interval.start_sec))
        end = pd.Timestamp(anchor + timedelta(seconds=interval.end_sec))
        exported: dict[str, str] = {}
        for field_name, source in sources.items():
            frame = pd.read_csv(source)
            if "timestamp_utc" not in frame.columns:
                log.warning("Synthetic artifact has no timestamp_utc column: %s", source)
                continue
            timestamps = pd.to_datetime(
                frame["timestamp_utc"], format="mixed", utc=True, errors="coerce"
            )
            sliced = frame[(timestamps >= start) & (timestamps <= end)]
            if sliced.empty:
                continue
            relative_path = Path("synthetic_ppg") / source.name
            sliced.to_csv(segment_directory / relative_path, index=False)
            exported[field_name] = relative_path.as_posix()
        return exported

    def _export_all_signal_slices(
        self, anchor: datetime, interval: LabelInterval, segment_directory: Path
    ) -> dict[str, object]:
        hr_files = self._export_signal_slice(
            self._hr_merged_df, anchor, interval, segment_directory, "heart_rate_bpm"
        )
        ecg_files = self._export_signal_slice(
            self._ecg_merged_df, anchor, interval, segment_directory, "ecg_waveform"
        )
        synthetic_artifacts = self._export_synthetic_artifact_slices(
            anchor, interval, segment_directory
        )
        ppg_path = synthetic_artifacts.get("synthetic_ppg_path")
        return {
            "hr_files": hr_files,
            "ecg_files": ecg_files,
            "ppg_files": [ppg_path] if ppg_path else [],
            "synthetic_artifacts": synthetic_artifacts,
        }

    def _export_signals_to_existing_segments(self):
        self._load_synced_signal()
        intervals = [interval for interval in self.state.intervals if interval.folder]
        if not intervals:
            QMessageBox.information(
                self, "No existing segments",
                "Import a labelled-segments manifest or export labelled segments first.",
            )
            return
        anchor = self._signal_anchor()
        if anchor is None:
            QMessageBox.warning(self, "Signals unavailable", "Video time anchor is unavailable.")
            return
        count = 0
        cleaned_legacy_count = 0
        for interval in intervals:
            seg_dir = Path(interval.folder)
            if not seg_dir.exists():
                continue
            legacy_csv_paths = self._legacy_segment_csv_paths(seg_dir)
            signal_meta = self._export_all_signal_slices(anchor, interval, seg_dir)
            if not any((
                signal_meta["hr_files"],
                signal_meta["ecg_files"],
                signal_meta["ppg_files"],
                signal_meta["synthetic_artifacts"],
            )):
                continue
            meta_path = seg_dir / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            meta["hr_files"] = signal_meta["hr_files"]
            meta["ecg_files"] = signal_meta["ecg_files"]
            meta["ppg_files"] = signal_meta["ppg_files"]
            meta["synced_hr_paths"] = self._relative_signal_paths(
                self.state.synced_hr_paths, seg_dir
            )
            meta["synced_ecg_paths"] = self._relative_signal_paths(
                self.state.synced_ecg_paths, seg_dir
            )
            for field_name in _SYNTHETIC_ARTIFACT_FIELDS:
                meta.pop(field_name, None)
            meta.update(signal_meta["synthetic_artifacts"])
            if self._remove_legacy_csv_cb.isChecked() and signal_meta["hr_files"]:
                try:
                    cleaned_legacy_count += cleanup_obsolete_paths(
                        legacy_csv_paths,
                        self.state.output_directory,
                        self.state.archive_removed_files,
                    )
                    meta.pop("signal_file", None)
                    meta.pop("synced_signal_path", None)
                except OSError as exc:
                    log.warning("Could not clean up legacy segment CSV files: %s", exc)
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            count += 1
        status = f"Exported all available signals to {count} existing segment(s)."
        if cleaned_legacy_count:
            action = "archived" if self.state.archive_removed_files else "removed"
            status += f" Legacy CSVs {action}: {cleaned_legacy_count}."
        self._status.setText(status)

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
        labels = [t.camera_label for t in self.state.tracks]
        options = ask_mosaic_options(self, default_name, self.state.mosaic_preset, labels,
                                     self._player.audio_camera)
        if options is None:
            return
        path, preset_key, preset_choice, audio_index = options
        self.state.mosaic_preset = preset_key

        t0 = self.state.tracks[0]
        fps = t0.fps or 30.0
        duration = iv.end_sec - iv.start_sec
        total_frames = int(duration * fps)
        audio_path = ""
        if 0 <= audio_index < len(self.state.tracks):
            audio_path = self.state.tracks[audio_index].final_output_path

        self._mosaic_worker = MosaicWorker(
            video_paths=valid,
            camera_labels=labels,
            fps=fps,
            total_frames=total_frames,
            video_duration_sec=duration,
            output_path=path,
            signal_df=self._hr_merged_df if self._hr_merged_df is not None else self._ecg_merged_df,
            start_sec=iv.start_sec,
            end_sec=iv.end_sec,
            quality_preset=preset_key,
            ffmpeg_path=self.state.ffmpeg_path,
            blur_faces=self.state.blur_faces,
            signal_time_zero=effective_signal_anchor(self.state),
            audio_path=audio_path,
        )
        self._mosaic_worker.progress.connect(self._on_mosaic_progress)
        self._mosaic_worker.finished.connect(self._on_mosaic_finished)
        self._set_busy(True)
        self._progress.setValue(0)
        self._status.setText(f"Exporting mosaic for '{iv.label}' ({preset_choice})…")
        self._mosaic_worker.start()

    def _on_mosaic_progress(self, current: int, total: int):
        if total > 0:
            self._progress.setValue(int(current * 100 / total))
        self._status.setText(f"Mosaic: frame {current} / {total}")

    def _on_mosaic_finished(self, success: bool, msg: str):
        self._set_busy(False)
        if success:
            self._progress.setValue(100)
            self._status.setText(f"Mosaic saved: {Path(msg).name}")
            QMessageBox.information(self, "Mosaic export", f"Mosaic video saved:\n{msg}")
        else:
            self._status.setText(f"Mosaic failed: {msg}")
            if "Cancelled" not in msg:
                QMessageBox.critical(self, "Error", f"Mosaic export failed:\n{msg}")
        self._mosaic_worker = None

