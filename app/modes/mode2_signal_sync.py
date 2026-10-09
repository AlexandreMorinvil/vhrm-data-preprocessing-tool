from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from PyQt6.QtCore import QSettings, Qt, QTime
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStyle,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from ..file_cleanup import cleanup_obsolete_paths
from ..mosaic_export import MosaicWorker
from ..signals import (
    get_loaders,
    load_signal_files,
    read_synced_signal_csvs,
    signal_file_type_name,
    signal_long_to_wide,
    write_synced_signal_csvs,
)
from ..state import (
    ProjectState,
    compute_signal_anchor,
    effective_signal_anchor,
    format_time_coherence_warnings,
    generate_sidecar,
    load_sidecar,
    video_timeline_duration_sec,
)
from ..widgets.layout import ModeWorkspace
from ..widgets.frame_preview import MultiCameraPlayer
from ..widgets.signal_plot import SignalPlot
from .common import ask_mosaic_options, export_frames_interactive, export_snapshot_interactive

log = logging.getLogger(__name__)

_SIGNAL_FILTER = "CSV / signal files (*.csv *.tsv *.txt);;All files (*)"
_META_FILTER = "Metadata sidecar (*.json);;All files (*)"


def _normalise_dt(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fmt_dt(dt: Optional[datetime]) -> str:
    if dt is None:
        return "unknown"
    return _normalise_dt(dt).isoformat(timespec="milliseconds")


def _fmt_time(dt: Optional[datetime]) -> str:
    if dt is None:
        return "unknown"
    return _normalise_dt(dt).strftime("%H:%M:%S.%f")[:-3]


def _qtime_from_dt(dt: Optional[datetime]) -> QTime:
    if dt is None:
        return QTime.currentTime()
    dt = _normalise_dt(dt)
    return QTime(dt.hour, dt.minute, dt.second, dt.microsecond // 1000)


def _dt_from_qtime(base: Optional[datetime], qtime: QTime) -> Optional[datetime]:
    if base is None:
        return None
    base = _normalise_dt(base)
    return base.replace(
        hour=qtime.hour(),
        minute=qtime.minute(),
        second=qtime.second(),
        microsecond=qtime.msec() * 1000,
    )


def _secs_to_qtime(sec: float) -> QTime:
    ms = int(round(max(0.0, sec) * 1000))
    h = min(ms // 3_600_000, 23)
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return QTime(h, m, s, ms)


def _qtime_to_secs(t: QTime) -> float:
    return t.hour() * 3600 + t.minute() * 60 + t.second() + t.msec() / 1000.0


class Mode2Widget(QWidget):
    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state
        self._hr_merged_df: Optional[pd.DataFrame] = None
        self._ecg_merged_df: Optional[pd.DataFrame] = None
        self._time_rows: list[dict] = []
        self._settings = QSettings("VideoResearchTool", "VRT")

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self._workspace = ModeWorkspace("signal_sync", self, left_width=640)
        root.addWidget(self._workspace)
        ws = self._workspace

        load_box = QWidget()
        load_lay = QVBoxLayout(load_box)
        load_lay.setContentsMargins(0, 0, 0, 0)
        meta_btn = QPushButton("Load from metadata file \u2026")
        meta_btn.clicked.connect(self._load_from_meta)
        load_lay.addWidget(meta_btn)
        self._load_status = QLabel("")
        self._load_status.setWordWrap(True)
        load_lay.addWidget(self._load_status)
        ws.add_section("Load project metadata", load_box)

        time_box = QWidget()
        time_lay = QVBoxLayout(time_box)
        time_lay.setContentsMargins(0, 0, 0, 0)
        time_hint = QLabel(
            "Tip: with \u201cReference point in video\u201d, find a visible clock or event in the "
            "videos, then press \u201cUse playhead\u201d."
        )
        time_hint.setWordWrap(True)
        time_hint.setStyleSheet("color: #8a8a8a;")
        time_lay.addWidget(time_hint)
        self._time_grid = QGridLayout()
        time_lay.addLayout(self._time_grid)
        self._check_time_btn = QPushButton("Check coherence")
        self._check_time_btn.clicked.connect(self._check_time_coherence)
        time_lay.addWidget(self._check_time_btn)
        self._time_status = QLabel("")
        self._time_status.setWordWrap(True)
        time_lay.addWidget(self._time_status)
        ws.add_section("Camera time correction", time_box)

        sources_box = QWidget()
        sources_lay = QVBoxLayout(sources_box)
        sources_lay.setContentsMargins(0, 0, 0, 0)
        loader_names = [getattr(l, "display_name", type(l).__name__) for l in get_loaders()]
        info = QLabel(f"Available loaders: {', '.join(loader_names) or 'none'}")
        info.setWordWrap(True)
        info.setStyleSheet("color: #8a8a8a;")
        sources_lay.addWidget(info)

        hr_grp = QGroupBox("Heart rate")
        hr_lay = QVBoxLayout(hr_grp)
        self._hr_file_list = QListWidget()
        self._hr_file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._hr_file_list.setMaximumHeight(110)
        hr_lay.addWidget(self._hr_file_list)
        hr_buttons = QHBoxLayout()
        hr_btn = QPushButton("Add HR file \u2026")
        hr_btn.clicked.connect(lambda: self._add_signal(self._hr_file_list, "HR"))
        hr_buttons.addWidget(hr_btn)
        self._remove_hr_btn = self._make_remove_signal_button(self._hr_file_list)
        hr_buttons.addWidget(self._remove_hr_btn)
        hr_lay.addLayout(hr_buttons)
        sources_lay.addWidget(hr_grp)

        ecg_grp = QGroupBox("ECG")
        ecg_lay = QVBoxLayout(ecg_grp)
        self._ecg_file_list = QListWidget()
        self._ecg_file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._ecg_file_list.setMaximumHeight(110)
        ecg_lay.addWidget(self._ecg_file_list)
        ecg_buttons = QHBoxLayout()
        ecg_btn = QPushButton("Add ECG file \u2026")
        ecg_btn.clicked.connect(lambda: self._add_signal(self._ecg_file_list, "ECG"))
        ecg_buttons.addWidget(ecg_btn)
        self._remove_ecg_btn = self._make_remove_signal_button(self._ecg_file_list)
        ecg_buttons.addWidget(self._remove_ecg_btn)
        ecg_lay.addLayout(ecg_buttons)
        sources_lay.addWidget(ecg_grp)

        self._load_btn = QPushButton("Load && synchronise HR / ECG")
        self._load_btn.setStyleSheet("font-weight:bold; padding:6px;")
        self._load_btn.clicked.connect(self._load_and_sync)
        sources_lay.addWidget(self._load_btn)

        self._remove_legacy_csv_cb = QCheckBox(
            "Remove legacy combined synchronized CSV files after synchronization"
        )
        self._remove_legacy_csv_cb.setToolTip(
            "Removes old signal_synced.csv, hr_synced.csv, and ecg_synced.csv files "
            "only after replacement per-sensor files are written successfully."
        )
        sources_lay.addWidget(self._remove_legacy_csv_cb)

        self._skip_btn = QPushButton("Skip (no signal)")
        self._skip_btn.clicked.connect(self._skip)
        sources_lay.addWidget(self._skip_btn)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        sources_lay.addWidget(self._status)
        ws.add_section("Source signal files", sources_box)

        synced_grp = QWidget()
        synced_lay = QGridLayout(synced_grp)
        synced_lay.setContentsMargins(0, 0, 0, 0)
        synced_lay.addWidget(QLabel("Heart rate"), 0, 0)
        synced_lay.addWidget(QLabel("ECG"), 0, 1)
        self._synced_hr_file_list = QListWidget()
        self._synced_ecg_file_list = QListWidget()
        self._synced_hr_file_list.setMinimumHeight(60)
        self._synced_ecg_file_list.setMinimumHeight(60)
        self._synced_hr_file_list.setMaximumHeight(100)
        self._synced_ecg_file_list.setMaximumHeight(100)
        synced_lay.addWidget(self._synced_hr_file_list, 1, 0)
        synced_lay.addWidget(self._synced_ecg_file_list, 1, 1)
        ws.add_section("Generated synchronized files", synced_grp)

        mosaic_box = QWidget()
        mosaic_lay = QVBoxLayout(mosaic_box)
        mosaic_lay.setContentsMargins(0, 0, 0, 0)
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
        self._mosaic_status.setWordWrap(True)
        mosaic_lay.addWidget(self._mosaic_status)
        ws.add_section("Mosaic video export", mosaic_box, expanded=False)
        self._mosaic_worker: MosaicWorker | None = None
        ws.finish_left()

        self._player = MultiCameraPlayer(face_blur_enabled=state.blur_faces)
        ws.add_work(self._player, 520)

        signal_area = QWidget()
        sig_lay = QVBoxLayout(signal_area)
        sig_lay.setContentsMargins(0, 0, 0, 0)
        toggles = QHBoxLayout()
        toggles.addWidget(QLabel("Show:"))
        self._show_hr = QCheckBox("Heart rate")
        self._show_ecg = QCheckBox("ECG")
        for key, box in (("hr", self._show_hr), ("ecg", self._show_ecg)):
            box.setChecked(self._settings.value(f"signal_sync/show_{key}", True, type=bool))
            box.toggled.connect(lambda checked, k=key: (
                self._settings.setValue(f"signal_sync/show_{k}", checked), self._show_selected_plot()))
            toggles.addWidget(box)
        toggles.addStretch()
        sig_lay.addLayout(toggles)
        self._plot = SignalPlot()
        self._plot.figure_export_context = lambda: {
            "output_directory": str(Path(self.state.output_directory) / "figures") if self.state.output_directory else "",
        }
        sig_lay.addWidget(self._plot, 1)
        ws.add_work(signal_area, 360)

        self._player.frame_changed.connect(self._on_frame_changed)
        self._player.position_changed.connect(self._plot.set_cursor)
        self._plot.seek_requested.connect(self._player.seek_seconds)
        self._player.export_frames_requested.connect(self._export_current_frames)
        self._player.snapshot_requested.connect(self._export_snapshot)
        self._player.install_shortcuts(self)

        self.refresh_from_state()

    def refresh_from_state(self):
        self._refresh_signal_file_lists()
        self._refresh_player()
        self._load_synced_signal()
        self._refresh_time_correction_ui()

    def _refresh_signal_file_lists(self):
        self._hr_file_list.clear()
        self._ecg_file_list.clear()
        self._synced_hr_file_list.clear()
        self._synced_ecg_file_list.clear()
        for p in self.state.hr_signal_paths or self.state.signal_paths:
            self._hr_file_list.addItem(self._make_signal_item(p, "HR"))
        for path in self.state.synced_hr_paths:
            self._synced_hr_file_list.addItem(self._make_synced_item(path, "HR"))
        for p in self.state.ecg_signal_paths:
            self._ecg_file_list.addItem(self._make_signal_item(p, "ECG"))
        for path in self.state.synced_ecg_paths:
            self._synced_ecg_file_list.addItem(self._make_synced_item(path, "ECG"))

    def _make_signal_item(self, path: str, signal_kind: str = "signal") -> QListWidgetItem:
        sensor_type = signal_file_type_name(path) if Path(path).exists() else f"{signal_kind} source (file unavailable)"
        item = QListWidgetItem(f"{Path(path).name} — {sensor_type}")
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(f"{path}\nDetected type: {sensor_type}")
        return item

    @staticmethod
    def _make_synced_item(path: str, signal_kind: str) -> QListWidgetItem:
        exists = Path(path).exists()
        availability = "Ready" if exists else "File unavailable"
        item = QListWidgetItem(f"{Path(path).name} — {availability}")
        item.setToolTip(f"Generated synchronized {signal_kind} file\n{path}")
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        return item

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
        self._refresh_time_correction_ui()
        self._refresh_signal_file_lists()
        n = len(self.state.tracks)
        self._load_status.setText(f"Loaded {n} camera(s) from sidecar.")

    def _refresh_player(self):
        """Reload the player/plot from the current state.tracks."""
        labels = [t.camera_label for t in self.state.tracks]
        self._player.set_cameras(labels)
        paths = [t.final_output_path for t in self.state.tracks]
        valid = [p for p in paths if p and Path(p).exists()]
        if valid:
            self._player.load_videos(valid)
        self._player.set_clock_anchor(effective_signal_anchor(self.state) if self.state.tracks else None)

    def _load_synced_signal(self):
        """Load synced signal CSV from state if available."""
        self._hr_merged_df = None
        self._ecg_merged_df = None
        hr_paths = [path for path in self.state.synced_hr_paths if Path(path).exists()]
        if hr_paths:
            try:
                self._hr_merged_df = read_synced_signal_csvs(hr_paths)
            except Exception as exc:
                log.warning("Could not load synchronized HR CSVs %s: %s", hr_paths, exc)
        ecg_paths = [path for path in self.state.synced_ecg_paths if Path(path).exists()]
        if ecg_paths:
            try:
                self._ecg_merged_df = read_synced_signal_csvs(ecg_paths)
            except Exception as exc:
                log.warning("Could not load synchronized ECG CSVs %s: %s", ecg_paths, exc)
        self._show_selected_plot()
        if self._hr_merged_df is not None or self._ecg_merged_df is not None:
            self._status.setText("Synchronized HR/ECG data loaded")

    def _add_signal(self, target: QListWidget, signal_kind: str):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select signal files", "", _SIGNAL_FILTER,
        )
        added = False
        for p in files:
            loader = signal_file_type_name(p)
            if signal_kind == "ECG" and loader != "ECG waveform":
                QMessageBox.warning(self, "Not an ECG file", f"{Path(p).name} is detected as {loader}.")
                continue
            if signal_kind == "HR" and loader == "ECG waveform":
                QMessageBox.warning(self, "Not an HR file", f"{Path(p).name} is an ECG waveform file.")
                continue
            existing = self._paths_from_list(target)
            if p not in existing:
                target.addItem(self._make_signal_item(p))
                added = True
        if added:
            self._update_source_paths()

    def _make_remove_signal_button(self, target: QListWidget) -> QPushButton:
        button = QPushButton("Remove selected")
        button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon))
        button.setToolTip("Remove selected source signals from the project without deleting the original files.")
        button.setEnabled(False)
        target.itemSelectionChanged.connect(lambda: button.setEnabled(bool(target.selectedItems())))
        button.clicked.connect(lambda: self._remove_signals(target))
        return button

    def _remove_signals(self, target: QListWidget):
        selected = target.selectedItems()
        if not selected:
            return
        for item in selected:
            target.takeItem(target.row(item))
        self._update_source_paths()
        self._status.setText("Removed source signals from the project. Original files unchanged.")

    def _update_source_paths(self):
        self.state.hr_signal_paths = self._paths_from_list(self._hr_file_list)
        self.state.signal_paths = list(self.state.hr_signal_paths)
        self.state.ecg_signal_paths = self._paths_from_list(self._ecg_file_list)
        self.state.mode2_complete = False

    # ------------------------------------------------------------------
    # Camera time correction
    # ------------------------------------------------------------------

    def _clear_time_grid(self):
        while self._time_grid.count():
            item = self._time_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _refresh_time_correction_ui(self):
        self._clear_time_grid()
        self._time_rows.clear()
        headers = ["Camera", "Mode", "Input", "Corrected start", "Not limiting size"]
        for col, text in enumerate(headers):
            label = QLabel(text)
            label.setStyleSheet("font-weight:bold;")
            self._time_grid.addWidget(label, 0, col)

        if not self.state.tracks:
            self._time_grid.addWidget(QLabel("Load project metadata to edit time corrections."), 1, 0, 1, 5)
            self._time_status.setText("")
            return

        for row_idx, track in enumerate(self.state.tracks, start=1):
            camera_label = QLabel(track.camera_label or f"Camera {row_idx}")
            camera_label.setToolTip(f"Filename time: {_fmt_time(track.parsed_start_datetime())}")

            mode_combo = QComboBox()
            mode_combo.addItem("No correction", "none")
            mode_combo.addItem("Reference start time", "reference_time")
            mode_combo.addItem("Reference point in video", "reference_video_time")
            mode_combo.addItem("Offset (seconds)", "offset")
            mode_idx = mode_combo.findData(track.time_correction_mode)
            mode_combo.setCurrentIndex(max(0, mode_idx))

            ref_edit = QTimeEdit()
            ref_edit.setDisplayFormat("HH:mm:ss.zzz")
            ref_dt = track.parsed_true_start_datetime() or track.corrected_start_datetime() or track.parsed_start_datetime()
            ref_edit.setTime(_qtime_from_dt(ref_dt))

            offset_spin = QDoubleSpinBox()
            offset_spin.setRange(-86400.0, 86400.0)
            offset_spin.setDecimals(3)
            offset_spin.setSingleStep(0.1)
            offset_spin.setValue(track.time_correction_offset_sec)

            point_container = QWidget()
            point_layout = QHBoxLayout(point_container)
            point_layout.setContentsMargins(0, 0, 0, 0)
            point_layout.addWidget(QLabel("Video:"))
            point_video = QTimeEdit()
            point_video.setDisplayFormat("HH:mm:ss.zzz")
            point_video.setTime(_secs_to_qtime(track.reference_video_time_sec))
            point_layout.addWidget(point_video)
            use_playhead_btn = QPushButton("Use playhead")
            use_playhead_btn.setToolTip("Use the current synced-video playhead time")
            point_layout.addWidget(use_playhead_btn)
            point_layout.addWidget(QLabel("Ref:"))
            point_ref = QTimeEdit()
            point_ref.setDisplayFormat("HH:mm:ss.zzz")
            point_ref_dt = track.parsed_video_reference_datetime() or track.corrected_start_datetime() or track.parsed_start_datetime()
            point_ref.setTime(_qtime_from_dt(point_ref_dt))
            point_layout.addWidget(point_ref)

            none_label = QLabel(f"Filename time {_fmt_time(track.parsed_start_datetime())}")
            input_container = QWidget()
            input_layout = QHBoxLayout(input_container)
            input_layout.setContentsMargins(0, 0, 0, 0)
            input_layout.addWidget(none_label)
            input_layout.addWidget(ref_edit)
            input_layout.addWidget(offset_spin)
            input_layout.addWidget(point_container)

            corrected_label = QLabel("")
            corrected_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

            not_limiting_cb = QCheckBox()
            not_limiting_cb.setChecked(not track.limits_common_duration)
            not_limiting_cb.setToolTip(
                "Exclude this camera when choosing the shared video duration for signal clipping, plotting, and mosaic export."
            )

            row = {
                "track_index": row_idx - 1,
                "mode": mode_combo,
                "none": none_label,
                "ref": ref_edit,
                "offset": offset_spin,
                "point_container": point_container,
                "point_video": point_video,
                "point_ref": point_ref,
                "point_use_playhead": use_playhead_btn,
                "corrected": corrected_label,
                "not_limiting": not_limiting_cb,
            }
            self._time_rows.append(row)

            mode_combo.currentIndexChanged.connect(lambda _=0, i=row_idx - 1: self._on_time_row_changed(i))
            ref_edit.timeChanged.connect(lambda _=None, i=row_idx - 1: self._on_time_row_changed(i))
            offset_spin.valueChanged.connect(lambda _=0.0, i=row_idx - 1: self._on_time_row_changed(i))
            point_video.timeChanged.connect(lambda _=None, i=row_idx - 1: self._on_time_row_changed(i))
            point_ref.timeChanged.connect(lambda _=None, i=row_idx - 1: self._on_time_row_changed(i))
            use_playhead_btn.clicked.connect(lambda _=False, i=row_idx - 1: self._use_playhead_for_time_row(i))
            not_limiting_cb.toggled.connect(lambda _=False, i=row_idx - 1: self._on_time_row_changed(i))

            self._time_grid.addWidget(camera_label, row_idx, 0)
            self._time_grid.addWidget(mode_combo, row_idx, 1)
            self._time_grid.addWidget(input_container, row_idx, 2)
            self._time_grid.addWidget(corrected_label, row_idx, 3)
            self._time_grid.addWidget(not_limiting_cb, row_idx, 4)
            self._update_time_row(row_idx - 1, write_state=False)

    def _on_time_row_changed(self, idx: int):
        self._update_time_row(idx, write_state=True)

    def _update_time_row(self, idx: int, write_state: bool):
        if idx < 0 or idx >= len(self._time_rows) or idx >= len(self.state.tracks):
            return
        row = self._time_rows[idx]
        track = self.state.tracks[idx]
        mode = row["mode"].currentData() or "none"

        row["none"].setVisible(mode == "none")
        row["ref"].setVisible(mode == "reference_time")
        row["offset"].setVisible(mode == "offset")
        row["point_container"].setVisible(mode == "reference_video_time")

        if write_state:
            base = track.parsed_start_datetime()
            track.time_correction_mode = mode
            track.limits_common_duration = not row["not_limiting"].isChecked()
            if mode == "none":
                track.true_start_datetime = None
                track.reference_video_time_sec = 0.0
                track.video_reference_datetime = None
                track.time_correction_offset_sec = 0.0
            elif mode == "reference_time":
                true_start = _dt_from_qtime(base, row["ref"].time())
                track.true_start_datetime = true_start.isoformat(timespec="milliseconds") if true_start else None
                track.reference_video_time_sec = 0.0
                track.video_reference_datetime = None
                track.time_correction_offset_sec = (
                    (true_start - base).total_seconds() if base is not None else 0.0
                )
            elif mode == "reference_video_time":
                reference_dt = _dt_from_qtime(base, row["point_ref"].time())
                video_time_sec = _qtime_to_secs(row["point_video"].time())
                track.true_start_datetime = None
                track.reference_video_time_sec = video_time_sec
                track.video_reference_datetime = reference_dt.isoformat(timespec="milliseconds") if reference_dt else None
                if base is not None and reference_dt is not None:
                    corrected_start = reference_dt - timedelta(seconds=video_time_sec)
                    track.time_correction_offset_sec = (corrected_start - base).total_seconds()
                else:
                    track.time_correction_offset_sec = 0.0
            else:
                track.true_start_datetime = None
                track.reference_video_time_sec = 0.0
                track.video_reference_datetime = None
                track.time_correction_offset_sec = row["offset"].value()

        if mode != "offset":
            row["offset"].blockSignals(True)
            row["offset"].setValue(track.time_correction_offset_sec)
            row["offset"].blockSignals(False)

        row["corrected"].setText(
            f"{track.time_correction_offset_sec:+.3f} -> {_fmt_time(track.corrected_start_datetime())}"
        )

    def _use_playhead_for_time_row(self, idx: int):
        if idx < 0 or idx >= len(self._time_rows):
            return
        fps = self._player.get_fps() or 30.0
        sec = self._player.current_frame / fps if fps > 0 else 0.0
        row = self._time_rows[idx]
        row["point_video"].setTime(_secs_to_qtime(sec))
        self._update_time_row(idx, write_state=True)

    def _apply_time_correction_ui(self):
        for idx in range(len(self._time_rows)):
            self._update_time_row(idx, write_state=True)

    def _time_anchor_and_warnings(self):
        self._apply_time_correction_ui()
        anchor, warnings = compute_signal_anchor(
            self.state.tracks,
            self.state.time_coherence_tolerance_sec,
        )
        return anchor, warnings

    def _check_time_coherence(self):
        anchor, warnings = self._time_anchor_and_warnings()
        anchor_text = _fmt_dt(anchor)
        if warnings:
            msg = format_time_coherence_warnings(warnings)
            self._time_status.setText(f"Warnings. Anchor: {anchor_text}")
            QMessageBox.warning(self, "Camera time correction", f"{msg}\n\nAnchor that would be used:\n{anchor_text}")
        else:
            self._time_status.setText(f"Coherent. Anchor: {anchor_text}")
            QMessageBox.information(self, "Camera time correction", f"No coherence warnings.\n\nAnchor:\n{anchor_text}")

    def _confirm_time_warnings(self, warnings: list[str], anchor: Optional[datetime]) -> bool:
        if not warnings:
            return True
        msg = format_time_coherence_warnings(warnings)
        reply = QMessageBox.question(
            self,
            "Camera time correction warning",
            f"{msg}\n\nSignal anchor to use:\n{_fmt_dt(anchor)}\n\nContinue signal synchronisation?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _load_and_sync(self):
        hr_paths = self._paths_from_list(self._hr_file_list)
        ecg_paths = self._paths_from_list(self._ecg_file_list)
        if not hr_paths and not ecg_paths:
            QMessageBox.information(self, "Info", "No HR or ECG files selected.")
            return

        self.state.hr_signal_paths = hr_paths
        self.state.signal_paths = list(hr_paths)
        self.state.ecg_signal_paths = ecg_paths
        anchor = None
        duration = 0.0
        if self.state.tracks:
            anchor, coherence_warnings = self._time_anchor_and_warnings()
            if not self._confirm_time_warnings(coherence_warnings, anchor):
                self._status.setText("Signal synchronisation cancelled after time warning.")
                return
            self.state.last_signal_anchor_datetime = anchor.isoformat() if anchor else None
            self.state.last_time_coherence_warnings = coherence_warnings
            duration = video_timeline_duration_sec(self.state.tracks)

        loaded_hr = self._load_clip_and_pivot(hr_paths, anchor, duration, False)
        loaded_ecg = self._load_clip_and_pivot(ecg_paths, anchor, duration, False)
        if loaded_hr is None and loaded_ecg is None:
            QMessageBox.warning(self, "Warning", "No signal data within the video time range.")
            return
        self._hr_merged_df = loaded_hr
        self._ecg_merged_df = loaded_ecg
        if loaded_hr is None:
            self.state.synced_hr_paths = []
            self.state.synced_hr_path = ""
            self.state.synced_signal_path = ""
        if loaded_ecg is None:
            self.state.synced_ecg_paths = []
            self.state.synced_ecg_path = ""

        out_dir = self.state.output_directory
        legacy_paths = set(self.state.legacy_synced_paths)
        legacy_hr_paths = legacy_paths.intersection(self.state.synced_hr_paths)
        legacy_ecg_paths = legacy_paths.intersection(self.state.synced_ecg_paths)
        cleaned_legacy_count = 0
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            if self._hr_merged_df is not None:
                self.state.synced_hr_paths = write_synced_signal_csvs(
                    self._hr_merged_df, out_dir, "heart_rate_bpm"
                )
                self.state.synced_hr_path = ""
                self.state.synced_signal_path = ""
            if self._ecg_merged_df is not None:
                self.state.synced_ecg_paths = write_synced_signal_csvs(
                    self._ecg_merged_df, out_dir, "ecg_waveform"
                )
                self.state.synced_ecg_path = ""
            if self._remove_legacy_csv_cb.isChecked():
                obsolete_paths: list[str | Path] = []
                if self._hr_merged_df is not None:
                    obsolete_paths.extend(legacy_hr_paths)
                    obsolete_paths.extend([
                        Path(out_dir) / "signal_synced.csv",
                        Path(out_dir) / "hr_synced.csv",
                    ])
                if self._ecg_merged_df is not None:
                    obsolete_paths.extend(legacy_ecg_paths)
                    obsolete_paths.append(Path(out_dir) / "ecg_synced.csv")
                generated_paths = {
                    str(Path(path).resolve())
                    for path in self.state.synced_hr_paths + self.state.synced_ecg_paths
                }
                obsolete_paths = [
                    path for path in obsolete_paths
                    if str(Path(path).resolve()) not in generated_paths
                ]
                try:
                    cleaned_legacy_count = cleanup_obsolete_paths(
                        obsolete_paths,
                        out_dir,
                        self.state.archive_removed_files,
                    )
                    self.state.legacy_synced_paths = [
                        path for path in self.state.legacy_synced_paths
                        if Path(path).exists()
                    ]
                except OSError as exc:
                    log.warning("Could not clean up legacy synchronized CSV files: %s", exc)
            try:
                generate_sidecar(self.state)
            except Exception as exc:
                log.warning("Could not update sidecar: %s", exc)

        self._refresh_signal_file_lists()
        self._show_selected_plot()
        status_parts = []
        if self._hr_merged_df is not None:
            status_parts.append(f"HR: {len(self._hr_merged_df)} samples")
        if self._ecg_merged_df is not None:
            status_parts.append(f"ECG: {len(self._ecg_merged_df)} samples")
        if cleaned_legacy_count:
            action = "archived" if self.state.archive_removed_files else "removed"
            status_parts.append(f"legacy CSVs {action}: {cleaned_legacy_count}")
        self._status.setText("; ".join(status_parts) + ".")
        self.state.mode2_complete = True

    @staticmethod
    def _paths_from_list(file_list: QListWidget) -> list[str]:
        return [
            path for i in range(file_list.count())
            if (path := file_list.item(i).data(Qt.ItemDataRole.UserRole))
        ]

    def _load_clip_and_pivot(self, paths, anchor, duration, include_average):
        if not paths:
            return None
        dfs, _types, failed_paths = load_signal_files(paths)
        for path in failed_paths:
            log.warning("Failed to load signal: %s", path)
        if not dfs:
            return None
        merged = pd.concat(dfs, ignore_index=True)
        if anchor is not None:
            end = pd.Timestamp(anchor + timedelta(seconds=duration))
            merged = merged[(merged["timestamp_utc"] >= pd.Timestamp(anchor)) & (merged["timestamp_utc"] <= end)]
        if merged.empty:
            return None
        return signal_long_to_wide(merged, include_average=include_average)

    def _show_selected_plot(self):
        panels = []
        if self._show_hr.isChecked() and self._hr_merged_df is not None:
            panels.append(("Heart rate", self._hr_merged_df, "BPM"))
        if self._show_ecg.isChecked() and self._ecg_merged_df is not None:
            panels.append(("ECG", self._ecg_merged_df, "ECG"))
        panels = [p for p in panels if "timestamp_utc" in p[1].columns and not p[1].empty]
        if not panels:
            self._plot.clear()
            return
        self._plot.set_time_zero(effective_signal_anchor(self.state) if self.state.tracks else None)
        self._plot.set_panels(panels, video_duration_sec=video_timeline_duration_sec(self.state.tracks))

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
        labels = [t.camera_label for t in self.state.tracks]
        options = ask_mosaic_options(self, default_name, self.state.mosaic_preset, labels,
                                     self._player.audio_camera)
        if options is None:
            return
        path, preset_key, preset_choice, audio_index = options
        self.state.mosaic_preset = preset_key

        t0 = self.state.tracks[0]
        fps = t0.fps or 30.0
        duration = video_timeline_duration_sec(self.state.tracks)
        total_frames = int(duration * fps) if duration > 0 else (t0.frame_count or 0)
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
            signal_df=self._hr_merged_df,
            quality_preset=preset_key,
            ffmpeg_path=self.state.ffmpeg_path,
            blur_faces=self.state.blur_faces,
            signal_time_zero=effective_signal_anchor(self.state),
            audio_path=audio_path,
        )
        self._mosaic_worker.progress.connect(self._on_mosaic_progress)
        self._mosaic_worker.finished.connect(self._on_mosaic_finished)
        self._mosaic_btn.setEnabled(False)
        self._mosaic_cancel_btn.setEnabled(True)
        self._mosaic_progress.setValue(0)
        self._mosaic_status.setText(f"Exporting ({preset_choice})\u2026")
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
        """Kept for compatibility; the cursor follows ``position_changed``."""

    def _export_current_frames(self):
        message = export_frames_interactive(self, self._player, self.state.output_directory, "signal_sync")
        if message:
            self._mosaic_status.setText(message)

    def _export_snapshot(self):
        message = export_snapshot_interactive(self, self._player, self._plot, self.state.output_directory, "signal_sync")
        if message:
            self._mosaic_status.setText(message)
