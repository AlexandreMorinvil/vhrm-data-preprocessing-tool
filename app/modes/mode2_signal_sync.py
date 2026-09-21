from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from PyQt6.QtCore import Qt, QTime
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
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

from ..mosaic_export import MOSAIC_PRESET_NAMES, MosaicWorker, normalise_mosaic_preset
from ..signals import (
    get_loaders,
    load_signal_files,
    read_synced_signal_csv,
    signal_file_type_name,
    signal_long_to_wide,
)
from ..state import (
    ProjectState,
    compute_signal_anchor,
    format_time_coherence_warnings,
    generate_sidecar,
    load_sidecar,
    populate_tracks_from_videos,
    video_timeline_duration_sec,
)
from ..widgets.layout import configure_main_splitter
from ..widgets.frame_preview import MultiCameraPlayer
from ..widgets.signal_plot import SignalPlot

log = logging.getLogger(__name__)

_SIGNAL_FILTER = "CSV / signal files (*.csv *.tsv *.txt);;All files (*)"
_META_FILTER = "Metadata sidecar (*.json);;All files (*)"
_VIDEO_FILTER = "Videos (*.mp4 *.mov *.lrf *.avi *.mkv);;All files (*)"


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

        time_grp = QGroupBox("Camera time correction")
        time_lay = QVBoxLayout(time_grp)
        self._time_grid = QGridLayout()
        time_lay.addLayout(self._time_grid)
        self._check_time_btn = QPushButton("Check coherence")
        self._check_time_btn.clicked.connect(self._check_time_coherence)
        time_lay.addWidget(self._check_time_btn)
        self._time_status = QLabel("")
        self._time_status.setWordWrap(True)
        time_lay.addWidget(self._time_status)
        ll.addWidget(time_grp)

        loader_names = [getattr(l, "display_name", type(l).__name__) for l in get_loaders()]
        info = QLabel(f"Available loaders: {', '.join(loader_names) or 'none'}")
        info.setWordWrap(True)
        ll.addWidget(info)

        hr_grp = QGroupBox("Heart-rate signals")
        hr_lay = QVBoxLayout(hr_grp)
        self._hr_file_list = QListWidget()
        hr_lay.addWidget(self._hr_file_list)
        hr_btn = QPushButton("Add HR file …")
        hr_btn.clicked.connect(lambda: self._add_signal(self._hr_file_list, "HR"))
        hr_lay.addWidget(hr_btn)
        ll.addWidget(hr_grp)

        ecg_grp = QGroupBox("ECG signals")
        ecg_lay = QVBoxLayout(ecg_grp)
        self._ecg_file_list = QListWidget()
        ecg_lay.addWidget(self._ecg_file_list)
        ecg_btn = QPushButton("Add ECG file …")
        ecg_btn.clicked.connect(lambda: self._add_signal(self._ecg_file_list, "ECG"))
        ecg_lay.addWidget(ecg_btn)
        ll.addWidget(ecg_grp)

        self._avg_checkbox = QCheckBox("Include average column")
        self._avg_checkbox.setChecked(state.include_signal_average)
        self._avg_checkbox.toggled.connect(self._on_avg_toggled)
        ll.addWidget(self._avg_checkbox)

        self._load_btn = QPushButton("Load && synchronise HR / ECG")
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
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)

        self._player = MultiCameraPlayer()
        rl.addWidget(self._player)

        self._plot_type = QComboBox()
        self._plot_type.addItems(["Heart rate", "ECG"])
        self._plot_type.currentIndexChanged.connect(self._show_selected_plot)
        rl.addWidget(self._plot_type)
        self._plot = SignalPlot()
        rl.addWidget(self._plot)
        splitter.addWidget(right)

        configure_main_splitter(splitter, left_scroll, right)

        self._player.frame_changed.connect(self._on_frame_changed)

        self._restore_from_state()

    def _restore_from_state(self):
        self._refresh_signal_file_lists()

        if self.state.tracks:
            labels = [t.camera_label for t in self.state.tracks]
            self._player.set_cameras(labels)
            paths = [t.final_output_path for t in self.state.tracks]
            valid = [p for p in paths if p and Path(p).exists()]
            if valid:
                self._player.load_videos(valid)
        self._refresh_time_correction_ui()

    def _refresh_signal_file_lists(self):
        self._hr_file_list.clear()
        self._ecg_file_list.clear()
        for p in self.state.hr_signal_paths or self.state.signal_paths:
            self._hr_file_list.addItem(self._make_signal_item(p, "HR"))
        if self.state.synced_hr_path:
            self._hr_file_list.addItem(self._make_synced_item(self.state.synced_hr_path, "HR"))
        for p in self.state.ecg_signal_paths:
            self._ecg_file_list.addItem(self._make_signal_item(p, "ECG"))
        if self.state.synced_ecg_path:
            self._ecg_file_list.addItem(self._make_synced_item(self.state.synced_ecg_path, "ECG"))

    def _make_signal_item(self, path: str, signal_kind: str = "signal") -> QListWidgetItem:
        sensor_type = signal_file_type_name(path) if Path(path).exists() else f"{signal_kind} source (file unavailable)"
        item = QListWidgetItem(f"{Path(path).name} — {sensor_type}")
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(f"{path}\nDetected type: {sensor_type}")
        return item

    @staticmethod
    def _make_synced_item(path: str, signal_kind: str) -> QListWidgetItem:
        availability = "Synchronized" if Path(path).exists() else "Synchronized file unavailable"
        item = QListWidgetItem(f"{Path(path).name} — {availability} {signal_kind}")
        item.setToolTip(path)
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
        self._refresh_time_correction_ui()
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
        self._hr_merged_df = None
        self._ecg_merged_df = None
        hr_path = self.state.synced_hr_path or self.state.synced_signal_path
        if hr_path and Path(hr_path).exists():
            try:
                self._hr_merged_df = read_synced_signal_csv(hr_path)
            except Exception as exc:
                log.warning("Could not load synchronized HR CSV %s: %s", hr_path, exc)
        ecg_path = self.state.synced_ecg_path
        if ecg_path and Path(ecg_path).exists():
            try:
                self._ecg_merged_df = read_synced_signal_csv(ecg_path)
            except Exception as exc:
                log.warning("Could not load synchronized ECG CSV %s: %s", ecg_path, exc)
        self._show_selected_plot()
        if self._hr_merged_df is not None or self._ecg_merged_df is not None:
            self._status.setText("Synchronized HR/ECG data loaded")

    def _add_signal(self, target: QListWidget, signal_kind: str):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select signal files", "", _SIGNAL_FILTER,
        )
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

    def _on_avg_toggled(self, checked):
        self.state.include_signal_average = checked

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
            self._time_grid.addWidget(QLabel("Load videos to edit time corrections."), 1, 0, 1, 5)
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

        loaded_hr = self._load_clip_and_pivot(hr_paths, anchor, duration, self.state.include_signal_average)
        loaded_ecg = self._load_clip_and_pivot(ecg_paths, anchor, duration, False)
        if loaded_hr is not None:
            self._hr_merged_df = loaded_hr
        if loaded_ecg is not None:
            self._ecg_merged_df = loaded_ecg
        if self._hr_merged_df is None and self._ecg_merged_df is None:
            QMessageBox.warning(self, "Warning", "No signal data within the video time range.")
            return

        out_dir = self.state.output_directory
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            if self._hr_merged_df is not None:
                self.state.synced_hr_path = str(Path(out_dir) / "hr_synced.csv")
                self.state.synced_signal_path = self.state.synced_hr_path
                self._hr_merged_df.to_csv(self.state.synced_hr_path, index=False)
            if self._ecg_merged_df is not None:
                self.state.synced_ecg_path = str(Path(out_dir) / "ecg_synced.csv")
                self._ecg_merged_df.to_csv(self.state.synced_ecg_path, index=False)
            try:
                generate_sidecar(self.state)
            except Exception as exc:
                log.warning("Could not update sidecar: %s", exc)

        self._show_selected_plot()
        status_parts = []
        if self._hr_merged_df is not None:
            status_parts.append(f"HR: {len(self._hr_merged_df)} samples")
        if self._ecg_merged_df is not None:
            status_parts.append(f"ECG: {len(self._ecg_merged_df)} samples")
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
        df = self._hr_merged_df if self._plot_type.currentIndex() == 0 else self._ecg_merged_df
        if df is None:
            self._plot.clear()
            return
        self._plot.set_data(df, video_duration_sec=video_timeline_duration_sec(self.state.tracks))

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
        duration = video_timeline_duration_sec(self.state.tracks)
        total_frames = int(duration * fps) if duration > 0 else (t0.frame_count or 0)
        labels = [t.camera_label for t in self.state.tracks]

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
        if not self.state.tracks:
            return
        fps = self.state.tracks[0].fps or 30.0
        time_sec = frame_no / fps
        self._plot.set_cursor(time_sec)
