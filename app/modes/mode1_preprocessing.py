from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..file_cleanup import remove_or_archive
from ..ffmpeg_utils import (
    FFmpegWorker,
    concatenate_segments,
    find_ffmpeg,
    find_ffprobe,
    probe_video,
    trim_video,
)
from ..state import ProjectState, VideoTrack, parse_dji_datetime, dji_datetime_str
from ..state import generate_sidecar
from ..widgets.frame_preview import MultiCameraPlayer
from ..widgets.layout import configure_main_splitter

log = logging.getLogger(__name__)

_VIDEO_FILTER = "Videos (*.mp4 *.mov *.lrf *.avi *.mkv);;All files (*)"


class _MetadataPanel(QGroupBox):
    def __init__(self, title="Metadata", parent=None):
        super().__init__(title, parent)
        layout = QFormLayout(self)
        self._rows: dict[str, QLabel] = {}

    def set_info(self, info: dict) -> None:
        for key in list(self._rows.keys()):
            self._rows[key].setParent(None)
            self._rows[key].deleteLater()
        self._rows.clear()
        layout: QFormLayout = self.layout()
        while layout.count():
            layout.removeRow(0)
        for k, v in info.items():
            lbl = QLabel(str(v))
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addRow(k + ":", lbl)
            self._rows[k] = lbl


class _CameraGroup(QGroupBox):
    segments_changed = pyqtSignal()

    def __init__(self, index: int, parent=None):
        super().__init__(f"Camera {index + 1}", parent)
        self.camera_index = index
        layout = QVBoxLayout(self)

        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("Add segments …")
        self._add_btn.clicked.connect(self._add_segments)
        btn_row.addWidget(self._add_btn)

        self._remove_btn = QPushButton("Remove selected")
        self._remove_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(self._remove_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._clear_all)
        btn_row.addWidget(self._clear_btn)
        layout.addLayout(btn_row)

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        layout.addWidget(self._list)

        self._label_edit = QLineEdit()
        self._label_edit.setPlaceholderText("Camera label (e.g. front, left)")
        layout.addWidget(self._label_edit)

        self._not_limiting_cb = QCheckBox("Not limiting size")
        self._not_limiting_cb.setToolTip(
            "Exclude this camera when choosing the common final duration. "
            "Useful when this camera ends earlier than the rest."
        )
        layout.addWidget(self._not_limiting_cb)

    @property
    def label(self) -> str:
        return self._label_edit.text().strip() or f"Camera {self.camera_index + 1}"

    @label.setter
    def label(self, text: str):
        self._label_edit.setText(text)

    @property
    def segment_paths(self) -> list[str]:
        return [self._list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self._list.count())]

    @segment_paths.setter
    def segment_paths(self, paths: list[str]):
        self._list.clear()
        for p in paths:
            item = QListWidgetItem(Path(p).name)
            item.setData(Qt.ItemDataRole.UserRole, p)
            item.setToolTip(p)
            self._list.addItem(item)

    @property
    def limits_common_duration(self) -> bool:
        return not self._not_limiting_cb.isChecked()

    @limits_common_duration.setter
    def limits_common_duration(self, value: bool):
        self._not_limiting_cb.setChecked(not value)

    def _add_segments(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, f"Select segments for Camera {self.camera_index + 1}",
            "", _VIDEO_FILTER,
        )
        if files:
            files.sort()
            for p in files:
                item = QListWidgetItem(Path(p).name)
                item.setData(Qt.ItemDataRole.UserRole, p)
                item.setToolTip(p)
                self._list.addItem(item)
            self.segments_changed.emit()

    def _remove_selected(self):
        for item in self._list.selectedItems():
            self._list.takeItem(self._list.row(item))
        self.segments_changed.emit()

    def _clear_all(self):
        self._list.clear()
        self.segments_changed.emit()


class Mode1Widget(QWidget):
    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state
        self._worker: Optional[FFmpegWorker] = None

        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)

        setup_group = QGroupBox("Setup")
        setup_lay = QFormLayout(setup_group)

        self._ffmpeg_edit = QLineEdit(state.ffmpeg_path or find_ffmpeg())
        browse_ff = QPushButton("…")
        browse_ff.setFixedWidth(30)
        browse_ff.clicked.connect(self._browse_ffmpeg)
        ff_row = QHBoxLayout()
        ff_row.addWidget(self._ffmpeg_edit)
        ff_row.addWidget(browse_ff)
        setup_lay.addRow("FFmpeg:", ff_row)

        self._out_dir_edit = QLineEdit(state.output_directory)
        browse_out = QPushButton("…")
        browse_out.setFixedWidth(30)
        browse_out.clicked.connect(self._browse_outdir)
        out_row = QHBoxLayout()
        out_row.addWidget(self._out_dir_edit)
        out_row.addWidget(browse_out)
        setup_lay.addRow("Output dir:", out_row)

        self._num_cameras_spin = QSpinBox()
        self._num_cameras_spin.setRange(1, 6)
        self._num_cameras_spin.setValue(state.num_cameras)
        self._num_cameras_spin.valueChanged.connect(self._rebuild_camera_groups)
        setup_lay.addRow("Cameras:", self._num_cameras_spin)
        left_layout.addWidget(setup_group)

        self._camera_area = QVBoxLayout()
        self._camera_groups: list[_CameraGroup] = []
        left_layout.addLayout(self._camera_area)

        sync_row = QHBoxLayout()
        sync_row.addWidget(QLabel("Audio sync duration (s):"))
        self._sync_duration_spin = QDoubleSpinBox()
        self._sync_duration_spin.setRange(0, 600)
        self._sync_duration_spin.setValue(60.0)
        self._sync_duration_spin.setDecimals(1)
        self._sync_duration_spin.setSingleStep(10)
        self._sync_duration_spin.setSpecialValueText("Full audio")
        self._sync_duration_spin.setToolTip(
            "How many seconds of audio to use for cross-correlation sync.\n"
            "Set to 0 to use the entire audio (slower)."
        )
        sync_row.addWidget(self._sync_duration_spin)
        left_layout.addLayout(sync_row)

        self._sync_pairwise_refinement_cb = QCheckBox("Refine audio sync with pairwise camera checks")
        self._sync_pairwise_refinement_cb.setChecked(False)
        self._sync_pairwise_refinement_cb.setToolTip(
            "Optional slower check for difficult recordings. It compares all camera pairs\n"
            "and keeps the offset set with the best overall consistency."
        )
        left_layout.addWidget(self._sync_pairwise_refinement_cb)

        self._delete_intermediates_cb = QCheckBox("Delete intermediate files automatically")
        self._delete_intermediates_cb.setChecked(True)
        self._delete_intermediates_cb.setToolTip(
            "When checked, concatenated and start-trimmed files are deleted\n"
            "as soon as the pipeline no longer needs them."
        )
        left_layout.addWidget(self._delete_intermediates_cb)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run preprocessing")
        self._run_btn.setStyleSheet("font-weight:bold; padding:8px;")
        self._run_btn.clicked.connect(self._run_preprocessing)
        btn_row.addWidget(self._run_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)
        btn_row.addWidget(self._cancel_btn)
        left_layout.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        self._progress.setValue(0)
        left_layout.addWidget(self._progress)

        self._status_label = QLabel("")
        left_layout.addWidget(self._status_label)

        self._log_area = QTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setMaximumHeight(120)
        left_layout.addWidget(self._log_area)

        left_layout.addStretch()
        scroll.setWidget(left)
        splitter.addWidget(scroll)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self._player = MultiCameraPlayer(face_blur_enabled=state.blur_faces)
        right_layout.addWidget(self._player)

        self._metadata_panel = _MetadataPanel("Video metadata")
        right_layout.addWidget(self._metadata_panel)
        splitter.addWidget(right)

        configure_main_splitter(splitter, scroll, right, 2, 3)

        self._rebuild_camera_groups()
        self._restore_from_state()

    def _browse_ffmpeg(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select ffmpeg executable")
        if path:
            self._ffmpeg_edit.setText(path)

    def _browse_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self._out_dir_edit.setText(d)

    def _rebuild_camera_groups(self):
        old_data = [(g.label, g.segment_paths, g.limits_common_duration) for g in self._camera_groups]
        for g in self._camera_groups:
            g.setParent(None)
            g.deleteLater()
        self._camera_groups.clear()
        n = self._num_cameras_spin.value()
        for i in range(n):
            g = _CameraGroup(i)
            if i < len(old_data):
                g.label = old_data[i][0]
                g.segment_paths = old_data[i][1]
                g.limits_common_duration = old_data[i][2]
            g.segments_changed.connect(self._preview_first_segments)
            self._camera_area.addWidget(g)
            self._camera_groups.append(g)
        labels = [g.label for g in self._camera_groups]
        self._player.set_cameras(labels)

    def _restore_from_state(self):
        for i, track in enumerate(self.state.tracks):
            if i < len(self._camera_groups):
                self._camera_groups[i].label = track.camera_label
                self._camera_groups[i].segment_paths = track.segment_paths
                self._camera_groups[i].limits_common_duration = track.limits_common_duration
        if self.state.tracks:
            paths = [t.final_output_path or t.concatenated_path for t in self.state.tracks]
            valid = [p for p in paths if p and Path(p).exists()]
            if valid:
                labels = [t.camera_label for t in self.state.tracks]
                self._player.set_cameras(labels)
                self._player.load_videos(valid)

    def _preview_first_segments(self):
        """Load the first segment of each camera into the preview player."""
        labels = [g.label for g in self._camera_groups]
        self._player.set_cameras(labels)
        paths = [g.segment_paths[0] if g.segment_paths else "" for g in self._camera_groups]
        if any(paths):
            self._player.load_videos(paths)

    def _log(self, msg: str):
        self._log_area.append(msg)
        log.info(msg)

    def _collect_state(self):
        self.state.ffmpeg_path = self._ffmpeg_edit.text().strip()
        self.state.output_directory = self._out_dir_edit.text().strip()
        self.state.num_cameras = self._num_cameras_spin.value()

    def _run_preprocessing(self):
        self._collect_state()
        ffmpeg = self.state.ffmpeg_path or find_ffmpeg()
        if not ffmpeg:
            QMessageBox.critical(self, "Error", "FFmpeg not found. Please set the path.")
            return
        out_dir = self.state.output_directory
        if not out_dir:
            QMessageBox.critical(self, "Error", "Please set an output directory.")
            return
        os.makedirs(out_dir, exist_ok=True)

        cameras = self._camera_groups
        for i, cam in enumerate(cameras):
            if not cam.segment_paths:
                QMessageBox.warning(self, "Warning", f"Camera {i+1} has no segments.")
                return

        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress.setValue(0)
        self._log("Starting preprocessing …")

        worker = FFmpegWorker(
            self._preprocessing_pipeline,
            ffmpeg=ffmpeg,
            out_dir=out_dir,
            cameras=cameras,
            delete_intermediates=self._delete_intermediates_cb.isChecked(),
            sync_audio_duration=self._sync_duration_spin.value() or None,
            sync_pairwise_refinement=self._sync_pairwise_refinement_cb.isChecked(),
        )
        worker.log_message.connect(self._log)
        worker.progress.connect(lambda v, m: (self._progress.setValue(v), self._status_label.setText(m)))
        worker.finished.connect(self._on_finished)
        self._worker = worker
        worker.start()

    def _cancel(self):
        if self._worker:
            self._worker.cancel()

    def _on_finished(self, ok: bool, msg: str):
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress.setValue(100 if ok else 0)
        self._status_label.setText(msg)
        self._log(msg)
        self._worker = None
        if ok:
            self.state.mode1_complete = True
            # Generate metadata sidecar
            try:
                sidecar = generate_sidecar(self.state)
                self._log(f"Wrote metadata sidecar: {Path(sidecar).name}")
            except Exception as exc:
                self._log(f"Warning: could not write sidecar: {exc}")
            paths = [t.final_output_path for t in self.state.tracks]
            labels = [t.camera_label for t in self.state.tracks]
            self._player.set_cameras(labels)
            self._player.load_videos(paths)
            self._update_metadata()

    def _update_metadata(self):
        info = {}
        ffprobe = find_ffprobe()
        for i, t in enumerate(self.state.tracks):
            p = t.final_output_path
            if p and Path(p).exists():
                try:
                    vi = probe_video(p, ffprobe)
                    info[f"Cam{i+1} codec"] = vi.get("codec", "")
                    info[f"Cam{i+1} resolution"] = f'{vi.get("width",0)}x{vi.get("height",0)}'
                    info[f"Cam{i+1} fps"] = f'{vi.get("fps",0):.2f}'
                    info[f"Cam{i+1} frames"] = vi.get("frame_count", 0)
                    info[f"Cam{i+1} duration"] = f'{vi.get("duration",0):.2f}s'
                    info[f"Cam{i+1} start"] = dji_datetime_str(t.parsed_start_datetime())
                except Exception as exc:
                    info[f"Cam{i+1} error"] = str(exc)
        self._metadata_panel.set_info(info)

    def _preprocessing_pipeline(self, *, ffmpeg, out_dir, cameras, delete_intermediates, sync_audio_duration, sync_pairwise_refinement, worker: FFmpegWorker):
        import shutil as _shutil
        num = len(cameras)
        tracks: list[VideoTrack] = []
        single_camera = (num == 1)

        def cleanup_intermediate(path: str) -> None:
            remove_or_archive(
                path,
                out_dir,
                self.state.archive_removed_files,
            )
            action = "Archived" if self.state.archive_removed_files else "Deleted"
            worker.log_message.emit(f"{action} intermediate: {Path(path).name}")

        # --- Concatenation (0–55%) -------------------------------------------
        for ci, cam in enumerate(cameras):
            if worker.is_cancelled:
                return
            seg_paths = cam.segment_paths
            label = cam.label

            first_dt = parse_dji_datetime(Path(seg_paths[0]).stem)

            step_label = f"Concatenating camera {ci+1}/{num} …"
            worker.progress.emit(int(ci * 55 / num), step_label)
            worker.log_message.emit(f"Camera {ci+1}: {len(seg_paths)} segment(s)")

            # Single camera with one segment: use original file directly
            if single_camera and len(seg_paths) == 1:
                worker.log_message.emit("Single camera, single segment — skipping concatenation.")
                final_name = f"cam{ci+1}_{label}_final.mp4"
                final_path = str(Path(out_dir) / final_name)
                _shutil.copy2(seg_paths[0], final_path)
                vinfo = probe_video(final_path, find_ffprobe())
                track = VideoTrack(
                    camera_index=ci,
                    camera_label=label,
                    segment_paths=seg_paths,
                    final_output_path=final_path,
                    fps=vinfo.get("fps", 0),
                    frame_count=vinfo.get("frame_count", 0),
                    width=vinfo.get("width", 0),
                    height=vinfo.get("height", 0),
                    codec=vinfo.get("codec", ""),
                    duration_sec=vinfo.get("duration", 0),
                    limits_common_duration=cam.limits_common_duration,
                )
                track.set_start_datetime(first_dt)
                tracks.append(track)
                continue

            concat_name = f"cam{ci+1}_{label}_concat.mp4"
            concat_path = str(Path(out_dir) / concat_name)

            base_pct = ci * 55 / num
            span_pct = 55 / num

            def _concat_progress(cur, total, _b=base_pct, _s=span_pct):
                frac = min(cur / total, 1.0) if total > 0 else 0
                worker.progress.emit(int(_b + frac * _s), step_label)

            concatenate_segments(
                seg_paths, concat_path, ffmpeg=ffmpeg,
                log_callback=lambda m: worker.log_message.emit(m),
                progress_callback=_concat_progress,
                cancel_check=lambda: worker.is_cancelled,
            )

            vinfo = probe_video(concat_path, find_ffprobe())

            track = VideoTrack(
                camera_index=ci,
                camera_label=label,
                segment_paths=seg_paths,
                concatenated_path=concat_path,
                fps=vinfo.get("fps", 0),
                frame_count=vinfo.get("frame_count", 0),
                width=vinfo.get("width", 0),
                height=vinfo.get("height", 0),
                codec=vinfo.get("codec", ""),
                duration_sec=vinfo.get("duration", 0),
                limits_common_duration=cam.limits_common_duration,
            )
            track.set_start_datetime(first_dt)
            tracks.append(track)

        if worker.is_cancelled:
            return

        # --- Single camera: skip sync & trim, use concat as final -------------
        if single_camera:
            t = tracks[0]
            if not t.final_output_path:
                # Multi-segment single camera: rename concat → final
                final_name = f"cam1_{t.camera_label}_final.mp4"
                final_path = str(Path(out_dir) / final_name)
                _shutil.copy2(t.concatenated_path, final_path)
                if delete_intermediates and t.concatenated_path and Path(t.concatenated_path).exists():
                    try:
                        cleanup_intermediate(t.concatenated_path)
                    except OSError as exc:
                        worker.log_message.emit(f"Could not clean up {t.concatenated_path}: {exc}")
                    t.concatenated_path = ""
                t.final_output_path = final_path
                vf = probe_video(final_path, find_ffprobe())
                t.duration_sec = vf.get("duration", 0)
                t.frame_count = vf.get("frame_count", 0)
                t.fps = vf.get("fps", t.fps)
            worker.log_message.emit("Single camera — skipping audio sync and trim steps.")
            worker.progress.emit(98, "Finalising …")
            self.state.tracks = tracks
            worker.progress.emit(100, "Done ✓")
            return

        # --- Audio sync (55–70%) ---------------------------------------------
        worker.progress.emit(55, "Synchronising audio …")
        first_segments = [cam.segment_paths[0] for cam in cameras]
        from ..audio_sync import compute_all_offsets
        offsets = compute_all_offsets(first_segments, ffmpeg=ffmpeg,
                                        audio_duration_sec=sync_audio_duration,
                                        pairwise_refinement=sync_pairwise_refinement,
                                        log_callback=lambda m: worker.log_message.emit(m))
        for i, off in enumerate(offsets):
            tracks[i].sync_offset_sec = off
            worker.log_message.emit(f"Camera {i+1} offset: {off:.4f}s")

        if worker.is_cancelled:
            return

        # --- Trim starts (70–85%) --------------------------------------------
        worker.progress.emit(70, "Trimming starts …")
        max_start_offset = max(t.sync_offset_sec for t in tracks)
        for i, t in enumerate(tracks):
            if worker.is_cancelled:
                return
            trim_start = max_start_offset - t.sync_offset_sec
            if trim_start > 0.01:
                trimmed_name = f"cam{i+1}_{t.camera_label}_trimstart.mp4"
                trimmed_path = str(Path(out_dir) / trimmed_name)
                expected_dur = t.duration_sec - trim_start

                step_label = f"Trimming start camera {i+1}/{num} …"
                worker.progress.emit(int(70 + i * 15 / num), step_label)

                def _trim_progress(cur_s, total_s, _i=i):
                    frac = min(cur_s / total_s, 1.0) if total_s > 0 else 0
                    worker.progress.emit(int(70 + (_i * 15 / num) + frac * 15 / num), step_label)

                trim_video(
                    t.concatenated_path, trimmed_path,
                    start_sec=trim_start, ffmpeg=ffmpeg,
                    log_callback=lambda m: worker.log_message.emit(m),
                    progress_callback=_trim_progress,
                    expected_duration_sec=expected_dur,
                    cancel_check=lambda: worker.is_cancelled,
                )
                t.trimstart_path = trimmed_path
                vinfo2 = probe_video(trimmed_path)
                t.duration_sec = vinfo2.get("duration", 0)
                t.frame_count = vinfo2.get("frame_count", 0)

                # Cleanup: concat file is no longer needed
                if delete_intermediates and t.concatenated_path and Path(t.concatenated_path).exists():
                    try:
                        cleanup_intermediate(t.concatenated_path)
                    except OSError as exc:
                        worker.log_message.emit(f"Could not clean up {t.concatenated_path}: {exc}")
                    t.concatenated_path = ""

        if worker.is_cancelled:
            return

        # --- Trim to common duration (85–98%) ---------------------------------
        worker.progress.emit(85, "Trimming to common duration …")
        limiting_tracks = [t for t in tracks if t.limits_common_duration and t.duration_sec > 0]
        if limiting_tracks:
            target_dur = min(t.duration_sec for t in limiting_tracks)
            excluded = [t.camera_label or f"Camera {i + 1}" for i, t in enumerate(tracks) if not t.limits_common_duration]
            worker.log_message.emit(
                f"Common duration from limiting cameras: {target_dur:.3f}s"
            )
            if excluded:
                worker.log_message.emit("Not limiting size: " + ", ".join(excluded))
        else:
            target_dur = max((t.duration_sec for t in tracks), default=0.0)
            worker.log_message.emit(
                "All cameras are marked not limiting size; keeping each available duration."
            )
        for i, t in enumerate(tracks):
            if worker.is_cancelled:
                return
            final_name = f"cam{i+1}_{t.camera_label}_final.mp4"
            final_path = str(Path(out_dir) / final_name)
            # Source for the end-trim is trimstart if it exists, else concat
            source = t.trimstart_path or t.concatenated_path

            step_label = f"Trimming end camera {i+1}/{num} …"
            worker.progress.emit(int(85 + i * 13 / num), step_label)

            if target_dur > 0 and t.duration_sec - target_dur > 0.1:
                def _trim_end_progress(cur_s, total_s, _i=i):
                    frac = min(cur_s / total_s, 1.0) if total_s > 0 else 0
                    worker.progress.emit(int(85 + (_i * 13 / num) + frac * 13 / num), step_label)

                trim_video(
                    source, final_path,
                    duration_sec=target_dur, ffmpeg=ffmpeg,
                    log_callback=lambda m: worker.log_message.emit(m),
                    progress_callback=_trim_end_progress,
                    expected_duration_sec=target_dur,
                    cancel_check=lambda: worker.is_cancelled,
                )
            else:
                import shutil
                shutil.copy2(source, final_path)
            t.final_output_path = final_path
            vinfo3 = probe_video(final_path)
            t.duration_sec = vinfo3.get("duration", 0)
            t.frame_count = vinfo3.get("frame_count", 0)
            t.fps = vinfo3.get("fps", t.fps)

            # Cleanup: source intermediate is no longer needed
            if delete_intermediates:
                if t.trimstart_path and Path(t.trimstart_path).exists():
                    try:
                        cleanup_intermediate(t.trimstart_path)
                    except OSError as exc:
                        worker.log_message.emit(f"Could not clean up {t.trimstart_path}: {exc}")
                    t.trimstart_path = ""
                elif t.concatenated_path and Path(t.concatenated_path).exists():
                    # No trimstart was created, so concat was used directly as source
                    try:
                        cleanup_intermediate(t.concatenated_path)
                    except OSError as exc:
                        worker.log_message.emit(f"Could not clean up {t.concatenated_path}: {exc}")
                    t.concatenated_path = ""

        if worker.is_cancelled:
            return

        frame_counts = [t.frame_count for t in tracks]
        worker.log_message.emit(f"Frame counts: {frame_counts}")
        if len(set(frame_counts)) > 1:
            worker.log_message.emit(
                "Warning: frame counts differ after trimming. This is expected when a shorter camera is marked not limiting size."
            )

        self.state.tracks = tracks
        worker.progress.emit(100, "Done ✓")
