from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLayout,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..mosaic_export import MOSAIC_PRESET_NAMES, MosaicWorker, normalise_mosaic_preset
from ..frame_export import export_player_frames
from ..signals import read_synced_signal_csvs
from ..state import (
    LabelInterval,
    ProjectState,
    load_labelled_segments_manifest,
    load_sidecar,
)
from ..widgets.frame_preview import MultiCameraPlayer
from ..widgets.layout import configure_main_splitter
from ..widgets.signal_plot import SignalPlot
from ..widgets.timeline import IntervalItem, TimelineWidget

log = logging.getLogger(__name__)


class _SegmentInfo:
    __slots__ = ("index", "label", "start_sec", "end_sec", "duration_sec", "folder", "color")

    def __init__(self, index: int, interval: LabelInterval):
        self.index = index
        self.label = interval.label
        self.start_sec = interval.start_sec
        self.end_sec = interval.end_sec
        self.duration_sec = interval.end_sec - interval.start_sec
        self.folder = interval.folder
        self.color = interval.color

    @property
    def dir_path(self) -> Path:
        return Path(self.folder)


class Mode4Widget(QWidget):
    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state
        self._segments: list[_SegmentInfo] = []
        self._current_seg: Optional[_SegmentInfo] = None
        self._synced_hr_df: Optional[pd.DataFrame] = None
        self._synced_ecg_df: Optional[pd.DataFrame] = None
        self._synced_ppg_df: Optional[pd.DataFrame] = None
        self._mosaic_worker: MosaicWorker | None = None
        self._seg_hr_df: Optional[pd.DataFrame] = None
        self._seg_ecg_df: Optional[pd.DataFrame] = None
        self._seg_ppg_df: Optional[pd.DataFrame] = None

        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)

        self._meta_btn = QPushButton("Load from metadata file …")
        self._meta_btn.clicked.connect(self._load_from_meta)
        ll.addWidget(self._meta_btn)
        self._load_btn = QPushButton("Load manifest CSV …")
        self._load_btn.clicked.connect(self._load_manifest)
        ll.addWidget(self._load_btn)

        self._global_view_btn = QPushButton("Show global view")
        self._global_view_btn.setToolTip("Clear the selected segment and show all labelled segments")
        self._global_view_btn.setEnabled(False)
        self._global_view_btn.clicked.connect(self._show_global_view)
        ll.addWidget(self._global_view_btn)

        self._seg_list = QListWidget()
        self._seg_list.currentRowChanged.connect(self._on_segment_selected)
        ll.addWidget(self._seg_list)

        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        ll.addWidget(self._info_label)

        # --- Mosaic export ---
        mosaic_grp = QGroupBox("Mosaic video export")
        mosaic_lay = QVBoxLayout(mosaic_grp)
        self._mosaic_btn = QPushButton("Export mosaic for selected segment")
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

        ll.addStretch()
        splitter.addWidget(left)

        self._right_content = QWidget()
        rl = QVBoxLayout(self._right_content)
        rl.setContentsMargins(4, 4, 4, 4)
        rl.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        self._timeline = TimelineWidget()
        self._timeline.setFixedHeight(50)
        rl.addWidget(self._timeline)

        self._player = MultiCameraPlayer(face_blur_enabled=state.blur_faces)
        self._player.setMinimumHeight(300)
        rl.addWidget(self._player)

        self._plot_type = QComboBox()
        self._plot_type.addItems(["Heart rate", "ECG", "Synthetic PPG"])
        self._plot_type.currentIndexChanged.connect(self._show_selected_plot)
        rl.addWidget(self._plot_type)

        self._plot = SignalPlot()
        self._plot.setFixedHeight(420)
        rl.addWidget(self._plot)
        self._right_content.setMinimumHeight(850)

        self._right_scroll = QScrollArea()
        self._right_scroll.setWidgetResizable(True)
        self._right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._right_scroll.verticalScrollBar().setSingleStep(40)
        self._right_scroll.verticalScrollBar().setPageStep(280)
        self._right_scroll.setStyleSheet("QScrollBar:vertical { width: 18px; }")
        self._right_scroll.setWidget(self._right_content)
        splitter.addWidget(self._right_scroll)

        configure_main_splitter(splitter, left, self._right_scroll)

        self._timeline.playhead_moved.connect(self._on_playhead)
        self._player.frame_changed.connect(self._on_frame_changed)
        self._player.export_frames_requested.connect(self._export_current_frames)

        self.refresh_from_state()

    def refresh_from_state(self):
        self._segments = [
            _SegmentInfo(index, interval)
            for index, interval in enumerate(self.state.intervals)
            if interval.folder
        ]
        self._seg_list.clear()
        self._current_seg = None
        self._seg_hr_df = None
        self._seg_ecg_df = None
        self._seg_ppg_df = None
        for segment in self._segments:
            self._seg_list.addItem(
                f"[{segment.index}] {segment.label}  ({segment.duration_sec:.1f}s)"
            )
        self._load_synced_signals_from_segments()
        total_duration = sum(segment.duration_sec for segment in self._segments)
        self._timeline.set_duration(total_duration)
        offset = 0.0
        items = []
        for segment in self._segments:
            items.append(IntervalItem(
                segment.label, offset, offset + segment.duration_sec, segment.color,
            ))
            offset += segment.duration_sec
        self._timeline.set_intervals(items)
        self._show_selected_plot()
        self._set_overview_info()

    def _load_from_meta(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select metadata sidecar", "", "Metadata sidecar (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            load_sidecar(path, self.state)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load metadata:\n{exc}")
            return
        self.refresh_from_state()
        self._info_label.setText(f"Loaded {len(self.state.tracks)} camera(s) from metadata.")

    def _load_manifest(self):
        default_path = self.state.segments_manifest_path or str(
            Path(self.state.output_directory) / "labelled_segments" / "manifest.csv"
        )
        path, _ = QFileDialog.getOpenFileName(
            self, "Select manifest.csv", default_path, "CSV (*.csv);;All files (*)",
        )
        if not path:
            return
        try:
            load_labelled_segments_manifest(path, self.state)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            QMessageBox.critical(self, "Invalid manifest", f"Failed to import labelled segments:\n{exc}")
            return
        self.refresh_from_state()

    def _load_synced_signals_from_segments(self):
        """Load shared synchronized HR and ECG files from state or segment metadata."""
        self._synced_hr_df = self._load_synced_signal(
            self.state.synced_hr_paths, "synced_hr_paths", "synced_signal_path"
        )
        self._synced_ecg_df = self._load_synced_signal(
            self.state.synced_ecg_paths, "synced_ecg_paths"
        )
        ppg_paths = [self.state.synthetic_ppg_path] if self.state.synthetic_ppg_path else []
        self._synced_ppg_df = self._load_synced_signal(
            ppg_paths, "synthetic_ppg_path"
        )

    def _load_synced_signal(
        self,
        state_paths: list[str],
        metadata_key: str,
        legacy_key: str | None = None,
    ) -> Optional[pd.DataFrame]:
        synced_paths = [Path(path) for path in state_paths if Path(path).exists()]
        if synced_paths:
            try:
                return read_synced_signal_csvs(synced_paths)
            except Exception as exc:
                log.error("Error loading shared %s files: %s", metadata_key, exc)

        for seg in self._segments:
            meta_path = seg.dir_path / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            relative_paths = meta.get(metadata_key)
            if relative_paths is None and legacy_key:
                legacy_path = meta.get(legacy_key)
                relative_paths = [legacy_path] if legacy_path else []
            synced_paths = [
                (seg.dir_path / path).resolve()
                for path in (relative_paths or [])
            ]
            synced_paths = [path for path in synced_paths if path.exists()]
            if not synced_paths:
                continue
            try:
                frame = read_synced_signal_csvs(synced_paths)
                log.info("Loaded %s from %s", metadata_key, synced_paths)
                return frame
            except Exception as exc:
                log.error("Error loading %s: %s", metadata_key, exc)
        return None

    def _show_selected_plot(self):
        plot_index = self._plot_type.currentIndex()
        if self._current_seg is not None:
            if plot_index == 0:
                df = self._seg_hr_df
            elif plot_index == 1:
                df = self._seg_ecg_df
            else:
                df = self._seg_ppg_df
            if df is None or df.empty:
                self._plot.clear()
                return
            self._plot.set_data(df, video_duration_sec=self._current_seg.duration_sec)
            return

        if plot_index == 0:
            df = self._synced_hr_df
        elif plot_index == 1:
            df = self._synced_ecg_df
        else:
            df = self._synced_ppg_df
        self._show_signal_overview(df)

    def _show_signal_overview(self, df: Optional[pd.DataFrame]):
        """Display the full synced signal with segment regions shaded."""
        if df is None or df.empty:
            self._plot.clear()
            return

        # Compute total duration from the segments for the overview
        if self._segments:
            max_end = max(s.end_sec for s in self._segments)
        else:
            max_end = 0
        self._plot.set_data(df, video_duration_sec=max_end)

        # Overlay shaded regions for each segment
        for seg in self._segments:
            self._plot._ax.axvspan(
                seg.start_sec, seg.end_sec,
                alpha=0.15, color=seg.color,
                label=seg.label,
            )
        # Avoid duplicate legend entries
        handles, labels = self._plot._ax.get_legend_handles_labels()
        seen = set()
        unique_h, unique_l = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                unique_h.append(h)
                unique_l.append(l)
        if unique_l:
            self._plot._ax.legend(unique_h, unique_l, fontsize=7, loc="upper right")
        self._plot._canvas.draw_idle()

    def _set_overview_info(self) -> None:
        if self._segments:
            source = (
                Path(self.state.segments_manifest_path).name
                if self.state.segments_manifest_path else "shared project state"
            )
            self._info_label.setText(f"Loaded {len(self._segments)} segments from {source}")
        else:
            self._info_label.setText("No labelled segments loaded.")

    def _show_global_view(self) -> None:
        self._current_seg = None
        self._seg_hr_df = None
        self._seg_ecg_df = None
        self._seg_ppg_df = None
        self._seg_list.blockSignals(True)
        self._seg_list.clearSelection()
        self._seg_list.setCurrentRow(-1)
        self._seg_list.blockSignals(False)
        self._player.set_cameras([])
        self._global_view_btn.setEnabled(False)
        self._show_selected_plot()
        self._set_overview_info()

    def _on_segment_selected(self, row: int):
        if row < 0:
            self._show_global_view()
            return
        if row >= len(self._segments):
            return
        seg = self._segments[row]
        self._current_seg = seg
        self._global_view_btn.setEnabled(True)

        seg_dir = seg.dir_path
        if not seg_dir.exists():
            self._info_label.setText(f"Folder not found: {seg_dir}")
            return

        # --- Load videos (prefer meta.json camera info, fallback to glob) ---
        meta_json = seg_dir / "meta.json"
        meta = {}
        if meta_json.exists():
            try:
                meta = json.loads(meta_json.read_text(encoding="utf-8"))
            except Exception:
                pass

        video_files = sorted(seg_dir.glob("cam*.*"))
        if video_files:
            labels = [f.stem for f in video_files]
            self._player.set_cameras(labels)
            self._player.load_videos([str(f) for f in video_files])
        else:
            labels = []
            self._player.set_cameras([])
        camera_rows = max(1, (len(labels) + 1) // 2)
        player_height = camera_rows * 190 + 70
        self._player.setMinimumHeight(player_height)
        self._right_content.setMinimumHeight(player_height + self._plot.height() + 140)

        # --- Load signals: prefer per-segment files, fallback to synced slices ---
        self._seg_hr_df = self._load_segment_signal(
            seg, meta, "hr_files", self._synced_hr_df, "signal_file"
        )
        self._seg_ecg_df = self._load_segment_signal(
            seg, meta, "ecg_files", self._synced_ecg_df
        )
        self._seg_ppg_df = self._load_segment_signal(
            seg, meta, "ppg_files", self._synced_ppg_df
        )
        self._show_selected_plot()

        # --- Info panel ---
        info_lines = [
            f"Segment: {seg.index}",
            f"Label: {seg.label}",
            f"Duration: {seg.duration_sec:.2f}s",
            f"Original time: {seg.start_sec:.2f} – {seg.end_sec:.2f}s",
        ]
        if meta:
            skip = {"index", "label", "start_sec", "end_sec", "duration_sec", "folder",
                "cameras", "signal_file", "hr_files", "ecg_files", "ppg_files",
                "synced_signal_path", "synced_hr_paths", "synced_ecg_paths",
                "synthetic_ppg_path", "generated_at"}
            for k, v in meta.items():
                if k not in skip:
                    info_lines.append(f"{k}: {v}")
            if meta.get("cameras"):
                info_lines.append(f"Cameras: {len(meta['cameras'])}")
        self._info_label.setText("\n".join(info_lines))

    def _load_segment_signal(
        self,
        seg: _SegmentInfo,
        meta: dict,
        files_key: str,
        synced_df: Optional[pd.DataFrame],
        legacy_key: str | None = None,
    ) -> Optional[pd.DataFrame]:
        signal_files = meta.get(files_key)
        if signal_files is None and legacy_key:
            legacy_file = meta.get(legacy_key, "signal.csv")
            signal_files = [legacy_file] if legacy_file else []
        signal_paths = [seg.dir_path / file_name for file_name in (signal_files or [])]
        signal_paths = [path for path in signal_paths if path.exists()]
        if signal_paths:
            try:
                return read_synced_signal_csvs(signal_paths)
            except Exception as exc:
                log.error("Error loading %s for segment %d: %s", files_key, seg.index, exc)

        if synced_df is not None:
            try:
                t0 = synced_df["timestamp_utc"].iloc[0]
                start_ts = t0 + pd.Timedelta(seconds=seg.start_sec)
                end_ts = t0 + pd.Timedelta(seconds=seg.end_sec)
                sliced = synced_df[
                    (synced_df["timestamp_utc"] >= start_ts)
                    & (synced_df["timestamp_utc"] <= end_ts)
                ].copy()
                if not sliced.empty:
                    return sliced
            except Exception as exc:
                log.error("Error slicing %s for segment %d: %s", files_key, seg.index, exc)
        return None

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

    def _export_current_frames(self):
        seg = self._current_seg
        if seg is None:
            QMessageBox.information(self, "Info", "Select a segment first.")
            return
        if not seg.dir_path.exists():
            QMessageBox.warning(self, "Warning", f"Folder not found: {seg.dir_path}")
            return

        try:
            written = export_player_frames(self._player, seg.dir_path, prefix=f"segment_{seg.index:04d}")
        except Exception as exc:
            QMessageBox.critical(self, "Synchronized capture", f"Capture export failed:\n{exc}")
            self._mosaic_status.setText(f"Capture export failed: {exc}")
            return

        capture_dir = written[0].parent
        self._mosaic_status.setText(f"Exported {len(written)} capture frame(s) to {capture_dir}")

    # ------------------------------------------------------------------
    # Mosaic export
    # ------------------------------------------------------------------

    def _export_mosaic(self):
        seg = self._current_seg
        if seg is None:
            QMessageBox.information(self, "Info", "Select a segment first.")
            return

        seg_dir = seg.dir_path
        video_files = sorted(seg_dir.glob("cam*.*"))
        if not video_files:
            QMessageBox.warning(self, "Warning", "No video files found in segment folder.")
            return

        video_paths = [str(f) for f in video_files]
        labels = [f.stem for f in video_files]

        default_name = str(seg_dir / f"mosaic_{seg.index:04d}_{seg.label.replace(' ', '_')}.mp4")
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

        fps = self._player.get_fps() or 30.0
        total_frames = int(seg.duration_sec * fps)
        if self._plot_type.currentIndex() == 1:
            signal_df = self._seg_ecg_df
        elif self._plot_type.currentIndex() == 2:
            signal_df = self._seg_ppg_df
        else:
            signal_df = self._seg_hr_df

        self._mosaic_worker = MosaicWorker(
            video_paths=video_paths,
            camera_labels=labels,
            fps=fps,
            total_frames=total_frames,
            video_duration_sec=seg.duration_sec,
            output_path=path,
            signal_df=signal_df,
            quality_preset=preset_key,
            ffmpeg_path=self.state.ffmpeg_path,
            blur_faces=self.state.blur_faces,
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
