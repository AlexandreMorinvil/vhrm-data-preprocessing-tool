from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
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

from ..mosaic_export import MOSAIC_PRESET_NAMES, MosaicWorker, normalise_mosaic_preset
from ..frame_export import export_player_frames
from ..signals import read_synced_signal_csv
from ..state import ProjectState
from ..widgets.frame_preview import MultiCameraPlayer
from ..widgets.layout import configure_main_splitter
from ..widgets.signal_plot import SignalPlot
from ..widgets.timeline import IntervalItem, TimelineWidget

log = logging.getLogger(__name__)


class _SegmentInfo:
    __slots__ = ("index", "label", "start_sec", "end_sec", "duration_sec", "folder", "base_dir", "color")

    def __init__(self, row: dict, base_dir: Path):
        self.index = int(row.get("index", 0))
        self.label = row.get("label", "")
        self.start_sec = float(row.get("start_sec", 0))
        self.end_sec = float(row.get("end_sec", 0))
        self.duration_sec = float(row.get("duration_sec", 0))
        self.folder = row.get("folder", "")
        self.base_dir = base_dir
        self.color = "#4488cc"

    @property
    def dir_path(self) -> Path:
        return self.base_dir / self.folder


class Mode4Widget(QWidget):
    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state
        self._segments: list[_SegmentInfo] = []
        self._current_seg: Optional[_SegmentInfo] = None
        self._synced_signal_df: Optional[pd.DataFrame] = None
        self._mosaic_worker: MosaicWorker | None = None
        self._seg_signal_df: Optional[pd.DataFrame] = None

        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)

        self._load_btn = QPushButton("Load manifest CSV …")
        self._load_btn.clicked.connect(self._load_manifest)
        ll.addWidget(self._load_btn)

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

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)

        self._timeline = TimelineWidget()
        self._timeline.setFixedHeight(50)
        rl.addWidget(self._timeline)

        self._player = MultiCameraPlayer()
        rl.addWidget(self._player)

        self._plot = SignalPlot()
        rl.addWidget(self._plot)
        splitter.addWidget(right)

        configure_main_splitter(splitter, left, right)

        self._timeline.playhead_moved.connect(self._on_playhead)
        self._player.frame_changed.connect(self._on_frame_changed)
        self._player.export_frames_requested.connect(self._export_current_frames)

    def _load_manifest(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select manifest.csv", "", "CSV (*.csv);;All files (*)",
        )
        if not path:
            return
        base_dir = Path(path).parent
        self._segments.clear()
        self._seg_list.clear()
        self._synced_signal_df = None
        self._current_seg = None

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seg = _SegmentInfo(row, base_dir)
                self._segments.append(seg)
                item = QListWidgetItem(f"[{seg.index}] {seg.label}  ({seg.duration_sec:.1f}s)")
                self._seg_list.addItem(item)

        # --- Discover synced signal from first segment's meta.json ---
        self._load_synced_signal_from_segments()

        # --- Timeline setup ---
        total_dur = sum(s.end_sec - s.start_sec for s in self._segments)
        self._timeline.set_duration(total_dur)
        items = []
        offset = 0.0
        for seg in self._segments:
            dur = seg.end_sec - seg.start_sec
            items.append(IntervalItem(seg.label, offset, offset + dur, seg.color))
            offset += dur
        self._timeline.set_intervals(items)

        # --- Show signal overview with shaded regions ---
        if self._synced_signal_df is not None:
            self._show_signal_overview()
        else:
            self._plot.clear()

        self._info_label.setText(f"Loaded {len(self._segments)} segments from {Path(path).name}")

    def _load_synced_signal_from_segments(self):
        """Read first segment's meta.json to find synced_signal_path and load it."""
        if not self._segments:
            return
        for seg in self._segments:
            meta_path = seg.dir_path / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            rel = meta.get("synced_signal_path")
            if not rel:
                continue
            synced_path = (seg.dir_path / rel).resolve()
            if not synced_path.exists():
                log.warning("Synced signal not found: %s", synced_path)
                continue
            try:
                self._synced_signal_df = read_synced_signal_csv(synced_path)
                log.info("Loaded synced signal from %s", synced_path)
            except Exception as exc:
                log.error("Error loading synced signal: %s", exc)
            break  # only need the first valid one

    def _show_signal_overview(self):
        """Display the full synced signal with segment regions shaded."""
        df = self._synced_signal_df
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
        t0 = df["timestamp_utc"].iloc[0]
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

    def _on_segment_selected(self, row: int):
        if row < 0 or row >= len(self._segments):
            return
        seg = self._segments[row]
        self._current_seg = seg

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
            self._player.set_cameras([])

        # --- Load signal: prefer per-segment, fallback to synced slice ---
        signal_loaded = False
        signal_file = meta.get("signal_file", "signal.csv")
        signal_csv = seg_dir / signal_file if signal_file else seg_dir / "signal.csv"
        if signal_csv.exists():
            try:
                df = read_synced_signal_csv(signal_csv)
                self._plot.set_data(df, video_duration_sec=seg.duration_sec)
                signal_loaded = True
            except Exception as exc:
                log.error("Error loading signal for segment %d: %s", seg.index, exc)

        if not signal_loaded and self._synced_signal_df is not None:
            # Fallback: slice from synced signal
            try:
                df = self._synced_signal_df
                t0 = df["timestamp_utc"].iloc[0]
                start_ts = t0 + pd.Timedelta(seconds=seg.start_sec)
                end_ts = t0 + pd.Timedelta(seconds=seg.end_sec)
                sliced = df[
                    (df["timestamp_utc"] >= start_ts)
                    & (df["timestamp_utc"] <= end_ts)
                ].copy()
                if not sliced.empty:
                    self._plot.set_data(sliced, video_duration_sec=seg.duration_sec)
                    signal_loaded = True
            except Exception as exc:
                log.error("Error slicing synced signal for segment %d: %s", seg.index, exc)

        if not signal_loaded:
            self._plot.clear()

        # --- Info panel ---
        info_lines = [
            f"Segment: {seg.index}",
            f"Label: {seg.label}",
            f"Duration: {seg.duration_sec:.2f}s",
            f"Original time: {seg.start_sec:.2f} – {seg.end_sec:.2f}s",
        ]
        if meta:
            skip = {"index", "label", "start_sec", "end_sec", "duration_sec", "folder",
                    "cameras", "signal_file", "synced_signal_path", "generated_at"}
            for k, v in meta.items():
                if k not in skip:
                    info_lines.append(f"{k}: {v}")
            if meta.get("cameras"):
                info_lines.append(f"Cameras: {len(meta['cameras'])}")
        self._info_label.setText("\n".join(info_lines))

        # Store the loaded signal for mosaic use
        if signal_loaded:
            self._seg_signal_df = self._plot._df
        else:
            self._seg_signal_df = None
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

        self._mosaic_worker = MosaicWorker(
            video_paths=video_paths,
            camera_labels=labels,
            fps=fps,
            total_frames=total_frames,
            video_duration_sec=seg.duration_sec,
            output_path=path,
            signal_df=self._seg_signal_df,
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
