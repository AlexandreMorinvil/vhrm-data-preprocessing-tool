"""PPG alignment: estimate and apply the video/signal offset of a project.

An algorithmic rPPG waveform is extracted from the face in one camera clip of a
labelled segment and compared with the synthetic PPG. The offset that best
aligns the two can be adjusted by hand (drag the curves, step between
correlation peaks), checked on the other segments, imported from another
project (for example the same participant's baseline session) and applied to
the project and its exported segments.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..ffmpeg_utils import FFmpegWorker, find_ffmpeg
from ..ppg_alignment import (
    Cancelled,
    FaceTrace,
    FaceTraceSettings,
    LagScan,
    cache_key,
    cache_paths,
    extract_face_trace,
    load_trace,
    prepare_reference,
    save_trace,
    scan_lags,
)
from ..rppg import RPPG_METHODS, dominant_rate_bpm, estimate_pulse
from ..signal_offset import (
    ALIGNMENT_DIRECTORY,
    alignment_directory,
    apply_signal_offset,
    read_offset_file,
    read_segment_meta,
    segment_applied_offset,
    segment_reference_start,
)
from ..state import LabelInterval, ProjectState, load_labelled_segments_manifest, load_sidecar
from ..widgets.alignment_plot import AlignmentPlot
from ..widgets.frame_preview import MultiCameraPlayer
from ..widgets.layout import ModeWorkspace

log = logging.getLogger(__name__)

_ROI_PREFIX = "Face ROI"
_SCAN_STEP_SEC = 0.02
_HR_MISMATCH_BPM = 8.0


class _Segment:
    def __init__(self, index: int, interval: LabelInterval):
        self.index = index
        self.interval = interval
        self.folder = Path(interval.folder)
        self.meta = read_segment_meta(self.folder)
        self.duration = interval.end_sec - interval.start_sec

    @property
    def title(self) -> str:
        return f"[{self.index}] {self.interval.label}  ({self.duration:.0f} s)"

    def cameras(self) -> list[tuple[str, Path]]:
        """(label, clip path) of the segment's camera clips."""
        found = []
        for camera in self.meta.get("cameras", []):
            path = self.folder / str(camera.get("video_file", ""))
            if camera.get("video_file") and path.exists():
                found.append((str(camera.get("label") or path.stem), path))
        if not found:
            found = [(path.stem, path) for path in sorted(self.folder.glob("cam*.mp4"))]
        return found


def _default_camera(labels: list[str]) -> int:
    for index, label in enumerate(labels):
        if "front" in label.lower():
            return index
    return 0


class PpgAlignmentWidget(QWidget):
    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state
        self._segments: list[_Segment] = []
        self._segment: Optional[_Segment] = None
        self._trace: Optional[FaceTrace] = None
        self._pulse: Optional[np.ndarray] = None
        self._reference_t = np.empty(0)
        self._reference = np.empty(0)
        self._reference_margin = 0.0
        self._scan: Optional[LagScan] = None
        self._offset = state.video_signal_offset_sec
        self._offset_note = ""
        self._window: Optional[tuple[float, float]] = None
        self._reference_cache: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
        self._visible_views: set[str] = set()
        self._new_segment = False
        self._estimate_worker: Optional[FFmpegWorker] = None
        self._estimate_failures: list[str] = []
        self._apply_worker: Optional[FFmpegWorker] = None
        self._apply_report = None

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self._workspace = ModeWorkspace("ppg_alignment", self, left_width=380)
        root.addWidget(self._workspace)
        ws = self._workspace

        # -- project data ---------------------------------------------------------
        data_box = QWidget()
        data_lay = QVBoxLayout(data_box)
        data_lay.setContentsMargins(0, 0, 0, 0)
        meta_btn = QPushButton("Load from metadata file …")
        meta_btn.clicked.connect(self._load_from_meta)
        data_lay.addWidget(meta_btn)
        manifest_btn = QPushButton("Load manifest CSV …")
        manifest_btn.clicked.connect(self._load_manifest)
        data_lay.addWidget(manifest_btn)
        self._data_label = QLabel("")
        self._data_label.setWordWrap(True)
        data_lay.addWidget(self._data_label)
        ws.add_section("Project data", data_box)

        # -- segments ---------------------------------------------------------------
        seg_box = QWidget()
        seg_lay = QVBoxLayout(seg_box)
        seg_lay.setContentsMargins(0, 0, 0, 0)
        self._segment_list = QListWidget()
        self._segment_list.setMinimumHeight(160)
        self._segment_list.setToolTip("✓ = rPPG already estimated for the selected camera")
        self._segment_list.currentRowChanged.connect(self._on_segment_selected)
        seg_lay.addWidget(self._segment_list)
        nav = QHBoxLayout()
        prev_btn = QPushButton("◀ Previous")
        prev_btn.clicked.connect(lambda: self._step_segment(-1))
        next_btn = QPushButton("Next ▶")
        next_btn.clicked.connect(lambda: self._step_segment(1))
        nav.addWidget(prev_btn)
        nav.addWidget(next_btn)
        seg_lay.addLayout(nav)
        ws.add_section("Segments", seg_box)

        # -- estimation -------------------------------------------------------------
        est_box = QWidget()
        est_lay = QVBoxLayout(est_box)
        est_lay.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self._camera_combo = QComboBox()
        self._camera_combo.setToolTip("Camera whose face is used for the rPPG estimate")
        self._camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        form.addRow("Camera:", self._camera_combo)
        self._method_combo = QComboBox()
        self._method_combo.addItems(RPPG_METHODS)
        self._method_combo.setToolTip(
            "POS and CHROM are robust to lighting changes; Green is the simplest.\n"
            "All three give a waveform with the polarity of a contact PPG."
        )
        self._method_combo.currentIndexChanged.connect(self._recompute_pulse)
        form.addRow("Method:", self._method_combo)
        self._skin_cb = QCheckBox("Skin-coloured pixels only")
        self._skin_cb.setChecked(True)
        self._skin_cb.setToolTip(
            "Average only skin-coloured pixels of the face box (eyes, hair and background are\n"
            "dimmed in the ROI video). Changing this requires a new estimate."
        )
        self._skin_cb.toggled.connect(self._on_camera_changed)
        form.addRow("", self._skin_cb)
        est_lay.addLayout(form)
        est_buttons = QHBoxLayout()
        self._estimate_btn = QPushButton("Estimate this segment")
        self._estimate_btn.setStyleSheet("font-weight:bold; padding:5px;")
        self._estimate_btn.clicked.connect(lambda: self._start_estimate(all_segments=False))
        est_buttons.addWidget(self._estimate_btn)
        self._estimate_all_btn = QPushButton("Estimate all")
        self._estimate_all_btn.setToolTip("Estimate every segment not yet estimated for this camera")
        self._estimate_all_btn.clicked.connect(lambda: self._start_estimate(all_segments=True))
        est_buttons.addWidget(self._estimate_all_btn)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel_estimate)
        est_buttons.addWidget(self._cancel_btn)
        est_lay.addLayout(est_buttons)
        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        est_lay.addWidget(self._progress)
        self._estimate_label = QLabel("")
        self._estimate_label.setWordWrap(True)
        est_lay.addWidget(self._estimate_label)
        ws.add_section("rPPG estimate", est_box)

        # -- search -----------------------------------------------------------------
        search_box = QWidget()
        search_lay = QVBoxLayout(search_box)
        search_lay.setContentsMargins(0, 0, 0, 0)
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Offsets from"))
        self._lag_min = QDoubleSpinBox()
        self._lag_max = QDoubleSpinBox()
        for spin, value in ((self._lag_min, -10.0), (self._lag_max, 10.0)):
            spin.setRange(-60.0, 60.0)
            spin.setDecimals(1)
            spin.setSingleStep(0.5)
            spin.setSuffix(" s")
            spin.setValue(value)
            spin.valueChanged.connect(self._schedule_scan)
        range_row.addWidget(self._lag_min)
        range_row.addWidget(QLabel("to"))
        range_row.addWidget(self._lag_max)
        search_lay.addLayout(range_row)
        window_row = QGridLayout()
        whole_btn = QPushButton("Window: whole segment")
        whole_btn.clicked.connect(self._window_whole)
        window_row.addWidget(whole_btn, 0, 0)
        view_btn = QPushButton("Window: visible range")
        view_btn.setToolTip("Use the part of the segment shown in the graph")
        view_btn.clicked.connect(lambda: self._plot.window_to_view())
        window_row.addWidget(view_btn, 0, 1)
        zoom_btn = QPushButton("Zoom to window")
        zoom_btn.clicked.connect(lambda: self._plot.zoom_to_window())
        window_row.addWidget(zoom_btn, 1, 0)
        full_btn = QPushButton("Show whole segment")
        full_btn.clicked.connect(lambda: self._plot.show_full_range())
        window_row.addWidget(full_btn, 1, 1)
        search_lay.addLayout(window_row)
        hint = QLabel(
            "The correlation uses the shaded window of the graph; drag its edges to "
            "restrict it to a calm part of the segment."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8a8a8a;")
        search_lay.addWidget(hint)
        ws.add_section("Search", search_box)

        # -- offset -----------------------------------------------------------------
        offset_box = QWidget()
        offset_lay = QVBoxLayout(offset_box)
        offset_lay.setContentsMargins(0, 0, 0, 0)
        value_row = QHBoxLayout()
        value_row.addWidget(QLabel("Offset:"))
        self._offset_spin = QDoubleSpinBox()
        self._offset_spin.setRange(-600.0, 600.0)
        self._offset_spin.setDecimals(3)
        self._offset_spin.setSingleStep(0.01)
        self._offset_spin.setSuffix(" s")
        self._offset_spin.setToolTip(
            "Shift applied to the signals. Positive: signals later on the video timeline.\n"
            "Type a value, use the arrows, or drag on the graph."
        )
        self._offset_spin.setValue(self._offset)
        self._offset_spin.valueChanged.connect(self._on_offset_spin)
        value_row.addWidget(self._offset_spin, 1)
        offset_lay.addLayout(value_row)
        peak_row = QHBoxLayout()
        prev_peak = QPushButton("◀ Previous peak")
        prev_peak.setToolTip("Move to the next correlation peak at a smaller offset")
        prev_peak.clicked.connect(lambda: self._step_peak(-1))
        nearest_peak = QPushButton("Nearest")
        nearest_peak.setToolTip("Snap to the correlation peak closest to the current offset")
        nearest_peak.clicked.connect(self._nearest_peak)
        next_peak = QPushButton("Next peak ▶")
        next_peak.setToolTip("Move to the next correlation peak at a larger offset")
        next_peak.clicked.connect(lambda: self._step_peak(1))
        peak_row.addWidget(prev_peak)
        peak_row.addWidget(nearest_peak)
        peak_row.addWidget(next_peak)
        offset_lay.addLayout(peak_row)
        best_row = QHBoxLayout()
        best_btn = QPushButton("Best in range")
        best_btn.setToolTip("Offset with the highest correlation in the search range")
        best_btn.clicked.connect(self._best_offset)
        best_row.addWidget(best_btn)
        applied_btn = QPushButton("Back to applied")
        applied_btn.clicked.connect(lambda: self._set_offset(self.state.video_signal_offset_sec, "applied"))
        best_row.addWidget(applied_btn)
        offset_lay.addLayout(best_row)
        self._applied_label = QLabel("")
        self._applied_label.setWordWrap(True)
        offset_lay.addWidget(self._applied_label)
        io_row = QHBoxLayout()
        import_btn = QPushButton("Import offset …")
        import_btn.setToolTip(
            "Read the offset applied in another project (its ppg_alignment/alignment.json,\n"
            "metadata sidecar, a segment meta.json or the .vrt project) to inspect it here."
        )
        import_btn.clicked.connect(self._import_offset)
        io_row.addWidget(import_btn)
        self._apply_btn = QPushButton("Apply to project …")
        self._apply_btn.setStyleSheet("font-weight:bold; padding:5px;")
        self._apply_btn.setToolTip(
            "Store this offset in the project and rewrite the signal files and timing of every\n"
            "exported segment (videos are not modified)."
        )
        self._apply_btn.clicked.connect(self._apply_offset)
        io_row.addWidget(self._apply_btn)
        offset_lay.addLayout(io_row)
        self._remove_btn = QPushButton("Remove offset …")
        self._remove_btn.setToolTip(
            "Return to the original camera timing (applies 0 s). To change the offset, simply\n"
            "apply the new value: applying replaces the previous offset."
        )
        self._remove_btn.clicked.connect(self._remove_offset)
        offset_lay.addWidget(self._remove_btn)
        self._apply_status = QLabel("")
        self._apply_status.setWordWrap(True)
        offset_lay.addWidget(self._apply_status)
        ws.add_section("Offset", offset_box)
        ws.finish_left()

        # -- work area -------------------------------------------------------------
        self._player = MultiCameraPlayer(face_blur_enabled=state.blur_faces)
        self._views_widget = QWidget()
        self._views_layout = QHBoxLayout(self._views_widget)
        self._views_layout.setContentsMargins(0, 0, 0, 0)
        self._player.add_transport_widget(self._views_widget)
        ws.add_work(self._player, 420)

        plot_area = QWidget()
        plot_lay = QVBoxLayout(plot_area)
        plot_lay.setContentsMargins(0, 0, 0, 0)
        self._metrics_label = QLabel("Select a segment and estimate its rPPG.")
        self._metrics_label.setWordWrap(True)
        self._metrics_label.setStyleSheet("font-family: Consolas, 'Courier New', monospace;")
        plot_lay.addWidget(self._metrics_label)
        self._plot = AlignmentPlot()
        plot_lay.addWidget(self._plot, 1)
        ws.add_work(plot_area, 420)

        self._scan_timer = QTimer(self)
        self._scan_timer.setSingleShot(True)
        self._scan_timer.setInterval(150)
        self._scan_timer.timeout.connect(self._run_scan)

        self._plot.offset_changed.connect(self._on_plot_offset)
        self._plot.window_changed.connect(self._on_window_changed)
        self._plot.seek_requested.connect(self._player.seek_seconds)
        self._player.frame_changed.connect(self._on_frame_changed)
        self._player.install_shortcuts(self)

        self.refresh_from_state()

    # ------------------------------------------------------------------ state
    def refresh_from_state(self) -> None:
        previous = self._segment.folder if self._segment is not None else None
        self._segments = [
            _Segment(index, interval)
            for index, interval in enumerate(self.state.intervals)
            if interval.folder and Path(interval.folder).exists()
        ]
        self._segment_list.blockSignals(True)
        self._segment_list.clear()
        for segment in self._segments:
            self._segment_list.addItem(segment.title)
        self._segment_list.blockSignals(False)
        self._update_segment_marks()
        if self._segments:
            self._data_label.setText(f"{len(self._segments)} labelled segment(s).")
        else:
            self._data_label.setText(
                "No exported labelled segments. Export them in Labelling, or load a "
                "metadata file and the labelled_segments/manifest.csv."
            )
        self._update_applied_label()
        row = next((i for i, s in enumerate(self._segments) if s.folder == previous), 0 if self._segments else -1)
        self._segment_list.setCurrentRow(row)
        if row < 0:
            self._on_segment_selected(-1)

    def refresh_camera_names(self) -> None:
        self.refresh_from_state()

    def _load_from_meta(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select metadata sidecar", self.state.output_directory or "",
            "Metadata sidecar (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            load_sidecar(path, self.state)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load metadata:\n{exc}")
            return
        self._offset = self.state.video_signal_offset_sec
        self._offset_spin.blockSignals(True)
        self._offset_spin.setValue(self._offset)
        self._offset_spin.blockSignals(False)
        self.refresh_from_state()

    def _load_manifest(self) -> None:
        default = self.state.segments_manifest_path or str(
            Path(self.state.output_directory) / "labelled_segments" / "manifest.csv"
        )
        path, _ = QFileDialog.getOpenFileName(self, "Select manifest.csv", default, "CSV (*.csv);;All files (*)")
        if not path:
            return
        try:
            load_labelled_segments_manifest(path, self.state)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            QMessageBox.critical(self, "Invalid manifest", f"Failed to import labelled segments:\n{exc}")
            return
        self.refresh_from_state()

    def _cache_dir(self) -> Optional[Path]:
        directory = alignment_directory(self.state)
        if directory is None and self._segments:
            directory = self._segments[0].folder.parent.parent / ALIGNMENT_DIRECTORY
        return directory / "cache" if directory is not None else None

    def _trace_settings(self) -> FaceTraceSettings:
        return FaceTraceSettings(skin_mask=self._skin_cb.isChecked())

    # ------------------------------------------------------------------ segments
    def _step_segment(self, step: int) -> None:
        if self._segments:
            row = max(0, min(len(self._segments) - 1, self._segment_list.currentRow() + step))
            self._segment_list.setCurrentRow(row)

    def _camera_label(self) -> str:
        return self._camera_combo.currentText()

    def _key(self, segment: _Segment) -> str:
        return cache_key(str(segment.folder), f"{self._camera_label()}_{'skin' if self._skin_cb.isChecked() else 'box'}")

    def _update_segment_marks(self) -> None:
        cache_dir = self._cache_dir()
        for row, segment in enumerate(self._segments):
            item = self._segment_list.item(row)
            if item is None:
                continue
            done = cache_dir is not None and cache_paths(cache_dir, self._key(segment))[0].exists()
            item.setText(segment.title + ("  ✓" if done else ""))

    def _on_segment_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._segments):
            self._segment = None
            self._trace = None
            self._pulse = None
            self._plot.clear()
            self._player.set_cameras([])
            self._metrics_label.setText("Select a segment and estimate its rPPG.")
            return
        self._segment = self._segments[row]
        self._segment.meta = read_segment_meta(self._segment.folder)
        labels = [label for label, _path in self._segment.cameras()]
        current = self._camera_combo.currentText()
        self._camera_combo.blockSignals(True)
        self._camera_combo.clear()
        self._camera_combo.addItems(labels)
        self._camera_combo.setCurrentIndex(labels.index(current) if current in labels else _default_camera(labels))
        self._camera_combo.blockSignals(False)
        self._update_segment_marks()  # the marks depend on the camera, known only now
        if not self._visible_views:
            self._visible_views = {self._camera_label(), _ROI_PREFIX}
        self._window = None
        self._new_segment = True
        self._plot.set_segment_duration(self._segment.duration)
        self._plot.set_window(0.0, self._segment.duration)
        self._load_reference()
        self._load_trace()
        self._update_applied_label()

    def _on_camera_changed(self, *_args) -> None:
        self._update_segment_marks()
        if self._segment is not None:
            self._load_trace()

    # ------------------------------------------------------------------ video views
    def _rebuild_views(self, include_roi: bool = True) -> None:
        for index in reversed(range(self._views_layout.count())):
            widget = self._views_layout.itemAt(index).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        position = self._player.current_sec if self._player.total_frames and not self._new_segment else 0.0
        self._new_segment = False
        if self._segment is None:
            self._player.set_cameras([])
            return
        self._views_layout.addWidget(QLabel("Show:"))
        views: list[tuple[str, Optional[Path]]] = list(self._segment.cameras())
        roi_label = f"{_ROI_PREFIX} ({self._camera_label()})"
        roi_path = None
        cache_dir = self._cache_dir()
        if include_roi and self._trace is not None and cache_dir is not None:
            candidate = cache_paths(cache_dir, self._key(self._segment))[1]
            roi_path = candidate if candidate.exists() else None
        views.append((roi_label, roi_path))
        shown: list[tuple[str, Path]] = []
        for label, path in views:
            key = _ROI_PREFIX if label.startswith(_ROI_PREFIX) else label
            box = QCheckBox(label)
            box.setChecked(key in self._visible_views)
            box.setEnabled(path is not None)
            if path is None:
                box.setToolTip("Estimate the rPPG of this segment to see the face region used")
            box.toggled.connect(lambda on, k=key: self._toggle_view(k, on))
            self._views_layout.addWidget(box)
            if path is not None and key in self._visible_views:
                shown.append((label, path))
        self._player.set_cameras([label for label, _ in shown])
        if shown:
            self._player.load_videos([str(path) for _, path in shown])
            start = self._segment.meta.get("segment_start_datetime_utc")
            self._player.set_clock_anchor(pd.Timestamp(start).to_pydatetime() if start else None)
            self._player.seek_seconds(position)

    def _toggle_view(self, key: str, on: bool) -> None:
        if on:
            self._visible_views.add(key)
        else:
            self._visible_views.discard(key)
        QTimer.singleShot(0, self._rebuild_views)

    def _on_frame_changed(self, frame: int) -> None:
        fps = self._player.get_fps()
        self._plot.set_cursor(frame / fps if fps > 0 else 0.0)

    # ------------------------------------------------------------------ signals
    def _reference_source(self) -> Optional[Path]:
        if self.state.synthetic_ppg_path and Path(self.state.synthetic_ppg_path).exists():
            return Path(self.state.synthetic_ppg_path)
        if self._segment is not None:
            relative = self._segment.meta.get("synthetic_ppg_path") or (self._segment.meta.get("ppg_files") or [None])[0]
            if relative and (self._segment.folder / relative).exists():
                return self._segment.folder / relative
        return None

    def _read_reference(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        mtime = path.stat().st_mtime
        cached = self._reference_cache.get(str(path))
        if cached is not None and cached[0] == mtime:
            return cached[1], cached[2]
        frame = pd.read_csv(path)
        column = "synthetic_ppg" if "synthetic_ppg" in frame.columns else next(
            c for c in frame.columns if c != "timestamp_utc"
        )
        stamps = pd.to_datetime(frame["timestamp_utc"], format="mixed", utc=True, errors="coerce")
        epoch = (stamps - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy(np.float64)
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
        self._reference_cache[str(path)] = (mtime, epoch, values)
        return epoch, values

    def _load_reference(self) -> None:
        self._reference_t = np.empty(0)
        self._reference = np.empty(0)
        self._reference_margin = float("inf")  # nothing more to load unless a source is found
        segment = self._segment
        source = self._reference_source()
        start = segment_reference_start(segment.meta, segment.interval, self.state) if segment else None
        if segment is None or source is None or start is None:
            self._plot.set_reference(self._reference_t, self._reference)
            return
        try:
            epoch, values = self._read_reference(source)
        except Exception as exc:
            log.exception("Cannot read synthetic PPG")
            self._metrics_label.setText(f"Cannot read the synthetic PPG {source.name}: {exc}")
            return
        start_epoch = start.timestamp()
        margin = self._search_margin()
        self._reference_margin = margin
        keep = (epoch >= start_epoch - margin) & (epoch <= start_epoch + segment.duration + margin)
        self._reference_t, self._reference = prepare_reference(epoch[keep] - start_epoch, values[keep])
        self._plot.set_reference(self._reference_t, self._reference)

    def _load_trace(self) -> None:
        self._trace = None
        self._pulse = None
        segment = self._segment
        cameras = dict(segment.cameras()) if segment else {}
        video = cameras.get(self._camera_label())
        cache_dir = self._cache_dir()
        if segment is not None and video is not None and cache_dir is not None:
            self._trace = load_trace(cache_dir, self._key(segment), str(video), self._trace_settings())
        self._rebuild_views()
        self._recompute_pulse()

    def _recompute_pulse(self, *_args) -> None:
        if self._trace is None:
            self._pulse = None
            self._plot.set_pulse(np.empty(0), np.empty(0))
            self._estimate_label.setText(
                "Not estimated yet for this camera." if self._segment is not None else ""
            )
            self._scan = None
            self._plot.set_scan(None)
            self._update_metrics()
            return
        self._pulse = estimate_pulse(self._trace.rgb, self._trace.fs, self._method_combo.currentText())
        self._plot.set_pulse(self._trace.times, self._pulse)
        coverage = self._trace.face_coverage
        text = f"Face found in {coverage:.0%} of the frames."
        if coverage < 0.9:
            text += " ⚠ The face was lost often: prefer another segment or camera."
        self._estimate_label.setText(text)
        self._schedule_scan()

    def _schedule_scan(self, *_args) -> None:
        self._scan_timer.start()

    def _on_window_changed(self, start: float, end: float) -> None:
        self._window = (start, end)
        self._schedule_scan()

    def _window_whole(self) -> None:
        if self._segment is not None:
            self._plot.set_window(0.0, self._segment.duration)
            self._on_window_changed(0.0, self._segment.duration)

    def _search_margin(self) -> float:
        return max(abs(self._lag_min.value()), abs(self._lag_max.value())) + 2.0

    def _run_scan(self) -> None:
        if self._segment is not None and self._search_margin() > self._reference_margin:
            self._load_reference()  # the search range grew beyond the loaded signal
        if self._pulse is None or not len(self._reference_t):
            self._scan = None
            self._plot.set_scan(None)
            self._update_metrics()
            return
        self._scan = scan_lags(
            self._trace.times, self._pulse, self._reference_t, self._reference,
            self._lag_min.value(), self._lag_max.value(), step=_SCAN_STEP_SEC, window=self._window,
        )
        self._plot.set_scan(self._scan)
        self._update_metrics()

    def _window_bounds(self) -> tuple[float, float]:
        if self._window is not None:
            return self._window
        return 0.0, self._segment.duration if self._segment else 0.0

    def _update_metrics(self) -> None:
        if self._segment is None:
            return
        if not len(self._reference_t):
            self._metrics_label.setText(
                "No synthetic PPG for this segment: generate it in Synthetic PPG (and load the "
                "project metadata), or export the segments with it."
            )
            return
        if self._pulse is None:
            self._metrics_label.setText(
                f"Offset {self._offset:+.3f} s. Estimate the rPPG of this segment to compare it with the PPG."
            )
            return
        start, end = self._window_bounds()
        parts = [f"Offset {self._offset:+.3f} s"]
        if self._scan is not None and self._scan.valid:
            here = self._scan.at(self._offset)
            best = self._scan.best()
            parts.append(f"r = {here:.2f}" if np.isfinite(here) else "r = n/a (outside the search range)")
            parts.append(
                f"best {best:+.3f} s (r = {np.nanmax(self._scan.correlation):.2f}, "
                f"{self._scan.peak_ratio():.2f}× next peak)"
            )
        times = self._trace.times
        in_window = (times >= start) & (times <= end)
        video_hr = dominant_rate_bpm(self._pulse[in_window], self._trace.fs)
        shifted = self._reference_t + self._offset
        ref_window = (shifted >= start) & (shifted <= end)
        ref_rate = 1.0 / float(np.median(np.diff(self._reference_t))) if len(self._reference_t) > 1 else 0.0
        ppg_hr = dominant_rate_bpm(self._reference[ref_window], ref_rate) if ref_rate else float("nan")
        parts.append(f"HR video {video_hr:.1f} / PPG {ppg_hr:.1f} bpm")
        parts.append(f"window {start:.1f}–{end:.1f} s")
        text = "  |  ".join(parts)
        if np.isfinite(video_hr) and np.isfinite(ppg_hr) and abs(video_hr - ppg_hr) > _HR_MISMATCH_BPM:
            text += ("\n⚠ The heart rates differ: the rPPG estimate is unreliable in this window "
                     "(motion, lighting or face lost). Choose a calmer window or segment.")
        self._metrics_label.setText(text)

    # ------------------------------------------------------------------ offset
    def _set_offset(self, offset: float, note: str = "") -> None:
        self._offset = round(float(offset), 4)
        if note:
            self._offset_note = note
        self._offset_spin.blockSignals(True)
        self._offset_spin.setValue(self._offset)
        self._offset_spin.blockSignals(False)
        self._plot.set_offset(self._offset)
        self._update_metrics()

    def _on_offset_spin(self, value: float) -> None:
        self._set_offset(value, "manual")

    def _on_plot_offset(self, offset: float, finished: bool) -> None:
        self._offset = round(offset, 4)
        self._offset_note = "manual"
        self._offset_spin.blockSignals(True)
        self._offset_spin.setValue(self._offset)
        self._offset_spin.blockSignals(False)
        if finished:
            self._update_metrics()

    def _require_scan(self) -> bool:
        if self._scan is None or not self._scan.valid:
            QMessageBox.information(self, "PPG alignment", "Estimate the rPPG of this segment first.")
            return False
        return True

    def _step_peak(self, direction: int) -> None:
        if not self._require_scan():
            return
        target = self._scan.step(self._offset, direction)
        if target is None:
            self._metrics_label.setText(self._metrics_label.text() + "\nNo further peak in the search range.")
            return
        self._set_offset(target, "peak")

    def _nearest_peak(self) -> None:
        if not self._require_scan() or not len(self._scan.peaks):
            return
        peaks = self._scan.peak_lags
        self._set_offset(float(peaks[np.argmin(np.abs(peaks - self._offset))]), "peak")

    def _best_offset(self) -> None:
        if self._require_scan():
            self._set_offset(self._scan.best(), "best")

    def _update_applied_label(self) -> None:
        applied = self.state.video_signal_offset_sec
        text = f"Applied to the project: {applied:+.3f} s"
        if self.state.video_signal_offset_note:
            text += f" — {self.state.video_signal_offset_note}"
        if self._segment is not None:
            segment_offset = segment_applied_offset(self._segment.meta, self.state)
            if abs(segment_offset - applied) > 1e-4:
                text += (f"\n⚠ This segment's files were cut with {segment_offset:+.3f} s; "
                         "apply the offset to update them.")
        self._applied_label.setText(text)
        self._plot.set_applied_offset(applied)

    def _import_offset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import offset", str(Path(self.state.output_directory).parent) if self.state.output_directory else "",
            "Offset sources (alignment.json *_meta.json meta.json *.json *.vrt);;All files (*)",
        )
        if not path:
            return
        try:
            offset, note = read_offset_file(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Import offset", str(exc))
            return
        self._set_offset(offset, f"imported from {Path(path).parent.name}/{Path(path).name}"
                         + (f" ({note})" if note else ""))
        self._apply_status.setText(
            f"Imported {offset:+.3f} s. Check the alignment on a few segments (Nearest peak refines it), "
            "then Apply to project."
        )

    def _estimate_record(self) -> dict:
        record = {"note_source": self._offset_note}
        if self._segment is not None and self._scan is not None and self._scan.valid:
            start, end = self._window_bounds()
            record.update({
                "segment": self._segment.folder.name,
                "camera": self._camera_label(),
                "method": self._method_combo.currentText(),
                "window_sec": [round(start, 2), round(end, 2)],
                "correlation": round(self._scan.at(self._offset), 4),
                "best_offset_sec": self._scan.best(),
                "best_correlation": round(float(np.nanmax(self._scan.correlation)), 4),
            })
        return record

    def _apply_offset(self) -> None:
        if self._apply_worker is not None:
            return
        segments = [s for s in self._segments if (s.folder / "meta.json").exists()]
        default_note = self._offset_note
        if self._segment is not None and self._offset_note in ("best", "peak", "manual"):
            default_note = (f"{self._offset_note} on {self._segment.folder.name} "
                            f"({self._camera_label()}, {self._method_combo.currentText()})")
        note, ok = QInputDialog.getText(
            self, "Apply offset",
            f"Apply {self._offset:+.3f} s (currently {self.state.video_signal_offset_sec:+.3f} s).\n\n"
            f"This rewrites the HR, ECG and synthetic PPG files and the timing in meta.json of "
            f"{len(segments)} exported segment(s), updates the metadata sidecar and records the "
            f"change in {ALIGNMENT_DIRECTORY}/alignment.json. Videos are not modified.\n\n"
            "Note stored with the offset:",
            QLineEdit.EchoMode.Normal, default_note,
        )
        if not ok:
            return
        self._start_apply(self._offset, note, self._estimate_record())

    def _remove_offset(self) -> None:
        if self._apply_worker is not None:
            return
        current = self.state.video_signal_offset_sec
        if abs(current) <= 1e-4:
            QMessageBox.information(self, "Remove offset", "No offset is applied to this project.")
            return
        segments = sum(1 for s in self._segments if (s.folder / "meta.json").exists())
        reply = QMessageBox.question(
            self, "Remove offset",
            f"Remove the applied offset ({current:+.3f} s) and return to the camera timing?\n\n"
            f"This rewrites the HR, ECG and synthetic PPG files and the timing in meta.json of "
            f"{segments} exported segment(s) and updates the metadata sidecar, exactly as Apply "
            "does with 0 s. Videos are not modified.",
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._start_apply(0.0, "offset removed (camera timing)", {})

    def _start_apply(self, offset: float, note: str, estimate: dict) -> None:
        self._player.stop()
        worker = FFmpegWorker(self._apply_job, offset=offset, note=note, estimate=estimate)
        worker.progress.connect(lambda value, message: (self._progress.setValue(value),
                                                        self._apply_status.setText(message)))
        worker.finished.connect(self._on_apply_finished)
        self._apply_worker = worker
        self._apply_btn.setEnabled(False)
        self._remove_btn.setEnabled(False)
        self._apply_status.setText("Applying…")
        worker.start()

    def _apply_job(self, *, offset, note, estimate, worker):
        self._apply_report = apply_signal_offset(
            self.state, offset, note, estimate,
            progress=lambda done, total: worker.progress.emit(
                int(100 * done / max(1, total)), f"Updating segment {done}/{total}…"),
        )

    def _on_apply_finished(self, ok: bool, message: str) -> None:
        self._apply_worker = None
        self._apply_btn.setEnabled(True)
        self._remove_btn.setEnabled(True)
        report = self._apply_report
        self._apply_report = None
        if not ok or report is None:
            QMessageBox.critical(self, "Apply offset", f"Applying the offset failed:\n{message}")
            self._apply_status.setText(f"Failed: {message}")
            return
        lines = [f"Applied {report.offset_sec:+.3f} s to {report.segments_updated} segment(s)."]
        if report.resliced:
            lines.append("Re-sliced: " + ", ".join(report.resliced) + ".")
        lines.append("Metadata sidecar updated." if report.sidecar_updated else "No metadata sidecar to update.")
        lines.append("Save the project (Ctrl+S) to keep the offset in the .vrt file.")
        lines.extend(f"⚠ {warning}" for warning in report.warnings)
        self._apply_status.setText("\n".join(lines))
        self._progress.setValue(100)
        for segment in self._segments:
            segment.meta = read_segment_meta(segment.folder)
        self._update_applied_label()
        if abs(self._offset - report.offset_sec) > 1e-4:  # after removing the offset, show what is applied now
            self._set_offset(report.offset_sec, "applied")

    # ------------------------------------------------------------------ estimation
    def _start_estimate(self, all_segments: bool) -> None:
        if self._estimate_worker is not None:
            return
        ffmpeg = self.state.ffmpeg_path or find_ffmpeg()
        cache_dir = self._cache_dir()
        if not ffmpeg:
            QMessageBox.critical(self, "PPG alignment", "FFmpeg not found.")
            return
        if cache_dir is None or self._segment is None:
            QMessageBox.information(self, "PPG alignment", "Select a labelled segment first.")
            return
        camera = self._camera_label()
        candidates = self._segments if all_segments else [self._segment]
        jobs = []
        for segment in candidates:
            video = dict(segment.cameras()).get(camera)
            if video is None:
                continue
            key = self._key(segment)
            if all_segments and load_trace(cache_dir, key, str(video), self._trace_settings()) is not None:
                continue
            trace_path, roi_path = cache_paths(cache_dir, key)
            jobs.append((segment.folder.name, str(video), key, str(roi_path)))
        if not jobs:
            self._estimate_label.setText(f"Every segment is already estimated for {camera}.")
            return
        # Release the ROI video of the shown segment: it may be rewritten.
        self._rebuild_views(include_roi=False)
        self._estimate_failures = []
        worker = FFmpegWorker(self._estimate_job, jobs=jobs, settings=self._trace_settings(),
                              ffmpeg=ffmpeg, cache_dir=cache_dir)
        worker.progress.connect(lambda value, message: (self._progress.setValue(value),
                                                        self._estimate_label.setText(message)))
        worker.finished.connect(self._on_estimate_finished)
        self._estimate_worker = worker
        self._estimate_btn.setEnabled(False)
        self._estimate_all_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress.setValue(0)
        worker.start()

    def _estimate_job(self, *, jobs, settings, ffmpeg, cache_dir, worker):
        for number, (name, video, key, roi_path) in enumerate(jobs):
            if worker.is_cancelled:
                return

            def progress(done, total, number=number, name=name):
                worker.progress.emit(
                    int(100 * (number + done / max(1, total)) / len(jobs)),
                    f"{name} ({number + 1}/{len(jobs)}): frame {done}/{total}",
                )

            try:
                trace = extract_face_trace(video, settings, ffmpeg=ffmpeg, roi_video_path=roi_path,
                                           progress=progress, cancel=lambda: worker.is_cancelled)
                save_trace(cache_dir, key, trace)
            except Cancelled:
                return
            except Exception as exc:
                log.exception("rPPG estimation failed for %s", video)
                self._estimate_failures.append(f"{name}: {exc}")

    def _cancel_estimate(self) -> None:
        if self._estimate_worker is not None:
            self._estimate_worker.cancel()

    def _on_estimate_finished(self, ok: bool, message: str) -> None:
        self._estimate_worker = None
        self._estimate_btn.setEnabled(True)
        self._estimate_all_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._update_segment_marks()
        if self._segment is not None:
            self._load_trace()
        if not ok:
            self._estimate_label.setText(f"Stopped: {message}")
        elif self._estimate_failures:
            self._progress.setValue(100)
            QMessageBox.warning(self, "rPPG estimate", "Some segments failed:\n" + "\n".join(self._estimate_failures))
        else:
            self._progress.setValue(100)
