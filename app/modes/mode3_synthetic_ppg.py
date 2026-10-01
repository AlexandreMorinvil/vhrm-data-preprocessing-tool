from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..file_cleanup import cleanup_obsolete_paths
from ..signals import is_aux_signal_column, read_synced_signal_csv, read_synced_signal_csvs
from ..state import (
    ProjectState,
    generate_sidecar,
    video_timeline_duration_sec,
)
from ..synthetic_ppg import SyntheticPpgResult, generate_synthetic_ppg
from ..widgets.frame_preview import MultiCameraPlayer
from ..widgets.layout import configure_main_splitter
from ..widgets.signal_plot import SignalPlot

log = logging.getLogger(__name__)

_SYNTHETIC_PPG_DIRECTORY = "synthetic_ppg"
_SYNTHETIC_ARTIFACT_NAMES = {
    "synthetic_ppg_path": "synthetic_ppg.csv",
    "synthetic_ppg_hr_path": "synthetic_ppg_hr.csv",
    "synthetic_ppg_hrv_path": "synthetic_ppg_hrv.csv",
    "synthetic_rr_path": "synthetic_ppg_rr.csv",
}


def synthetic_ppg_artifact_paths(output_directory: str | Path) -> dict[str, Path]:
    directory = Path(output_directory) / _SYNTHETIC_PPG_DIRECTORY
    return {
        field_name: directory / filename
        for field_name, filename in _SYNTHETIC_ARTIFACT_NAMES.items()
    }


class _GenerationWorker(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        source_path: str,
        settings: dict,
        hr_reference_path: str | None = None,
    ):
        super().__init__()
        self.source_path = source_path
        self.settings = settings
        self.hr_reference_path = hr_reference_path

    def run(self) -> None:
        try:
            ecg = read_synced_signal_csv(self.source_path)
            hr_timestamps = None
            if self.hr_reference_path:
                reference = read_synced_signal_csv(self.hr_reference_path)
                hr_timestamps = pd.DatetimeIndex(reference["timestamp_utc"])
            result = generate_synthetic_ppg(
                ecg,
                hr_timestamps=hr_timestamps,
                **self.settings,
            )
        except Exception as exc:
            log.exception("Synthetic PPG generation failed")
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(result)


def _preferred_hr_reference_path(paths: list[str]) -> str | None:
    available = [path for path in paths if Path(path).is_file()]
    zephyr_paths = [path for path in available if "zephyr" in Path(path).stem.lower()]
    return zephyr_paths[0] if zephyr_paths else (available[0] if available else None)


def _read_timestamped_csv(path: str) -> pd.DataFrame | None:
    if not path or not Path(path).exists():
        return None
    frame = pd.read_csv(path)
    if "timestamp_utc" not in frame.columns:
        return None
    frame["timestamp_utc"] = pd.to_datetime(
        frame["timestamp_utc"], format="mixed", utc=True, errors="coerce"
    )
    return frame.dropna(subset=["timestamp_utc"]).reset_index(drop=True)


def _combine_timestamped(frames: list[pd.DataFrame | None]) -> pd.DataFrame | None:
    usable = [frame.set_index("timestamp_utc") for frame in frames if frame is not None and not frame.empty]
    if not usable:
        return None
    return pd.concat(usable, axis=1).sort_index().reset_index()


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if column == "timestamp_utc" or is_aux_signal_column(str(column)):
            continue
        values = pd.to_numeric(result[column], errors="coerce")
        span = values.max() - values.min()
        result[column] = (values - values.min()) / span if span > 0 else 0.0
    return result


def _comparison_text(
    reference: pd.DataFrame,
    reference_column: str,
    estimate: pd.DataFrame,
    estimate_column: str,
    unit: str,
) -> str:
    left = reference[["timestamp_utc", reference_column]].dropna().sort_values("timestamp_utc")
    right = estimate[["timestamp_utc", estimate_column]].dropna().sort_values("timestamp_utc")
    if left.empty or right.empty:
        return "Not enough overlapping values for comparison."
    aligned = pd.merge_asof(
        left,
        right,
        on="timestamp_utc",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=1),
    ).dropna()
    if aligned.empty:
        return "No values align within one second."
    errors = aligned[estimate_column] - aligned[reference_column]
    correlation = aligned[reference_column].corr(aligned[estimate_column])
    return (
        f"n={len(aligned):,}   MAE={np.mean(np.abs(errors)):.2f} {unit}   "
        f"RMSE={np.sqrt(np.mean(errors**2)):.2f} {unit}   "
        f"Bias={errors.mean():+.2f} {unit}   r={correlation:.3f}"
    )


SYNTHETIC_PPG_VIEWS = (
    "ECG + synthetic PPG (normalized)",
    "Synthetic PPG",
    "Heart rate comparison",
    "HRV comparison",
    "Detected RR intervals",
)


def build_synthetic_ppg_view(
    view_index: int,
    *,
    ppg: pd.DataFrame | None,
    heart_rate: pd.DataFrame | None,
    hrv: pd.DataFrame | None,
    rr: pd.DataFrame | None,
    ecg: pd.DataFrame | None,
    sensor_hr: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, str]:
    """Return the plot frame and comparison metrics for one of ``SYNTHETIC_PPG_VIEWS``."""
    frame: pd.DataFrame | None = None
    metric_text = ""
    if view_index == 0:
        renamed_ppg = ppg.rename(columns={"synthetic_ppg": "Synthetic PPG"}) if ppg is not None else None
        frame = _combine_timestamped([ecg, renamed_ppg])
        frame = _normalise_columns(frame) if frame is not None else None
    elif view_index == 1:
        frame = ppg
    elif view_index == 2:
        generated = (
            heart_rate.rename(columns={"heart_rate_bpm": "Synthetic PPG HR (10 s FFT)"})
            if heart_rate is not None else None
        )
        frame = _combine_timestamped([sensor_hr, generated])
        if sensor_hr is not None and heart_rate is not None:
            reference_columns = [
                column for column in sensor_hr.columns
                if column != "timestamp_utc" and not is_aux_signal_column(str(column))
            ]
            if reference_columns:
                zephyr_columns = [
                    column for column in reference_columns
                    if str(column).lower().startswith("zephyr")
                ]
                reference_column = zephyr_columns[0] if zephyr_columns else reference_columns[0]
                metric_text = _comparison_text(
                    sensor_hr, reference_column, heart_rate, "heart_rate_bpm", "BPM"
                )
    elif view_index == 3:
        references = None
        reference_column = None
        if sensor_hr is not None:
            hrv_columns = [column for column in sensor_hr.columns if str(column).endswith("__HRV")]
            if hrv_columns:
                reference_column = hrv_columns[0]
                references = sensor_hr[["timestamp_utc", reference_column]].copy()
                references[reference_column] = pd.to_numeric(references[reference_column], errors="coerce")
                references = references[references[reference_column].between(1, 1000)]
                references = references.rename(columns={reference_column: "Zephyr HRV"})
                reference_column = "Zephyr HRV"
        generated = None
        if hrv is not None:
            generated = hrv[["timestamp_utc", "sdnn_ms", "rmssd_ms"]].rename(
                columns={"sdnn_ms": "Synthetic PPG SDNN", "rmssd_ms": "Synthetic PPG RMSSD"}
            )
        frame = _combine_timestamped([references, generated])
        if references is not None and hrv is not None and reference_column is not None:
            metric_text = _comparison_text(references, reference_column, hrv, "sdnn_ms", "ms")
    else:
        frame = rr.rename(columns={"rr_interval_ms": "ECG RR interval"}) if rr is not None else None
    return frame, metric_text


class SyntheticPpgWidget(QWidget):
    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state
        self._worker: _GenerationWorker | None = None
        self._ppg: pd.DataFrame | None = None
        self._heart_rate: pd.DataFrame | None = None
        self._hrv: pd.DataFrame | None = None
        self._rr: pd.DataFrame | None = None
        self._ecg: pd.DataFrame | None = None
        self._sensor_hr: pd.DataFrame | None = None

        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(4, 4, 4, 4)

        source_group = QGroupBox("ECG source")
        source_layout = QFormLayout(source_group)
        self._source = QComboBox()
        source_layout.addRow("Synchronized ECG:", self._source)
        self._source_note = QLabel(
            "Peak detection and PPG synthesis use only the ECG samples imported and cropped during Signal sync."
        )
        self._source_note.setWordWrap(True)
        source_layout.addRow(self._source_note)
        self._hr_reference_note = QLabel("")
        self._hr_reference_note.setWordWrap(True)
        source_layout.addRow("HR timestamps:", self._hr_reference_note)
        controls_layout.addWidget(source_group)

        settings_group = QGroupBox("Generation settings")
        settings_layout = QFormLayout(settings_group)
        self._fs = self._spin(20.0, 1000.0, 125.0, 1.0, 1)
        self._low_hz = self._spin(0.1, 100.0, 5.0, 0.5, 1)
        self._high_hz = self._spin(0.2, 200.0, 25.0, 0.5, 1)
        self._peak_distance = self._spin(0.1, 2.0, 0.35, 0.01, 2)
        self._prominence = self._spin(0.01, 10.0, 4.0, 0.25, 2)
        settings_layout.addRow("PPG sampling rate (Hz):", self._fs)
        settings_layout.addRow("ECG band-pass low (Hz):", self._low_hz)
        settings_layout.addRow("ECG band-pass high (Hz):", self._high_hz)
        settings_layout.addRow("Minimum R-peak distance (s):", self._peak_distance)
        settings_layout.addRow("Prominence factor:", self._prominence)
        controls_layout.addWidget(settings_group)

        self._generate_button = QPushButton("Generate synthetic PPG")
        self._generate_button.setStyleSheet("font-weight:bold; padding:8px;")
        self._generate_button.clicked.connect(self._generate)
        controls_layout.addWidget(self._generate_button)
        self._skip_button = QPushButton("Skip synthetic PPG")
        self._skip_button.clicked.connect(self._skip)
        controls_layout.addWidget(self._skip_button)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        controls_layout.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        controls_layout.addWidget(self._status)
        self._metrics = QLabel("")
        self._metrics.setWordWrap(True)
        controls_layout.addWidget(self._metrics)
        controls_layout.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(controls)
        splitter.addWidget(left_scroll)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        self._player = MultiCameraPlayer(face_blur_enabled=state.blur_faces)
        self._player.setMinimumHeight(300)
        right_layout.addWidget(self._player)
        self._view = QComboBox()
        self._view.addItems(SYNTHETIC_PPG_VIEWS)
        self._view.currentIndexChanged.connect(self._show_selected_plot)
        right_layout.addWidget(self._view)
        self._plot = SignalPlot()
        self._plot.setFixedHeight(420)
        right_layout.addWidget(self._plot)
        right.setMinimumHeight(800)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setWidget(right)
        splitter.addWidget(right_scroll)
        configure_main_splitter(splitter, left_scroll, right_scroll)

        self._player.frame_changed.connect(self._on_frame_changed)
        self.refresh_from_state()

    @staticmethod
    def _spin(
        minimum: float,
        maximum: float,
        value: float,
        step: float,
        decimals: int,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def refresh_from_state(self) -> None:
        selected = self._source.currentData()
        self._source.clear()
        for path in self.state.synced_ecg_paths:
            if Path(path).is_file():
                self._source.addItem(Path(path).name, path)
        if selected:
            index = self._source.findData(selected)
            if index >= 0:
                self._source.setCurrentIndex(index)
        reference_path = _preferred_hr_reference_path(self.state.synced_hr_paths)
        self._hr_reference_note.setText(
            Path(reference_path).name
            if reference_path
            else "One-second generated timeline (no synchronized HR reference)"
        )
        self._refresh_player()
        self._load_artifacts()

    def _refresh_player(self) -> None:
        labels = [track.camera_label for track in self.state.tracks]
        self._player.set_cameras(labels)
        paths = [track.final_output_path for track in self.state.tracks]
        valid_paths = [path for path in paths if path and Path(path).exists()]
        if valid_paths:
            self._player.load_videos(valid_paths)

    def _load_artifacts(self) -> None:
        self._ppg = _read_timestamped_csv(self.state.synthetic_ppg_path)
        self._heart_rate = _read_timestamped_csv(self.state.synthetic_ppg_hr_path)
        self._hrv = _read_timestamped_csv(self.state.synthetic_ppg_hrv_path)
        self._rr = _read_timestamped_csv(self.state.synthetic_rr_path)
        try:
            self._ecg = read_synced_signal_csvs([
                path for path in self.state.synced_ecg_paths if Path(path).exists()
            ])
            self._sensor_hr = read_synced_signal_csvs([
                path for path in self.state.synced_hr_paths if Path(path).exists()
            ])
        except Exception as exc:
            log.warning("Could not load synchronized comparison signals: %s", exc)
        self._show_selected_plot()

    def _generate(self) -> None:
        source_path = self._source.currentData()
        if not source_path:
            QMessageBox.information(
                self,
                "Synthetic PPG",
                "Import and synchronize an ECG file in Signal sync first.",
            )
            return

        settings = {
            "fs": self._fs.value(),
            "low_hz": self._low_hz.value(),
            "high_hz": self._high_hz.value(),
            "min_peak_distance_s": self._peak_distance.value(),
            "prominence_factor": self._prominence.value(),
        }
        self._worker = _GenerationWorker(
            source_path,
            settings,
            _preferred_hr_reference_path(self.state.synced_hr_paths),
        )
        self._worker.succeeded.connect(self._generation_succeeded)
        self._worker.failed.connect(self._generation_failed)
        self._generate_button.setEnabled(False)
        self._skip_button.setEnabled(False)
        self._progress.setRange(0, 0)
        self._status.setText("Detecting ECG peaks and generating PPG, HR, and HRV...")
        self._worker.start()

    def _generation_succeeded(self, result: SyntheticPpgResult) -> None:
        try:
            output = Path(self.state.output_directory)
            paths = synthetic_ppg_artifact_paths(output)
            next(iter(paths.values())).parent.mkdir(parents=True, exist_ok=True)
            for frame, field_name in (
                (result.ppg, "synthetic_ppg_path"),
                (result.heart_rate, "synthetic_ppg_hr_path"),
                (result.hrv, "synthetic_ppg_hrv_path"),
                (result.peaks, "synthetic_rr_path"),
            ):
                frame.to_csv(paths[field_name], index=False, date_format="%Y-%m-%dT%H:%M:%S.%fZ")
                setattr(self.state, field_name, str(paths[field_name]))
            legacy_paths = [
                output / filename for filename in _SYNTHETIC_ARTIFACT_NAMES.values()
            ]
            cleanup_obsolete_paths(
                legacy_paths,
                output,
                self.state.archive_removed_files,
            )
            self.state.synthetic_ppg_source_ecg_path = str(self._source.currentData())
            self.state.mode3_complete = True
            generate_sidecar(self.state)
        except Exception as exc:
            self._generation_failed(str(exc))
            return

        self._finish_worker()
        self._load_artifacts()
        self._status.setText(
            f"Generated {len(result.ppg):,} PPG samples from {len(result.source_peak_times):,} ECG peaks; "
            f"{len(result.heart_rate):,} HR and {len(result.hrv):,} HRV estimates saved in "
            f"{_SYNTHETIC_PPG_DIRECTORY}/."
        )

    def _generation_failed(self, message: str) -> None:
        self._finish_worker()
        self._status.setText(f"Generation failed: {message}")
        QMessageBox.critical(self, "Synthetic PPG generation", message)

    def _finish_worker(self) -> None:
        self._generate_button.setEnabled(True)
        self._skip_button.setEnabled(True)
        self._progress.setRange(0, 100)
        self._progress.setValue(100 if self.state.mode3_complete else 0)
        if self._worker is not None:
            self._worker.deleteLater()
        self._worker = None

    def _skip(self) -> None:
        self.state.mode3_complete = True
        self._status.setText("Synthetic PPG generation skipped.")

    def _show_selected_plot(self) -> None:
        frame, metric_text = build_synthetic_ppg_view(
            self._view.currentIndex(),
            ppg=self._ppg,
            heart_rate=self._heart_rate,
            hrv=self._hrv,
            rr=self._rr,
            ecg=self._ecg,
            sensor_hr=self._sensor_hr,
        )

        if frame is None or frame.empty:
            self._plot.clear()
        else:
            self._plot.set_data(
                frame,
                video_duration_sec=video_timeline_duration_sec(self.state.tracks),
            )
        self._metrics.setText(metric_text)

    def _on_frame_changed(self, frame_number: int) -> None:
        if not self.state.tracks:
            return
        fps = self.state.tracks[0].fps or 30.0
        self._plot.set_cursor(frame_number / fps)