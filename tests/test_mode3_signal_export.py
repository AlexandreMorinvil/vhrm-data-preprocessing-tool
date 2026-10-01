import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from app.modes.mode3_labelling import Mode3Widget
from app.state import LabelInterval, ProjectState


class Mode3SignalExportTests(unittest.TestCase):
    def test_exports_all_synthetic_artifacts_to_recreated_segment_subfolder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_directory = root / "source" / "synthetic_ppg"
            source_directory.mkdir(parents=True)
            timestamps = [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:01Z",
                "2026-01-01T00:00:02Z",
                "2026-01-01T00:00:03Z",
            ]
            artifacts = {
                "synthetic_ppg_path": ("synthetic_ppg.csv", "synthetic_ppg"),
                "synthetic_ppg_hr_path": ("synthetic_ppg_hr.csv", "heart_rate_bpm"),
                "synthetic_ppg_hrv_path": ("synthetic_ppg_hrv.csv", "sdnn_ms"),
                "synthetic_rr_path": ("synthetic_ppg_rr.csv", "rr_interval_ms"),
            }
            state = ProjectState()
            for field_name, (file_name, value_column) in artifacts.items():
                path = source_directory / file_name
                pd.DataFrame({
                    "timestamp_utc": timestamps,
                    value_column: [10, 11, 12, 13],
                }).to_csv(path, index=False)
                setattr(state, field_name, str(path))

            segment_directory = root / "segment"
            stale_directory = segment_directory / "synthetic_ppg"
            stale_directory.mkdir(parents=True)
            (stale_directory / "stale.csv").touch()

            exported = Mode3Widget._export_synthetic_artifact_slices(
                SimpleNamespace(state=state),
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                LabelInterval("Test", 1.0, 2.0),
                segment_directory,
            )

            self.assertEqual(
                exported,
                {
                    field_name: f"synthetic_ppg/{file_name}"
                    for field_name, (file_name, _value_column) in artifacts.items()
                },
            )
            self.assertFalse((stale_directory / "stale.csv").exists())
            for relative_path in exported.values():
                frame = pd.read_csv(segment_directory / relative_path)
                self.assertEqual(len(frame), 2)
                self.assertEqual(
                    frame["timestamp_utc"].tolist(), timestamps[1:3]
                )

    def test_all_signals_export_updates_existing_segment_without_touching_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            segment_directory = root / "labelled_segments" / "0000_Test"
            segment_directory.mkdir(parents=True)
            video_path = segment_directory / "cam1_Front.mp4"
            video_path.write_bytes(b"existing-video")
            (segment_directory / "meta.json").write_text(
                '{"cameras": [{"video_file": "cam1_Front.mp4"}], '
                '"synthetic_ppg_hrv_path": "obsolete.csv"}',
                encoding="utf-8",
            )

            synthetic_path = root / "synthetic_ppg" / "synthetic_ppg.csv"
            synthetic_path.parent.mkdir()
            pd.DataFrame({
                "timestamp_utc": [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:01Z",
                ],
                "synthetic_ppg": [0.1, 0.2],
            }).to_csv(synthetic_path, index=False)
            interval = LabelInterval("Test", 0.0, 1.0, folder=str(segment_directory))
            state = ProjectState(
                output_directory=str(root),
                intervals=[interval],
                synthetic_ppg_path=str(synthetic_path),
            )
            timestamps = pd.to_datetime([
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:01Z",
            ])
            widget = SimpleNamespace(
                state=state,
                _hr_merged_df=pd.DataFrame({
                    "timestamp_utc": timestamps,
                    "Polar_1": [70, 71],
                }),
                _ecg_merged_df=pd.DataFrame({
                    "timestamp_utc": timestamps,
                    "ECG_1": [0.3, 0.4],
                }),
                _remove_legacy_csv_cb=Mock(),
                _status=Mock(),
                _load_synced_signal=Mock(return_value=True),
                _signal_anchor=Mock(
                    return_value=datetime(2026, 1, 1, tzinfo=timezone.utc)
                ),
                _legacy_segment_csv_paths=Mode3Widget._legacy_segment_csv_paths,
                _relative_signal_paths=Mode3Widget._relative_signal_paths,
            )
            widget._remove_legacy_csv_cb.isChecked.return_value = False
            widget._export_signal_slice = Mode3Widget._export_signal_slice
            widget._export_synthetic_artifact_slices = MethodType(
                Mode3Widget._export_synthetic_artifact_slices, widget
            )
            widget._export_all_signal_slices = MethodType(
                Mode3Widget._export_all_signal_slices, widget
            )

            Mode3Widget._export_signals_to_existing_segments(widget)

            self.assertEqual(video_path.read_bytes(), b"existing-video")
            self.assertTrue((segment_directory / "polar_1.csv").exists())
            self.assertTrue((segment_directory / "ecg_1.csv").exists())
            self.assertTrue(
                (segment_directory / "synthetic_ppg" / "synthetic_ppg.csv").exists()
            )
            metadata = pd.read_json(segment_directory / "meta.json", typ="series")
            self.assertEqual(metadata["hr_files"], ["polar_1.csv"])
            self.assertEqual(metadata["ecg_files"], ["ecg_1.csv"])
            self.assertEqual(
                metadata["ppg_files"], ["synthetic_ppg/synthetic_ppg.csv"]
            )
            self.assertEqual(
                metadata["synthetic_ppg_path"],
                "synthetic_ppg/synthetic_ppg.csv",
            )
            self.assertNotIn("synthetic_ppg_hrv_path", metadata)


if __name__ == "__main__":
    unittest.main()