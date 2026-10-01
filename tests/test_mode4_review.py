import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from app.modes.mode4_review import Mode4Widget


class Mode4ReviewSignalTests(unittest.TestCase):
    def test_loads_segment_ecg_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            segment_directory = Path(temporary_directory)
            pd.DataFrame({
                "timestamp_utc": ["2026-01-01T00:00:00Z"],
                "ecg_waveform": [0.25],
            }).to_csv(segment_directory / "ecg_1.csv", index=False)
            segment = SimpleNamespace(
                index=0,
                start_sec=0.0,
                end_sec=1.0,
                dir_path=segment_directory,
            )

            frame = Mode4Widget._load_segment_signal(
                SimpleNamespace(),
                segment,
                {"ecg_files": ["ecg_1.csv"]},
                "ecg_files",
                None,
            )

            self.assertIsNotNone(frame)
            self.assertEqual(list(frame.columns), ["timestamp_utc", "ECG_1"])
            self.assertEqual(frame["ECG_1"].iloc[0], 0.25)

    @staticmethod
    def _plot_widget(plot_index: int, **attributes) -> SimpleNamespace:
        widget = SimpleNamespace(
            _plot_type=SimpleNamespace(currentIndex=lambda: plot_index),
            _current_seg=SimpleNamespace(duration_sec=12.5),
            _seg_hr_df=None,
            _seg_ecg_df=None,
            _seg_synthetic={},
            _synced_hr_df=None,
            _synced_ecg_df=None,
            _synced_synthetic={},
            _plot=Mock(),
            _metrics_label=Mock(),
        )
        for name, value in attributes.items():
            setattr(widget, name, value)
        widget._selected_view = lambda: Mode4Widget._selected_view(widget)
        return widget

    def test_ecg_selection_displays_segment_ecg(self) -> None:
        ecg_frame = pd.DataFrame({
            "timestamp_utc": pd.to_datetime(["2026-01-01T00:00:00Z"]),
            "ECG_1": [0.25],
        })
        widget = self._plot_widget(1, _seg_ecg_df=ecg_frame)

        Mode4Widget._show_selected_plot(widget)

        widget._plot.set_data.assert_called_once_with(
            ecg_frame, video_duration_sec=12.5
        )
        widget._plot.clear.assert_not_called()

    def test_segment_heart_rate_comparison_combines_sensor_and_synthetic_hr(self) -> None:
        timestamps = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z"])
        sensor_hr = pd.DataFrame({"timestamp_utc": timestamps, "Zephyr_1": [60.0, 62.0]})
        synthetic_hr = pd.DataFrame({"timestamp_utc": timestamps, "heart_rate_bpm": [61.0, 63.0]})
        widget = self._plot_widget(
            4,
            _seg_hr_df=sensor_hr,
            _seg_synthetic={"synthetic_ppg_hr_path": synthetic_hr},
        )

        Mode4Widget._show_selected_plot(widget)

        frame = widget._plot.set_data.call_args.args[0]
        self.assertEqual(
            list(frame.columns),
            ["timestamp_utc", "Zephyr_1", "Synthetic PPG HR (10 s FFT)"],
        )
        self.assertIn("MAE=1.00 BPM", widget._metrics_label.setText.call_args.args[0])

    def test_segment_rr_view_loads_exported_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            segment_directory = Path(temporary_directory)
            (segment_directory / "synthetic_ppg").mkdir()
            pd.DataFrame({
                "timestamp_utc": ["2026-01-01T00:00:00Z"],
                "rr_interval_ms": [800.0],
            }).to_csv(segment_directory / "synthetic_ppg" / "synthetic_ppg_rr.csv", index=False)
            segment = SimpleNamespace(index=0, start_sec=0.0, end_sec=1.0, dir_path=segment_directory)
            widget = SimpleNamespace(_synced_synthetic={}, state=SimpleNamespace(last_signal_anchor_datetime=None))
            widget._slice_to_segment = lambda *args: Mode4Widget._slice_to_segment(widget, *args)

            artifacts = Mode4Widget._load_segment_synthetic_artifacts(
                widget,
                segment,
                {"synthetic_rr_path": "synthetic_ppg/synthetic_ppg_rr.csv"},
            )

            self.assertEqual(artifacts["synthetic_rr_path"]["rr_interval_ms"].iloc[0], 800.0)
            self.assertIsNone(artifacts["synthetic_ppg_hrv_path"])

    def test_global_view_clears_selected_segment(self) -> None:
        widget = SimpleNamespace(
            _current_seg=SimpleNamespace(index=2),
            _seg_hr_df=Mock(),
            _seg_ecg_df=Mock(),
            _seg_list=Mock(),
            _player=Mock(),
            _global_view_btn=Mock(),
            _show_selected_plot=Mock(),
            _set_overview_info=Mock(),
        )

        Mode4Widget._show_global_view(widget)

        self.assertIsNone(widget._current_seg)
        self.assertIsNone(widget._seg_hr_df)
        self.assertIsNone(widget._seg_ecg_df)
        widget._seg_list.clearSelection.assert_called_once_with()
        widget._seg_list.setCurrentRow.assert_called_once_with(-1)
        widget._player.set_cameras.assert_called_once_with([])
        widget._global_view_btn.setEnabled.assert_called_once_with(False)
        widget._show_selected_plot.assert_called_once_with()
        widget._set_overview_info.assert_called_once_with()

    def test_cleared_selection_restores_global_view(self) -> None:
        widget = SimpleNamespace(_segments=[], _show_global_view=Mock())

        Mode4Widget._on_segment_selected(widget, -1)

        widget._show_global_view.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
