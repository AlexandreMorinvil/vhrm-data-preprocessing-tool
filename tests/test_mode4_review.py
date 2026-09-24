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

    def test_ecg_selection_displays_segment_ecg(self) -> None:
        ecg_frame = pd.DataFrame({
            "timestamp_utc": pd.to_datetime(["2026-01-01T00:00:00Z"]),
            "ECG_1": [0.25],
        })
        widget = SimpleNamespace(
            _plot_type=SimpleNamespace(currentIndex=lambda: 1),
            _current_seg=SimpleNamespace(duration_sec=12.5),
            _seg_hr_df=None,
            _seg_ecg_df=ecg_frame,
            _plot=Mock(),
        )

        Mode4Widget._show_selected_plot(widget)

        widget._plot.set_data.assert_called_once_with(
            ecg_frame, video_duration_sec=12.5
        )
        widget._plot.clear.assert_not_called()

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
