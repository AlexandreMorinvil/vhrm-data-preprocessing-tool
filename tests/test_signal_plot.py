import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
from PyQt6.QtWidgets import QApplication

from app.widgets.signal_plot import SignalPlot


class SignalPlotWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.plot = SignalPlot()
        frame = pd.DataFrame({
            "timestamp_utc": pd.to_datetime([
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:05Z",
                "2026-01-01T00:00:10Z",
                "2026-01-01T00:00:15Z",
                "2026-01-01T00:00:20Z",
            ]),
            "ECG_1": [0.0, 0.5, -0.25, 0.75, 0.0],
        })
        self.plot.set_data(frame, video_duration_sec=20.0)

    def tearDown(self) -> None:
        self.plot.close()

    def assertVisibleRange(self, start_sec: float, end_sec: float) -> None:
        actual_start, actual_end = self.plot._ax.get_xlim()
        self.assertAlmostEqual(actual_start, start_sec)
        self.assertAlmostEqual(actual_end, end_sec)

    def test_selects_and_moves_time_window(self) -> None:
        self.plot._on_range_selected(2.0, 7.0)
        self.assertVisibleRange(2.0, 7.0)

        self.plot._pan_window(1)
        self.assertVisibleRange(6.0, 11.0)

        self.plot._pan_window(-1)
        self.assertVisibleRange(2.0, 7.0)

        self.plot._window_slider.setValue(10_000)
        self.assertVisibleRange(15.0, 20.0)

        self.plot._show_full_range()
        self.assertVisibleRange(0.0, 20.0)

    def test_adjusts_y_limits_automatically_and_manually(self) -> None:
        self.plot._on_range_selected(2.0, 12.0)
        auto_min, auto_max = self.plot._ax.get_ylim()
        self.assertAlmostEqual(auto_min, -0.2875)
        self.assertAlmostEqual(auto_max, 0.5375)

        self.plot._set_zero_y_min()
        y_min, y_max = self.plot._ax.get_ylim()
        self.assertEqual(y_min, 0.0)
        self.assertAlmostEqual(y_max, 0.5375)

        self.plot._set_hr_y_max()
        self.assertEqual(self.plot._ax.get_ylim(), (0.0, 220.0))

        self.plot._y_min.setValue(-2.0)
        self.plot._y_max.setValue(3.0)
        self.assertEqual(self.plot._ax.get_ylim(), (-2.0, 3.0))

        self.plot._auto_y_min.setChecked(True)
        self.plot._auto_y_max.setChecked(True)
        restored_min, restored_max = self.plot._ax.get_ylim()
        self.assertAlmostEqual(restored_min, -0.2875)
        self.assertAlmostEqual(restored_max, 0.5375)


if __name__ == "__main__":
    unittest.main()
