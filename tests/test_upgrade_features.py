import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QApplication

from app.face_privacy import FaceAnonymizer, FacePrivacySettings
from app.state import LabelInterval, ProjectState
from app.timefmt import format_hms_ms, parse_time_text
from app.widgets.frame_preview import frame_to_ms, ms_to_frame
from app.widgets.timeline import IntervalItem, TimelineWidget, colour_for_label


class TimeFormattingTests(unittest.TestCase):
    def test_parses_common_time_formats(self) -> None:
        self.assertAlmostEqual(parse_time_text("1:02:03.5"), 3723.5)
        self.assertAlmostEqual(parse_time_text("02:03"), 123.0)
        self.assertAlmostEqual(parse_time_text("12,25"), 12.25)
        self.assertAlmostEqual(parse_time_text("f600", fps=60.0), 10.0)
        self.assertIsNone(parse_time_text("abc"))
        self.assertIsNone(parse_time_text("f10"))
        self.assertEqual(format_hms_ms(3723.5), "01:02:03.500")

    def test_frame_position_mapping_round_trips_at_ntsc_rate(self) -> None:
        fps = 60000 / 1001
        for frame in list(range(0, 5000)) + [204_915, 204_916]:
            self.assertEqual(ms_to_frame(frame_to_ms(frame, fps), fps), frame)


class FaceTrackingTests(unittest.TestCase):
    def test_missed_detections_stay_masked_for_persistence_window(self) -> None:
        tracker = FaceAnonymizer(FacePrivacySettings(persistence_sec=0.5), fps=10)
        face = (100.0, 100.0, 40.0, 40.0, 0.9)
        self.assertEqual(len(tracker.update_with_detections([face], 0.0)), 1)
        self.assertEqual(len(tracker.update_with_detections([face], 0.1)), 1)
        held = tracker.update_with_detections([], 0.4)
        self.assertEqual(len(held), 1)
        self.assertGreater(held[0][2], 40.0, "unseen faces grow to cover motion")
        self.assertEqual(tracker.update_with_detections([], 0.7), [])

    def test_moving_face_is_matched_to_existing_track(self) -> None:
        tracker = FaceAnonymizer(FacePrivacySettings(persistence_sec=0.5), fps=10)
        tracker.update_with_detections([(100.0, 100.0, 40.0, 40.0, 0.9)], 0.0)
        boxes = tracker.update_with_detections([(110.0, 100.0, 40.0, 40.0, 0.9)], 0.1)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0][0], 110.0)

    def test_settings_round_trip_and_clamp(self) -> None:
        settings = FacePrivacySettings.from_dict({"style": "bogus", "strength": 99, "unknown": 1})
        self.assertEqual(settings.style, "blur")
        self.assertEqual(settings.strength, 10)
        self.assertEqual(FacePrivacySettings.from_dict(settings.to_dict()), settings)


class TimelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.timeline = TimelineWidget()
        self.timeline.resize(1000, 90)
        self.timeline.set_duration(100.0)
        self.timeline.set_intervals([
            IntervalItem("A", 10.0, 20.0),
            IntervalItem("B", 30.0, 40.0),
            IntervalItem("C", 60.0, 70.0),
        ])

    def tearDown(self) -> None:
        self.timeline.close()

    def test_zoomed_view_maps_time_to_pixels(self) -> None:
        self.timeline.set_content_margins(100, 0)
        self.timeline.set_view_range(25.0, 50.0)
        self.assertEqual(self.timeline.view_range(), (25.0, 50.0))
        self.assertAlmostEqual(self.timeline._sec_to_x(25.0), 100.0)
        self.assertAlmostEqual(self.timeline._x_to_sec(self.timeline._sec_to_x(37.5)), 37.5)

    def test_view_is_clamped_to_session(self) -> None:
        self.timeline.set_view_range(90.0, 130.0)
        self.assertEqual(self.timeline.view_range(), (60.0, 100.0))

    def test_neighbour_limits_prevent_overlap(self) -> None:
        self.assertEqual(self.timeline._neighbour_limits(1), (20.0, 60.0))
        self.assertEqual(self.timeline._neighbour_limits(0), (0.0, 30.0))
        self.assertEqual(self.timeline._neighbour_limits(2), (40.0, 100.0))

    def test_follow_playhead_pages_the_view(self) -> None:
        self.timeline.set_view_range(0.0, 20.0)
        self.timeline.set_playhead(50.0)
        start, end = self.timeline.view_range()
        self.assertLessEqual(start, 50.0)
        self.assertGreaterEqual(end, 50.0)

    def test_unknown_label_colour_is_stable(self) -> None:
        self.assertEqual(colour_for_label("Custom", []), colour_for_label("Custom", []))
        self.assertEqual(colour_for_label("B", ["A", "B"]), "#cc4444")


class VisualExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _plot(self):
        from app.widgets.signal_plot import SignalPlot

        plot = SignalPlot()
        anchor = pd.Timestamp("2026-01-01T12:00:00Z")
        hr = pd.DataFrame({
            "timestamp_utc": anchor + pd.to_timedelta(np.arange(0, 60, 1.0) + 5.0, unit="s"),
            "Zephyr_1": np.linspace(60, 90, 60),
        })
        ecg = pd.DataFrame({
            "timestamp_utc": anchor + pd.to_timedelta(np.arange(0, 60, 0.004), unit="s"),
            "ECG_1": np.sin(np.arange(15000) / 20),
        })
        plot.set_time_zero(anchor)
        plot.set_panels([("Heart rate", hr, "BPM"), ("ECG", ecg, "ECG")], video_duration_sec=60.0)
        plot.set_intervals([LabelInterval("Rest", 10.0, 20.0, "#4488cc"),
                            LabelInterval("Squat", 30.0, 45.0, "#cc4444")])
        self.addCleanup(plot.close)
        return plot

    def test_signals_are_placed_relative_to_the_anchor(self) -> None:
        plot = self._plot()
        self.assertEqual(plot.x_range(), (0.0, 64.0))
        hr_line = next(line for line in plot._signal_lines if line.get_label() == "Zephyr_1")
        self.assertAlmostEqual(float(hr_line.get_xdata()[0]), 5.0)

    def test_exports_figures_and_csv(self) -> None:
        from app.visual_export import FigureOptions, export_interval_figures, export_plot_csv, render_signal_figure, save_figure

        plot = self._plot()
        snapshot = plot.snapshot()
        with tempfile.TemporaryDirectory() as directory:
            fig = render_signal_figure(snapshot, FigureOptions(x_range=(0.0, 60.0), dpi=60))
            for ext in ("png", "svg", "pdf"):
                self.assertTrue(save_figure(fig, Path(directory) / f"f.{ext}").stat().st_size > 0)
            labels_only = render_signal_figure(snapshot, FigureOptions(x_range=(0.0, 60.0), labels_only=True, dpi=60))
            self.assertEqual(len(labels_only.axes), 1)
            written = export_interval_figures(
                snapshot, [(0, "Rest", 10.0, 20.0), (1, "Squat", 30.0, 45.0)], directory,
                FigureOptions(x_range=(0.0, 1.0), dpi=50),
            )
            self.assertEqual([p.name for p in written], ["0000_Rest.png", "0001_Squat.png"])
            rows = export_plot_csv(snapshot, Path(directory) / "data.csv", (10.0, 11.0))
            table = pd.read_csv(Path(directory) / "data.csv")
            self.assertEqual(len(table), rows)
            self.assertEqual(list(table.columns[:3]), ["video_time_sec", "timestamp_utc", "label"])
            self.assertEqual(set(table["label"].dropna()), {"Rest"})


class LabellingWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _widget(self):
        from app.modes.mode3_labelling import Mode3Widget

        state = ProjectState(intervals=[
            LabelInterval("Baseline", 10.0, 20.0),
            LabelInterval("Squat", 30.0, 40.0),
        ])
        widget = Mode3Widget(state)
        widget._timeline.set_duration(100.0)
        self.addCleanup(widget.close)
        return widget, state

    def test_undo_and_redo_interval_creation(self) -> None:
        widget, state = self._widget()
        widget._on_interval_created(50.0, 55.0)
        self.assertEqual(len(state.intervals), 3)
        widget._undo_action()
        self.assertEqual(len(widget.state.intervals), 2)
        widget._redo_action()
        self.assertEqual([iv.start_sec for iv in widget.state.intervals], [10.0, 30.0, 50.0])

    def test_rejects_overlapping_resize(self) -> None:
        widget, state = self._widget()
        widget._on_interval_resized(0, 10.0, 35.0)
        self.assertEqual(state.intervals[0].end_sec, 20.0)
        widget._on_interval_resized(0, 12.0, 25.0)
        self.assertEqual((state.intervals[0].start_sec, state.intervals[0].end_sec), (12.0, 25.0))

    def test_split_and_label_keys(self) -> None:
        widget, state = self._widget()
        widget._split_interval(1, 35.0)
        self.assertEqual([(iv.label, iv.start_sec, iv.end_sec) for iv in state.intervals],
                         [("Baseline", 10.0, 20.0), ("Squat", 30.0, 35.0), ("Squat", 35.0, 40.0)])
        widget._choose_label(1)
        self.assertEqual(widget._current_label(), state.labels_library[1])

    def test_start_end_label_creates_interval_at_playhead(self) -> None:
        widget, state = self._widget()
        widget._timeline.set_playhead(60.0)
        widget._start_label_at_playhead()
        widget._timeline.set_playhead(64.5)
        widget._end_label_at_playhead()
        self.assertEqual((state.intervals[-1].start_sec, state.intervals[-1].end_sec), (60.0, 64.5))
        self.assertIsNone(widget._pending_start)


if __name__ == "__main__":
    unittest.main()
