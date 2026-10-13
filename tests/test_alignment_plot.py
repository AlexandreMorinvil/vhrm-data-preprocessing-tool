import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication

from app.ppg_alignment import scan_lags
from app.widgets.alignment_plot import MODE_SHIFT, MODE_ZOOM, AlignmentPlot

_app = QApplication.instance() or QApplication([])


class AlignmentPlotMouseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plot = AlignmentPlot()
        self.plot.resize(1200, 700)
        self.plot.show()
        times = np.arange(0, 60, 1 / 30)
        reference_t = np.arange(-12, 72, 1 / 125)
        reference = np.sin(2 * np.pi * 1.1 * reference_t)
        pulse = np.sin(2 * np.pi * 1.1 * (times - 0.4))
        self.plot.set_segment_duration(60.0)
        self.plot.set_window(0.0, 60.0)
        self.plot.set_pulse(times, pulse)
        self.plot.set_reference(reference_t, reference)
        self.plot.set_scan(scan_lags(times, pulse, reference_t, reference, -5, 5))
        self.offsets: list[tuple[float, bool]] = []
        self.windows: list[tuple[float, float]] = []
        self.plot.window_changed.connect(lambda low, high: self.windows.append((low, high)))
        self.plot.offset_changed.connect(lambda value, done: self.offsets.append((value, done)))
        _app.processEvents()
        self.view = self.plot._signal_plot.getViewBox()
        self.view.setXRange(0, 60, padding=0)
        _app.processEvents()

    def tearDown(self) -> None:
        self.plot.close()

    def _point(self, view, x: float, y: float) -> QPoint:
        scene = view.mapViewToScene(QPointF(x, y))
        return self.plot._graphics.mapFromScene(scene)

    def _drag(self, view, start: tuple[float, float], end: tuple[float, float],
              modifiers=Qt.KeyboardModifier.NoModifier, grab_offset_px: int = 0) -> None:
        """Drag between two data points; *grab_offset_px* moves both points horizontally (where the press lands)."""
        viewport = self.plot._graphics.viewport()
        begin, finish = self._point(view, *start), self._point(view, *end)
        begin += QPoint(grab_offset_px, 0)
        finish += QPoint(grab_offset_px, 0)
        left = Qt.MouseButton.LeftButton

        def send(kind, position, button, buttons):
            event = QMouseEvent(kind, QPointF(position), QPointF(viewport.mapToGlobal(position)),
                                button, buttons, modifiers)
            QApplication.sendEvent(viewport, event)

        send(QEvent.Type.MouseButtonPress, begin, left, left)
        for step in range(1, 11):
            position = begin + (finish - begin) * step / 10
            time.sleep(0.03)  # pyqtgraph drops move events that arrive faster than the display rate
            send(QEvent.Type.MouseMove, position, Qt.MouseButton.NoButton, left)
        send(QEvent.Type.MouseButtonRelease, finish, left, Qt.MouseButton.NoButton)
        _app.processEvents()

    def test_default_drag_zooms_and_can_be_undone(self) -> None:
        self.assertEqual(self.plot.drag_mode, MODE_ZOOM)
        self._drag(self.view, (10, 0), (20, 0))
        low, high = self.view.viewRange()[0]
        self.assertAlmostEqual(low, 10, delta=0.3)
        self.assertAlmostEqual(high, 20, delta=0.3)
        self.assertEqual(self.offsets, [])
        self.assertEqual(self.plot.offset, 0.0)
        self.plot.undo_zoom()
        low, high = self.view.viewRange()[0]
        self.assertAlmostEqual(low, 0, delta=0.3)
        self.assertAlmostEqual(high, 60, delta=0.3)

    def test_shift_drag_moves_the_ppg_in_zoom_mode(self) -> None:
        self._drag(self.view, (10, 0), (12, 0), Qt.KeyboardModifier.ShiftModifier)
        self.assertAlmostEqual(self.plot.offset, 2.0, delta=0.15)
        self.assertTrue(self.offsets[-1][1])
        low, high = self.view.viewRange()[0]
        self.assertAlmostEqual(high - low, 60, delta=0.3)

    def test_shift_mode_moves_the_ppg_without_zooming(self) -> None:
        self.plot.set_drag_mode(MODE_SHIFT)
        self._drag(self.view, (30, 0), (29, 0))
        self.assertAlmostEqual(self.plot.offset, -1.0, delta=0.15)
        low, high = self.view.viewRange()[0]
        self.assertAlmostEqual(high - low, 60, delta=0.3)

    def test_ctrl_drag_pans(self) -> None:
        self.view.setXRange(20, 30, padding=0)
        _app.processEvents()
        self._drag(self.view, (25, 0), (23, 0), Qt.KeyboardModifier.ControlModifier)
        low, high = self.view.viewRange()[0]
        self.assertAlmostEqual(high - low, 10, delta=0.3)
        self.assertGreater(low, 20.5)
        self.assertEqual(self.offsets, [])

    def test_correlation_plot_zooms_and_full_range_restores_it(self) -> None:
        lag_view = self.plot._lag_plot.getViewBox()
        self._drag(lag_view, (-2, 0), (2, 0))
        low, high = lag_view.viewRange()[0]
        self.assertAlmostEqual(low, -2, delta=0.1)
        self.assertAlmostEqual(high, 2, delta=0.1)
        self.plot.reset_zoom()
        low, high = lag_view.viewRange()[0]
        self.assertLess(low, -4.9)
        self.assertGreater(high, 4.9)


    def _middle_y(self) -> float:
        low, high = self.view.viewRange()[1]
        return (low + high) / 2

    def test_window_handles_point_outward_and_can_be_grabbed(self) -> None:
        self.plot.set_window(10.0, 40.0)
        _app.processEvents()
        middle = self._middle_y()
        # Press on the start handle's half disc, 7 px to the left of the line.
        self._drag(self.view, (10, middle), (15, middle), grab_offset_px=-7)
        self.assertAlmostEqual(self.plot.window()[0], 15, delta=0.3)
        # Press on the end handle's half disc, 7 px to the right of the line.
        self._drag(self.view, (40, middle), (45, middle), grab_offset_px=7)
        self.assertAlmostEqual(self.plot.window()[1], 45, delta=0.3)
        self.assertAlmostEqual(self.windows[-1][0], 15, delta=0.3)
        self.assertAlmostEqual(self.windows[-1][1], 45, delta=0.3)
        low, high = self.view.viewRange()[0]
        self.assertAlmostEqual(high - low, 60, delta=0.3)  # no zoom happened
        self.assertEqual(self.offsets, [])

    def test_edges_have_a_wide_grab_band_along_their_whole_height(self) -> None:
        self.plot.set_window(10.0, 40.0)
        _app.processEvents()
        low, high = self.view.viewRange()[1]
        near_top = high - (high - low) * 0.1
        self._drag(self.view, (40, near_top), (30, near_top), grab_offset_px=-6)
        self.assertAlmostEqual(self.plot.window()[1], 30, delta=0.3)

    def test_edges_cannot_cross(self) -> None:
        self.plot.set_window(10.0, 20.0)
        _app.processEvents()
        middle = self._middle_y()
        self._drag(self.view, (10, middle), (50, middle), grab_offset_px=-7)
        start, end = self.plot.window()
        self.assertAlmostEqual(end, 20.0, delta=1e-6)
        self.assertLessEqual(start, end - 0.5 + 1e-6)

    def test_window_follows_a_new_segment_without_old_limits(self) -> None:
        self.plot.set_window(0.0, 60.0)
        self.plot.set_segment_duration(570.0)
        self.plot.set_window(100.0, 570.0)
        self.assertEqual(self.plot.window(), (100.0, 570.0))
        self.plot.set_segment_duration(30.0)
        self.plot.set_window(0.0, 30.0)
        self.assertEqual(self.plot.window(), (0.0, 30.0))


if __name__ == "__main__":
    unittest.main()
