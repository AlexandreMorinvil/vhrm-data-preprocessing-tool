import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtWidgets import QApplication, QCheckBox, QMenu, QMessageBox

from app.modes.mode5_ppg_alignment import PpgAlignmentWidget
from app.ppg_alignment import FaceTrace, FaceTraceSettings, cache_key, cache_paths, save_trace
from app.signal_offset import apply_signal_offset, read_segment_meta
from app.state import LabelInterval, ProjectState
from app.synthetic_ppg import synthesize_from_peak_times
from tests import test_ppg_alignment, test_signal_offset

_app = QApplication.instance() or QApplication([])


class SegmentMarkTests(unittest.TestCase):
    def test_estimated_segments_are_marked_right_after_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intervals = []
            for index, label in enumerate(("Resting", "Recovery")):
                folder = root / "labelled_segments" / f"{index:04d}_{label}"
                folder.mkdir(parents=True)
                for number, view in enumerate(("Back-left", "Front"), start=1):
                    (folder / f"cam{number}_Camera_{view}.mp4").write_bytes(b"")
                cameras = [{"label": f"Camera_{view}", "video_file": f"cam{number}_Camera_{view}.mp4"}
                           for number, view in enumerate(("Back-left", "Front"), start=1)]
                (folder / "meta.json").write_text(json.dumps({"cameras": cameras}), encoding="utf-8")
                intervals.append(LabelInterval(label, index * 100.0, index * 100.0 + 60.0, folder=str(folder)))
            estimated = Path(intervals[1].folder)
            trace_path = cache_paths(root / "ppg_alignment" / "cache",
                                     cache_key(str(estimated), "Camera_Front_skin"))[0]
            trace_path.parent.mkdir(parents=True)
            trace_path.write_bytes(b"")

            # The section starts empty, then metadata and manifest are loaded.
            state = ProjectState()
            widget = PpgAlignmentWidget(state)
            state.output_directory = str(root)
            state.intervals = intervals
            widget.refresh_from_state()

            texts = [widget._segment_list.item(row).text() for row in range(widget._segment_list.count())]
            self.assertFalse(texts[0].endswith("✓"))
            self.assertTrue(texts[1].endswith("✓"))

            # The face tracking options are off by default (original estimates) and keep
            # their own estimates apart.
            segment = widget._segments[1]
            self.assertEqual(widget._key(segment), cache_key(str(estimated), "Camera_Front_skin"))
            self.assertEqual(widget._trace_settings(), FaceTraceSettings())
            widget._every_frame_cb.setChecked(True)
            widget._centred_cb.setChecked(True)
            self.assertEqual(widget._key(segment), cache_key(str(estimated), "Camera_Front_skin_every_frame_centred"))
            self.assertTrue(widget._trace_settings().detect_every_frame)
            self.assertTrue(widget._trace_settings().centred_smoothing)
            self.assertFalse(widget._segment_list.item(1).text().endswith("✓"))
            widget._every_frame_cb.setChecked(False)
            widget._centred_cb.setChecked(False)
            self.assertTrue(widget._segment_list.item(1).text().endswith("✓"))
            widget._player.shutdown()


class EstimateByLabelTests(unittest.TestCase):
    def test_estimate_all_can_be_limited_to_one_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intervals = []
            for index, label in enumerate(("Resting", "Curl", "Resting")):
                folder = root / "labelled_segments" / f"{index:04d}_{label}"
                folder.mkdir(parents=True)
                (folder / "cam1_Camera_Front.mp4").write_bytes(b"")
                (folder / "meta.json").write_text(json.dumps(
                    {"cameras": [{"label": "Camera_Front", "video_file": "cam1_Camera_Front.mp4"}]}),
                    encoding="utf-8")
                intervals.append(LabelInterval(label, index * 100.0, index * 100.0 + 60.0, folder=str(folder)))
            state = ProjectState()
            state.output_directory = str(root)
            state.ffmpeg_path = "ffmpeg"
            state.intervals = intervals
            widget = PpgAlignmentWidget(state)
            combo = widget._estimate_label_combo
            self.assertEqual([combo.itemText(i) for i in range(combo.count())],
                             ["All labels", "Resting (2)", "Curl (1)"])

            def queued() -> list[str]:
                with mock.patch.object(widget, "_estimate_job") as job:
                    widget._start_estimate(all_segments=True)
                    deadline = time.time() + 30
                    while widget._estimate_worker is not None and time.time() < deadline:
                        _app.processEvents()
                        time.sleep(0.01)
                return [name for name, *_rest in job.call_args.kwargs["jobs"]]

            combo.setCurrentIndex(combo.findData("Resting"))
            self.assertEqual(queued(), ["0000_Resting", "0002_Resting"])
            # The chosen label survives a refresh of the list.
            widget.refresh_from_state()
            self.assertEqual(combo.currentData(), "Resting")
            combo.setCurrentIndex(0)
            self.assertEqual(queued(), ["0000_Resting", "0001_Curl", "0002_Resting"])
            widget._player.shutdown()


class CameraViewTests(unittest.TestCase):
    def test_switching_camera_shows_its_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "labelled_segments" / "0000_Resting"
            folder.mkdir(parents=True)
            cameras = []
            for number, view in enumerate(("Back-left", "Front", "Side"), start=1):
                (folder / f"cam{number}_Camera_{view}.mp4").write_bytes(b"")
                cameras.append({"label": f"Camera_{view}", "video_file": f"cam{number}_Camera_{view}.mp4"})
            (folder / "meta.json").write_text(json.dumps({"cameras": cameras}), encoding="utf-8")
            state = ProjectState()
            state.output_directory = directory
            state.intervals = [LabelInterval("Resting", 0.0, 60.0, folder=str(folder))]
            widget = PpgAlignmentWidget(state)

            def checked() -> set[str]:
                boxes = (widget._views_layout.itemAt(i).widget() for i in range(widget._views_layout.count()))
                return {box.text() for box in boxes if isinstance(box, QCheckBox) and box.isChecked()}

            self.assertEqual(widget._camera_label(), "Camera_Front")
            self.assertIn("Camera_Front", checked())
            widget._camera_combo.setCurrentText("Camera_Back-left")
            self.assertIn("Camera_Back-left", checked())
            self.assertNotIn("Camera_Front", checked())

            # A camera the user shows by hand stays shown when it becomes the selected one.
            widget._toggle_view("Camera_Side", True)
            widget._camera_combo.setCurrentText("Camera_Side")
            _app.processEvents()
            self.assertTrue({"Camera_Back-left", "Camera_Side"} <= checked())
            widget._player.shutdown()


class RemoveOffsetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project = test_signal_offset.SignalOffsetTests("test_apply_shifts_anchor_and_reslices_every_signal")
        self.project.setUp()
        self.state = self.project.state
        apply_signal_offset(self.state, 1.5, note="baseline estimate")
        self.widget = PpgAlignmentWidget(self.state)

    def tearDown(self) -> None:
        self.widget._player.shutdown()
        self.project.tearDown()

    def _remove(self, answer) -> None:
        with mock.patch.object(QMessageBox, "question", return_value=answer):
            self.widget._remove_offset()
        deadline = time.time() + 30
        while self.widget._apply_worker is not None and time.time() < deadline:
            _app.processEvents()
            time.sleep(0.01)

    def test_remove_returns_to_camera_timing_and_shows_it(self) -> None:
        self._remove(QMessageBox.StandardButton.Yes)
        self.assertEqual(self.state.video_signal_offset_sec, 0.0)
        self.assertEqual(read_segment_meta(self.project.segment)["video_signal_offset_sec"], 0.0)
        self.assertEqual(self.widget._offset_spin.value(), 0.0)
        self.assertTrue(self.widget._remove_btn.isEnabled())

    def test_cancelling_keeps_the_offset(self) -> None:
        self._remove(QMessageBox.StandardButton.No)
        self.assertEqual(self.state.video_signal_offset_sec, 1.5)
        self.assertEqual(read_segment_meta(self.project.segment)["video_signal_offset_sec"], 1.5)


_START = pd.Timestamp("2026-05-22T12:00:00Z")


def _write_segment(root: Path, index: int, label: str, start_sec: float, rr_sec: float,
                   offset: float, stretch: float = 1.0, seed: int = 0,
                   pulse_speed: float = 1.0) -> LabelInterval:
    """A 60 s segment with its synthetic PPG, ECG peaks and the cached face trace of a
    video whose pulse is the PPG shifted by *offset* (positive: the signals are later).
    A *pulse_speed* other than 1 makes the video pulse faster than the PPG (bad rPPG)."""
    folder = root / "labelled_segments" / f"{index:04d}_{label}"
    (folder / "synthetic_ppg").mkdir(parents=True)
    video = folder / "cam2_Camera_Front.mp4"
    video.write_bytes(b"")
    segment_start = _START + pd.Timedelta(seconds=start_sec)
    rng = np.random.default_rng(seed)
    rr = rr_sec * rng.uniform(0.9, 1.1, int(400 / rr_sec))
    peaks = segment_start - pd.Timedelta(seconds=20) + pd.to_timedelta(np.concatenate(([0.0], np.cumsum(rr))), unit="s")
    ppg = synthesize_from_peak_times(pd.DatetimeIndex(peaks), fs=125.0).ppg
    elapsed = (ppg["timestamp_utc"] - ppg["timestamp_utc"].iloc[0]).dt.total_seconds()
    ppg["timestamp_utc"] = ppg["timestamp_utc"].iloc[0] + pd.to_timedelta(elapsed * stretch, unit="s")
    ppg.to_csv(folder / "synthetic_ppg" / "synthetic_ppg.csv", index=False)
    pd.DataFrame({"timestamp_utc": peaks}).to_csv(folder / "synthetic_ppg" / "synthetic_ppg_rr.csv", index=False)
    (folder / "meta.json").write_text(json.dumps({
        "cameras": [{"label": "Camera_Front", "video_file": video.name}],
        "segment_start_datetime_utc": segment_start.isoformat(),
        "synthetic_ppg_path": "synthetic_ppg/synthetic_ppg.csv",
        "synthetic_rr_path": "synthetic_ppg/synthetic_ppg_rr.csv",
    }), encoding="utf-8")

    times = np.arange(0, 60, 1 / test_ppg_alignment.FS)
    ppg_epoch = (ppg["timestamp_utc"] - segment_start).dt.total_seconds().to_numpy()
    blood_volume = np.interp((times - offset) * pulse_speed, ppg_epoch, ppg["synthetic_ppg"].to_numpy())
    trace = FaceTrace(
        fs=test_ppg_alignment.FS, rgb=test_ppg_alignment._skin_rgb(blood_volume - 0.5, seed=seed),
        detected=np.ones(len(times), dtype=bool), boxes=np.full((len(times), 4), np.nan),
        frame_size=(10, 10), video_path=str(video), settings=FaceTraceSettings(),
    )
    save_trace(root / "ppg_alignment" / "cache", cache_key(str(folder), "Camera_Front_skin"), trace)
    return LabelInterval(label, start_sec, start_sec + 60.0, folder=str(folder))


class PooledScanTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)

    def tearDown(self) -> None:
        self.widget._player.shutdown()
        self._directory.cleanup()

    def _open(self, intervals: list[LabelInterval]) -> None:
        state = ProjectState()
        state.output_directory = str(self.root)
        state.intervals = intervals
        self.widget = PpgAlignmentWidget(state)

    def _wait_for_pool(self) -> None:
        deadline = time.time() + 60
        while time.time() < deadline:
            _app.processEvents()
            if self.widget._pool_worker is None and not self.widget._scan_timer.isActive():
                return
            time.sleep(0.01)
        self.fail("Pooling did not finish")

    def test_pools_every_estimated_segment(self) -> None:
        self._open([
            _write_segment(self.root, 0, "Resting", 0.0, rr_sec=0.95, offset=1.3, seed=1),
            _write_segment(self.root, 1, "Curl", 500.0, rr_sec=0.6, offset=1.3, seed=2),
        ])
        self.widget._pool_cb.setChecked(True)
        self._wait_for_pool()

        self.assertEqual([name for name, _key, _sum in self.widget._pool_segments],
                         ["0000_Resting", "0001_Curl"])
        self.widget._best_offset()
        self.assertAlmostEqual(self.widget._offset, 1.3, delta=0.04)
        text = self.widget._metrics_label.text()
        self.assertIn("Pooled 6 of 6 windows", text)
        self.assertIn("0001_Curl", text)
        record = self.widget._estimate_record()
        self.assertEqual(record["segment"], "pooled")
        self.assertEqual(record["segments"], ["0000_Resting", "0001_Curl"])
        self.assertFalse(self.widget._reference_warning.isVisibleTo(self.widget))

        # Changing the method recomputes; switching pooling off returns to the segment.
        self.widget._method_combo.setCurrentText("CHROM")
        self._wait_for_pool()
        self.assertEqual(len(self.widget._pool_windows), 4)
        self.widget._pool_cb.setChecked(False)
        self._wait_for_pool()
        self.assertNotIn("Pooled", self.widget._metrics_label.text())

    def test_pools_only_the_ticked_segments(self) -> None:
        self._open([
            _write_segment(self.root, 0, "Resting", 0.0, rr_sec=0.95, offset=1.3, seed=1),
            _write_segment(self.root, 1, "Curl", 500.0, rr_sec=0.6, offset=1.3, seed=2),
        ])
        items = [self.widget._segment_list.item(row) for row in range(2)]
        self.assertIsNone(items[0].data(Qt.ItemDataRole.CheckStateRole))  # no tick box until pooling
        self.widget._pool_cb.setChecked(True)
        self._wait_for_pool()
        self.assertEqual([item.checkState() for item in items], [Qt.CheckState.Checked] * 2)

        items[0].setCheckState(Qt.CheckState.Unchecked)
        self._wait_for_pool()
        self.assertEqual([name for name, _key, _sum in self.widget._pool_segments], ["0001_Curl"])
        self.assertIn("1 estimated segment(s) unticked", self.widget._metrics_label.text())
        self.assertEqual(self.widget._estimate_record()["segments"], ["0001_Curl"])

        self.widget._select_pooled(False)
        self._wait_for_pool()
        self.assertEqual(self.widget._pool_segments, [])
        self.assertIn("No segment is pooled", self.widget._metrics_label.text())

        # The selection is kept while pooling is off and when the list is refreshed.
        self.widget._pool_cb.setChecked(False)
        self.assertIsNone(items[0].data(Qt.ItemDataRole.CheckStateRole))
        self.widget._select_pooled(True)
        self.widget._pool_cb.setChecked(True)
        self.widget._segment_list.item(0).setCheckState(Qt.CheckState.Unchecked)
        self.widget.refresh_from_state()
        self._wait_for_pool()
        self.assertEqual(self.widget._segment_list.item(0).checkState(), Qt.CheckState.Unchecked)
        self.assertEqual([name for name, _key, _sum in self.widget._pool_segments], ["0001_Curl"])

    def test_right_click_ticks_the_segments_of_a_label(self) -> None:
        self._open([
            _write_segment(self.root, 0, "Resting", 0.0, rr_sec=0.95, offset=1.3, seed=1),
            _write_segment(self.root, 1, "Curl", 500.0, rr_sec=0.6, offset=1.3, seed=2),
            _write_segment(self.root, 2, "Resting", 1000.0, rr_sec=0.8, offset=1.3, seed=3),
        ])
        segment_list = self.widget._segment_list
        self.widget._pool_cb.setChecked(True)
        self._wait_for_pool()

        def right_click(row: int, choice: str) -> None:
            def choose(menu, *_args):
                return next(action for action in menu.actions() if action.text().startswith(choice))
            with mock.patch.object(segment_list, "itemAt", return_value=segment_list.item(row)), \
                    mock.patch.object(QMenu, "exec", new=choose):
                self.widget._segment_menu(QPoint(0, 0))
            self._wait_for_pool()

        def ticked() -> list[bool]:
            return [segment_list.item(row).checkState() == Qt.CheckState.Checked for row in range(3)]

        right_click(2, "Untick all “Resting” segments (2)")
        self.assertEqual(ticked(), [False, True, False])
        self.assertEqual([name for name, _key, _sum in self.widget._pool_segments], ["0001_Curl"])
        right_click(0, "Tick all “Resting”")
        self.assertEqual(ticked(), [True, True, True])
        right_click(0, "Tick only “Resting”")
        self.assertEqual(ticked(), [True, False, True])
        self.assertEqual([name for name, _key, _sum in self.widget._pool_segments],
                         ["0000_Resting", "0002_Resting"])

    def test_shows_every_window_when_none_passes_the_heart_rate_check(self) -> None:
        self._open([_write_segment(self.root, 0, "Resting", 0.0, rr_sec=0.8, offset=0.5, seed=3,
                                   pulse_speed=1.25)])
        self.widget._pool_cb.setChecked(True)
        self._wait_for_pool()

        self.assertTrue(self.widget._pool_fallback)
        self.assertTrue(self.widget._scan.valid)  # the graph is not empty
        text = self.widget._metrics_label.text()
        self.assertIn("UNRELIABLE", text)
        self.assertIn("HR video/PPG (bpm):", text)
        self.assertIsNone(self.widget._estimate_record()["max_hr_difference_bpm"])

        # A larger tolerance accepts the windows without scanning them again.
        self.widget._pool_hr_spin.setValue(60)
        self._wait_for_pool()
        self.assertFalse(self.widget._pool_fallback)
        self.assertIn("Pooled 3 of 3 windows", self.widget._metrics_label.text())
        self.assertEqual(self.widget._estimate_record()["max_hr_difference_bpm"], 60)
        self.assertEqual(len(self.widget._pool_windows), 1)

    def test_warns_about_a_drifting_synthetic_ppg(self) -> None:
        self._open([_write_segment(self.root, 0, "Resting", 0.0, rr_sec=0.8, offset=0.0, stretch=1.0006)])
        self.assertTrue(self.widget._reference_warning.isVisibleTo(self.widget))
        self.assertIn("Regenerate it in Synthetic PPG", self.widget._reference_warning.text())


if __name__ == "__main__":
    unittest.main()
