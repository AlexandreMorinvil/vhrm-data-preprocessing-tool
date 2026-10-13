import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QMessageBox

from app.modes.mode5_ppg_alignment import PpgAlignmentWidget
from app.ppg_alignment import cache_key, cache_paths
from app.signal_offset import apply_signal_offset, read_segment_meta
from app.state import LabelInterval, ProjectState
from tests import test_signal_offset

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


if __name__ == "__main__":
    unittest.main()
