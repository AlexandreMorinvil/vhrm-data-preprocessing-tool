import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from app.modes.mode5_ppg_alignment import PpgAlignmentWidget
from app.ppg_alignment import cache_key, cache_paths
from app.state import LabelInterval, ProjectState

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


if __name__ == "__main__":
    unittest.main()
