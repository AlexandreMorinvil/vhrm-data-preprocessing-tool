import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QMessageBox

from app import dataset_templates
from app.dataset_templates import TemplateStore
from app.state import LabelInterval, ProjectState, VideoTrack, generate_sidecar, is_other_dataset, load_sidecar

_YES = QMessageBox.StandardButton.Yes
_CANCEL = QMessageBox.StandardButton.Cancel


class StartOverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from app.face_privacy import get_privacy_settings, set_privacy_settings
        from main import MainWindow

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        # No default template from the user's own settings.
        store = TemplateStore(QSettings(str(self.root / "settings.ini"), QSettings.Format.IniFormat))
        self.addCleanup(setattr, dataset_templates, "_store", dataset_templates._store)
        dataset_templates._store = store
        # MainWindow loads the stored privacy settings into a module global.
        self.addCleanup(set_privacy_settings, get_privacy_settings())
        self.window = MainWindow()
        self.addCleanup(self.window.deleteLater)
        self.dataset_a = self.make_dataset("a", "Front")
        self.dataset_b = self.make_dataset("b", "Side")

    def make_dataset(self, name: str, camera: str) -> str:
        folder = self.root / name
        folder.mkdir()
        state = ProjectState(output_directory=str(folder),
                             tracks=[VideoTrack(camera_label=camera, duration_sec=10.0)])
        return generate_sidecar(state)

    def load_a_with_interval(self, folder: str = "") -> None:
        self.window._load_dataset(self.dataset_a)
        self.window.state.intervals.append(LabelInterval("Rest", 0.0, 5.0, folder=folder))

    def test_other_dataset_is_recognised_by_its_folder(self):
        state = ProjectState()
        self.assertFalse(is_other_dataset(state, self.dataset_b))
        load_sidecar(self.dataset_a, state)
        self.assertFalse(is_other_dataset(state, self.dataset_a))
        self.assertTrue(is_other_dataset(state, self.dataset_b))
        self.assertTrue(is_other_dataset(ProjectState(intervals=[LabelInterval("Rest", 0, 1)]),
                                         self.dataset_b))

    def test_new_project_clears_every_section(self):
        self.load_a_with_interval()
        self.window._select_mode(1)
        old_section = self.window._mode_widgets[1]
        with patch("main.QMessageBox.question", return_value=_YES) as question:
            self.window._new_project()
        question.assert_called_once()
        self.assertIn("1 interval(s)", question.call_args.args[2])
        self.assertEqual(self.window.state.tracks, [])
        self.assertEqual(self.window.state.intervals, [])
        self.assertEqual(self.window.state.output_directory, "")
        self.assertIsNot(self.window._mode_widgets[1], old_section)
        self.assertIs(self.window._mode_widgets[1].state, self.window.state)
        self.assertEqual(sum(w is not None for w in self.window._mode_widgets), 1)

    def test_cancelling_new_project_keeps_everything(self):
        self.load_a_with_interval()
        state = self.window.state
        with patch("main.QMessageBox.question", return_value=_CANCEL):
            self.window._new_project()
        self.assertIs(self.window.state, state)
        self.assertEqual(len(state.intervals), 1)

    def test_new_project_on_empty_state_does_not_ask(self):
        with patch("main.QMessageBox.question") as question:
            self.window._new_project()
        question.assert_not_called()

    def test_loading_another_dataset_in_a_section_starts_over(self):
        self.load_a_with_interval(folder=str(self.root / "a" / "labelled_segments" / "0000_Rest"))
        self.window._select_mode(1)
        section = self.window._mode_widgets[1]
        with patch("app.modes.mode2_signal_sync.QFileDialog.getOpenFileName",
                   return_value=(self.dataset_b, "")), \
                patch("main.QMessageBox.question") as question:
            section._load_from_meta()
        question.assert_not_called()  # the interval was exported: nothing is lost
        state = self.window.state
        self.assertEqual([t.camera_label for t in state.tracks], ["Side"])
        self.assertEqual(state.intervals, [])
        self.assertEqual(Path(state.output_directory), self.root / "b")
        self.assertIsNot(self.window._mode_widgets[1], section)
        self.assertIs(self.window._mode_widgets[1].state, state)

    def test_unexported_intervals_are_confirmed_before_switching_dataset(self):
        self.load_a_with_interval()
        with patch("main.QMessageBox.question", return_value=_CANCEL):
            self.window._load_dataset(self.dataset_b)
        self.assertEqual([t.camera_label for t in self.window.state.tracks], ["Front"])
        self.assertEqual(len(self.window.state.intervals), 1)

    def test_reloading_the_same_dataset_keeps_intervals(self):
        self.load_a_with_interval()
        state = self.window.state
        self.window._select_mode(1)
        section = self.window._mode_widgets[1]
        with patch("app.modes.mode2_signal_sync.QFileDialog.getOpenFileName",
                   return_value=(self.dataset_a, "")):
            section._load_from_meta()
        self.assertIs(self.window.state, state)
        self.assertIs(self.window._mode_widgets[1], section)
        self.assertEqual(len(state.intervals), 1)


if __name__ == "__main__":
    unittest.main()
