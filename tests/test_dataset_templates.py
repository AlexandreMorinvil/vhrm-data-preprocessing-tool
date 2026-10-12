import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

from app import dataset_templates
from app.dataset_templates import (
    CameraTemplate,
    DatasetTemplate,
    PreprocessingOptions,
    TemplateStore,
    apply_labels,
    describe,
    rename_label,
)
from app.state import LabelInterval, ProjectState


def _template(name="Gym", **parts):
    defaults = dict(
        cameras=[CameraTemplate("front"), CameraTemplate("side", False, 12.5)],
        labels=["Rest", "Squat"],
        preprocessing=PreprocessingOptions(30.0, True, False),
    )
    defaults.update(parts)
    return DatasetTemplate(name=name, **defaults)


class _IsolatedStore(unittest.TestCase):
    """Point the shared template store at a throw-away settings file."""

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.settings_path = str(Path(directory.name) / "settings.ini")
        self.store = self.new_store()
        previous = dataset_templates._store
        dataset_templates._store = self.store
        self.addCleanup(setattr, dataset_templates, "_store", previous)

    def new_store(self) -> TemplateStore:
        return TemplateStore(QSettings(self.settings_path, QSettings.Format.IniFormat))


class TemplateStoreTests(_IsolatedStore):
    def test_round_trip_and_default(self) -> None:
        self.store.save(_template())
        self.store.set_default("Gym")
        reloaded = self.new_store()
        self.assertEqual(reloaded.names(), ["Gym"])
        self.assertEqual(reloaded.get("Gym"), _template())
        self.assertEqual(reloaded.default(), _template())

    def test_saving_keeps_parts_left_out(self) -> None:
        self.store.save(_template())
        self.store.save(DatasetTemplate("Gym", labels=["Walk"]))
        saved = self.store.get("Gym")
        self.assertEqual(saved.labels, ["Walk"])
        self.assertEqual(saved.cameras, _template().cameras)
        self.assertEqual(saved.preprocessing, _template().preprocessing)

    def test_partial_template(self) -> None:
        self.store.save(DatasetTemplate("Labels only", labels=["A", " A ", "", "B"]))
        template = self.new_store().get("Labels only")
        self.assertEqual(template.labels, ["A", "B"])
        self.assertIsNone(template.cameras)
        self.assertEqual(template.parts(), ["labels"])

    def test_rejects_unnamed_or_empty(self) -> None:
        with self.assertRaises(ValueError):
            self.store.save(_template(name="  "))
        with self.assertRaises(ValueError):
            self.store.save(DatasetTemplate("Nothing"))

    def test_rename_and_delete_follow_default(self) -> None:
        self.store.save(_template())
        self.store.save(_template(name="Lab"))
        self.store.set_default("Gym")
        with self.assertRaises(ValueError):
            self.store.rename("Gym", "Lab")
        self.store.rename("Gym", "Gym v2")
        self.assertEqual(self.store.default_name, "Gym v2")
        self.store.delete("Gym v2")
        self.assertEqual(self.store.default_name, "")
        self.assertEqual(self.new_store().names(), ["Lab"])

    def test_invalid_stored_data_is_ignored(self) -> None:
        settings = QSettings(self.settings_path, QSettings.Format.IniFormat)
        settings.setValue("dataset_templates/data", "{not json")
        settings.sync()
        self.assertEqual(self.new_store().names(), [])

    def test_describe_mentions_every_part(self) -> None:
        text = describe(_template())
        self.assertIn("front, side", text)
        self.assertIn("not limiting size: side", text)
        self.assertIn("Rest, Squat", text)
        self.assertIn("30 s", text)


class ApplyLabelsTests(unittest.TestCase):
    def test_replaces_library_but_keeps_labels_in_use(self) -> None:
        state = ProjectState(intervals=[LabelInterval("Old", 0.0, 1.0), LabelInterval("Rest", 1.0, 2.0)])
        kept = apply_labels(state, ["Rest", "Squat", "Rest"])
        self.assertEqual(state.labels_library, ["Rest", "Squat", "Old"])
        self.assertEqual(kept, ["Old"])


class RenameLabelTests(unittest.TestCase):
    def _state(self):
        return ProjectState(labels_library=["Rest", "Squat"], intervals=[
            LabelInterval("Rest", 0.0, 1.0), LabelInterval("Squat", 1.0, 2.0), LabelInterval("Rest", 2.0, 3.0),
        ])

    def test_renames_library_entry_in_place_and_every_interval(self) -> None:
        state = self._state()
        self.assertEqual(rename_label(state, "Rest", "Baseline"), 2)
        self.assertEqual(state.labels_library, ["Baseline", "Squat"])
        self.assertEqual([iv.label for iv in state.intervals], ["Baseline", "Squat", "Baseline"])

    def test_renaming_onto_existing_label_merges(self) -> None:
        state = self._state()
        self.assertEqual(rename_label(state, "Squat", "Rest"), 1)
        self.assertEqual(state.labels_library, ["Rest"])
        self.assertEqual({iv.label for iv in state.intervals}, {"Rest"})

    def test_label_missing_from_library_is_added(self) -> None:
        state = self._state()
        state.intervals.append(LabelInterval("Typo", 3.0, 4.0))
        self.assertEqual(rename_label(state, "Typo", "Lunge"), 1)
        self.assertEqual(state.labels_library, ["Rest", "Squat", "Lunge"])


class TemplateScreenTests(_IsolatedStore):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _preprocessing(self, state=None):
        from app.modes.mode1_preprocessing import Mode1Widget

        widget = Mode1Widget(state or ProjectState())
        self.addCleanup(widget.close)
        return widget

    def test_preprocessing_apply_and_capture(self) -> None:
        widget = self._preprocessing()
        widget._apply_template(_template(cameras=[
            CameraTemplate("front"), CameraTemplate("side", False, 12.5), CameraTemplate("top"),
        ]))
        self.assertEqual(widget._num_cameras_spin.value(), 3)
        self.assertEqual([g.label for g in widget._camera_groups], ["front", "side", "top"])
        self.assertFalse(widget._camera_groups[1].limits_common_duration)
        self.assertEqual(widget._camera_groups[1].sync_audio_start_offset_sec, 12.5)
        self.assertEqual([p.label_text for p in widget._player.previews], ["front", "side", "top"])
        self.assertEqual(widget._sync_duration_spin.value(), 30.0)
        self.assertTrue(widget._sync_pairwise_refinement_cb.isChecked())
        self.assertFalse(widget._delete_intermediates_cb.isChecked())
        self.assertEqual(widget.state.labels_library, ["Rest", "Squat"])

        captured = widget._capture_template()
        self.assertEqual([c.label for c in captured.cameras], ["front", "side", "top"])
        self.assertEqual(captured.labels, ["Rest", "Squat"])
        self.assertEqual(captured.preprocessing, PreprocessingOptions(30.0, True, False))

    def test_default_template_fills_new_projects_only(self) -> None:
        self.store.save(_template())
        self.store.set_default("Gym")
        fresh = self._preprocessing()
        self.assertEqual([g.label for g in fresh._camera_groups], ["front", "side"])
        # Labels are set when the project is created, not by this screen.
        self.assertEqual(fresh.state.labels_library, ProjectState().labels_library)

        opened = self._preprocessing(ProjectState(project_path="study.vrt", num_cameras=4))
        self.assertEqual(opened._num_cameras_spin.value(), 4)
        self.assertEqual([g.label_text for g in opened._camera_groups], ["", "", "", ""])

    def test_labelling_applies_labels_with_undo(self) -> None:
        from app.modes.mode3_labelling import Mode3Widget

        state = ProjectState(intervals=[LabelInterval("Baseline", 10.0, 20.0)])
        widget = Mode3Widget(state)
        self.addCleanup(widget.close)
        before = list(state.labels_library)
        self.store.save(_template())
        widget._template_bar.select("Gym")
        widget._template_bar.apply_selected()
        self.assertEqual(state.labels_library, ["Rest", "Squat", "Baseline"])
        self.assertEqual(widget._label_combo.count(), 3)
        widget._undo_action()
        self.assertEqual(state.labels_library, before)

    def test_save_dialog_stores_checked_parts(self) -> None:
        from app.widgets.template_bar import TemplateSaveDialog

        self.store.save(DatasetTemplate("Gym", labels=["Walk"]))
        dialog = TemplateSaveDialog(_template(name=""), self.store, "Gym")
        self.addCleanup(dialog.close)
        self.assertIn("already exists", dialog._note.text())
        dialog._parts["labels"].setChecked(False)
        self.store.save(dialog.template())
        saved = self.store.get("Gym")
        self.assertEqual(saved.labels, ["Walk"])
        self.assertEqual(saved.cameras, _template().cameras)

    def test_new_project_starts_with_default_labels(self) -> None:
        from main import MainWindow

        self.assertEqual(MainWindow._fresh_state().labels_library, ProjectState().labels_library)
        self.store.save(_template())
        self.store.set_default("Gym")
        state = MainWindow._fresh_state(blur_faces=True)
        self.assertEqual(state.labels_library, ["Rest", "Squat"])
        self.assertTrue(state.blur_faces)

    def test_labelling_capture_uses_project_cameras(self) -> None:
        from app.modes.mode3_labelling import Mode3Widget
        from app.state import VideoTrack

        state = ProjectState(tracks=[VideoTrack(camera_label="front"), VideoTrack(camera_label="side")])
        widget = Mode3Widget(state)
        self.addCleanup(widget.close)
        captured = widget._capture_template()
        self.assertEqual([c.label for c in captured.cameras], ["front", "side"])
        self.assertEqual(captured.labels, state.labels_library)
        self.assertIsNone(captured.preprocessing)


if __name__ == "__main__":
    unittest.main()
