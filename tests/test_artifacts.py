import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app import artifacts
from app.artifacts import (
    ALIGNMENT_FILES,
    ARTIFACTS_MANIFEST,
    FULL_VIDEOS,
    INTERMEDIATE_VIDEOS,
    ArtifactFile,
    default_artifacts_name,
    delete_artifacts,
    destination_problem,
    existing_destinations,
    find_artifacts,
    move_artifacts,
    read_manifest,
    restore_artifacts,
    restore_checks,
    scan_artifacts_folder,
)
from app.state import ProjectState, VideoTrack


def _write(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class ArtifactTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.dataset = self.base / "jzc11"
        self.segment_clip = _write(self.dataset / "labelled_segments" / "0000_Baseline" / "cam1_Front.mp4")
        _write(self.dataset / "labelled_segments" / "manifest.csv", b"label,start_sec,end_sec,folder\n")
        self.final = _write(self.dataset / "cam1_Front_final.mp4", b"final video")
        self.concat = _write(self.dataset / "cam2_Back_concat.mp4", b"concat")
        self.cache = _write(self.dataset / "ppg_alignment" / "cache" / "0000_Baseline__Front_skin.npz", b"npz")
        self.record = _write(self.dataset / "ppg_alignment" / "alignment.json", b"{}")
        self.mosaic = _write(self.dataset / "mosaic.mp4")
        self.signal = _write(self.dataset / "polar_1.csv")

    def found(self, state=None):
        return find_artifacts(self.dataset, state)


class FindArtifactsTests(ArtifactTestCase):
    def test_categorises_root_videos_and_alignment_files(self):
        found = self.found()
        self.assertEqual([item.path for item in found[FULL_VIDEOS]], [self.final])
        self.assertEqual([item.path for item in found[INTERMEDIATE_VIDEOS]], [self.concat])
        self.assertEqual(
            sorted(item.relative for item in found[ALIGNMENT_FILES]),
            sorted([Path("ppg_alignment/alignment.json"),
                    Path("ppg_alignment/cache/0000_Baseline__Front_skin.npz")]),
        )
        self.assertEqual(found[FULL_VIDEOS][0].size, len(b"final video"))

    def test_never_lists_segments_signals_or_other_videos(self):
        listed = {item.path for files in self.found().values() for item in files}
        self.assertNotIn(self.segment_clip, listed)
        self.assertNotIn(self.mosaic, listed)
        self.assertNotIn(self.signal, listed)

    def test_includes_final_video_recorded_by_project_under_another_name(self):
        renamed = _write(self.dataset / "front camera.mp4")
        state = ProjectState(output_directory=str(self.dataset),
                             tracks=[VideoTrack(final_output_path=str(renamed))])
        self.assertIn(renamed, [item.path for item in self.found(state)[FULL_VIDEOS]])

    def test_default_name_uses_dataset_folder(self):
        self.assertEqual(default_artifacts_name(self.dataset), "jzc11_artifacts")

    def test_missing_folder_has_no_artifacts(self):
        found = find_artifacts(self.base / "missing")
        self.assertTrue(all(not files for files in found.values()))


class MoveArtifactsTests(ArtifactTestCase):
    def test_moves_with_same_structure_and_removes_emptied_folders(self):
        destination = self.base / "jzc11_artifacts"
        found = self.found()
        files = found[FULL_VIDEOS] + found[ALIGNMENT_FILES]

        result = move_artifacts(files, self.dataset, destination)

        self.assertEqual(len(result.done), 3)
        self.assertFalse(result.failures)
        self.assertEqual((destination / "cam1_Front_final.mp4").read_bytes(), b"final video")
        self.assertTrue((destination / "ppg_alignment" / "alignment.json").exists())
        self.assertTrue((destination / "ppg_alignment" / "cache" / "0000_Baseline__Front_skin.npz").exists())
        self.assertFalse(self.final.exists())
        self.assertFalse((self.dataset / "ppg_alignment").exists())
        self.assertTrue(self.segment_clip.exists())
        self.assertTrue(self.concat.exists())

    def test_copies_across_volumes(self):
        destination = self.base / "artifacts"
        progress = []
        with patch.object(artifacts, "_same_volume", return_value=False):
            result = move_artifacts(self.found()[FULL_VIDEOS], self.dataset, destination,
                                    progress=lambda done, total, name: progress.append((done, total)))
        self.assertEqual(len(result.done), 1)
        self.assertFalse(self.final.exists())
        self.assertEqual((destination / "cam1_Front_final.mp4").read_bytes(), b"final video")
        self.assertFalse(list(destination.glob("*.partial")))
        self.assertEqual(progress[-1][0], progress[-1][1])

    def test_cancelled_copy_keeps_source_and_leaves_no_partial_file(self):
        destination = self.base / "artifacts"
        with patch.object(artifacts, "_same_volume", return_value=False):
            result = move_artifacts(self.found()[FULL_VIDEOS], self.dataset, destination,
                                    is_cancelled=iter([False, True, True, True]).__next__)
        self.assertTrue(result.cancelled)
        self.assertFalse(result.done)
        self.assertTrue(self.final.exists())
        self.assertFalse(any(destination.rglob("*")))

    def test_existing_file_is_kept_unless_overwrite(self):
        destination = self.base / "artifacts"
        _write(destination / "cam1_Front_final.mp4", b"old")
        files = self.found()[FULL_VIDEOS]
        self.assertEqual(existing_destinations(files, destination), [destination / "cam1_Front_final.mp4"])

        result = move_artifacts(files, self.dataset, destination)
        self.assertEqual(len(result.failures), 1)
        self.assertTrue(self.final.exists())
        self.assertEqual((destination / "cam1_Front_final.mp4").read_bytes(), b"old")

        result = move_artifacts(files, self.dataset, destination, overwrite=True)
        self.assertEqual(len(result.done), 1)
        self.assertEqual((destination / "cam1_Front_final.mp4").read_bytes(), b"final video")

    def test_rejects_destination_inside_or_containing_dataset(self):
        self.assertIsNotNone(destination_problem(self.dataset, self.dataset / "jzc11_artifacts"))
        self.assertIsNotNone(destination_problem(self.dataset, self.base))
        self.assertIsNotNone(destination_problem(self.dataset, ""))
        self.assertIsNone(destination_problem(self.dataset, self.base / "jzc11_artifacts"))
        with self.assertRaises(ValueError):
            move_artifacts(self.found()[FULL_VIDEOS], self.dataset, self.dataset / "out")


class RestoreArtifactsTests(ArtifactTestCase):
    def setUp(self):
        super().setUp()
        self.artifacts = self.base / "jzc11_artifacts"

    def move_out(self, *categories):
        found = self.found()
        files = [item for category in categories for item in found[category]]
        result = move_artifacts(files, self.dataset, self.artifacts)
        self.assertEqual(len(result.done), len(files))
        return files

    def scan(self, state=None):
        scan = scan_artifacts_folder(self.artifacts, state)
        return scan, [item for files in scan.found.values() for item in files]

    def test_move_records_origin_and_sizes_in_manifest(self):
        self.move_out(FULL_VIDEOS)
        manifest = read_manifest(self.artifacts)
        self.assertEqual(Path(manifest["dataset"]), self.dataset)
        self.assertEqual(manifest["dataset_name"], "jzc11")
        self.assertEqual(manifest["files"]["cam1_Front_final.mp4"]["size"], len(b"final video"))
        self.assertEqual(manifest["files"]["cam1_Front_final.mp4"]["category"], FULL_VIDEOS)

    def test_round_trip_restores_everything_and_removes_artifacts_folder(self):
        before = {path.relative_to(self.dataset): path.read_bytes()
                  for path in self.dataset.rglob("*") if path.is_file()}
        self.move_out(FULL_VIDEOS, ALIGNMENT_FILES, INTERMEDIATE_VIDEOS)
        scan, files = self.scan()
        self.assertEqual(len(files), 4)
        self.assertEqual(restore_checks(self.artifacts, self.dataset, files, scan.manifest), ([], []))

        result = restore_artifacts(files, self.artifacts, self.dataset)

        self.assertEqual(len(result.done), 4)
        after = {path.relative_to(self.dataset): path.read_bytes()
                 for path in self.dataset.rglob("*") if path.is_file()}
        self.assertEqual(after, before)
        self.assertFalse(self.artifacts.exists())

    def test_partial_restore_keeps_the_rest_listed(self):
        self.move_out(FULL_VIDEOS, ALIGNMENT_FILES)
        scan, _files = self.scan()
        restore_artifacts(scan.found[FULL_VIDEOS], self.artifacts, self.dataset)
        self.assertTrue(self.final.exists())
        self.assertEqual(set(read_manifest(self.artifacts)["files"]),
                         {"ppg_alignment/alignment.json", "ppg_alignment/cache/0000_Baseline__Front_skin.npz"})

    def test_scan_ignores_unknown_and_partial_files_and_uses_manifest_categories(self):
        renamed = _write(self.dataset / "front camera.mp4", b"renamed")
        state = ProjectState(output_directory=str(self.dataset),
                             tracks=[VideoTrack(final_output_path=str(renamed))])
        move_artifacts(find_artifacts(self.dataset, state)[FULL_VIDEOS], self.dataset, self.artifacts)
        _write(self.artifacts / "notes.txt")
        _write(self.artifacts / "ppg_alignment" / "cache" / "x.npz.partial")
        scan, _files = self.scan()  # without the project: the manifest still knows the video
        self.assertEqual(sorted(item.relative.name for item in scan.found[FULL_VIDEOS]),
                         ["cam1_Front_final.mp4", "front camera.mp4"])
        self.assertEqual(sorted(scan.ignored),
                         sorted([Path("notes.txt"), Path("ppg_alignment/cache/x.npz.partial")]))
        self.assertNotIn(Path(ARTIFACTS_MANIFEST), scan.ignored)

    def test_scan_without_manifest_recognises_artifact_layout(self):
        _write(self.artifacts / "cam1_Front_final.mp4")
        _write(self.artifacts / "ppg_alignment" / "alignment.json")
        _write(self.artifacts / "labelled_segments" / "0000_Baseline" / "cam1_Front.mp4")
        scan, files = self.scan()
        self.assertIsNone(scan.manifest)
        self.assertEqual(len(files), 2)
        self.assertIn(Path("labelled_segments/0000_Baseline/cam1_Front.mp4"), scan.ignored)
        errors, warnings = restore_checks(self.artifacts, self.dataset, files, None)
        self.assertEqual(errors, [])
        self.assertTrue(any(ARTIFACTS_MANIFEST in message for message in warnings))

    def test_warns_when_artifacts_come_from_another_dataset(self):
        self.move_out(FULL_VIDEOS)
        other = self.base / "jzc10"
        other.mkdir()
        scan, files = self.scan()
        errors, warnings = restore_checks(self.artifacts, other, files, scan.manifest)
        self.assertEqual(errors, [])
        self.assertTrue(any("'jzc11'" in message for message in warnings))

    def test_warns_about_changed_missing_and_foreign_files(self):
        self.move_out(FULL_VIDEOS, ALIGNMENT_FILES)
        (self.artifacts / "cam1_Front_final.mp4").write_bytes(b"edited since")
        (self.artifacts / "ppg_alignment" / "alignment.json").unlink()
        state = ProjectState(tracks=[VideoTrack(final_output_path=str(self.dataset / "cam1_Back_final.mp4"))])
        scan, files = self.scan()
        _errors, warnings = restore_checks(self.artifacts, self.dataset, files, scan.manifest, state)
        text = "\n".join(warnings)
        self.assertIn("changed size", text)
        self.assertIn("no longer in the artifacts folder", text)
        self.assertIn("Not videos of the open project", text)

    def test_errors_for_missing_overlapping_or_full_target(self):
        self.move_out(FULL_VIDEOS)
        scan, files = self.scan()
        errors, _ = restore_checks(self.base / "missing", self.dataset, files, None)
        self.assertTrue(errors)
        nested = self.dataset / "nested_artifacts"
        nested.mkdir()
        errors, _ = restore_checks(nested, self.dataset, files, None)
        self.assertTrue(errors)
        errors, _ = restore_checks(self.artifacts, self.dataset, [], scan.manifest)
        self.assertTrue(errors)
        with patch.object(artifacts, "_same_volume", return_value=False), \
                patch.object(artifacts.shutil, "disk_usage", return_value=type("Usage", (), {"free": 1})()):
            errors, _ = restore_checks(self.artifacts, self.dataset, files, scan.manifest)
        self.assertTrue(any("free space" in message for message in errors))

    def test_conflicting_dataset_file_is_kept_unless_overwrite(self):
        self.move_out(FULL_VIDEOS)
        _write(self.final, b"regenerated")
        scan, files = self.scan()
        self.assertEqual(existing_destinations(files, self.dataset), [self.final])
        result = restore_artifacts(files, self.artifacts, self.dataset)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(self.final.read_bytes(), b"regenerated")
        result = restore_artifacts(files, self.artifacts, self.dataset, overwrite=True)
        self.assertEqual(len(result.done), 1)
        self.assertEqual(self.final.read_bytes(), b"final video")

    def test_restores_across_volumes(self):
        self.move_out(FULL_VIDEOS)
        _scan, files = self.scan()
        with patch.object(artifacts, "_same_volume", return_value=False):
            result = restore_artifacts(files, self.artifacts, self.dataset)
        self.assertEqual(len(result.done), 1)
        self.assertEqual(self.final.read_bytes(), b"final video")
        self.assertFalse(self.artifacts.exists())

    def test_refuses_to_restore_into_labelled_segments(self):
        crafted = _write(self.artifacts / "labelled_segments" / "0000_Baseline" / "cam1_Front.mp4", b"bad")
        item = ArtifactFile(crafted, crafted.relative_to(self.artifacts), 3, FULL_VIDEOS)
        result = restore_artifacts([item], self.artifacts, self.dataset)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(self.segment_clip.read_bytes(), b"x")
        self.assertTrue(crafted.exists())

    def test_rejects_artifacts_folder_inside_dataset(self):
        with self.assertRaises(ValueError):
            restore_artifacts([], self.dataset / "ppg_alignment", self.dataset)


class DeleteArtifactsTests(ArtifactTestCase):
    def test_deletes_selected_files_and_emptied_folders(self):
        result = delete_artifacts(self.found()[ALIGNMENT_FILES], self.dataset)
        self.assertEqual(len(result.done), 2)
        self.assertFalse((self.dataset / "ppg_alignment").exists())
        self.assertTrue(self.final.exists())
        self.assertTrue(self.dataset.exists())

    def test_refuses_labelled_segment_and_external_files(self):
        outside = _write(self.base / "elsewhere.mp4")
        files = [
            ArtifactFile(self.segment_clip, self.segment_clip.relative_to(self.dataset), 1, FULL_VIDEOS),
            ArtifactFile(outside, Path("elsewhere.mp4"), 1, FULL_VIDEOS),
        ]
        result = delete_artifacts(files, self.dataset)
        self.assertEqual(len(result.failures), 2)
        self.assertTrue(self.segment_clip.exists())
        self.assertTrue(outside.exists())


class ArtifactsWidgetTests(ArtifactTestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        from PyQt6.QtCore import QSettings
        settings = QSettings("VideoResearchTool", "VRT")
        saved = {key: settings.value(key) for key in ("artifacts/categories", "artifacts/action",
                                                       "artifacts/location")}

        def restore():
            for key, value in saved.items():
                if value is None:
                    settings.remove(key)
                else:
                    settings.setValue(key, value)

        self.addCleanup(restore)
        for key in saved:
            settings.remove(key)

    def widget(self):
        from app.modes.mode7_artifacts import ArtifactsWidget
        widget = ArtifactsWidget(ProjectState(output_directory=str(self.dataset)))
        self.addCleanup(widget.deleteLater)
        return widget

    def test_defaults_to_full_videos_moved_next_to_dataset(self):
        widget = self.widget()
        self.assertEqual(widget._destination(), self.base / "jzc11_artifacts")
        self.assertEqual([item.path for item in widget._selected_files()], [self.final])
        self.assertTrue(widget._move_radio.isChecked())
        self.assertIn("1 file(s)", widget._run_btn.text())
        self.assertEqual(widget._tree.topLevelItemCount(), 1)

    def test_selection_name_and_location_drive_the_destination(self):
        widget = self.widget()
        widget._category_checks[ALIGNMENT_FILES].setChecked(True)
        self.assertEqual(len(widget._selected_files()), 3)
        self.assertEqual(widget._tree.topLevelItemCount(), 2)
        widget._name_edit.setText("custom")
        widget._location_edit.setText(str(self.base / "store"))
        widget._on_location_edited()
        self.assertEqual(widget._destination(), self.base / "store" / "custom")
        child = widget._tree.topLevelItem(0).child(0)
        self.assertEqual(child.text(2), str(Path("custom") / "cam1_Front_final.mp4"))
        self.assertEqual(child.toolTip(2), str(self.base / "store" / "custom" / "cam1_Front_final.mp4"))

    def test_job_moves_selected_files(self):
        widget = self.widget()
        worker = type("Worker", (), {"is_cancelled": False, "progress": type(
            "Signal", (), {"emit": lambda *args: None})()})()
        files = widget._selected_files()
        widget._cleanup_job(files=files, dataset=self.dataset, destination=widget._destination(),
                            action="move", overwrite=False, segments_dir=widget._segments_dir(self.dataset),
                            worker=worker)
        self.assertEqual(len(widget._result.done), 1)
        self.assertTrue((self.base / "jzc11_artifacts" / "cam1_Front_final.mp4").exists())
        widget._rescan()
        check = widget._category_checks[FULL_VIDEOS]
        self.assertIn("none found", check.text())
        self.assertTrue(check.isEnabled())
        check.setChecked(False)
        self.assertFalse(check.isChecked())

    def test_restore_mode_lists_the_artifacts_folder_and_moves_back(self):
        artifacts_dir = self.base / "jzc11_artifacts"
        move_artifacts(self.found()[FULL_VIDEOS], self.dataset, artifacts_dir)
        _write(artifacts_dir / "notes.txt")
        widget = self.widget()
        self.assertIn("none found", widget._category_checks[FULL_VIDEOS].text())

        widget._restore_radio.setChecked(True)

        self.assertIn("1 file(s)", widget._category_checks[FULL_VIDEOS].text())
        self.assertEqual([item.relative for item in widget._selected_files()], [Path("cam1_Front_final.mp4")])
        self.assertIn("Restore 1 file(s)", widget._run_btn.text())
        self.assertEqual(widget._destination_caption.text(), "Artifacts folder:")
        self.assertEqual(widget._tree.topLevelItemCount(), 2)  # the video, and notes.txt left in place
        self.assertEqual(widget._tree.topLevelItem(0).child(0).text(2), str(Path("jzc11") / "cam1_Front_final.mp4"))
        self.assertNotIn("\u2716", widget._summary.text())

        worker = type("Worker", (), {"is_cancelled": False, "progress": type(
            "Signal", (), {"emit": lambda *args: None})()})()
        widget._cleanup_job(files=widget._selected_files(), dataset=self.dataset, destination=artifacts_dir,
                            action="restore", overwrite=False, segments_dir=widget._segments_dir(self.dataset),
                            worker=worker)
        self.assertEqual(len(widget._result.done), 1)
        self.assertEqual(self.final.read_bytes(), b"final video")
        self.assertTrue((artifacts_dir / "notes.txt").exists())

    def test_preview_flags_files_that_already_exist_at_the_target(self):
        artifacts_dir = self.base / "jzc11_artifacts"
        move_artifacts(self.found()[FULL_VIDEOS], self.dataset, artifacts_dir)
        _write(self.final, b"regenerated")
        widget = self.widget()
        widget._restore_radio.setChecked(True)
        self.assertTrue(widget._tree.topLevelItem(0).child(0).text(2).endswith("already exists"))
        self.assertIn("1 file(s) already exist in the dataset", widget._summary.text())

    def test_restore_mode_is_not_remembered(self):
        from PyQt6.QtCore import QSettings
        widget = self.widget()
        widget._restore_radio.setChecked(True)
        self.assertEqual(QSettings("VideoResearchTool", "VRT").value("artifacts/action"), "move")
        self.assertTrue(self.widget()._move_radio.isChecked())


class ArtifactsTabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_artifacts_tab_is_on_the_right_bar_and_bars_switch_highlight(self):
        from app.face_privacy import get_privacy_settings, set_privacy_settings
        from app.modes.mode7_artifacts import ArtifactsWidget
        from main import MainWindow

        # MainWindow loads the stored privacy settings into a module global.
        self.addCleanup(set_privacy_settings, get_privacy_settings())
        with patch("main.MainWindow._on_tab_changed") as changed:
            window = MainWindow()
            self.addCleanup(window.deleteLater)
            self.assertEqual(window._right_tabs.count(), 1)
            self.assertEqual(window._right_tabs.tabText(0), "7. Artifacts")

            window._right_tabs.tabBarClicked.emit(0)
            changed.assert_called_with(6)
            self.assertTrue(window._mode_tabs.property("inactive"))
            self.assertFalse(window._right_tabs.property("inactive"))

            # The left bar's current tab is clicked again: it must be reselected.
            window._mode_tabs.tabBarClicked.emit(window._mode_tabs.currentIndex())
            changed.assert_called_with(0)
            self.assertFalse(window._mode_tabs.property("inactive"))
            self.assertTrue(window._right_tabs.property("inactive"))

        window._mode_widgets[6] = None
        window._ensure_mode_widget(6)
        self.assertIsInstance(window._mode_widgets[6], ArtifactsWidget)


if __name__ == "__main__":
    unittest.main()
