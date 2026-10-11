import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.camera_rename import CameraRenameCancelled, rename_cameras, validate_camera_labels
from app.state import LabelInterval, ProjectState, VideoTrack, generate_sidecar, load_sidecar


class CameraRenameTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.video = self.root / "cam1_Wrong_final.mp4"
        self.video.write_bytes(b"original video bytes")
        self.source = self.root / "DJI_original.mp4"
        self.source.write_bytes(b"source recording")
        self.state = ProjectState(
            output_directory=str(self.root),
            tracks=[VideoTrack(camera_label="Wrong", final_output_path=str(self.video),
                               segment_paths=[str(self.source)], fps=30, duration_sec=12)],
            num_cameras=1,
        )
        self.sidecar = Path(generate_sidecar(self.state))

    def add_segment(self, folder=None):
        folder = folder or self.root / "labelled_segments" / "0000_Wrong"
        folder.mkdir(parents=True)
        (folder / "cam1_Wrong.mp4").write_bytes(b"clip bytes")
        (folder / "polar_1.csv").write_text("sensor data", encoding="utf-8")
        meta = {"label": "Wrong", "custom": {"preserve": True}, "cameras": [
            {"label": "Wrong", "video_file": "cam1_Wrong.mp4", "fps": 30}
        ], "hr_files": ["polar_1.csv"]}
        (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        self.state.intervals = [LabelInterval("Wrong", 0, 12, folder=str(folder))]
        return folder

    def test_renames_videos_metadata_project_and_live_track(self):
        folder = self.add_segment()
        project = self.root / "session.vrt"
        self.state.save(project)
        raw = json.loads(self.sidecar.read_text())
        raw["extra_field"] = {"preserved": "Wrong"}
        self.sidecar.write_text(json.dumps(raw), encoding="utf-8")
        track = self.state.tracks[0]

        self.assertEqual(rename_cameras(self.state, ["Front"]), 2)

        new_video = self.root / "cam1_Front_final.mp4"
        self.assertEqual(new_video.read_bytes(), b"original video bytes")
        self.assertFalse(self.video.exists())
        self.assertEqual((folder / "cam1_Front.mp4").read_bytes(), b"clip bytes")
        self.assertFalse((folder / "cam1_Wrong.mp4").exists())
        self.assertEqual(self.source.read_bytes(), b"source recording")
        self.assertEqual((folder / "polar_1.csv").read_text(), "sensor data")
        self.assertIs(self.state.tracks[0], track)
        self.assertEqual(track.camera_label, "Front")
        self.assertEqual(track.final_output_path, str(new_video))
        self.assertEqual(self.state.intervals[0].label, "Wrong")
        meta = json.loads((folder / "meta.json").read_text())
        self.assertEqual(meta["cameras"][0]["label"], "Front")
        self.assertEqual(meta["cameras"][0]["video_file"], "cam1_Front.mp4")
        self.assertEqual(meta["label"], "Wrong")
        self.assertEqual(meta["custom"], {"preserve": True})
        sidecar = json.loads(self.sidecar.read_text())
        self.assertEqual(sidecar["extra_field"], {"preserved": "Wrong"})
        self.assertEqual(sidecar["cameras"][0]["final_video_path"], new_video.name)
        loaded = ProjectState.load(project)
        self.assertEqual(loaded.tracks[0].camera_label, "Front")
        self.assertEqual(loaded.tracks[0].final_output_path, str(new_video))
        from_sidecar = ProjectState()
        load_sidecar(str(self.sidecar), from_sidecar)
        self.assertEqual(from_sidecar.tracks[0].final_output_path, str(new_video))

    def test_custom_imported_metadata_filename_is_updated(self):
        custom = self.root / "custom-import.json"
        self.sidecar.rename(custom)
        loaded = ProjectState()
        load_sidecar(str(custom), loaded)
        self.assertEqual(loaded.metadata_path, str(custom))
        rename_cameras(loaded, ["Side view"])
        self.assertEqual(json.loads(custom.read_text())["cameras"][0]["label"], "Side view")
        self.assertFalse(self.sidecar.exists())

    def test_existing_destination_aborts_without_changing_anything(self):
        folder = self.add_segment()
        (folder / "cam1_Front.mp4").write_bytes(b"do not overwrite")
        before = self.state.to_dict()
        metadata = self.sidecar.read_bytes()
        with self.assertRaises(FileExistsError):
            rename_cameras(self.state, ["Front"])
        self.assertEqual(self.state.to_dict(), before)
        self.assertEqual(self.sidecar.read_bytes(), metadata)
        self.assertTrue(self.video.exists())
        self.assertEqual((folder / "cam1_Front.mp4").read_bytes(), b"do not overwrite")

    def test_metadata_failure_restores_all_files_and_written_json(self):
        folder = self.add_segment()
        before = self.state.to_dict()
        backups = {p: p.read_bytes() for p in (self.sidecar, folder / "meta.json")}
        original_replace = Path.replace
        count = 0

        def fail_second_replace(source, target):
            nonlocal count
            count += 1
            if count == 2:
                raise PermissionError("simulated locked metadata")
            return original_replace(source, target)

        with patch.object(Path, "replace", fail_second_replace):
            with self.assertRaises(PermissionError):
                rename_cameras(self.state, ["Front"])
        self.assertEqual(self.state.to_dict(), before)
        self.assertTrue(self.video.exists())
        self.assertTrue((folder / "cam1_Wrong.mp4").exists())
        self.assertFalse((self.root / "cam1_Front_final.mp4").exists())
        for path, data in backups.items():
            self.assertEqual(path.read_bytes(), data)
        self.assertEqual(list(self.root.rglob(".camera-*.tmp")), [])

    def test_move_failure_restores_staged_files(self):
        self.add_segment()
        original_rename = Path.rename
        count = 0

        def fail_second_rename(source, target):
            nonlocal count
            count += 1
            if count == 2:
                raise PermissionError("simulated locked video")
            return original_rename(source, target)

        with patch.object(Path, "rename", fail_second_rename):
            with self.assertRaises(PermissionError):
                rename_cameras(self.state, ["Front"])
        self.assertTrue(self.video.exists())
        self.assertEqual(self.state.tracks[0].camera_label, "Wrong")
        self.assertEqual(list(self.root.rglob(".camera-*.tmp")), [])

    def test_case_only_rename(self):
        rename_cameras(self.state, ["wrong"])
        self.assertIn("cam1_wrong_final.mp4", [p.name for p in self.root.iterdir()])
        self.assertEqual(self.state.tracks[0].camera_label, "wrong")
        self.assertEqual(json.loads(self.sidecar.read_text())["cameras"][0]["final_video_path"],
                         "cam1_wrong_final.mp4")

    def test_two_cameras_can_swap_labels_without_swapping_recordings(self):
        second = self.root / "cam2_Front_final.mp4"
        second.write_bytes(b"second camera")
        self.state.tracks.append(VideoTrack(camera_index=1, camera_label="Front", final_output_path=str(second)))
        generate_sidecar(self.state)
        rename_cameras(self.state, ["Front", "Wrong"])
        self.assertEqual((self.root / "cam1_Front_final.mp4").read_bytes(), b"original video bytes")
        self.assertEqual((self.root / "cam2_Wrong_final.mp4").read_bytes(), b"second camera")

    def test_external_segment_folder_and_legacy_clip_without_metadata(self):
        folder = self.add_segment(self.root / "external" / "segment")
        (folder / "meta.json").unlink()
        rename_cameras(self.state, ["Front"])
        self.assertTrue((folder / "cam1_Front.mp4").exists())

    def test_unrelated_project_sidecar_is_not_modified(self):
        unrelated = self.root / "other_meta.json"
        data = json.dumps({"cameras": [{"label": "Wrong", "final_video_path": "other.mp4"}]})
        unrelated.write_text(data, encoding="utf-8")
        rename_cameras(self.state, ["Front"])
        self.assertEqual(unrelated.read_text(), data)

    def test_retained_intermediate_and_multiple_matching_sidecars(self):
        intermediate = self.root / "cam1_Wrong_trimstart.mp4"
        intermediate.write_bytes(b"trimmed")
        self.state.tracks[0].trimstart_path = str(intermediate)
        other = self.root / "session_meta.json"
        other.write_bytes(self.sidecar.read_bytes())
        self.assertEqual(rename_cameras(self.state, ["Front"]), 2)
        self.assertEqual(self.state.tracks[0].trimstart_path, str(self.root / "cam1_Front_trimstart.mp4"))
        self.assertEqual(json.loads(other.read_text())["cameras"][0]["label"], "Front")

    def test_captured_frame_filenames_follow_camera_name(self):
        captures = self.root / "synchronized_captures" / "review_frame000012"
        captures.mkdir(parents=True)
        old = captures / "cam01_cam1_Wrong_final_frame000012.png"
        old.write_bytes(b"frame pixels")
        unrelated = captures / "review_snapshot.png"
        unrelated.write_bytes(b"composite snapshot")
        self.assertEqual(rename_cameras(self.state, ["Front view"]), 2)
        self.assertFalse(old.exists())
        self.assertEqual((captures / "cam01_cam1_Front_view_final_frame000012.png").read_bytes(),
                         b"frame pixels")
        self.assertEqual(unrelated.read_bytes(), b"composite snapshot")

    def test_progress_reports_files_and_completion(self):
        self.add_segment()
        events = []
        paths = {}
        rename_cameras(self.state, ["Front"], progress=lambda *event: events.append(event),
                       path_changes=paths)
        self.assertEqual(events[0][0], 0)
        self.assertEqual(events[-1][0], 100)
        self.assertEqual([percent for percent, _ in events], sorted(percent for percent, _ in events))
        self.assertTrue(any("Updating metadata:" in text for _, text in events))
        self.assertEqual(paths[str(self.video)], str(self.root / "cam1_Front_final.mp4"))

    def test_cancel_during_planning_does_not_change_any_files(self):
        before = self.state.to_dict()
        with self.assertRaises(CameraRenameCancelled):
            rename_cameras(self.state, ["Front"], cancelled=lambda: True)
        self.assertEqual(self.state.to_dict(), before)
        self.assertTrue(self.video.exists())

    def test_cancel_after_metadata_writes_restores_entire_transaction(self):
        folder = self.add_segment()
        before = self.state.to_dict()
        backup = {p: p.read_bytes() for p in (self.sidecar, folder / "meta.json")}
        cancel = threading.Event()
        events = []

        def report(percent, message):
            events.append(message)
            if message == "Finishing camera rename…":
                cancel.set()

        with self.assertRaises(CameraRenameCancelled):
            rename_cameras(self.state, ["Front"], progress=report, cancelled=cancel.is_set)
        self.assertEqual(self.state.to_dict(), before)
        self.assertTrue(self.video.exists())
        self.assertTrue((folder / "cam1_Wrong.mp4").exists())
        self.assertFalse((folder / "cam1_Front.mp4").exists())
        for path, content in backup.items():
            self.assertEqual(path.read_bytes(), content)
        self.assertIn("Restoring previous filenames and metadata…", events)
        self.assertEqual(list(self.root.rglob(".camera-*.tmp")), [])

    def test_does_not_read_or_copy_video_contents(self):
        read_bytes = Path.read_bytes

        def metadata_only(path):
            if path.suffix == ".mp4":
                self.fail("Rename must not read gigabytes of video contents.")
            return read_bytes(path)

        with patch.object(Path, "read_bytes", metadata_only):
            rename_cameras(self.state, ["Front"])

    def test_noop_does_not_touch_files(self):
        before = self.sidecar.read_bytes()
        self.assertEqual(rename_cameras(self.state, ["Wrong"]), 0)
        self.assertEqual(self.sidecar.read_bytes(), before)

    def test_rejects_invalid_names_and_wrong_camera_count(self):
        for label in ("", " ", "../oops", "A/B", "A\\B", 'A"B', "A:B", "A?B", "A*B", "A|B", "A<B", "A>B", "trailing.", "\x00", "x" * 101):
            with self.subTest(label=label), self.assertRaises(ValueError):
                rename_cameras(self.state, [label])
        with self.assertRaises(ValueError):
            rename_cameras(self.state, [])
        with self.assertRaises(ValueError):
            validate_camera_labels(["Front", "front"])
        self.assertEqual(validate_camera_labels([" Front view ", "Côté"]), ["Front view", "Côté"])


class CameraRenameUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_dialog_edits_all_camera_labels(self):
        from app.widgets.camera_rename_dialog import CameraRenameDialog
        dialog = CameraRenameDialog(ProjectState(tracks=[
            VideoTrack(camera_label="Wrong"), VideoTrack(camera_label="Side")
        ]))
        self.addCleanup(dialog.close)
        dialog._edits[0].setText("Front")
        self.assertEqual(dialog.labels(), ["Front", "Side"])
        dialog.accept()
        self.assertEqual(dialog.result(), dialog.DialogCode.Accepted)

    def test_dialog_rejects_invalid_name_without_closing(self):
        from app.widgets.camera_rename_dialog import CameraRenameDialog
        dialog = CameraRenameDialog(ProjectState(tracks=[VideoTrack(camera_label="Wrong")]))
        self.addCleanup(dialog.close)
        dialog._edits[0].setText("../bad")
        with patch("app.widgets.camera_rename_dialog.QMessageBox.warning") as warning:
            dialog.accept()
        warning.assert_called_once()
        self.assertEqual(dialog.result(), dialog.DialogCode.Rejected)

    def test_main_window_runs_rename_off_thread_and_keeps_gui_responsive(self):
        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QWidget
        from main import MainWindow
        gui_thread = threading.get_ident()
        state = ProjectState(tracks=[VideoTrack(camera_label="Wrong")])
        player = Mock()
        refresh = Mock()
        window = QWidget()
        self.addCleanup(window.close)
        window.state = state
        window._mode_widgets = [SimpleNamespace(refresh_camera_names=refresh)]
        window._players = lambda: [player]
        window._reload_modes = Mock()
        window.statusBar = lambda: Mock()
        events = []
        player.suspend_for_camera_rename.side_effect = lambda: events.append("release")
        gate = threading.Event()

        def rename(snapshot, labels, **kwargs):
            self.assertNotEqual(threading.get_ident(), gui_thread)
            self.assertIsNot(snapshot, state)
            # This gate is released by a GUI timer only while the worker runs.
            self.assertTrue(gate.wait(3), "The Qt event loop was blocked by renaming.")
            self.assertEqual(state.tracks[0].camera_label, "Wrong")
            snapshot.tracks[0].camera_label = labels[0]
            events.append("rename")
            return 1

        with patch("app.widgets.camera_rename_dialog.CameraRenameDialog") as dialog_type, \
                patch("app.camera_rename_worker.rename_cameras", side_effect=rename):
            dialog_type.return_value.exec.return_value = True
            dialog_type.return_value.labels.return_value = ["Front"]
            QTimer.singleShot(25, gate.set)
            MainWindow._rename_cameras(window)
        self.assertEqual(events, ["release", "rename"])
        self.assertEqual(state.tracks[0].camera_label, "Front")
        player.shutdown.assert_not_called()
        player.resume_after_camera_rename.assert_called_once_with({}, ["Front"])
        refresh.assert_called_once()
        window._reload_modes.assert_not_called()

    def test_progress_dialog_cancel_waits_until_rollback_finishes(self):
        from app.widgets.camera_rename_progress import CameraRenameProgressDialog
        progress = CameraRenameProgressDialog()
        self.addCleanup(progress.deleteLater)
        cancel = Mock()
        progress.cancel_requested.connect(cancel)
        progress.show()
        progress.reject()
        progress.close()
        cancel.assert_called_once()
        self.assertTrue(progress.isVisible())
        self.assertFalse(progress._cancel.isEnabled())
        progress.finish()
        self.assertFalse(progress.isVisible())

    def test_name_only_controls_do_not_reload_video_or_signals(self):
        from PyQt6.QtWidgets import QGridLayout, QLabel, QWidget
        from app.modes.mode1_preprocessing import Mode1Widget
        from app.modes.mode2_signal_sync import Mode2Widget
        from app.modes.mode3_labelling import Mode3Widget
        state = ProjectState(tracks=[VideoTrack(camera_label="Front")])
        group = SimpleNamespace(label="Wrong")
        preprocessing = SimpleNamespace(state=state, _camera_groups=[group], _update_metadata=Mock())
        Mode1Widget.refresh_camera_names(preprocessing)
        self.assertEqual(group.label, "Front")
        preprocessing._update_metadata.assert_not_called()
        content = QWidget()
        self.addCleanup(content.close)
        grid = QGridLayout(content)
        label = QLabel("Wrong")
        grid.addWidget(label, 1, 0)
        sync = SimpleNamespace(state=state, _time_grid=grid, _load_synced_signal=Mock())
        Mode2Widget.refresh_camera_names(sync)
        self.assertEqual(label.text(), "Front")
        sync._load_synced_signal.assert_not_called()
        labelling = SimpleNamespace(_refresh_time_summary=Mock(), _load_synced_signal=Mock())
        Mode3Widget.refresh_camera_names(labelling)
        labelling._refresh_time_summary.assert_called_once()
        labelling._load_synced_signal.assert_not_called()

    def test_cancelled_worker_leaves_shared_state_and_previews_unchanged(self):
        from PyQt6.QtWidgets import QWidget
        from main import MainWindow
        window = QWidget()
        self.addCleanup(window.close)
        window.state = ProjectState(tracks=[VideoTrack(camera_label="Wrong")])
        window._mode_widgets = []
        player = Mock()
        window._players = lambda: [player]
        window._reload_modes = Mock()
        status = Mock()
        window.statusBar = lambda: status
        with patch("app.widgets.camera_rename_dialog.CameraRenameDialog") as dialog_type, \
                patch("app.camera_rename_worker.rename_cameras", side_effect=CameraRenameCancelled), \
                patch("main.QMessageBox.critical") as message:
            dialog_type.return_value.exec.return_value = True
            dialog_type.return_value.labels.return_value = ["Front"]
            MainWindow._rename_cameras(window)
        self.assertEqual(window.state.tracks[0].camera_label, "Wrong")
        message.assert_not_called()
        player.resume_after_camera_rename.assert_called_once_with({}, ["Front"])
        self.assertIn("cancelled", status.showMessage.call_args.args[0])

    def test_preview_path_update_preserves_cached_metadata_and_position(self):
        from PyQt6.QtCore import QSize
        from app.widgets.frame_preview import MultiCameraPlayer
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory).resolve() / "cam1_Wrong_final.mp4"
            new = old.with_name("cam1_Front_final.mp4")
            new.write_bytes(b"dummy")
            player = MultiCameraPlayer()
            self.addCleanup(player.shutdown)
            self.addCleanup(player.close)
            player.set_cameras(["Wrong"])
            preview = player.previews[0]
            preview._video_path = str(old)
            preview._frame_count = 12084
            preview._fps = 59.94
            preview._frame_size = QSize(2688, 1512)
            player._position_ms = 54321
            with patch("app.widgets.frame_preview._probe_with_cv2", side_effect=AssertionError("Must not re-probe")):
                player.suspend_for_camera_rename()
                player.resume_after_camera_rename({str(old): str(new)}, ["Front"])
            self.assertIs(player.previews[0], preview)
            self.assertEqual(preview.label_text, "Front")
            self.assertEqual(preview.video_path, str(new))
            self.assertEqual(preview.frame_count, 12084)
            self.assertEqual(preview.fps, 59.94)
            self.assertEqual(player._position_ms, 54321)

    def test_failed_worker_restores_players_without_rebuilding_or_applying_names(self):
        from PyQt6.QtWidgets import QWidget
        from main import MainWindow
        window = QWidget()
        self.addCleanup(window.close)
        window.state = ProjectState(tracks=[VideoTrack(camera_label="Wrong")])
        window._mode_widgets = []
        player = Mock()
        window._players = lambda: [player]
        window._reload_modes = Mock()
        with patch("app.widgets.camera_rename_dialog.CameraRenameDialog") as dialog_type, \
                patch("app.camera_rename_worker.rename_cameras", side_effect=PermissionError("locked video")), \
                patch("main.QMessageBox.critical") as message:
            dialog_type.return_value.exec.return_value = True
            dialog_type.return_value.labels.return_value = ["Front"]
            MainWindow._rename_cameras(window)
        self.assertEqual(window.state.tracks[0].camera_label, "Wrong")
        message.assert_called_once()
        player.resume_after_camera_rename.assert_called_once_with({}, ["Front"])
        window._reload_modes.assert_not_called()


if __name__ == "__main__":
    unittest.main()