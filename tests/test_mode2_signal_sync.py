import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from app.modes.mode2_signal_sync import Mode2Widget
from app.state import ProjectState


class Mode2SignalSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def make_widget(self, state: ProjectState) -> Mode2Widget:
        widget = Mode2Widget(state)
        self.addCleanup(widget.close)
        return widget

    def test_removes_multiple_hr_sources_without_deleting_files_or_ecg(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = [root / f"source_{index}.csv" for index in range(4)]
            for path in paths:
                path.write_text("original source", encoding="utf-8")
            hr_paths = [str(path) for path in paths[:3]]
            ecg_paths = [str(paths[3])]
            state = ProjectState(
                hr_signal_paths=hr_paths,
                signal_paths=list(hr_paths),
                ecg_signal_paths=ecg_paths,
                mode2_complete=True,
            )
            widget = self.make_widget(state)
            self.assertFalse(widget._remove_hr_btn.isEnabled())
            widget._hr_file_list.item(0).setSelected(True)
            widget._hr_file_list.item(2).setSelected(True)
            self.assertTrue(widget._remove_hr_btn.isEnabled())

            widget._remove_hr_btn.click()

            self.assertEqual(state.hr_signal_paths, [hr_paths[1]])
            self.assertEqual(state.signal_paths, [hr_paths[1]])
            self.assertEqual(state.ecg_signal_paths, ecg_paths)
            self.assertFalse(state.mode2_complete)
            self.assertTrue(all(path.read_text(encoding="utf-8") == "original source" for path in paths))
            widget.refresh_from_state()
            self.assertEqual(widget._paths_from_list(widget._hr_file_list), [hr_paths[1]])

    def test_removes_last_legacy_hr_source_without_restoring_it_on_refresh(self) -> None:
        state = ProjectState(signal_paths=["old_hr.csv"])
        widget = self.make_widget(state)
        widget._hr_file_list.item(0).setSelected(True)

        widget._remove_hr_btn.click()
        widget.refresh_from_state()

        self.assertEqual(state.hr_signal_paths, [])
        self.assertEqual(state.signal_paths, [])
        self.assertEqual(widget._hr_file_list.count(), 0)
        self.assertFalse(widget._remove_hr_btn.isEnabled())

    def test_ecg_source_can_be_removed_and_replaced_before_sync(self) -> None:
        state = ProjectState(hr_signal_paths=["hr.csv"], ecg_signal_paths=["old_ecg.csv"])
        widget = self.make_widget(state)
        widget._ecg_file_list.item(0).setSelected(True)
        widget._remove_ecg_btn.click()
        self.assertEqual(state.ecg_signal_paths, [])

        with patch("app.modes.mode2_signal_sync.QFileDialog.getOpenFileNames", return_value=(["new_ecg.csv"], "")), patch(
            "app.modes.mode2_signal_sync.signal_file_type_name", return_value="ECG waveform"
        ):
            widget._add_signal(widget._ecg_file_list, "ECG")

        widget.refresh_from_state()
        self.assertEqual(state.ecg_signal_paths, ["new_ecg.csv"])
        self.assertEqual(state.hr_signal_paths, ["hr.csv"])
        self.assertEqual(widget._paths_from_list(widget._ecg_file_list), ["new_ecg.csv"])
        restored = ProjectState.from_dict(state.to_dict())
        self.assertEqual(restored.ecg_signal_paths, ["new_ecg.csv"])

    def test_removal_without_selection_does_not_change_state(self) -> None:
        state = ProjectState(hr_signal_paths=["hr.csv"], mode2_complete=True)
        widget = self.make_widget(state)

        widget._remove_signals(widget._hr_file_list)

        self.assertEqual(state.hr_signal_paths, ["hr.csv"])
        self.assertTrue(state.mode2_complete)

    def test_sync_drops_cached_data_and_output_references_for_removed_signal_kind(self) -> None:
        for removed_kind in ("HR", "ECG"):
            with self.subTest(removed_kind=removed_kind), tempfile.TemporaryDirectory() as temporary_directory:
                state = ProjectState(
                    output_directory=temporary_directory,
                    hr_signal_paths=["hr.csv"],
                    ecg_signal_paths=["ecg.csv"],
                    synced_hr_paths=["old_hr.csv"],
                    synced_hr_path="old_hr.csv",
                    synced_signal_path="old_hr.csv",
                    synced_ecg_paths=["old_ecg.csv"],
                    synced_ecg_path="old_ecg.csv",
                )
                widget = self.make_widget(state)
                timestamps = pd.to_datetime(["2026-01-01T12:00:00Z"])
                hr_data = pd.DataFrame({"timestamp_utc": timestamps, "Polar_1": [70]})
                ecg_data = pd.DataFrame({"timestamp_utc": timestamps, "ECG_1": [0.25]})
                widget._hr_merged_df = hr_data
                widget._ecg_merged_df = ecg_data
                target = widget._hr_file_list if removed_kind == "HR" else widget._ecg_file_list
                target.item(0).setSelected(True)
                widget._remove_signals(target)
                loaded_data = [None, ecg_data] if removed_kind == "HR" else [hr_data, None]

                with patch.object(widget, "_load_clip_and_pivot", side_effect=loaded_data):
                    widget._load_and_sync()

                self.assertTrue(state.mode2_complete)
                if removed_kind == "HR":
                    self.assertIsNone(widget._hr_merged_df)
                    self.assertEqual(state.synced_hr_paths, [])
                    self.assertEqual(state.synced_hr_path, "")
                    self.assertEqual(state.synced_signal_path, "")
                    self.assertEqual(widget._synced_hr_file_list.count(), 0)
                    self.assertEqual(len(state.synced_ecg_paths), 1)
                else:
                    self.assertIsNone(widget._ecg_merged_df)
                    self.assertEqual(state.synced_ecg_paths, [])
                    self.assertEqual(state.synced_ecg_path, "")
                    self.assertEqual(widget._synced_ecg_file_list.count(), 0)
                    self.assertEqual(len(state.synced_hr_paths), 1)

    def test_failed_sync_does_not_reexport_cached_signals(self) -> None:
        widget = self.make_widget(ProjectState(hr_signal_paths=["invalid_hr.csv"]))
        widget._hr_merged_df = pd.DataFrame({"Polar_1": [70]})

        with patch.object(widget, "_load_clip_and_pivot", return_value=None), patch(
            "app.modes.mode2_signal_sync.QMessageBox.warning"
        ) as warning, patch("app.modes.mode2_signal_sync.write_synced_signal_csvs") as write_signals:
            widget._load_and_sync()

        warning.assert_called_once()
        write_signals.assert_not_called()
        self.assertFalse(widget.state.mode2_complete)


if __name__ == "__main__":
    unittest.main()