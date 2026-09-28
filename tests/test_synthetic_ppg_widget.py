import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from app.modes.mode3_synthetic_ppg import (
    SyntheticPpgWidget,
    _SYNTHETIC_ARTIFACT_NAMES,
    _preferred_hr_reference_path,
    synthetic_ppg_artifact_paths,
)
from app.file_cleanup import cleanup_obsolete_paths
from app.state import ProjectState, generate_sidecar, load_sidecar


class SyntheticPpgWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_lists_only_synchronized_ecg_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_path = root / "raw_ecg.csv"
            raw_path.touch()
            synced_path = root / "ecg_1.csv"
            pd.DataFrame({
                "timestamp_utc": ["2026-01-01T12:00:00Z"],
                "ecg_waveform": [0.25],
            }).to_csv(synced_path, index=False)
            state = ProjectState(
                ecg_signal_paths=[str(raw_path)],
                synced_ecg_paths=[str(synced_path)],
            )

            widget = SyntheticPpgWidget(state)

            self.assertEqual(widget._source.count(), 1)
            self.assertEqual(widget._source.currentData(), str(synced_path))
            widget.close()

    def test_prefers_zephyr_timestamps_for_generated_hr(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            polar_path = root / "polar_1.csv"
            zephyr_path = root / "zephyr_1.csv"
            polar_path.touch()
            zephyr_path.touch()

            selected = _preferred_hr_reference_path([
                str(polar_path),
                str(zephyr_path),
            ])

            self.assertEqual(selected, str(zephyr_path))

    def test_artifacts_are_grouped_in_dedicated_directory(self) -> None:
        paths = synthetic_ppg_artifact_paths(Path("project_output"))

        self.assertEqual(
            {path.parent for path in paths.values()},
            {Path("project_output") / "synthetic_ppg"},
        )
        self.assertEqual(
            {path.name for path in paths.values()},
            {
                "synthetic_ppg.csv",
                "synthetic_ppg_hr.csv",
                "synthetic_ppg_hrv.csv",
                "synthetic_ppg_rr.csv",
            },
        )

    def test_sidecar_preserves_nested_synthetic_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / "synthetic_ppg" / "synthetic_ppg.csv"
            artifact.parent.mkdir()
            artifact.touch()
            state = ProjectState(
                output_directory=str(root),
                synthetic_ppg_path=str(artifact),
            )

            sidecar_path = generate_sidecar(state)
            loaded = ProjectState()
            load_sidecar(sidecar_path, loaded)

            self.assertTrue(Path(loaded.synthetic_ppg_path).samefile(artifact))

    def test_legacy_root_artifacts_can_be_cleaned_after_migration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            legacy_paths = []
            for filename in _SYNTHETIC_ARTIFACT_NAMES.values():
                path = root / filename
                path.touch()
                legacy_paths.append(path)
            for path in synthetic_ppg_artifact_paths(root).values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            removed = cleanup_obsolete_paths(legacy_paths, root, archive=False)

            self.assertEqual(removed, 4)
            self.assertFalse(any(path.exists() for path in legacy_paths))
            self.assertTrue(
                all(path.exists() for path in synthetic_ppg_artifact_paths(root).values())
            )


if __name__ == "__main__":
    unittest.main()