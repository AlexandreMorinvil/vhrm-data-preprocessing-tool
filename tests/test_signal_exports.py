import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from app.signals import load_signal_files, read_synced_signal_csvs, write_synced_signal_csvs
from app.state import ProjectState


class SignalExportTests(unittest.TestCase):
    def test_numbers_single_sensor_of_each_type(self) -> None:
        class FakeLoader:
            def __init__(self, sensor_type: str) -> None:
                self.sensor_type = sensor_type
                self.display_name = sensor_type

            def load(self, _path: str) -> pd.DataFrame:
                return pd.DataFrame({
                    "timestamp_utc": pd.to_datetime(["2026-01-01T00:00:00Z"]),
                    "value": [70],
                })

        loaders = {
            "polar-source.csv": FakeLoader("Polar"),
            "zephyr-source.csv": FakeLoader("Zephyr"),
        }
        with patch("app.signals.get_signal_loader", side_effect=loaders.get):
            frames, _types, failures = load_signal_files(list(loaders))

        self.assertFalse(failures)
        self.assertEqual(
            [frame["sensor_id"].iloc[0] for frame in frames],
            ["Polar_1", "Zephyr_1"],
        )

    def test_writes_one_file_per_sensor_without_average(self) -> None:
        frame = pd.DataFrame({
            "timestamp_utc": pd.to_datetime([
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00.4Z",
            ], format="mixed"),
            "Polar": [70, 70],
            "Zephyr": [80, 80],
            "averaged": [75, 75],
            "aux__Zephyr__BR": [None, 12],
        })

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_synced_signal_csvs(
                frame, temporary_directory, "heart_rate_bpm"
            )

            self.assertEqual(
                {Path(path).name for path in paths},
                {"polar_1.csv", "zephyr_1.csv"},
            )
            self.assertFalse((Path(temporary_directory) / "average.csv").exists())
            zephyr = pd.read_csv(Path(temporary_directory) / "zephyr_1.csv")
            self.assertEqual(
                list(zephyr.columns),
                ["timestamp_utc", "heart_rate_bpm", "BR"],
            )

    def test_reads_separated_sensor_files_into_wide_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pd.DataFrame({
                "timestamp_utc": ["2026-01-01T00:00:00Z"],
                "heart_rate_bpm": [70],
            }).to_csv(root / "polar_1.csv", index=False)
            pd.DataFrame({
                "timestamp_utc": ["2026-01-01T00:00:00.4Z"],
                "heart_rate_bpm": [80],
                "BR": [12],
            }).to_csv(root / "zephyr.csv", index=False)

            frame = read_synced_signal_csvs([
                root / "polar_1.csv",
                root / "zephyr.csv",
            ])

            self.assertIsNotNone(frame)
            self.assertEqual(
                set(frame.columns),
                {"timestamp_utc", "Polar_1", "Zephyr", "aux__Zephyr__BR"},
            )

    def test_reads_legacy_combined_signal_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "hr_synced.csv"
            pd.DataFrame({
                "timestamp_utc": ["2026-01-01T00:00:00Z"],
                "Polar": [70],
                "Zephyr": [80],
            }).to_csv(path, index=False)

            frame = read_synced_signal_csvs([path])

            self.assertIsNotNone(frame)
            self.assertEqual(
                list(frame.columns), ["timestamp_utc", "Polar", "Zephyr"]
            )

    def test_reads_synthetic_ppg_as_a_separated_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "synthetic_ppg.csv"
            pd.DataFrame({
                "timestamp_utc": ["2026-01-01T00:00:00Z"],
                "synthetic_ppg": [0.75],
            }).to_csv(path, index=False)

            frame = read_synced_signal_csvs([path])

            self.assertIsNotNone(frame)
            self.assertEqual(list(frame.columns), ["timestamp_utc", "Synthetic_Ppg"])

    def test_migrates_legacy_project_signal_paths(self) -> None:
        state = ProjectState.from_dict({
            "synced_signal_path": "hr_synced.csv",
            "synced_ecg_path": "ecg_synced.csv",
        })

        self.assertEqual(state.synced_hr_paths, ["hr_synced.csv"])
        self.assertEqual(state.synced_ecg_paths, ["ecg_synced.csv"])
        self.assertEqual(
            state.legacy_synced_paths,
            ["hr_synced.csv", "ecg_synced.csv"],
        )


if __name__ == "__main__":
    unittest.main()