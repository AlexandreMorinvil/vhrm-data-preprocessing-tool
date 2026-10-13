import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from app.segment_signals import OFFSET_KEY, segment_timing_fields
from app.signal_offset import (
    apply_signal_offset,
    read_offset_file,
    read_segment_meta,
    segment_applied_offset,
    segment_reference_start,
)
from app.state import (
    LabelInterval,
    ProjectState,
    VideoTrack,
    effective_signal_anchor,
    generate_sidecar,
    load_sidecar,
    project_signal_anchor,
)

ANCHOR = datetime(2026, 5, 29, 19, 9, 7, tzinfo=timezone.utc)


def _timestamps(start: datetime, seconds: float, step: float) -> list[str]:
    return [(start + timedelta(seconds=i * step)).isoformat() for i in range(int(seconds / step))]


class SignalOffsetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        synthetic = self.root / "synthetic_ppg" / "synthetic_ppg.csv"
        synthetic.parent.mkdir()
        stamps = _timestamps(ANCHOR - timedelta(seconds=10), 60, 0.1)
        pd.DataFrame({"timestamp_utc": stamps, "synthetic_ppg": range(len(stamps))}).to_csv(synthetic, index=False)
        zephyr = self.root / "zephyr_1.csv"
        pd.DataFrame({"timestamp_utc": _timestamps(ANCHOR - timedelta(seconds=10), 60, 1.0),
                      "heart_rate_bpm": 70}).to_csv(zephyr, index=False)
        self.segment = self.root / "labelled_segments" / "0000_Resting"
        self.segment.mkdir(parents=True)
        self.interval = LabelInterval("Resting", 10.0, 20.0, folder=str(self.segment))
        meta = {"label": "Resting", "start_sec": 10.0, "end_sec": 20.0,
                **segment_timing_fields(ANCHOR, self.interval, 0.0)}
        (self.segment / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        self.state = ProjectState(
            output_directory=str(self.root),
            intervals=[self.interval],
            synthetic_ppg_path=str(synthetic),
            synced_hr_paths=[str(zephyr)],
            last_signal_anchor_datetime=ANCHOR.isoformat(),
            tracks=[VideoTrack(camera_label="Front", start_datetime=ANCHOR.isoformat(), duration_sec=40.0)],
        )
        self.sidecar = Path(generate_sidecar(self.state))

    def tearDown(self) -> None:
        self._directory.cleanup()

    def _first_slice_time(self, name: str) -> pd.Timestamp:
        frame = pd.read_csv(self.segment / name)
        return pd.to_datetime(frame["timestamp_utc"], utc=True, format="ISO8601").min()

    def test_apply_shifts_anchor_and_reslices_every_signal(self) -> None:
        report = apply_signal_offset(self.state, 1.5, note="baseline estimate")

        self.assertEqual(report.segments_updated, 1)
        meta = read_segment_meta(self.segment)
        expected_anchor = ANCHOR - timedelta(seconds=1.5)
        self.assertEqual(datetime.fromisoformat(meta["signal_anchor_datetime_utc"]), expected_anchor)
        self.assertEqual(datetime.fromisoformat(meta["segment_start_datetime_utc"]),
                         expected_anchor + timedelta(seconds=10))
        self.assertEqual(meta[OFFSET_KEY], 1.5)
        self.assertEqual(meta["video_signal_offset_note"], "baseline estimate")
        self.assertEqual(self._first_slice_time("synthetic_ppg/synthetic_ppg.csv"),
                         pd.Timestamp(ANCHOR + timedelta(seconds=8.5)))
        self.assertEqual(meta["hr_files"], ["zephyr_1.csv"])
        self.assertEqual(self._first_slice_time("zephyr_1.csv"), pd.Timestamp(ANCHOR + timedelta(seconds=9)))

        self.assertEqual(self.state.video_signal_offset_sec, 1.5)
        self.assertEqual(effective_signal_anchor(self.state), expected_anchor)
        sidecar = json.loads(self.sidecar.read_text(encoding="utf-8"))
        self.assertEqual(datetime.fromisoformat(sidecar["signal_anchor_datetime_utc"]), expected_anchor)
        self.assertEqual(sidecar[OFFSET_KEY], 1.5)
        record = json.loads((self.root / "ppg_alignment" / "alignment.json").read_text(encoding="utf-8"))
        self.assertEqual(record[OFFSET_KEY], 1.5)
        self.assertEqual(len(record["history"]), 1)

    def test_applying_again_replaces_the_offset_instead_of_adding_it(self) -> None:
        apply_signal_offset(self.state, 1.5)
        apply_signal_offset(self.state, -0.25)

        meta = read_segment_meta(self.segment)
        self.assertEqual(datetime.fromisoformat(meta["signal_anchor_datetime_utc"]),
                         ANCHOR + timedelta(seconds=0.25))
        self.assertEqual(effective_signal_anchor(self.state), ANCHOR + timedelta(seconds=0.25))
        sidecar = json.loads(self.sidecar.read_text(encoding="utf-8"))
        self.assertEqual(datetime.fromisoformat(sidecar["signal_anchor_datetime_utc"]),
                         ANCHOR + timedelta(seconds=0.25))
        record = json.loads((self.root / "ppg_alignment" / "alignment.json").read_text(encoding="utf-8"))
        self.assertEqual([entry[OFFSET_KEY] for entry in record["history"]], [1.5, -0.25])

    def test_reference_start_ignores_the_applied_offset(self) -> None:
        before = segment_reference_start(read_segment_meta(self.segment), self.interval, self.state)
        apply_signal_offset(self.state, 2.0)
        meta = read_segment_meta(self.segment)
        self.assertEqual(segment_reference_start(meta, self.interval, self.state), before)
        self.assertEqual(before, ANCHOR + timedelta(seconds=10))
        self.assertEqual(segment_applied_offset(meta, self.state), 2.0)

    def test_segments_exported_before_offsets_count_as_zero(self) -> None:
        meta = {"segment_start_datetime_utc": ANCHOR.isoformat()}
        self.state.video_signal_offset_sec = 3.0
        self.assertEqual(segment_applied_offset(meta, self.state), 0.0)
        self.assertEqual(segment_reference_start(meta, self.interval, self.state), ANCHOR)

    def test_offset_survives_project_and_sidecar_round_trips(self) -> None:
        apply_signal_offset(self.state, -6.53, note="jzc11 baseline")
        project = self.root / "project.vrt"
        self.state.save(project)
        self.assertEqual(ProjectState.load(project).video_signal_offset_sec, -6.53)
        loaded = ProjectState()
        load_sidecar(str(self.sidecar), loaded)
        self.assertEqual(loaded.video_signal_offset_sec, -6.53)
        self.assertEqual(loaded.video_signal_offset_note, "jzc11 baseline")
        for path in (project, self.sidecar, self.root / "ppg_alignment" / "alignment.json",
                     self.segment / "meta.json"):
            with self.subTest(path=path.name):
                self.assertEqual(read_offset_file(path)[0], -6.53)

    def test_reading_a_file_without_offset_is_an_error(self) -> None:
        other = self.root / "other.json"
        other.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            read_offset_file(other)

    def test_camera_anchor_is_corrected_by_the_offset(self) -> None:
        self.state.last_signal_anchor_datetime = None
        self.state.tracks[0].time_correction_mode = "offset"
        self.state.tracks[0].time_correction_offset_sec = -6.0
        anchor, _warnings = project_signal_anchor(self.state)
        self.assertEqual(anchor, ANCHOR - timedelta(seconds=6))
        self.state.video_signal_offset_sec = 1.25
        anchor, _warnings = project_signal_anchor(self.state)
        self.assertEqual(anchor, ANCHOR - timedelta(seconds=7.25))
        self.assertEqual(effective_signal_anchor(self.state), ANCHOR - timedelta(seconds=7.25))


if __name__ == "__main__":
    unittest.main()
