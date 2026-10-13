import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.ffmpeg_utils import find_ffmpeg
from app.ppg_alignment import (
    FaceTrace,
    FaceTraceSettings,
    cache_paths,
    choose_face,
    extract_face_trace,
    load_trace,
    prepare_reference,
    roi_from_face,
    save_trace,
    scan_lags,
)
from app.rppg import RPPG_METHODS, dominant_rate_bpm, estimate_pulse

FS = 30.0


def _irregular_pulse(times: np.ndarray, seed: int = 0) -> np.ndarray:
    """Blood-volume-like waveform with beat-to-beat variability (breaks the periodic ambiguity)."""
    rng = np.random.default_rng(seed)
    beats = np.cumsum(rng.uniform(0.75, 0.95, size=int(times[-1] / 0.7) + 20)) - 5.0
    phase = np.interp(times, beats, np.arange(len(beats)))
    return np.cos(2 * np.pi * phase)


def _skin_rgb(blood_volume: np.ndarray, seed: int = 1) -> np.ndarray:
    """Mean skin colour: every channel darkens with blood volume, green the most."""
    rng = np.random.default_rng(seed)
    base = np.array([180.0, 120.0, 100.0])
    absorption = np.array([0.0015, 0.004, 0.002])
    rgb = base * (1.0 - absorption * blood_volume[:, None])
    return rgb + rng.normal(0.0, 0.05, rgb.shape)


class RppgTests(unittest.TestCase):
    def test_every_method_returns_contact_ppg_polarity_and_rate(self) -> None:
        times = np.arange(0, 60, 1 / FS)
        blood_volume = np.sin(2 * np.pi * 1.2 * times)
        rgb = _skin_rgb(blood_volume)
        for method in RPPG_METHODS:
            with self.subTest(method=method):
                pulse = estimate_pulse(rgb, FS, method)
                core = slice(int(5 * FS), int(55 * FS))
                correlation = np.corrcoef(pulse[core], blood_volume[core])[0, 1]
                self.assertGreater(correlation, 0.8)
                self.assertAlmostEqual(dominant_rate_bpm(pulse, FS), 72.0, delta=1.5)

    def test_missing_frames_are_interpolated(self) -> None:
        times = np.arange(0, 20, 1 / FS)
        rgb = _skin_rgb(np.sin(2 * np.pi * 1.0 * times))
        rgb[100:110] = np.nan
        self.assertTrue(np.isfinite(estimate_pulse(rgb, FS, "POS")).all())


class LagScanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference_times = np.arange(-15, 75, 1 / 125.0)
        self.reference = _irregular_pulse(self.reference_times)
        self.pulse_times = np.arange(0, 60, 1 / FS)

    def _pulse_with_offset(self, offset: float) -> np.ndarray:
        # The video pulse at t equals the reference at t - offset.
        return np.interp(self.pulse_times - offset, self.reference_times, self.reference)

    def test_recovers_known_offset(self) -> None:
        reference_t, reference = prepare_reference(self.reference_times, self.reference)
        for offset in (-6.5, 0.0, 1.31):
            with self.subTest(offset=offset):
                scan = scan_lags(self.pulse_times, self._pulse_with_offset(offset),
                                 reference_t, reference, -10, 10)
                self.assertAlmostEqual(scan.best(), offset, delta=0.015)
                self.assertGreater(scan.at(offset), 0.95)
                self.assertGreater(scan.peak_ratio(), 1.0)

    def test_steps_between_neighbouring_peaks(self) -> None:
        reference_t, reference = prepare_reference(self.reference_times, self.reference)
        scan = scan_lags(self.pulse_times, self._pulse_with_offset(1.0), reference_t, reference, -4, 4)
        best = scan.best()
        after = scan.step(best, 1)
        before = scan.step(best, -1)
        self.assertGreater(after, best + 0.5)
        self.assertLess(after, best + 1.1)
        self.assertLess(before, best - 0.5)
        self.assertGreater(before, best - 1.1)
        self.assertAlmostEqual(scan.step(after, -1), best, delta=1e-6)

    def test_window_restricts_the_samples_used(self) -> None:
        reference_t, reference = prepare_reference(self.reference_times, self.reference)
        pulse = self._pulse_with_offset(2.0)
        pulse[self.pulse_times > 30] = np.random.default_rng(3).normal(size=(self.pulse_times > 30).sum())
        scan = scan_lags(self.pulse_times, pulse, reference_t, reference, -4, 4, window=(0.0, 30.0))
        self.assertAlmostEqual(scan.best(), 2.0, delta=0.015)

    def test_no_overlap_gives_an_invalid_scan(self) -> None:
        scan = scan_lags(self.pulse_times, self._pulse_with_offset(0.0), np.array([500.0, 501.0]),
                         np.array([0.0, 1.0]), -2, 2)
        self.assertFalse(scan.valid)
        self.assertIsNone(scan.best())
        self.assertIsNone(scan.step(0.0, 1))


class FaceRoiTests(unittest.TestCase):
    def test_roi_keeps_the_centre_of_the_face_inside_the_frame(self) -> None:
        self.assertEqual(roi_from_face(np.array([100, 50, 100, 120]), 0.8, (640, 480)), (110, 50, 80, 120))
        x, y, w, h = roi_from_face(np.array([600, 400, 100, 120]), 0.8, (640, 480))
        self.assertLessEqual(x + w, 640)
        self.assertLessEqual(y + h, 480)

    def test_follows_the_previous_face_before_the_largest(self) -> None:
        previous = np.array([10.0, 10.0, 50.0, 50.0])
        detections = [(12, 11, 50, 50, 0.9), (300, 300, 120, 120, 0.9)]
        np.testing.assert_array_equal(choose_face(detections, previous), [12, 11, 50, 50])
        np.testing.assert_array_equal(choose_face(detections, None), [300, 300, 120, 120])
        self.assertIsNone(choose_face([], previous))


class TraceCacheTests(unittest.TestCase):
    def test_cache_is_reused_only_for_the_same_video_and_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "cam2_Front.mp4"
            video.write_bytes(b"video")
            settings = FaceTraceSettings()
            trace = FaceTrace(fs=30.0, rgb=np.ones((4, 3)), detected=np.ones(4, bool),
                              boxes=np.zeros((4, 4)), frame_size=(540, 960),
                              video_path=str(video), settings=settings)
            save_trace(directory, "key", trace)
            loaded = load_trace(directory, "key", str(video), settings)
            self.assertIsNotNone(loaded)
            np.testing.assert_array_equal(loaded.rgb, trace.rgb)
            self.assertEqual(loaded.frame_size, (540, 960))
            self.assertIsNone(load_trace(directory, "key", str(video), FaceTraceSettings(skin_mask=False)))
            video.write_bytes(b"re-exported video")
            self.assertIsNone(load_trace(directory, "key", str(video), settings))


@unittest.skipUnless(find_ffmpeg(), "FFmpeg is required")
class ExtractionTests(unittest.TestCase):
    def test_decodes_at_the_analysis_rate_and_writes_the_roi_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "clip.mp4"
            subprocess.run(
                [find_ffmpeg(), "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=60:duration=2",
                 "-pix_fmt", "yuv420p", str(video)],
                check=True,
            )
            roi_path = Path(directory) / "cache" / "clip_roi.mp4"  # folder does not exist yet
            trace = extract_face_trace(str(video), FaceTraceSettings(hardware_decoding=False),
                                       ffmpeg=find_ffmpeg(), roi_video_path=str(roi_path))
            self.assertAlmostEqual(len(trace.rgb), 60, delta=1)
            self.assertFalse(trace.detected.any())  # no face in the test pattern
            self.assertTrue(roi_path.exists())
            self.assertFalse(any(Path(directory, "cache").glob("*.partial.*")))
            self.assertEqual(cache_paths(directory, "clip")[1].name, "clip_roi.mp4")


if __name__ == "__main__":
    unittest.main()
