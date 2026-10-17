import json
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
    _BoxTracker,
    cache_paths,
    choose_face,
    extract_face_trace,
    load_trace,
    prepare_reference,
    roi_from_face,
    save_trace,
    pool_windows,
    scan_lags,
    window_correlations,
    windowed_correlation,
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

    def test_unreliable_windows_are_left_out(self) -> None:
        reference_t, reference = prepare_reference(self.reference_times, self.reference)
        pulse = self._pulse_with_offset(1.0)
        noisy = self.pulse_times > 30
        # Motion: a strong 2.6 Hz oscillation dominates the pulse in the second half.
        pulse[noisy] = 5 * np.sin(2 * np.pi * 2.6 * self.pulse_times[noisy])
        detected = np.ones(len(pulse), dtype=bool)

        sums = windowed_correlation(self.pulse_times, pulse, FS, reference_t, reference, -3, 3,
                                    window_sec=20, hop_sec=10, detected=detected)
        self.assertEqual(sums.windows, 5)
        self.assertEqual(sums.accepted, 2)  # 0-20 s and 10-30 s
        self.assertAlmostEqual(sums.scan().best(), 1.0, delta=0.03)

        detected[:] = False
        lost = windowed_correlation(self.pulse_times, pulse, FS, reference_t, reference, -3, 3,
                                    window_sec=20, hop_sec=10, detected=detected)
        self.assertEqual(lost.accepted, 0)
        self.assertFalse(lost.scan().valid)

    def test_reliability_criteria_apply_after_scanning(self) -> None:
        reference_t, reference = prepare_reference(self.reference_times, self.reference)
        # A rPPG locked on 1.6 Hz: correlated with nothing, at the wrong heart rate.
        pulse = np.sin(2 * np.pi * 1.6 * self.pulse_times)
        lags, windows = window_correlations(self.pulse_times, pulse, FS, reference_t, reference, -3, 3,
                                            window_sec=20, hop_sec=10)
        self.assertEqual(len(windows), 5)
        self.assertTrue(all(np.isfinite(window.correlation).any() for window in windows))
        self.assertEqual(pool_windows(lags, windows).accepted, 0)
        self.assertEqual(pool_windows(lags, windows, max_hr_difference_bpm=None).accepted, 5)
        self.assertAlmostEqual(windows[0].video_hr, 96.0, delta=1.5)

    def test_pooling_segments_at_different_heart_rates_resolves_the_beat_ambiguity(self) -> None:
        offset = 1.3
        segments = []
        # Resting and exercise, perfectly periodic so each segment alone is ambiguous
        # (their peaks coincide again only 20 s away, outside the search range).
        for rate_hz in (1.0, 1.45):
            reference_times = np.arange(-15, 75, 1 / 125.0)
            reference_t, reference = prepare_reference(reference_times, np.sin(2 * np.pi * rate_hz * reference_times))
            pulse = np.sin(2 * np.pi * rate_hz * (self.pulse_times - offset))
            segments.append(windowed_correlation(self.pulse_times, pulse, FS, reference_t, reference, -4, 4))
        single_ratios = [segment.scan().peak_ratio() for segment in segments]
        self.assertLess(max(single_ratios), 1.02)

        pooled = segments[0]
        pooled.add(segments[1])
        self.assertEqual(pooled.accepted, 6)
        self.assertAlmostEqual(pooled.scan().best(), offset, delta=0.03)
        self.assertGreater(pooled.scan().peak_ratio(), max(single_ratios) + 0.01)

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


def _track(settings: FaceTraceSettings, faces: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Boxes of _BoxTracker for a face at *faces[i]* in frame i, and the frames detected on."""
    detected_on: list[int] = []

    def detect(index):
        detected_on.append(int(index))
        return [(*faces[index], 0.9)]

    # The tracker passes the frame to the detector: here the frame is its index.
    tracker = _BoxTracker(settings, detect)
    released = []
    for index in range(len(faces)):
        released += tracker.push(index, index)
    released += tracker.flush()
    assert [frame for frame, _box, _detected in released] == list(range(len(faces)))
    return np.array([box for _frame, box, _detected in released]), detected_on


class BoxTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        # A face bobbing up and down 3 face heights every 2 s (squats), at 30 frames/s.
        t = np.arange(120) / FS
        y = 200 + 150 * np.sin(2 * np.pi * 0.5 * t)
        self.faces = np.column_stack([np.full(len(t), 100.0), y, np.full(len(t), 100.0), np.full(len(t), 100.0)])

    def _lag(self, boxes: np.ndarray) -> float:
        """Median vertical error of the box, in face heights."""
        return float(np.median(np.abs(boxes[:, 1] - self.faces[:, 1])) / 100.0)

    def test_original_tracking_detects_every_tenth_frame_and_blends(self) -> None:
        settings = FaceTraceSettings()
        boxes, detected_on = _track(settings, self.faces)
        self.assertEqual(detected_on, list(range(0, 120, 10)))
        np.testing.assert_allclose(boxes[0], self.faces[0])
        blended = (1 - settings.box_smoothing) * self.faces[0] + settings.box_smoothing * self.faces[10]
        np.testing.assert_allclose(boxes[10], blended)
        np.testing.assert_allclose(boxes[15], blended)  # held until the next detection
        self.assertGreater(self._lag(boxes), 0.5)  # trails the moving face

    def test_detecting_every_frame_follows_a_moving_face(self) -> None:
        boxes, detected_on = _track(FaceTraceSettings(detect_every_frame=True), self.faces)
        self.assertEqual(detected_on, list(range(120)))
        self.assertLess(self._lag(boxes), 0.2)

    def test_centred_smoothing_does_not_lag(self) -> None:
        boxes, _ = _track(FaceTraceSettings(detect_every_frame=True, centred_smoothing=True), self.faces)
        self.assertLess(self._lag(boxes), 0.02)
        # Between detections every 10th frame, the box is interpolated rather than held.
        boxes, detected_on = _track(FaceTraceSettings(centred_smoothing=True), self.faces)
        self.assertEqual(detected_on, list(range(0, 120, 10)))
        self.assertLess(self._lag(boxes), 0.15)

    def test_centred_smoothing_keeps_the_face_size_steady(self) -> None:
        faces = self.faces.copy()
        faces[50, 2:] = 160.0  # one oversized detection
        boxes, _ = _track(FaceTraceSettings(detect_every_frame=True, centred_smoothing=True), faces)
        np.testing.assert_allclose(boxes[50, 2:], [100.0, 100.0])


class TraceCacheTests(unittest.TestCase):
    def test_traces_cached_before_the_tracking_options_stay_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "cam2_Front.mp4"
            video.write_bytes(b"video")
            trace = FaceTrace(fs=30.0, rgb=np.ones((4, 3)), detected=np.ones(4, bool),
                              boxes=np.zeros((4, 4)), frame_size=(540, 960),
                              video_path=str(video), settings=FaceTraceSettings())
            save_trace(directory, "key", trace)
            with np.load(cache_paths(directory, "key")[0]) as data:
                stored = json.loads(str(data["signature"]))["settings"]
            self.assertNotIn("detect_every_frame", stored)
            self.assertNotIn("centred_smoothing", stored)
            self.assertIsNotNone(load_trace(directory, "key", str(video), FaceTraceSettings()))
            self.assertIsNone(load_trace(directory, "key", str(video), FaceTraceSettings(detect_every_frame=True)))
            self.assertIsNone(load_trace(directory, "key", str(video), FaceTraceSettings(centred_smoothing=True)))


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
