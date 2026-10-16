import unittest

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

from app import synthetic_ppg
from app.synthetic_ppg import (
    detect_ecg_peaks,
    estimate_ppg_heart_rate,
    estimate_ppg_metrics,
    generate_synthetic_ppg,
    pulse_phase_drift,
    synthesize_from_peak_times,
)


def _ppgsynth_ode(rr: np.ndarray, fs: float) -> np.ndarray:
    """The PPGSynth equations integrated numerically (tight tolerances)."""
    boundaries = np.cumsum(rr)
    times = np.arange(int(np.ceil(rr.sum() * fs))) / fs

    def equations(time, point):
        index = min(int(np.searchsorted(boundaries, time, side="right")), rr.size - 1)
        x_value, y_value, _ = point
        omega = 2.0 * np.pi / rr[index]
        dx_dt, dy_dt = -omega * y_value, omega * x_value
        dtheta_dt = (-y_value * dx_dt + x_value * dy_dt) / (x_value**2 + y_value**2)
        delta = np.arctan2(y_value, x_value) - synthetic_ppg._THETA
        dz_dt = -np.sum(
            synthetic_ppg._AMPLITUDE * delta * np.exp(-((delta / synthetic_ppg._WIDTH) ** 2)) / 2.0
        ) * dtheta_dt
        return [dx_dt, dy_dt, dz_dt]

    solution = solve_ivp(equations, (0.0, times[-1]), (-1.0, 0.0, 0.0), t_eval=times,
                         rtol=1e-10, atol=1e-12, max_step=0.01)
    return times, solution.y[2]


def _pulse_fractions(values: np.ndarray, fs: float, peak_times: np.ndarray) -> np.ndarray:
    """Position of every PPG pulse within its RR interval (0 = R peak, 1 = next R peak)."""
    indices, _ = find_peaks(values, distance=round(0.3 * fs), prominence=0.2)
    pulse_times = indices / fs
    beat = np.searchsorted(peak_times, pulse_times, side="right") - 1
    return (pulse_times - peak_times[beat]) / np.diff(peak_times)[beat]


class SyntheticPpgTimingTests(unittest.TestCase):
    def test_closed_form_is_the_ppgsynth_model_without_its_per_beat_ramp(self) -> None:
        rr = np.random.default_rng(0).uniform(0.55, 1.1, 8)
        fs = 125.0
        times, z = _ppgsynth_ode(rr, fs)
        boundaries = np.cumsum(rr)
        beat = np.minimum(np.searchsorted(boundaries, times, side="right"), rr.size - 1)
        fraction = (times - (boundaries - rr)[beat]) / rr[beat]
        start, end = synthetic_ppg._beat_shape(np.array((-np.pi, np.pi)))
        z -= (end - start) * (beat + fraction)  # the model's z changes by end - start every beat
        expected = (z - z.min()) / np.ptp(z)

        np.testing.assert_allclose(synthetic_ppg._generate_exact_timing_ppg(rr, fs), expected, atol=1e-6)

    def test_pulse_keeps_its_place_in_the_beat_over_a_long_recording(self) -> None:
        # About 40 minutes: the numerical integration used before drifted by a third of a beat.
        rng = np.random.default_rng(1)
        rr = np.clip(0.8 + np.cumsum(rng.normal(0.0, 0.01, 3000)) * 0.05, 0.45, 1.2)
        rr += rng.normal(0.0, 0.03, rr.size)
        fs = 125.0
        values = synthetic_ppg._generate_exact_timing_ppg(rr, fs)
        peak_times = np.concatenate(([0.0], np.cumsum(rr)))

        fractions = _pulse_fractions(values, fs, peak_times)

        self.assertEqual(len(fractions), len(rr))
        self.assertLess(np.ptp(fractions), 0.02)
        self.assertAlmostEqual(float(np.median(fractions)), 0.247, delta=0.005)
        self.assertLess(np.ptp(values[: int(60 * fs)]), 1.0 + 1e-9)
        self.assertGreater(np.ptp(values[-int(60 * fs):]), 0.9)  # no ramp squashing the end

    def test_phase_drift_detects_a_pulse_sliding_against_the_ecg(self) -> None:
        rng = np.random.default_rng(2)
        rr = rng.uniform(0.7, 0.9, 1500)
        fs = 125.0
        values = synthetic_ppg._generate_exact_timing_ppg(rr, fs)
        times = np.arange(values.size) / fs
        peak_times = np.concatenate(([0.0], np.cumsum(rr)))

        self.assertLess(pulse_phase_drift(times, values, peak_times), 0.01)
        # Pulses arriving 0.25 s later by the end of the recording.
        drifting = times + 0.25 * times / times[-1]
        self.assertAlmostEqual(pulse_phase_drift(drifting, values, peak_times), 0.25 / 0.8, delta=0.05)
        self.assertTrue(np.isnan(pulse_phase_drift(times[:500], values[:500], peak_times[:5])))


    def test_synthesis_uses_only_supplied_ecg_peaks(self) -> None:
        peaks = pd.DatetimeIndex(pd.to_datetime([
            "2026-01-01T12:00:00.500Z",
            "2026-01-01T12:00:01.500Z",
            "2026-01-01T12:00:02.500Z",
        ]))

        result = synthesize_from_peak_times(peaks, fs=20.0)

        self.assertTrue(result.source_peak_times.equals(peaks))
        self.assertEqual(result.ppg["timestamp_utc"].iloc[0], peaks[0])
        self.assertEqual(
            result.ppg["timestamp_utc"].iloc[-1],
            peaks[-1] - pd.Timedelta(milliseconds=50),
        )

    def test_synthesis_does_not_require_outside_peaks(self) -> None:
        peaks = pd.DatetimeIndex(pd.to_datetime([
            "2026-01-01T12:00:00.500Z",
            "2026-01-01T12:00:01.500Z",
            "2026-01-01T12:00:02.500Z",
        ]))

        result = synthesize_from_peak_times(peaks, fs=20.0)

        self.assertTrue(result.source_peak_times.equals(peaks))
        self.assertEqual(len(result.ppg), 40)
        self.assertTrue(result.ppg["synthetic_ppg"].between(0.0, 1.0).all())

    def test_synthesis_matches_requested_hr_timestamps(self) -> None:
        peaks = pd.date_range("2026-01-01T12:00:00Z", periods=21, freq="1s")
        reference_times = pd.date_range(
            "2026-01-01T11:59:59.750Z", periods=22, freq="1s"
        )

        result = synthesize_from_peak_times(
            peaks,
            fs=40.0,
            hr_timestamps=reference_times,
        )

        self.assertTrue(
            pd.DatetimeIndex(result.heart_rate["timestamp_utc"]).equals(
                reference_times
            )
        )

    def test_detects_periodic_ecg_peaks(self) -> None:
        fs = 250.0
        elapsed = np.arange(0.0, 10.0, 1.0 / fs)
        values = np.zeros_like(elapsed)
        for peak_time in np.arange(1.0, 9.1, 1.0):
            values += np.exp(-0.5 * ((elapsed - peak_time) / 0.015) ** 2)
        ecg = pd.DataFrame({
            "timestamp_utc": pd.Timestamp("2026-01-01T12:00:00Z")
            + pd.to_timedelta(elapsed, unit="s"),
            "ecg_waveform": values,
        })

        peaks = detect_ecg_peaks(ecg)

        detected_elapsed = (
            peaks["timestamp_utc"] - ecg["timestamp_utc"].iloc[0]
        ).dt.total_seconds().to_numpy()
        self.assertEqual(len(peaks), 9)
        np.testing.assert_allclose(detected_elapsed, np.arange(1.0, 9.1, 1.0), atol=0.02)

    def test_ecg_detector_rejects_secondary_waves_between_beats(self) -> None:
        fs = 250.0
        elapsed = np.arange(0.0, 30.0, 1.0 / fs)
        expected_peaks = np.arange(1.0, 29.1, 1.0)
        values = 0.03 * np.sin(2 * np.pi * 0.3 * elapsed)
        for peak_time in expected_peaks:
            amplitude = 1.0 if peak_time < 15.0 else 0.3
            values += amplitude * np.exp(-0.5 * ((elapsed - peak_time) / 0.015) ** 2)
            values += 0.18 * amplitude * np.exp(
                -0.5 * ((elapsed - peak_time - 0.42) / 0.035) ** 2
            )
        ecg = pd.DataFrame({
            "timestamp_utc": pd.Timestamp("2026-01-01T12:00:00Z")
            + pd.to_timedelta(elapsed, unit="s"),
            "ecg_waveform": values,
        })

        result = generate_synthetic_ppg(ecg, fs=40.0)
        peaks = result.peaks

        detected_elapsed = (
            peaks["timestamp_utc"] - ecg["timestamp_utc"].iloc[0]
        ).dt.total_seconds().to_numpy()
        self.assertEqual(len(peaks), len(expected_peaks))
        np.testing.assert_allclose(detected_elapsed, expected_peaks, atol=0.02)
        self.assertAlmostEqual(result.heart_rate["heart_rate_bpm"].median(), 60.0, delta=2.0)
        self.assertLess(result.heart_rate["heart_rate_bpm"].max(), 70.0)

    def test_ecg_detector_preserves_genuine_high_rate_beats(self) -> None:
        fs = 250.0
        elapsed = np.arange(0.0, 12.0, 1.0 / fs)
        expected_peaks = np.arange(1.0, 11.1, 0.4)
        values = np.zeros_like(elapsed)
        for peak_time in expected_peaks:
            values += np.exp(-0.5 * ((elapsed - peak_time) / 0.015) ** 2)
        ecg = pd.DataFrame({
            "timestamp_utc": pd.Timestamp("2026-01-01T12:00:00Z")
            + pd.to_timedelta(elapsed, unit="s"),
            "ecg_waveform": values,
        })

        peaks = detect_ecg_peaks(ecg)

        detected_elapsed = (
            peaks["timestamp_utc"] - ecg["timestamp_utc"].iloc[0]
        ).dt.total_seconds().to_numpy()
        self.assertEqual(len(peaks), len(expected_peaks))
        np.testing.assert_allclose(detected_elapsed, expected_peaks, atol=0.02)

    def test_estimates_windowed_hr_and_hrv_from_constant_pulse_train(self) -> None:
        fs = 40.0
        elapsed = np.arange(0.0, 306.0, 1.0 / fs)
        values = np.sin(2 * np.pi * elapsed) + 0.35 * np.sin(4 * np.pi * elapsed)
        ppg = pd.DataFrame({
            "timestamp_utc": pd.Timestamp("2026-01-01T12:00:00Z")
            + pd.to_timedelta(elapsed, unit="s"),
            "synthetic_ppg": values,
        })

        heart_rate, hrv = estimate_ppg_metrics(ppg, fs=fs)

        self.assertGreater(len(heart_rate), 290)
        self.assertLess(len(heart_rate), 305)
        self.assertAlmostEqual(heart_rate["heart_rate_bpm"].median(), 60.0, delta=2.0)
        time_steps = heart_rate["timestamp_utc"].diff().dropna().dt.total_seconds()
        self.assertTrue(np.allclose(time_steps, 1.0))
        self.assertFalse(hrv.empty)
        self.assertAlmostEqual(hrv["sdnn_ms"].max(), 0.0, places=6)
        self.assertAlmostEqual(hrv["rmssd_ms"].max(), 0.0, places=6)

    def test_hrv_uses_expanding_window_for_short_recordings(self) -> None:
        fs = 40.0
        elapsed = np.arange(0.0, 120.0, 1.0 / fs)
        ppg = pd.DataFrame({
            "timestamp_utc": pd.Timestamp("2026-01-01T12:00:00Z")
            + pd.to_timedelta(elapsed, unit="s"),
            "synthetic_ppg": np.sin(2 * np.pi * elapsed),
        })

        _heart_rate, hrv = estimate_ppg_metrics(ppg, fs=fs)

        self.assertFalse(hrv.empty)
        self.assertEqual(hrv["accepted_intervals"].min(), 30)
        self.assertLess(hrv["accepted_intervals"].max(), 300)
        self.assertTrue(hrv["accepted_intervals"].is_monotonic_increasing)

    def test_fft_hr_rejects_sustained_second_harmonic_lock(self) -> None:
        fs = 40.0
        elapsed = np.arange(0.0, 45.0, 1.0 / fs)
        fundamental = np.sin(2 * np.pi * 1.25 * elapsed)
        second_harmonic = np.sin(2 * np.pi * 2.5 * elapsed)
        values = np.where(
            elapsed < 20.0,
            fundamental + 0.25 * second_harmonic,
            0.4 * fundamental + second_harmonic,
        )
        ppg = pd.DataFrame({
            "timestamp_utc": pd.Timestamp("2026-01-01T12:00:00Z")
            + pd.to_timedelta(elapsed, unit="s"),
            "synthetic_ppg": values,
        })

        heart_rate = estimate_ppg_heart_rate(ppg, fs=fs)

        self.assertLess(heart_rate["heart_rate_bpm"].max(), 100.0)
        self.assertAlmostEqual(heart_rate["heart_rate_bpm"].median(), 75.0, delta=3.0)

    def test_fft_hr_recovers_when_harmonic_lock_starts_immediately(self) -> None:
        fs = 40.0
        elapsed = np.arange(0.0, 30.0, 1.0 / fs)
        fundamental = 0.4 * np.sin(2 * np.pi * 1.25 * elapsed)
        second_harmonic = np.sin(2 * np.pi * 2.5 * elapsed)
        ppg = pd.DataFrame({
            "timestamp_utc": pd.Timestamp("2026-01-01T12:00:00Z")
            + pd.to_timedelta(elapsed, unit="s"),
            "synthetic_ppg": fundamental + second_harmonic,
        })

        heart_rate = estimate_ppg_heart_rate(ppg, fs=fs)

        self.assertLess(heart_rate["heart_rate_bpm"].max(), 100.0)
        self.assertAlmostEqual(heart_rate["heart_rate_bpm"].median(), 75.0, delta=3.0)

    def test_recent_hr_prevents_false_autocorrelation_halving(self) -> None:
        fs = 40.0
        elapsed = np.arange(0.0, 45.0, 1.0 / fs)
        cycles = np.floor(elapsed * 1.25).astype(int)
        amplitude = np.where(
            elapsed < 15.0,
            1.0,
            np.where(cycles % 2 == 0, 1.0, 0.35),
        )
        ppg = pd.DataFrame({
            "timestamp_utc": pd.Timestamp("2026-01-01T12:00:00Z")
            + pd.to_timedelta(elapsed, unit="s"),
            "synthetic_ppg": amplitude * np.sin(2 * np.pi * 1.25 * elapsed),
        })

        heart_rate = estimate_ppg_heart_rate(ppg, fs=fs)

        self.assertGreater(heart_rate["heart_rate_bpm"].min(), 70.0)
        self.assertAlmostEqual(heart_rate["heart_rate_bpm"].median(), 75.0, delta=3.0)

    def test_fft_hr_matches_reference_timestamps_and_refines_frequency(self) -> None:
        fs = 40.0
        elapsed = np.arange(0.0, 30.0, 1.0 / fs)
        frequency_hz = 1.23
        ppg = pd.DataFrame({
            "timestamp_utc": pd.Timestamp("2026-01-01T12:00:00.250Z")
            + pd.to_timedelta(elapsed, unit="s"),
            "synthetic_ppg": np.sin(2 * np.pi * frequency_hz * elapsed),
        })
        reference_times = pd.date_range(
            "2026-01-01T12:00:00Z", periods=31, freq="1s"
        )

        heart_rate = estimate_ppg_heart_rate(
            ppg,
            fs=fs,
            target_timestamps=reference_times,
        )

        self.assertTrue(
            pd.DatetimeIndex(heart_rate["timestamp_utc"]).equals(reference_times)
        )
        self.assertEqual(len(heart_rate), len(reference_times))
        self.assertAlmostEqual(
            heart_rate["heart_rate_bpm"].median(),
            frequency_hz * 60.0,
            delta=0.4,
        )
        self.assertGreater(heart_rate["heart_rate_bpm"].nunique(), 1)


if __name__ == "__main__":
    unittest.main()