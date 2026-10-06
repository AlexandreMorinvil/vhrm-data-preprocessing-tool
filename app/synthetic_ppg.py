"""ECG-driven synthetic PPG generation.

The dynamical model is adapted from PPGSynth by Tang et al. and from the
project's Python port. PPGSynth is licensed under GNU GPL v3. This modified
implementation uses exact cumulative RR boundaries instead of expanding each
interval with ``ceil(fs * RR)`` samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import butter, filtfilt, find_peaks, sosfiltfilt
from scipy.sparse import diags, eye
from scipy.sparse.linalg import factorized


_THETA = np.asarray((-1.6184, 0.8903), dtype=float)
_AMPLITUDE = np.asarray((0.7482, 0.0444), dtype=float)
_WIDTH = np.asarray((0.9353, 1.9499), dtype=float)
_TOOLBOX_DETREND_LAMBDA = 100.0
_TOOLBOX_REFERENCE_FS = 30.0


@dataclass(frozen=True)
class SyntheticPpgResult:
    ppg: pd.DataFrame
    heart_rate: pd.DataFrame
    hrv: pd.DataFrame
    peaks: pd.DataFrame
    source_peak_times: pd.DatetimeIndex
    rr_intervals_ms: np.ndarray
    sampling_rate_hz: float


def detect_ecg_peaks(
    ecg: pd.DataFrame,
    *,
    low_hz: float = 5.0,
    high_hz: float = 25.0,
    min_peak_distance_s: float = 0.35,
    prominence_factor: float = 4.0,
) -> pd.DataFrame:
    """Detect ECG R peaks and return their absolute timestamps and RR intervals."""

    value_column = "ecg_waveform" if "ecg_waveform" in ecg.columns else "value"
    if "timestamp_utc" not in ecg.columns or value_column not in ecg.columns:
        raise ValueError("ECG data must contain timestamp_utc and ecg_waveform/value columns")
    samples = ecg[["timestamp_utc", value_column]].copy()
    samples["timestamp_utc"] = pd.to_datetime(samples["timestamp_utc"], utc=True, errors="coerce")
    samples[value_column] = pd.to_numeric(samples[value_column], errors="coerce")
    samples = samples.dropna().sort_values("timestamp_utc").drop_duplicates("timestamp_utc")
    if len(samples) < 10:
        raise ValueError("ECG data does not contain enough valid samples")

    elapsed = (
        samples["timestamp_utc"] - samples["timestamp_utc"].iloc[0]
    ).dt.total_seconds().to_numpy()
    intervals = np.diff(elapsed)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    if intervals.size == 0:
        raise ValueError("Could not determine the ECG sampling rate")
    fs = 1.0 / float(np.median(intervals))
    if not 0 < low_hz < high_hz < fs / 2.0:
        raise ValueError(f"ECG band-pass must be below Nyquist ({fs / 2.0:.2f} Hz)")

    values = samples[value_column].to_numpy(float)
    filtered = sosfiltfilt(
        butter(3, (low_hz, high_hz), btype="bandpass", fs=fs, output="sos"),
        values,
    )
    robust_scale = float(np.median(np.abs(filtered - np.median(filtered))))
    if not np.isfinite(robust_scale) or robust_scale <= np.finfo(float).eps:
        raise ValueError("ECG filtering produced no measurable signal variation")
    peak_indices, properties = find_peaks(
        filtered,
        distance=max(1, round(min_peak_distance_s * fs)),
        prominence=prominence_factor * robust_scale,
    )
    candidate_prominences = properties["prominences"]
    local_qrs_prominence = (
        pd.Series(
            candidate_prominences,
            index=pd.DatetimeIndex(samples["timestamp_utc"].iloc[peak_indices]),
        )
        .rolling("10s", center=True, min_periods=1)
        .quantile(0.9)
        .to_numpy()
    )
    accepted = candidate_prominences >= 0.25 * local_qrs_prominence
    peak_indices = peak_indices[accepted]
    if peak_indices.size < 2:
        raise ValueError("Too few ECG peaks detected; inspect the signal or detector settings")

    peak_times = pd.DatetimeIndex(samples["timestamp_utc"].iloc[peak_indices])
    result = pd.DataFrame({
        "timestamp_utc": peak_times,
        "rr_interval_ms": pd.Series(peak_times).diff().dt.total_seconds().mul(1000.0),
        "ecg_waveform": values[peak_indices],
        "filtered_ecg": filtered[peak_indices],
        "prominence": candidate_prominences[accepted],
    })
    return result.reset_index(drop=True)


def _centered_local_median(values: np.ndarray, width: int = 11) -> np.ndarray:
    radius = width // 2
    return np.asarray([
        np.median(values[max(0, index - radius) : min(values.size, index + radius + 1)])
        for index in range(values.size)
    ])


def _timestamp_differences_seconds(timestamps: pd.DatetimeIndex) -> np.ndarray:
    return (
        pd.Series(timestamps)
        .diff()
        .dt.total_seconds()
        .iloc[1:]
        .to_numpy(dtype=float)
    )


def _refine_peak_frequency(
    frequencies: np.ndarray,
    power: np.ndarray,
    peak_index: int,
) -> float:
    """Refine a spectral peak with a three-bin log-parabolic estimate."""

    if peak_index <= 0 or peak_index >= power.size - 1:
        return float(frequencies[peak_index])
    log_power = np.log(
        np.maximum(
            power[peak_index - 1 : peak_index + 2],
            np.finfo(float).tiny,
        )
    )
    denominator = log_power[0] - 2.0 * log_power[1] + log_power[2]
    if abs(denominator) <= np.finfo(float).eps:
        return float(frequencies[peak_index])
    offset = float(
        np.clip(
            0.5 * (log_power[0] - log_power[2]) / denominator,
            -0.5,
            0.5,
        )
    )
    return float(frequencies[peak_index] + offset * (frequencies[1] - frequencies[0]))


def _estimate_autocorrelation_bpm(
    power: np.ndarray,
    *,
    nfft: int,
    window_samples: int,
    fs: float,
    low_hz: float,
    high_hz: float,
) -> float:
    """Estimate the fundamental period from overlap-normalized autocorrelation."""

    autocorrelation = np.fft.irfft(power, n=nfft)[:window_samples]
    minimum_lag = max(1, round(fs / high_hz))
    maximum_lag = min(window_samples - 2, round(fs / low_hz))
    lags = np.arange(minimum_lag, maximum_lag + 1)
    scores = autocorrelation[lags] / (window_samples - lags)
    peaks, _ = find_peaks(scores)
    peak_offset = int(peaks[np.argmax(scores[peaks])]) if peaks.size else int(np.argmax(scores))
    lag = float(lags[peak_offset])
    if 0 < peak_offset < scores.size - 1:
        denominator = (
            scores[peak_offset - 1]
            - 2.0 * scores[peak_offset]
            + scores[peak_offset + 1]
        )
        if abs(denominator) > np.finfo(float).eps:
            lag += float(np.clip(
                0.5
                * (scores[peak_offset - 1] - scores[peak_offset + 1])
                / denominator,
                -0.5,
                0.5,
            ))
    return float(60.0 * fs / lag)


def estimate_ppg_heart_rate(
    ppg: pd.DataFrame,
    *,
    fs: float,
    window_seconds: float = 10.0,
    step_seconds: float = 1.0,
    low_hz: float = 0.6,
    high_hz: float = 3.3,
    target_timestamps: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Estimate one-second HR with the rPPG-Toolbox-style FFT pipeline."""

    values = ppg["synthetic_ppg"].to_numpy(float)
    timestamps = pd.DatetimeIndex(pd.to_datetime(ppg["timestamp_utc"], utc=True))
    window_samples = int(round(window_seconds * fs))
    step_samples = int(round(step_seconds * fs))
    if window_samples < 9 or values.size < window_samples:
        return pd.DataFrame(columns=["timestamp_utc", "heart_rate_bpm"])
    if step_samples < 1:
        raise ValueError("HR estimate step must be at least one sample")
    if not 0 < low_hz < high_hz < fs / 2.0:
        raise ValueError("HR frequency limits must satisfy 0 < low < high < fs/2")

    second_difference = diags(
        (
            np.ones(window_samples - 2),
            -2 * np.ones(window_samples - 2),
            np.ones(window_samples - 2),
        ),
        (0, 1, 2),
        shape=(window_samples - 2, window_samples),
        format="csc",
    )
    detrend_lambda = _TOOLBOX_DETREND_LAMBDA * (fs / _TOOLBOX_REFERENCE_FS) ** 2
    solve_trend = factorized(
        eye(window_samples, format="csc")
        + detrend_lambda**2 * (second_difference.T @ second_difference)
    )
    numerator, denominator = butter(1, [low_hz, high_hz], btype="bandpass", fs=fs)
    nfft = 1 << (8 * window_samples - 1).bit_length()
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / fs)
    in_band = (frequencies >= low_hz) & (frequencies <= high_hz)

    if target_timestamps is None:
        windows = [
            (
                start,
                timestamps[start]
                + (timestamps[start + window_samples - 1] - timestamps[start]) / 2,
            )
            for start in range(0, values.size - window_samples + 1, step_samples)
        ]
    else:
        targets = pd.DatetimeIndex(
            pd.to_datetime(target_timestamps, utc=True)
        ).sort_values().drop_duplicates()
        latest_start = values.size - window_samples
        windows = []
        for target in targets:
            center_sample = round((target - timestamps[0]).total_seconds() * fs)
            start = max(0, min(center_sample - window_samples // 2, latest_start))
            windows.append((start, target))

    estimates = []
    recent_bpm: list[float] = []
    cached_spectra: dict[int, np.ndarray] = {}
    for start, output_time in windows:
        stop = start + window_samples
        power = cached_spectra.get(start)
        if power is None:
            window = values[start:stop]
            detrended = window - solve_trend(window)
            filtered = filtfilt(numerator, denominator, detrended)
            power = np.abs(np.fft.rfft(filtered, n=nfft)) ** 2
            cached_spectra[start] = power
        peak_index = np.flatnonzero(in_band)[np.argmax(power[in_band])]
        peak_frequency = _refine_peak_frequency(frequencies, power, peak_index)
        bpm = float(peak_frequency * 60.0)
        half_center = int(np.argmin(np.abs(frequencies - peak_frequency / 2.0)))
        half_candidates = np.arange(max(1, half_center - 2), half_center + 3)
        half_index = int(half_candidates[np.argmax(power[half_candidates])])
        half_frequency = _refine_peak_frequency(frequencies, power, half_index)
        half_bpm = float(half_frequency * 60.0)
        half_power_ratio = power[half_index] / power[peak_index]
        autocorrelation_bpm = _estimate_autocorrelation_bpm(
            power,
            nfft=nfft,
            window_samples=window_samples,
            fs=fs,
            low_hz=low_hz,
            high_hz=high_hz,
        )
        autocorrelation_support = (
            1.6 * autocorrelation_bpm < bpm < 2.4 * autocorrelation_bpm
            and abs(half_bpm - autocorrelation_bpm) <= 0.3 * autocorrelation_bpm
        )
        continuity_support = False
        dominant_disagrees_with_history = not recent_bpm
        if recent_bpm:
            recent_median = float(np.median(recent_bpm[-5:]))
            dominant_disagrees_with_history = (
                abs(bpm - recent_median) > 0.1 * recent_median
            )
            continuity_support = (
                1.6 * recent_median < bpm < 2.4 * recent_median
                and abs(half_bpm - recent_median) <= 0.3 * recent_median
            )
        if (
            autocorrelation_support and dominant_disagrees_with_history
        ) or (
            half_power_ratio >= 0.2 and continuity_support
        ):
            bpm = half_bpm
        estimates.append({
            "timestamp_utc": output_time,
            "heart_rate_bpm": bpm,
        })
        recent_bpm.append(bpm)
    return pd.DataFrame(estimates)


def estimate_ppg_metrics(
    ppg: pd.DataFrame,
    *,
    fs: float,
    window_beats: int = 300,
    min_window_beats: int = 30,
    step_seconds: float = 1.0,
    hr_timestamps: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate windowed FFT HR and rolling pulse-interval SDNN/RMSSD."""

    values = ppg["synthetic_ppg"].to_numpy(float)
    timestamps = pd.DatetimeIndex(pd.to_datetime(ppg["timestamp_utc"], utc=True))
    heart_rate = estimate_ppg_heart_rate(
        ppg,
        fs=fs,
        step_seconds=step_seconds,
        target_timestamps=hr_timestamps,
    )
    amplitude_range = float(np.ptp(values))
    if amplitude_range <= np.finfo(float).eps:
        raise ValueError("Synthetic PPG has no measurable amplitude variation")
    peak_indices, _ = find_peaks(
        values,
        distance=max(1, round(0.3 * fs)),
        prominence=0.05 * amplitude_range,
    )
    hrv_columns = [
        "timestamp_utc", "sdnn_ms", "rmssd_ms", "accepted_intervals",
        "rejected_intervals", "quality_fraction",
    ]
    if peak_indices.size < 3:
        return heart_rate, pd.DataFrame(columns=hrv_columns)

    peak_times = timestamps[peak_indices]
    interval_end_times = peak_times[1:]
    intervals_ms = _timestamp_differences_seconds(peak_times) * 1000.0
    local_ratio = intervals_ms / _centered_local_median(intervals_ms)
    accepted = (
        (intervals_ms >= 300.0)
        & (intervals_ms <= 2000.0)
        & (local_ratio >= 0.6)
        & (local_ratio <= 1.6)
    )
    accepted_indices = np.flatnonzero(accepted)
    min_window_beats = max(3, min(min_window_beats, window_beats))
    if accepted_indices.size < min_window_beats:
        return heart_rate, pd.DataFrame(columns=hrv_columns)

    valid_times = interval_end_times[accepted_indices]
    first_output = valid_times[min_window_beats - 1].ceil(f"{step_seconds}s")
    last_output = interval_end_times[-1].floor(f"{step_seconds}s")
    output_times = pd.date_range(first_output, last_output, freq=f"{step_seconds}s")
    estimates = []
    for output_time in output_times:
        valid_stop = int(valid_times.searchsorted(output_time, side="right"))
        if valid_stop < min_window_beats:
            continue
        window_count = min(valid_stop, window_beats)
        window_indices = accepted_indices[valid_stop - window_count : valid_stop]
        window = intervals_ms[window_indices]
        window_start = valid_times[valid_stop - window_count] - pd.to_timedelta(
            window[0], unit="ms"
        )
        in_span = (interval_end_times > window_start) & (interval_end_times <= output_time)
        total_count = int(np.count_nonzero(in_span))
        accepted_count = int(np.count_nonzero(accepted & in_span))
        successive = np.diff(window)[np.diff(window_indices) == 1]
        estimates.append({
            "timestamp_utc": output_time,
            "sdnn_ms": float(np.std(window, ddof=1)),
            "rmssd_ms": float(np.sqrt(np.mean(successive**2))) if successive.size else np.nan,
            "accepted_intervals": window_count,
            "rejected_intervals": total_count - accepted_count,
            "quality_fraction": accepted_count / total_count if total_count else 0.0,
        })
    return heart_rate, pd.DataFrame(estimates, columns=hrv_columns)


def _generate_exact_timing_ppg(rr_intervals_s: np.ndarray, fs: float) -> np.ndarray:
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling rate must be positive and finite")
    rr = np.asarray(rr_intervals_s, dtype=float).reshape(-1)
    if rr.size == 0 or not np.all(np.isfinite(rr)) or np.any(rr <= 0):
        raise ValueError("RR intervals must be positive and finite")

    duration = float(rr.sum())
    sample_count = int(np.ceil(duration * fs))
    times = np.arange(sample_count, dtype=float) / fs
    boundaries = np.cumsum(rr)

    def equations(time: float, point: np.ndarray) -> np.ndarray:
        interval_index = min(int(np.searchsorted(boundaries, time, side="right")), rr.size - 1)
        x_value, y_value, _ = point
        omega = 2.0 * np.pi / rr[interval_index]
        dx_dt = -omega * y_value
        dy_dt = omega * x_value
        radius_squared = x_value * x_value + y_value * y_value
        dtheta_dt = (-y_value * dx_dt + x_value * dy_dt) / radius_squared
        phase_delta = np.arctan2(y_value, x_value) - _THETA
        dz_dt = -np.sum(
            _AMPLITUDE
            * phase_delta
            * np.exp(-((phase_delta / _WIDTH) ** 2))
            / 2.0
        ) * dtheta_dt
        return np.asarray((dx_dt, dy_dt, dz_dt), dtype=float)

    solution = solve_ivp(
        equations,
        (0.0, float(times[-1])),
        (-1.0, 0.0, 0.0),
        method="RK45",
        t_eval=times,
    )
    if not solution.success:
        raise RuntimeError(f"PPG integration failed: {solution.message}")
    signal = solution.y[2]
    low = float(signal.min())
    span = float(signal.max() - low)
    return np.zeros_like(signal) if span <= np.finfo(float).eps else (signal - low) / span


def synthesize_from_peak_times(
    peak_times: pd.DatetimeIndex,
    *,
    fs: float = 125.0,
    hr_timestamps: pd.DatetimeIndex | None = None,
) -> SyntheticPpgResult:
    """Generate a timestamped PPG from all supplied ECG peak timestamps."""

    peaks = pd.DatetimeIndex(peak_times).sort_values().drop_duplicates()
    if len(peaks) < 2:
        raise ValueError("At least two ECG peaks are required for PPG synthesis")
    rr_intervals_s = _timestamp_differences_seconds(peaks)
    signal = _generate_exact_timing_ppg(rr_intervals_s, fs)
    timestamps = peaks[0] + pd.to_timedelta(np.arange(signal.size) / fs, unit="s")
    ppg = pd.DataFrame({
        "timestamp_utc": timestamps,
        "synthetic_ppg": signal,
    })
    if len(peaks) >= 4:
        heart_rate, hrv = estimate_ppg_metrics(
            ppg,
            fs=fs,
            hr_timestamps=hr_timestamps,
        )
    else:
        heart_rate = pd.DataFrame(columns=["timestamp_utc", "heart_rate_bpm"])
        hrv = pd.DataFrame(columns=[
            "timestamp_utc", "sdnn_ms", "rmssd_ms", "accepted_intervals",
            "rejected_intervals", "quality_fraction",
        ])
    peak_frame = pd.DataFrame({
        "timestamp_utc": peaks,
        "rr_interval_ms": pd.Series(peaks).diff().dt.total_seconds().mul(1000.0),
    })
    return SyntheticPpgResult(
        ppg=ppg,
        heart_rate=heart_rate,
        hrv=hrv,
        peaks=peak_frame,
        source_peak_times=peaks,
        rr_intervals_ms=rr_intervals_s * 1000.0,
        sampling_rate_hz=float(fs),
    )


def generate_synthetic_ppg(
    ecg: pd.DataFrame,
    *,
    fs: float = 125.0,
    low_hz: float = 5.0,
    high_hz: float = 25.0,
    min_peak_distance_s: float = 0.35,
    prominence_factor: float = 4.0,
    hr_timestamps: pd.DatetimeIndex | None = None,
) -> SyntheticPpgResult:
    """Detect ECG peaks and generate synchronized PPG, HR, and HRV outputs."""

    detected = detect_ecg_peaks(
        ecg,
        low_hz=low_hz,
        high_hz=high_hz,
        min_peak_distance_s=min_peak_distance_s,
        prominence_factor=prominence_factor,
    )
    result = synthesize_from_peak_times(
        pd.DatetimeIndex(detected["timestamp_utc"]),
        fs=fs,
        hr_timestamps=hr_timestamps,
    )
    return SyntheticPpgResult(
        ppg=result.ppg,
        heart_rate=result.heart_rate,
        hrv=result.hrv,
        peaks=detected,
        source_peak_times=result.source_peak_times,
        rr_intervals_ms=result.rr_intervals_ms,
        sampling_rate_hz=result.sampling_rate_hz,
    )