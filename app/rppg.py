"""Algorithmic (training-free) remote PPG from the mean colour of a skin region.

The functions take the per-frame mean R, G, B of a face region and return a
pulse waveform sampled at the video rate. They are used to check, and correct,
the time alignment between the camera videos and the physiological signals.

References:
- POS: Wang, den Brinker, Stuijk, de Haan (2017), "Algorithmic principles of
  remote PPG", IEEE TBME 64(7).
- CHROM: de Haan, Jeanne (2013), "Robust pulse rate from chrominance-based
  rPPG", IEEE TBME 60(10).
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

RPPG_METHODS = ("POS", "CHROM", "Green")
PULSE_BAND_HZ = (0.7, 3.5)


def _fill_gaps(rgb: np.ndarray) -> np.ndarray:
    """Linearly interpolate rows containing NaN (frames without a usable region)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("Expected an (n, 3) array of mean R, G, B values")
    valid = np.all(np.isfinite(rgb), axis=1)
    if valid.all() or not valid.any():
        return np.nan_to_num(rgb)
    index = np.arange(len(rgb))
    filled = rgb.copy()
    for channel in range(3):
        filled[:, channel] = np.interp(index, index[valid], rgb[valid, channel])
    return filled


def pos(rgb: np.ndarray, fs: float, window_sec: float = 1.6) -> np.ndarray:
    """Plane-Orthogonal-to-Skin pulse signal (overlap-added over sliding windows)."""
    rgb = _fill_gaps(rgb)
    count = len(rgb)
    window = max(2, int(round(window_sec * fs)))
    pulse = np.zeros(count)
    if count < window:
        return pulse
    projection = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    for start in range(count - window + 1):
        block = rgb[start:start + window].T
        mean = block.mean(axis=1, keepdims=True)
        mean[mean == 0] = 1.0
        s = projection @ (block / mean)
        std_1 = s[1].std()
        h = s[0] + (s[0].std() / std_1 if std_1 > 0 else 0.0) * s[1]
        pulse[start:start + window] += h - h.mean()
    return pulse


def chrom(rgb: np.ndarray, fs: float, window_sec: float = 1.6) -> np.ndarray:
    """Chrominance-based pulse signal (overlap-added over sliding windows)."""
    rgb = _fill_gaps(rgb)
    count = len(rgb)
    window = max(2, int(round(window_sec * fs)))
    pulse = np.zeros(count)
    if count < window:
        return pulse
    for start in range(count - window + 1):
        block = rgb[start:start + window]
        mean = block.mean(axis=0)
        mean[mean == 0] = 1.0
        normalised = block / mean
        x = 3.0 * normalised[:, 0] - 2.0 * normalised[:, 1]
        y = 1.5 * normalised[:, 0] + normalised[:, 1] - 1.5 * normalised[:, 2]
        std_y = y.std()
        h = x - (x.std() / std_y if std_y > 0 else 0.0) * y
        pulse[start:start + window] += (h - h.mean()) * np.hanning(window)
    return pulse


def green(rgb: np.ndarray, fs: float) -> np.ndarray:
    """Inverted, detrended green channel (blood absorbs green light)."""
    rgb = _fill_gaps(rgb)
    values = rgb[:, 1]
    mean = values.mean() if len(values) else 0.0
    return -(values - mean) / (mean if mean else 1.0)


def bandpass(values: np.ndarray, fs: float, band_hz: tuple[float, float] = PULSE_BAND_HZ,
             order: int = 2) -> np.ndarray:
    """Zero-phase band-pass filter (no time shift is introduced)."""
    values = np.asarray(values, dtype=np.float64)
    nyquist = fs / 2.0
    low, high = band_hz
    high = min(high, nyquist * 0.95)
    if len(values) < 3 * (2 * order + 1) or low >= high:
        return values - (values.mean() if len(values) else 0.0)
    sos = sp_signal.butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    padlen = min(len(values) - 1, 3 * (2 * order + 1) * 4)
    return sp_signal.sosfiltfilt(sos, values - values.mean(), padlen=padlen)


def standardise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return values
    centred = values - np.nanmean(values)
    deviation = np.nanstd(centred)
    return centred / deviation if deviation > 0 else centred


def estimate_pulse(rgb: np.ndarray, fs: float, method: str = "POS",
                   band_hz: tuple[float, float] = PULSE_BAND_HZ) -> np.ndarray:
    """Band-passed, standardised pulse waveform from mean skin colour.

    Every method returns the polarity of a contact PPG: the waveform rises with
    the blood volume (when the skin gets darker in green). The POS projection
    has the opposite sign, so it is inverted here; without this, aligning a
    reference PPG to POS would put it half a heartbeat out of phase.
    """
    method = method.upper()
    if method == "POS":
        raw = -pos(rgb, fs)
    elif method == "CHROM":
        raw = chrom(rgb, fs)
    elif method == "GREEN":
        raw = green(rgb, fs)
    else:
        raise ValueError(f"Unknown rPPG method: {method}")
    return standardise(bandpass(raw, fs, band_hz))


def dominant_rate_bpm(values: np.ndarray, fs: float,
                      band_hz: tuple[float, float] = PULSE_BAND_HZ) -> float:
    """Frequency of the strongest spectral peak inside *band_hz*, in beats per minute."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < max(8, int(fs * 4)):
        return float("nan")
    frequencies, power = sp_signal.periodogram(
        values - values.mean(), fs, window="hann", nfft=max(len(values), 8192)
    )
    mask = (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
    if not mask.any():
        return float("nan")
    return float(frequencies[mask][np.argmax(power[mask])] * 60.0)
