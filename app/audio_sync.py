from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from .ffmpeg_utils import extract_audio

log = logging.getLogger(__name__)


def _load_audio(wav_path: str) -> tuple[np.ndarray, int]:
    import librosa
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    return y, sr


def compute_sync_offset(
    reference_path: str,
    target_path: str,
    ffmpeg: str = "",
    max_offset_sec: float = 60.0,
    audio_duration_sec: float | None = 60.0,
) -> float:
    with tempfile.TemporaryDirectory() as tmp:
        ref_wav = str(Path(tmp) / "ref.wav")
        tgt_wav = str(Path(tmp) / "tgt.wav")
        extract_audio(reference_path, ref_wav, ffmpeg=ffmpeg,
                       duration_sec=audio_duration_sec)
        extract_audio(target_path, tgt_wav, ffmpeg=ffmpeg,
                       duration_sec=audio_duration_sec)

        ref_y, sr = _load_audio(ref_wav)
        tgt_y, _ = _load_audio(tgt_wav)

    import librosa
    ref_env = librosa.onset.onset_strength(y=ref_y, sr=sr)
    tgt_env = librosa.onset.onset_strength(y=tgt_y, sr=sr)

    n = len(ref_env) + len(tgt_env) - 1
    fft_size = 1
    while fft_size < n:
        fft_size <<= 1

    ref_fft = np.fft.rfft(ref_env, fft_size)
    tgt_fft = np.fft.rfft(tgt_env, fft_size)
    corr = np.fft.irfft(ref_fft * np.conj(tgt_fft), fft_size)
    corr = np.concatenate([corr[-(len(tgt_env) - 1):], corr[:len(ref_env)]])

    hop_length = 512
    max_lag_frames = int(max_offset_sec * sr / hop_length)
    mid = len(tgt_env) - 1
    lo = max(0, mid - max_lag_frames)
    hi = min(len(corr), mid + max_lag_frames + 1)
    search = corr[lo:hi]
    best = int(np.argmax(search)) + lo - mid

    offset_sec = best * hop_length / sr
    log.info("Sync offset: %.4f s (%d onset frames)", offset_sec, best)
    return offset_sec


def compute_all_offsets(
    segment_first_per_camera: list[str],
    ffmpeg: str = "",
    max_offset_sec: float = 60.0,
    audio_duration_sec: float | None = 60.0,
) -> list[float]:
    if not segment_first_per_camera:
        return []
    offsets = [0.0]
    for i in range(1, len(segment_first_per_camera)):
        off = compute_sync_offset(
            segment_first_per_camera[0],
            segment_first_per_camera[i],
            ffmpeg=ffmpeg,
            max_offset_sec=max_offset_sec,
            audio_duration_sec=audio_duration_sec,
        )
        offsets.append(off)
    return offsets
