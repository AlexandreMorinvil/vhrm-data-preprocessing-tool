from __future__ import annotations

import logging
import tempfile
from collections import deque
from itertools import combinations
from pathlib import Path

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


def _offsets_from_tree(
    tree_edges: tuple[tuple[int, int, float], ...],
    num_cameras: int,
) -> list[float] | None:
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(num_cameras)]
    for left, right, offset in tree_edges:
        adjacency[left].append((right, offset))
        adjacency[right].append((left, -offset))

    offsets: list[float | None] = [None] * num_cameras
    offsets[0] = 0.0
    pending: deque[int] = deque([0])
    while pending:
        current = pending.popleft()
        current_offset = offsets[current]
        if current_offset is None:
            continue
        for neighbor, relation in adjacency[current]:
            if offsets[neighbor] is None:
                offsets[neighbor] = current_offset + relation
                pending.append(neighbor)

    if any(offset is None for offset in offsets):
        return None
    return [float(offset) for offset in offsets]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _best_pairwise_offsets(
    relations: list[tuple[int, int, float]],
    num_cameras: int,
) -> tuple[list[float], tuple[float, float, float]]:
    if num_cameras <= 1:
        return [0.0] if num_cameras == 1 else [], (0.0, 0.0, 0.0)
    if num_cameras == 2:
        return [0.0, relations[0][2]], (0.0, 0.0, 0.0)

    best_offsets: list[float] | None = None
    best_score: tuple[float, float, float] | None = None
    for tree_edges in combinations(relations, num_cameras - 1):
        offsets = _offsets_from_tree(tree_edges, num_cameras)
        if offsets is None:
            continue

        residuals = [
            abs((offsets[right] - offsets[left]) - offset)
            for left, right, offset in relations
        ]
        score = (_median(residuals), float(np.mean(residuals)), max(residuals))
        if best_score is None or score < best_score:
            best_score = score
            best_offsets = offsets

    if best_offsets is None or best_score is None:
        raise RuntimeError("Could not derive a connected pairwise audio-sync solution.")
    return best_offsets, best_score


def compute_all_offsets(
    segment_first_per_camera: list[str],
    ffmpeg: str = "",
    max_offset_sec: float = 60.0,
    audio_duration_sec: float | None = 60.0,
    pairwise_refinement: bool = False,
    log_callback=None,
) -> list[float]:
    if not segment_first_per_camera:
        return []
    if len(segment_first_per_camera) == 1:
        return [0.0]
    if pairwise_refinement:
        relations: list[tuple[int, int, float]] = []
        total = len(segment_first_per_camera)
        for left in range(total):
            for right in range(left + 1, total):
                off = compute_sync_offset(
                    segment_first_per_camera[left],
                    segment_first_per_camera[right],
                    ffmpeg=ffmpeg,
                    max_offset_sec=max_offset_sec,
                    audio_duration_sec=audio_duration_sec,
                )
                relations.append((left, right, off))
                msg = f"Pairwise audio offset camera {left + 1}->{right + 1}: {off:.4f}s"
                log.info(msg)
                if log_callback is not None:
                    log_callback(msg)

        offsets, score = _best_pairwise_offsets(relations, total)
        msg = (
            "Pairwise audio refinement residuals "
            f"median={score[0]:.4f}s mean={score[1]:.4f}s max={score[2]:.4f}s"
        )
        log.info(msg)
        if log_callback is not None:
            log_callback(msg)
        return offsets

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
