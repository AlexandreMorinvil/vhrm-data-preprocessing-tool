from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)


def find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    return path if path else ""


def find_ffprobe() -> str:
    path = shutil.which("ffprobe")
    return path if path else ""


def probe_video(path: str, ffprobe: str = "") -> dict[str, Any]:
    ffprobe = ffprobe or find_ffprobe()
    if not ffprobe:
        raise RuntimeError("ffprobe not found")
    cmd = [
        ffprobe, "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr}")
    data = json.loads(result.stdout)

    video_stream = None
    audio_stream = None
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and video_stream is None:
            video_stream = s
        if s.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = s

    fmt = data.get("format", {})
    info: dict[str, Any] = {
        "path": path,
        "file_size": os.path.getsize(path),
        "duration": float(fmt.get("duration", 0)),
        "format_name": fmt.get("format_name", ""),
    }

    if video_stream:
        info["width"] = int(video_stream.get("width", 0))
        info["height"] = int(video_stream.get("height", 0))
        info["codec"] = video_stream.get("codec_name", "")
        r_frame_rate = video_stream.get("r_frame_rate", "0/1")
        num, den = r_frame_rate.split("/")
        info["fps"] = float(num) / float(den) if float(den) else 0.0
        info["frame_count"] = int(video_stream.get("nb_frames", 0))
        if info["frame_count"] == 0 and info["fps"] > 0:
            info["frame_count"] = int(info["duration"] * info["fps"])
    else:
        info.update({"width": 0, "height": 0, "codec": "", "fps": 0.0, "frame_count": 0})

    info["has_audio"] = audio_stream is not None
    return info


def _run_ffmpeg(args: list[str], ffmpeg: str = "", log_callback=None) -> str:
    ffmpeg = ffmpeg or find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")
    cmd = [ffmpeg, "-y"] + args
    log.info("FFmpeg: %s", " ".join(cmd))
    if log_callback:
        log_callback(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        err = proc.stderr
        log.error("FFmpeg failed: %s", err)
        raise RuntimeError(f"FFmpeg error:\n{err}")
    return proc.stderr


def _drain_pipe(pipe) -> list[str]:
    """Read *pipe* in a background thread and return a list holding the result.

    Returns a 1-element list; the caller should ``join()`` the thread then
    read ``result[0]``.  This prevents the classic deadlock where both stdout
    and stderr are piped but only one is consumed.
    """
    result: list[str] = [""]

    def _reader():
        try:
            result[0] = pipe.read()
        except Exception:
            pass

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return result, t  # type: ignore[return-value]


def concatenate_segments(
    segment_paths: list[str],
    output_path: str,
    ffmpeg: str = "",
    log_callback=None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> str:
    """Concatenate video segments.

    *progress_callback(current_bytes, total_bytes)* is called every ~500 ms
    with file-size progress (only when there are multiple segments).
    """
    if len(segment_paths) == 1:
        shutil.copy2(segment_paths[0], output_path)
        return output_path

    expected_size = sum(os.path.getsize(p) for p in segment_paths)

    list_file = output_path + ".concat.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for seg in segment_paths:
            escaped = seg.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    try:
        if progress_callback is not None:
            # Use Popen so we can poll file size while ffmpeg runs
            ffmpeg = ffmpeg or find_ffmpeg()
            if not ffmpeg:
                raise RuntimeError("ffmpeg not found")
            cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0",
                   "-i", list_file, "-c", "copy", output_path]
            log.info("FFmpeg: %s", " ".join(cmd))
            if log_callback:
                log_callback(f"Running: {' '.join(cmd)}")
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE)
            stderr_buf, stderr_thread = _drain_pipe(proc.stderr)
            while proc.poll() is None:
                if cancel_check and cancel_check():
                    proc.kill()
                    proc.wait()
                    stderr_thread.join(timeout=2)
                    raise RuntimeError("Cancelled by user")
                try:
                    cur = os.path.getsize(output_path) if os.path.exists(output_path) else 0
                except OSError:
                    cur = 0
                progress_callback(cur, expected_size)
                time.sleep(0.5)
            stderr_thread.join(timeout=5)
            if proc.returncode != 0:
                err = stderr_buf[0] if isinstance(stderr_buf[0], str) else stderr_buf[0].decode(errors="replace")
                log.error("FFmpeg failed: %s", err)
                raise RuntimeError(f"FFmpeg error:\n{err}")
            progress_callback(expected_size, expected_size)
        else:
            _run_ffmpeg(
                ["-f", "concat", "-safe", "0", "-i", list_file,
                 "-c", "copy", output_path],
                ffmpeg=ffmpeg, log_callback=log_callback,
            )
    finally:
        if os.path.exists(list_file):
            os.remove(list_file)
    return output_path


def trim_video(
    input_path: str,
    output_path: str,
    start_sec: float = 0.0,
    duration_sec: float = 0.0,
    ffmpeg: str = "",
    log_callback=None,
    progress_callback: Optional[Callable[[float, float], None]] = None,
    expected_duration_sec: float = 0.0,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> str:
    """Trim a video with ``-c copy``.

    When *progress_callback* and *expected_duration_sec* are provided, FFmpeg
    is started with ``-progress pipe:1`` and *progress_callback(current_sec,
    total_sec)* is called as ``out_time_us`` lines arrive.
    """
    ffmpeg = ffmpeg or find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")

    args: list[str] = []
    if start_sec > 0:
        args += ["-ss", f"{start_sec:.6f}"]
    args += ["-i", input_path]
    if duration_sec > 0:
        args += ["-t", f"{duration_sec:.6f}"]
    args += ["-c", "copy"]

    use_progress = progress_callback is not None and expected_duration_sec > 0
    if use_progress:
        args += ["-progress", "pipe:1"]

    cmd = [ffmpeg, "-y"] + args + [output_path]
    log.info("FFmpeg: %s", " ".join(cmd))
    if log_callback:
        log_callback(f"Running: {' '.join(cmd)}")

    if use_progress:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        stderr_buf, stderr_thread = _drain_pipe(proc.stderr)
        for line in proc.stdout:  # type: ignore[union-attr]
            if cancel_check and cancel_check():
                proc.kill()
                proc.wait()
                stderr_thread.join(timeout=2)
                raise RuntimeError("Cancelled by user")
            line = line.strip()
            if line.startswith("out_time_us="):
                try:
                    us = int(line.split("=", 1)[1])
                    progress_callback(us / 1_000_000, expected_duration_sec)
                except (ValueError, IndexError):
                    pass
        proc.wait()
        stderr_thread.join(timeout=5)
        if proc.returncode != 0:
            err = stderr_buf[0]
            log.error("FFmpeg failed: %s", err)
            raise RuntimeError(f"FFmpeg error:\n{err}")
        progress_callback(expected_duration_sec, expected_duration_sec)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            log.error("FFmpeg failed: %s", result.stderr)
            raise RuntimeError(f"FFmpeg error:\n{result.stderr}")
    return output_path


def extract_audio(
    input_path: str,
    output_path: str,
    ffmpeg: str = "",
    log_callback=None,
) -> str:
    _run_ffmpeg(
        ["-i", input_path, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", output_path],
        ffmpeg=ffmpeg,
        log_callback=log_callback,
    )
    return output_path


def get_frame_count(path: str, ffprobe: str = "") -> int:
    info = probe_video(path, ffprobe)
    return info.get("frame_count", 0)


class FFmpegWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)
    log_message = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def run(self) -> None:
        try:
            self._func(*self._args, worker=self, **self._kwargs)
            if self._cancelled:
                self.finished.emit(False, "Cancelled by user.")
            else:
                self.finished.emit(True, "Completed successfully.")
        except Exception as exc:
            log.exception("FFmpegWorker error")
            self.finished.emit(False, str(exc))
