from __future__ import annotations

from pathlib import Path
import subprocess
import threading

import cv2
import numpy as np


_FACE_MARGIN_RATIO = 0.22
_PIXEL_BLOCKS = 10
_DETECTION_MAX_DIMENSION = 960
_YUNET_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "face_detection_yunet_2023mar.onnx"
)
_detector_local = threading.local()


def _yunet_detector() -> cv2.FaceDetectorYN:
    detector = getattr(_detector_local, "yunet", None)
    if detector is None:
        if not _YUNET_MODEL_PATH.exists():
            raise RuntimeError(f"Face privacy model not found: {_YUNET_MODEL_PATH}")
        detector = cv2.FaceDetectorYN.create(
            str(_YUNET_MODEL_PATH),
            "",
            (320, 320),
            0.5,
            0.3,
            5000,
            cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU,
        )
        _detector_local.yunet = detector
    return detector


def _detect_faces(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    scale = min(1.0, _DETECTION_MAX_DIMENSION / max(width, height))
    if scale < 1.0:
        sample = cv2.resize(
            frame,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        sample = frame

    detector = _yunet_detector()
    detector.setInputSize((sample.shape[1], sample.shape[0]))
    _status, faces = detector.detect(sample)
    if faces is None:
        return []

    inverse_scale = 1.0 / scale
    return [
        tuple(
            int(round(float(value) * inverse_scale))
            for value in face[:4]
        )
        for face in faces
    ]


def _expanded_region(
    face: tuple[int, int, int, int], frame_width: int, frame_height: int
) -> tuple[int, int, int, int]:
    x, y, width, height = face
    margin_x = int(round(width * _FACE_MARGIN_RATIO))
    margin_y = int(round(height * _FACE_MARGIN_RATIO))
    return (
        max(0, x - margin_x),
        max(0, y - margin_y),
        min(frame_width, x + width + margin_x),
        min(frame_height, y + height + margin_y),
    )


def anonymize_faces(frame: np.ndarray) -> np.ndarray:
    """Return a copy of a BGR frame with detected faces strongly pixelated."""
    if frame.size == 0:
        return frame.copy()

    result = frame.copy()
    height, width = frame.shape[:2]
    for face in _detect_faces(frame):
        x0, y0, x1, y1 = _expanded_region(face, width, height)
        region = result[y0:y1, x0:x1]
        if region.size == 0:
            continue
        block_width = max(1, min(_PIXEL_BLOCKS, region.shape[1]))
        block_height = max(1, min(_PIXEL_BLOCKS, region.shape[0]))
        pixelated = cv2.resize(
            region, (block_width, block_height), interpolation=cv2.INTER_LINEAR
        )
        result[y0:y1, x0:x1] = cv2.resize(
            pixelated,
            (region.shape[1], region.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    return result


def export_anonymized_video_segment(
    source_path: str,
    output_path: str,
    start_sec: float,
    duration_sec: float,
    ffmpeg: str,
) -> None:
    """Encode an anonymized video segment while copying its source audio."""
    capture = cv2.VideoCapture(source_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid video dimensions: {source_path}")

    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_sec) * 1000.0)
    frame_limit = max(0, int(round(duration_sec * fps)))
    command = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-ss", str(max(0.0, start_sec)),
        "-t", str(max(0.0, duration_sec)),
        "-i", source_path,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        for _ in range(frame_limit):
            ok, frame = capture.read()
            if not ok:
                break
            process.stdin.write(anonymize_faces(frame).tobytes())  # type: ignore[union-attr]
    except BrokenPipeError:
        pass
    finally:
        capture.release()
        if process.stdin is not None:
            process.stdin.close()
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg anonymized export failed:\n{stderr}")