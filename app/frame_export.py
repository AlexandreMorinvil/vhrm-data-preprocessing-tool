from __future__ import annotations

import re
from pathlib import Path

import cv2


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_name(value: str, fallback: str) -> str:
    name = _SAFE_NAME_RE.sub("_", value.strip()).strip("._")
    return name or fallback


def _format_cursor_time(sec: float) -> str:
    ms_total = int(round(max(0.0, sec) * 1000))
    hours = ms_total // 3_600_000
    ms_total %= 3_600_000
    minutes = ms_total // 60_000
    ms_total %= 60_000
    seconds = ms_total // 1000
    millis = ms_total % 1000
    return f"{hours:02d}h{minutes:02d}m{seconds:02d}s{millis:03d}ms"


def _unique_dir(base_dir: Path, name: str) -> Path:
    candidate = base_dir / name
    if not candidate.exists():
        return candidate
    index = 2
    while True:
        candidate = base_dir / f"{name}_{index:02d}"
        if not candidate.exists():
            return candidate
        index += 1


def export_player_frames(player, output_root: str | Path, prefix: str = "capture") -> list[Path]:
    """Export the frame currently shown by each loaded camera preview."""
    previews = [
        preview for preview in player.previews
        if preview.video_path and Path(preview.video_path).exists() and preview.frame_count > 0
    ]
    if not previews:
        raise ValueError("No loaded camera videos to export.")

    fps = player.get_fps() or 30.0
    cursor_sec = player.current_frame / fps if fps > 0 else 0.0
    base_dir = Path(output_root) / "synchronized_captures"
    capture_dir = _unique_dir(
        base_dir,
        f"{_safe_name(prefix, 'capture')}_{_format_cursor_time(cursor_sec)}_frame{player.current_frame:06d}",
    )
    capture_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    failures: list[str] = []
    for index, preview in enumerate(previews, start=1):
        cap = cv2.VideoCapture(preview.video_path)
        if not cap.isOpened():
            failures.append(Path(preview.video_path).name)
            continue
        try:
            frame_no = max(0, min(preview.current_frame, preview.frame_count - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok:
            failures.append(Path(preview.video_path).name)
            continue

        camera_name = _safe_name(Path(preview.video_path).stem, f"camera{index:02d}")
        out_path = capture_dir / f"cam{index:02d}_{camera_name}_frame{frame_no:06d}.png"
        if not cv2.imwrite(str(out_path), frame):
            failures.append(out_path.name)
            continue
        written.append(out_path)

    if not written:
        raise RuntimeError("Could not export any camera frames.")
    if failures:
        failed = ", ".join(failures)
        raise RuntimeError(f"Exported {len(written)} frame(s), but failed for: {failed}")
    return written