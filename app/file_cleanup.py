from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path


ARCHIVE_DIRECTORY_NAME = "obsolete_files"


def remove_or_archive(
    path: str | Path,
    project_root: str | Path,
    archive: bool,
) -> Path | None:
    source = Path(path)
    if not source.exists():
        return None
    if not archive:
        if source.is_dir():
            shutil.rmtree(source)
        else:
            source.unlink()
        return None

    root = Path(project_root).resolve()
    archive_root = root / ARCHIVE_DIRECTORY_NAME
    try:
        relative = source.resolve().relative_to(root)
    except ValueError:
        relative = Path("external") / source.name

    destination = archive_root / relative
    if destination.exists():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = destination.with_name(
            f"{destination.stem}_{timestamp}{destination.suffix}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(shutil.move(str(source), str(destination)))


def cleanup_obsolete_paths(
    paths: list[str | Path],
    project_root: str | Path,
    archive: bool,
) -> int:
    root = Path(project_root).resolve()
    cleaned = 0
    seen: set[Path] = set()
    for path in paths:
        source = Path(path)
        try:
            resolved = source.resolve()
            resolved.relative_to(root)
        except ValueError:
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        remove_or_archive(resolved, root, archive)
        cleaned += 1
    return cleaned