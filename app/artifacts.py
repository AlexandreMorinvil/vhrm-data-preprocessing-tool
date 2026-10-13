"""Find and remove the processing artifacts left in a finished dataset folder.

Artifacts are files the pipeline needed but the dataset does not:

* the full-length synchronised videos (``cam1_<label>_final.mp4``) in the root
  of the output folder, once they have been cut into labelled segments;
* intermediate preprocessing videos (``_concat`` / ``_trimstart``) kept when
  intermediates were not deleted;
* the PPG alignment working files (``ppg_alignment/``: face-trace cache, ROI
  videos and the alignment log). The applied offset itself is also stored in
  every segment's ``meta.json`` and in the metadata sidecar.

They are either deleted or moved to a separate artifacts folder that mirrors
the dataset layout (``ppg_alignment/...`` stays in ``ppg_alignment/``, root
videos go to the root). Labelled segments are never touched. Moved artifacts
are listed in ``artifacts_manifest.json`` so they can be restored into the
dataset they came from.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

from .signal_offset import ALIGNMENT_DIRECTORY
from .state import ProjectState

log = logging.getLogger(__name__)

ARTIFACTS_SUFFIX = "_artifacts"
ARTIFACTS_MANIFEST = "artifacts_manifest.json"
SEGMENTS_DIRECTORY = "labelled_segments"

FULL_VIDEOS = "full_videos"
INTERMEDIATE_VIDEOS = "intermediate_videos"
ALIGNMENT_FILES = "alignment_files"

CATEGORY_TITLES = {
    FULL_VIDEOS: "Full-length videos",
    INTERMEDIATE_VIDEOS: "Intermediate videos",
    ALIGNMENT_FILES: "PPG alignment files",
}
CATEGORY_DESCRIPTIONS = {
    FULL_VIDEOS: "Synchronised and trimmed full-session videos (cam*_final) in the "
                 "dataset root. The labelled segments keep their own clips.",
    INTERMEDIATE_VIDEOS: "Concatenated / start-trimmed videos (cam*_concat, cam*_trimstart) "
                         "kept from preprocessing.",
    ALIGNMENT_FILES: f"Everything in {ALIGNMENT_DIRECTORY}/: face-trace cache, ROI videos "
                     "and the alignment log. Applied offsets stay in the segments' meta.json.",
}
CATEGORIES = (FULL_VIDEOS, ALIGNMENT_FILES, INTERMEDIATE_VIDEOS)

_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
_FINAL_RE = re.compile(r"^cam\d+_.*_final$", re.IGNORECASE)
_INTERMEDIATE_RE = re.compile(r"^cam\d+_.*_(concat|trimstart)$", re.IGNORECASE)
_PARTIAL_SUFFIX = ".partial"
_COPY_CHUNK = 8 * 1024 * 1024


@dataclass
class ArtifactFile:
    path: Path
    relative: Path
    size: int
    category: str


@dataclass
class CleanupResult:
    done: list[ArtifactFile] = field(default_factory=list)
    failures: list[tuple[ArtifactFile, str]] = field(default_factory=list)
    cancelled: bool = False

    @property
    def bytes_done(self) -> int:
        return sum(item.size for item in self.done)


@dataclass
class ArtifactsFolderScan:
    """Content of an artifacts folder, as candidates for a restore."""
    found: dict[str, list[ArtifactFile]]
    ignored: list[Path] = field(default_factory=list)  # relative paths that are not artifacts
    manifest: Optional[dict] = None


def _empty_categories() -> dict[str, list[ArtifactFile]]:
    return {category: [] for category in CATEGORIES}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def dataset_directory(state: ProjectState) -> Optional[Path]:
    """The project output folder (the folder containing ``labelled_segments``)."""
    if state.output_directory:
        return Path(state.output_directory)
    if state.segments_manifest_path:
        return Path(state.segments_manifest_path).parent.parent
    return None


def default_artifacts_name(dataset_dir: str | Path) -> str:
    return f"{Path(dataset_dir).resolve().name}{ARTIFACTS_SUFFIX}"


def segments_directory(dataset_dir: str | Path, state: Optional[ProjectState] = None) -> Path:
    if state is not None and state.segments_manifest_path:
        manifest_dir = Path(state.segments_manifest_path).parent
        if manifest_dir.parent.resolve() == Path(dataset_dir).resolve():
            return manifest_dir
    return Path(dataset_dir) / SEGMENTS_DIRECTORY


def project_video_names(state: Optional[ProjectState]) -> tuple[set[str], set[str]]:
    """Lower-case file names of the project's final and intermediate videos."""
    finals: set[str] = set()
    intermediates: set[str] = set()
    if state is not None:
        for track in state.tracks:
            for value, bucket in ((track.final_output_path, finals),
                                  (track.concatenated_path, intermediates),
                                  (track.trimstart_path, intermediates)):
                if value:
                    bucket.add(Path(value).name.lower())
    return finals, intermediates


def _root_video_category(path: Path, finals: set[str], intermediates: set[str]) -> Optional[str]:
    if path.suffix.lower() not in _VIDEO_SUFFIXES:
        return None
    name = path.name.lower()
    if name in finals or _FINAL_RE.match(path.stem):
        return FULL_VIDEOS
    if name in intermediates or _INTERMEDIATE_RE.match(path.stem):
        return INTERMEDIATE_VIDEOS
    return None


def _artifact(path: Path, root: Path, category: str) -> ArtifactFile:
    return ArtifactFile(path=path, relative=path.relative_to(root), size=path.stat().st_size,
                        category=category)


def find_artifacts(dataset_dir: str | Path,
                   state: Optional[ProjectState] = None) -> dict[str, list[ArtifactFile]]:
    """Return the artifacts of every category found in *dataset_dir*.

    Only files directly in the dataset root (videos) and inside
    ``ppg_alignment/`` are considered, so the labelled segments are never
    listed.
    """
    root = Path(dataset_dir).resolve()
    found = _empty_categories()
    if not root.is_dir():
        return found
    # Names recorded by the project, in case the cameras were renamed to an
    # unusual pattern.
    finals, intermediates = project_video_names(state)
    for path in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        category = _root_video_category(path, finals, intermediates) if path.is_file() else None
        if category:
            found[category].append(_artifact(path.resolve(), root, category))

    alignment = root / ALIGNMENT_DIRECTORY
    if alignment.is_dir():
        for path in sorted(alignment.rglob("*"), key=lambda p: str(p).lower()):
            if path.is_file():
                found[ALIGNMENT_FILES].append(_artifact(path.resolve(), root, ALIGNMENT_FILES))
    return found


def scan_artifacts_folder(artifacts_dir: str | Path,
                          state: Optional[ProjectState] = None) -> ArtifactsFolderScan:
    """List the files of an artifacts folder that can go back into a dataset.

    Files recorded in the manifest keep their category; others are recognised
    with the same rules as in the dataset. Anything else (and leftovers of an
    interrupted copy) is ignored and stays where it is.
    """
    root = Path(artifacts_dir).resolve()
    scan = ArtifactsFolderScan(found=_empty_categories())
    if not root.is_dir():
        return scan
    scan.manifest = read_manifest(root)
    listed = (scan.manifest or {}).get("files", {})
    finals, intermediates = project_video_names(state)
    for path in sorted(root.rglob("*"), key=lambda p: str(p).lower()):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative == Path(ARTIFACTS_MANIFEST):
            continue
        category = None
        recorded = listed.get(relative.as_posix(), {}).get("category")
        if path.name.endswith(_PARTIAL_SUFFIX):
            category = None
        elif recorded in CATEGORIES:
            category = recorded
        elif relative.parts[0] == ALIGNMENT_DIRECTORY and len(relative.parts) > 1:
            category = ALIGNMENT_FILES
        elif len(relative.parts) == 1:
            category = _root_video_category(path, finals, intermediates)
        if category:
            scan.found[category].append(_artifact(path.resolve(), root, category))
        else:
            scan.ignored.append(relative)
    return scan


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def read_manifest(artifacts_dir: str | Path) -> Optional[dict]:
    path = Path(artifacts_dir) / ARTIFACTS_MANIFEST
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("Ignoring unreadable %s: %s", path, exc)
        return None
    if not isinstance(data, dict) or not isinstance(data.get("files", {}), dict):
        log.warning("Ignoring malformed %s", path)
        return None
    data.setdefault("files", {})
    return data


def _update_manifest(artifacts_dir: Path, dataset_dir: Path,
                     added: Iterable[ArtifactFile] = (), removed: Iterable[ArtifactFile] = ()) -> None:
    added = list(added)
    removed = list(removed)
    if not added and not removed:
        return
    manifest = read_manifest(artifacts_dir) or {"files": {}}
    if added:
        manifest["dataset"] = str(dataset_dir.resolve())
        manifest["dataset_name"] = dataset_dir.resolve().name
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for item in added:
        manifest["files"][item.relative.as_posix()] = {
            "size": item.size, "category": item.category, "moved_at": now,
        }
    for item in removed:
        manifest["files"].pop(item.relative.as_posix(), None)
    path = artifacts_dir / ARTIFACTS_MANIFEST
    if not manifest["files"]:
        path.unlink(missing_ok=True)
        return
    temporary = path.with_name(path.name + _PARTIAL_SUFFIX)
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def destination_problem(dataset_dir: str | Path, destination: str | Path) -> Optional[str]:
    """Why *destination* cannot be the artifacts folder of the dataset, or None when it can."""
    dataset = Path(dataset_dir)
    target = Path(destination)
    if not str(destination).strip() or not target.is_absolute():
        return "Choose a destination folder."
    if _is_within(target, dataset):
        return "The artifacts folder must be outside the dataset folder."
    if _is_within(dataset, target):
        return "The artifacts folder cannot contain the dataset folder."
    if target.exists() and not target.is_dir():
        return f"{target} exists and is not a folder."
    return None


def _existing_ancestor(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if candidate.exists():
            return candidate
    return path


def space_problem(files: Iterable[ArtifactFile], target_root: str | Path) -> Optional[str]:
    """Whether the target drive is too small for the files that must be copied."""
    anchor = _existing_ancestor(Path(target_root))
    needed = sum(item.size for item in files if not _same_volume(item.path, anchor))
    if not needed:
        return None
    try:
        free = shutil.disk_usage(anchor).free
    except OSError:
        return None
    if needed > free:
        return (f"Not enough free space on {anchor.anchor or anchor}: {format_size(needed)} needed, "
                f"{format_size(free)} available.")
    return None


def restore_checks(artifacts_dir: str | Path, dataset_dir: str | Path, files: list[ArtifactFile],
                   manifest: Optional[dict],
                   state: Optional[ProjectState] = None) -> tuple[list[str], list[str]]:
    """Errors that prevent restoring *files*, and warnings to confirm first."""
    artifacts = Path(artifacts_dir)
    dataset = Path(dataset_dir)
    errors: list[str] = []
    warnings: list[str] = []
    if not dataset.is_dir():
        errors.append("The dataset folder does not exist.")
    if not artifacts.is_dir():
        errors.append("The artifacts folder does not exist.")
    if errors:
        return errors, warnings
    problem = destination_problem(dataset, artifacts)
    if problem:
        errors.append(problem)
    if not files:
        errors.append("There is nothing to restore.")
    problem = space_problem(files, dataset)
    if problem:
        errors.append(problem)

    if manifest is None:
        warnings.append(f"No {ARTIFACTS_MANIFEST} in this folder: it cannot be checked that the "
                        "files come from this dataset.")
    else:
        origin = manifest.get("dataset") or ""
        name = manifest.get("dataset_name") or (Path(origin).name if origin else "")
        if origin and Path(origin).resolve() == dataset.resolve():
            pass
        elif name and name.lower() != dataset.resolve().name.lower():
            warnings.append(f"These artifacts were moved out of the dataset '{name}' ({origin}), "
                            f"not out of '{dataset.resolve().name}'.")
        elif origin:
            warnings.append(f"These artifacts were moved out of {origin}; the dataset has since "
                            "moved or been renamed.")
        listed = manifest.get("files", {})
        changed = [item.relative for item in files
                   if item.relative.as_posix() in listed
                   and listed[item.relative.as_posix()].get("size") not in (None, item.size)]
        if changed:
            warnings.append(f"{len(changed)} file(s) changed size since they were moved: "
                            + ", ".join(str(path) for path in changed[:5])
                            + (" …" if len(changed) > 5 else ""))
        present = {item.relative.as_posix() for item in files}
        missing = [key for key in listed if key not in present and not (artifacts / key).exists()]
        if missing:
            warnings.append(f"{len(missing)} file(s) recorded as moved are no longer in the artifacts "
                            "folder: " + ", ".join(missing[:5]) + (" …" if len(missing) > 5 else ""))

    finals, _intermediates = project_video_names(state)
    if finals:
        foreign = [item.relative for item in files
                   if item.category == FULL_VIDEOS and item.relative.name.lower() not in finals]
        if foreign:
            warnings.append("Not videos of the open project: "
                            + ", ".join(str(path) for path in foreign[:5])
                            + (" …" if len(foreign) > 5 else ""))
    return errors, warnings


def _check_removable(item: ArtifactFile, dataset_dir: Path, segments_dir: Path) -> None:
    if not _is_within(item.path, dataset_dir) or _is_within(item.path, segments_dir):
        raise ValueError(f"Refusing to touch {item.path}: not a dataset artifact.")


def existing_destinations(files: Iterable[ArtifactFile], destination: str | Path) -> list[Path]:
    return [Path(destination) / item.relative for item in files
            if (Path(destination) / item.relative).exists()]


# ---------------------------------------------------------------------------
# Delete / move / restore
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[int, int, str], None]


def _remove_empty_parents(paths: Iterable[Path], stop_at: Path) -> None:
    """Remove directories emptied by the clean-up, up to (not including) *stop_at*."""
    stop = stop_at.resolve()
    directories = {parent for path in paths for parent in path.parents
                   if parent != stop and _is_within(parent, stop)}
    for directory in sorted(directories, key=lambda d: len(d.parts), reverse=True):
        try:
            directory.rmdir()  # only succeeds when empty
        except OSError:
            pass


def _copy_then_remove(source: Path, target: Path, report: Callable[[int], None],
                      is_cancelled: Callable[[], bool]) -> bool:
    """Copy across volumes with progress; returns False when cancelled."""
    partial = target.with_name(target.name + _PARTIAL_SUFFIX)
    try:
        with open(source, "rb") as src, open(partial, "wb") as dst:
            while True:
                if is_cancelled():
                    raise InterruptedError
                chunk = src.read(_COPY_CHUNK)
                if not chunk:
                    break
                dst.write(chunk)
                report(len(chunk))
        shutil.copystat(source, partial)
        os.replace(partial, target)
    except InterruptedError:
        partial.unlink(missing_ok=True)
        return False
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    try:
        source.unlink()
    except OSError:
        target.unlink(missing_ok=True)  # keep a single copy, at the source
        raise
    return True


def _same_volume(source: Path, target_dir: Path) -> bool:
    try:
        return os.stat(source).st_dev == os.stat(target_dir).st_dev
    except OSError:
        return False


def _move_file(source: Path, target: Path, overwrite: bool, report: Callable[[int], None],
               is_cancelled: Callable[[], bool]) -> bool:
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if _same_volume(source, target.parent):
        os.replace(source, target)
        return True
    return _copy_then_remove(source, target, report, is_cancelled)


def _run(files: list[ArtifactFile], source_root: Path, action, check: Callable[[ArtifactFile], None],
         progress: Optional[ProgressCallback], is_cancelled: Callable[[], bool]) -> CleanupResult:
    total = sum(item.size for item in files) or 1
    done_bytes = 0
    result = CleanupResult()
    for item in files:
        if is_cancelled():
            result.cancelled = True
            break
        name = str(item.relative)
        copied = 0

        def report(count: int) -> None:
            nonlocal copied
            copied += count
            if progress is not None:
                progress(done_bytes + copied, total, name)

        try:
            check(item)
            if progress is not None:
                progress(done_bytes, total, name)
            if not action(item, report):
                result.cancelled = True
                break
            result.done.append(item)
        except (OSError, ValueError) as exc:
            result.failures.append((item, str(exc)))
        done_bytes += item.size
        if progress is not None:
            progress(done_bytes, total, name)
    _remove_empty_parents([item.path for item in result.done], source_root)
    return result


def delete_artifacts(files: list[ArtifactFile], dataset_dir: str | Path,
                     progress: Optional[ProgressCallback] = None,
                     is_cancelled: Callable[[], bool] = lambda: False,
                     segments_dir: Optional[Path] = None) -> CleanupResult:
    root = Path(dataset_dir).resolve()
    segments = (segments_dir or root / SEGMENTS_DIRECTORY).resolve()

    def delete(item: ArtifactFile, _report) -> bool:
        item.path.unlink()
        return True

    return _run(files, root, delete, lambda item: _check_removable(item, root, segments),
                progress, is_cancelled)


def move_artifacts(files: list[ArtifactFile], dataset_dir: str | Path, destination: str | Path,
                   overwrite: bool = False,
                   progress: Optional[ProgressCallback] = None,
                   is_cancelled: Callable[[], bool] = lambda: False,
                   segments_dir: Optional[Path] = None) -> CleanupResult:
    """Move *files* below *destination*, keeping their dataset-relative paths."""
    problem = destination_problem(dataset_dir, destination)
    if problem:
        raise ValueError(problem)
    root = Path(dataset_dir).resolve()
    segments = (segments_dir or root / SEGMENTS_DIRECTORY).resolve()
    target_root = Path(destination)

    def move(item: ArtifactFile, report) -> bool:
        return _move_file(item.path, target_root / item.relative, overwrite, report, is_cancelled)

    result = _run(files, root, move, lambda item: _check_removable(item, root, segments),
                  progress, is_cancelled)
    if result.done:
        _update_manifest(target_root.resolve(), root, added=result.done)
    return result


def restore_artifacts(files: list[ArtifactFile], artifacts_dir: str | Path, dataset_dir: str | Path,
                      overwrite: bool = False,
                      progress: Optional[ProgressCallback] = None,
                      is_cancelled: Callable[[], bool] = lambda: False,
                      segments_dir: Optional[Path] = None) -> CleanupResult:
    """Move *files* from *artifacts_dir* back to the same place in *dataset_dir*.

    The artifacts folder is removed when nothing is left in it.
    """
    problem = destination_problem(dataset_dir, artifacts_dir)
    if problem:
        raise ValueError(problem)
    source_root = Path(artifacts_dir).resolve()
    dataset = Path(dataset_dir).resolve()
    segments = (segments_dir or dataset / SEGMENTS_DIRECTORY).resolve()

    def check(item: ArtifactFile) -> None:
        target = dataset / item.relative
        if (not _is_within(item.path, source_root) or item.relative == Path(ARTIFACTS_MANIFEST)
                or not _is_within(target, dataset) or _is_within(target, segments)):
            raise ValueError(f"Refusing to restore {item.relative}: not a dataset artifact.")

    def restore(item: ArtifactFile, report) -> bool:
        return _move_file(item.path, dataset / item.relative, overwrite, report, is_cancelled)

    result = _run(files, source_root, restore, check, progress, is_cancelled)
    _update_manifest(source_root, dataset, removed=result.done)
    try:
        source_root.rmdir()  # only succeeds when everything went back
    except OSError:
        pass
    return result


def format_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    value = float(size)
    for unit in ("KB", "MB", "GB", "TB"):
        value /= 1024
        if value < 1024 or unit == "TB":
            break
    return f"{value:.1f} {unit}"
