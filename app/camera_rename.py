"""Transactional camera naming updates, without modifying original recordings."""
from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from .state import ProjectState, _relativise_paths


def validate_camera_labels(labels: list[str]) -> list[str]:
    """Require labels usable verbatim in the application's Windows filenames."""
    cleaned = [label.strip() for label in labels]
    for label in cleaned:
        if (not label or label in {".", ".."} or label.endswith(".")
                or re.search(r'[<>:"/\\|?*\x00-\x1f]', label)):
            raise ValueError("Camera names must be non-empty and cannot contain "
                             'Windows filename characters: < > : " / \\ | ? *')
        if len(label) > 100:
            raise ValueError("Camera names must be at most 100 characters long.")
    if len({label.casefold() for label in cleaned}) != len(cleaned):
        raise ValueError("Each camera must have a different name.")
    return cleaned


class CameraRenameCancelled(Exception):
    """The operation was cancelled and any changes have been rolled back."""


def rename_cameras(
    state: ProjectState, labels: list[str], *,
    progress: Callable[[int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    path_changes: dict[str, str] | None = None,
) -> int:
    """Rename generated camera files and persist references as one transaction.

    Returns the number of renamed files. Callers must release video decoders and
    stop export workers first. Existing JSON fields are preserved; original
    segment recordings and sensor files are never renamed. On an I/O failure,
    restore the previous files and metadata before changing shared state.
    """
    def report(percent: int, message: str, *, check_cancel: bool = True) -> None:
        if progress is not None:
            progress(percent, message)
        if check_cancel and cancelled is not None and cancelled():
            raise CameraRenameCancelled("Camera rename cancelled; previous files and metadata restored.")

    report(0, "Checking camera names…")
    if len(labels) != len(state.tracks):
        raise ValueError("Provide one name for each camera in the project.")
    labels = validate_camera_labels(labels)
    changes = [(i, track.camera_label, labels[i])
               for i, track in enumerate(state.tracks)
               if track.camera_label != labels[i]]
    if not changes:
        return 0

    moves: dict[Path, Path] = {}
    source_recordings = {Path(p).resolve() for t in state.tracks for p in t.segment_paths}

    def add_move(source: Path, target: Path) -> None:
        source = source.resolve()
        # Resolve the directory only: resolving an existing case-only target on
        # Windows would silently recover the OLD spelling of its filename.
        target = target.parent.resolve() / target.name
        if str(source) == str(target):
            return
        if source.resolve() in source_recordings:
            raise ValueError(f"Cannot rename an original source recording: {source}")
        if source in moves and str(moves[source]) != str(target):
            raise ValueError(f"Two cameras reference the same generated file: {source}")
        moves[source] = target

    # Include retained intermediates and the actual imported final-video path,
    # even if it did not originally use the application's naming convention.
    for i, old, new in changes:
        track = state.tracks[i]
        report(5, f"Finding generated files for {old}…")
        for field in ("final_output_path", "concatenated_path", "trimstart_path"):
            value = getattr(track, field)
            if not value:
                continue
            source = Path(value)
            prefix = f"cam{i + 1}_{old}"
            if source.stem == prefix or source.stem.startswith(prefix + "_"):
                name = f"cam{i + 1}_{new}" + source.name[len(prefix):]
            elif field == "final_output_path":
                name = f"cam{i + 1}_{new}_final{source.suffix}"
            else:
                continue
            add_move(source, source.with_name(name))

    # Only discover project sidecars and labelled clips, not unrelated JSON or
    # arbitrary similarly named files elsewhere on disk.
    roots = {Path(state.output_directory).resolve()} if state.output_directory else set()
    roots.update(Path(t.final_output_path).resolve().parent
                 for t in state.tracks if t.final_output_path)
    if state.metadata_path:
        roots.add(Path(state.metadata_path).resolve().parent)
    folders = {Path(iv.folder).resolve() for iv in state.intervals if iv.folder}
    if state.segments_manifest_path and Path(state.segments_manifest_path).parent.is_dir():
        folders.update(p for p in Path(state.segments_manifest_path).resolve().parent.iterdir()
                       if p.is_dir())
    for root in roots:
        report(10, f"Finding labelled segments in {root}…")
        segment_root = root / "labelled_segments"
        if segment_root.is_dir():
            folders.update(p for p in segment_root.iterdir() if p.is_dir())

    documents: dict[Path, dict] = {}
    metadata_files = {p for root in roots for p in root.glob("*_meta.json")}
    metadata_files.update(p for root in roots for p in root.glob("*.vrt-meta.json"))
    metadata_files.update(folder / "meta.json" for folder in folders
                          if (folder / "meta.json").exists())
    if state.metadata_path:
        metadata_files.add(Path(state.metadata_path).resolve())
    for path in sorted(metadata_files):
        report(15, f"Reading camera metadata: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid camera metadata: {path}")
        cameras = raw.get("cameras", [])
        relevant = False
        for i, old, new in changes:
            if i >= len(cameras) or cameras[i].get("label") != old:
                continue
            camera = cameras[i]
            # Match sidecars by actual final path, not just a common label.
            value = camera.get("final_video_path")
            if value and (path.parent / value).resolve() != Path(state.tracks[i].final_output_path).resolve():
                continue
            camera["label"] = new
            relevant = True
            clip = camera.get("video_file")
            if clip:
                source = path.parent / clip
                target = source.with_name(f"cam{i + 1}_{new}{source.suffix}")
                add_move(source, target)
        if relevant:
            documents[path] = raw

    # Legacy clips may have no metadata at all.
    for folder in folders:
        report(20, f"Checking labelled clips: {folder}")
        for i, old, new in changes:
            for source in folder.glob(f"cam{i + 1}_*.*"):
                if source.stem == f"cam{i + 1}_{old}" and source.suffix.lower() in {
                    ".mp4", ".mov", ".mkv", ".avi", ".m4v"
                }:
                    add_move(source, source.with_name(f"cam{i + 1}_{new}{source.suffix}"))

    # Individual frame captures encode the video's sanitized stem in their
    # filename. Composite snapshots contain rendered labels and need re-export.
    capture_roots = {root / "synchronized_captures" for root in roots | folders}
    capture_roots = {root for root in capture_roots if root.is_dir()}
    if capture_roots:
        from .frame_export import _safe_name

        stem_changes = {
            _safe_name(source.stem, "camera"): _safe_name(target.stem, "camera")
            for source, target in moves.items()
        }
        for root in capture_roots:
            report(25, f"Scanning captured frames: {root}")
            for source in root.rglob("*.png"):
                report(25, f"Checking captured frame: {source.name}")
                for old_stem, new_stem in stem_changes.items():
                    match = re.fullmatch(
                        rf"(cam\d+_){re.escape(old_stem)}(_frame\d+\.png)", source.name
                    )
                    if match:
                        add_move(source, source.with_name(
                            f"{match[1]}{new_stem}{match[2]}"
                        ))
                        break

    actual_moves = {s: t for s, t in moves.items() if s.exists()}
    report(30, f"Checking conflicts for {len(actual_moves)} files…")
    targets = [os.path.normcase(str(p)) for p in moves.values()]
    if len(targets) != len(set(targets)):
        raise ValueError("Renaming would give two files the same filename.")
    for source, target in moves.items():
        report(30, f"Checking destination: {target.name}")
        if target.exists() and target not in actual_moves:
            raise FileExistsError(f"Cannot rename camera: destination already exists:\n{target}")

    source_names = {source.name.casefold() for source in moves}

    def update_paths(value, base: Path):
        if isinstance(value, dict):
            return {k: update_paths(v, base) for k, v in value.items()}
        if isinstance(value, list):
            return [update_paths(v, base) for v in value]
        if isinstance(value, str) and value:
            if Path(value).name.casefold() not in source_names:
                return value
            candidate = (base / value).resolve()
            if candidate in moves:
                target = moves[candidate]
                return str(target) if Path(value).is_absolute() else os.path.relpath(target, base)
        return value

    updated = copy.deepcopy(state)
    for i, _, new in changes:
        updated.tracks[i].camera_label = new
    updated_data = update_paths(updated.to_dict(), Path.cwd())
    updated = ProjectState.from_dict(updated_data)
    for path, raw in documents.items():
        documents[path] = update_paths(raw, path.parent)
    if state.project_path:
        project = Path(state.project_path).resolve()
        documents[project] = _relativise_paths(updated.to_dict(), project.parent)

    # Stage all source files first (supports case-only renames on Windows),
    # then publish each target. Metadata writes use sibling atomic replacements.
    backups = {p: p.read_bytes() if p.exists() else None for p in documents}
    report(35, f"Ready to rename {len(actual_moves)} files and update {len(documents)} metadata files.")
    staged: dict[Path, Path] = {}
    published: list[Path] = []
    temporary_metadata: list[Path] = []
    written: list[Path] = []
    total_steps = max(1, 2 * len(actual_moves) + len(documents))
    step = 0
    try:
        for source in actual_moves:
            report(35 + int(60 * step / total_steps), f"Preparing file: {source.name}")
            temporary = source.with_name(f".camera-rename-{uuid4().hex}.tmp")
            source.rename(temporary)
            staged[source] = temporary
            step += 1
        for source, target in actual_moves.items():
            report(35 + int(60 * step / total_steps), f"Renaming file: {target.name}")
            if target.exists():
                raise FileExistsError(f"Destination appeared during rename: {target}")
            staged[source].rename(target)
            published.append(source)
            step += 1
        for path, raw in documents.items():
            report(35 + int(60 * step / total_steps), f"Updating metadata: {path}")
            temporary = path.with_name(f".camera-meta-{uuid4().hex}.tmp")
            temporary_metadata.append(temporary)
            temporary.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
            written.append(path)
            step += 1
        report(95, "Finishing camera rename…")
    except Exception as exc:
        report(95, "Restoring previous filenames and metadata…", check_cancel=False)
        rollback_errors = []
        # Stage published destinations again before restoring sources.
        for source in reversed(published):
            try:
                actual_moves[source].rename(staged[source])
            except OSError as error:
                rollback_errors.append(str(error))
        for source, temporary in staged.items():
            try:
                if temporary.exists():
                    temporary.rename(source)
            except OSError as error:
                rollback_errors.append(str(error))
        for path in reversed(written):
            try:
                original = backups[path]
                if original is None:
                    path.unlink()
                else:
                    path.write_bytes(original)
            except OSError as error:
                rollback_errors.append(str(error))
        if rollback_errors:
            raise RuntimeError(f"Rename failed: {exc}. Some files could not be restored: "
                               + "; ".join(rollback_errors)) from exc
        raise
    finally:
        for temporary in temporary_metadata:
            if temporary.exists():
                temporary.unlink()

    # Keep existing track objects: mode widgets may retain references to them.
    for current, replacement in zip(state.tracks, updated.tracks):
        current.__dict__.update(replacement.__dict__)
    if path_changes is not None:
        path_changes.update({str(source): str(target) for source, target in moves.items()})
    report(100, f"Camera names updated; {len(actual_moves)} file(s) renamed.", check_cancel=False)
    return len(actual_moves)