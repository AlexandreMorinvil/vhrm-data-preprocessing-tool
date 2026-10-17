"""Final clean-up: delete the processing artifacts, move them out of the dataset, or move them back."""
from __future__ import annotations

import html
import logging
import os
from pathlib import Path
from typing import Callable, Optional

from PyQt6 import sip
from PyQt6.QtCore import QSettings, Qt, QUrl
from PyQt6.QtGui import QBrush, QColor, QDesktopServices
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..artifacts import (
    CATEGORIES,
    CATEGORY_DESCRIPTIONS,
    CATEGORY_TITLES,
    FULL_VIDEOS,
    ArtifactFile,
    ArtifactsFolderScan,
    CleanupResult,
    dataset_directory,
    default_artifacts_name,
    delete_artifacts,
    destination_problem,
    existing_destinations,
    find_artifacts,
    format_size,
    move_artifacts,
    restore_artifacts,
    restore_checks,
    scan_artifacts_folder,
    segments_directory,
    space_problem,
)
from ..ffmpeg_utils import FFmpegWorker
from ..state import ProjectState
from ..widgets.layout import ModeWorkspace

log = logging.getLogger(__name__)

_MOVE = "move"
_DELETE = "delete"
_RESTORE = "restore"
_TITLES = {_MOVE: "Move artifacts", _DELETE: "Delete artifacts", _RESTORE: "Restore artifacts"}


def _settings() -> QSettings:
    return QSettings("VideoResearchTool", "VRT")


def _empty_found() -> dict[str, list[ArtifactFile]]:
    return {category: [] for category in CATEGORIES}


def _listing(paths: list, limit: int = 8) -> str:
    shown = "\n".join(str(path) for path in paths[:limit])
    return shown + (f"\n… and {len(paths) - limit} more" if len(paths) > limit else "")


class ArtifactsWidget(QWidget):
    def __init__(self, state: ProjectState, busy_check: Optional[Callable[[], bool]] = None, parent=None):
        super().__init__(parent)
        self.state = state
        self._busy_check = busy_check
        self._found = _empty_found()
        self._restore_scan = ArtifactsFolderScan(found=_empty_found())
        self._state_dataset = ""
        self._worker: FFmpegWorker | None = None
        self._result: Optional[CleanupResult] = None
        self._job: dict = {}
        self._suspended_players: list = []
        # Camera tiles whose video was moved away: (player, tile index, path),
        # reloaded when the file is restored.
        self._released: list[tuple[object, int, str]] = []

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self._workspace = ModeWorkspace("artifacts", self, left_width=400)
        root.addWidget(self._workspace)
        ws = self._workspace

        # -- dataset ----------------------------------------------------------------
        dataset_box = QWidget()
        dataset_lay = QVBoxLayout(dataset_box)
        dataset_lay.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self._dataset_edit = QLineEdit()
        self._dataset_edit.setPlaceholderText("Project output folder")
        self._dataset_edit.setToolTip("The folder that contains labelled_segments/ and the full-length videos.")
        self._dataset_edit.editingFinished.connect(self._rescan)
        row.addWidget(self._dataset_edit)
        browse = QPushButton("…")
        browse.setFixedWidth(30)
        browse.clicked.connect(self._browse_dataset)
        row.addWidget(browse)
        dataset_lay.addLayout(row)
        rescan = QPushButton("Rescan")
        rescan.clicked.connect(self._rescan)
        dataset_lay.addWidget(rescan)
        self._dataset_info = QLabel("")
        self._dataset_info.setWordWrap(True)
        dataset_lay.addWidget(self._dataset_info)
        ws.add_section("Dataset", dataset_box)

        # -- artifact categories --------------------------------------------------------
        stored = str(_settings().value("artifacts/categories", f"{FULL_VIDEOS}") or "")
        selected = {value for value in stored.split(",") if value}
        category_box = QWidget()
        category_lay = QVBoxLayout(category_box)
        category_lay.setContentsMargins(0, 0, 0, 0)
        self._category_checks: dict[str, QCheckBox] = {}
        for category in CATEGORIES:
            check = QCheckBox(CATEGORY_TITLES[category])
            check.setChecked(category in selected)
            check.setToolTip(CATEGORY_DESCRIPTIONS[category])
            check.toggled.connect(self._on_categories_changed)
            category_lay.addWidget(check)
            self._category_checks[category] = check
        ws.add_section("Artifacts", category_box)

        # -- action ---------------------------------------------------------------------
        action_box = QWidget()
        action_lay = QVBoxLayout(action_box)
        action_lay.setContentsMargins(0, 0, 0, 0)
        self._move_radio = QRadioButton("Move to an artifacts folder (same structure)")
        self._delete_radio = QRadioButton("Delete permanently")
        self._restore_radio = QRadioButton("Restore from an artifacts folder (move back)")
        self._restore_radio.setToolTip("Move previously moved artifacts back to their place in the dataset.")
        group = QButtonGroup(self)
        for button in (self._move_radio, self._delete_radio, self._restore_radio):
            group.addButton(button)
            action_lay.addWidget(button)
        # Restoring is occasional: the page always opens on moving or deleting.
        if str(_settings().value("artifacts/action", _MOVE)) == _DELETE:
            self._delete_radio.setChecked(True)
        else:
            self._move_radio.setChecked(True)
        group.buttonToggled.connect(lambda _button, checked: checked and self._on_action_changed())

        self._move_form = QWidget()
        form = QFormLayout(self._move_form)
        form.setContentsMargins(18, 2, 0, 0)
        location_row = QHBoxLayout()
        self._location_edit = QLineEdit(str(_settings().value("artifacts/location", "") or ""))
        self._location_edit.setPlaceholderText("Next to the dataset folder")
        self._location_edit.setToolTip("Folder that contains the artifacts folder.")
        self._location_edit.editingFinished.connect(self._on_location_edited)
        location_row.addWidget(self._location_edit)
        location_browse = QPushButton("…")
        location_browse.setFixedWidth(30)
        location_browse.clicked.connect(self._browse_location)
        location_row.addWidget(location_browse)
        reset = QToolButton()
        reset.setText("↺")
        reset.setToolTip("Use the folder that contains the dataset folder")
        reset.clicked.connect(self._reset_location)
        location_row.addWidget(reset)
        form.addRow("Location:", location_row)
        self._name_edit = QLineEdit()
        self._name_edit.setToolTip("Name of the artifacts folder; empty uses the dataset folder name + _artifacts.")
        self._name_edit.textChanged.connect(self._update_destination)
        form.addRow("Folder name:", self._name_edit)
        self._choose_artifacts_btn = QPushButton("Choose artifacts folder …")
        self._choose_artifacts_btn.setToolTip("Pick the artifacts folder directly (sets location and name).")
        self._choose_artifacts_btn.clicked.connect(self._browse_artifacts_folder)
        form.addRow("", self._choose_artifacts_btn)
        self._destination_caption = QLabel("Destination:")
        self._destination_label = QLabel("")
        self._destination_label.setWordWrap(True)
        self._destination_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._destination_caption, self._destination_label)
        action_lay.addWidget(self._move_form)
        ws.add_section("Action", action_box)

        # -- run ------------------------------------------------------------------------
        run_box = QWidget()
        run_lay = QVBoxLayout(run_box)
        run_lay.setContentsMargins(0, 0, 0, 0)
        self._run_btn = QPushButton("")
        self._run_btn.setStyleSheet("font-weight:bold; padding:5px;")
        self._run_btn.clicked.connect(self._run)
        run_lay.addWidget(self._run_btn)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)
        run_lay.addWidget(self._cancel_btn)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        run_lay.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        run_lay.addWidget(self._status)
        self._open_btn = QPushButton("Open artifacts folder")
        self._open_btn.clicked.connect(self._open_destination)
        run_lay.addWidget(self._open_btn)
        ws.add_section("Clean up", run_box)
        ws.finish_left()

        # -- preview --------------------------------------------------------------------
        preview = QWidget()
        preview_lay = QVBoxLayout(preview)
        preview_lay.setContentsMargins(4, 4, 4, 4)
        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        self._summary.setTextFormat(Qt.TextFormat.RichText)
        preview_lay.addWidget(self._summary)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["File", "Size", "Destination"])
        self._tree.setAlternatingRowColors(True)
        header = self._tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self._tree.setColumnWidth(0, 420)
        preview_lay.addWidget(self._tree, 1)
        ws.add_work(preview, 600)

        self._on_action_changed()
        self.refresh_from_state()

    # ------------------------------------------------------------------ state
    def refresh_from_state(self) -> None:
        if self._worker is not None:
            return
        directory = dataset_directory(self.state)
        derived = str(directory) if directory is not None else ""
        if derived != self._state_dataset:
            # Follow the project unless another folder was chosen for it.
            self._state_dataset = derived
            self._dataset_edit.setText(derived)
        self._rescan()

    def _action(self) -> str:
        if self._restore_radio.isChecked():
            return _RESTORE
        return _DELETE if self._delete_radio.isChecked() else _MOVE

    def _dataset(self) -> Optional[Path]:
        text = self._dataset_edit.text().strip()
        return Path(text) if text else None

    def _project_state_for(self, dataset: Path) -> Optional[ProjectState]:
        """The project state when *dataset* is the project's own output folder."""
        if self._state_dataset and Path(self._state_dataset).resolve() == dataset.resolve():
            return self.state
        return None

    def _segments_dir(self, dataset: Path) -> Path:
        return segments_directory(dataset, self._project_state_for(dataset))

    def _rescan(self) -> None:
        dataset = self._dataset()
        if dataset is None or not dataset.is_dir():
            self._found = _empty_found()
            self._dataset_info.setText(
                "Load the project metadata, or choose its output folder." if dataset is None
                else "⚠ This folder does not exist."
            )
        else:
            self._found = find_artifacts(dataset, self._project_state_for(dataset))
            self._dataset_info.setText(self._segments_text(dataset))
        self._update_destination()

    def _segments_text(self, dataset: Path) -> str:
        segments = self._segments_dir(dataset)
        manifest = segments / "manifest.csv"
        if not manifest.exists():
            return (f"⚠ No {segments.name}/manifest.csv in this folder. The full-length videos may "
                    "still be needed to export the labelled segments.")
        count = sum(1 for path in segments.iterdir() if path.is_dir())
        return f"Labelled segments: {count} folder(s) in {segments.name}/ (never modified here)."

    def _active_found(self) -> dict[str, list[ArtifactFile]]:
        """Candidates of the current action: in the dataset, or in the artifacts folder."""
        return self._restore_scan.found if self._action() == _RESTORE else self._found

    def _update_categories(self) -> None:
        found = self._active_found()
        where = "in the artifacts folder" if self._action() == _RESTORE else "found"
        for category, check in self._category_checks.items():
            files = found[category]
            if files:
                size = format_size(sum(item.size for item in files))
                check.setText(f"{CATEGORY_TITLES[category]} — {len(files)} file(s), {size}")
            else:
                check.setText(f"{CATEGORY_TITLES[category]} — none {where}")
            # Empty categories stay checkable: the choice is remembered for the next dataset.
            check.setEnabled(self._worker is None)

    def _on_categories_changed(self, *_args) -> None:
        selected = [category for category, check in self._category_checks.items() if check.isChecked()]
        _settings().setValue("artifacts/categories", ",".join(selected))
        self._update_preview()

    def _on_action_changed(self) -> None:
        action = self._action()
        if action != _RESTORE:
            _settings().setValue("artifacts/action", action)
        self._move_form.setVisible(action != _DELETE)
        self._open_btn.setVisible(action != _DELETE)
        self._choose_artifacts_btn.setVisible(action == _RESTORE)
        self._destination_caption.setText("Artifacts folder:" if action == _RESTORE else "Destination:")
        self._update_destination()

    # ------------------------------------------------------------------ destination
    def _browse_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Dataset folder", self._dataset_edit.text())
        if path:
            self._dataset_edit.setText(path)
            self._rescan()

    def _browse_location(self) -> None:
        start = self._location_edit.text() or self._default_location()
        path = QFileDialog.getExistingDirectory(self, "Folder that contains the artifacts folder", start)
        if path:
            self._location_edit.setText(path)
            self._on_location_edited()

    def _browse_artifacts_folder(self) -> None:
        destination = self._destination()
        start = str(destination) if destination is not None and destination.is_dir() else (
            self._location_edit.text() or self._default_location())
        path = QFileDialog.getExistingDirectory(self, "Artifacts folder to restore", start)
        if path:
            folder = Path(path)
            self._location_edit.setText(str(folder.parent))
            self._name_edit.setText(folder.name)
            self._on_location_edited()

    def _on_location_edited(self) -> None:
        _settings().setValue("artifacts/location", self._location_edit.text().strip())
        self._update_destination()

    def _reset_location(self) -> None:
        self._location_edit.clear()
        self._on_location_edited()

    def _default_location(self) -> str:
        dataset = self._dataset()
        return str(dataset.resolve().parent) if dataset is not None else ""

    def _destination(self) -> Optional[Path]:
        """The artifacts folder: where artifacts are moved to, or restored from."""
        dataset = self._dataset()
        if dataset is None:
            return None
        location = self._location_edit.text().strip() or self._default_location()
        name = self._name_edit.text().strip() or default_artifacts_name(dataset)
        return Path(location) / name if location else None

    def _update_destination(self, *_args) -> None:
        dataset = self._dataset()
        restoring = self._action() == _RESTORE
        self._name_edit.setPlaceholderText(default_artifacts_name(dataset) if dataset is not None else "")
        destination = self._destination()
        problem = destination_problem(dataset, destination) if dataset and destination else None
        if restoring and destination is not None and not problem and not destination.is_dir():
            problem = "This folder does not exist."
        self._restore_scan = (
            scan_artifacts_folder(destination, self._project_state_for(dataset))
            if restoring and destination is not None and not problem
            else ArtifactsFolderScan(found=_empty_found())
        )
        if destination is None:
            self._destination_label.setText("")
        elif problem:
            self._destination_label.setText(f"<span style='color:#e05050'>⚠ {problem}</span>")
        elif restoring:
            self._destination_label.setText(str(destination))
        else:
            exists = " (exists — files are added to it)" if destination.exists() else ""
            self._destination_label.setText(f"{destination}{exists}")
        self._open_btn.setEnabled(destination is not None and destination.is_dir())
        self._update_categories()
        self._update_preview()

    def _open_destination(self) -> None:
        destination = self._destination()
        if destination is not None and destination.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(destination)))

    # ------------------------------------------------------------------ preview
    def _selected_files(self) -> list[ArtifactFile]:
        found = self._active_found()
        return [item for category, check in self._category_checks.items()
                if check.isChecked() for item in found[category]]

    def _restore_validation(self, files: list[ArtifactFile]) -> tuple[list[str], list[str]]:
        dataset = self._dataset()
        destination = self._destination()
        if dataset is None or destination is None:
            return ["Choose the dataset and the artifacts folder."], []
        return restore_checks(destination, dataset, files, self._restore_scan.manifest,
                              self._project_state_for(dataset))

    def _update_preview(self) -> None:
        action = self._action()
        dataset = self._dataset()
        destination = self._destination()
        found = self._active_found()
        conflicts = 0
        self._tree.clear()
        for category, check in self._category_checks.items():
            files = found[category]
            if not check.isChecked() or not files:
                continue
            size = sum(item.size for item in files)
            top = QTreeWidgetItem([f"{CATEGORY_TITLES[category]} ({len(files)})", format_size(size), ""])
            font = top.font(0)
            font.setBold(True)
            top.setFont(0, font)
            top.setToolTip(0, CATEGORY_DESCRIPTIONS[category])
            for item in files:
                if action == _DELETE:
                    target = shown = "deleted"
                else:
                    base = dataset if action == _RESTORE else destination
                    target = str(base / item.relative) if base else ""
                    shown = str(Path(base.name) / item.relative) if base else ""
                    if base and (base / item.relative).exists():
                        conflicts += 1
                        shown += "  — already exists"
                child = QTreeWidgetItem([str(item.relative), format_size(item.size), shown])
                child.setToolTip(0, str(item.path))
                child.setToolTip(2, target)
                if shown.endswith("already exists"):
                    child.setForeground(2, QBrush(QColor("#e0a030")))
                top.addChild(child)
            self._tree.addTopLevelItem(top)
            top.setExpanded(len(files) <= 20)
        if action == _RESTORE and self._restore_scan.ignored:
            ignored = self._restore_scan.ignored
            top = QTreeWidgetItem([f"Not artifacts — left in place ({len(ignored)})", "", ""])
            top.setToolTip(0, "Files the restore does not recognise as artifacts of a dataset.")
            grey = QBrush(QColor("#888888"))
            for relative in ignored:
                child = QTreeWidgetItem([str(relative), "", "not restored"])
                for column in range(3):
                    child.setForeground(column, grey)
                top.addChild(child)
            self._tree.addTopLevelItem(top)

        files = self._selected_files()
        size = format_size(sum(item.size for item in files))
        verb = {_MOVE: "Move", _DELETE: "Delete", _RESTORE: "Restore"}[action]
        target_name = "dataset" if action == _RESTORE else "artifacts folder"
        conflict_text = (f"<br><span style='color:#e0a030'>⚠ {conflicts} file(s) already exist in the "
                         f"{target_name}; you will be asked whether to replace them.</span>" if conflicts else "")
        if files:
            if action == _RESTORE:
                errors, warnings = self._restore_validation(files)
                origin = (self._restore_scan.manifest or {}).get("dataset")
                source = f", moved out of {html.escape(origin)}," if origin and not errors and not warnings else ""
                text = f"Restore {len(files)} file(s), {size}{source} to their place in the dataset."
                text += "".join(f"<br><span style='color:#e05050'>✖ {html.escape(message)}</span>" for message in errors)
                text += "".join(f"<br><span style='color:#e0a030'>⚠ {html.escape(message)}</span>" for message in warnings)
                self._summary.setText(text + conflict_text)
            else:
                self._summary.setText(
                    f"{verb} {len(files)} file(s), {size}. Labelled segments, signals and metadata "
                    "stay in the dataset." + conflict_text
                )
            self._run_btn.setText(f"{verb} {len(files)} file(s) ({size}) …")
        else:
            if action == _RESTORE:
                empty = "No artifacts to restore in this folder."
            else:
                empty = "No artifacts found in this folder."
            self._summary.setText("Select the artifacts." if any(found.values()) else empty)
            self._run_btn.setText(f"{verb} artifacts …")
        self._run_btn.setEnabled(bool(files) and self._worker is None)

    # ------------------------------------------------------------------ run
    def _run(self) -> None:
        dataset = self._dataset()
        files = self._selected_files()
        action = self._action()
        title = _TITLES[action]
        if dataset is None or not files:
            return
        if self._busy_check is not None and self._busy_check():
            QMessageBox.warning(self, title, "Wait for preprocessing, exports or estimations to finish first.")
            return
        destination = self._destination()
        if action == _RESTORE:
            files, overwrite = self._confirm_restore(dataset, destination, files)
        else:
            overwrite = self._confirm_clean_up(action, dataset, destination, files)
        if files is None or overwrite is None:
            return

        self._suspend_players()
        self._result = None
        self._job = {"action": action, "destination": destination}
        worker = FFmpegWorker(
            self._cleanup_job, files=files, dataset=dataset, destination=destination,
            action=action, overwrite=overwrite, segments_dir=self._segments_dir(dataset),
        )
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        self._worker = worker
        self._set_busy(True)
        self._progress.setValue(0)
        self._status.setText({_MOVE: "Moving …", _DELETE: "Deleting …", _RESTORE: "Restoring …"}[action])
        worker.start()

    def _category_details(self, files: list[ArtifactFile]) -> str:
        lines = []
        for category in CATEGORIES:
            selected = [item for item in files if item.category == category]
            if selected:
                lines.append(f"• {CATEGORY_TITLES[category]}: {len(selected)} file(s), "
                             f"{format_size(sum(item.size for item in selected))}")
        return "\n".join(lines)

    def _confirm_clean_up(self, action: str, dataset: Path, destination: Optional[Path],
                          files: list[ArtifactFile]) -> Optional[bool]:
        """Validate and confirm a move or delete; returns *overwrite*, or None to stop."""
        title = _TITLES[action]
        moving = action == _MOVE
        if moving:
            problem = destination_problem(dataset, destination) if destination else "Choose a destination folder."
            problem = problem or space_problem(files, destination)
            if problem:
                QMessageBox.warning(self, title, problem)
                return None
        size = format_size(sum(item.size for item in files))
        details = self._category_details(files)
        videos = any(item.category == FULL_VIDEOS for item in files)
        warning = ""
        if videos and not (self._segments_dir(dataset) / "manifest.csv").exists():
            warning = ("\n\n⚠ No labelled segments were found in this folder: the full-length videos "
                       "are the only copy of the recording.")
        notes = ("\n\nThe Preprocessing, Labelling and PPG alignment sections can no longer play the "
                 "full session once its videos are gone." if videos else "")
        if not moving:
            box = QMessageBox.warning(
                self, title,
                f"Permanently delete {len(files)} file(s) ({size}) from\n{dataset}?\n\n{details}{warning}{notes}"
                "\n\nThis cannot be undone.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            return False if box == QMessageBox.StandardButton.Yes else None
        box = QMessageBox.question(
            self, title,
            f"Move {len(files)} file(s) ({size}) from\n{dataset}\nto\n{destination}?\n\n{details}{warning}{notes}"
            "\n\nThey can be moved back with “Restore from an artifacts folder”.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if box != QMessageBox.StandardButton.Yes:
            return None
        conflicts = existing_destinations(files, destination)
        if not conflicts:
            return False
        box = QMessageBox.question(
            self, title,
            f"{len(conflicts)} file(s) already exist in the artifacts folder:\n{_listing(conflicts)}\n\n"
            "Replace them?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return True if box == QMessageBox.StandardButton.Yes else None

    def _confirm_restore(self, dataset: Path, artifacts: Optional[Path], files: list[ArtifactFile]
                         ) -> tuple[Optional[list[ArtifactFile]], Optional[bool]]:
        """Validate and confirm a restore; returns the files and *overwrite*, or Nones to stop."""
        title = _TITLES[_RESTORE]
        errors, warnings = self._restore_validation(files)
        if errors:
            QMessageBox.warning(self, title, "Cannot restore:\n\n" + "\n".join(f"• {e}" for e in errors))
            return None, None

        overwrite = False
        conflicts = existing_destinations(files, dataset)
        if conflicts:
            box = QMessageBox(QMessageBox.Icon.Question, title,
                              f"{len(conflicts)} file(s) already exist in the dataset:\n{_listing(conflicts)}\n\n"
                              "Replace them with the artifacts, or keep the dataset's files and skip these?",
                              parent=self)
            replace = box.addButton("Replace", QMessageBox.ButtonRole.DestructiveRole)
            skip = box.addButton("Skip existing", QMessageBox.ButtonRole.AcceptRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(skip)
            box.exec()
            if box.clickedButton() is replace:
                overwrite = True
            elif box.clickedButton() is skip:
                existing = {path.resolve() for path in conflicts}
                files = [item for item in files if (dataset / item.relative).resolve() not in existing]
                if not files:
                    QMessageBox.information(self, title, "Every file already exists in the dataset; nothing to restore.")
                    return None, None
            else:
                return None, None

        size = format_size(sum(item.size for item in files))
        warning_text = "".join(f"\n\n⚠ {message}" for message in warnings)
        ignored = self._restore_scan.ignored
        ignored_text = (f"\n\n{len(ignored)} other file(s) are not artifacts and stay in the artifacts folder."
                        if ignored else "")
        box = QMessageBox.question(
            self, title,
            f"Move {len(files)} file(s) ({size}) from\n{artifacts}\nback to\n{dataset}?\n\n"
            f"{self._category_details(files)}{warning_text}{ignored_text}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel if warnings else QMessageBox.StandardButton.Yes,
        )
        if box != QMessageBox.StandardButton.Yes:
            return None, None
        return files, overwrite

    def _cleanup_job(self, *, files, dataset, destination, action, overwrite, segments_dir, worker):
        verb = {_MOVE: "Moving", _DELETE: "Deleting", _RESTORE: "Restoring"}[action]

        def progress(done: int, total: int, name: str) -> None:
            worker.progress.emit(int(100 * done / total), f"{verb} {name}")

        options = dict(progress=progress, is_cancelled=lambda: worker.is_cancelled, segments_dir=segments_dir)
        if action == _MOVE:
            self._result = move_artifacts(files, dataset, destination, overwrite=overwrite, **options)
        elif action == _RESTORE:
            self._result = restore_artifacts(files, destination, dataset, overwrite=overwrite, **options)
        else:
            self._result = delete_artifacts(files, dataset, **options)

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def _on_progress(self, value: int, message: str) -> None:
        self._progress.setValue(value)
        self._status.setText(message)

    def _on_finished(self, ok: bool, message: str) -> None:
        self._worker = None
        result = self._result
        action = self._job.get("action", _MOVE)
        title = _TITLES[action]
        restoring = action == _RESTORE
        self._resume_players([] if restoring or result is None else result.done,
                             result.done if restoring and result is not None else [])
        self._set_busy(False)
        if result is None:
            self._status.setText(f"Failed: {message}")
            QMessageBox.critical(self, title, f"Failed:\n{message}")
        else:
            verb = {_MOVE: "Moved", _DELETE: "Deleted", _RESTORE: "Restored"}[action]
            text = f"{verb} {len(result.done)} file(s), {format_size(result.bytes_done)}."
            if action == _MOVE and result.done:
                text += f"\nArtifacts folder: {self._job['destination']}"
            if restoring and result.done and not Path(self._job["destination"]).exists():
                text += "\nThe artifacts folder was empty and has been removed."
            if result.cancelled:
                place = "artifacts folder" if restoring else "dataset"
                text += f"\nCancelled; the remaining files were left in the {place}."
            self._status.setText(text)
            if not result.cancelled and not result.failures:
                self._progress.setValue(100)
            for item, error in result.failures:
                log.warning("%s failed for %s: %s", title, item.path, error)
            if result.failures:
                lines = [f"{item.relative}: {error}" for item, error in result.failures]
                QMessageBox.warning(
                    self, title,
                    f"{len(result.failures)} file(s) could not be processed and were left in place:\n"
                    f"{_listing(lines)}\n\nClose any program using them and try again.",
                )
            log.info(text.replace("\n", " "))
        self._rescan()

    def _set_busy(self, busy: bool) -> None:
        self._run_btn.setEnabled(not busy and bool(self._selected_files()))
        self._cancel_btn.setEnabled(busy)
        for widget in (self._dataset_edit, self._move_radio, self._delete_radio, self._restore_radio,
                       self._move_form, *self._category_checks.values()):
            widget.setEnabled(not busy)
        if not busy:
            self._update_categories()

    # ------------------------------------------------------------------ players
    def _players(self) -> list:
        from ..widgets.frame_preview import MultiCameraPlayer

        return self.window().findChildren(MultiCameraPlayer)

    def _suspend_players(self) -> None:
        """Release the video files held open by the other sections (Windows locks them)."""
        self._suspended_players = self._players()
        for player in self._suspended_players:
            player.suspend_for_camera_rename()

    def _resume_players(self, removed: list[ArtifactFile], restored: list[ArtifactFile]) -> None:
        gone = {os.path.normcase(str(item.path)) for item in removed}
        for player in self._suspended_players:
            for index, preview in enumerate(player.previews):
                path = preview.video_path
                if path and os.path.normcase(str(Path(path).resolve())) in gone:
                    self._released.append((player, index, path))
                    preview.load_video(path)  # the file is gone: shows "Missing video"
            player.resume_after_camera_rename({}, [])
        self._suspended_players = []
        if restored:
            self._reload_restored_videos()

    def _reload_restored_videos(self) -> None:
        """Show again the camera videos that are back in the dataset."""
        by_player: dict[int, tuple[object, dict[int, str]]] = {}
        remaining = []
        for player, index, path in self._released:
            if sip.isdeleted(player):
                continue
            if Path(path).exists():
                by_player.setdefault(id(player), (player, {}))[1][index] = path
            else:
                remaining.append((player, index, path))
        self._released = remaining
        for player, paths in by_player.values():
            current = [preview.video_path for preview in player.previews]
            player.load_videos([paths.get(index, value) for index, value in enumerate(current)])
