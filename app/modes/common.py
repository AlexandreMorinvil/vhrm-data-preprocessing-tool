"""Helpers shared by the mode screens (dialogs, captures, snapshots)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QWidget,
)

from ..frame_export import export_player_frames, export_player_snapshot
from ..mosaic_export import MOSAIC_PRESET_NAMES, normalise_mosaic_preset

log = logging.getLogger(__name__)


class MosaicExportDialog(QDialog):
    """Choose output file, speed/quality preset and the sound source."""

    def __init__(self, default_path: str, preset_key: str, camera_labels: list[str],
                 audio_camera: int = 0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export mosaic video")
        self.setMinimumWidth(520)
        form = QFormLayout(self)
        path_row = QHBoxLayout()
        self._path = QLineEdit(default_path)
        browse = QPushButton("…")
        browse.setFixedWidth(30)
        browse.clicked.connect(self._browse)
        path_row.addWidget(self._path)
        path_row.addWidget(browse)
        form.addRow("Output file:", path_row)
        self._preset = QComboBox()
        self._preset.addItems(MOSAIC_PRESET_NAMES)
        self._preset.setCurrentIndex({"speed": 0, "balanced": 1, "quality": 2}.get(
            normalise_mosaic_preset(preset_key), 1))
        form.addRow("Speed vs quality:", self._preset)
        self._audio = QComboBox()
        self._audio.addItem("No sound", -1)
        for index, label in enumerate(camera_labels):
            self._audio.addItem(f"Sound from {label}", index)
        position = self._audio.findData(audio_camera)
        self._audio.setCurrentIndex(position if position >= 0 else 0)
        form.addRow("Audio:", self._audio)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save mosaic video", self._path.text(),
                                              "MP4 video (*.mp4);;All files (*)")
        if path:
            self._path.setText(path)

    def _accept(self) -> None:
        if not self._path.text().strip():
            QMessageBox.warning(self, "Export mosaic video", "Choose an output file.")
            return
        self.accept()

    def values(self) -> tuple[str, str, str, int]:
        path = self._path.text().strip()
        if not path.lower().endswith(".mp4"):
            path += ".mp4"
        choice = self._preset.currentText()
        return path, normalise_mosaic_preset(choice), choice, int(self._audio.currentData())


def ask_mosaic_options(parent: QWidget, default_path: str, preset_key: str, camera_labels: list[str],
                       audio_camera: int = 0) -> Optional[tuple[str, str, str, int]]:
    dialog = MosaicExportDialog(default_path, preset_key, camera_labels, audio_camera, parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.values()


def export_frames_interactive(parent: QWidget, player, output_root: str, prefix: str) -> Optional[str]:
    """Export the current frame of every camera; returns a status message."""
    if not output_root:
        output_root = QFileDialog.getExistingDirectory(parent, "Folder for captured frames")
        if not output_root:
            return None
    try:
        written = export_player_frames(player, output_root, prefix=prefix)
    except Exception as exc:
        QMessageBox.critical(parent, "Synchronized capture", f"Capture export failed:\n{exc}")
        return f"Capture export failed: {exc}"
    return f"Exported {len(written)} capture frame(s) to {written[0].parent}"


def export_snapshot_interactive(parent: QWidget, player, plot, output_root: str, prefix: str) -> Optional[str]:
    """Export one composite image (all cameras + signals around the playhead)."""
    if not output_root:
        output_root = QFileDialog.getExistingDirectory(parent, "Folder for the snapshot")
        if not output_root:
            return None
    snapshot = plot.snapshot() if plot is not None and plot.panel_count else None
    try:
        path = export_player_snapshot(player, snapshot, output_root, prefix=prefix)
    except Exception as exc:
        QMessageBox.critical(parent, "Composite snapshot", f"Snapshot export failed:\n{exc}")
        return f"Snapshot export failed: {exc}"
    QMessageBox.information(parent, "Composite snapshot", f"Saved\n{path}")
    return f"Saved snapshot {Path(path).name}"
