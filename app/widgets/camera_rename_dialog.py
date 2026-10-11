from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit,
    QMessageBox, QScrollArea, QVBoxLayout, QWidget,
)

from ..camera_rename import validate_camera_labels
from ..state import ProjectState


class CameraRenameDialog(QDialog):
    def __init__(self, state: ProjectState, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Rename cameras")
        self.resize(540, 360)
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Rename cameras throughout this project, including final videos, "
            "labelled clips, metadata and the saved project.\n"
            "Original recordings and sensor files are not changed. "
            "Existing text drawn into exported mosaics or images cannot be changed; "
            "export those again to use the new names."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        form = QFormLayout(content)
        self._edits: list[QLineEdit] = []
        for i, track in enumerate(state.tracks):
            edit = QLineEdit(track.camera_label)
            edit.setMaxLength(100)
            form.addRow(f"Camera {i + 1}:", edit)
            self._edits.append(edit)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Rename")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def labels(self) -> list[str]:
        return [edit.text().strip() for edit in self._edits]

    def accept(self) -> None:
        try:
            validate_camera_labels(self.labels())
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid camera names", str(exc))
            return
        super().accept()