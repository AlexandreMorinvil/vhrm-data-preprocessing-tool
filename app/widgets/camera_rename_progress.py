from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QDialog, QLabel, QProgressBar, QPushButton, QVBoxLayout


class CameraRenameProgressDialog(QDialog):
    """Stay open on Cancel/Escape/close until the worker finishes its rollback."""

    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Renaming cameras")
        self.resize(620, 210)
        self._complete = False
        self._cancelling = False
        layout = QVBoxLayout(self)
        self._status = QLabel("Releasing video decoders…")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)
        explanation = QLabel(
            "Files are renamed without copying or re-encoding the videos. "
            "Cancel restores the original filenames and metadata."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self._cancel = QPushButton("Cancel")
        self._cancel.clicked.connect(self.reject)
        layout.addWidget(self._cancel)

    def update_progress(self, percent: int, message: str) -> None:
        self._progress.setValue(percent)
        if not self._cancelling:
            self._status.setText(message)

    def reject(self) -> None:
        if self._complete:
            super().reject()
        elif not self._cancelling:
            self._cancelling = True
            self._cancel.setEnabled(False)
            self._status.setText("Cancelling… waiting for any file operation to finish and restoring previous files.")
            self.cancel_requested.emit()

    def closeEvent(self, event) -> None:
        if not self._complete:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)

    def finish(self) -> None:
        self._complete = True
        self.accept()