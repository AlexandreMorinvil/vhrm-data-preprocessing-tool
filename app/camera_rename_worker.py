"""Run the filesystem transaction away from Qt's GUI thread."""
from __future__ import annotations

import copy
import logging
import threading
import time

from PyQt6.QtCore import QThread, pyqtSignal

from .camera_rename import CameraRenameCancelled, rename_cameras
from .state import ProjectState

log = logging.getLogger(__name__)


class CameraRenameWorker(QThread):
    progress = pyqtSignal(int, str)

    def __init__(self, state: ProjectState, labels: list[str], parent=None):
        super().__init__(parent)
        # Never mutate shared state while its widgets belong to the GUI thread.
        self.state = copy.deepcopy(state)
        self.labels = list(labels)
        self.path_changes: dict[str, str] = {}
        self.count = 0
        self.error = ""
        self.was_cancelled = False
        self._cancel = threading.Event()
        self._last_progress_time = 0.0
        self._last_percent = -1

    def cancel(self) -> None:
        self._cancel.set()

    def _report(self, percent: int, message: str) -> None:
        # Bound queued GUI events when a project has thousands of captures.
        now = time.monotonic()
        if percent != self._last_percent or now - self._last_progress_time >= 0.1:
            log.info("Camera rename (%d%%): %s", percent, message)
            self.progress.emit(percent, message)
            self._last_progress_time = now
            self._last_percent = percent

    def run(self) -> None:
        try:
            self.count = rename_cameras(
                self.state, self.labels, progress=self._report,
                cancelled=self._cancel.is_set, path_changes=self.path_changes,
            )
        except CameraRenameCancelled:
            self.was_cancelled = True
        except Exception as exc:
            log.exception("Camera rename failed")
            self.error = str(exc)