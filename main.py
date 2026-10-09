from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QStackedWidget,
    QStatusBar,
    QTabBar,
    QToolBar,
    QWidget,
)

from app import theme
from app.face_privacy import FacePrivacySettings, get_privacy_settings, set_privacy_settings
from app.state import ProjectState

log = logging.getLogger(__name__)

_PROJECT_FILTER = "VRT Project (*.vrt);;All files (*)"
_MAX_RECENT = 8
_MODE_NAMES = ("1. Preprocessing", "2. Signal sync", "3. Synthetic PPG", "4. Labelling", "5. Review")

_DARK_QSS = """
QWidget { background: #1e1e1e; color: #d4d4d4; font-size: 13px; }
QToolTip { background: #2d2d30; color: #e0e0e0; border: 1px solid #555; }
QGroupBox { border: 1px solid #444; margin-top: 8px; padding-top: 12px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QPushButton { background: #2d2d30; border: 1px solid #555; padding: 4px 10px; border-radius: 3px; }
QPushButton:hover { background: #3e3e42; }
QPushButton:checked { background: #0e5a9e; border-color: #1c7fd6; }
QPushButton:disabled { color: #6a6a6a; border-color: #3a3a3a; }
QToolButton { background: transparent; border: 1px solid transparent; padding: 3px; border-radius: 3px; }
QToolButton:hover { background: #3e3e42; border-color: #555; }
QLineEdit, QSpinBox, QDoubleSpinBox, QTimeEdit, QComboBox { background: #2d2d30; border: 1px solid #555; padding: 3px; }
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled { color: #6a6a6a; }
QComboBox QAbstractItemView { background: #252526; selection-background-color: #0e5a9e; }
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QTimeEdit::up-button, QTimeEdit::down-button { background: #3e3e42; border: 1px solid #555; width: 16px; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow, QTimeEdit::up-arrow { image: none; border-left: 4px solid transparent; border-right: 4px solid transparent; border-bottom: 5px solid #cccccc; width: 0; height: 0; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow, QTimeEdit::down-arrow { image: none; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 5px solid #cccccc; width: 0; height: 0; }
QSlider::groove:horizontal { height: 6px; background: #444; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #0e5a9e; border-radius: 3px; }
QSlider::handle:horizontal { width: 14px; margin: -4px 0; background: #1c8ae6; border-radius: 7px; }
QListWidget, QTableWidget { background: #252526; alternate-background-color: #2a2a2b; }
QListWidget::item:selected, QTableWidget::item:selected { background: #0e5a9e; color: #ffffff; }
QHeaderView::section { background: #2d2d30; color: #cccccc; border: 0; border-right: 1px solid #3a3a3a; padding: 3px; }
QTextEdit { background: #1b1b1b; }
QProgressBar { text-align: center; border: 1px solid #555; background: #2d2d30; }
QProgressBar::chunk { background: #0078d4; }
QTabBar::tab { background: #2d2d30; padding: 6px 16px; border: 1px solid #444; }
QTabBar::tab:selected { background: #0078d4; color: #ffffff; }
QSplitter::handle { background: #2b2b2b; }
QSplitter::handle:hover { background: #0e5a9e; }
QScrollBar:vertical { background: #1e1e1e; width: 12px; }
QScrollBar::handle:vertical { background: #4a4a4a; min-height: 24px; border-radius: 4px; }
QScrollBar:horizontal { background: #1e1e1e; height: 12px; }
QScrollBar::handle:horizontal { background: #4a4a4a; min-width: 24px; border-radius: 4px; }
QMenu { background: #252526; border: 1px solid #444; }
QMenu::item:selected { background: #0e5a9e; }
QCheckBox::indicator { width: 14px; height: 14px; }
"""

_LIGHT_QSS = """
QWidget { font-size: 13px; }
QTabBar::tab { padding: 6px 16px; }
QTabBar::tab:selected { background: #0078d4; color: #ffffff; }
QSplitter::handle:hover { background: #0078d4; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Research Tool")
        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(
                min(1600, int(available.width() * 0.92)),
                min(1000, int(available.height() * 0.92)),
            )
        else:
            self.resize(1400, 900)

        self._settings = QSettings("VideoResearchTool", "VRT")
        geometry = self._settings.value("main/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        privacy = self._settings.value("privacy/settings")
        if privacy:
            try:
                set_privacy_settings(FacePrivacySettings.from_dict(json.loads(privacy)))
            except (TypeError, ValueError):
                log.warning("Ignoring invalid stored privacy settings")
        self.state = ProjectState(
            archive_removed_files=self._settings.value(
                "archive_removed_files", False, type=bool
            ),
            blur_faces=self._settings.value("blur_faces", False, type=bool),
        )
        self._saved_snapshot = self._state_snapshot()

        self._build_menus()
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()

        self._apply_theme(str(self._settings.value("theme", "dark")))
        self._mode_tabs.setCurrentIndex(0)
        self._on_tab_changed(0)

    # ------------------------------------------------------------------ menus
    def _build_menus(self):
        mb: QMenuBar = self.menuBar()

        file_menu = mb.addMenu("&File")
        new_act = QAction("&New project", self)
        new_act.setShortcut(QKeySequence("Ctrl+N"))
        new_act.triggered.connect(self._new_project)
        file_menu.addAction(new_act)

        open_act = QAction("&Open project …", self)
        open_act.setShortcut(QKeySequence("Ctrl+O"))
        open_act.triggered.connect(self._open_project)
        file_menu.addAction(open_act)

        self._recent_menu = file_menu.addMenu("Open &recent")
        self._rebuild_recent_menu()

        save_act = QAction("&Save project", self)
        save_act.setShortcut(QKeySequence("Ctrl+S"))
        save_act.triggered.connect(self._save_project)
        file_menu.addAction(save_act)

        save_as_act = QAction("Save project &as …", self)
        save_as_act.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_as_act.triggered.connect(self._save_project_as)
        file_menu.addAction(save_as_act)

        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence("Ctrl+Q"))
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        view_menu = mb.addMenu("&View")
        dark_act = QAction("Dark theme", self)
        dark_act.triggered.connect(lambda: self._apply_theme("dark"))
        view_menu.addAction(dark_act)
        light_act = QAction("Light theme", self)
        light_act.triggered.connect(lambda: self._apply_theme("light"))
        view_menu.addAction(light_act)
        view_menu.addSeparator()
        side_act = QAction("Show / hide side panel", self)
        side_act.setShortcut(QKeySequence("Ctrl+B"))
        side_act.triggered.connect(self._toggle_side_panel)
        view_menu.addAction(side_act)
        popout_act = QAction("Cameras in a separate window", self)
        popout_act.setShortcut(QKeySequence("Ctrl+Shift+W"))
        popout_act.triggered.connect(self._toggle_popout)
        view_menu.addAction(popout_act)
        reset_act = QAction("Reset layout of this section", self)
        reset_act.triggered.connect(self._reset_layout)
        view_menu.addAction(reset_act)

        options_menu = mb.addMenu("&Options")
        self._archive_removed_action = QAction(
            "Archive removed files instead of deleting", self
        )
        self._archive_removed_action.setCheckable(True)
        self._archive_removed_action.setChecked(self.state.archive_removed_files)
        self._archive_removed_action.setToolTip(
            "Move obsolete files into the output folder's obsolete_files directory."
        )
        self._archive_removed_action.toggled.connect(self._set_archive_removed_files)
        options_menu.addAction(self._archive_removed_action)

        self._blur_faces_action = QAction("Blur faces for privacy", self)
        self._blur_faces_action.setCheckable(True)
        self._blur_faces_action.setChecked(self.state.blur_faces)
        self._blur_faces_action.setToolTip(
            "Anonymize detected faces in previews, captured frames, and exported videos."
        )
        self._blur_faces_action.toggled.connect(self._set_blur_faces)
        options_menu.addAction(self._blur_faces_action)
        privacy_act = QAction("Face privacy settings …", self)
        privacy_act.triggered.connect(self._privacy_settings)
        options_menu.addAction(privacy_act)

        help_menu = mb.addMenu("&Help")
        keys_act = QAction("&Keyboard shortcuts", self)
        keys_act.setShortcut(QKeySequence("F1"))
        keys_act.triggered.connect(self._show_shortcuts)
        help_menu.addAction(keys_act)
        about_act = QAction("&About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _build_toolbar(self):
        tb = QToolBar("Modes")
        tb.setMovable(False)
        self.addToolBar(tb)

        self._mode_tabs = QTabBar()
        for name in _MODE_NAMES:
            self._mode_tabs.addTab(name)
        self._mode_tabs.currentChanged.connect(self._on_tab_changed)
        tb.addWidget(self._mode_tabs)

    def _build_central(self):
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._mode_widgets: list[QWidget] = [None, None, None, None, None]

    def _build_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._project_label = QLabel("")
        sb.addPermanentWidget(self._project_label)
        self._update_project_label()
        from app.widgets.frame_preview import multimedia_backend_available

        if not multimedia_backend_available():
            warning = QLabel(
                "⚠ Video playback with sound needs PyQt6 ≥ 6.8 — run: "
                "pip install -r requirements.txt"
            )
            warning.setStyleSheet("color: #e0a030;")
            sb.addWidget(warning)

    # ------------------------------------------------------------------ modes
    def _ensure_mode_widget(self, index: int):
        if self._mode_widgets[index] is not None:
            return False
        if index == 0:
            from app.modes.mode1_preprocessing import Mode1Widget
            w = Mode1Widget(self.state)
        elif index == 1:
            from app.modes.mode2_signal_sync import Mode2Widget
            w = Mode2Widget(self.state)
        elif index == 2:
            from app.modes.mode3_synthetic_ppg import SyntheticPpgWidget
            w = SyntheticPpgWidget(self.state)
        elif index == 3:
            from app.modes.mode3_labelling import Mode3Widget
            w = Mode3Widget(self.state)
        elif index == 4:
            from app.modes.mode4_review import Mode4Widget
            w = Mode4Widget(self.state)
        else:
            w = QWidget()
        self._mode_widgets[index] = w
        self._stack.addWidget(w)
        return True

    def _on_tab_changed(self, index: int):
        created = self._ensure_mode_widget(index)
        widget = self._mode_widgets[index]
        if not created:
            refresh = getattr(widget, "refresh_from_state", None)
            if refresh is not None:
                refresh()
        self._stack.setCurrentWidget(widget)
        self.state.active_mode = index + 1
        self.statusBar().showMessage(_MODE_NAMES[index], 3000)

    def _current_mode(self):
        return self._stack.currentWidget()

    def _players(self):
        from app.widgets.frame_preview import MultiCameraPlayer

        return self.findChildren(MultiCameraPlayer)

    # ------------------------------------------------------------------ project
    def _state_snapshot(self) -> str:
        data = self.state.to_dict()
        for key in ("active_mode", "archive_removed_files", "blur_faces"):
            data.pop(key, None)
        return json.dumps(data, sort_keys=True, default=str)

    def _has_unsaved_changes(self) -> bool:
        return self._state_snapshot() != self._saved_snapshot

    def _confirm_discard(self) -> bool:
        if not self._has_unsaved_changes():
            return True
        reply = QMessageBox.question(
            self, "Unsaved changes",
            "The project has unsaved changes (for example labels or signal settings).\n"
            "Save them before continuing?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Cancel:
            return False
        if reply == QMessageBox.StandardButton.Save:
            return self._save_project()
        return True

    def _new_project(self):
        if not self._confirm_discard():
            return
        self.state = ProjectState(
            archive_removed_files=self._archive_removed_action.isChecked(),
            blur_faces=self._blur_faces_action.isChecked(),
        )
        self._saved_snapshot = self._state_snapshot()
        self._reload_modes()
        self.setWindowTitle("Video Research Tool — New project")
        self._update_project_label()

    def _open_project(self, path: str = ""):
        if not self._confirm_discard():
            return
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open project", "", _PROJECT_FILTER)
        if path:
            try:
                self.state = ProjectState.load(path)
                self.state.archive_removed_files = self._archive_removed_action.isChecked()
                self.state.blur_faces = self._blur_faces_action.isChecked()
                self._saved_snapshot = self._state_snapshot()
                self._reload_modes()
                self.setWindowTitle(f"Video Research Tool — {Path(path).stem}")
                self._add_recent(path)
                self._update_project_label()
            except Exception as exc:
                QMessageBox.critical(self, "Error", f"Failed to open project:\n{exc}")

    def _save_project(self) -> bool:
        if self.state.project_path:
            self.state.save(self.state.project_path)
            self._saved_snapshot = self._state_snapshot()
            self._add_recent(self.state.project_path)
            self.statusBar().showMessage("Saved.", 3000)
            return True
        return self._save_project_as()

    def _save_project_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(self, "Save project as", "", _PROJECT_FILTER)
        if not path:
            return False
        if not path.endswith(".vrt"):
            path += ".vrt"
        self.state.save(path)
        self._saved_snapshot = self._state_snapshot()
        self.setWindowTitle(f"Video Research Tool — {Path(path).stem}")
        self._add_recent(path)
        self._update_project_label()
        self.statusBar().showMessage(f"Saved to {path}", 3000)
        return True

    def _reload_modes(self):
        for player in self._players():
            player.shutdown()
        for i, w in enumerate(self._mode_widgets):
            if w is not None:
                self._stack.removeWidget(w)
                w.deleteLater()
        self._mode_widgets = [None, None, None, None, None]
        self._on_tab_changed(self._mode_tabs.currentIndex())

    def _update_project_label(self):
        path = self.state.project_path
        self._project_label.setText(f"Project: {Path(path).name}" if path else "Project: (unsaved)")

    def _recent_paths(self) -> list[str]:
        value = self._settings.value("recent_projects", [])
        if isinstance(value, str):
            value = [value]
        return [p for p in (value or []) if p]

    def _add_recent(self, path: str):
        path = str(Path(path).resolve())
        recent = [p for p in self._recent_paths() if p != path]
        recent.insert(0, path)
        self._settings.setValue("recent_projects", recent[:_MAX_RECENT])
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self):
        self._recent_menu.clear()
        recent = self._recent_paths()
        if not recent:
            action = self._recent_menu.addAction("(none)")
            action.setEnabled(False)
            return
        for path in recent:
            action = self._recent_menu.addAction(Path(path).name)
            action.setToolTip(path)
            action.setEnabled(Path(path).exists())
            action.triggered.connect(lambda _c=False, p=path: self._open_project(p))

    # ------------------------------------------------------------------ view
    def _apply_theme(self, name: str):
        name = "light" if name == "light" else "dark"
        self.setStyleSheet(_DARK_QSS if name == "dark" else _LIGHT_QSS)
        theme.set_theme(name)
        self._settings.setValue("theme", name)

    def _toggle_side_panel(self):
        from app.widgets.layout import find_workspace

        workspace = find_workspace(self._current_mode())
        if workspace is not None:
            workspace.toggle_left_panel()

    def _reset_layout(self):
        from app.widgets.layout import find_workspace

        workspace = find_workspace(self._current_mode())
        if workspace is not None:
            workspace.reset_layout()

    def _toggle_popout(self):
        from app.widgets.frame_preview import MultiCameraPlayer

        mode = self._current_mode()
        player = mode.findChild(MultiCameraPlayer) if mode is not None else None
        if player is not None:
            player.toggle_popout()

    # ------------------------------------------------------------------ options
    def _set_archive_removed_files(self, enabled: bool):
        self.state.archive_removed_files = enabled
        self._settings.setValue("archive_removed_files", enabled)
        policy = "archived" if enabled else "deleted permanently"
        self.statusBar().showMessage(f"Removed files will be {policy}.", 3000)

    def _set_blur_faces(self, enabled: bool):
        self.state.blur_faces = enabled
        self._settings.setValue("blur_faces", enabled)
        for player in self._players():
            player.set_face_blur_enabled(enabled)
        status = "enabled" if enabled else "disabled"
        self.statusBar().showMessage(f"Privacy face blurring {status}.", 3000)

    def _privacy_settings(self):
        from app.widgets.frame_preview import MultiCameraPlayer
        from app.widgets.privacy_dialog import PrivacySettingsDialog

        mode = self._current_mode()
        player = mode.findChild(MultiCameraPlayer) if mode is not None else None

        def frames():
            if player is None:
                return []
            return [(p.label_text, p.read_exact_frame()) for p in player.previews if p.video_path]

        dialog = PrivacySettingsDialog(get_privacy_settings(), frames, self)
        if dialog.exec():
            settings = dialog.settings()
            set_privacy_settings(settings)
            self._settings.setValue("privacy/settings", json.dumps(settings.to_dict()))
            for p in self._players():
                p.refresh_privacy()
            self.statusBar().showMessage("Face privacy settings updated.", 3000)

    # ------------------------------------------------------------------ help
    def _show_shortcuts(self):
        from app.widgets.frame_preview import SHORTCUT_HELP

        labelling = (
            ("I / O", "Start label / end label at the playhead (Labelling)"),
            ("1 … 9", "Choose label from the library (Labelling)"),
            ("R", "Apply the current label to the selected interval"),
            ("S", "Split the interval under the playhead"),
            ("N / P", "Jump to next / previous interval"),
            ("Delete", "Delete the selected interval"),
            ("Esc", "Cancel a pending label start"),
            ("Ctrl+Z / Ctrl+Y", "Undo / redo interval changes"),
            ("Page Up / Page Down", "Previous / next segment (Review)"),
            ("Timeline & graphs", "Wheel = zoom, Shift+wheel or drag = pan, click = seek"),
            ("Graphs", "Ctrl+drag = zoom to range, Shift+drag = draw an interval (Labelling)"),
            ("Timeline", "Drag empty space = new interval, drag edges = resize, drag selected = move "
                         "(Shift disables snapping)"),
            ("Ctrl+B", "Show / hide the side panel"),
            ("Ctrl+Shift+W", "Cameras in a separate window"),
        )
        rows = "".join(
            f"<tr><td style='padding-right:14px'><b>{key}</b></td><td>{text}</td></tr>"
            for key, text in (*SHORTCUT_HELP, *labelling)
        )
        box = QMessageBox(self)
        box.setWindowTitle("Keyboard shortcuts")
        box.setText(f"<table>{rows}</table>")
        box.exec()

    def _show_about(self):
        QMessageBox.about(
            self, "About",
            "Video Research Tool\n\n"
            "Multi-camera DJI video preprocessing, signal synchronisation, "
            "synthetic PPG generation, labelling, and review browser.\n\n"
            "Synthetic PPG generation adapts PPGSynth by Tang et al. under "
            "GNU GPL v3. See THIRD_PARTY_NOTICES.md and LICENSES/PPGSynth-GPL-3.0.txt.\n\n"
            "This program comes with absolutely no warranty. Built with PyQt6."
        )

    # ------------------------------------------------------------------ close
    def closeEvent(self, event):
        if not self._confirm_discard():
            event.ignore()
            return
        self._settings.setValue("main/geometry", self.saveGeometry())
        for player in self._players():
            player.shutdown()
        super().closeEvent(event)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Silence extremely verbose third-party debug loggers
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("librosa").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    app = QApplication(sys.argv)
    app.setApplicationName("Video Research Tool")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
