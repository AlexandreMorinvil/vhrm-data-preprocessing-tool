from __future__ import annotations

import logging
import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QStackedWidget,
    QStatusBar,
    QTabBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app.state import ProjectState

log = logging.getLogger(__name__)

_PROJECT_FILTER = "VRT Project (*.vrt);;All files (*)"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Research Tool")
        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(
                min(1400, int(available.width() * 0.9)),
                min(900, int(available.height() * 0.9)),
            )
        else:
            self.resize(1400, 900)

        self.state = ProjectState()
        self._settings = QSettings("VideoResearchTool", "VRT")

        self._build_menus()
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()

        self._apply_theme("dark")
        self._mode_tabs.setCurrentIndex(0)
        self._on_tab_changed(0)

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

        help_menu = mb.addMenu("&Help")
        about_act = QAction("&About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _build_toolbar(self):
        tb = QToolBar("Modes")
        tb.setMovable(False)
        self.addToolBar(tb)

        self._mode_tabs = QTabBar()
        self._mode_tabs.addTab("1. Preprocessing")
        self._mode_tabs.addTab("2. Signal sync")
        self._mode_tabs.addTab("3. Labelling")
        self._mode_tabs.addTab("4. Review")
        self._mode_tabs.currentChanged.connect(self._on_tab_changed)
        tb.addWidget(self._mode_tabs)

    def _build_central(self):
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._mode_widgets: list[QWidget] = [None, None, None, None]

    def _build_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._status_project = QAction("No project", self)
        sb.addWidget(QWidget())
        self._status_label = sb

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
            from app.modes.mode3_labelling import Mode3Widget
            w = Mode3Widget(self.state)
        elif index == 3:
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
        self.statusBar().showMessage(f"Mode {index + 1}", 3000)

    def _new_project(self):
        self.state = ProjectState()
        self._reload_modes()
        self.setWindowTitle("Video Research Tool — New project")

    def _open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", _PROJECT_FILTER)
        if path:
            try:
                self.state = ProjectState.load(path)
                self._reload_modes()
                self.setWindowTitle(f"Video Research Tool — {Path(path).stem}")
            except Exception as exc:
                QMessageBox.critical(self, "Error", f"Failed to open project:\n{exc}")

    def _save_project(self):
        if self.state.project_path:
            self.state.save(self.state.project_path)
            self.statusBar().showMessage("Saved.", 3000)
        else:
            self._save_project_as()

    def _save_project_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save project as", "", _PROJECT_FILTER)
        if path:
            if not path.endswith(".vrt"):
                path += ".vrt"
            self.state.save(path)
            self.setWindowTitle(f"Video Research Tool — {Path(path).stem}")
            self.statusBar().showMessage(f"Saved to {path}", 3000)

    def _reload_modes(self):
        for i, w in enumerate(self._mode_widgets):
            if w is not None:
                self._stack.removeWidget(w)
                w.deleteLater()
        self._mode_widgets = [None, None, None, None]
        self._on_tab_changed(self._mode_tabs.currentIndex())

    def _apply_theme(self, name: str):
        if name == "dark":
            qss = (
                "QWidget { background: #1e1e1e; color: #cccccc; font-size: 13px; }"
                "QGroupBox { border: 1px solid #444; margin-top: 8px; padding-top: 12px; }"
                "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
                "QPushButton { background: #2d2d30; border: 1px solid #555; padding: 4px 12px; border-radius: 3px; }"
                "QPushButton:hover { background: #3e3e42; }"
                "QLineEdit, QSpinBox, QComboBox { background: #2d2d30; border: 1px solid #555; padding: 3px; }"
                "QSpinBox::up-button, QSpinBox::down-button { background: #3e3e42; border: 1px solid #555; width: 16px; }"
                "QSpinBox::up-button:hover, QSpinBox::down-button:hover { background: #555; }"
                "QSpinBox::up-arrow { image: none; border-left: 4px solid transparent; border-right: 4px solid transparent; border-bottom: 5px solid #cccccc; width: 0; height: 0; }"
                "QSpinBox::down-arrow { image: none; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 5px solid #cccccc; width: 0; height: 0; }"
                "QSlider::groove:horizontal { height: 6px; background: #444; border-radius: 3px; }"
                "QSlider::handle:horizontal { width: 14px; margin: -4px 0; background: #0078d4; border-radius: 7px; }"
                "QListWidget { background: #252526; }"
                "QTextEdit { background: #1b1b1b; }"
                "QProgressBar { text-align: center; border: 1px solid #555; background: #2d2d30; }"
                "QProgressBar::chunk { background: #0078d4; }"
                "QTabBar::tab { background: #2d2d30; padding: 6px 16px; border: 1px solid #444; }"
                "QTabBar::tab:selected { background: #0078d4; color: #ffffff; }"
            )
        else:
            qss = (
                "QWidget { font-size: 13px; }"
                "QTabBar::tab { padding: 6px 16px; }"
                "QTabBar::tab:selected { background: #0078d4; color: #ffffff; }"
            )
        self.setStyleSheet(qss)
        self._settings.setValue("theme", name)

    def _show_about(self):
        QMessageBox.about(
            self, "About",
            "Video Research Tool\n\n"
            "Multi-camera DJI video preprocessing, signal synchronisation, "
            "labelling, and review browser.\n\n"
            "Built with PyQt6."
        )


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Silence extremely verbose third-party debug loggers
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("librosa").setLevel(logging.INFO)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
