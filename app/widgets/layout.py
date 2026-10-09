"""Shared layout building blocks for the mode screens."""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QSettings, Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QLayout,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

LEFT_PANE_MIN_WIDTH = 300
RIGHT_PANE_MIN_WIDTH = 480
LEFT_PANE_DEFAULT_WIDTH = 360


def configure_main_splitter(
    splitter: QSplitter,
    left: QWidget,
    right: QWidget,
    left_stretch: int = 1,
    right_stretch: int = 3,
) -> None:
    left.setMinimumWidth(LEFT_PANE_MIN_WIDTH)
    right.setMinimumWidth(RIGHT_PANE_MIN_WIDTH)
    splitter.setChildrenCollapsible(False)
    splitter.setStretchFactor(0, left_stretch)
    splitter.setStretchFactor(1, right_stretch)
    splitter.setSizes([LEFT_PANE_DEFAULT_WIDTH, RIGHT_PANE_MIN_WIDTH * 3])


def _settings() -> QSettings:
    return QSettings("VideoResearchTool", "VRT")


class CollapsibleSection(QWidget):
    """A titled section whose content can be folded away (state remembered)."""

    def __init__(self, title: str, content: QWidget | QLayout, expanded: bool = True,
                 settings_key: str = "", parent=None):
        super().__init__(parent)
        self._key = settings_key
        if settings_key:
            expanded = _settings().value(f"sections/{settings_key}", expanded, type=bool)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)
        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._toggle.setStyleSheet(
            "QToolButton { border: none; font-weight: bold; text-align: left; padding: 3px 2px; }"
        )
        self._toggle.toggled.connect(self.set_expanded)
        layout.addWidget(self._toggle)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #444;")
        layout.addWidget(line)
        if isinstance(content, QLayout):
            holder = QWidget()
            holder.setLayout(content)
            content = holder
        self._content = content
        layout.addWidget(content)
        self.set_expanded(expanded)

    @property
    def content(self) -> QWidget:
        return self._content

    def set_expanded(self, expanded: bool) -> None:
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        if self._toggle.isChecked() != expanded:
            self._toggle.setChecked(expanded)
        self._content.setVisible(expanded)
        if self._key:
            _settings().setValue(f"sections/{self._key}", expanded)


class ModeWorkspace(QWidget):
    """Collapsible control panel on the left, resizable work area on the right.

    The work area is a vertical splitter (videos / signals / timeline...);
    splitter sizes are remembered per mode.
    """

    def __init__(self, key: str, parent=None, left_width: int = LEFT_PANE_DEFAULT_WIDTH):
        super().__init__(parent)
        self._key = key
        self._left_default = left_width
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self._hsplit = QSplitter(Qt.Orientation.Horizontal)
        self._hsplit.setHandleWidth(5)
        root.addWidget(self._hsplit)

        self._left_scroll = QScrollArea()
        self._left_scroll.setWidgetResizable(True)
        self._left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left = QWidget()
        self.left_layout = QVBoxLayout(left)
        self.left_layout.setContentsMargins(6, 4, 6, 4)
        self.left_layout.setSpacing(4)
        self._left_scroll.setWidget(left)
        self._left_scroll.setMinimumWidth(LEFT_PANE_MIN_WIDTH)
        self._hsplit.addWidget(self._left_scroll)

        self._vsplit = QSplitter(Qt.Orientation.Vertical)
        self._vsplit.setHandleWidth(5)
        self._vsplit.setChildrenCollapsible(False)
        self._hsplit.addWidget(self._vsplit)
        self._hsplit.setCollapsible(0, True)
        self._hsplit.setCollapsible(1, False)
        self._hsplit.setStretchFactor(0, 0)
        self._hsplit.setStretchFactor(1, 1)
        self._hsplit.setSizes([left_width, 1400])
        self._saved_left_width = left_width
        self._default_vsizes: list[int] = []

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(400)
        self._save_timer.timeout.connect(self._save_state)
        self._hsplit.splitterMoved.connect(lambda *_: self._save_timer.start())
        self._vsplit.splitterMoved.connect(lambda *_: self._save_timer.start())
        self._restored = False

    # ---------------------------------------------------------------- left panel
    def add_left(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self.left_layout.addWidget(widget, stretch)
        return widget

    def add_section(self, title: str, content: QWidget | QLayout, expanded: bool = True) -> CollapsibleSection:
        key = f"{self._key}/{title}"
        section = CollapsibleSection(title, content, expanded, settings_key=key)
        self.left_layout.addWidget(section)
        return section

    def finish_left(self) -> None:
        self.left_layout.addStretch(1)

    def toggle_left_panel(self) -> None:
        sizes = self._hsplit.sizes()
        if sizes[0] > 0:
            self._saved_left_width = sizes[0]
            self._hsplit.setSizes([0, sizes[0] + sizes[1]])
        else:
            total = sum(sizes)
            width = max(LEFT_PANE_MIN_WIDTH, self._saved_left_width)
            self._hsplit.setSizes([width, max(1, total - width)])
        self._save_timer.start()

    # ---------------------------------------------------------------- work area
    def add_work(self, widget: QWidget, size: int) -> QWidget:
        """Append a pane to the vertical work area with a default height."""
        self._vsplit.addWidget(widget)
        self._default_vsizes.append(size)
        self._vsplit.setStretchFactor(self._vsplit.count() - 1, 1 if size >= 250 else 0)
        self._vsplit.setSizes(self._default_vsizes)
        return widget

    def reset_layout(self) -> None:
        self._hsplit.setSizes([self._left_default, max(1, self.width() - self._left_default)])
        self._vsplit.setSizes(self._default_vsizes)
        self._save_timer.start()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._restored:
            self._restored = True
            settings = _settings()
            h = settings.value(f"layout/{self._key}/h")
            v = settings.value(f"layout/{self._key}/v")
            if h is not None:
                self._hsplit.restoreState(h)
            if v is not None and self._vsplit.count() == len(self._default_vsizes):
                self._vsplit.restoreState(v)

    def _save_state(self) -> None:
        settings = _settings()
        settings.setValue(f"layout/{self._key}/h", self._hsplit.saveState())
        settings.setValue(f"layout/{self._key}/v", self._vsplit.saveState())


def find_workspace(widget: Optional[QWidget]) -> Optional[ModeWorkspace]:
    if widget is None:
        return None
    if isinstance(widget, ModeWorkspace):
        return widget
    return widget.findChild(ModeWorkspace)
