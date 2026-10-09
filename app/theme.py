"""Application colour theme shared by custom-painted widgets and plots."""
from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QObject, pyqtSignal


@dataclass(frozen=True)
class Palette:
    name: str
    background: str
    panel: str
    lane: str
    text: str
    muted_text: str
    grid: str
    axis: str
    playhead: str
    accent: str


DARK = Palette(
    name="dark",
    background="#1e1e1e",
    panel="#252526",
    lane="#2a2a2a",
    text="#d4d4d4",
    muted_text="#8a8a8a",
    grid="#3a3a3a",
    axis="#9a9a9a",
    playhead="#ff4040",
    accent="#0078d4",
)

LIGHT = Palette(
    name="light",
    background="#ffffff",
    panel="#f3f3f3",
    lane="#e8e8e8",
    text="#202020",
    muted_text="#6a6a6a",
    grid="#d0d0d0",
    axis="#505050",
    playhead="#d00000",
    accent="#0078d4",
)


class _ThemeNotifier(QObject):
    changed = pyqtSignal(object)


notifier = _ThemeNotifier()
_current = DARK


def current() -> Palette:
    return _current


def set_theme(name: str) -> None:
    global _current
    _current = LIGHT if name == "light" else DARK
    notifier.changed.emit(_current)
