"""Reusable dataset setups (cameras, labels, preprocessing options).

Templates are stored once per user in the application settings, so they are
available in every project. Each part of a template is optional: ``None``
means "not part of this template", which lets a template hold only labels, or
only cameras, and lets a later save update one part without touching the rest.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Optional

from PyQt6.QtCore import QObject, QSettings, pyqtSignal

from .state import ProjectState

log = logging.getLogger(__name__)

_SETTINGS_KEY = "dataset_templates/data"
_FORMAT_VERSION = 1

# Template parts, in display order.
PART_CAMERAS = "cameras"
PART_LABELS = "labels"
PART_PREPROCESSING = "preprocessing"
PART_TITLES = {
    PART_CAMERAS: "Cameras",
    PART_LABELS: "Labels",
    PART_PREPROCESSING: "Audio sync & cleanup options",
}


@dataclass
class CameraTemplate:
    label: str = ""
    limits_common_duration: bool = True
    sync_audio_start_offset_sec: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CameraTemplate:
        return cls(
            label=str(d.get("label", "")).strip(),
            limits_common_duration=bool(d.get("limits_common_duration", True)),
            sync_audio_start_offset_sec=max(0.0, float(d.get("sync_audio_start_offset_sec", 0.0))),
        )


@dataclass
class PreprocessingOptions:
    sync_audio_duration_sec: float = 60.0
    sync_pairwise_refinement: bool = False
    delete_intermediates: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PreprocessingOptions:
        return cls(
            sync_audio_duration_sec=max(0.0, float(d.get("sync_audio_duration_sec", 60.0))),
            sync_pairwise_refinement=bool(d.get("sync_pairwise_refinement", False)),
            delete_intermediates=bool(d.get("delete_intermediates", True)),
        )


@dataclass
class DatasetTemplate:
    name: str
    cameras: Optional[list[CameraTemplate]] = None
    labels: Optional[list[str]] = None
    preprocessing: Optional[PreprocessingOptions] = None

    def has_part(self, part: str) -> bool:
        return getattr(self, part) is not None

    def parts(self) -> list[str]:
        return [part for part in PART_TITLES if self.has_part(part)]

    def without(self, parts) -> DatasetTemplate:
        """Return a copy with *parts* removed."""
        return replace(self, **{part: None for part in parts})

    def merged_over(self, older: Optional[DatasetTemplate]) -> DatasetTemplate:
        """Fill the parts missing from this template with those of *older*."""
        if older is None:
            return self
        return replace(self, **{
            part: getattr(older, part)
            for part in PART_TITLES if not self.has_part(part)
        })

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DatasetTemplate:
        cameras = d.get(PART_CAMERAS)
        labels = d.get(PART_LABELS)
        preprocessing = d.get(PART_PREPROCESSING)
        return cls(
            name=str(d.get("name", "")).strip(),
            cameras=[CameraTemplate.from_dict(c) for c in cameras] if cameras is not None else None,
            labels=clean_labels(labels) if labels is not None else None,
            preprocessing=PreprocessingOptions.from_dict(preprocessing) if preprocessing is not None else None,
        )


def clean_labels(labels) -> list[str]:
    """Strip names and drop empty or duplicate entries, keeping the order."""
    return list(dict.fromkeys(str(label).strip() for label in labels if str(label).strip()))


def describe_part(template: DatasetTemplate, part: str) -> str:
    """One-line human description of a template part."""
    if part == PART_CAMERAS and template.cameras is not None:
        names = [cam.label or f"Camera {i + 1}" for i, cam in enumerate(template.cameras)]
        extras = []
        not_limiting = [n for n, cam in zip(names, template.cameras) if not cam.limits_common_duration]
        if not_limiting:
            extras.append("not limiting size: " + ", ".join(not_limiting))
        offsets = [f"{n} {cam.sync_audio_start_offset_sec:g} s"
                   for n, cam in zip(names, template.cameras) if cam.sync_audio_start_offset_sec > 0]
        if offsets:
            extras.append("audio sync start offset: " + ", ".join(offsets))
        text = f"{len(names)} camera(s): " + ", ".join(names)
        return text + (f" ({'; '.join(extras)})" if extras else "")
    if part == PART_LABELS and template.labels is not None:
        return f"{len(template.labels)} label(s): " + (", ".join(template.labels) or "none")
    if part == PART_PREPROCESSING and template.preprocessing is not None:
        opts = template.preprocessing
        duration = (f"{opts.sync_audio_duration_sec:g} s" if opts.sync_audio_duration_sec > 0
                    else "full audio")
        return (f"audio sync on {duration}, pairwise refinement "
                f"{'on' if opts.sync_pairwise_refinement else 'off'}, delete intermediates "
                f"{'on' if opts.delete_intermediates else 'off'}")
    return "not included"


def describe(template: DatasetTemplate) -> str:
    lines = [f"{PART_TITLES[part]}: {describe_part(template, part)}"
             for part in template.parts()]
    return "\n".join(lines) or "Empty template"


def apply_labels(state: ProjectState, labels: list[str]) -> list[str]:
    """Replace the label library with *labels*.

    Labels still used by existing intervals are kept (appended at the end) so
    no interval loses its label. Returns the labels that were kept that way.
    """
    library = clean_labels(labels)
    kept = []
    for interval in state.intervals:
        if interval.label and interval.label not in library:
            library.append(interval.label)
            kept.append(interval.label)
    state.labels_library = library
    return kept



class TemplateStore(QObject):
    """Templates saved in the user's application settings."""

    changed = pyqtSignal()

    def __init__(self, settings: Optional[QSettings] = None, parent=None):
        super().__init__(parent)
        self._settings = settings or QSettings("VideoResearchTool", "VRT")
        self._templates: dict[str, DatasetTemplate] = {}
        self._default = ""
        self._load()

    # ---------------------------------------------------------------- queries
    def names(self) -> list[str]:
        return sorted(self._templates, key=str.casefold)

    def get(self, name: str) -> Optional[DatasetTemplate]:
        return self._templates.get(name)

    def templates(self) -> list[DatasetTemplate]:
        return [self._templates[name] for name in self.names()]

    @property
    def default_name(self) -> str:
        return self._default

    def default(self) -> Optional[DatasetTemplate]:
        return self._templates.get(self._default)

    # ---------------------------------------------------------------- changes
    def save(self, template: DatasetTemplate) -> DatasetTemplate:
        """Store *template*; parts it does not include keep their saved values."""
        name = template.name.strip()
        if not name:
            raise ValueError("A template needs a name.")
        template = replace(template, name=name).merged_over(self._templates.get(name))
        if not template.parts():
            raise ValueError("A template must include at least cameras, labels or options.")
        self._templates[name] = template
        self._commit()
        return template

    def delete(self, name: str) -> None:
        if self._templates.pop(name, None) is not None:
            if self._default == name:
                self._default = ""
            self._commit()

    def rename(self, old: str, new: str) -> None:
        new = new.strip()
        if old not in self._templates:
            raise KeyError(old)
        if not new:
            raise ValueError("A template needs a name.")
        if new != old and new in self._templates:
            raise ValueError(f"A template named “{new}” already exists.")
        self._templates[new] = replace(self._templates.pop(old), name=new)
        if self._default == old:
            self._default = new
        self._commit()

    def set_default(self, name: str) -> None:
        """Use *name* for new projects ("" for none)."""
        if name and name not in self._templates:
            raise KeyError(name)
        self._default = name
        self._commit()

    # ---------------------------------------------------------------- storage
    def _load(self) -> None:
        raw = self._settings.value(_SETTINGS_KEY, "")
        if not raw:
            return
        try:
            data = json.loads(raw)
            for item in data.get("templates", []):
                template = DatasetTemplate.from_dict(item)
                if template.name and template.parts():
                    self._templates[template.name] = template
        except (TypeError, ValueError, AttributeError) as exc:
            log.warning("Ignoring invalid stored dataset templates: %s", exc)
            return
        default = data.get("default", "")
        self._default = default if default in self._templates else ""

    def _commit(self) -> None:
        data = {
            "version": _FORMAT_VERSION,
            "default": self._default,
            "templates": [t.to_dict() for t in self.templates()],
        }
        self._settings.setValue(_SETTINGS_KEY, json.dumps(data, ensure_ascii=False))
        self._settings.sync()
        self.changed.emit()


_store: Optional[TemplateStore] = None


def template_store() -> TemplateStore:
    """The application-wide store shared by every screen."""
    global _store
    if _store is None:
        _store = TemplateStore()
    return _store
