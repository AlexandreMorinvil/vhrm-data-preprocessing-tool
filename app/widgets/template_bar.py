"""Quick access to saved dataset templates (pick, apply, save)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..dataset_templates import (
    PART_TITLES,
    DatasetTemplate,
    TemplateStore,
    describe,
    describe_part,
    template_store,
)


class TemplateSaveDialog(QDialog):
    """Name a template and choose which captured parts it stores."""

    def __init__(self, captured: DatasetTemplate, store: TemplateStore, name: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save dataset template")
        self.setMinimumWidth(480)
        self._store = store
        self._captured = captured
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self._name = QComboBox()
        self._name.setEditable(True)
        self._name.addItems(store.names())
        self._name.setCurrentText(name)
        self._name.lineEdit().setPlaceholderText("e.g. Gym protocol — 3 cameras")
        self._name.currentTextChanged.connect(self._update_note)
        form.addRow("Template name:", self._name)
        layout.addLayout(form)

        layout.addWidget(QLabel("Include:"))
        self._parts: dict[str, QCheckBox] = {}
        for part, title in PART_TITLES.items():
            if not captured.has_part(part):
                continue
            box = QCheckBox(title.replace("&", "&&"))
            box.setChecked(True)
            layout.addWidget(box)
            detail = QLabel(describe_part(captured, part))
            detail.setWordWrap(True)
            detail.setStyleSheet("color: #8a8a8a; margin-left: 22px;")
            layout.addWidget(detail)
            self._parts[part] = box

        self._note = QLabel("")
        self._note.setWordWrap(True)
        layout.addWidget(self._note)
        self._default_cb = QCheckBox("Use this template at startup")
        self._default_cb.setToolTip(
            "The application starts with this template's cameras, labels and options."
        )
        layout.addWidget(self._default_cb)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._update_note()

    def name(self) -> str:
        return self._name.currentText().strip()

    def template(self) -> DatasetTemplate:
        excluded = [part for part, box in self._parts.items() if not box.isChecked()]
        return DatasetTemplate(name=self.name(), cameras=self._captured.cameras,
                               labels=self._captured.labels,
                               preprocessing=self._captured.preprocessing).without(excluded)

    def use_as_default(self) -> bool:
        return self._default_cb.isChecked()

    def _update_note(self) -> None:
        existing = self._store.get(self.name())
        self._default_cb.setChecked(bool(existing) and self._store.default_name == self.name())
        if existing is None:
            self._note.setText("")
            return
        self._note.setText(
            f"“{existing.name}” already exists and will be updated. "
            "Parts you leave unchecked keep their saved values."
        )

    def accept(self) -> None:
        if not self.name():
            QMessageBox.warning(self, "Save dataset template", "Enter a template name.")
            return
        if not any(box.isChecked() for box in self._parts.values()):
            QMessageBox.warning(self, "Save dataset template", "Choose at least one part to save.")
            return
        super().accept()


class TemplateBar(QWidget):
    """Template picker with Apply / Save buttons for one screen.

    *capture* returns the current setup of the screen as a template (parts the
    screen cannot provide are ``None``). *apply_parts* lists the parts the
    screen can fill in; applying emits :attr:`apply_requested` with only those.
    """

    apply_requested = pyqtSignal(object)
    message = pyqtSignal(str)

    def __init__(self, capture: Callable[[], DatasetTemplate], apply_parts: tuple[str, ...],
                 store: Optional[TemplateStore] = None, parent=None):
        super().__init__(parent)
        self._capture = capture
        self._apply_parts = apply_parts
        self._store = store or template_store()

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self._combo = QComboBox()
        self._combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._combo.setMinimumContentsLength(10)
        self._combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._combo.currentIndexChanged.connect(self._update_buttons)
        row.addWidget(self._combo, 1)

        applies = " and ".join(PART_TITLES[p].lower() for p in apply_parts)
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setToolTip(f"Fill in the {applies} from the selected template")
        self._apply_btn.clicked.connect(self.apply_selected)
        row.addWidget(self._apply_btn)

        self._save_btn = QPushButton("Save…")
        self._save_btn.setToolTip("Save the current setup as a template (or update one)")
        self._save_btn.clicked.connect(self.save_current)
        row.addWidget(self._save_btn)

        self._more_btn = QToolButton()
        self._more_btn.setText("⋯")
        self._more_btn.setToolTip("More template actions")
        self._more_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._more_btn.setStyleSheet("QToolButton::menu-indicator { image: none; width: 0; }")
        menu = QMenu(self._more_btn)
        self._default_action = QAction("Use at startup", self)
        self._default_action.setCheckable(True)
        self._default_action.triggered.connect(self._toggle_default)
        menu.addAction(self._default_action)
        menu.addSeparator()
        self._rename_action = menu.addAction("Rename…", self._rename)
        self._delete_action = menu.addAction("Delete…", self._delete)
        self._more_btn.setMenu(menu)
        row.addWidget(self._more_btn)

        self._store.changed.connect(self._reload)
        self._reload()
        if self._store.default_name:
            self.select(self._store.default_name)

    # ---------------------------------------------------------------- public
    def selected_name(self) -> str:
        return self._combo.currentData() or ""

    def select(self, name: str) -> None:
        index = self._combo.findData(name)
        if index >= 0:
            self._combo.setCurrentIndex(index)

    def apply_selected(self) -> None:
        template = self._store.get(self.selected_name())
        if template is None:
            return
        parts = [p for p in self._apply_parts if template.has_part(p)]
        if not parts:
            wanted = " or ".join(PART_TITLES[p].lower() for p in self._apply_parts)
            QMessageBox.information(self, "Apply template",
                                    f"“{template.name}” has no {wanted} to apply here.")
            return
        others = [p for p in PART_TITLES if p not in parts]
        self.apply_requested.emit(template.without(others))

    def save_current(self) -> None:
        captured = self._capture()
        if not captured.parts():
            QMessageBox.information(self, "Save dataset template", "There is nothing to save yet.")
            return
        dialog = TemplateSaveDialog(captured, self._store, self.selected_name(), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        saved = self._store.save(dialog.template())
        if dialog.use_as_default():
            self._store.set_default(saved.name)
        elif self._store.default_name == saved.name:
            self._store.set_default("")
        self.select(saved.name)
        self.message.emit(f"Saved template “{saved.name}”.")

    # ---------------------------------------------------------------- internals
    def _reload(self) -> None:
        current = self.selected_name()
        self._combo.blockSignals(True)
        self._combo.clear()
        for template in self._store.templates():
            is_default = template.name == self._store.default_name
            self._combo.addItem(f"★ {template.name}" if is_default else template.name, template.name)
            tooltip = describe(template) + ("\n★ Used at startup" if is_default else "")
            self._combo.setItemData(self._combo.count() - 1, tooltip, role=Qt.ItemDataRole.ToolTipRole)
        if self._combo.count() == 0:
            self._combo.setPlaceholderText("No templates yet — use Save…")
        index = self._combo.findData(current)
        self._combo.setCurrentIndex(index if index >= 0 else (0 if self._combo.count() else -1))
        self._combo.blockSignals(False)
        self._update_buttons()

    def _update_buttons(self) -> None:
        template = self._store.get(self.selected_name())
        has = template is not None
        self._apply_btn.setEnabled(has)
        self._rename_action.setEnabled(has)
        self._delete_action.setEnabled(has)
        self._default_action.setEnabled(has)
        self._default_action.setChecked(has and self._store.default_name == template.name)
        tooltip = "Saved dataset templates (★ = used at startup)"
        self._combo.setToolTip(f"{describe(template)}\n\n{tooltip}" if has else tooltip)

    def _toggle_default(self, checked: bool) -> None:
        name = self.selected_name()
        if not name:
            return
        self._store.set_default(name if checked else "")
        self.message.emit(f"The application will start with “{name}”." if checked
                          else "The application will start without a template.")

    def _rename(self) -> None:
        old = self.selected_name()
        if not old:
            return
        new, ok = QInputDialog.getText(self, "Rename template", f"New name for “{old}”:", text=old)
        if not ok or not new.strip() or new.strip() == old:
            return
        try:
            self._store.rename(old, new)
        except ValueError as exc:
            QMessageBox.warning(self, "Rename template", str(exc))
            return
        self.select(new.strip())

    def _delete(self) -> None:
        name = self.selected_name()
        if not name:
            return
        reply = QMessageBox.question(self, "Delete template", f"Delete the template “{name}”?")
        if reply == QMessageBox.StandardButton.Yes:
            self._store.delete(name)
            self.message.emit(f"Deleted template “{name}”.")
