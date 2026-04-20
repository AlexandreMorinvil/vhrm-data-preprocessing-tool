"""Dialog for subdividing labelled intervals by count or duration."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)


class SubdivideDialog(QDialog):
    """Modal dialog that returns (mode, value) where mode is 'count' or 'duration'."""

    def __init__(self, interval_label: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Subdivide interval")
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)

        if interval_label:
            layout.addWidget(QLabel(f"Subdivide: <b>{interval_label}</b>"))

        # --- By count ---
        self._count_radio = QRadioButton("Split into N equal segments")
        self._count_radio.setChecked(True)
        layout.addWidget(self._count_radio)

        count_row = QHBoxLayout()
        count_row.addSpacing(20)
        count_row.addWidget(QLabel("N:"))
        self._count_spin = QSpinBox()
        self._count_spin.setRange(2, 999)
        self._count_spin.setValue(2)
        count_row.addWidget(self._count_spin)
        count_row.addStretch()
        layout.addLayout(count_row)

        # --- By duration ---
        self._dur_radio = QRadioButton("Split into segments of X seconds")
        layout.addWidget(self._dur_radio)

        dur_row = QHBoxLayout()
        dur_row.addSpacing(20)
        dur_row.addWidget(QLabel("Duration (s):"))
        self._dur_spin = QDoubleSpinBox()
        self._dur_spin.setRange(0.5, 36000.0)
        self._dur_spin.setValue(30.0)
        self._dur_spin.setDecimals(1)
        self._dur_spin.setSingleStep(5.0)
        dur_row.addWidget(self._dur_spin)
        dur_row.addStretch()
        layout.addLayout(dur_row)

        # Mutual exclusion
        self._group = QButtonGroup(self)
        self._group.addButton(self._count_radio, 0)
        self._group.addButton(self._dur_radio, 1)
        self._group.idToggled.connect(self._on_toggle)
        self._dur_spin.setEnabled(False)

        # Buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _on_toggle(self, btn_id: int, checked: bool):
        self._count_spin.setEnabled(self._count_radio.isChecked())
        self._dur_spin.setEnabled(self._dur_radio.isChecked())

    def result_values(self) -> tuple[str, float]:
        """Return ``('count', N)`` or ``('duration', seconds)``."""
        if self._count_radio.isChecked():
            return ("count", float(self._count_spin.value()))
        return ("duration", self._dur_spin.value())
