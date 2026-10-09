"""Dialog to export signal figures, label overviews and per-interval batches."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QSettings, Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from ..timefmt import format_hms_ms, parse_time_text
from ..visual_export import (
    FIGURE_FORMATS,
    FigureOptions,
    export_interval_figures,
    render_signal_figure,
    safe_filename,
    save_figure,
)

log = logging.getLogger(__name__)

SIZE_PRESETS = (
    ("Wide figure (10 × 5 in)", 10.0, 5.0),
    ("Slide 16:9 (13.3 × 7.5 in)", 13.33, 7.5),
    ("Paper, single column (3.5 × 2.6 in)", 3.5, 2.6),
    ("Paper, double column (7.2 × 3.6 in)", 7.2, 3.6),
    ("Label overview strip (12 × 1.6 in)", 12.0, 1.6),
)


class FigureExportDialog(QDialog):
    def __init__(self, snapshot, context: Optional[dict] = None,
                 initial_range: Optional[tuple[float, float]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export figure")
        self._snapshot = snapshot
        self._context = context or {}
        self._settings = QSettings("VideoResearchTool", "VRT")
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(250)
        self._preview_timer.timeout.connect(self._render_preview)

        root = QHBoxLayout(self)
        form_col = QVBoxLayout()
        root.addLayout(form_col, 0)

        # --- Range --------------------------------------------------------
        range_box = QGroupBox("Time range")
        range_lay = QVBoxLayout(range_box)
        self._range_group = QButtonGroup(self)
        self._r_view = QRadioButton(f"Current view ({self._fmt_range(snapshot.view)})")
        self._r_full = QRadioButton(f"Full session ({self._fmt_range(snapshot.data_range)})")
        selected = self._context.get("selected_interval")
        self._r_selected = QRadioButton(
            f"Selected interval: {selected[1]} ({self._fmt_range(selected[2:4])})" if selected else "Selected interval"
        )
        self._r_selected.setEnabled(bool(selected))
        self._r_custom = QRadioButton("Custom")
        batch = self._context.get("intervals") or []
        self._r_batch = QRadioButton(f"One figure per labelled interval ({len(batch)})")
        self._r_batch.setEnabled(bool(batch))
        for index, button in enumerate((self._r_view, self._r_full, self._r_selected, self._r_custom, self._r_batch)):
            self._range_group.addButton(button, index)
            range_lay.addWidget(button)
        custom_row = QHBoxLayout()
        custom_row.addSpacing(20)
        self._custom_start = QLineEdit(format_hms_ms(snapshot.view[0]))
        self._custom_end = QLineEdit(format_hms_ms(snapshot.view[1]))
        custom_row.addWidget(QLabel("from"))
        custom_row.addWidget(self._custom_start)
        custom_row.addWidget(QLabel("to"))
        custom_row.addWidget(self._custom_end)
        range_lay.addLayout(custom_row)
        pad_row = QHBoxLayout()
        pad_row.addSpacing(20)
        pad_row.addWidget(QLabel("Padding around intervals:"))
        self._padding = QDoubleSpinBox()
        self._padding.setRange(0.0, 3600.0)
        self._padding.setSuffix(" s")
        self._padding.setValue(0.0)
        pad_row.addWidget(self._padding)
        pad_row.addStretch()
        range_lay.addLayout(pad_row)
        if initial_range is not None:
            self._custom_start.setText(format_hms_ms(initial_range[0]))
            self._custom_end.setText(format_hms_ms(initial_range[1]))
            self._r_custom.setChecked(True)
        elif selected and self._context.get("prefer_selected"):
            self._r_selected.setChecked(True)
        else:
            self._r_view.setChecked(True)
        form_col.addWidget(range_box)

        # --- Content ------------------------------------------------------
        content_box = QGroupBox("Content")
        content_lay = QVBoxLayout(content_box)
        self._panel_checks: list[QCheckBox] = []
        for panel in snapshot.panels:
            check = QCheckBox(panel.title or panel.y_label or "Signal")
            check.setChecked(True)
            self._panel_checks.append(check)
            content_lay.addWidget(check)
        self._label_track = QCheckBox("Label timeline track")
        self._label_track.setChecked(bool(snapshot.intervals))
        self._label_track.setEnabled(bool(snapshot.intervals))
        self._labels_only = QCheckBox("Label timeline only (overview of labels)")
        self._labels_only.setEnabled(bool(snapshot.intervals))
        self._shading = QCheckBox("Shade labelled intervals on signals")
        self._shading.setChecked(True)
        self._cursor = QCheckBox("Show playhead")
        self._legend = QCheckBox("Legends")
        self._legend.setChecked(True)
        for widget in (self._label_track, self._labels_only, self._shading, self._cursor, self._legend):
            content_lay.addWidget(widget)
        form_col.addWidget(content_box)

        # --- Format -------------------------------------------------------
        fmt_box = QGroupBox("Format")
        fmt_lay = QFormLayout(fmt_box)
        self._title = QLineEdit()
        self._title.setPlaceholderText("Optional title")
        fmt_lay.addRow("Title:", self._title)
        self._preset = QComboBox()
        for name, _w, _h in SIZE_PRESETS:
            self._preset.addItem(name)
        self._preset.addItem("Custom")
        fmt_lay.addRow("Size preset:", self._preset)
        size_row = QHBoxLayout()
        self._width = QDoubleSpinBox()
        self._width.setRange(1.0, 60.0)
        self._width.setSuffix(" in")
        self._height = QDoubleSpinBox()
        self._height.setRange(0.8, 60.0)
        self._height.setSuffix(" in")
        size_row.addWidget(self._width)
        size_row.addWidget(QLabel("×"))
        size_row.addWidget(self._height)
        fmt_lay.addRow("Size:", size_row)
        self._dpi = QSpinBox()
        self._dpi.setRange(50, 1200)
        self._dpi.setValue(int(self._settings.value("figure/dpi", 300)))
        fmt_lay.addRow("Resolution (DPI):", self._dpi)
        self._font = QDoubleSpinBox()
        self._font.setRange(5.0, 30.0)
        self._font.setValue(float(self._settings.value("figure/font", 9.0)))
        fmt_lay.addRow("Font size (pt):", self._font)
        self._axis = QComboBox()
        self._axis.addItem("Video time (h:mm:ss)", "video")
        self._axis.addItem("Seconds", "seconds")
        self._axis.addItem("Clock time (UTC)", "clock")
        self._axis.setCurrentIndex(max(0, self._axis.findData(snapshot.time_axis_mode)))
        fmt_lay.addRow("Time axis:", self._axis)
        self._format = QComboBox()
        for fmt in FIGURE_FORMATS:
            self._format.addItem(fmt.upper(), fmt)
        self._format.setCurrentIndex(max(0, self._format.findData(self._settings.value("figure/format", "png"))))
        fmt_lay.addRow("File type:", self._format)
        form_col.addWidget(fmt_box)
        self._preset.currentIndexChanged.connect(self._apply_preset)
        self._preset.setCurrentIndex(int(self._settings.value("figure/preset", 0)))
        self._apply_preset(self._preset.currentIndex())

        buttons = QHBoxLayout()
        buttons.addStretch()
        save_btn = QPushButton("Save…")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._save)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        buttons.addWidget(save_btn)
        buttons.addWidget(close_btn)
        form_col.addLayout(buttons)

        preview_col = QVBoxLayout()
        root.addLayout(preview_col, 1)
        preview_col.addWidget(QLabel("Preview"))
        self._preview = QLabel()
        self._preview.setMinimumSize(560, 360)
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setStyleSheet("background: white; border: 1px solid #888;")
        preview_col.addWidget(self._preview, 1)
        self._preview_note = QLabel("")
        self._preview_note.setWordWrap(True)
        preview_col.addWidget(self._preview_note)

        for widget in (self._custom_start, self._custom_end, self._title):
            widget.textChanged.connect(lambda _t: self._preview_timer.start())
        for widget in (self._label_track, self._labels_only, self._shading, self._cursor, self._legend, *self._panel_checks):
            widget.toggled.connect(lambda _c: self._preview_timer.start())
        for widget in (self._width, self._height, self._font, self._padding):
            widget.valueChanged.connect(lambda _v: self._preview_timer.start())
        self._axis.currentIndexChanged.connect(lambda _i: self._preview_timer.start())
        self._range_group.idToggled.connect(lambda _i, _c: self._preview_timer.start())
        self._width.valueChanged.connect(self._mark_custom)
        self._height.valueChanged.connect(self._mark_custom)
        self.resize(1250, 720)
        self._preview_timer.start()

    @staticmethod
    def _fmt_range(rng) -> str:
        return f"{format_hms_ms(rng[0])} – {format_hms_ms(rng[1])}"

    def _apply_preset(self, index: int) -> None:
        if 0 <= index < len(SIZE_PRESETS):
            _name, width, height = SIZE_PRESETS[index]
            for spin, value in ((self._width, width), (self._height, height)):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
            if index == len(SIZE_PRESETS) - 1:
                self._labels_only.setChecked(True)
        self._preview_timer.start()

    def _mark_custom(self, *_args) -> None:
        self._preset.blockSignals(True)
        self._preset.setCurrentIndex(len(SIZE_PRESETS))
        self._preset.blockSignals(False)

    def _range(self) -> Optional[tuple[float, float]]:
        checked = self._range_group.checkedId()
        if checked == 0:
            return self._snapshot.view
        if checked == 1:
            return self._snapshot.data_range
        if checked == 2:
            selected = self._context.get("selected_interval")
            pad = self._padding.value()
            return (selected[2] - pad, selected[3] + pad) if selected else None
        if checked == 3:
            start = parse_time_text(self._custom_start.text())
            end = parse_time_text(self._custom_end.text())
            if start is None or end is None or end <= start:
                return None
            return start, end
        intervals = self._context.get("intervals") or []
        if intervals:
            first = intervals[0]
            return first[2] - self._padding.value(), first[3] + self._padding.value()
        return None

    def _options(self, x_range) -> FigureOptions:
        return FigureOptions(
            x_range=x_range,
            panel_indices=[i for i, c in enumerate(self._panel_checks) if c.isChecked()],
            include_label_track=self._label_track.isChecked(),
            include_shading=self._shading.isChecked(),
            include_cursor=self._cursor.isChecked(),
            include_legend=self._legend.isChecked(),
            title=self._title.text().strip(),
            width_in=self._width.value(),
            height_in=self._height.value(),
            dpi=self._dpi.value(),
            font_size=self._font.value(),
            time_axis_mode=self._axis.currentData(),
            labels_only=self._labels_only.isChecked(),
        )

    def _render_preview(self) -> None:
        x_range = self._range()
        if x_range is None:
            self._preview_note.setText("Enter a valid time range (e.g. 0:12:30.5 to 0:13:00).")
            return
        options = self._options(x_range)
        options.dpi = 80
        try:
            fig = render_signal_figure(self._snapshot, options)
        except Exception as exc:
            self._preview.clear()
            self._preview_note.setText(str(exc))
            return
        fig.canvas.draw()
        buffer = fig.canvas.buffer_rgba()
        width, height = fig.canvas.get_width_height()
        image = QImage(bytes(buffer), width, height, QImage.Format.Format_RGBA8888).copy()
        self._preview.setPixmap(QPixmap.fromImage(image).scaled(
            self._preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        note = (f"Output: {self._width.value():.2f} × {self._height.value():.2f} in at {self._dpi.value()} DPI "
                f"= {int(self._width.value() * self._dpi.value())} × {int(self._height.value() * self._dpi.value())} px")
        if self._range_group.checkedId() == 4:
            note += " — preview shows the first interval."
        self._preview_note.setText(note)

    def _remember(self) -> None:
        self._settings.setValue("figure/dpi", self._dpi.value())
        self._settings.setValue("figure/font", self._font.value())
        self._settings.setValue("figure/format", self._format.currentData())
        self._settings.setValue("figure/preset", self._preset.currentIndex())

    def _save(self) -> None:
        fmt = self._format.currentData()
        out_dir = Path(self._context.get("output_directory") or self._settings.value("figure/last_dir", "") or ".")
        self._remember()
        if self._range_group.checkedId() == 4:
            folder = QFileDialog.getExistingDirectory(self, "Folder for interval figures", str(out_dir / "figures"))
            if not folder:
                return
            intervals = [(i, label, s, e) for i, label, s, e in self._context.get("intervals", [])]
            progress = QProgressDialog("Exporting figures…", "Cancel", 0, len(intervals), self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)

            def on_progress(done, total):
                progress.setValue(done)
                QApplication.processEvents()

            try:
                written = export_interval_figures(
                    self._snapshot, intervals, folder, self._options((0.0, 1.0)), fmt=fmt,
                    padding_sec=self._padding.value(), progress=on_progress,
                )
            except Exception as exc:
                QMessageBox.critical(self, "Export figure", f"Export failed:\n{exc}")
                return
            finally:
                progress.close()
            self._settings.setValue("figure/last_dir", folder)
            QMessageBox.information(self, "Export figure", f"Saved {len(written)} figure(s) in\n{folder}")
            return
        x_range = self._range()
        if x_range is None:
            QMessageBox.warning(self, "Export figure", "The time range is not valid.")
            return
        stem = self._title.text().strip() or ("labels_overview" if self._labels_only.isChecked() else "signals")
        default = out_dir / f"{safe_filename(stem)}.{fmt}"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save figure", str(default), f"{fmt.upper()} (*.{fmt});;All files (*)"
        )
        if not path:
            return
        if not Path(path).suffix:
            path += f".{fmt}"
        try:
            save_figure(render_signal_figure(self._snapshot, self._options(x_range)), path, self._dpi.value())
        except Exception as exc:
            QMessageBox.critical(self, "Export figure", f"Export failed:\n{exc}")
            return
        self._settings.setValue("figure/last_dir", str(Path(path).parent))
        QMessageBox.information(self, "Export figure", f"Saved\n{path}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._preview_timer.start()
