"""Face privacy settings with a live test on the current video frame."""
from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
)

from ..face_privacy import (
    FacePrivacySettings,
    _expanded_region,
    apply_privacy_masks,
    detect_faces_scored,
)

FrameProvider = Callable[[], list[tuple[str, Optional[np.ndarray]]]]


def _to_pixmap(image: np.ndarray, max_w: int, max_h: int) -> QPixmap:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimage = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimage).scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation)


class PrivacySettingsDialog(QDialog):
    def __init__(self, settings: FacePrivacySettings, frame_provider: Optional[FrameProvider] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Face privacy settings")
        self._frames: list[tuple[str, Optional[np.ndarray]]] = frame_provider() if frame_provider else []
        self._frames = [(label, frame) for label, frame in self._frames if frame is not None]
        root = QHBoxLayout(self)

        form_col = QVBoxLayout()
        form = QFormLayout()
        self._style = QComboBox()
        self._style.addItem("Smooth blur (recommended)", "blur")
        self._style.addItem("Pixelate", "pixelate")
        self._style.addItem("Solid mask", "solid")
        self._style.setCurrentIndex(max(0, self._style.findData(settings.style)))
        form.addRow("Style:", self._style)
        self._strength = QSlider(Qt.Orientation.Horizontal)
        self._strength.setRange(1, 10)
        self._strength.setValue(settings.strength)
        form.addRow("Strength:", self._strength)
        self._quality = QComboBox()
        self._quality.addItem("Fast (misses small faces)", "fast")
        self._quality.addItem("Balanced — faces down to ~60 px in 4K", "balanced")
        self._quality.addItem("Thorough — faces down to ~35 px in 4K (slow)", "thorough")
        self._quality.setCurrentIndex(max(0, self._quality.findData(settings.detection_quality)))
        form.addRow("Detection:", self._quality)
        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(0.05, 0.95)
        self._threshold.setSingleStep(0.05)
        self._threshold.setValue(settings.score_threshold)
        self._threshold.setToolTip("Lower = more faces caught (and more false positives).")
        form.addRow("Sensitivity threshold:", self._threshold)
        self._margin = QDoubleSpinBox()
        self._margin.setRange(0.0, 1.5)
        self._margin.setSingleStep(0.05)
        self._margin.setValue(settings.margin_ratio)
        self._margin.setToolTip("Extra area around each detected face (fraction of the face size).")
        form.addRow("Margin:", self._margin)
        self._persistence = QDoubleSpinBox()
        self._persistence.setRange(0.0, 5.0)
        self._persistence.setSingleStep(0.1)
        self._persistence.setSuffix(" s")
        self._persistence.setValue(settings.persistence_sec)
        self._persistence.setToolTip("Keep masking a face this long after it was last detected (videos).")
        form.addRow("Keep mask after miss:", self._persistence)
        form_col.addLayout(form)
        note = QLabel(
            "Exports process every frame with tracking. The on-screen preview samples frames "
            "a few times per second, so masks can trail fast movement slightly.\n"
            "Automatic detection can still miss faces (e.g. seen from behind or heavily "
            "occluded); review exported material before sharing."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #8a8a8a;")
        form_col.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form_col.addStretch()
        form_col.addWidget(buttons)
        root.addLayout(form_col)

        preview_col = QVBoxLayout()
        top = QHBoxLayout()
        top.addWidget(QLabel("Test on current frame:"))
        self._camera = QComboBox()
        for label, _frame in self._frames:
            self._camera.addItem(label)
        top.addWidget(self._camera, 1)
        preview_col.addLayout(top)
        self._preview = QLabel("Load videos in the current section to test detection on a real frame.")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(640, 520)
        self._preview.setWordWrap(True)
        preview_col.addWidget(self._preview, 1)
        self._summary = QLabel("")
        preview_col.addWidget(self._summary)
        root.addLayout(preview_col, 1)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._update_preview)
        for widget in (self._style, self._quality, self._camera):
            widget.currentIndexChanged.connect(lambda _i: self._timer.start())
        self._strength.valueChanged.connect(lambda _v: self._timer.start())
        for spin in (self._threshold, self._margin):
            spin.valueChanged.connect(lambda _v: self._timer.start())
        if self._frames:
            self._timer.start()

    def settings(self) -> FacePrivacySettings:
        return FacePrivacySettings(
            style=self._style.currentData(),
            strength=self._strength.value(),
            detection_quality=self._quality.currentData(),
            score_threshold=self._threshold.value(),
            margin_ratio=self._margin.value(),
            persistence_sec=self._persistence.value(),
        ).normalised()

    def _update_preview(self) -> None:
        index = self._camera.currentIndex()
        if not (0 <= index < len(self._frames)):
            return
        frame = self._frames[index][1]
        settings = self.settings()
        faces = detect_faces_scored(frame, settings)
        h, w = frame.shape[:2]
        regions = [_expanded_region(face, w, h, settings.margin_ratio) for face in faces]
        masked = apply_privacy_masks(frame, regions, settings)
        boxed = frame.copy()
        thickness = max(2, w // 400)
        for (x, y, fw, fh, score), (x0, y0, x1, y1) in zip(faces, regions):
            cv2.rectangle(boxed, (int(x), int(y)), (int(x + fw), int(y + fh)), (0, 220, 255), thickness)
            cv2.rectangle(boxed, (x0, y0), (x1, y1), (255, 160, 0), max(1, thickness // 2))
            cv2.putText(boxed, f"{score:.2f}", (int(x), max(20, int(y) - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.6, w / 2000), (0, 220, 255), thickness)
        combined = np.hstack([boxed, masked])
        self._preview.setPixmap(_to_pixmap(combined, self._preview.width(), self._preview.height()))
        self._summary.setText(
            f"{len(faces)} face(s) detected. Left: detections (yellow) and masked area (blue). Right: result."
        )
