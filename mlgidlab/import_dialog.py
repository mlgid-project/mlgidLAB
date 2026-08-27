"""Dialog for importing pre-converted images as one N-frame scan.

For q-space maps produced outside mlgidLAB: the user picks the output
.h5 path, optionally supplies the q-axis ranges (else the axes fall
back to pixel indices), an incidence angle, and whether to flip frames
vertically. The heavy work happens in ``workers.ImportWorker`` via
``conversion.import_converted_stack`` — this dialog only collects the
parameters.

Kept Qt-only and free of fabio/h5py imports except the one lazy probe
that reads the first image's shape for the summary line.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mlgidlab import file_dialogs

import logging

logger = logging.getLogger(__name__)


def _range_spin(minimum: float, maximum: float, value: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(4)
    spin.setValue(value)
    return spin


class ImportConvertedDialog(QDialog):
    """Collects the parameters for a converted-image import.

    ``values()`` is valid only after ``exec()`` returned Accepted.
    The output-path picker runs through ``file_dialogs``, which owns
    both the shared browsing directory and the constant-time icon
    provider that keeps a detector-image folder listing instantly.
    """

    def __init__(self, paths: list[Path], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import images as converted scan")
        self._paths = list(paths)

        layout = QVBoxLayout(self)
        summary = QLabel(self._summary_text())
        summary.setWordWrap(True)
        layout.addWidget(summary)

        form = QFormLayout()

        default_out = (
            self._paths[0].parent
            / f"{self._paths[0].parent.name}_imported.h5"
        )
        self.out_path = QLineEdit(str(default_out))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_output)
        out_row = QHBoxLayout()
        out_row.setContentsMargins(0, 0, 0, 0)
        out_row.addWidget(self.out_path)
        out_row.addWidget(browse)
        out_widget = QWidget()
        out_widget.setLayout(out_row)
        form.addRow("Output file:", out_widget)

        self.entry_name = QLineEdit("entry_0000")
        form.addRow("Entry name:", self.entry_name)

        # Optional q axes — without them the entry shows pixel indices.
        self.use_q_ranges = QCheckBox(
            "I know the q ranges (1/Å) of these maps"
        )
        self.use_q_ranges.toggled.connect(self._on_use_q_toggled)
        form.addRow("", self.use_q_ranges)
        self.qxy_min = _range_spin(-100.0, 100.0, 0.0)
        self.qxy_max = _range_spin(-100.0, 100.0, 2.0)
        self.qz_min = _range_spin(-100.0, 100.0, 0.0)
        self.qz_max = _range_spin(-100.0, 100.0, 2.0)
        qxy_row = QHBoxLayout()
        qxy_row.setContentsMargins(0, 0, 0, 0)
        qxy_row.addWidget(self.qxy_min)
        qxy_row.addWidget(QLabel("to"))
        qxy_row.addWidget(self.qxy_max)
        qxy_widget = QWidget()
        qxy_widget.setLayout(qxy_row)
        form.addRow("q_xy range:", qxy_widget)
        qz_row = QHBoxLayout()
        qz_row.setContentsMargins(0, 0, 0, 0)
        qz_row.addWidget(self.qz_min)
        qz_row.addWidget(QLabel("to"))
        qz_row.addWidget(self.qz_max)
        qz_widget = QWidget()
        qz_widget.setLayout(qz_row)
        form.addRow("q_z range:", qz_widget)

        # Wavelength unlocks the pipeline: with real q ranges AND a
        # wavelength the importer writes a full instrument block (the
        # detector fields as documented zero placeholders — detection,
        # fitting and matching only consume image, q axes, wavelength
        # and incidence angle), so the standard pipeline runs on the
        # imported scan. Gated with the q ranges — without real axes
        # the pipeline would be meaningless anyway.
        self.wavelength_input = _range_spin(0.0, 100.0, 0.0)
        self.wavelength_input.setSuffix(" Å")
        self.wavelength_input.setSpecialValueText("unknown")
        self.wavelength_input.setToolTip(
            "The beam wavelength these maps were converted with. "
            "Providing it (together with the q ranges) makes the "
            "imported scan usable for detection, fitting and matching. "
            "Leave at 'unknown' for view-only import."
        )
        form.addRow("Wavelength:", self.wavelength_input)
        self._on_use_q_toggled(False)

        self.ai_input = _range_spin(0.0, 90.0, 0.0)
        self.ai_input.setToolTip(
            "Recorded as instrument/angle_of_incidence for every frame."
        )
        form.addRow("Incidence angle (deg):", self.ai_input)

        self.flip_vertical = QCheckBox(
            "Flip images vertically (q_z direction reversed in source)"
        )
        form.addRow("", self.flip_vertical)

        layout.addLayout(form)

        note = QLabel(
            "<i>With q ranges AND a wavelength, the imported scan is "
            "usable for detection, fitting and matching. Without them "
            "it imports view-only (no detector geometry exists for "
            "imported images).</i>"
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _summary_text(self) -> str:
        shape = "unknown"
        try:
            import fabio

            from mlgidlab import file_model

            file_model._quiet_fabio()
            data = fabio.open(str(self._paths[0])).data
            shape = f"{data.shape[0]} × {data.shape[1]}"
        except Exception:
            logger.debug("suppressed shape probe in import dialog", exc_info=True)
        return (
            f"<b>{len(self._paths)}</b> image files "
            f"(first frame: {shape} px) will be stacked as one scan "
            "in frame order shown in the batch."
        )

    def _on_use_q_toggled(self, checked: bool) -> None:
        for w in (
            self.qxy_min, self.qxy_max, self.qz_min, self.qz_max,
            self.wavelength_input,
        ):
            w.setEnabled(checked)

    def _browse_output(self) -> None:
        # The field's *name* carries over; the directory is the
        # app-wide browsing one (``file_dialogs``), same as every
        # other picker.
        path, _ = file_dialogs.save_file(
            self, "Save imported scan as",
            "NeXus / HDF5 (*.h5 *.hdf5 *.nxs)",
            suggested_name=self.out_path.text().strip(),
            default_suffix="h5",
        )
        if path:
            self.out_path.setText(path)

    def _validate_and_accept(self) -> None:
        if not self.out_path.text().strip():
            QMessageBox.warning(
                self, "Import", "Choose an output file path."
            )
            return
        if not self.entry_name.text().strip():
            QMessageBox.warning(self, "Import", "Entry name is empty.")
            return
        if self.use_q_ranges.isChecked():
            if (
                self.qxy_min.value() >= self.qxy_max.value()
                or self.qz_min.value() >= self.qz_max.value()
            ):
                QMessageBox.warning(
                    self,
                    "Import",
                    "Each q range needs min < max.",
                )
                return
        self.accept()

    def values(self) -> dict:
        """The collected import parameters (call after Accepted)."""
        use_q = self.use_q_ranges.isChecked()
        wavelength = self.wavelength_input.value()
        return {
            "out_path": Path(self.out_path.text().strip()),
            "entry_name": self.entry_name.text().strip(),
            "qxy_range": (
                (self.qxy_min.value(), self.qxy_max.value()) if use_q else None
            ),
            "qz_range": (
                (self.qz_min.value(), self.qz_max.value()) if use_q else None
            ),
            "ai": self.ai_input.value(),
            "flip_vertical": self.flip_vertical.isChecked(),
            "wavelength_A": (
                wavelength if use_q and wavelength > 0 else None
            ),
        }
