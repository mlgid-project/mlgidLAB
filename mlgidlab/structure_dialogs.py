"""Dialogs for creating things in an HDF5 file.

Split out of ``structure_panel`` so the panel stays a panel; the same
split ``phase_views_dialogs`` makes for the phase-views window.

Both dialogs are pure input forms: they collect what the user typed and
hand it back. Nothing here opens a file, and nothing here validates
against one — a name that collides or a shape that will not allocate is
reported by ``h5_edit``, where the file is, in the same message the user
would get from any other failed edit.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QWidget,
)

from mlgidlab import nexus_schema
from mlgidlab.h5_edit import DTYPE_CHOICES
from mlgidlab.widgets import make_form

#: The template entry that means "no template": a plain group with
#: whatever class the user types, or none at all.
PLAIN_GROUP = "Plain group"


class NewGroupDialog(QDialog):
    """Name, NeXus class and optional template for a new group.

    The template is the friendly half: picking *Sample (NXsample)* both
    stamps the class and pre-creates the fields that group nearly always
    has, so the user is editing values rather than looking up what an
    NXsample is supposed to contain.
    """

    def __init__(self, parent: QWidget | None = None, *, parent_path: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("New group")
        form = make_form(self)

        if parent_path:
            where = QLabel(f"Inside {parent_path}", self)
            where.setProperty("status", "muted")
            form.addRow("", where)

        self.template_combo = QComboBox(self)
        self.template_combo.addItem(PLAIN_GROUP)
        for key in nexus_schema.template_names():
            self.template_combo.addItem(nexus_schema.TEMPLATES[key].label, key)
        self.template_combo.currentIndexChanged.connect(self._on_template)
        form.addRow("Template:", self.template_combo)

        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("sample, instrument, notes, …")
        form.addRow("Name:", self.name_edit)

        self.class_combo = QComboBox(self)
        self.class_combo.setEditable(True)
        self.class_combo.addItem("")
        self.class_combo.addItems(nexus_schema.NX_CLASSES)
        self.class_combo.currentTextChanged.connect(self._on_class)
        form.addRow("NX_class:", self.class_combo)

        self.help = QLabel("", self)
        self.help.setProperty("status", "muted")
        self.help.setWordWrap(True)
        form.addRow("", self.help)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _on_template(self, _index: int) -> None:
        key = self.template_combo.currentData()
        if key is None:
            self.help.setText("")
            return
        template = nexus_schema.TEMPLATES[key]
        self.class_combo.setCurrentText(template.nx_class)
        if not self.name_edit.text().strip():
            # A sensible default name: the class without its NX prefix,
            # which is what these groups are conventionally called.
            self.name_edit.setText(template.nx_class[2:].lower())
        fields = ", ".join(f.name for f in template.fields)
        self.help.setText(
            f"{template.help}  Creates: {fields}." if fields else template.help
        )

    def _on_class(self, text: str) -> None:
        if self.template_combo.currentData() is None:
            self.help.setText(nexus_schema.class_help(text.strip()))

    def result_values(self) -> tuple[str, str, str | None]:
        """``(name, nx_class, template_key)``; the key is None for a plain group."""
        return (
            self.name_edit.text().strip(),
            self.class_combo.currentText().strip(),
            self.template_combo.currentData(),
        )


class NewDatasetDialog(QDialog):
    """Name, type, shape and fill value for a new dataset.

    Shape is typed as a comma-separated list, empty meaning a scalar,
    which is how NeXus writes most metadata fields. The units box exists
    because a NeXus number without units is an unfinished field, and
    offering the right ones up front is cheaper than a later audit.
    """

    def __init__(self, parent: QWidget | None = None, *, parent_path: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("New field")
        form = make_form(self)

        if parent_path:
            where = QLabel(f"Inside {parent_path}", self)
            where.setProperty("status", "muted")
            form.addRow("", where)

        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("wavelength, exposure_time, …")
        self.name_edit.textChanged.connect(self._on_name)
        form.addRow("Name:", self.name_edit)

        self.type_combo = QComboBox(self)
        self.type_combo.addItems(DTYPE_CHOICES)
        self.type_combo.setCurrentText("float64")
        form.addRow("Type:", self.type_combo)

        self.shape_edit = QLineEdit(self)
        self.shape_edit.setPlaceholderText("empty for a single value, or 3, 4")
        form.addRow("Shape:", self.shape_edit)

        self.value_edit = QLineEdit(self)
        self.value_edit.setPlaceholderText("0")
        form.addRow("Value / fill:", self.value_edit)

        self.units_combo = QComboBox(self)
        self.units_combo.setEditable(True)
        self.units_combo.addItems(nexus_schema.GENERIC_UNITS)
        form.addRow("Units:", self.units_combo)

        self.resizable_check = QCheckBox("Growable along the first axis", self)
        self.resizable_check.setToolTip(
            "Allows rows to be appended later without rewriting the dataset.")
        form.addRow("", self.resizable_check)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _on_name(self, name: str) -> None:
        """Offer the units the field's name implies, keeping what was typed."""
        current = self.units_combo.currentText()
        self.units_combo.clear()
        self.units_combo.addItems(nexus_schema.suggest_units(name.strip()))
        self.units_combo.setCurrentText(current)

    def result_values(self) -> tuple[str, str, str, str, str, bool]:
        """``(name, dtype, shape_text, value_text, units, resizable)``."""
        return (
            self.name_edit.text().strip(),
            self.type_combo.currentText(),
            self.shape_edit.text().strip(),
            self.value_edit.text().strip(),
            self.units_combo.currentText().strip(),
            self.resizable_check.isChecked(),
        )


def parse_shape(text: str) -> tuple[int, ...]:
    """``"3, 4"`` -> ``(3, 4)``; empty -> ``()``, a scalar.

    Raises ``ValueError`` with a message fit for a dialog.
    """
    text = (text or "").strip().strip("()[]")
    if not text:
        return ()
    parts = [p.strip() for p in text.split(",") if p.strip()]
    try:
        dims = tuple(int(p) for p in parts)
    except ValueError:
        raise ValueError(
            "A shape is a comma-separated list of whole numbers, "
            "for example 3, 4. Leave it empty for a single value."
        ) from None
    if any(d < 0 for d in dims):
        raise ValueError("A dimension cannot be negative.")
    return dims
