"""Compact colour-grid popup for recolouring matched structures.

A small frameless ``Qt.Popup`` shown under a legend swatch: a 4 x 10 grid
of preset swatches (the matched palette plus lighter/darker shades), an
"Automatic" button that returns the structure to its palette colour, and
a "More..." button opening the full ``QColorDialog``. The widget knows
nothing about structures — it just emits the chosen hex colour (or a
reset request) and closes; the caller wires the signals.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mlgidlab.image_viewer import MATCHED_PALETTE

_GRID_COLUMNS = len(MATCHED_PALETTE)
_SWATCH_PX = 18


def grid_colors() -> list[str]:
    """The popup's preset hexes: the matched palette row, then a lighter
    and two darker shade rows of the same hues (40 in total)."""
    shades = (
        lambda c: c,
        lambda c: c.lighter(140),
        lambda c: c.darker(130),
        lambda c: c.darker(180),
    )
    return [
        shade(QColor(base)).name()
        for shade in shades
        for base in MATCHED_PALETTE
    ]


class ColorGridPopup(QWidget):
    """Colour chooser popup. ``colorPicked(hex)`` for a concrete choice,
    ``resetPicked()`` for "Automatic"; either closes the popup. Clicking
    anywhere outside dismisses it without a signal (``Qt.Popup``)."""

    colorPicked = Signal(str)
    resetPicked = Signal()

    def __init__(
        self, parent: QWidget | None = None, current: str | None = None
    ) -> None:
        super().__init__(parent, Qt.WindowType.Popup)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        # Starting colour for the QColorDialog behind "More...".
        self._current = current

        box = QVBoxLayout(self)
        box.setContentsMargins(6, 6, 6, 6)
        box.setSpacing(6)

        grid = QGridLayout()
        grid.setSpacing(2)
        self._swatch_buttons: list[QToolButton] = []
        for i, hex_color in enumerate(grid_colors()):
            btn = QToolButton(self)
            btn.setFixedSize(_SWATCH_PX, _SWATCH_PX)
            btn.setToolTip(hex_color)
            btn.setStyleSheet(
                f"background-color: {hex_color}; border: 1px solid #444;"
            )
            btn.clicked.connect(lambda _=False, c=hex_color: self._pick(c))
            grid.addWidget(btn, i // _GRID_COLUMNS, i % _GRID_COLUMNS)
            self._swatch_buttons.append(btn)
        box.addLayout(grid)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        self._auto_button = QPushButton("Automatic")
        self._auto_button.setToolTip(
            "Back to the automatically assigned palette colour"
        )
        self._auto_button.clicked.connect(self._reset)
        actions.addWidget(self._auto_button)
        self._more_button = QPushButton("More...")
        self._more_button.setToolTip("Pick any colour")
        self._more_button.clicked.connect(self._more)
        actions.addWidget(self._more_button)
        box.addLayout(actions)

    def show_under(self, widget: QWidget) -> None:
        """Open the popup just below ``widget`` (the clicked swatch)."""
        self.adjustSize()
        self.move(widget.mapToGlobal(QPoint(0, widget.height())))
        self.show()

    def _pick(self, hex_color: str) -> None:
        self.colorPicked.emit(hex_color)
        self.close()

    def _reset(self) -> None:
        self.resetPicked.emit()
        self.close()

    def _more(self) -> None:
        # Hide first: a Qt.Popup would dismiss itself the moment the
        # modal dialog grabs focus, taking this slot's widget with it
        # (WA_DeleteOnClose). Hidden, it survives until we close() below.
        self.hide()
        color = QColorDialog.getColor(
            QColor(self._current) if self._current else Qt.GlobalColor.white,
            self.parentWidget(),
            "Structure colour",
        )
        if color.isValid():
            self.colorPicked.emit(color.name())
        self.close()
