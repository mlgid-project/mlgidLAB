"""The Structure tab: what the file browser's selection actually is.

The third central tab, beside Image and Data. The File-browser dock
stays the one place a file is navigated; this panel answers the
questions the tree cannot show in a row — the full path, the type and
shape, every attribute, what a link points at, whether the pipeline
depends on this node, and what is wrong with the file as a whole.

Display only. Everything it shows is handed to it by
``MainWindow``'s ``StructureMixin``, which owns the file handles; the
panel opens nothing and reads nothing. Later stages add the editing
affordances on top of exactly these sections.
"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mlgidlab import icons
from mlgidlab.h5_edit import Issue, NodeInfo
from mlgidlab.widgets import (
    GAP,
    PAD,
    Card,
    attach_empty_hint,
    skin_item_view,
)

import logging

logger = logging.getLogger(__name__)

#: Level -> the skin's semantic state name, for the issue rows and the
#: Check card's header status.
_ISSUE_STATUS = {"error": "error", "warning": "warn", "info": "info"}

#: Shown in the header when nothing is selected.
_EMPTY_HINT = (
    "Select a group or a dataset in the File browser to see what it is."
)


def _human_bytes(count: int) -> str:
    """``4600`` -> ``4.6 kB``. Decimal units, matching what h5 tools print."""
    value = float(count)
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000
    return f"{value:.1f} TB"  # pragma: no cover - unreachable, kept explicit


def describe_node(info: NodeInfo) -> str:
    """The one-line type summary under the path.

    Everything a person wants before deciding whether to touch a node:
    what it is, what it holds, how big it is, and how it is stored.
    """
    if info.kind == "missing":
        return "not in this file"
    if info.kind == "link":
        return info.link.describe()
    if info.kind == "group":
        children = info.n_children or 0
        parts = [
            info.nx_class or "group",
            f"{children} child" if children == 1 else f"{children} children",
        ]
        if info.link.kind != "hard":
            parts.append(info.link.describe())
        return " · ".join(parts)

    shape = " × ".join(str(n) for n in info.shape) if info.shape else "scalar"
    parts = ["compound" if info.is_compound else info.dtype, shape]
    if info.nbytes:
        parts.append(_human_bytes(info.nbytes))
    if info.chunks:
        parts.append("chunked")
    if info.compression:
        parts.append(str(info.compression))
    if info.maxshape and any(n is None for n in info.maxshape):
        parts.append("resizable")
    if info.link.kind != "hard":
        parts.append(info.link.describe())
    return " · ".join(parts)


class StructurePanel(QWidget):
    """Read-out of one HDF5 node, plus the file's health."""

    #: The user asked for the file to be re-checked.
    recheckRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._info: NodeInfo | None = None

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        # The path line elides and the tables scroll internally, so the
        # page never needs a horizontal bar of its own.
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        page = QWidget(scroll)
        column = QVBoxLayout(page)
        column.setContentsMargins(PAD * 2, PAD * 2, PAD * 2, PAD * 2)
        column.setSpacing(PAD * 2)

        column.addLayout(self._build_header(page))
        column.addWidget(self._build_attributes_card(page))
        self._value_card = self._build_value_card(page)
        column.addWidget(self._value_card)
        self._link_card = self._build_link_card(page)
        column.addWidget(self._link_card)
        column.addWidget(self._build_check_card(page))
        column.addStretch(1)

        scroll.setWidget(page)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self.clear()

    # -- construction ------------------------------------------------------

    def _build_header(self, parent: QWidget) -> QVBoxLayout:
        box = QVBoxLayout()
        box.setSpacing(2)

        top = QHBoxLayout()
        top.setSpacing(GAP)
        self._path_label = QLabel(parent)
        self._path_label.setProperty("role", "card-title")
        self._path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        # The path is the one thing worth copying out of this panel, so
        # it stays selectable and wraps rather than eliding into a "…"
        # the user cannot recover.
        self._path_label.setWordWrap(True)
        top.addWidget(self._path_label, 1)

        self._badge = QLabel(parent)
        self._badge.setProperty("status", "warn")
        self._badge.hide()
        top.addWidget(self._badge, 0, Qt.AlignmentFlag.AlignTop)
        box.addLayout(top)

        self._type_label = QLabel(parent)
        self._type_label.setProperty("status", "muted")
        self._type_label.setWordWrap(True)
        box.addWidget(self._type_label)

        self._note_label = QLabel(parent)
        self._note_label.setProperty("status", "muted")
        self._note_label.setWordWrap(True)
        self._note_label.hide()
        box.addWidget(self._note_label)
        return box

    def _build_attributes_card(self, parent: QWidget) -> Card:
        card = Card("Attributes", parent=parent)
        self._attr_card = card
        self._attr_table = QTableWidget(0, 2, card)
        skin_item_view(self._attr_table)
        self._attr_table.setHorizontalHeaderLabels(["Name", "Value"])
        self._attr_table.verticalHeader().setVisible(False)
        self._attr_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._attr_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self._attr_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._attr_table.setMinimumHeight(120)
        attach_empty_hint(self._attr_table, "No attributes on this node.")
        card.body_layout.addWidget(self._attr_table)
        return card

    def _build_value_card(self, parent: QWidget) -> Card:
        card = Card("Value", parent=parent)
        self._value_label = QLabel(card)
        self._value_label.setWordWrap(True)
        self._value_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        card.body_layout.addWidget(self._value_label)
        return card

    def _build_link_card(self, parent: QWidget) -> Card:
        card = Card("Link", parent=parent)
        self._link_label = QLabel(card)
        self._link_label.setWordWrap(True)
        self._link_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        card.body_layout.addWidget(self._link_label)
        return card

    def _build_check_card(self, parent: QWidget) -> Card:
        card = Card("Check", parent=parent)
        self._check_card = card
        row = QHBoxLayout()
        row.setSpacing(GAP)
        self._recheck_btn = QToolButton(card)
        self._recheck_btn.setText("Re-check")
        self._recheck_btn.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        icons.bind(self._recheck_btn, "refresh")
        self._recheck_btn.setAutoRaise(True)
        self._recheck_btn.setToolTip(
            "Re-read the file's top-level layout and report anything that "
            "would stop the viewer or the pipeline from opening it."
        )
        self._recheck_btn.clicked.connect(self.recheckRequested)
        row.addWidget(self._recheck_btn)
        row.addStretch(1)
        card.body_layout.addLayout(row)

        self._issue_table = QTableWidget(0, 2, card)
        skin_item_view(self._issue_table)
        self._issue_table.setHorizontalHeaderLabels(["Where", "What"])
        self._issue_table.verticalHeader().setVisible(False)
        self._issue_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._issue_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        issue_header = self._issue_table.horizontalHeader()
        issue_header.setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        issue_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._issue_table.setMinimumHeight(90)
        attach_empty_hint(
            self._issue_table,
            "Nothing to report — the viewer and the pipeline can read "
            "this file's layout.",
        )
        card.body_layout.addWidget(self._issue_table)
        return card

    # -- population --------------------------------------------------------

    def clear(self, message: str = "") -> None:
        """Empty every section. ``message`` replaces the default hint."""
        self._info = None
        self._path_label.setText(message or _EMPTY_HINT)
        self._type_label.setText("")
        self._badge.hide()
        self._note_label.hide()
        self._attr_table.setRowCount(0)
        self._attr_card.set_status("")
        self._value_card.hide()
        self._link_card.hide()

    def set_note(self, text: str, level: str = "muted") -> None:
        """A line under the type summary — why editing is off, mostly."""
        self._note_label.setText(text)
        self._note_label.setProperty("status", level)
        style = self._note_label.style()
        style.unpolish(self._note_label)
        style.polish(self._note_label)
        self._note_label.setVisible(bool(text))

    def set_node(
        self,
        info: NodeInfo,
        attrs: dict[str, Any],
        preview: str = "",
        *,
        file_label: str = "",
    ) -> None:
        """Show one node. ``file_label`` prefixes the path when set."""
        self._info = info
        self._path_label.setText(
            f"{file_label}::{info.path}" if file_label else info.path)
        self._type_label.setText(describe_node(info))

        if info.protection:
            self._badge.setText(f"⛨  {info.protection}")
            self._badge.setToolTip(
                "The viewer or the pipeline reads this node. Editing it is "
                "allowed; anything destructive will ask first."
            )
            self._badge.show()
        else:
            self._badge.hide()

        self._fill_attributes(attrs)

        if info.kind == "dataset":
            self._value_label.setText(preview or "(empty)")
            self._value_card.show()
        else:
            self._value_card.hide()

        if info.link.kind != "hard":
            self._link_label.setText(info.link.describe())
            self._link_card.show()
        else:
            self._link_card.hide()

    def _fill_attributes(self, attrs: dict[str, Any]) -> None:
        from mlgidlab.h5_edit import format_value

        self._attr_table.setRowCount(0)
        for name, value in attrs.items():
            row = self._attr_table.rowCount()
            self._attr_table.insertRow(row)
            self._attr_table.setItem(row, 0, QTableWidgetItem(str(name)))
            item = QTableWidgetItem(format_value(value))
            item.setToolTip(format_value(value, limit=2000))
            self._attr_table.setItem(row, 1, item)
        count = self._attr_table.rowCount()
        self._attr_card.set_status(f"{count}" if count else "")

    def set_issues(self, issues: list[Issue]) -> None:
        """Fill the Check card. An empty list reads as a clean bill."""
        self._issue_table.setRowCount(0)
        worst = ""
        for issue in issues:
            row = self._issue_table.rowCount()
            self._issue_table.insertRow(row)
            where = QTableWidgetItem(issue.path)
            self._issue_table.setItem(row, 0, where)
            text = issue.message
            if issue.fix:
                text = f"{text}  →  {issue.fix}"
            item = QTableWidgetItem(text)
            item.setToolTip(text)
            self._issue_table.setItem(row, 1, item)
            if issue.level == "error":
                worst = "error"
            elif issue.level == "warning" and worst != "error":
                worst = "warning"
            elif not worst:
                worst = "info"
        counts = {
            level: sum(1 for i in issues if i.level == level)
            for level in ("error", "warning", "info")
        }
        if not issues:
            self._check_card.set_status("clean", "ok")
            return
        summary = ", ".join(
            f"{n} {name}" for name, n in (
                ("error", counts["error"]),
                ("warning", counts["warning"]),
                ("note", counts["info"]),
            ) if n
        )
        self._check_card.set_status(summary, _ISSUE_STATUS.get(worst, "muted"))

    # -- read-back, for the tests and for later stages ---------------------

    @property
    def current(self) -> NodeInfo | None:
        return self._info

    def attribute_rows(self) -> list[tuple[str, str]]:
        return [
            (self._attr_table.item(r, 0).text(), self._attr_table.item(r, 1).text())
            for r in range(self._attr_table.rowCount())
        ]

    def issue_rows(self) -> list[tuple[str, str]]:
        return [
            (self._issue_table.item(r, 0).text(),
             self._issue_table.item(r, 1).text())
            for r in range(self._issue_table.rowCount())
        ]
