"""Small shared Qt widgets and builders used across panels.

Deliberately light on imports: the conversion panel and the figure-export
window used to carry verbatim copies of these because importing them from
``pipeline_panel`` would have pulled its mlgidbase-heavy import chain
along. The refined rule is *no import that reaches the analysis backend*
— ``skin`` and ``theme_tokens`` are stdlib/Qt-free and therefore fine,
and importing the variant names from ``skin`` keeps one definition
instead of two that can drift.
"""
from __future__ import annotations

from typing import Callable

from mlgidlab.skin import DANGER, PRIMARY  # noqa: F401  (re-exported)

from PySide6.QtCore import QEvent, QObject, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QProgressBar,
    QProgressDialog,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


def set_variant(widget: QWidget, variant: str = "") -> QWidget:
    """Tag ``widget`` so the skin paints it as a primary/destructive action.

    Returns the widget, so a construction line stays a single statement::

        self.btn_run = set_variant(QPushButton("Run"), PRIMARY)

    The re-polish matters when the variant changes *after* the widget was
    first polished (e.g. an emphasis that comes and goes): Qt caches the
    resolved style, so an attribute selector is not re-evaluated on its
    own. Passing ``""`` clears the tag.
    """
    widget.setProperty("variant", variant)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    return widget


def skin_item_view(view: QWidget) -> QWidget:
    """Tag an item view so the skin paints its grid, header and rows.

    Presentation only, deliberately: selection behaviour, edit triggers,
    sorting and column sizing stay at the call sites, because those are
    behaviour that the table tests pin and the two tables genuinely
    differ on. It replaces the padding stylesheet that was copy-pasted
    into both of them, and adds what qdarkstyle never shipped — it has
    no QTableView rules at all, so alternating rows fell back to the OS
    palette and were near-invisible in the light theme.
    """
    view.setProperty("mlgid", "table")
    view.setAlternatingRowColors(True)
    return view


def skin_progress(widget: QWidget) -> QWidget:
    """Tag a progress bar so the skin fills it with the accent.

    Accepts a ``QProgressDialog`` too and tags the bar Qt builds inside
    it, which is otherwise unreachable: the dialog's own objectName
    would not match a ``QProgressBar`` selector, and an ancestor-scoped
    rule is exactly what the skin's scoping rule forbids.

    Opt-in rather than a blanket ``QProgressBar`` rule because silx and
    pyFAI have progress bars of their own, and a bare element selector
    would repaint them (see the module docstring in ``skin``).
    """
    bar = widget
    if isinstance(widget, QProgressDialog):
        bar = widget.findChild(QProgressBar)
        if bar is None:                       # pragma: no cover - defensive
            return widget
    bar.setProperty("mlgid", "progress")
    style = bar.style()
    style.unpolish(bar)
    style.polish(bar)
    return widget


def make_form(parent: QWidget | None = None) -> QFormLayout:
    """Build a QFormLayout configured to wrap long rows.

    ``WrapLongRows`` keeps labels next to their fields when there's
    horizontal space and stacks the label above the field when the
    panel is narrow. This stops form rows from forcing the panel
    wider than the dock and is what makes the parent QScrollArea's
    ``ScrollBarAlwaysOff`` horizontal policy work in practice.
    """
    form = QFormLayout(parent) if parent is not None else QFormLayout()
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    return form


class CollapsibleSection(QWidget):
    """Section header (clickable) + body widget that hides on collapse.

    Qt has no built-in expander, so this is a small QToolButton + QFrame
    combo. Hosts add controls to ``body_layout``. ``expandedChanged`` lets
    a coordinator (e.g. an accordion group) react when the user opens or
    closes the section.
    """

    expandedChanged = Signal(bool)

    def __init__(self, title: str, *, expanded: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._toggle = QToolButton(self)
        self._toggle.setObjectName("SectionHeader")
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setIconSize(QSize(14, 14))
        # Look comes from the skin's QToolButton#SectionHeader rules
        # (transparent ground, hairline rule, accent on hover). It has to
        # restate background and border explicitly, because qdarkstyle
        # ships 19 QToolButton rules that would otherwise give all 13
        # headers full button chrome.
        self._set_marker(expanded)
        self._toggle.toggled.connect(self._on_toggled)
        outer.addWidget(self._toggle)

        self._body = QFrame(self)
        self._body.setFrameShape(QFrame.Shape.NoFrame)
        self.body_layout = QVBoxLayout(self._body)
        self.body_layout.setContentsMargins(16, 0, 4, 6)
        self.body_layout.setSpacing(4)
        self._body.setVisible(expanded)
        outer.addWidget(self._body)

    def _set_marker(self, expanded: bool) -> None:
        """Point the chevron down (open) or right (closed).

        ``icons.bind`` both sets the glyph and registers the button, so a
        later theme switch repaints it; re-binding on every toggle keeps
        the registered name in step with the state. The import is local
        and guarded so a packaging slip degrades to Qt's built-in
        triangle instead of leaving a blank header.
        """
        name = "chevron-down" if expanded else "chevron-right"
        try:
            from mlgidlab import icons

            icons.bind(self._toggle, name)
            if not self._toggle.icon().isNull():
                self._toggle.setArrowType(Qt.ArrowType.NoArrow)
                return
        except Exception:  # pragma: no cover - defensive
            pass
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )

    def _on_toggled(self, checked: bool) -> None:
        self._apply_state(checked)
        self.expandedChanged.emit(checked)

    def _apply_state(self, expanded: bool) -> None:
        self._body.setVisible(expanded)
        self._set_marker(expanded)


def make_debounced_timer(parent: QWidget, ms: int, slot: Callable[[], None]) -> QTimer:
    """A single-shot timer wired to ``slot`` — the debounce idiom.

    Callers restart it (``timer.start()``) on every input event; the
    slot fires once, ``ms`` after the input settles.
    """
    timer = QTimer(parent)
    timer.setSingleShot(True)
    timer.setInterval(ms)
    timer.timeout.connect(slot)
    return timer


def make_progress_dialog(parent, label, *, title, maximum=0):
    """A window-modal, uncancelable QProgressDialog shown immediately.

    The shape every long blocking operation uses (image import, raw
    conversion, self-update): no cancel button because interrupting the
    underlying operation midway is unsafe (half-written files, a
    half-installed package), minimum duration 0 so it appears at once.
    ``maximum=0`` gives an indeterminate busy bar.
    """
    dlg = QProgressDialog(label, "", 0, maximum, parent)
    dlg.setWindowTitle(title)
    dlg.setWindowModality(Qt.WindowModality.WindowModal)
    dlg.setCancelButton(None)
    dlg.setMinimumDuration(0)
    skin_progress(dlg)
    dlg.show()
    return dlg


def spin_double(lo: float, hi: float, default: float, decimals: int = 3) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setDecimals(decimals)
    s.setRange(lo, hi)
    s.setValue(default)
    return s


def spin_int(default: int, *, lo: int = 0, hi: int = 144) -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(default)
    return s


def row_wrap(layout: QHBoxLayout) -> QWidget:
    w = QWidget()
    w.setLayout(layout)
    return w


def make_pen_swatch(style: dict, width: int = 26, height: int = 12) -> QPixmap:
    """Render a small line preview matching an overlay's pen color/style."""
    pix = QPixmap(width, height)
    pix.fill(Qt.GlobalColor.transparent)
    pen = QPen(QColor(style["color"]), 2)
    pen.setStyle(style["style"])
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(pen)
    painter.drawLine(2, height // 2, width - 2, height // 2)
    painter.end()
    return pix


class ComboWheelBlocker(QObject):
    """Application-wide event filter that swallows wheel events on
    CLOSED comboboxes.

    Qt's default lets the mouse wheel step through a combobox's items
    on hover, so scrolling a dock accidentally changes settings (entry,
    overlay CIF/orientation, display style, …). Only the closed
    combobox is filtered — the popup list is a separate widget and
    keeps normal wheel scrolling.
    """

    def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
        if event.type() == QEvent.Type.Wheel and isinstance(obj, QComboBox):
            return True
        return False
