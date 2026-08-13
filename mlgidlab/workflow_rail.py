"""The stage strip above the image: where this scan stands.

mlgidLAB drives a linear pipeline — convert, detect, fit, match, track —
but those live in three different dock tabs (Pipeline, Conversion, Scan
tracking), so nothing on screen says what order they go in or which of
them has already run. This is that missing spine.

It deliberately holds no pipeline logic. Clicking a stage raises the dock
that owns it; clicking its run glyph clicks that dock's own Run button,
so every ``PipelineCommand`` is still built in exactly one place, with
the kwargs the panel collected. The rail cannot run something the panel
would refuse: the glyph mirrors that button's enabled state.

Counts are for the **current frame**, and the chips say so. Entry-wide
totals would mean reading peak-table shapes across every frame, which on
a 602-frame scan belongs on a worker thread; the frame the user is
looking at is already in memory.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mlgidlab import icons
from mlgidlab.widgets import GAP, PAD

#: (key, label, glyph). Order is the order of the workflow, which is the
#: whole point of the strip.
STAGES = (
    ("convert", "Convert", "dock-conversion"),
    ("detect", "Detect", "dock-pipeline"),
    ("fit", "Fit", "dock-peaks"),
    ("match", "Match", "dock-expected"),
    ("track", "Track", "dock-tracking"),
)


class _StageChip(QWidget):
    """One stage: name, state line, and a run glyph."""

    activated = Signal(str)
    runRequested = Signal(str)

    def __init__(self, key: str, label: str, glyph: str,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)

        self.button = QToolButton(self)
        self.button.setProperty("stage", "chip")
        self.button.setText(label)
        self.button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.clicked.connect(lambda: self.activated.emit(self.key))
        icons.bind(self.button, glyph)
        row.addWidget(self.button)

        self.run_button = QToolButton(self)
        self.run_button.setProperty("stage", "run")
        self.run_button.setToolTip(f"Run {label.lower()}")
        self.run_button.clicked.connect(
            lambda: self.runRequested.emit(self.key))
        icons.bind(self.run_button, "play")
        row.addWidget(self.run_button)

        # The state line sits under the pair, so a chip is two rows tall
        # at most and the strip stays a strip.
        self.state = QLabel("", self)
        self.state.setProperty("status", "muted")

    def set_state(self, text: str, level: str = "muted") -> None:
        self.state.setText(text)
        self.state.setProperty("status", level)
        style = self.state.style()
        style.unpolish(self.state)
        style.polish(self.state)
        self.button.setProperty("state", level)
        style = self.button.style()
        style.unpolish(self.button)
        style.polish(self.button)

    def set_available(self, available: bool, reason: str = "") -> None:
        self.button.setEnabled(available)
        if reason:
            self.button.setToolTip(reason)

    def set_runnable(self, runnable: bool) -> None:
        self.run_button.setEnabled(runnable)


class WorkflowRail(QWidget):
    """The five stages, left to right, above the central view."""

    #: A stage was clicked: bring the dock that owns it forward.
    stageActivated = Signal(str)
    #: A stage's run glyph was clicked.
    stageRunRequested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("mlgid", "rail")
        outer = QHBoxLayout(self)
        outer.setContentsMargins(PAD, 2, PAD, 2)
        outer.setSpacing(GAP)

        self.chips: dict[str, _StageChip] = {}
        for index, (key, label, glyph) in enumerate(STAGES):
            if index:
                arrow = QLabel("→")
                arrow.setProperty("status", "muted")
                outer.addWidget(arrow)
            chip = _StageChip(key, label, glyph, self)
            chip.activated.connect(self.stageActivated)
            chip.runRequested.connect(self.stageRunRequested)
            column = QVBoxLayout()
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(0)
            column.addWidget(chip)
            column.addWidget(chip.state)
            outer.addLayout(column)
            self.chips[key] = chip
        outer.addStretch(1)

        line = QFrame(self)
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Plain)
        line.setFixedHeight(1)

    # -- state ---------------------------------------------------------

    def set_state(self, key: str, text: str, level: str = "muted") -> None:
        chip = self.chips.get(key)
        if chip is not None:
            chip.set_state(text, level)

    def set_available(self, key: str, available: bool, reason: str = "") -> None:
        chip = self.chips.get(key)
        if chip is not None:
            chip.set_available(available, reason)

    def set_runnable(self, key: str, runnable: bool) -> None:
        chip = self.chips.get(key)
        if chip is not None:
            chip.set_runnable(runnable)

    def set_mode(self, kind: str | None) -> None:
        """Adapt to the session kind.

        Raw sessions can only be converted; converted (NeXus) sessions
        are past that stage. Unavailable stages stay visible and greyed
        rather than disappearing — the point of the strip is to show the
        whole workflow, including the parts that do not apply yet.
        """
        is_raw = kind == "raw"
        self.set_available("convert", is_raw,
                           "Raw sessions only — this file is already converted"
                           if not is_raw else "")
        for key in ("detect", "fit", "match", "track"):
            self.set_available(
                key, kind == "nexus",
                "Convert this raw scan first" if is_raw else "")
