"""Pipeline launcher: collapsible Detection / Fitting / Matching sections.

Each section exposes the full kwarg surface of the underlying ``mlgidBASE``
method (see ``mlgidbase/main.py``: ``run_detection``, ``run_fitting``,
``run_matching``). The panel knows nothing about the active session — the
host wires a ``get_active_entry`` callback so the entry-scope dropdowns can
resolve "Active entry" at click time.

Defaults intentionally scope every run to the *active* entry rather than to
all entries (mlgidBASE's own default). The viewer shows one entry at a time
and per-entry runs sidestep failures on incompatible sibling entries — the
user can still pick "All entries" explicitly when they want a sweep.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mlgidlab.pipeline import PipelineCommand, is_mlgidbase_available


# Sentinels for the entry-scope and frame-scope dropdowns. The panel resolves
# them to mlgidBASE-shaped kwargs at click time so the host MainWindow stays
# the single source of truth for "what's active right now".
ENTRY_ACTIVE = "Active entry"
ENTRY_ALL = "All entries"

FRAME_ACTIVE = "Active frame"
FRAME_ALL = "All frames"

# Human-readable titles for the mlgidBASE method names emitted via
# ``PipelineWorker.frameProgress`` — used by the progress label and
# the status-bar tail. Falls through to the raw op name when unmapped.
_OP_TITLES: dict[str, str] = {
    "run_detection": "Detection",
    "run_fitting": "Fitting",
    "run_matching": "Matching",
}


def _make_form(parent: QWidget | None = None) -> QFormLayout:
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


class _CollapsibleSection(QWidget):
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
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        # Section header: no border, bold, full width — matches dark theme.
        self._toggle.setStyleSheet(
            "QToolButton { border: none; padding: 4px 0px; font-weight: bold; }"
        )
        self._toggle.toggled.connect(self._on_toggled)
        outer.addWidget(self._toggle)

        self._body = QFrame(self)
        self._body.setFrameShape(QFrame.Shape.NoFrame)
        self.body_layout = QVBoxLayout(self._body)
        self.body_layout.setContentsMargins(16, 0, 4, 6)
        self.body_layout.setSpacing(4)
        self._body.setVisible(expanded)
        outer.addWidget(self._body)

    def is_expanded(self) -> bool:
        return self._toggle.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        """Open or close without re-emitting (used by accordion peers)."""
        if self._toggle.isChecked() == expanded:
            return
        # blockSignals avoids a re-entrant accordion update — the coordinator
        # already knows it caused this state change.
        self._toggle.blockSignals(True)
        try:
            self._toggle.setChecked(expanded)
        finally:
            self._toggle.blockSignals(False)
        self._apply_state(expanded)

    def _on_toggled(self, checked: bool) -> None:
        self._apply_state(checked)
        self.expandedChanged.emit(checked)

    def _apply_state(self, expanded: bool) -> None:
        self._body.setVisible(expanded)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )


class PipelinePanel(QWidget):
    """Buttons + parameter controls for the three mlgidbase pipeline stages.

    Emits ``runRequested(PipelineCommand)`` when the user clicks a Run
    button; the main window owns the threading and file-handle juggling.
    """

    runRequested = Signal(PipelineCommand)
    # Emitted when the user clicks "Parse CIFs". The host runs a worker
    # thread to do the actual parsing (slow for raw CIFs) and posts the
    # result back via ``set_cif_pattern``.
    parseCifsRequested = Signal(str)
    # Log routing: panels emit messages and the host forwards them to a
    # shared Logs dock. Keeping ``append_log`` / ``clear_log`` as the
    # public surface so existing call sites don't change.
    logMessage = Signal(str)
    logCleared = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._available = is_mlgidbase_available()
        # Resolved by the host so panel stays decoupled from MainWindow.
        # If unset (or returns None) "Active entry/frame" silently falls
        # back to mlgidBASE's own None-default (all entries / all frames).
        self._get_active_entry: Callable[[], str | None] | None = None
        self._get_active_frame: Callable[[], int | None] | None = None
        # CIF cache: ``_cached_cif_input`` is the raw text the user
        # parsed against; ``_cached_cif_obj`` is the resulting CifPattern
        # (or other pre-loaded object). Run Matching forwards the cache
        # when present and the input hasn't changed since the parse.
        self._cached_cif_input: str | None = None
        self._cached_cif_obj: object | None = None
        self._build_ui()

    # -- Public API used by MainWindow --

    def set_active_entry_resolver(
        self, fn: Callable[[], str | None]
    ) -> None:
        self._get_active_entry = fn

    def set_active_frame_resolver(
        self, fn: Callable[[], int | None]
    ) -> None:
        self._get_active_frame = fn

    def append_log(self, msg: str) -> None:
        """Forward ``msg`` to the shared Logs dock via ``logMessage``."""
        self.logMessage.emit(msg)

    def clear_log(self) -> None:
        """Ask the shared Logs dock to wipe its contents."""
        self.logCleared.emit()

    def set_running(self, running: bool) -> None:
        if not self._available:
            return
        self.btn_detect.setEnabled(not running)
        self.btn_fit.setEnabled(not running)
        # Match + run-all both additionally require an active matching
        # source — let _update_match_enabled re-evaluate that gate when
        # a run finishes.
        if running:
            self.btn_match.setEnabled(False)
            self.btn_run_all.setEnabled(False)
        else:
            self._update_match_enabled()
        # Hide both progress rows when a run ends. Either bar will
        # reappear on the next ``on_frame_progress`` /
        # ``on_queue_progress`` call if the upcoming run is
        # multi-frame / multi-entry; single-everything runs leave them
        # hidden. Clear the paint-state memoisation so the next run
        # always repaints (even if its first emit happens to match
        # the last emit of the previous run).
        if not running and hasattr(self, "_progress_bar"):
            self._progress_bar.hide()
            self._progress_label.hide()
            self._frame_progress_last_state = None
            self._frame_progress_last_visible = False
        if not running and hasattr(self, "_entry_progress_bar"):
            self._entry_progress_bar.hide()
            self._entry_progress_label.hide()
            self._entry_progress_last_state = None
            self._entry_progress_last_visible = False

    def on_queue_progress(
        self, current: int, total: int, entry: str, op_name: str
    ) -> None:
        """Update the entry-level progress row.

        Driven by the host whenever a queued "All entries" run starts a
        new entry. ``current`` is 1-indexed (the entry just starting),
        ``total`` is the number of entries in the run. Hidden when
        ``total <= 1`` so single-entry runs do not get a meaningless
        1/1 bar. Label reads e.g. ``Detection · entry 3/8 · entry_DBTTF``.

        Memoises the last paint state and skips redundant widget
        writes so back-to-back identical updates do not trigger paint
        events.
        """
        if not hasattr(self, "_entry_progress_bar"):
            return
        if total <= 1:
            if getattr(self, "_entry_progress_last_visible", True):
                self._entry_progress_bar.hide()
                self._entry_progress_label.hide()
                self._entry_progress_last_visible = False
            return
        new_state = (int(current), int(total), str(entry), str(op_name))
        if getattr(self, "_entry_progress_last_state", None) == new_state:
            return
        self._entry_progress_last_state = new_state
        title = _OP_TITLES.get(op_name, op_name)
        entry_part = f" · {entry}" if entry else ""
        # setRange is cheap when the range is unchanged but still
        # forwards through QProgressBar; only call it when total moves.
        if self._entry_progress_bar.maximum() != total:
            self._entry_progress_bar.setRange(0, total)
        self._entry_progress_bar.setValue(current)
        self._entry_progress_label.setText(
            f"{title} · entry {current}/{total}{entry_part}"
        )
        if not getattr(self, "_entry_progress_last_visible", False):
            self._entry_progress_label.show()
            self._entry_progress_bar.show()
            self._entry_progress_last_visible = True

    def on_frame_progress(
        self, done: int, total: int, op_name: str, entry: str
    ) -> None:
        """Update the multi-frame progress row.

        Hides both widgets when ``total <= 1`` so single-frame runs
        contribute no UI noise. Otherwise paints a determinate bar
        and a one-line label, e.g. ``Detection · entry_4P · 3/12 frames``.
        ``op_name`` is the mlgidBASE method name (``run_detection`` /
        ``run_fitting`` / ``run_matching``) which we map to a human
        title; ``entry`` may be the literal entry name or a sentinel
        like ``"all entries"`` when the run iterates the whole file.

        Memoises the last paint state and skips redundant widget
        writes so a fast pipeline emitting many frame events per
        second does not turn into a paint storm.
        """
        if not hasattr(self, "_progress_bar"):
            return
        if op_name == "track_peaks":
            # track_peaks is one opaque mlgidBASE call dominated by the
            # file open and the all-against-all IoU/graph step, not the
            # (fast) per-frame reads — a determinate "N/M frames" bar
            # just sits at 0 then flickers to full. Show a busy
            # indicator (indeterminate marquee) for the whole run
            # instead; set it up once (later ticks carry no new info).
            if getattr(self, "_frame_progress_last_state", None) != ("track_peaks",):
                self._frame_progress_last_state = ("track_peaks",)
                self._progress_bar.setRange(0, 0)   # 0..0 == busy marquee
                self._progress_label.setText("Tracking peaks…")
                self._progress_label.show()
                self._progress_bar.show()
                self._frame_progress_last_visible = True
            return
        if total <= 1:
            if getattr(self, "_frame_progress_last_visible", True):
                self._progress_bar.hide()
                self._progress_label.hide()
                self._frame_progress_last_visible = False
            return
        new_state = (int(done), int(total), str(op_name), str(entry))
        if getattr(self, "_frame_progress_last_state", None) == new_state:
            return
        self._frame_progress_last_state = new_state
        title = _OP_TITLES.get(op_name, op_name)
        entry_part = f" · {entry}" if entry else ""
        if self._progress_bar.maximum() != total:
            self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(done)
        self._progress_label.setText(f"{title}{entry_part} · {done}/{total} frames")
        if not getattr(self, "_frame_progress_last_visible", False):
            self._progress_label.show()
            self._progress_bar.show()
            self._frame_progress_last_visible = True

    # -- UI construction --

    def _build_ui(self) -> None:
        # Outer layout hosts only the scroll area — content widget owns
        # all the actual section margins so scrollbars sit flush against
        # the dock edge.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        # Vertical scroll fires only when content overflows; horizontal
        # is hard-locked off so a narrow dock collapses form rows
        # (labels wrap above fields, see ``_make_form``) instead of
        # introducing an x-axis scrollbar.
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        content = QWidget()
        inner = QVBoxLayout(content)
        inner.setContentsMargins(8, 8, 8, 8)
        inner.setSpacing(4)

        # Progress rows — two stacked levels.
        #   * Entry row: only visible during an "All entries" run, shows
        #     "entry K/N · <entry>" and a determinate bar over the queue.
        #   * Frame row: visible during any multi-frame run, shows
        #     "<Op> · <entry> · K/N frames" and a bar over the entry's
        #     frames.
        # Both rows hide on idle and on single-{frame,entry} runs so the
        # progress region collapses to zero pixels in the steady state.
        self._entry_progress_label = QLabel("")
        self._entry_progress_label.setStyleSheet("color: #aaa;")
        self._entry_progress_label.hide()
        self._entry_progress_bar = QProgressBar()
        self._entry_progress_bar.setRange(0, 1)
        self._entry_progress_bar.setValue(0)
        self._entry_progress_bar.setTextVisible(False)
        self._entry_progress_bar.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._entry_progress_bar.hide()
        inner.addWidget(self._entry_progress_label)
        inner.addWidget(self._entry_progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet("color: #aaa;")
        self._progress_label.hide()
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._progress_bar.hide()
        inner.addWidget(self._progress_label)
        inner.addWidget(self._progress_bar)

        if not self._available:
            hint = QLabel(
                "<b>mlgidbase</b> is not installed in this environment.<br><br>"
                "Install it to enable detection, fitting, and matching:"
                "<pre>  pip install mlgidbase</pre>"
            )
            hint.setWordWrap(True)
            inner.addWidget(hint)
            inner.addStretch(1)
            scroll.setWidget(content)
            return

        # Run-all button — created up-front because the matching
        # section's initial _update_match_enabled call (during section
        # construction) reaches for self.btn_run_all to set its gated
        # state. The widget is added to the layout further down so it
        # sits pinned to the bottom of the dock.
        self.btn_run_all = QPushButton("Run full pipeline")
        self.btn_run_all.setToolTip(
            "Run Detection, Fitting, and Matching back-to-back using the "
            "current section kwargs.\n\n"
            "Disabled until the active matching source (CIF or pickle) "
            "has a value."
        )
        self.btn_run_all.setEnabled(False)
        self.btn_run_all.clicked.connect(self._on_run_all)

        # Sections are independent now: any combination of them can be
        # open at once. Detection starts open so the user has something
        # actionable on first sight.
        self._sections: list[_CollapsibleSection] = [
            self._build_detection_section(),
            self._build_fitting_section(),
            self._build_matching_section(),
        ]
        for s in self._sections:
            inner.addWidget(s)

        # Logs live in their own dock now (see MainWindow._logs_dock); the
        # trailing stretch keeps the sections at the top of the scroll
        # area when they don't fill the visible height. The run-all
        # button sits below the stretch so it stays pinned to the
        # bottom regardless of how many sections are expanded.
        inner.addStretch(1)
        inner.addWidget(self.btn_run_all)
        scroll.setWidget(content)

    def _build_detection_section(self) -> QWidget:
        section = _CollapsibleSection("Detection", expanded=True)
        form = _make_form()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)

        # Entry scope — "Active entry" is the default to keep runs aligned
        # with what's on screen and avoid surprises on multi-entry files.
        self.det_entry_scope = QComboBox()
        self.det_entry_scope.addItems([ENTRY_ACTIVE, ENTRY_ALL])
        form.addRow("Entry:", self.det_entry_scope)

        self.det_frame_scope = QComboBox()
        self.det_frame_scope.addItems([FRAME_ALL, FRAME_ACTIVE])
        form.addRow("Frames:", self.det_frame_scope)

        # YAML config picker — passed straight through to mlgidBASE's
        # ``config_detect`` argument when non-empty.
        self.det_config_path = QLineEdit()
        self.det_config_path.setPlaceholderText("(default config)")
        self.det_config_path.setToolTip(
            "Optional YAML config file passed to mlgidDETECT as "
            "config_detect. Leave blank to use the built-in defaults."
        )
        det_browse = QPushButton("Browse…")
        det_browse.clicked.connect(self._browse_detect_config)
        det_clear = QPushButton("Clear")
        det_clear.clicked.connect(lambda: self.det_config_path.setText(""))
        det_cfg_row = QWidget()
        det_cfg_h = QHBoxLayout(det_cfg_row)
        det_cfg_h.setContentsMargins(0, 0, 0, 0)
        det_cfg_h.setSpacing(4)
        det_cfg_h.addWidget(self.det_config_path, 1)
        det_cfg_h.addWidget(det_browse)
        det_cfg_h.addWidget(det_clear)
        form.addRow("Config (yaml):", det_cfg_row)

        # Model type — empty string means "use mlgidbase default".
        self.det_model_type = QComboBox()
        self.det_model_type.addItems(["(default)", "faster_rcnn", "dino"])
        self.det_model_type.setToolTip(
            "Detection model architecture. Leave on (default) unless your "
            "config selects a different backbone."
        )
        form.addRow("Model:", self.det_model_type)

        section.body_layout.addLayout(form)
        self.btn_detect = QPushButton("Run detection")
        self.btn_detect.clicked.connect(self._on_run_detection)
        section.body_layout.addWidget(self.btn_detect)
        return section

    def _build_fitting_section(self) -> QWidget:
        section = _CollapsibleSection("Fitting", expanded=False)
        form = _make_form()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)

        self.fit_entry_scope = QComboBox()
        self.fit_entry_scope.addItems([ENTRY_ACTIVE, ENTRY_ALL])
        form.addRow("Entry:", self.fit_entry_scope)

        self.fit_frame_scope = QComboBox()
        self.fit_frame_scope.addItems([FRAME_ALL, FRAME_ACTIVE])
        form.addRow("Frames:", self.fit_frame_scope)

        # Match mlgidbase defaults exactly so an unedited form yields the
        # same behaviour as a bare ``analysis.run_fitting()`` call.
        self.fit_crit_angle = QDoubleSpinBox()
        self.fit_crit_angle.setDecimals(3)
        self.fit_crit_angle.setRange(0.0, 90.0)
        self.fit_crit_angle.setSingleStep(0.5)
        self.fit_crit_angle.setValue(0.0)
        self.fit_crit_angle.setSuffix(" °")
        self.fit_crit_angle.setToolTip(
            "Maximum allowed misorientation angle between peaks within a cluster."
        )
        form.addRow("Critical angle:", self.fit_crit_angle)

        self.fit_dist_peaks = QDoubleSpinBox()
        self.fit_dist_peaks.setDecimals(2)
        self.fit_dist_peaks.setRange(0.0, 1000.0)
        self.fit_dist_peaks.setSingleStep(1.0)
        self.fit_dist_peaks.setValue(10.0)
        self.fit_dist_peaks.setToolTip(
            "Distance threshold for peak clustering (px in detector frame)."
        )
        form.addRow("Cluster dist (peaks):", self.fit_dist_peaks)

        self.fit_dist_rings = QDoubleSpinBox()
        self.fit_dist_rings.setDecimals(2)
        self.fit_dist_rings.setRange(0.0, 1000.0)
        self.fit_dist_rings.setSingleStep(1.0)
        self.fit_dist_rings.setValue(10.0)
        self.fit_dist_rings.setToolTip("Distance threshold for ring clustering.")
        form.addRow("Cluster dist (rings):", self.fit_dist_rings)

        self.fit_cluster_extend = QSpinBox()
        self.fit_cluster_extend.setRange(0, 100)
        self.fit_cluster_extend.setValue(2)
        self.fit_cluster_extend.setToolTip(
            "Number of neighboring peaks to include in cluster expansion."
        )
        form.addRow("Cluster extend:", self.fit_cluster_extend)

        self.fit_theta_fixed = QCheckBox()
        self.fit_theta_fixed.setChecked(True)
        self.fit_theta_fixed.setToolTip(
            "Hold theta fixed during clustering. Default in mlgidBASE."
        )
        form.addRow("Theta fixed:", self.fit_theta_fixed)

        self.fit_use_pool = QCheckBox()
        self.fit_use_pool.setChecked(False)
        self.fit_use_pool.setToolTip(
            "Use multiprocessing for fitting (faster on large stacks, "
            "but interleaves logs unpredictably)."
        )
        form.addRow("Use pool:", self.fit_use_pool)

        self.fit_debug = QCheckBox()
        self.fit_debug.setChecked(False)
        form.addRow("Debug:", self.fit_debug)

        section.body_layout.addLayout(form)
        self.btn_fit = QPushButton("Run fitting")
        self.btn_fit.clicked.connect(self._on_run_fitting)
        section.body_layout.addWidget(self.btn_fit)
        return section

    def _build_matching_section(self) -> QWidget:
        section = _CollapsibleSection("Matching", expanded=False)
        form = _make_form()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)

        self.match_entry_scope = QComboBox()
        self.match_entry_scope.addItems([ENTRY_ACTIVE, ENTRY_ALL])
        form.addRow("Entry:", self.match_entry_scope)

        self.match_frame_scope = QComboBox()
        self.match_frame_scope.addItems([FRAME_ALL, FRAME_ACTIVE])
        form.addRow("Frames:", self.match_frame_scope)

        # Mutually-exclusive radio buttons in a single QButtonGroup —
        # one sits at the start of each input row, so the user picks
        # the active source by ticking the radio next to the field
        # they want to use. The unticked row's widgets are greyed out
        # to remove any ambiguity about which value Run-Matching
        # consumes.
        self.rb_cif_source = QRadioButton()
        self.rb_pickle_source = QRadioButton()
        self.rb_cif_source.setToolTip(
            "Use the raw CIF input below as the matching source."
        )
        self.rb_pickle_source.setToolTip(
            "Use the pickle file below as the matching source."
        )
        self._source_group = QButtonGroup(self)
        self._source_group.setExclusive(True)
        self._source_group.addButton(self.rb_cif_source)
        self._source_group.addButton(self.rb_pickle_source)
        self.rb_cif_source.setChecked(True)
        # Either toggled signal fires for both buttons in an exclusive
        # group, so we get one update per user click. ``buttonToggled``
        # gives us the QAbstractButton + new state.
        self._source_group.buttonToggled.connect(
            lambda *_: self._on_match_source_changed()
        )

        self.cif_path = QLineEdit()
        self.cif_path.setPlaceholderText("Raw .cif file(s) or folder…")
        self.cif_path.setToolTip(
            "Raw CIF input forwarded as ``cif_prepr`` to "
            "mlgidBASE.run_matching. Accepts:\n"
            "  • one or more raw .cif files (semicolon-separated)\n"
            "  • a folder containing .cif files\n\n"
            "ExpParameters are auto-derived from the active NeXus file's "
            "instrument metadata."
        )
        self._cif_browse_btn = QPushButton("Browse…")
        self._cif_browse_btn.clicked.connect(self._browse_cif)
        self._cif_browse_dir_btn = QPushButton("Folder…")
        self._cif_browse_dir_btn.setToolTip(
            "Pick a directory; every .cif inside is used."
        )
        self._cif_browse_dir_btn.clicked.connect(self._browse_cif_dir)
        cif_row = QWidget()
        cif_h = QHBoxLayout(cif_row)
        cif_h.setContentsMargins(0, 0, 0, 0)
        cif_h.addWidget(self.rb_cif_source)
        cif_h.addWidget(self.cif_path, 1)
        cif_h.addWidget(self._cif_browse_btn)
        cif_h.addWidget(self._cif_browse_dir_btn)
        form.addRow("CIF input:", cif_row)

        # Parse button + cache-state label. CIF preprocessing simulates
        # every input pattern, which is slow — caching the resulting
        # CifPattern across multiple Run-Matching clicks is a huge time
        # saver. The cache is invalidated whenever the input text changes.
        self.btn_parse_cifs = QPushButton("Parse CIFs")
        self.btn_parse_cifs.setToolTip(
            "Pre-load the CIF input into a CifPattern that subsequent "
            "matching runs reuse. Without this, every Run Matching "
            "re-parses the CIFs from scratch.\n\n"
            "Applies only to the CIF input above; pickles are forwarded "
            "directly without preprocessing."
        )
        self.btn_parse_cifs.clicked.connect(self._on_parse_cifs)
        self.btn_parse_cifs.setEnabled(False)
        # Compact busy bar shown while the parse worker runs. CifPattern
        # simulates every CIF in one opaque call (no per-CIF callback to
        # hook), so the bar is indeterminate — but it moves, which the
        # static "Parsing…" text alone does not.
        self.cif_parse_bar = QProgressBar()
        self.cif_parse_bar.setRange(0, 0)
        self.cif_parse_bar.setTextVisible(False)
        self.cif_parse_bar.setFixedWidth(110)
        self.cif_parse_bar.setFixedHeight(14)
        self.cif_parse_bar.hide()
        self.cif_cache_label = QLabel("Not parsed")
        self.cif_cache_label.setStyleSheet("color: #aaa; font-style: italic;")
        cif_parse_row = QWidget()
        cif_parse_h = QHBoxLayout(cif_parse_row)
        cif_parse_h.setContentsMargins(0, 0, 0, 0)
        cif_parse_h.addWidget(self.btn_parse_cifs)
        cif_parse_h.addWidget(self.cif_parse_bar)
        cif_parse_h.addWidget(self.cif_cache_label, 1)
        form.addRow("", cif_parse_row)
        # Any edit invalidates the cache and re-enables the button.
        self.cif_path.textChanged.connect(self._on_cif_input_changed)

        # Separate pickle input — preprocessed CifPattern files skip the
        # raw-CIF preprocessing step entirely; mlgidBASE.run_matching
        # accepts a path-to-pickle string verbatim. Pickle path takes
        # priority over the CIF input above when both are set.
        self.pickle_path = QLineEdit()
        self.pickle_path.setPlaceholderText(
            "Preprocessed CIF pickle (.pickle / .pkl)…"
        )
        self.pickle_path.setToolTip(
            "Path to a preprocessed CifPattern pickle. Forwarded as "
            "``cif_prepr`` to mlgidBASE.run_matching when ``Source`` "
            "above is set to ``Pickle file``."
        )
        self._pickle_browse_btn = QPushButton("Browse…")
        self._pickle_browse_btn.clicked.connect(self._browse_pickle)
        self._pickle_clear_btn = QPushButton("Clear")
        self._pickle_clear_btn.clicked.connect(lambda: self.pickle_path.setText(""))
        pickle_row = QWidget()
        pickle_h = QHBoxLayout(pickle_row)
        pickle_h.setContentsMargins(0, 0, 0, 0)
        pickle_h.addWidget(self.rb_pickle_source)
        pickle_h.addWidget(self.pickle_path, 1)
        pickle_h.addWidget(self._pickle_browse_btn)
        pickle_h.addWidget(self._pickle_clear_btn)
        form.addRow("Pickle input:", pickle_row)

        self.peaks_type = QComboBox()
        self.peaks_type.addItems(["segments", "rings"])
        form.addRow("Peaks type:", self.peaks_type)

        # Probability threshold (mlgidBASE alias: ``threshold``). The two
        # are equivalent on the mlgidBASE side; we send ``threshold`` since
        # that's what the underlying _run_matching prefers.
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setSingleStep(0.05)
        self.threshold.setDecimals(2)
        self.threshold.setValue(0.5)
        self.threshold.setToolTip(
            "Minimum probability for a CIF candidate to be accepted."
        )
        form.addRow("Probability threshold:", self.threshold)

        self.intensity_threshold = QDoubleSpinBox()
        self.intensity_threshold.setRange(0.0, 1e9)
        self.intensity_threshold.setSingleStep(1.0)
        self.intensity_threshold.setDecimals(3)
        self.intensity_threshold.setValue(0.0)
        self.intensity_threshold.setToolTip(
            "Minimum peak intensity to consider during matching."
        )
        form.addRow("Intensity threshold:", self.intensity_threshold)

        self.device = QComboBox()
        self.device.addItems(["cpu", "cuda"])
        form.addRow("Device:", self.device)

        section.body_layout.addLayout(form)

        self.btn_match = QPushButton("Run matching")
        self.btn_match.setEnabled(False)
        self.btn_match.clicked.connect(self._on_run_matching)
        # Gate run button on the active source having text — matching
        # can't proceed without an input value.
        self.cif_path.textChanged.connect(self._update_match_enabled)
        self.pickle_path.textChanged.connect(self._update_match_enabled)
        section.body_layout.addWidget(self.btn_match)
        # Apply the initial enabled / disabled state now that all
        # widgets exist; running the slot earlier would crash because
        # btn_match isn't built until this line.
        self._on_match_source_changed()
        return section

    def _update_match_enabled(self) -> None:
        # Run-Match and Run-Full-Pipeline are both gated on the *selected*
        # source having a value; text in the inactive row is irrelevant.
        # Run-Full-Pipeline shares the gate because Matching is the last
        # stage of the chain and it can't proceed without a source.
        if self._use_pickle_source():
            has_input = bool(self.pickle_path.text().strip())
        else:
            has_input = bool(self.cif_path.text().strip())
        self.btn_match.setEnabled(has_input)
        self.btn_run_all.setEnabled(has_input)

    def _use_pickle_source(self) -> bool:
        return self.rb_pickle_source.isChecked()

    def _on_match_source_changed(self, *_args) -> None:
        """Enable only the input row matching the active source.

        Greying the unused row out (instead of hiding it) keeps the
        layout stable and lets the user see what's typed in the
        inactive field — useful when alternating between sources.
        Run-Matching gating is recomputed because the enabled-text
        check now points at the other field.
        """
        use_pickle = self._use_pickle_source()
        # CIF inputs + parse machinery are CIF-only.
        for w in (
            self.cif_path,
            self._cif_browse_btn,
            self._cif_browse_dir_btn,
            self.btn_parse_cifs,
            self.cif_cache_label,
        ):
            w.setEnabled(not use_pickle)
        # Re-apply the parse button's text-based gating after the
        # source flip so it doesn't sit enabled on an empty CIF field.
        if not use_pickle:
            self.btn_parse_cifs.setEnabled(bool(self.cif_path.text().strip()))
        # Pickle widgets.
        for w in (
            self.pickle_path,
            self._pickle_browse_btn,
            self._pickle_clear_btn,
        ):
            w.setEnabled(use_pickle)
        self._update_match_enabled()

    # -- Click handlers --

    def _on_run_detection(self) -> None:
        kwargs: dict = {}
        self._inject_entry_scope(self.det_entry_scope, kwargs)
        self._inject_frame_scope(self.det_frame_scope, kwargs)
        cfg = self.det_config_path.text().strip()
        if cfg:
            kwargs["config_detect"] = cfg
        model = self.det_model_type.currentText()
        if model and not model.startswith("("):
            kwargs["model_type"] = model
        self.runRequested.emit(PipelineCommand("run_detection", kwargs))

    def _on_run_fitting(self) -> None:
        kwargs: dict = {}
        self._inject_entry_scope(self.fit_entry_scope, kwargs)
        self._inject_frame_scope(self.fit_frame_scope, kwargs)
        kwargs["crit_angle"] = float(self.fit_crit_angle.value())
        kwargs["clustering_distance_peaks"] = float(self.fit_dist_peaks.value())
        kwargs["clustering_distance_rings"] = float(self.fit_dist_rings.value())
        kwargs["clustering_extend"] = int(self.fit_cluster_extend.value())
        kwargs["theta_fixed"] = bool(self.fit_theta_fixed.isChecked())
        kwargs["use_pool"] = bool(self.fit_use_pool.isChecked())
        kwargs["debug"] = bool(self.fit_debug.isChecked())
        self.runRequested.emit(PipelineCommand("run_fitting", kwargs))

    def _on_run_all(self) -> None:
        """Chain Detection → Fitting → Matching as three queued commands.

        Each stage reuses its per-stage handler so kwarg-building stays
        in one place. The host's runRequested dispatcher queues each
        emitted command onto the same worker thread, so the three
        stages run sequentially without overlapping. "All entries"
        scope is still expanded per-entry inside each stage.

        Errors in earlier stages don't abort the chain — the existing
        queue logs and continues, matching the per-entry batch behaviour.
        """
        # Belt-and-braces — the button is gated, but a programmatic
        # caller could still reach this with no source set.
        if self._use_pickle_source():
            if not self.pickle_path.text().strip():
                return
        else:
            if not self.cif_path.text().strip():
                return
        self._on_run_detection()
        self._on_run_fitting()
        self._on_run_matching()

    def matching_kwargs(self) -> dict | None:
        """Base ``run_matching`` kwargs (no entry/frame scope) from the
        panel's current CIF source + parameters, or ``None`` when no
        CIF source is configured.

        The Source selector decides which input is used; the inactive
        row is greyed out and its content ignored. mlgidBASE.run_matching's
        ``load_cif_prepr`` accepts a path-to-pickle string verbatim, so
        the pickle path is forwarded untouched. For raw CIFs we reuse
        the cached CifPattern when the input hasn't changed since the
        last parse, otherwise we send the string and let the worker
        build the pattern. Public so the host can chain a matching pass
        (e.g. after Interpolate-track's gap fills) with the SAME
        settings the panel's own Run matching would use.
        """
        if self._use_pickle_source():
            pkl = self.pickle_path.text().strip()
            if not pkl:
                return None
            cif_value: object = pkl
        else:
            cif = self.cif_path.text().strip()
            if not cif:
                return None
            if self._cached_cif_obj is not None and self._cached_cif_input == cif:
                cif_value = self._cached_cif_obj
            else:
                cif_value = cif
        return {
            "cif_prepr": cif_value,
            "peaks_type": self.peaks_type.currentText(),
            "threshold": float(self.threshold.value()),
            "intensity_threshold": float(self.intensity_threshold.value()),
            "device": self.device.currentText(),
        }

    def cif_source_text(self) -> str:
        """The RAW .cif source text (folder / file / ``;``-list) when
        the CIF input row is active — ``""`` for the pickle source or
        an empty field. Lets the host build CIF-restricted matching
        commands (Interpolate track) even when ``matching_kwargs()``
        returns the cached pre-parsed ``CifPattern`` object, which
        cannot be subset."""
        if self._use_pickle_source():
            return ""
        return self.cif_path.text().strip()

    def _on_run_matching(self) -> None:
        kwargs = self.matching_kwargs()
        if kwargs is None:
            return
        self._inject_entry_scope(self.match_entry_scope, kwargs)
        self._inject_frame_scope(self.match_frame_scope, kwargs)
        self.runRequested.emit(PipelineCommand("run_matching", kwargs))

    def _on_parse_cifs(self) -> None:
        text = self.cif_path.text().strip()
        if not text:
            return
        # Disable the button + show in-progress text. The host runs the
        # parse on a worker thread and posts the result back via
        # set_cif_pattern.
        self.btn_parse_cifs.setEnabled(False)
        self.btn_parse_cifs.setText("Parsing…")
        self.cif_parse_bar.show()
        self.cif_cache_label.setText("Parsing…")
        self.cif_cache_label.setStyleSheet("color: #ffeb3b; font-style: italic;")
        self.parseCifsRequested.emit(text)

    def _on_cif_input_changed(self, text: str) -> None:
        """Invalidate the cache + re-enable the parse button on edit."""
        text = text.strip()
        # Only invalidate if the text actually differs from the cache —
        # programmatic Browse → setText that exactly matches the cached
        # input shouldn't blow it away.
        if self._cached_cif_input is not None and text != self._cached_cif_input:
            self._cached_cif_obj = None
            self._cached_cif_input = None
            self.cif_cache_label.setText("Input changed; re-parse")
            self.cif_cache_label.setStyleSheet(
                "color: #ff6b6b; font-style: italic;"
            )
        elif self._cached_cif_input is None:
            self.cif_cache_label.setText("Not parsed")
            self.cif_cache_label.setStyleSheet(
                "color: #aaa; font-style: italic;"
            )
        self.btn_parse_cifs.setText("Parse CIFs")
        self.btn_parse_cifs.setEnabled(bool(text))

    def clear_cif_cache(self) -> None:
        """Forget any cached CifPattern.

        Used by the host on session swap because ``ExpParameters`` are
        derived from the active NeXus file's instrument metadata — a
        cache built against file A's params can't be safely reused when
        running matching on file B.
        """
        if self._cached_cif_obj is None and self._cached_cif_input is None:
            return
        self._cached_cif_obj = None
        self._cached_cif_input = None
        self.cif_cache_label.setText("Not parsed (active file changed)")
        self.cif_cache_label.setStyleSheet("color: #aaa; font-style: italic;")
        self.btn_parse_cifs.setText("Parse CIFs")
        self.btn_parse_cifs.setEnabled(bool(self.cif_path.text().strip()))

    def set_cif_pattern(self, obj: object | None, error: Exception | None) -> None:
        """Host posts the parse result here. None+exception → error state.

        On success, ``obj`` is cached against the current input text and
        every subsequent Run Matching reuses it.
        """
        self.btn_parse_cifs.setText("Parse CIFs")
        self.btn_parse_cifs.setEnabled(True)
        self.cif_parse_bar.hide()
        if error is not None or obj is None:
            self._cached_cif_obj = None
            self._cached_cif_input = None
            msg = str(error) if error is not None else "(empty result)"
            # Keep the user's input alone so they can retry from the same
            # text; just flag the cache as failed.
            self.cif_cache_label.setText(f"Parse failed — {msg[:60]}")
            self.cif_cache_label.setStyleSheet(
                "color: #ff6b6b; font-style: italic;"
            )
            return
        self._cached_cif_input = self.cif_path.text().strip()
        self._cached_cif_obj = obj
        # Surface the CIF count + the kind of input so the user can see
        # what's been cached (CifPattern exposes ``cifs``).
        n = len(getattr(obj, "cifs", []) or [])
        self.cif_cache_label.setText(f"Parsed: {n} CIF(s) cached")
        self.cif_cache_label.setStyleSheet(
            "color: #4ade80; font-style: italic;"
        )

    # -- Internals --

    def _inject_entry_scope(self, combo: QComboBox, kwargs: dict) -> None:
        """Translate the entry-scope dropdown into mlgidBASE's ``entry`` kwarg.

        - ``ENTRY_ACTIVE``: insert ``entry=<active>`` (skip if no resolver
          or no active entry — mlgidBASE will then iterate all).
        - ``ENTRY_ALL``: leave ``entry`` out of kwargs so mlgidBASE
          defaults to all entries (the host's queue dispatcher then
          expands this into per-entry runs).
        - any other value: a literal entry name appended after
          ACTIVE/ALL, used when the user explicitly picks one entry.
        """
        choice = combo.currentText()
        if choice == ENTRY_ALL or not choice:
            return
        if choice == ENTRY_ACTIVE:
            if self._get_active_entry is None:
                return
            active = self._get_active_entry()
            if active:
                kwargs["entry"] = active
            return
        # Literal entry name — sent verbatim.
        kwargs["entry"] = choice

    def set_available_entries(self, entries: list[str]) -> None:
        """Refresh the per-entry options in all three entry-scope combos.

        Called by the host whenever the active session changes (or after
        a pipeline op that might have added entries — though no current
        op does). Rebuilds the items as
        ``[ACTIVE, ALL, entry_0000, entry_0001, …]`` while preserving
        the user's prior selection if it's still valid.
        """
        if not self._available:
            return
        for combo in (
            self.det_entry_scope,
            self.fit_entry_scope,
            self.match_entry_scope,
        ):
            previous = combo.currentText()
            combo.blockSignals(True)
            try:
                combo.clear()
                combo.addItems([ENTRY_ACTIVE, ENTRY_ALL, *entries])
                # Restore the prior selection when still in the new list,
                # otherwise default back to "Active entry".
                idx = combo.findText(previous)
                combo.setCurrentIndex(idx if idx >= 0 else 0)
            finally:
                combo.blockSignals(False)

    def _inject_frame_scope(self, combo: QComboBox, kwargs: dict) -> None:
        if combo.currentText() != FRAME_ACTIVE:
            return
        if self._get_active_frame is None:
            return
        active = self._get_active_frame()
        if active is not None:
            kwargs["frame_num"] = int(active)

    def _browse_detect_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select detection config (YAML)",
            "",
            "YAML (*.yaml *.yml);;All files (*)",
        )
        if path:
            self.det_config_path.setText(path)

    def _browse_cif(self) -> None:
        """Pick one-or-more raw .cif files.

        Multi-select joins paths with ``;``; ``pipeline.execute`` splits
        this back out and wraps the CIFs in a CifPattern at run time.
        Pickle input has its own dedicated picker — see ``_browse_pickle``.
        """
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select CIF file(s)",
            "",
            "CIF files (*.cif);;All files (*)",
        )
        if not paths:
            return
        self.cif_path.setText(";".join(paths))

    def _browse_cif_dir(self) -> None:
        """Pick a directory of .cif files. Forwarded as a single path."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select folder with CIF files", ""
        )
        if directory:
            self.cif_path.setText(directory)

    def _browse_pickle(self) -> None:
        """Pick a single preprocessed CifPattern pickle file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select preprocessed CIF pickle",
            "",
            "Pickle (*.pickle *.pkl);;All files (*)",
        )
        if path:
            self.pickle_path.setText(path)
