"""Construction of the central widget, the eight dock panels with all cross-object signal wiring, the ghost-tab-bar cleanup, the window title and the status bar.

Plain mixin over ``MainWindow``: no __init__, no Signals; all state
lives on the combined class. Split out of ``main_window`` in the 2026
source split.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction, QFont, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from mlgidlab.browser_widgets import _MlgidHdf5TreeView
from mlgidlab.conversion_panel import ConversionPanel
from mlgidlab.image_viewer import (
    GIWAXSImageViewer,
    OVERLAY_STYLE,
    SIM_MISSED_COLOR,
)
from mlgidlab.main_window_constants import (
    APP_NAME,
    DEFAULT_PLAYBACK_FRAME_MS,
)
from mlgidlab.parameter_panel import ParameterPanel
from mlgidlab.peaks_table_panel import PeaksTablePanel
from mlgidlab.pipeline_panel import PipelinePanel
from mlgidlab.profile_viewer import ProfileViewer
from mlgidlab.scan_tracking_panel import ScanTrackingPanel
from mlgidlab.update_ui import _UpdateBanner
from mlgidlab.widgets import (
    PRIMARY,
    make_debounced_timer,
    make_pen_swatch as _make_pen_swatch,
    set_variant,
    skin_progress,
)
from silx.gui.data.DataViewerFrame import DataViewerFrame
from mlgidlab import icons

import logging

logger = logging.getLogger(__name__)


class BuildMixin:
    def _build_central(self) -> None:
        self.viewer = GIWAXSImageViewer(self)
        self.data_viewer = DataViewerFrame(self)

        self.tabs = QTabWidget(self)
        # documentMode flattens the tab-pane border so the image fills
        # the full tab area without the small inset that lets pyqtgraph
        # show a few pixels of scrollable margin.
        self.tabs.setDocumentMode(True)
        # documentMode is partial under qdarkstyle — the pane keeps a
        # small border + padding from the dark stylesheet which traps
        # ~2 px of overflow from the central widget. Override via an
        # explicit zero-pad stylesheet so the viewer fills flush.
        self.tabs.setStyleSheet(
            "QTabWidget::pane { border: 0px; padding: 0px; margin: 0px; }"
        )
        self.tabs.addTab(self.viewer, "Image")
        self.tabs.addTab(self.data_viewer, "Data")
        # Render a deferred (clicked-while-hidden) tree node when the user
        # switches to the Data tab — see ``_set_or_defer_data_node``.
        self.tabs.currentChanged.connect(self._on_main_tab_changed)

        # Central column: a hidden "update available" banner above the tabs
        # (shown by the startup update check — see _on_update_check_finished).
        central = QWidget(self)
        col = QVBoxLayout(central)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        self._update_banner = _UpdateBanner(central)
        self._update_banner.hide()
        self._update_banner.installRequested.connect(
            self._on_install_update_requested
        )
        col.addWidget(self._update_banner)
        col.addWidget(self.tabs)
        self.setCentralWidget(central)

    def _build_docks(self) -> None:
        # Make the side docks own the bottom corners so the bottom Profile
        # dock stays aligned with the central image/data tabs.
        self.setCorner(
            Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.setCorner(
            Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea
        )

        # Left: HDF5 tree (silx) — subclass swaps the root icon for raw
        # sessions so NeXus and raw files are distinguishable at a glance.
        self.tree = _MlgidHdf5TreeView(self)
        # Sorting is DISABLED on purpose. silx's NexusSortFilterProxyModel
        # sorts the NAME column by resolving each node — ``lessThan`` calls
        # ``__isNXentry`` (reads ``node.obj.attrs['NX_class']``) and
        # ``childDatasetLessThan(..., 'start_time')``, both of which
        # DEREFERENCE the external link. Expanding a master that links 226
        # external scans therefore opened all 226 scans on the GUI thread
        # (h5py holds the GIL across those network opens) — the freeze that
        # made the file browser unusable and entry clicks unresponsive.
        # With sorting off, ``lessThan`` is never called, so only the rows
        # actually scrolled into view resolve (for their icon), lazily.
        # Entries show in the file's native order (acquisition order for the
        # track_order pygid masters this targets).
        self.tree.setSortingEnabled(False)
        # Single-click silently updates Data tab; double-click jumps to it.
        self.tree.selectionModel().selectionChanged.connect(
            self._on_tree_selection_changed
        )
        self.tree.activated.connect(self._on_tree_activated)
        # Delete (browser-focused) removes the selected file from the tree,
        # mirroring File → Close (Ctrl+W). See _remove_selected_file_from_browser.
        self.tree.deleteFileRequested.connect(self._remove_selected_file_from_browser)
        # Small header row above the tree with the manual Refresh action:
        # re-sync every open session with the filesystem (close deleted
        # originals, reload changed ones). Also on F5.
        self._tree_refresh_btn = QToolButton()
        self._tree_refresh_btn.setText("Refresh")
        self._tree_refresh_btn.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        icons.bind(self._tree_refresh_btn, "refresh")
        self._tree_refresh_btn.setAutoRaise(True)
        self._tree_refresh_btn.setToolTip(
            "Re-check every open file on disk (F5): close files whose "
            "original was deleted (unsaved changes are kept open), "
            "reload files that changed on disk (unless they have "
            "unsaved changes)."
        )
        self._tree_refresh_btn.clicked.connect(self._refresh_file_tree)
        tree_header = QHBoxLayout()
        tree_header.setContentsMargins(2, 2, 2, 0)
        tree_header.addWidget(self._tree_refresh_btn)
        tree_header.addStretch(1)
        tree_box = QWidget(self)
        tree_box_layout = QVBoxLayout(tree_box)
        tree_box_layout.setContentsMargins(0, 0, 0, 0)
        tree_box_layout.setSpacing(0)
        tree_box_layout.addLayout(tree_header)
        tree_box_layout.addWidget(self.tree)
        self._tree_dock = QDockWidget("File browser", self)
        self._tree_dock.setWidget(tree_box)
        self._tree_dock.setObjectName("FileBrowserDock")
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._tree_dock)
        refresh_action = QAction("Refresh file browser", self)
        refresh_action.setShortcut(QKeySequence("F5"))
        refresh_action.triggered.connect(self._refresh_file_tree)
        self.addAction(refresh_action)

        # Right: entry selector + overlay toggles
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        form = QFormLayout()
        self.entry_combo = QComboBox()
        self.entry_combo.currentTextChanged.connect(self._on_entry_changed)
        form.addRow("Entry:", self.entry_combo)

        # Frame-navigation controls live on the *image viewer's*
        # toolbar (alongside the Log-scale checkbox) so they're
        # reachable from any right-dock tab — not just Display. We
        # still build them here because MainWindow owns the slot
        # wiring (slider valueChanged → viewer, play timer, …);
        # ``viewer.insert_frame_controls`` re-parents them onto the
        # toolbar in ``_build_docks`` once the toolbar is ready.
        # Hidden for single-frame stacks where they have no function.
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setSingleStep(1)
        self.frame_slider.setPageStep(1)
        self.frame_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        # Editable frame index next to the slider: type an exact frame
        # and confirm with Enter (keyboard tracking is off so partial
        # input like the "2" of "25" doesn't seek mid-typing).
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, 0)
        self.frame_spin.setKeyboardTracking(False)
        self.frame_spin.setToolTip(
            "Current frame — type an index and press Enter to jump."
        )
        self.frame_spin.valueChanged.connect(self._on_frame_spin_changed)
        # Compact "/ max" readout completing the spinbox ("idx / max").
        # The "Frame" word was dropped to save toolbar space — its
        # meaning is obvious from context (next to play / prev / next
        # icons and a slider).
        self.frame_label = QLabel("—")
        self.frame_label.setMinimumWidth(36)
        self.frame_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        # Play / pause toggle. Drives a QTimer that calls
        # viewer.set_frame(current + 1) on every tick; stops when the
        # last frame is reached. Standard-icon based so qdarkstyle
        # picks up the right colour automatically.
        self.play_button = QToolButton()
        self.play_button.setCheckable(True)
        self.play_button.setIcon(icons.icon("play"))
        self.play_button.setToolTip(
            "Play frames from the current position to the end.\n"
            "Stops at the last frame; click again to pause."
        )
        self.play_button.toggled.connect(self._on_play_toggled)
        # Previous / next single-step buttons. Step by one frame and
        # clamp at boundaries (the buttons disable themselves at
        # frame 0 / last via ``_refresh_frame_nav_enabled``).
        self.prev_frame_button = QToolButton()
        icons.bind(self.prev_frame_button, "prev")
        self.prev_frame_button.setToolTip("Previous frame")
        self.prev_frame_button.setAutoRepeat(True)
        self.prev_frame_button.setAutoRepeatDelay(300)
        self.prev_frame_button.setAutoRepeatInterval(80)
        self.prev_frame_button.clicked.connect(self._on_prev_frame_clicked)
        self.next_frame_button = QToolButton()
        icons.bind(self.next_frame_button, "next")
        self.next_frame_button.setToolTip("Next frame")
        self.next_frame_button.setAutoRepeat(True)
        self.next_frame_button.setAutoRepeatDelay(300)
        self.next_frame_button.setAutoRepeatInterval(80)
        self.next_frame_button.clicked.connect(self._on_next_frame_clicked)
        # Driver for playback. Interval + step are resolved from
        # QSettings (see ``_compute_play_schedule``) on every Play
        # start, so a setting change picks up on the next press
        # without restarting the timer. Default mode is "time per
        # frame" at 50 ms = 20 fps. Requested rates below 50 ms /
        # frame don't speed up the timer — instead, the play-head
        # advances by ``self._play_step`` frames per tick so the
        # target total time is honoured while the timer stays at the
        # 20 fps practical ceiling.
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(DEFAULT_PLAYBACK_FRAME_MS)
        self._play_timer.timeout.connect(self._on_play_tick)
        layout.addLayout(form)
        # Hand the controls to the image viewer's toolbar. Order reads
        # left-to-right: prev / play / next / slider / index / "max".
        self.viewer.insert_frame_controls([
            self.prev_frame_button,
            self.play_button,
            self.next_frame_button,
            self.frame_slider,
            self.frame_spin,
            self.frame_label,
        ])
        # Start hidden — only useful once a multi-frame stack is loaded.
        self._set_frame_slider_visible(False)

        layout.addWidget(QLabel("Overlays"))
        # Manual peaks intentionally omitted: the GUI now keeps at most
        # one manual box per frame (drawn → replaced → committed via
        # Add-to-fitted/detected, removed via Esc / Delete), so a
        # visibility toggle for "all manual peaks" no longer has work
        # to do. The viewer's internal _visibility["manual"] stays True
        # by default — see GIWAXSImageViewer.__init__.
        self._overlay_checks: dict[str, QCheckBox] = {}
        for kind, label in (
            ("detected", "Detected peaks"),
            ("fitted", "Fitted peaks"),
        ):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            swatch = QLabel()
            swatch.setPixmap(_make_pen_swatch(OVERLAY_STYLE[kind]))
            row.addWidget(swatch)
            chk = QCheckBox(label)
            chk.setChecked(True)
            chk.toggled.connect(
                lambda v, k=kind: self.viewer.set_overlay_visible(k, v)
            )
            row.addWidget(chk)
            row.addStretch(1)
            row_widget = QWidget()
            row_widget.setLayout(row)
            layout.addWidget(row_widget)
            self._overlay_checks[kind] = chk

            # Per-detected min-score slider. Sits indented under the
            # Detected checkbox so the affordance is right next to
            # the layer it controls. Range 0–100 → 0.00–1.00 cutoff
            # forwarded to ``viewer.set_detected_score_cutoff``.
            # Initial value is seeded to the minimum score on the
            # current frame in ``_seed_detected_score_slider`` so
            # the default shows every detection; the user drags up
            # to hide weak ones.
            if kind == "detected":
                score_row = QHBoxLayout()
                score_row.setContentsMargins(20, 0, 0, 0)
                score_row.setSpacing(6)
                score_row.addWidget(QLabel("Min score:"))
                self._detected_score_slider = QSlider(Qt.Orientation.Horizontal)
                self._detected_score_slider.setRange(0, 100)
                self._detected_score_slider.setValue(0)
                self._detected_score_slider.setToolTip(
                    "Hide detected peaks whose model score is below "
                    "the cutoff. The slider starts at the lowest "
                    "score on the current frame so nothing is hidden "
                    "by default."
                )
                self._detected_score_slider.valueChanged.connect(
                    self._on_detected_score_changed
                )
                score_row.addWidget(self._detected_score_slider, 1)
                self._detected_score_value_label = QLabel("0.00")
                self._detected_score_value_label.setMinimumWidth(36)
                score_row.addWidget(self._detected_score_value_label)
                score_row_widget = QWidget()
                score_row_widget.setLayout(score_row)
                layout.addWidget(score_row_widget)

        # Matched peaks: master toggle + per-structure rows. The per-structure
        # rows are rebuilt on every frame change because different frames can
        # have different matching solutions.
        matched_master_row = QHBoxLayout()
        matched_master_row.setContentsMargins(0, 0, 0, 0)
        matched_master_row.setSpacing(6)
        # Match the Detected/Fitted layout exactly: those rows put a
        # 26×12 pixmap-bearing QLabel before the checkbox. QLabel
        # renders with different content margins depending on whether
        # it carries a pixmap or not, so a setFixedSize-only label
        # ends up a couple pixels off vertically. Give the matched
        # spacer a transparent pixmap of the same dimensions so its
        # sizing semantics line up byte-for-byte with the real
        # swatches above.
        _ref_swatch = _make_pen_swatch(OVERLAY_STYLE["detected"])
        _spacer_pixmap = QPixmap(_ref_swatch.size())
        _spacer_pixmap.fill(Qt.GlobalColor.transparent)
        _matched_swatch_spacer = QLabel()
        _matched_swatch_spacer.setPixmap(_spacer_pixmap)
        matched_master_row.addWidget(_matched_swatch_spacer)
        self._matched_master_check = QCheckBox("Matched peaks")
        self._matched_master_check.setChecked(True)
        self._matched_master_check.toggled.connect(self._on_matched_master_toggled)
        # Per-structure checkboxes are rebuilt in _refresh_matched_panel
        # but kept indexed here so the master-toggle cascade and the
        # "single structure on while master off" promotion path can
        # reach them by uid.
        self._matched_struct_checkboxes: dict[str, QCheckBox] = {}
        matched_master_row.addWidget(self._matched_master_check)
        matched_master_row.addStretch(1)
        # Matched display style: boxes (default) vs q-map-style markers
        # (circles for peaks, dashed arcs for rings) — nicer for scan
        # playback where varying box sizes are distracting.
        matched_master_row.addWidget(QLabel("style:"))
        self._matched_style_combo = QComboBox()
        self._matched_style_combo.addItem("Boxes", userData="boxes")
        self._matched_style_combo.addItem("Markers", userData="markers")
        self._matched_style_combo.setToolTip(
            "How matched structures are drawn on the image: 'Boxes' "
            "(per-peak boxes) or 'Markers' (hollow circles for peaks, "
            "dashed arcs for rings — the q-map look, steadier during "
            "playback since it ignores per-peak box size)."
        )
        self._matched_style_combo.currentIndexChanged.connect(
            self._on_matched_style_changed
        )
        matched_master_row.addWidget(self._matched_style_combo)
        matched_master_widget = QWidget()
        matched_master_widget.setLayout(matched_master_row)
        layout.addWidget(matched_master_widget)

        # Substring filter for the per-structure rows below. Useful
        # when matching has been run against a folder of many CIFs
        # (``cif_organic`` has 32 entries — 32 rows is hard to scan
        # by eye). Filter is case-insensitive, applied live as the
        # user types. Indented to align with the structure rows.
        matched_filter_row = QHBoxLayout()
        matched_filter_row.setContentsMargins(20, 0, 0, 0)
        matched_filter_row.setSpacing(6)
        matched_filter_row.addWidget(QLabel("Filter:"))
        self._matched_filter_edit = QLineEdit()
        self._matched_filter_edit.setPlaceholderText("CIF name substring…")
        self._matched_filter_edit.setClearButtonEnabled(True)
        self._matched_filter_edit.textChanged.connect(self._apply_matched_filter)
        matched_filter_row.addWidget(self._matched_filter_edit, 1)
        matched_filter_widget = QWidget()
        matched_filter_widget.setLayout(matched_filter_row)
        layout.addWidget(matched_filter_widget)

        # Min-probability slider — hides matched rows whose structure
        # probability falls below the cutoff. Composes with the
        # CIF-substring filter above and the per-structure visibility
        # checkboxes below. Integer slider 0–100 represents a 0.00–1.00
        # threshold; rendered live next to the slider for readability.
        prob_row = QHBoxLayout()
        prob_row.setContentsMargins(20, 0, 0, 0)
        prob_row.setSpacing(6)
        prob_row.addWidget(QLabel("Min p:"))
        self._matched_prob_slider = QSlider(Qt.Orientation.Horizontal)
        self._matched_prob_slider.setRange(0, 100)
        self._matched_prob_slider.setValue(0)
        self._matched_prob_slider.setToolTip(
            "Hide matched structures whose probability is below the "
            "cutoff. Composes with the CIF-name filter above."
        )
        self._matched_prob_slider.valueChanged.connect(
            self._on_matched_prob_changed
        )
        prob_row.addWidget(self._matched_prob_slider, 1)
        self._matched_prob_value_label = QLabel("0.00")
        self._matched_prob_value_label.setMinimumWidth(36)
        prob_row.addWidget(self._matched_prob_value_label)
        prob_widget = QWidget()
        prob_widget.setLayout(prob_row)
        layout.addWidget(prob_widget)

        # Container for the dynamic per-structure rows. Indented so it reads
        # as a sub-list of the master toggle.
        self._matched_struct_container = QWidget()
        self._matched_struct_layout = QVBoxLayout(self._matched_struct_container)
        self._matched_struct_layout.setContentsMargins(20, 0, 0, 0)
        self._matched_struct_layout.setSpacing(2)
        layout.addWidget(self._matched_struct_container)
        # Per-uid row widgets — used by _apply_matched_filter to
        # show / hide individual rows without rebuilding from data.
        self._matched_struct_rows: dict[str, QWidget] = {}
        # Per-uid structure probability snapshot. Populated in
        # ``_refresh_matched_panel`` and consumed by the min-p
        # slider filter in ``_apply_matched_filter``.
        self._matched_struct_probs: dict[str, float] = {}
        # Shown when a non-empty filter hides every row (distinct
        # from the "no matched solutions" empty-list label, which is
        # torn down with the rest of the rows on every rebuild).
        self._matched_filter_empty_label: QLabel | None = None

        # Twin legend for the Expected-pattern dock: the same master
        # toggle and per-structure rows, rebuilt alongside the Display
        # ones in _refresh_matched_panel with checkbox state mirrored
        # both ways, so structures can be shown/hidden without
        # switching back to the Display tab. Built here so the initial
        # _refresh_matched_panel call below can populate it; the
        # container joins the Expected-pattern dock's layout when that
        # dock is assembled. Deliberately NOT part of
        # _sim_section_container: matched structures exist (and stay
        # toggleable) even while no CIF cache is parsed and the rest
        # of that section is disabled.
        self._sim_legend_container = QWidget()
        sim_legend_box = QVBoxLayout(self._sim_legend_container)
        sim_legend_box.setContentsMargins(0, 0, 0, 0)
        sim_legend_box.setSpacing(2)
        sim_legend_master_row = QHBoxLayout()
        sim_legend_master_row.setContentsMargins(0, 0, 0, 0)
        sim_legend_master_row.setSpacing(6)
        self._sim_legend_master_check = QCheckBox("Matched peaks")
        self._sim_legend_master_check.setChecked(
            self._matched_master_check.isChecked()
        )
        self._sim_legend_master_check.setToolTip(
            "Mirror of the Display tab's Matched-peaks master toggle "
            "(the two stay in sync): show or hide every "
            "matched-structure overlay. The CIF-name filter and the "
            "min-probability slider live in the Display tab and also "
            "apply to this list."
        )
        self._sim_legend_master_check.toggled.connect(
            self._on_matched_master_toggled
        )
        sim_legend_master_row.addWidget(self._sim_legend_master_check)
        sim_legend_master_row.addStretch(1)
        sim_legend_master_widget = QWidget()
        sim_legend_master_widget.setLayout(sim_legend_master_row)
        sim_legend_box.addWidget(sim_legend_master_widget)
        self._sim_legend_struct_container = QWidget()
        self._sim_legend_struct_layout = QVBoxLayout(
            self._sim_legend_struct_container
        )
        self._sim_legend_struct_layout.setContentsMargins(20, 0, 0, 0)
        self._sim_legend_struct_layout.setSpacing(2)
        sim_legend_box.addWidget(self._sim_legend_struct_container)
        self._sim_legend_struct_checkboxes: dict[str, QCheckBox] = {}
        self._sim_legend_struct_rows: dict[str, QWidget] = {}
        self._sim_legend_filter_empty_label: QLabel | None = None
        # Every "Unmatched fitted peaks" checkbox currently built (one
        # per legend) — kept for cross-legend sync in
        # _on_unmatched_row_toggled.
        self._unmatched_row_checks: list[QCheckBox] = []

        self._refresh_matched_panel(0, [])
        self.viewer.matchedStructuresChanged.connect(self._refresh_matched_panel)

        # "Expected pattern": forward-simulated reflections of a parsed
        # CIF, rendered as a display-only overlay. Fed by the Pipeline
        # panel's cached CifPattern (Parse CIFs) — the whole section is
        # disabled until a cache exists. Wrapped in one container so
        # enable/disable + tooltip reach every row at once. The
        # cifCacheChanged connection is made after the Pipeline panel
        # is constructed (it doesn't exist yet at this point).
        sim_box = QVBoxLayout()
        sim_box.setContentsMargins(0, 0, 0, 0)
        sim_box.setSpacing(2)
        sim_master_row = QHBoxLayout()
        sim_master_row.setContentsMargins(0, 0, 0, 0)
        sim_master_row.setSpacing(6)
        sim_swatch = QLabel()
        sim_swatch.setPixmap(_make_pen_swatch({
            "color": SIM_MISSED_COLOR,
            "style": Qt.PenStyle.DashLine,
            "width": 1.5,
        }))
        sim_master_row.addWidget(sim_swatch)
        self._sim_master_check = QCheckBox("Show expected pattern")
        self._sim_master_check.setChecked(False)
        self._sim_master_check.setToolTip(
            "Overlay the forward-simulated reflections of a parsed CIF "
            "(diamonds for spots, dashed arcs for rings; marker size "
            "encodes simulated intensity). Display-only — nothing is "
            "written to the file."
        )
        self._sim_master_check.toggled.connect(self._on_sim_master_toggled)
        sim_master_row.addWidget(self._sim_master_check)
        sim_master_row.addStretch(1)
        sim_master_widget = QWidget()
        sim_master_widget.setLayout(sim_master_row)
        sim_box.addWidget(sim_master_widget)

        sim_cif_row = QHBoxLayout()
        sim_cif_row.setContentsMargins(20, 0, 0, 0)
        sim_cif_row.setSpacing(6)
        sim_cif_row.addWidget(QLabel("CIF:"))
        self._sim_cif_combo = QComboBox()
        self._sim_cif_combo.setToolTip(
            "Structure to simulate, from the parsed CIF cache. Entries "
            "matched on the current frame are marked."
        )
        self._sim_cif_combo.currentIndexChanged.connect(
            self._on_sim_cif_changed
        )
        sim_cif_row.addWidget(self._sim_cif_combo, 1)
        sim_cif_widget = QWidget()
        sim_cif_widget.setLayout(sim_cif_row)
        sim_box.addWidget(sim_cif_widget)

        sim_orient_row = QHBoxLayout()
        sim_orient_row.setContentsMargins(20, 0, 0, 0)
        sim_orient_row.setSpacing(6)
        sim_orient_row.addWidget(QLabel("Orientation:"))
        self._sim_orient_mode = QComboBox()
        self._sim_orient_mode.addItem("matched", userData="matched")
        self._sim_orient_mode.addItem(
            "random (powder rings)", userData="random"
        )
        self._sim_orient_mode.addItem(
            "user-specified hkl", userData="user"
        )
        self._sim_orient_mode.setToolTip(
            "How to pick the texture orientation to render: 'matched' "
            "follows the structure's matched orientation(s) on the "
            "current frame; 'random' is the orientation-free powder "
            "ring pattern (randomly oriented crystallites); "
            "'user-specified' renders the (h k l) you type."
        )
        self._sim_orient_mode.currentIndexChanged.connect(
            self._on_sim_orient_mode_changed
        )
        sim_orient_row.addWidget(self._sim_orient_mode)
        self._sim_matched_combo = QComboBox()
        self._sim_matched_combo.setToolTip(
            "Matched orientation to render, labelled with its matching "
            "probability on the current frame."
        )
        self._sim_matched_combo.currentIndexChanged.connect(
            self._on_sim_matched_changed
        )
        sim_orient_row.addWidget(self._sim_matched_combo, 1)
        self._sim_hkl_edit = QLineEdit()
        self._sim_hkl_edit.setPlaceholderText("h k l   e.g. 0 0 1")
        self._sim_hkl_edit.setToolTip(
            "Miller indices of the orientation to render: three "
            "integers separated by spaces or commas. Validated "
            "against the structure's precomputed orientations; "
            "equivalent spellings (2 2 0, 0 0 -1) resolve to the "
            "simulated one (1 1 0, 0 0 1)."
        )
        self._sim_hkl_edit.editingFinished.connect(
            self._on_sim_hkl_edited
        )
        self._sim_hkl_edit.setVisible(False)
        sim_orient_row.addWidget(self._sim_hkl_edit, 1)
        sim_orient_widget = QWidget()
        sim_orient_widget.setLayout(sim_orient_row)
        sim_box.addWidget(sim_orient_widget)
        # One-line state/validation hint under the orientation row:
        # invalid typed hkl, equivalent-spelling note, or "no matched
        # orientation on this frame". Hidden when there is nothing to
        # say.
        self._sim_orient_hint = QLabel()
        self._sim_orient_hint.setContentsMargins(20, 0, 0, 0)
        self._sim_orient_hint.setWordWrap(True)
        self._sim_orient_hint.setVisible(False)
        sim_box.addWidget(self._sim_orient_hint)

        sim_int_row = QHBoxLayout()
        sim_int_row.setContentsMargins(20, 0, 0, 0)
        sim_int_row.setSpacing(6)
        sim_int_row.addWidget(QLabel("Min int:"))
        # Structure-factor intensities span decades, so a linear slider
        # is useless here (everything interesting sits between 0 and
        # ~1% of the strongest reflection). A lone spinbox with
        # adaptive-decimal stepping instead: the arrow/wheel step is
        # always one decade below the current value's magnitude, so it
        # moves in useful increments at 0.01% just as at 50%.
        self._sim_int_spin = QDoubleSpinBox()
        self._sim_int_spin.setRange(0.0, 100.0)
        self._sim_int_spin.setDecimals(4)
        self._sim_int_spin.setStepType(
            QAbstractSpinBox.StepType.AdaptiveDecimalStepType
        )
        self._sim_int_spin.setSuffix(" %")
        self._sim_int_spin.setValue(1.0)
        self._sim_int_spin.setToolTip(
            "Hide simulated reflections whose intensity is below this "
            "percentage of the pattern's strongest reflection. "
            "Arrows/wheel step relative to the current value, so tiny "
            "cutoffs stay reachable."
        )
        self._sim_int_spin.valueChanged.connect(
            self._on_sim_int_spin_changed
        )
        sim_int_row.addWidget(self._sim_int_spin, 1)
        sim_int_widget = QWidget()
        sim_int_widget.setLayout(sim_int_row)
        sim_box.addWidget(sim_int_widget)

        # Selection row: bulk-select the reflections the data does not
        # account for, plus a live count of what's queued. Clicking
        # individual markers on the image toggles them one by one.
        sim_sel_row = QHBoxLayout()
        sim_sel_row.setContentsMargins(20, 0, 0, 0)
        sim_sel_row.setSpacing(6)
        self._sim_select_missed_btn = QPushButton("Select missed")
        self._sim_select_missed_btn.setToolTip(
            "Select every visible simulated reflection that is neither "
            "explained by a matched peak nor covered by any fitted "
            "peak on this frame. Click markers on the image to toggle "
            "reflections individually."
        )
        self._sim_select_missed_btn.clicked.connect(
            lambda: self.viewer.select_missed_simulated(
                self.viewer.current_frame
            )
        )
        sim_sel_row.addWidget(self._sim_select_missed_btn)
        self._sim_sel_count_label = QLabel("0 selected")
        sim_sel_row.addWidget(self._sim_sel_count_label)
        sim_sel_row.addStretch(1)
        sim_sel_widget = QWidget()
        sim_sel_widget.setLayout(sim_sel_row)
        sim_box.addWidget(sim_sel_widget)
        self.viewer.simulationSelectionChanged.connect(
            self._on_sim_selection_changed
        )

        # The action button: inject the selected reflections as real
        # detected boxes, 2D-fit each, re-match the frame. Enabled only
        # with a non-empty selection on a frame that already has
        # detected/fitted datasets, while the pipeline is idle.
        sim_add_row = QHBoxLayout()
        sim_add_row.setContentsMargins(20, 0, 0, 0)
        sim_add_row.setSpacing(6)
        self._sim_add_btn = set_variant(
            QPushButton("Add selected peaks (fit + match)"), PRIMARY)
        self._sim_add_btn.setEnabled(False)
        self._sim_add_btn.clicked.connect(self._on_add_predicted_peaks)
        sim_add_row.addWidget(self._sim_add_btn)
        sim_add_row.addStretch(1)
        sim_add_widget = QWidget()
        sim_add_widget.setLayout(sim_add_row)
        sim_box.addWidget(sim_add_widget)

        # The all-frames sweep: every matched (CIF, orientation) combo
        # found anywhere in the entry becomes a template applied to
        # EVERY frame — "here should be a peak, can we find it?".
        # Independent of the overlay checkbox and of the reflection
        # selection.
        sim_sweep_row = QHBoxLayout()
        sim_sweep_row.setContentsMargins(20, 0, 0, 0)
        sim_sweep_row.setSpacing(6)
        self._sim_sweep_btn = QPushButton(
            "Find all matched structures (all frames)"
        )
        self._sim_sweep_btn.clicked.connect(self._on_fill_all_matched)
        sim_sweep_row.addWidget(self._sim_sweep_btn)
        sim_sweep_row.addStretch(1)
        sim_sweep_widget = QWidget()
        sim_sweep_widget.setLayout(sim_sweep_row)
        sim_box.addWidget(sim_sweep_widget)
        # The per-structure reflection caps live in the sweep's
        # confirmation dialog (one spinbox per found structure) — a
        # permanent cap row here would crowd the dock for a setting
        # that only matters at launch time.
        self._update_sim_add_button()

        self._sim_section_container = QWidget()
        self._sim_section_container.setLayout(sim_box)
        self._sim_section_container.setEnabled(False)
        # Not added to the Display panel: the whole Expected-pattern
        # workflow lives in its own tabbed dock (built right after the
        # Pipeline dock below).
        # (stem, hkl) of the pattern currently installed in the viewer
        # (None = overlay clear). Lets frame changes rebuild the combo
        # labels without re-applying an unchanged pattern — re-applying
        # would drop the reflection selection.
        self._sim_applied: tuple | None = None
        # User-typed orientation, kept only when it validated against
        # the selected CIF's precomputed set (None = empty/invalid).
        self._sim_user_hkl: tuple | None = None
        # Until the user touches the mode combo, the mode follows the
        # data: "matched" when the frame has matched orientations for
        # the CIF, "random" (powder) otherwise — the pre-mode default.
        self._sim_mode_auto = True
        self.viewer.matchedStructuresChanged.connect(
            self._refresh_sim_matched_entries
        )

        layout.addSpacing(6)

        self.parameter_panel = ParameterPanel(self)
        layout.addWidget(self.parameter_panel)

        # Note: the long polar-mode hint that used to live here has
        # moved to **Help → Controls & shortcuts…** to free up
        # vertical space in the Display dock.

        layout.addStretch(1)

        # Wrap the dock content in a QScrollArea so files with many
        # matched structures (one row per (CIF, hkl) match) don't push
        # the parameter panel and shortcut hint off the bottom of the
        # screen. Vertical scrolling kicks in on demand; horizontal is
        # locked off so narrow docks wrap their form rows instead of
        # introducing an x-axis scrollbar.
        display_scroll = QScrollArea(self)
        display_scroll.setWidgetResizable(True)
        display_scroll.setFrameShape(QFrame.Shape.NoFrame)
        display_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        display_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        display_scroll.setWidget(panel)
        self._display_dock = QDockWidget("Display", self)
        self._display_dock.setWidget(display_scroll)
        self._display_dock.setObjectName("DisplayDock")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._display_dock)

        # Pipeline dock — tabified with Display on the right.
        self.pipeline_panel = PipelinePanel(self)
        # Let the panel resolve "Active entry" / "Active frame" at click time
        # without pulling MainWindow into its imports. Returning None for
        # either falls through to mlgidBASE's all-entries / all-frames default.
        self.pipeline_panel.set_active_entry_resolver(
            lambda: self.entry_combo.currentText() or None
        )
        self.pipeline_panel.set_active_frame_resolver(
            lambda: self.viewer.current_frame if self.session is not None else None
        )
        self.pipeline_panel.set_frame_count_resolver(
            lambda: self.viewer.n_frames if self.session is not None else None
        )
        self.pipeline_panel.runRequested.connect(self._on_run_requested)
        self.pipeline_panel.parseCifsRequested.connect(self._on_parse_cifs_requested)
        self.pipeline_panel.cifCacheChanged.connect(self._on_cif_cache_changed)
        self._update_sim_section_state()
        self._pipeline_dock = QDockWidget("Pipeline", self)
        self._pipeline_dock.setWidget(self.pipeline_panel)
        self._pipeline_dock.setObjectName("PipelineDock")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._pipeline_dock)
        self.tabifyDockWidget(self._display_dock, self._pipeline_dock)

        # Expected pattern dock — the forward-simulation overlay plus
        # the add/sweep workflow, in its own right-side tab so the
        # Display dock stays a pure view-settings panel. Same scroll
        # treatment as Display.
        sim_panel = QWidget()
        sim_panel_layout = QVBoxLayout(sim_panel)
        sim_panel_layout.setContentsMargins(8, 8, 8, 8)
        sim_panel_layout.addWidget(self._sim_section_container)
        # The matched-structures twin legend (built next to the Display
        # legend above) — outside the section container so it stays
        # enabled even when the simulation workflow is greyed out.
        sim_panel_layout.addSpacing(8)
        sim_panel_layout.addWidget(self._sim_legend_container)
        sim_panel_layout.addStretch(1)
        sim_scroll = QScrollArea(self)
        sim_scroll.setWidgetResizable(True)
        sim_scroll.setFrameShape(QFrame.Shape.NoFrame)
        sim_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        sim_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        sim_scroll.setWidget(sim_panel)
        self._sim_dock = QDockWidget("Expected pattern", self)
        self._sim_dock.setWidget(sim_scroll)
        self._sim_dock.setObjectName("ExpectedPatternDock")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._sim_dock)
        self.tabifyDockWidget(self._pipeline_dock, self._sim_dock)

        # Peaks panel — sortable per-frame view of detected / fitted /
        # matched peaks with bidirectional click-sync to the image
        # viewer's selection. The dock itself is built further down
        # so it can be tabified with the Profile dock at the bottom
        # of the window (the two are read together — peak row +
        # cross-section profile — so sharing a tab area is more
        # practical than burying Peaks among the right-side
        # control docks).
        self.peaks_table_panel = PeaksTablePanel(self)

        # Conversion dock — mode-exclusive sibling of the Pipeline dock.
        # Visible only when the active session is a RawSession; switching
        # between Nexus and Raw sessions hides one and shows the other.
        # Both share the same dock slot (tabified with Display) so the
        # right side never grows beyond two visible tabs.
        self.conversion_panel = ConversionPanel(self)
        self.conversion_panel.conversionRunRequested.connect(
            self._on_conversion_run
        )
        # Let the conversion panel's in-GUI calibration dialog ask
        # for the currently displayed raw frame so it can pre-load
        # the user's image without an extra browse step. Returns
        # None when no raw session is active or the viewer hasn't
        # been populated yet.
        self.conversion_panel.set_active_raw_frame_resolver(
            self._active_raw_frame_for_calibration
        )
        # Flip the live raw preview when the panel's fliplr/flipud checkboxes
        # toggle, so the user sees the orientation the conversion will produce.
        self.conversion_panel.rawFlipsChanged.connect(self._on_raw_flips_changed)
        self._conversion_dock = QDockWidget("Conversion", self)
        self._conversion_dock.setWidget(self.conversion_panel)
        self._conversion_dock.setObjectName("ConversionDock")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._conversion_dock)
        # Peaks is no longer on the right side, so the chain is just
        # Display | Pipeline | Conversion | Logs. Conversion is
        # tabified with Pipeline (they're mode-exclusive siblings)
        # so the visible tab triplet is Display | <Pipeline or
        # Conversion> | Logs.
        self.tabifyDockWidget(self._sim_dock, self._conversion_dock)
        # Default state matches the default session (none): pipeline dock
        # shown so the user can see what would be available once they
        # open a converted file. ``_apply_session_mode`` handles toggles
        # from then on.
        self._conversion_dock.setVisible(False)

        # Shared Logs dock — tabified next to Display / Pipeline / Conversion.
        # Both panels emit ``logMessage`` / ``logCleared``; we route them
        # through this single widget so the log history is visible in
        # either mode (and a switch from Conversion to NeXus doesn't hide
        # the running log).
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFont(QFont("monospace"))
        self._log_view.setMaximumBlockCount(4000)
        self._log_view.setPlaceholderText(
            "Pipeline and conversion logs land here."
        )
        self._logs_dock = QDockWidget("Logs", self)
        self._logs_dock.setWidget(self._log_view)
        self._logs_dock.setObjectName("LogsDock")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._logs_dock)
        self.tabifyDockWidget(self._conversion_dock, self._logs_dock)

        # Route both panels' log messages into the shared widget. Both
        # panels' ``append_log`` / ``clear_log`` already emit these
        # signals — every existing call site keeps working.
        self.pipeline_panel.logMessage.connect(self._log_view.appendPlainText)
        self.pipeline_panel.logCleared.connect(self._log_view.clear)
        self.conversion_panel.logMessage.connect(self._log_view.appendPlainText)
        self.conversion_panel.logCleared.connect(self._log_view.clear)

        self._display_dock.raise_()

        # Bottom: profile viewer + peaks table, tabified together.
        # Default to ~30% of window height so the central image
        # stays the main focus. Profile is the first tab (raised)
        # because the live cross-section is more frequently read
        # than the peak table; the peak table sits behind it,
        # one click away.
        self.profile_viewer = ProfileViewer(self)
        self._profile_dock = QDockWidget("Profiles", self)
        self._profile_dock.setWidget(self.profile_viewer)
        self._profile_dock.setObjectName("ProfileDock")
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._profile_dock)
        self._peaks_dock = QDockWidget("Peaks", self)
        self._peaks_dock.setWidget(self.peaks_table_panel)
        self._peaks_dock.setObjectName("PeaksDock")
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._peaks_dock)
        self.tabifyDockWidget(self._profile_dock, self._peaks_dock)
        # Scan tracking joins the bottom tab group behind Profiles and
        # Peaks — it shares their per-peak viewing context (click a
        # track, see the peak) and, like them, only means something for
        # converted multi-frame data.
        self.scan_tracking_panel = ScanTrackingPanel(self)
        self._scan_tracking_dock = QDockWidget("Scan tracking", self)
        self._scan_tracking_dock.setWidget(self.scan_tracking_panel)
        self._scan_tracking_dock.setObjectName("ScanTrackingDock")
        self.addDockWidget(
            Qt.DockWidgetArea.BottomDockWidgetArea, self._scan_tracking_dock
        )
        self.tabifyDockWidget(self._peaks_dock, self._scan_tracking_dock)
        self.scan_tracking_panel.trackRequested.connect(
            self._on_track_scan_requested
        )
        self.scan_tracking_panel.showViewsRequested.connect(
            self._on_show_phase_views
        )
        self.scan_tracking_panel.trackRowSelected.connect(
            self._on_track_row_selected
        )
        self.scan_tracking_panel.onlyTrackedToggled.connect(
            self._on_only_tracked_toggled
        )
        self.scan_tracking_panel.interpolateRequested.connect(
            self._on_interpolate_tracks_requested
        )
        self.scan_tracking_panel.trackDeleteRequested.connect(
            self._on_scan_track_deleted
        )
        # Image -> table sync: selecting a fitted peak anywhere (image
        # click or Peaks-dock row) highlights its cross-frame track row
        # and surfaces the frame span in the status bar.
        self.viewer.selectionChanged.connect(
            self._forward_selection_to_scan_tracking
        )
        self._profile_dock.raise_()
        self.resizeDocks(
            [self._profile_dock], [max(self.height() // 3, 280)], Qt.Orientation.Vertical
        )
        # Default column widths. Both areas are tuned together — the
        # file browser was previously squeezed to ~100 px (truncated
        # HDF5 paths), and after Peaks moved to the bottom the right
        # dock area's natural sizeHint shrank from ~560 to ~240. The
        # pinned values below split the difference: 280 leaves enough
        # room for typical file paths and 350 gives the Display dock's
        # form rows headroom without dominating the central image.
        self.resizeDocks(
            [self._tree_dock, self._display_dock],
            [260, 350],
            Qt.Orientation.Horizontal,
        )
        self.viewer.frameChanged.connect(self.profile_viewer.set_frame)
        # Bidirectional Display-dock slider sync: viewer pushes frame
        # changes into the slider (e.g. user scrubs the pyqtgraph
        # timeline below the image), and the slider's valueChanged
        # already pushes back into the viewer via _on_frame_slider_changed.
        self.viewer.frameChanged.connect(self._on_viewer_frame_changed)
        # Bidirectional sync between 2D ROI and profile-edge regions. The
        # profile viewer only handles ManualPeak, so we filter the
        # SelectedPeak-typed signals down to the manual case before forwarding.
        self.viewer.selectionChanged.connect(self._forward_selection_to_profile)
        self.viewer.peakGeometryChanged.connect(self._forward_geom_to_profile)
        self.profile_viewer.peakGeometryChanged.connect(self.viewer.update_peak_geometry_external)
        # Detected-peak profile region drag — live updates flow into
        # the viewer's in-memory PeakTable so the colored overlay
        # tracks the drag; the disk write fires once on drag-end via
        # _on_detected_border_commit (mirrors how the image-side ROI
        # drag commits via peakRowWriteRequested).
        self.profile_viewer.detectedPeakGeometryChanged.connect(
            self.viewer.update_detected_geometry_external
        )
        self.profile_viewer.detectedPeakBorderCommit.connect(
            self._on_detected_border_commit
        )
        # The faint fitted-preview box for the selected manual peak follows
        # the profile viewer's 1D Gaussian fits. It also has to drop when
        # the selection changes away from a manual peak.
        self.profile_viewer.fitParamsChanged.connect(self._update_fitted_preview)
        self.viewer.selectionChanged.connect(self._on_selection_for_preview)
        # Live readout in the parameter panel — routed through a
        # host dispatcher (instead of straight to ``set_fits``) so
        # 2D mode can substitute pygidfit's refined values for
        # scipy's 1D fits. The dispatcher matches what
        # Add-to-fitted (2D) will actually save, so the readout no
        # longer claims to commit scipy-derived numbers in 2D mode.
        self.profile_viewer.fitParamsChanged.connect(self._dispatch_fit_params)

        # Parameter readout — both selection and geometry changes feed the same slot.
        self.viewer.selectionChanged.connect(self.parameter_panel.set_peak)
        self.viewer.peakGeometryChanged.connect(self.parameter_panel.set_peak)

        # Peaks table sync. Image → table: mirror the selection onto
        # the relevant row (auto-switches tab). Table → image: route
        # row clicks back through the viewer's selection setter.
        self.viewer.selectionChanged.connect(self.peaks_table_panel.set_external_selection)
        self.viewer.frameChanged.connect(self._refresh_peaks_table_on_frame)
        self.peaks_table_panel.peakSelectedFromTable.connect(
            self._on_peak_selected_from_table
        )

        # Commit / delete actions on the parameter panel. Add-to-detected and
        # delete reuse the existing PipelineWorker path.
        self.parameter_panel.addToDetectedRequested.connect(self._on_add_to_detected)
        self.parameter_panel.addToFittedRequested.connect(self._on_add_to_fitted)
        # Batch 2D fit of the multi-selection. The button on the
        # parameter panel emits ``batchFit2DRequested``; the host runs
        # pygidfit inside a modal QProgressDialog (see
        # ``_on_batch_fit_2d``). Enable state is re-evaluated on every
        # selectionsChanged + saveAsRingChanged tick by
        # ``_refresh_fit_buttons``.
        self.parameter_panel.batchFit2DRequested.connect(self._on_batch_fit_2d)
        self.viewer.selectionsChanged.connect(
            lambda _sels: self._refresh_fit_buttons()
        )
        self.parameter_panel.saveAsRingChanged.connect(
            lambda _r: self._refresh_fit_buttons()
        )
        # Apply once so the cold-start panel reflects the initial
        # empty-selection state (both fit buttons hidden).
        self._refresh_fit_buttons()
        # Refresh the cyan preview overlay immediately when the user
        # toggles ring/segment — otherwise the preview would lag until
        # the next fit recompute.
        self.parameter_panel.saveAsRingChanged.connect(self._on_save_as_ring_changed)
        # Flipping the 1D / 2D fit-mode radios changes how the dashed
        # cyan preview's box widths are computed — re-invoke the
        # preview slot with the cached 1D fits so the box redraws at
        # the new mode's convention immediately, instead of lagging
        # until the next fit recompute (frame change, ROI drag, etc.).
        self.parameter_panel.fitModeChanged.connect(self._on_fit_mode_changed)
        # Live 2D preview: run pygidfit on selection / frame / mode
        # changes so the profile fits + cyan box mirror what
        # Add-to-fitted (2D) will save. Selection already calls
        # ``_refresh_2d_preview`` via ``_on_selection_for_preview``;
        # also wire frame + mode + ring so all four user-triggers
        # refresh the override.
        self.viewer.frameChanged.connect(
            lambda _f: self._refresh_2d_preview()
        )
        self.parameter_panel.fitModeChanged.connect(
            lambda _m: self._refresh_2d_preview()
        )
        self.parameter_panel.saveAsRingChanged.connect(
            lambda _r: self._refresh_2d_preview()
        )
        # ROI drag-end: the user repositions a manual / detected box
        # to a new peak. ``peakGeometryChanged`` fires per drag tick
        # (too slow for pygidfit), ``peakGeometryDragFinished`` fires
        # once after the handle settles. Cache fingerprint includes
        # the geometry, so the next refresh is a real recompute.
        self.viewer.peakGeometryDragFinished.connect(
            lambda _sel: self._refresh_2d_preview()
        )
        # Live 2D preview during drag: debounce pygidfit runs so the
        # cyan preview + parameter readout track the user's drag.
        # Per-tick refits would freeze the GUI (pygidfit is ~100-
        # 500 ms); a single-shot timer that restarts on every drag
        # tick collapses bursts into one fit every ~150 ms while
        # idle, then a final fit on drag-end via the existing
        # ``peakGeometryDragFinished`` wiring. While the timer is
        # pending, ``_current_pygidfit_box_for_selection`` returns
        # the *cached* refined box (ignoring the geometry fingerprint
        # mismatch) so the cyan box stays at pygidfit's last known
        # result instead of falling through to scipy 1D widths.
        self._drag_pygidfit_timer = make_debounced_timer(
            self, 150, self._refresh_2d_preview
        )
        self.viewer.peakGeometryChanged.connect(
            self._schedule_drag_2d_preview
        )
        # Pink fit curves on the profile plots are hidden only when
        # a manual / detected peak is selected in 2D mode (the
        # source-of-fit case where the 2D-projected pink preview
        # was misleading). Selection changes also re-evaluate so
        # switching to a fitted peak immediately restores its
        # overlay — see ``_apply_fit_curve_visibility``.
        self.parameter_panel.fitModeChanged.connect(
            lambda _m: self._apply_fit_curve_visibility()
        )
        self.parameter_panel.saveAsRingChanged.connect(
            lambda _r: self._apply_fit_curve_visibility()
        )
        self.viewer.selectionChanged.connect(
            lambda _s: self._apply_fit_curve_visibility()
        )
        # Apply the initial state so a cold-start GUI reflects the
        # parameter panel's default mode + no selection.
        self._apply_fit_curve_visibility()
        self.parameter_panel.deletePeakRequested.connect(
            lambda: self._on_delete_peak_requested(self.viewer.selected_peak)
        )

        # Direct-h5py geometry writes for detected/fitted ROI edits.
        self.viewer.peakRowWriteRequested.connect(self._on_peak_row_write_requested)
        # Delete keypress on file-resident peaks.
        self.viewer.deletePeakRequested.connect(self._on_delete_peak_requested)
        # Delete keypress with >= 2 detected peaks selected → bulk delete.
        self.viewer.deletePeaksRequested.connect(self._on_delete_peaks_requested)

        # Keep _ring_pre_geom in sync with the manual peak it points at.
        # When the user replaces the box (single-box policy) while ring
        # is active, the new box also needs ring expansion; when the
        # box is removed (Esc / Delete / Add-to-detected), the stash
        # goes stale and must be invalidated.
        self.viewer.manualPeakRemoved.connect(self._on_manual_peak_removed)
        self.viewer.manualPeakAdded.connect(self._on_manual_peak_added)

    def _hide_stale_dock_tab_bars(self) -> None:
        """Hide ghost dock tab bars.

        Qt quirk: tabifying the Scan-tracking dock into the existing
        Profiles/Peaks bottom group can leave the group's OLD two-tab
        bar behind as a second, stale ``QTabBar`` — painted into the
        file-browser corner of the window. The ghost's signature is a
        MainWindow-child dock tab bar whose tab set is a STRICT subset
        of another dock tab bar's (the real bar of the same group);
        distinct dock groups have disjoint tab sets and never match.
        Ran on show and after every dock re-tabify; a no-op when Qt
        behaves.
        """
        from PySide6.QtWidgets import QTabBar

        infos = [
            (tb, {tb.tabText(i) for i in range(tb.count())})
            for tb in self.findChildren(QTabBar)
            if tb.parent() is self
        ]
        for tb, labs in infos:
            if labs and any(
                labs < other for bar, other in infos if bar is not tb
            ):
                tb.hide()

    def _update_title(self) -> None:
        if self.session is None:
            self.setWindowTitle(APP_NAME)
            self._update_status_file()
            return
        marker = "*" if self.session.dirty else ""
        self.setWindowTitle(
            f"{self.session.original_path.name}{marker} — {APP_NAME}"
        )
        self._update_status_file()

    def _build_status_bar(self) -> None:
        """Permanent status-bar widgets: file / entry / frame / pipeline + cursor.

        Each label lives in the status bar's permanent-widget slot so
        Qt's transient ``showMessage`` calls (PNG / CSV export confirmations)
        still render correctly alongside them — Qt clears the transient
        message after its timeout but leaves the permanent labels alone.
        """
        sb = self.statusBar()
        self._sb_file = QLabel("no file")
        self._sb_entry = QLabel("")
        self._sb_frame = QLabel("")
        self._sb_pipeline = QLabel("idle")
        self._sb_cursor = QLabel("")
        for w in (self._sb_file, self._sb_entry, self._sb_frame,
                  self._sb_pipeline, self._sb_cursor):
            # Light separation so the eye can scan the row. The divider
            # colour and padding come from the skin, which is why this
            # is a role tag rather than a stylesheet: the old hardcoded
            # #444 stayed dark-grey in the light theme. The file name is
            # the one field that says *what you are looking at*, so it
            # keeps the full text colour while the rest read as context.
            w.setProperty(
                "role", "sb-cell-active" if w is self._sb_file else "sb-cell")
            sb.addPermanentWidget(w)
        # The cursor readout is the chattiest widget; let it stretch
        # so values don't truncate, others stay tight.
        self._sb_cursor.setMinimumWidth(360)
        self.viewer.cursorMoved.connect(self._on_status_cursor_moved)
        self._status_cursor_visible = True
        # Bottom-left, non-modal open-progress: a small busy (indeterminate)
        # bar + a stage label, shown while a file loads in the background
        # (copy + first frame warmed off the GUI thread by CopyWorker). The
        # bar is indeterminate on purpose: the open no longer resolves the
        # external links, so there is no granular per-entry progress to
        # show — a continuously marching bar reads as "smoothly loading"
        # during the (GUI-responsive) worker phase, whereas a determinate
        # bar would just jump between a couple of coarse stages. The
        # worker's ``progress`` ticks drive the LABEL ("Copying file",
        # "Loading first entry"). Non-modal so the window stays usable; the
        # file only appears in the browser once the load finishes
        # (_on_open_finished). addWidget puts these on the left, before the
        # transient showMessage text.
        self._sb_open_label = QLabel("")
        self._sb_open_bar = skin_progress(QProgressBar())
        self._sb_open_bar.setRange(0, 0)          # indeterminate "busy" march
        self._sb_open_bar.setMaximumWidth(110)
        self._sb_open_bar.setTextVisible(False)
        self._sb_open_label.hide()
        self._sb_open_bar.hide()
        sb.addWidget(self._sb_open_label)
        sb.addWidget(self._sb_open_bar)
        # Separate determinate bar for the chunked file-browser fill
        # ("Browser: n/N files"). Its OWN widget pair on purpose: the
        # open bar above is driven concurrently by the async first-image
        # load ("Loading <entry>…" → dismissed on arrival), so sharing
        # it would let that dismiss hide an still-running browser fill.
        self._sb_tree_label = QLabel("")
        self._sb_tree_bar = skin_progress(QProgressBar())
        self._sb_tree_bar.setRange(0, 100)
        self._sb_tree_bar.setMaximumWidth(110)
        self._sb_tree_bar.setTextVisible(False)
        self._sb_tree_label.hide()
        self._sb_tree_bar.hide()
        sb.addWidget(self._sb_tree_label)
        sb.addWidget(self._sb_tree_bar)

    def _update_status_file(self) -> None:
        if self.session is None:
            self._sb_file.setText("no file")
            return
        marker = "*" if self.session.dirty else ""
        self._sb_file.setText(
            f"{self.session.original_path.name}{marker}"
        )

    def _active_raw_frame_for_calibration(self):
        """Return a 2D ndarray to seed the pyFAI calibration dialog.

        For multi-frame raw scans (the typical calibrant case —
        LaB6 / Si / CeO2 measured as a short scan to boost ring
        statistics) this returns the per-pixel mean across all
        frames so faint outer rings come out of the noise. For a
        single-frame stack the lone frame is returned unchanged.

        Returns None when no raw session is active, when the
        viewer is in NeXus mode, or when the raw stack hasn't been
        populated yet — the dialog then opens with an empty image
        slot and the user can browse to a file from inside it.

        Lazy stacks (``LazyRawStack``) are averaged over at most the
        first 20 frames, read one at a time — a calibrant scan is
        short, and materializing an arbitrary beamtime stack here
        would re-introduce the multi-GB GUI-thread read the lazy
        path exists to avoid.
        """
        try:
            stack = getattr(self.viewer, "_raw_image_stack", None)
        except Exception:
            logger.debug("suppressed exception in MainWindow._active_raw_frame_for_calibration", exc_info=True)
            return None
        if stack is None:
            return None
        if not isinstance(stack, np.ndarray):
            try:
                n = int(stack.shape[0])
                if n == 0:
                    return None
                if n == 1:
                    return np.asarray(stack[0])
                take = min(n, 20)
                acc = np.zeros(stack.shape[1:], dtype=np.float64)
                for i in range(take):
                    acc += stack[i]
                return acc / take
            except Exception:
                logger.debug("suppressed exception in MainWindow._active_raw_frame_for_calibration", exc_info=True)
                return None
        try:
            arr = np.asarray(stack)
            if arr.ndim != 3 or arr.shape[0] == 0:
                # Defensive: viewer should always hand back a 3D
                # stack in raw mode, but if something upstream
                # changes that contract fall back to whatever the
                # current-frame index points at.
                idx = int(self.viewer.current_frame)
                if 0 <= idx < arr.shape[0]:
                    return arr[idx]
                return arr[0] if arr.shape[0] else None
            if arr.shape[0] == 1:
                return arr[0]
            # Mean in float64 to keep the average stable for high-
            # dynamic-range detector data; pyFAI's image model
            # accepts arbitrary numeric dtypes.
            return arr.mean(axis=0, dtype=np.float64)
        except Exception:
            logger.debug("suppressed exception in MainWindow._active_raw_frame_for_calibration", exc_info=True)
            return None

    def _update_status_entry(self) -> None:
        entry = self.entry_combo.currentText() if hasattr(self, "entry_combo") else ""
        self._sb_entry.setText(entry or "")

    def _update_status_frame(self) -> None:
        n = getattr(self.viewer, "n_frames", 0)
        if n <= 0:
            self._sb_frame.setText("")
            return
        cur = int(getattr(self.viewer, "current_frame", 0))
        # Frames are 0-indexed everywhere else in the GUI (peak rows,
        # NeXus group keys), so the denominator is the max index
        # ``n - 1``, not the count. 17-frame stack → "frame 16 / 16"
        # at the end. Single-frame entries elide the "/ total" since
        # there's no navigation possible.
        if n == 1:
            self._sb_frame.setText(f"frame {cur}")
        else:
            self._sb_frame.setText(f"frame {cur} / {n - 1}")

    def _update_status_pipeline(self, command=None, *, running: bool) -> None:
        if not running:
            self._sb_pipeline.setText("idle")
            # Drop any progress tail from the previous run so a stale
            # "3/12 frames" counter doesn't haunt the status bar after
            # an op finishes.
            self._pipe_progress_tail = ""
            return
        if command is None:
            self._sb_pipeline.setText("running…")
            return
        op = command.op_name if hasattr(command, "op_name") else str(command)
        entry = command.kwargs.get("entry") if hasattr(command, "kwargs") else None
        # Fold in the entry-queue position and the most recent
        # ``frameProgress`` tail when each is known. Multi-entry runs
        # get "· entry K/N"; multi-frame runs get "· K/N frames";
        # single-of-both contributes nothing.
        head = f"running: {op} on {entry}" if entry else f"running: {op}"
        entry_tail = ""
        if getattr(self, "_entry_queue_total", 0) > 1:
            entry_tail = (
                f" · entry {self._entry_queue_pos}/{self._entry_queue_total}"
            )
        frame_tail = getattr(self, "_pipe_progress_tail", "")
        self._sb_pipeline.setText(head + entry_tail + frame_tail)

    def _on_pipeline_frame_progress(
        self, done: int, total: int, op_name: str, entry: str
    ) -> None:
        """Mirror ``PipelineWorker.frameProgress`` into the status bar.

        Single-frame and indeterminate runs (``total <= 1``) clear the
        tail so the existing "running: op on entry" remains unadorned.
        Skips the status-bar repaint when the tail string is unchanged
        from the last emit — a fast pipeline can fire many
        ``frameProgress`` signals per second and an unchanged
        ``setText`` still schedules a paint event.
        """
        # Interpolate-track chain: drive the loading dialog with the
        # SAME per-frame ticks the Pipeline panel's bar shows, summed
        # across the chain's commands (fill + re-match passes), so the
        # two never disagree and the dialog no longer idles at ~95%.
        chain = self._interp_chain
        cmd = getattr(self, "_pipe_command", None)
        dlg = self._track_progress_dialog
        if (
            chain is not None
            and dlg is not None
            and cmd is not None
            and id(cmd) in chain["ticks"]
        ):
            ticks = chain["ticks"][id(cmd)]
            overall = chain["base"] + min(int(done), ticks)
            label = chain.get("label", "Interpolate track")
            if op_name in ("interpolate_tracks", "inject_fitted_peaks"):
                dlg.setLabelText(
                    f"{label}: fitting injected boxes… "
                    f"({min(int(done), ticks)}/{ticks} frames)"
                )
            elif op_name == "run_matching":
                dlg.setLabelText(
                    f"{label}: matching "
                    f"{cmd.kwargs.get('peaks_type', 'segments')}… "
                    f"({min(int(done), ticks)}/{ticks} frames)"
                )
            # setValue MUST come last: on a window-modal QProgressDialog
            # it runs processEvents(), which can deliver this command's
            # queued opFinished re-entrantly — completing the chain and
            # tearing the dialog down (``_track_progress_dialog`` is
            # None again on return). Nothing may touch the dialog after
            # this call.
            dlg.setValue(int(round(100 * overall / chain["total"])))

        # If that re-entrant opFinished drained the queue, the status
        # bar already reads "idle" — do not resurrect a stale
        # "running: …" line for a command that is finished.
        if self._pipe_thread is None and not self._pipeline_queue:
            return

        if op_name == "track_peaks":
            # Busy (indeterminate) in the panel — no frame count to show.
            new_tail = " · tracking…"
        elif total <= 1:
            new_tail = ""
        else:
            new_tail = f" · {done}/{total} frames"
        if getattr(self, "_pipe_progress_tail", "") == new_tail:
            return
        self._pipe_progress_tail = new_tail
        # Re-render the status line so the new tail is visible
        # immediately. Reuse the existing op + entry from the in-flight
        # command rather than rebuilding here.
        cmd = getattr(self, "_pipe_command", None)
        self._update_status_pipeline(cmd, running=True)

    def _on_status_cursor_moved(self, info) -> None:
        if not self._status_cursor_visible:
            self._sb_cursor.setText("")
            return
        if not info:
            self._sb_cursor.setText("")
            return
        mode = info.get("mode")
        inten = info.get("intensity", float("nan"))
        inten_str = "—" if inten != inten else f"{inten:.3g}"  # NaN check
        if mode == "pixel":
            self._sb_cursor.setText(
                f"row={info['row']}, col={info['col']}, I={inten_str}"
            )
        elif mode == "cartesian":
            self._sb_cursor.setText(
                f"q_xy={info['q_xy']:.3f}, q_z={info['q_z']:.3f}, I={inten_str}"
            )
        elif mode == "polar":
            self._sb_cursor.setText(
                f"r={info['r']:.3f}, θ={info['theta']:.1f}°, I={inten_str}"
            )
        else:
            self._sb_cursor.setText("")

    def _set_cursor_readout_visible(self, visible: bool) -> None:
        self._status_cursor_visible = bool(visible)
        self._sb_cursor.setVisible(self._status_cursor_visible)

    def _update_actions(self) -> None:
        has_session = self.session is not None
        # Save/Save As only apply to NeXus sessions — raw sessions have no
        # writable temp copy. Close still works either way.
        is_nexus = has_session and self.session.kind == "nexus"
        self.action_save.setEnabled(is_nexus)
        self.action_save_as.setEnabled(is_nexus)
        self.action_close_file.setEnabled(has_session)
