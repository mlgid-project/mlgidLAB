"""Phase-tracking views window (non-modal, lazily built, kept alive).

Renders one ``phase_tracking.TrackingPayload`` — the exact arrays
mlgidBASE's ``track_peaks`` computed — as four interactive pyqtgraph
tabs, following the ideas prototyped in the peak_matching_frames
notebook:

1. **Trajectories** — per-track colored trace of a selectable axis
   (|q| / angle / amplitude / q_xy / q_z) vs frame number (upstream's
   "evolution" figure, interactive). Clicking a point asks the host to
   jump the image viewer to that frame.
2. **q-map overlay** — per-track trajectories and mean-position markers
   in (q_xy, q_z), optionally over the nan-mean Cartesian image of a
   chosen frame window (worker-computed).
3. **Amplitude evolution** — per-track amplitude vs frame,
   median-smoothed and nanmax-normalized like the notebook.
4. **Radial waterfall** — frame × |q| intensity heatmap of the whole
   scan (worker-computed, cached per entry), independent of tracking.

The window never touches the tracking math: the payload is pushed by
the host, and the two image aids are computed by
``workers.ScanProfileWorker`` on QThreads owned here (read-only, own
h5py handles). "Save official mlgidBASE figures…" emits
``saveFiguresRequested`` for the host to run upstream's real matplotlib
export on the GUI thread.
"""
from __future__ import annotations

import csv
import logging
import os

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mlgidlab.phase_tracking import (
    AXIS_LABELS,
    AXIS_NAMES,
    TrackingPayload,
    smooth_normalize_amplitude,
)
from mlgidlab.workers import ScanProfileWorker

logger = logging.getLogger(__name__)


# Group key for tracks with no matched phase.
_UNMATCHED = "— unmatched —"


def _track_pen(k: int, n_tracks: int):
    """Distinct per-track color, cycling hues (tab20-flavored)."""
    return pg.intColor(k, hues=max(9, min(n_tracks, 20)), maxValue=255)


def _log_display(data: np.ndarray) -> np.ndarray:
    """log10 view of an intensity array; non-positive/NaN -> NaN."""
    out = np.full(data.shape, np.nan, dtype=np.float32)
    positive = np.isfinite(data) & (data > 0)
    out[positive] = np.log10(data[positive], dtype=np.float32)
    return out


class _ExportDialog(QDialog):
    """Pick which views to export and — for images — how to style them.

    ``views`` is ``[(key, label, enabled, tooltip), ...]``; disabled
    entries stay listed (with the tooltip saying what is missing) so
    the user learns what CAN be exported. ``structures`` (image mode)
    is ``[(key, label, css_color, checked), ...]`` — the initial check
    state mirrors the window's structure toggles, but the dialog's
    selection is authoritative for the export (WYSIWYG start, then
    override). OK is enabled only while at least one view is ticked.
    """

    _PEN_STYLES = (
        ("Solid", Qt.PenStyle.SolidLine),
        ("Dashed", Qt.PenStyle.DashLine),
        ("Dotted", Qt.PenStyle.DotLine),
        ("Dash-dot", Qt.PenStyle.DashDotLine),
    )

    def __init__(
        self, parent, mode: str, views, structures=(), current_key=None,
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self.setWindowTitle(
            "Export plot images" if mode == "image"
            else "Export plot data (CSV)"
        )
        layout = QVBoxLayout(self)
        views_box = QGroupBox("Plots (one file each)", self)
        views_lay = QVBoxLayout(views_box)
        self._view_checks: dict = {}
        for key, label, enabled, tip in views:
            chk = QCheckBox(label, views_box)
            chk.setEnabled(enabled)
            if tip:
                chk.setToolTip(tip)
            chk.setChecked(enabled and key == current_key)
            chk.toggled.connect(self._update_ok)
            views_lay.addWidget(chk)
            self._view_checks[key] = chk
        layout.addWidget(views_box)
        note = QLabel(
            "Files are written next to the chosen name, suffixed per "
            "plot (e.g. <name>_trajectories). Existing files are "
            "overwritten.", self,
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self._struct_checks: dict = {}
        if mode == "image":
            if structures:
                struct_box = QGroupBox("Structures to include", self)
                struct_lay = QVBoxLayout(struct_box)
                for key, label, color, checked in structures:
                    chk = QCheckBox(label, struct_box)
                    chk.setChecked(bool(checked))
                    chk.setStyleSheet(f"color: {color};")
                    struct_lay.addWidget(chk)
                    self._struct_checks[key] = chk
                layout.addWidget(struct_box)
            style_box = QGroupBox("Style", self)
            form = QFormLayout(style_box)
            self.line_width = QDoubleSpinBox(style_box)
            self.line_width.setRange(0.5, 10.0)
            self.line_width.setSingleStep(0.5)
            self.line_width.setValue(1.0)
            form.addRow("Line width", self.line_width)
            self.line_style = QComboBox(style_box)
            for name, pen_style in self._PEN_STYLES:
                self.line_style.addItem(name, userData=pen_style)
            form.addRow("Line style", self.line_style)
            self.marker_size = QSpinBox(style_box)
            self.marker_size.setRange(0, 20)
            self.marker_size.setValue(4)
            self.marker_size.setSpecialValueText("hidden")
            form.addRow("Marker size", self.marker_size)
            self.image_width = QSpinBox(style_box)
            self.image_width.setRange(400, 8000)
            self.image_width.setSingleStep(200)
            self.image_width.setValue(1600)
            self.image_width.setSuffix(" px")
            self.image_width.setToolTip(
                "PNG output width in pixels (height keeps the plot's "
                "aspect ratio). Ignored for SVG, which is scale-free."
            )
            form.addRow("Image width", self.image_width)
            self.white_bg = QCheckBox("White background", style_box)
            self.white_bg.setToolTip(
                "Render on white with dark axes (publication style) "
                "instead of the current theme."
            )
            form.addRow("", self.white_bg)
            style_note = QLabel(
                "Line/marker styling applies to the curve plots; the "
                "waterfall and mean-image heatmaps are unaffected.",
                style_box,
            )
            style_note.setWordWrap(True)
            form.addRow(style_note)
            layout.addWidget(style_box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._update_ok()

    def _update_ok(self) -> None:
        self._ok.setEnabled(
            any(c.isChecked() for c in self._view_checks.values())
        )

    def selected_views(self) -> list:
        return [k for k, c in self._view_checks.items() if c.isChecked()]

    def selected_structures(self):
        """Structure keys to draw, or None when there is no structure
        group (no matching yet -> nothing to filter)."""
        if not self._struct_checks:
            return None
        return {k for k, c in self._struct_checks.items() if c.isChecked()}

    def style(self) -> dict:
        return {
            "line_width": float(self.line_width.value()),
            "line_style": self.line_style.currentData(),
            "marker_size": int(self.marker_size.value()),
            "width_px": int(self.image_width.value()),
            "white_bg": self.white_bg.isChecked(),
        }


class PhaseViewsWindow(QMainWindow):
    """Four interactive views over one tracking payload."""

    frameJumpRequested = Signal(int)
    # (base_path, axis) — host runs upstream track_peaks with
    # save_fig=True on the GUI thread (silx detached).
    saveFiguresRequested = Signal(str, str)

    def __init__(
        self, parent: QWidget | None = None, busy_probe=None
    ) -> None:
        """``busy_probe``: optional callable returning True while the
        host runs a pipeline command. pygid opens the NeXus file in
        r+ mode even for reads, so a view worker's long-lived read
        handle and a pipeline run must never overlap — the probe lets
        this window refuse to start a worker mid-run (the host guards
        the opposite direction)."""
        super().__init__(parent)
        self._busy_probe = busy_probe
        self.setWindowTitle("Phase tracking views")
        self.resize(900, 640)

        self._payload: TrackingPayload | None = None
        self._file_path = None
        self._entry: str | None = None
        self._n_frames = 0
        # Worker-computed caches, keyed by (file_path, entry).
        self._waterfall: dict | None = None
        self._mean_image: dict | None = None
        self._worker_thread: QThread | None = None
        self._worker: ScanProfileWorker | None = None
        # Phase identity for the q-map + amplitude grouping:
        # {track_index: [cif, ...]} (dominant first) + a stable colour
        # per unique CIF. ``_struct_checks`` maps each structure key
        # (CIF or _UNMATCHED) to its show/hide checkbox; ``_amp_labels``
        # holds the grouped-view TextItems for teardown.
        self._track_phases: dict = {}
        self._phase_colors: dict = {}
        self._struct_checks: dict = {}
        self._amp_labels: list = []
        # Interquartile-band items of the amplitude tab's median mode.
        # FillBetweenItems are NOT pyqtgraph data items, so
        # ``_clear_overlays`` misses them — torn down explicitly.
        self._amp_fills: list = []
        # Track indices that are rings — drawn on the q-map as dashed
        # quarter-circle arcs at their mean |q|, not as point trajectories.
        self._ring_tracks: set = set()
        # Temporary render overrides applied while exporting images
        # (structure subset, line width/style, marker size). Empty in
        # normal operation; the style helpers consult it.
        self._export_overrides: dict = {}
        # Last theme applied by the host — restored after a white-
        # background export.
        self._theme_colors = (
            pg.getConfigOption("background"),
            pg.getConfigOption("foreground"),
        )

        # Central layout: a window-wide structure-selection bar above the
        # tabs (so the per-CIF show/hide toggles govern every phase view —
        # trajectories, q-map, amplitude — from any tab), then the tabs.
        central = QWidget(self)
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        self._struct_toggle_widget = QWidget(central)
        self._struct_toggle_layout = QHBoxLayout(self._struct_toggle_widget)
        self._struct_toggle_layout.setContentsMargins(6, 2, 6, 2)
        self._struct_toggle_layout.addWidget(QLabel("Structures:", central))
        self._struct_toggle_layout.addStretch(1)
        self._struct_toggle_widget.setVisible(False)
        central_layout.addWidget(self._struct_toggle_widget)
        self._tabs = QTabWidget(central)
        central_layout.addWidget(self._tabs)
        self.setCentralWidget(central)

        self._build_trajectories_tab()
        self._build_qmap_tab()
        self._build_amplitude_tab()
        self._build_waterfall_tab()

        act_data = QAction("Export plot data (CSV)…", self)
        act_data.setToolTip(
            "Write the numbers behind the selected plots to CSV files "
            "(one per plot) for further analysis or custom plotting."
        )
        act_data.triggered.connect(self._on_export_data)
        act_images = QAction("Export plot images…", self)
        act_images.setToolTip(
            "Render the selected plots to PNG/SVG, with export styling: "
            "structure subset, line width/style, marker size, output "
            "width, white background."
        )
        act_images.triggered.connect(self._on_export_images)
        act_save = QAction("Save official mlgidBASE figures…", self)
        act_save.setToolTip(
            "Run mlgidBASE track_peaks with save_fig=True: writes the "
            "upstream *_qxy_qz / *_evolution matplotlib figures."
        )
        act_save.triggered.connect(self._on_save_figures)
        toolbar = self.addToolBar("Export")
        toolbar.setObjectName("PhaseViewsExportBar")
        toolbar.setMovable(False)
        toolbar.addAction(act_data)
        toolbar.addAction(act_images)
        toolbar.addSeparator()
        toolbar.addAction(act_save)

        self.statusBar().showMessage("No tracking results yet.")

    # --- Tab construction ---

    def _controls_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        return row

    def _make_plot(self, x_label: str, y_label: str) -> pg.PlotWidget:
        plot = pg.PlotWidget(self)
        plot.setLabel("bottom", x_label)
        plot.setLabel("left", y_label)
        plot.showGrid(x=True, y=True, alpha=0.2)
        return plot

    def _build_trajectories_tab(self) -> None:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        controls = self._controls_row()
        controls.addWidget(QLabel("Axis:", tab))
        self.axis_combo = QComboBox(tab)
        for name in AXIS_NAMES:
            self.axis_combo.addItem(AXIS_LABELS[name], userData=name)
        self.axis_combo.setCurrentIndex(AXIS_NAMES.index("radius"))
        self.axis_combo.currentIndexChanged.connect(
            lambda _i: self._refresh_trajectories()
        )
        controls.addWidget(self.axis_combo)
        controls.addWidget(QLabel("Colour:", tab))
        self.traj_color = QComboBox(tab)
        self.traj_color.addItem("fitted (per track)", userData="fitted")
        self.traj_color.addItem("matched (per phase)", userData="matched")
        self.traj_color.setToolTip(
            "'fitted' gives each track its own colour; 'matched' colours "
            "tracks by the matched crystal phase (CIF) they belong to, so "
            "tracks of the same phase share a colour (unmatched grey). "
            "Needs Matching to have run."
        )
        self.traj_color.model().item(1).setEnabled(False)  # 'matched' off until phases
        self.traj_color.currentIndexChanged.connect(
            lambda _i: self._refresh_trajectories()
        )
        controls.addWidget(self.traj_color)
        controls.addStretch(1)
        hint = QLabel("Click a point to jump the viewer to that frame.", tab)
        controls.addWidget(hint)
        layout.addLayout(controls)
        self.traj_plot = self._make_plot("Frame #", AXIS_LABELS["radius"])
        layout.addWidget(self.traj_plot)
        self._tabs.addTab(tab, "Trajectories")

    def _build_qmap_tab(self) -> None:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        controls = self._controls_row()
        controls.addWidget(QLabel("Mean image frames", tab))
        self.qmap_lo = QSpinBox(tab)
        self.qmap_hi = QSpinBox(tab)
        for spin in (self.qmap_lo, self.qmap_hi):
            spin.setRange(0, 0)
            controls.addWidget(spin)
        self.btn_mean_image = QPushButton("Compute mean image", tab)
        self.btn_mean_image.setToolTip(
            "Average the Cartesian frames in the chosen window "
            "(nan-aware) and draw it under the trajectories."
        )
        self.btn_mean_image.clicked.connect(self._on_compute_mean_image)
        controls.addWidget(self.btn_mean_image)
        self.qmap_log = QCheckBox("Log intensity", tab)
        self.qmap_log.setChecked(True)
        self.qmap_log.toggled.connect(lambda _c: self._refresh_qmap())
        controls.addWidget(self.qmap_log)
        self.qmap_phase = QCheckBox("Color by matched phase", tab)
        self.qmap_phase.setToolTip(
            "Colour each track by the matched crystal phase (CIF) its "
            "tracked peaks belong to, with a legend; unmatched tracks "
            "are grey. Needs Matching to have run."
        )
        self.qmap_phase.setEnabled(False)
        self.qmap_phase.toggled.connect(lambda _c: self._refresh_qmap())
        controls.addWidget(self.qmap_phase)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.qmap_plot = self._make_plot("q_xy (Å⁻¹)", "q_z (Å⁻¹)")
        self.qmap_plot.getPlotItem().getViewBox().setAspectLocked(True)
        self.qmap_image = pg.ImageItem()
        self.qmap_image.setZValue(-10)
        self._apply_image_cmap(self.qmap_image)
        self.qmap_plot.addItem(self.qmap_image)
        self._qmap_legend = self.qmap_plot.getPlotItem().addLegend(
            offset=(-10, 10)
        )
        self._qmap_legend.setVisible(False)
        layout.addWidget(self.qmap_plot)
        self._tabs.addTab(tab, "q-map overlay")

    def _build_amplitude_tab(self) -> None:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        controls = self._controls_row()
        controls.addWidget(QLabel("Smoothing window", tab))
        self.amp_window = QSpinBox(tab)
        self.amp_window.setRange(1, 99)
        self.amp_window.setSingleStep(2)
        self.amp_window.setValue(7)
        self.amp_window.setToolTip(
            "Median-filter size over each track's frame-ordered "
            "amplitudes (1 = raw)."
        )
        self.amp_window.valueChanged.connect(
            lambda _v: self._refresh_amplitude()
        )
        controls.addWidget(self.amp_window)
        self.amp_normalize = QCheckBox("Normalize per track", tab)
        self.amp_normalize.setChecked(True)
        self.amp_normalize.toggled.connect(
            lambda _c: self._refresh_amplitude()
        )
        controls.addWidget(self.amp_normalize)
        self.amp_group = QCheckBox("Group by structure", tab)
        self.amp_group.setToolTip(
            "Stack the tracks by matched crystal phase (CIF): each "
            "structure at its own vertical offset (amplitudes are "
            "normalized, so the y-scale is arbitrary), coloured by "
            "phase. Needs Matching to have run."
        )
        self.amp_group.setEnabled(False)
        self.amp_group.toggled.connect(lambda _c: self._refresh_amplitude())
        controls.addWidget(self.amp_group)
        self.amp_median = QCheckBox("Median per structure", tab)
        self.amp_median.setToolTip(
            "Draw ONE curve per matched structure: the per-frame MEDIAN "
            "amplitude across all of the structure's tracks, with a "
            "transparent interquartile band (25th-75th percentile) "
            "around it — instead of every individual track. Respects "
            "the smoothing window and (ungrouped) the normalize toggle; "
            "combines with \"Group by structure\" to stack one median "
            "band per structure. Needs Matching to have run."
        )
        self.amp_median.setEnabled(False)
        self.amp_median.toggled.connect(lambda _c: self._refresh_amplitude())
        controls.addWidget(self.amp_median)
        controls.addStretch(1)
        layout.addLayout(controls)
        # The per-structure show/hide toggles live in a window-wide bar
        # (built in __init__) so they also govern the q-map/trajectories.
        self.amp_plot = self._make_plot(
            "Frame #", "Amplitude (normalized)"
        )
        layout.addWidget(self.amp_plot)
        self._tabs.addTab(tab, "Amplitude evolution")

    def _build_waterfall_tab(self) -> None:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        controls = self._controls_row()
        self.btn_waterfall = QPushButton("Compute waterfall", tab)
        self.btn_waterfall.setToolTip(
            "Resample every frame to polar and stack the angular-mean "
            "radial profiles into a frame × |q| heatmap. Independent "
            "of tracking; cached per entry."
        )
        self.btn_waterfall.clicked.connect(self._on_compute_waterfall)
        controls.addWidget(self.btn_waterfall)
        self.waterfall_log = QCheckBox("Log intensity", tab)
        self.waterfall_log.setChecked(True)
        self.waterfall_log.toggled.connect(
            lambda _c: self._refresh_waterfall()
        )
        controls.addWidget(self.waterfall_log)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.waterfall_plot = self._make_plot("Frame #", "|q| (Å⁻¹)")
        self.waterfall_image = pg.ImageItem()
        self._apply_image_cmap(self.waterfall_image)
        self.waterfall_plot.addItem(self.waterfall_image)
        layout.addWidget(self.waterfall_plot)
        self._tabs.addTab(tab, "Radial waterfall")

    # --- Host API ---

    def set_context(self, file_path, entry: str, n_frames: int) -> None:
        """Point the window at the active session/entry. Invalidates
        the worker-computed caches when the target changed."""
        changed = (file_path, entry) != (self._file_path, self._entry)
        self._file_path = file_path
        self._entry = entry
        self._n_frames = int(n_frames)
        hi = max(0, self._n_frames - 1)
        for spin in (self.qmap_lo, self.qmap_hi):
            spin.setRange(0, hi)
        if changed:
            self.qmap_lo.setValue(0)
            self.qmap_hi.setValue(hi)
            self._waterfall = None
            self._mean_image = None
            self._refresh_qmap()
            self._refresh_waterfall()

    def set_payload(self, payload: TrackingPayload | None) -> None:
        self._payload = payload
        self._refresh_trajectories()
        self._refresh_qmap()
        self._refresh_amplitude()
        if payload is None:
            self.statusBar().showMessage("No tracking results yet.")
        else:
            self.statusBar().showMessage(
                f"{payload.n_tracks} tracks · {payload.n_members} "
                f"members · entry {payload.entry}"
            )

    def set_track_phases(self, mapping: dict) -> None:
        """Install ``{track_index: [cif, ...]}`` (dominant phase first)
        for the q-map "Color by matched phase" overlay.

        Assigns a stable colour per unique CIF, enables the toggle only
        when there are phases (untick + disable otherwise), and
        re-renders the q-map.
        """
        self._track_phases = {int(k): list(v) for k, v in mapping.items()}
        cifs = sorted({c for lst in self._track_phases.values() for c in lst})
        hues = max(9, len(cifs))
        self._phase_colors = {
            cif: pg.intColor(i, hues=hues, maxValue=255)
            for i, cif in enumerate(cifs)
        }
        has_phases = bool(self._track_phases)
        self._rebuild_structure_toggles()
        # Trajectories 'matched' colour option follows phase availability.
        self.traj_color.model().item(1).setEnabled(has_phases)
        if not has_phases and self.traj_color.currentData() == "matched":
            self.traj_color.setCurrentIndex(0)   # triggers _refresh_trajectories
        else:
            self._refresh_trajectories()
        self.amp_group.setEnabled(has_phases)
        self.amp_median.setEnabled(has_phases)
        if not has_phases and self.amp_median.isChecked():
            self.amp_median.setChecked(False)  # triggers _refresh_amplitude
        if not has_phases and self.amp_group.isChecked():
            self.amp_group.setChecked(False)   # triggers _refresh_amplitude
        else:
            self._refresh_amplitude()
        self.qmap_phase.setEnabled(has_phases)
        if not has_phases and self.qmap_phase.isChecked():
            self.qmap_phase.setChecked(False)   # triggers _refresh_qmap
        else:
            self._refresh_qmap()

    def _rebuild_structure_toggles(self) -> None:
        """Rebuild the per-structure show/hide checkboxes from the phase
        map (one per CIF present + 'unmatched' when any track has no
        phase). Checkbox text is tinted with the phase colour so it ties
        to the grouped amplitude bands / q-map legend. Hidden when there
        are no phases."""
        for chk in self._struct_checks.values():
            self._struct_toggle_layout.removeWidget(chk)
            chk.setParent(None)
            chk.deleteLater()
        self._struct_checks = {}
        # Toggles are meaningful only once Matching has identified at
        # least one phase; with no matches every track is "unmatched"
        # and the panel is just noise.
        keys = self._structure_keys() if self._track_phases else []
        for key in keys:
            color = self._structure_color(key)
            label = "unmatched" if key == _UNMATCHED else key
            chk = QCheckBox(label, self._struct_toggle_widget)
            chk.setChecked(True)
            chk.setStyleSheet(f"color: {pg.mkColor(color).name()};")
            chk.toggled.connect(lambda _c: self._on_structure_toggle())
            # Insert before the trailing stretch.
            self._struct_toggle_layout.insertWidget(
                self._struct_toggle_layout.count() - 1, chk
            )
            self._struct_checks[key] = chk
        self._struct_toggle_widget.setVisible(bool(self._struct_checks))

    def _structure_keys(self) -> list:
        """Ordered structure keys across the tracks: matched CIFs
        (sorted) then 'unmatched' if any track has no phase."""
        if self._payload is None:
            return []
        cifs = sorted({
            v[0] for v in self._track_phases.values() if v
        })
        has_unmatched = any(
            not self._track_phases.get(k)
            for k in range(self._payload.n_tracks)
        )
        return cifs + ([_UNMATCHED] if has_unmatched else [])

    def _structure_color(self, key: str):
        return self._phase_colors.get(
            key, pg.mkColor(*self._UNMATCHED_COLOR)
        )

    def _structure_key_of(self, k: int) -> str:
        cifs = self._track_phases.get(k)
        return cifs[0] if cifs else _UNMATCHED

    def _structure_key_visible(self, key: str) -> bool:
        """Whether a structure key is drawn: the export-time structure
        subset wins over the window's toggle bar while an image export
        is rendering."""
        structs = self._export_overrides.get("structures")
        if structs is not None:
            return key in structs
        chk = self._struct_checks.get(key)
        return chk is None or chk.isChecked()

    def _structure_visible(self, k: int) -> bool:
        return self._structure_key_visible(self._structure_key_of(k))

    # --- Export style overrides (image export) ---

    def _curve_pen(self, color, base_width: float = 1.0,
                   emphasized: bool = False):
        """Pen for a data curve, honouring the export overrides: the
        chosen line width (+1 for emphasized curves like the median)
        and line style. Without overrides, exactly the historical pen."""
        ov = self._export_overrides
        if "line_width" in ov:
            width = float(ov["line_width"]) + (1.0 if emphasized else 0.0)
        else:
            width = float(base_width)
        pen = pg.mkPen(color, width=width)
        style = ov.get("line_style")
        if style is not None:
            pen.setStyle(style)
        return pen

    def _emph_width(self, base: float) -> float:
        """Width for emphasized non-line pens (ring arcs, mean-position
        marker outlines) — these keep their own pen STYLE (a ring's
        dash is semantic) but follow the export line width."""
        ov = self._export_overrides
        if "line_width" in ov:
            return float(ov["line_width"]) + 1.0
        return float(base)

    def _symbol_kwargs(self, base_size: int, color) -> dict:
        """Data-point symbol kwargs for ``plot()``: export marker-size
        override applied; size 0 hides the markers entirely."""
        size = int(self._export_overrides.get("marker_size", base_size))
        if size <= 0:
            return {"symbol": None}
        return {
            "symbol": "o", "symbolSize": size,
            "symbolBrush": color, "symbolPen": None,
        }

    def _on_structure_toggle(self) -> None:
        """A per-structure show/hide checkbox changed: refresh every
        phase view that filters tracks by structure."""
        self._refresh_trajectories()
        self._refresh_qmap()
        self._refresh_amplitude()

    def _apply_image_cmap(self, image_item) -> None:
        """Apply the GUI's GIWAXS colormap (matches the main image
        viewer's ``DEFAULT_COLORMAP``) to a heatmap ImageItem, so the
        waterfall / mean image read like the rest of the app instead of
        raw greyscale."""
        from mlgidlab.image_viewer import DEFAULT_COLORMAP

        cmap = None
        for source in ("matplotlib", None):
            try:
                cmap = (
                    pg.colormap.get(DEFAULT_COLORMAP, source=source)
                    if source else pg.colormap.get(DEFAULT_COLORMAP)
                )
            except Exception:
                cmap = None
            if cmap is not None:
                break
        if cmap is not None:
            image_item.setColorMap(cmap)

    def set_ring_tracks(self, ring_tracks) -> None:
        """Mark which track indices are rings (drawn as dashed arcs on
        the q-map). Re-renders the q-map."""
        self._ring_tracks = {int(k) for k in ring_tracks}
        self._refresh_qmap()

    def clear(self) -> None:
        self._ring_tracks = set()
        self.set_track_phases({})
        self.set_payload(None)

    def apply_theme_colors(self, background, foreground) -> None:
        """Runtime theme swap (same contract as the viewer/profile)."""
        self._theme_colors = (background, foreground)
        self._apply_plot_colors(background, foreground)

    def _apply_plot_colors(self, background, foreground) -> None:
        for plot in (self.traj_plot, self.qmap_plot, self.amp_plot,
                     self.waterfall_plot):
            try:
                plot.setBackground(background)
                for axis_name in ("left", "bottom"):
                    axis = plot.getPlotItem().getAxis(axis_name)
                    axis.setPen(foreground)
                    axis.setTextPen(foreground)
            except Exception:
                logger.debug(
                    "suppressed exception in PhaseViewsWindow."
                    "apply_theme_colors", exc_info=True,
                )

    # --- View refreshes ---

    def _clear_overlays(self, plot: pg.PlotWidget, keep=()) -> None:
        for item in list(plot.getPlotItem().listDataItems()):
            plot.removeItem(item)
        # ImageItems are not data items; they persist across refreshes.
        del keep

    def _refresh_trajectories(self) -> None:
        self._clear_overlays(self.traj_plot)
        axis = self.axis_combo.currentData() or "radius"
        self.traj_plot.setLabel("left", AXIS_LABELS[axis])
        payload = self._payload
        if payload is None:
            return
        phase_mode = (
            self.traj_color.currentData() == "matched"
            and bool(self._track_phases)
        )
        values = payload.axis_values(axis)
        for k in range(payload.n_tracks):
            if not self._structure_visible(k):
                continue
            members = payload.track_members(k)
            frames = payload.frame_num[members]
            color = self._qmap_track_color(k, payload.n_tracks, phase_mode)
            curve = self.traj_plot.plot(
                frames, values[members],
                pen=self._curve_pen(color),
                **self._symbol_kwargs(4, color),
            )
            curve.sigPointsClicked.connect(self._on_traj_points_clicked)

    def _on_traj_points_clicked(self, _item, points) -> None:
        if len(points):
            self.frameJumpRequested.emit(int(round(points[0].pos().x())))

    def _refresh_qmap(self) -> None:
        self._clear_overlays(self.qmap_plot)
        mi = self._mean_image
        if mi is not None:
            data = np.asarray(mi["data"], dtype=np.float32)
            shown = _log_display(data) if self.qmap_log.isChecked() else data
            # ImageItem axis order: axis0 -> x. Data is (n_qz, n_qxy),
            # so transpose puts q_xy on x / q_z on y.
            self.qmap_image.setImage(shown.T, autoLevels=True)
            q_xy, q_z = mi["q_xy"], mi["q_z"]
            x0, x1 = float(np.min(q_xy)), float(np.max(q_xy))
            y0, y1 = float(np.min(q_z)), float(np.max(q_z))
            self.qmap_image.setRect(x0, y0, x1 - x0, y1 - y0)
            self.qmap_image.setVisible(True)
        else:
            self.qmap_image.setVisible(False)
        payload = self._payload
        phase_mode = self.qmap_phase.isChecked() and bool(self._track_phases)
        if payload is None:
            self._qmap_legend.clear()
            self._qmap_legend.setVisible(False)
            return
        for k in range(payload.n_tracks):
            if not self._structure_visible(k):
                continue
            color = self._qmap_track_color(k, payload.n_tracks, phase_mode)
            if k in self._ring_tracks:
                # A ring spans the whole azimuth at a fixed |q|, so it
                # reads as a dashed quarter-circle arc (0deg along q_xy
                # to 90deg along q_z) at its mean radius — not a cluster
                # of points.
                r = payload.track_mean("radius", k)
                theta = np.linspace(0.0, np.pi / 2, 100)
                self.qmap_plot.plot(
                    r * np.cos(theta), r * np.sin(theta),
                    pen=pg.mkPen(
                        color, width=self._emph_width(2),
                        style=Qt.PenStyle.DashLine,
                    ),
                )
                continue
            members = payload.track_members(k)
            self.qmap_plot.plot(
                payload.q_xy[members], payload.q_z[members],
                pen=self._curve_pen(color),
            )
            self.qmap_plot.plot(
                [payload.track_mean("q_xy", k)],
                [payload.track_mean("q_z", k)],
                pen=None, symbol="o", symbolSize=10,
                symbolBrush=None,
                symbolPen=pg.mkPen(color, width=self._emph_width(2)),
            )
        if phase_mode:
            self._rebuild_phase_legend(payload)
        else:
            self._qmap_legend.clear()
            self._qmap_legend.setVisible(False)

    # Neutral grey for tracks with no matched phase.
    _UNMATCHED_COLOR = (150, 150, 150)

    def _qmap_track_color(self, k: int, n_tracks: int, phase_mode: bool):
        """Colour for track ``k`` on the q-map: per-CIF phase colour
        (dominant phase, grey if unmatched) when phase mode is on,
        otherwise the per-track cycling colour."""
        if not phase_mode:
            return _track_pen(k, n_tracks)
        cifs = self._track_phases.get(k)
        if cifs:
            return self._phase_colors[cifs[0]]
        return pg.mkColor(*self._UNMATCHED_COLOR)

    def _rebuild_phase_legend(self, payload) -> None:
        """One legend entry per unique dominant CIF present among the
        VISIBLE tracks (+ 'unmatched' if any visible track has no
        phase) — so hiding a structure drops it from the legend too."""
        self._qmap_legend.clear()
        seen: list = []
        has_unmatched = False
        for k in range(payload.n_tracks):
            if not self._structure_visible(k):
                continue
            cifs = self._track_phases.get(k)
            if cifs:
                if cifs[0] not in seen:
                    seen.append(cifs[0])
            else:
                has_unmatched = True
        for cif in seen:
            sample = pg.PlotDataItem(
                pen=pg.mkPen(self._phase_colors[cif], width=2)
            )
            self._qmap_legend.addItem(sample, cif)
        if has_unmatched:
            sample = pg.PlotDataItem(
                pen=pg.mkPen(self._UNMATCHED_COLOR, width=2)
            )
            self._qmap_legend.addItem(sample, "unmatched")
        self._qmap_legend.setVisible(True)

    def _clear_amp_labels(self) -> None:
        for lbl in self._amp_labels:
            self.amp_plot.removeItem(lbl)
        self._amp_labels = []
        for fill in self._amp_fills:
            self.amp_plot.removeItem(fill)
        self._amp_fills = []

    def _amp_median_series(self, tracks, window, normalize):
        """Per-frame MEDIAN + interquartile bounds of the given tracks'
        amplitudes.

        Each track's series goes through the same smoothing window /
        per-track normalization as the individual curves, then a track
        with several same-frame members contributes their mean once.
        Per frame the statistics run over the tracks that HAVE a value
        there (a track absent from a frame does not drag the median
        down). Returns ``(frames, median, q25, q75)`` — the 25th/75th
        percentile bounds collapse onto the median where only one track
        contributes — or ``None`` with no data.
        """
        per_frame: dict = {}
        for k in tracks:
            members = self._payload.track_members(k)
            frames, amps = smooth_normalize_amplitude(
                self._payload.frame_num[members],
                self._payload.amplitude[members],
                window=window, normalize=normalize,
            )
            track_vals: dict = {}
            for f, a in zip(frames, amps):
                track_vals.setdefault(int(f), []).append(float(a))
            for f, vals in track_vals.items():
                per_frame.setdefault(f, []).append(
                    float(np.mean(vals))
                )
        if not per_frame:
            return None
        xs = np.array(sorted(per_frame), dtype=float)
        med = np.array([np.median(per_frame[int(f)]) for f in xs])
        q25 = np.array([np.percentile(per_frame[int(f)], 25) for f in xs])
        q75 = np.array([np.percentile(per_frame[int(f)], 75) for f in xs])
        return xs, med, q25, q75

    def _amp_median_band(self, tracks, window, normalize, baseline, color):
        """Draw one structure's median amplitude curve with a
        transparent interquartile band — 25th to 75th percentile
        (``pg.FillBetweenItem`` between the bound curves, tracked in
        ``_amp_fills`` because fills are not pyqtgraph data items)."""
        series = self._amp_median_series(tracks, window, normalize)
        if series is None:
            return
        xs, med, q25, q75 = series
        qc = pg.mkColor(color)
        band_pen = pg.mkPen(
            (qc.red(), qc.green(), qc.blue(), 0)
        )
        lo = self.amp_plot.plot(xs, q25 + baseline, pen=band_pen)
        hi = self.amp_plot.plot(xs, q75 + baseline, pen=band_pen)
        fill = pg.FillBetweenItem(
            lo, hi,
            brush=pg.mkBrush(qc.red(), qc.green(), qc.blue(), 60),
        )
        self.amp_plot.addItem(fill)
        self._amp_fills.append(fill)
        self.amp_plot.plot(
            xs, med + baseline,
            pen=self._curve_pen(color, base_width=2, emphasized=True),
            **self._symbol_kwargs(4, color),
        )

    def _amp_extreme_lines(self, tracks, window, baseline, color):
        """Thin horizontal reference lines at a structure's amplitude
        MAXIMUM and MINIMUM (grouped view): the extremes of its
        per-frame median series (per-track normalized, like the grouped
        curves), spanning the structure's frame range — so every point
        of the band reads off how far it sits from the structure's own
        max/min."""
        series = self._amp_median_series(tracks, window, True)
        if series is None or not len(series[0]):
            return
        xs, med, _q25, _q75 = series
        span = [float(xs.min()), float(xs.max())]
        pen = pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine)
        for y in (float(np.nanmax(med)), float(np.nanmin(med))):
            self.amp_plot.plot(span, [y + baseline] * 2, pen=pen)

    def _amp_curve(self, k, window, normalize, baseline, color):
        members = self._payload.track_members(k)
        frames, amps = smooth_normalize_amplitude(
            self._payload.frame_num[members],
            self._payload.amplitude[members],
            window=window, normalize=normalize,
        )
        self.amp_plot.plot(
            frames, amps + baseline,
            pen=self._curve_pen(color),
            **self._symbol_kwargs(3, color),
        )

    def _refresh_amplitude(self) -> None:
        self._clear_overlays(self.amp_plot)
        self._clear_amp_labels()
        left_axis = self.amp_plot.getPlotItem().getAxis("left")
        payload = self._payload
        grouped = self.amp_group.isChecked() and bool(self._track_phases)
        # Grouped bands are always per-track normalized (the offset needs
        # a fixed 0..1 range), so the manual normalize toggle is moot.
        self.amp_normalize.setEnabled(not grouped)
        if payload is None:
            left_axis.setStyle(showValues=True)
            self.amp_plot.setLabel("left", "Amplitude")
            return
        window = int(self.amp_window.value())
        median = self.amp_median.isChecked() and bool(self._track_phases)

        if grouped:
            # Stack structures: each CIF group at its own vertical
            # offset, normalized per track (y-scale arbitrary), coloured
            # by phase, with a text label per band. Hidden structures
            # (unticked) drop out and the remaining bands re-pack. In
            # median mode each band is ONE median curve + IQR band
            # instead of the group's individual tracks.
            left_axis.setStyle(showValues=False)
            self.amp_plot.setLabel("left", "Structure (stacked, arb.)")
            groups: dict = {}
            for k in range(payload.n_tracks):
                groups.setdefault(self._structure_key_of(k), []).append(k)
            keys = [
                key for key in self._structure_keys()
                if self._structure_key_visible(key)
            ]
            x0 = float(payload.frame_num.min()) if payload.n_members else 0.0
            for g, key in enumerate(keys):
                baseline = g * self._AMP_GROUP_STEP
                color = self._structure_color(key)
                tracks = groups.get(key, [])
                if median:
                    self._amp_median_band(
                        tracks, window, True, baseline, color
                    )
                else:
                    for k in tracks:
                        self._amp_curve(k, window, True, baseline, color)
                # Thin dashed reference lines at this structure's
                # median max / min.
                self._amp_extreme_lines(tracks, window, baseline, color)
                label = pg.TextItem(
                    text="unmatched" if key == _UNMATCHED else key,
                    color=color, anchor=(0.0, 0.5),
                )
                label.setPos(x0, baseline + 0.5)
                self.amp_plot.addItem(label)
                self._amp_labels.append(label)
            return

        # Ungrouped: overlaid curves (structure toggles still filter
        # which tracks show). Median mode collapses each structure's
        # tracks into one phase-coloured median curve + IQR band.
        left_axis.setStyle(showValues=True)
        normalize = self.amp_normalize.isChecked()
        self.amp_plot.setLabel(
            "left",
            "Amplitude (normalized)" if normalize
            else AXIS_LABELS["amplitude"],
        )
        if median:
            groups = {}
            for k in range(payload.n_tracks):
                groups.setdefault(self._structure_key_of(k), []).append(k)
            for key in self._structure_keys():
                if not self._structure_key_visible(key):
                    continue
                self._amp_median_band(
                    groups.get(key, []), window, normalize, 0.0,
                    self._structure_color(key),
                )
            return
        for k in range(payload.n_tracks):
            if not self._structure_visible(k):
                continue
            self._amp_curve(
                k, window, normalize, 0.0, _track_pen(k, payload.n_tracks)
            )

    # Vertical spacing between stacked structure bands (each track is
    # normalized to 0..1, so 1.3 leaves a ~0.3 gap).
    _AMP_GROUP_STEP = 1.3

    def _refresh_waterfall(self) -> None:
        wf = self._waterfall
        if wf is None:
            self.waterfall_image.setVisible(False)
            return
        data = np.asarray(wf["data"], dtype=np.float32)
        shown = (
            _log_display(data)
            if self.waterfall_log.isChecked()
            else data
        )
        # data is (n_frames, n_radius): axis0 (frame) -> x, axis1 -> y.
        self.waterfall_image.setImage(shown, autoLevels=True)
        radius = np.asarray(wf["radius"], dtype=float)
        r0, r1 = float(radius.min()), float(radius.max())
        self.waterfall_image.setRect(0.0, r0, data.shape[0], r1 - r0)
        self.waterfall_image.setVisible(True)

    # --- Worker plumbing (waterfall / mean image) ---

    @property
    def worker_active(self) -> bool:
        """True while a waterfall / mean-image worker holds its read
        handle on the file (the host refuses to start pipeline runs
        during that window — see ``busy_probe``)."""
        return self._worker_thread is not None

    def _start_worker(self, worker: ScanProfileWorker) -> bool:
        if self._worker_thread is not None:
            self.statusBar().showMessage(
                "A view computation is already running…", 4000
            )
            return False
        if self._busy_probe is not None and self._busy_probe():
            self.statusBar().showMessage(
                "Wait for the running pipeline command to finish — the "
                "file cannot be read while mlgidbase holds it.", 6000
            )
            return False
        self.btn_waterfall.setEnabled(False)
        self.btn_mean_image.setEnabled(False)
        thread = QThread(self)
        worker.moveToThread(thread)
        worker.progress.connect(self._on_worker_progress)
        worker.finished.connect(self._on_worker_finished)
        thread.started.connect(worker.run)
        self._worker_thread, self._worker = thread, worker
        thread.start()
        return True

    def _on_worker_progress(self, done: int, total: int) -> None:
        self.statusBar().showMessage(f"Computing… frame {done}/{total}")

    def _on_worker_finished(self, result: object, error) -> None:
        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait()
            self._worker_thread.deleteLater()
            self._worker_thread = None
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        self.btn_waterfall.setEnabled(True)
        self.btn_mean_image.setEnabled(True)
        if error is not None:
            self.statusBar().showMessage(f"View computation failed: {error}")
            QMessageBox.warning(self, "Phase tracking views", str(error))
            return
        if not isinstance(result, dict):
            return
        if result.get("kind") == "waterfall":
            self._waterfall = result
            self._refresh_waterfall()
        elif result.get("kind") == "mean_image":
            self._mean_image = result
            self._refresh_qmap()
        self.statusBar().showMessage("Done.", 3000)

    def _on_compute_waterfall(self) -> None:
        if self._file_path is None or not self._entry:
            return
        self._start_worker(ScanProfileWorker(
            self._file_path, self._entry, "waterfall",
        ))

    def _on_compute_mean_image(self) -> None:
        if self._file_path is None or not self._entry:
            return
        lo = int(self.qmap_lo.value())
        hi = int(self.qmap_hi.value())
        if hi < lo:
            lo, hi = hi, lo
        self._start_worker(ScanProfileWorker(
            self._file_path, self._entry, "mean_image",
            frame_lo=lo, frame_hi=hi,
        ))

    # --- Plot data / image export ---

    _VIEW_KEYS = ("trajectories", "qmap", "amplitude", "waterfall")

    def _current_view_key(self) -> str:
        return self._VIEW_KEYS[self._tabs.currentIndex()]

    def _view_availability(self) -> list:
        """``(key, label, enabled, tooltip)`` per view — disabled
        entries carry a hint about what has to run first."""
        payload_ok = self._payload is not None
        need_track = "Run \"Track peaks (mlgidBASE)\" first."
        return [
            ("trajectories", "Trajectories", payload_ok,
             "" if payload_ok else need_track),
            ("qmap", "q-map overlay",
             payload_ok or self._mean_image is not None,
             "" if payload_ok or self._mean_image is not None
             else need_track),
            ("amplitude", "Amplitude evolution", payload_ok,
             "" if payload_ok else need_track),
            ("waterfall", "Radial waterfall", self._waterfall is not None,
             "" if self._waterfall is not None
             else "Compute the waterfall first."),
        ]

    def _structure_dialog_entries(self) -> list:
        """``(key, label, css_color, checked)`` per structure for the
        image-export dialog; initial check state mirrors the window's
        toggle bar."""
        if not self._track_phases:
            return []
        return [
            (
                key,
                "unmatched" if key == _UNMATCHED else key,
                pg.mkColor(self._structure_color(key)).name(),
                self._structure_key_visible(key),
            )
            for key in self._structure_keys()
        ]

    def _track_structure_label(self, k: int) -> str:
        key = self._structure_key_of(k)
        return "unmatched" if key == _UNMATCHED else key

    @staticmethod
    def _write_csv(path, header, rows, comment=None) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            if comment:
                fh.write(f"# {comment}\n")
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(rows)

    @staticmethod
    def _write_matrix_csv(
        path, corner, col_axis, row_axis, data, comment=None,
    ) -> None:
        """2D array as CSV: header row = column axis, first column =
        row axis (``corner`` labels both, e.g. ``frame\\|q|``)."""
        data = np.asarray(data, dtype=float)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            if comment:
                fh.write(f"# {comment}\n")
            writer = csv.writer(fh)
            writer.writerow(
                [corner] + [f"{float(v):.8g}" for v in col_axis]
            )
            for label, row in zip(row_axis, data):
                writer.writerow([label] + [f"{v:.8g}" for v in row])

    def _member_export_rows(self, extra_columns) -> list:
        """One CSV row per track member (ALL tracks — the structure
        toggles are a viewing aid, the export is complete). ``extra_
        columns(k, m)`` appends the per-view value columns."""
        payload = self._payload
        rows = []
        for k in range(payload.n_tracks):
            label = self._track_structure_label(k)
            all_structs = ";".join(self._track_phases.get(k, []))
            ring = int(k in self._ring_tracks)
            for m in payload.track_members(k):
                rows.append(
                    [k, label, all_structs, ring,
                     int(payload.frame_num[m])] + extra_columns(k, m)
                )
        return rows

    def _trajectories_export(self):
        payload = self._payload
        radius, angle = payload.radius, payload.angle
        header = ["track", "structure", "all_structures", "is_ring",
                  "frame", "radius", "angle", "amplitude", "q_xy", "q_z"]
        rows = self._member_export_rows(lambda k, m: [
            f"{radius[m]:.8g}", f"{angle[m]:.8g}",
            f"{payload.amplitude[m]:.8g}",
            f"{payload.q_xy[m]:.8g}", f"{payload.q_z[m]:.8g}",
        ])
        return header, rows

    def _qmap_export(self):
        payload = self._payload
        header = ["track", "structure", "all_structures", "is_ring",
                  "frame", "q_xy", "q_z", "track_mean_q_xy",
                  "track_mean_q_z", "track_mean_radius"]
        means = {
            k: (payload.track_mean("q_xy", k),
                payload.track_mean("q_z", k),
                payload.track_mean("radius", k))
            for k in range(payload.n_tracks)
        }
        rows = self._member_export_rows(lambda k, m: [
            f"{payload.q_xy[m]:.8g}", f"{payload.q_z[m]:.8g}",
            f"{means[k][0]:.8g}", f"{means[k][1]:.8g}",
            f"{means[k][2]:.8g}",
        ])
        return header, rows

    def _amplitude_export(self):
        """Amplitude tab numbers AS DISPLAYED: honours the smoothing
        window, the normalize toggle (forced on when grouped) and the
        median mode (-> per-structure median/quartile columns). The
        grouped view's vertical offsets are display-only and NOT
        included."""
        payload = self._payload
        grouped = self.amp_group.isChecked() and bool(self._track_phases)
        median = self.amp_median.isChecked() and bool(self._track_phases)
        window = int(self.amp_window.value())
        normalize = True if grouped else self.amp_normalize.isChecked()
        settings = (
            f"window={window}, normalize={normalize}, "
            f"mode={'median per structure' if median else 'per track'}; "
            "grouped-display offsets not included"
        )
        if median:
            header = ["structure", "frame", "median", "q25", "q75"]
            rows = []
            groups: dict = {}
            for k in range(payload.n_tracks):
                groups.setdefault(self._structure_key_of(k), []).append(k)
            for key in self._structure_keys():
                series = self._amp_median_series(
                    groups.get(key, []), window, normalize
                )
                if series is None:
                    continue
                label = "unmatched" if key == _UNMATCHED else key
                for x, med, lo, hi in zip(*series):
                    rows.append([
                        label, int(x), f"{med:.8g}",
                        f"{lo:.8g}", f"{hi:.8g}",
                    ])
            return header, rows, settings
        header = ["track", "structure", "frame", "amplitude"]
        rows = []
        for k in range(payload.n_tracks):
            label = self._track_structure_label(k)
            members = payload.track_members(k)
            frames, amps = smooth_normalize_amplitude(
                payload.frame_num[members], payload.amplitude[members],
                window=window, normalize=normalize,
            )
            for f, a in zip(frames, amps):
                rows.append([k, label, int(f), f"{a:.8g}"])
        return header, rows, settings

    def _export_data_files(self, keys, base: str) -> list:
        """Write one CSV per selected view (plus the q-map's mean-image
        matrix when computed) as ``<base>_<view>.csv``; returns the
        written paths."""
        entry = self._entry or (
            self._payload.entry if self._payload is not None else ""
        )
        provenance = f"mlgidLAB scan-tracking export, entry={entry}"
        written = []

        def _write(suffix, header, rows, comment):
            path = f"{base}_{suffix}.csv"
            self._write_csv(path, header, rows, comment=comment)
            written.append(path)

        if "trajectories" in keys and self._payload is not None:
            header, rows = self._trajectories_export()
            _write("trajectories", header, rows, provenance)
        if "qmap" in keys:
            if self._payload is not None:
                header, rows = self._qmap_export()
                _write("qmap_tracks", header, rows, provenance)
            if self._mean_image is not None:
                mi = self._mean_image
                path = f"{base}_qmap_mean_image.csv"
                lo, hi = int(self.qmap_lo.value()), int(self.qmap_hi.value())
                self._write_matrix_csv(
                    path, "q_z\\q_xy", mi["q_xy"],
                    [f"{float(v):.8g}" for v in mi["q_z"]], mi["data"],
                    comment=(
                        f"{provenance}; nan-mean Cartesian image of "
                        f"frames {min(lo, hi)}-{max(lo, hi)}, linear "
                        "intensity"
                    ),
                )
                written.append(path)
        if "amplitude" in keys and self._payload is not None:
            header, rows, settings = self._amplitude_export()
            _write("amplitude", header, rows, f"{provenance}; {settings}")
        if "waterfall" in keys and self._waterfall is not None:
            wf = self._waterfall
            data = np.asarray(wf["data"], dtype=float)
            path = f"{base}_waterfall.csv"
            self._write_matrix_csv(
                path, "frame\\|q|", wf["radius"],
                list(range(data.shape[0])), data,
                comment=(
                    f"{provenance}; angular-mean radial profile per "
                    "frame, linear intensity"
                ),
            )
            written.append(path)
        return written

    def _refresh_track_views(self) -> None:
        """Re-render the three payload-driven plots (the waterfall does
        not consume the export overrides)."""
        self._refresh_trajectories()
        self._refresh_qmap()
        self._refresh_amplitude()

    def _export_image_files(
        self, keys, base: str, ext: str, *, structures=None,
        line_width=None, line_style=None, marker_size=None,
        width_px: int = 1600, white_bg: bool = False,
    ) -> list:
        """Render the selected views to ``<base>_<view><ext>`` with the
        export styling applied, then restore the on-screen rendering.
        Returns the written paths."""
        from pyqtgraph.exporters import ImageExporter, SVGExporter

        overrides: dict = {}
        if structures is not None:
            overrides["structures"] = set(structures)
        if line_width is not None:
            overrides["line_width"] = float(line_width)
        if line_style is not None:
            overrides["line_style"] = line_style
        if marker_size is not None:
            overrides["marker_size"] = int(marker_size)
        plots = {
            "trajectories": self.traj_plot,
            "qmap": self.qmap_plot,
            "amplitude": self.amp_plot,
            "waterfall": self.waterfall_plot,
        }
        written = []
        self._export_overrides = overrides
        if white_bg:
            self._apply_plot_colors("w", "k")
        try:
            if overrides:
                self._refresh_track_views()
            for key in keys:
                path = f"{base}_{key}{ext}"
                item = plots[key].getPlotItem()
                if ext == ".svg":
                    SVGExporter(item).export(path)
                else:
                    exporter = ImageExporter(item)
                    try:
                        exporter.parameters()["width"] = int(width_px)
                    except Exception:
                        logger.debug(
                            "ImageExporter width parameter rejected; "
                            "exporting at native size", exc_info=True,
                        )
                    exporter.export(path)
                written.append(path)
        finally:
            self._export_overrides = {}
            if white_bg:
                self._apply_plot_colors(*self._theme_colors)
            if overrides:
                self._refresh_track_views()
        return written

    def _report_written(self, written) -> None:
        if not written:
            self.statusBar().showMessage("Nothing was exported.", 6000)
            return
        folder = os.path.dirname(written[0])
        names = ", ".join(os.path.basename(p) for p in written)
        self.statusBar().showMessage(
            f"Exported {len(written)} file(s) to {folder}: {names}", 10000
        )

    def _on_export_data(self) -> None:
        views = self._view_availability()
        if not any(enabled for _k, _l, enabled, _t in views):
            QMessageBox.information(
                self, "Export plot data",
                "Nothing to export yet — run \"Track peaks "
                "(mlgidBASE)\" or compute a waterfall first.",
            )
            return
        dlg = _ExportDialog(
            self, "data", views, current_key=self._current_view_key(),
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        keys = dlg.selected_views()
        if not keys:
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export plot data (CSV)", "tracking_data.csv",
            "CSV files (*.csv)",
        )
        if not path:
            return
        if path.lower().endswith(".csv"):
            path = path[:-4]
        self._report_written(self._export_data_files(keys, path))

    def _on_export_images(self) -> None:
        views = self._view_availability()
        if not any(enabled for _k, _l, enabled, _t in views):
            QMessageBox.information(
                self, "Export plot images",
                "Nothing to export yet — run \"Track peaks "
                "(mlgidBASE)\" or compute a waterfall first.",
            )
            return
        dlg = _ExportDialog(
            self, "image", views,
            structures=self._structure_dialog_entries(),
            current_key=self._current_view_key(),
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        keys = dlg.selected_views()
        if not keys:
            return
        path, chosen_filter = QFileDialog.getSaveFileName(
            self, "Export plot images", "tracking_plots.png",
            "PNG image (*.png);;SVG image (*.svg)",
        )
        if not path:
            return
        ext = (
            ".svg"
            if path.lower().endswith(".svg")
            or "svg" in chosen_filter.lower()
            else ".png"
        )
        for known in (".png", ".svg"):
            if path.lower().endswith(known):
                path = path[: -len(known)]
        style = dlg.style()
        written = self._export_image_files(
            keys, path, ext,
            structures=dlg.selected_structures(),
            line_width=style["line_width"],
            line_style=style["line_style"],
            marker_size=style["marker_size"],
            width_px=style["width_px"],
            white_bg=style["white_bg"],
        )
        self._report_written(written)

    # --- Official figure export ---

    def _on_save_figures(self) -> None:
        if self._payload is None:
            QMessageBox.information(
                self, "Save figures",
                "Run \"Track peaks (mlgidBASE)\" first.",
            )
            return
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Save official mlgidBASE tracking figures",
            "peak_tracking.png",
            "PNG image (*.png);;SVG image (*.svg);;PDF (*.pdf)",
        )
        if not path:
            return
        axis = self.axis_combo.currentData() or "radius"
        self.saveFiguresRequested.emit(path, axis)

    # --- Lifecycle ---

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        # Let a running view worker finish cleanly; it is read-only
        # and short-lived.
        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait()
        super().closeEvent(event)
