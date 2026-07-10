from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from PySide6.QtCore import (
    QCoreApplication,
    QEvent,
    QEventLoop,
    QMetaObject,
    QObject,
    QProcess,
    QSettings,
    QSignalBlocker,
    Qt,
    QThread,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QFrame,
    QPlainTextEdit,
    QProgressBar,
    QProgressDialog,
    QRadioButton,
    QDoubleSpinBox,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStyle,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from silx.gui.data.DataViewerFrame import DataViewerFrame
from silx.gui.hdf5 import Hdf5TreeModel, Hdf5TreeView
from silx.gui.hdf5.NexusSortFilterProxyModel import NexusSortFilterProxyModel

from mlgidlab import file_model
from mlgidlab import frame_range
from mlgidlab import peak_clipboard
from mlgidlab import update_check
from mlgidlab.image_viewer import (
    GIWAXSImageViewer,
    MATCHED_STYLE,
    UNMATCHED_COLOR,
    ManualPeak,
    OVERLAY_KINDS,
    OVERLAY_STYLE,
    SelectedPeak,
)
from mlgidlab.parameter_panel import ParameterPanel
from mlgidlab import phase_tracking
from mlgidlab.peaks_table_panel import PeaksTablePanel
from mlgidlab.phase_views_window import PhaseViewsWindow
from mlgidlab.pipeline import (
    PipelineCommand,
    is_mlgidbase_available,
)
from mlgidlab.pipeline_panel import PipelinePanel
from mlgidlab.profile_viewer import ProfileViewer
from mlgidlab.conversion_panel import ConversionPanel
from mlgidlab.scan_tracking_panel import ScanTrackingPanel
from mlgidlab.session import BaseSession, NexusSession, RawSession, Session
from mlgidlab.workers import (
    CifParseWorker,
    ConversionWorker,
    CopyWorker,
    EntryLoadWorker,
    PipelineWorker,
    PrefetchWorker,
)

import logging
logger = logging.getLogger(__name__)

APP_NAME = "mlgidLAB"
NEXUS_FILTER = "HDF5 / NeXus (*.h5 *.hdf5 *.nxs);;All files (*)"
# Open dialog auto-classifies NeXus vs raw HDF5 vs fabio image (TIFF/CBF/EDF);
# one combined filter does for all, with per-family filters below it.
OPEN_FILTER = (
    "All supported (*.h5 *.hdf5 *.nxs *.tif *.tiff *.cbf *.edf);;"
    "HDF5 / NeXus (*.h5 *.hdf5 *.nxs);;"
    "Detector images (*.tif *.tiff *.cbf *.edf);;"
    "All files (*)"
)


def _natural_key(path: Path):
    """Sort key so ``img_2`` precedes ``img_10`` (digits compared numerically)."""
    import re

    return [
        int(tok) if tok.isdigit() else tok.lower()
        for tok in re.split(r"(\d+)", path.name)
    ]


def _make_pen_swatch(style: dict, width: int = 26, height: int = 12) -> QPixmap:
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


def _pygidfit_to_fit_like(fit_2d):
    """Adapter exposing ``.center`` / ``.fwhm`` / ``.amplitude`` over a
    pygidfit ``ManualFitResult`` so ``ParameterPanel.set_fits`` can
    consume it interchangeably with scipy ``GaussianFit``s.

    pygidfit stores box widths as ``2σ`` (pipeline convention; see
    ``manual_fit.fit_one_peak``), so ``fwhm = 2σ × √(2 ln 2) =
    radius_width × √(2 ln 2) ≈ 1.177 × radius_width``. Used when
    the parameter readout is showing the 2D-mode preview of what
    Add-to-fitted (2D) will save.
    """
    from types import SimpleNamespace
    width_to_fwhm = float(np.sqrt(2.0 * np.log(2.0)))
    rfit = SimpleNamespace(
        center=float(fit_2d.radius),
        fwhm=float(fit_2d.radius_width) * width_to_fwhm,
        amplitude=float(fit_2d.amplitude),
    )
    afit = SimpleNamespace(
        center=float(fit_2d.angle),
        fwhm=float(fit_2d.angle_width) * width_to_fwhm,
        amplitude=float(fit_2d.amplitude),
    )
    return rfit, afit


class _CallbackAction:
    """Generic ``_Action`` (per the image_viewer protocol) that
    delegates ``undo`` / ``redo`` to caller-supplied callables.

    Lets host-level operations participate in the viewer's undo
    stack without dragging file-mutation code across the viewer
    abstraction. Used by ``_on_add_to_fitted`` so Ctrl+Z reverts a
    just-committed fitted row.
    """

    def __init__(self, undo_fn, redo_fn) -> None:
        self._undo_fn = undo_fn
        self._redo_fn = redo_fn

    def undo(self, _viewer) -> None:
        self._undo_fn()

    def redo(self, _viewer) -> None:
        self._redo_fn()


def _make_color_swatch(color: str, width: int = 26, height: int = 12) -> QPixmap:
    """Solid-line swatch in the given color — used for the matched-peaks
    master row where only the colour matters and there is no per-row
    line style to mirror.
    """
    return _make_pen_swatch(
        {"color": color, "style": MATCHED_STYLE["style"]}, width, height
    )




class _MlgidHdf5TreeModel(Hdf5TreeModel):
    """Silx tree model that swaps the file-root icon for raw sessions.

    The default ``Hdf5TreeModel`` uses ``SP_FileIcon`` for every loaded
    HDF5 file. Distinguishing converted-NeXus files (the pipeline runs
    on these) from raw detector files (they need conversion first)
    helps the user spot which is which when both are open in the file
    browser dock at the same time.

    The set of "raw" filesystem paths is owned by ``MainWindow`` and
    pushed in via ``set_raw_paths``; the model emits ``dataChanged``
    so existing rows refresh without a full rebuild.

    All read-only model overrides (``data`` / ``flags`` / ``rowCount`` /
    ``columnCount`` / ``hasChildren`` / ``index``) are wrapped in
    defensive try/except blocks that swallow ``ValueError`` /
    ``RecursionError`` / ``KeyError`` / ``OSError`` / ``RuntimeError``
    and return safe defaults. silx's ``Hdf5Item`` keeps an
    ``h5py.Group`` reference that can outlive the file handle
    (post-pipeline-run detach/reattach, session swap, file close);
    silx's own model methods don't defend against that and raise
    ``ValueError: Invalid group (or file) id`` from inside
    ``len(self.obj)``. Qt's QSortFilterProxyModel then re-fires the
    failing call through every proxy layer, producing a stack-busting
    recursion (40+ frames of ``QSortFilterProxyModel::data`` →
    ``QSortFilterProxyModel::rowCount`` → ``mapToSource``). Swallowing
    the error at our layer stops the storm at its source; the view
    paints a blank row for that frame, which the next natural repaint
    (after the proxy/source rebuild that follows the silx-dance
    completes) overwrites with the correct content.
    """

    # Exceptions raised when an Hdf5Item holds a stale h5py reference
    # or when Qt re-fires a failed call recursively.
    _STALE_EXC = (ValueError, KeyError, OSError, RuntimeError, RecursionError)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._raw_paths: set[str] = set()
        from PySide6.QtWidgets import QApplication, QStyle
        style = QApplication.style()
        self._raw_icon = style.standardIcon(QStyle.StandardPixmap.SP_DriveHDIcon)
        self._nexus_icon = style.standardIcon(QStyle.StandardPixmap.SP_FileIcon)

    def set_raw_paths(self, paths) -> None:
        # No dataChanged.emit here on purpose. Forcing silx's tree to
        # repaint while h5py items may still be in lazy-init has
        # produced reentrancy storms in QSortFilterProxyModel under
        # PySide6. Icons just take effect on the next natural paint
        # (resize / scroll / new insert), which is good enough.
        self._raw_paths = {str(p) for p in paths}

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if (
            role == Qt.ItemDataRole.DecorationRole
            and index.column() == self.NAME_COLUMN
            and not index.parent().isValid()
        ):
            try:
                node = self.nodeFromIndex(index)
                obj = getattr(node, "obj", None)
                if obj is not None:
                    filename = getattr(obj, "filename", None)
                    if filename:
                        if str(filename) in self._raw_paths:
                            return self._raw_icon
                        return self._nexus_icon
            except self._STALE_EXC:
                # If silx / h5py is in a transient bad state, fall
                # through to super().data() rather than propagating
                # an exception that Qt would re-fire endlessly.
                pass
        try:
            return super().data(index, role)
        except self._STALE_EXC:
            # silx's Hdf5Item.dataDescription walks `len(self.obj)` on
            # a possibly-stale h5py group and propagates a ValueError
            # ("Invalid group (or file) id") through every proxy layer.
            # Returning None lets the view paint a blank cell; the next
            # repaint after the silx-dance completes shows real data.
            return None

    def flags(self, index):
        try:
            return super().flags(index)
        except self._STALE_EXC:
            return Qt.ItemFlag.NoItemFlags

    def rowCount(self, parent=None):
        try:
            if parent is None:
                return super().rowCount()
            return super().rowCount(parent)
        except self._STALE_EXC:
            return 0

    def columnCount(self, parent=None):
        try:
            if parent is None:
                return super().columnCount()
            return super().columnCount(parent)
        except self._STALE_EXC:
            return 0

    def hasChildren(self, parent=None):
        try:
            if parent is None:
                return super().hasChildren()
            return super().hasChildren(parent)
        except self._STALE_EXC:
            return False

    def index(self, row, column, parent=None):
        try:
            if parent is None:
                return super().index(row, column)
            return super().index(row, column, parent)
        except self._STALE_EXC:
            from PySide6.QtCore import QModelIndex
            return QModelIndex()


class _MlgidHdf5TreeView(Hdf5TreeView):
    """Hdf5TreeView that builds its default model from our subclass.

    Also disables silx's built-in file-drop handler so all drag-and-
    drop events fall through to ``MainWindow.dropEvent``. Silx's
    default behaviour is to accept any URL drop on the tree and call
    ``insertFileAsync`` directly — that creates an orphan tree node
    with no matching ``Session`` in our session list, and later
    queries (selection changes, pipeline detach/reattach) blow up
    against the orphan's stale h5py handle.

    Emits ``deleteFileRequested`` when the user presses Delete with a
    tree row selected, so MainWindow can close the file the selection
    belongs to. Scoped to the tree's own ``keyPressEvent`` (fires only
    when the browser holds focus), so it never collides with the image
    viewer's Delete = "delete the selected peak" binding.
    """

    deleteFileRequested = Signal()

    def createDefaultModel(self):
        model = _MlgidHdf5TreeModel(self)
        model.setFileDropEnabled(False)
        proxy = NexusSortFilterProxyModel(self)
        proxy.setSourceModel(model)
        return proxy

    def keyPressEvent(self, ev) -> None:  # type: ignore[override]
        if (
            ev.key() == Qt.Key.Key_Delete
            and self.selectionModel() is not None
            and self.selectionModel().hasSelection()
        ):
            self.deleteFileRequested.emit()
            ev.accept()
            return
        super().keyPressEvent(ev)


class _ExportPeaksDialog(QDialog):
    """Modal kind/scope picker for Tools → Export peaks as CSV.

    Two QButtonGroups hold the kind (Detected/Fitted/Matched) and the
    scope (Active frame / Active entry / All entries) respectively;
    Active-frame is greyed when the active stack has only one frame
    so the option doesn't masquerade as different from Active-entry.
    """

    def __init__(self, parent: QWidget, *, has_multiple_frames: bool) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export peaks as CSV")
        layout = QVBoxLayout(self)

        kind_box = QGroupBox("Peak kind")
        kind_layout = QVBoxLayout(kind_box)
        self._rb_detected = QRadioButton("Detected")
        self._rb_fitted = QRadioButton("Fitted (with fit errors)")
        self._rb_matched = QRadioButton("Matched (flattened: one row per peak)")
        self._rb_fitted.setChecked(True)
        for rb in (self._rb_detected, self._rb_fitted, self._rb_matched):
            kind_layout.addWidget(rb)
        layout.addWidget(kind_box)

        scope_box = QGroupBox("Scope")
        scope_layout = QVBoxLayout(scope_box)
        self._rb_frame = QRadioButton("Active frame")
        self._rb_entry = QRadioButton("Active entry (all frames)")
        self._rb_all = QRadioButton("All entries (one combined CSV)")
        self._rb_entry.setChecked(True)
        if not has_multiple_frames:
            self._rb_frame.setEnabled(False)
            self._rb_frame.setToolTip(
                "Available only when the active entry has more than one frame."
            )
        for rb in (self._rb_frame, self._rb_entry, self._rb_all):
            scope_layout.addWidget(rb)
        layout.addWidget(scope_box)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def selected_kind(self) -> str:
        if self._rb_detected.isChecked():
            return "detected"
        if self._rb_matched.isChecked():
            return "matched"
        return "fitted"

    def selected_scope(self) -> str:
        if self._rb_frame.isChecked():
            return "frame"
        if self._rb_all.isChecked():
            return "all"
        return "entry"


# Playback settings persisted via QSettings under the keys below. The
# defaults give a 2× speed-up over the previous fixed 100 ms interval
# while still leaving headroom for cold-cache disk reads (~70-100 ms
# per fresh frame on local SSD). Users who want true frame-by-frame
# stepping can dial Frame interval up; users who want a fixed total
# duration (e.g. 5 s overview regardless of frame count) can flip to
# Total play time.
PLAYBACK_MODE_FRAME = "frame_interval_ms"
PLAYBACK_MODE_TOTAL = "total_time_s"
DEFAULT_PLAYBACK_FRAME_MS = 50          # 20 fps — was 100 ms / 10 fps
DEFAULT_PLAYBACK_TOTAL_S = 3.0
PLAYBACK_FRAME_MS_MIN = 10              # 100 fps requested ceiling
PLAYBACK_FRAME_MS_MAX = 2000            # 0.5 fps floor
PLAYBACK_TOTAL_S_MIN = 0.5
PLAYBACK_TOTAL_S_MAX = 600.0            # 10 minutes max

# Real tick cap. The eye stops perceiving extra frames much above
# ~20 fps, and large frames cannot be painted faster than ~50 ms
# regardless. When the user requests a faster per-frame rate (e.g.
# 3 s total over 300 frames = 10 ms / frame) we keep the timer at
# 50 ms and skip frames instead — see ``_compute_play_schedule``.
PLAYBACK_TICK_FLOOR_MS = 50


class _SettingsDialog(QDialog):
    """Application-wide settings dialog.

    Currently only carries the frame-playback section, but its
    layout reserves room for future settings groups (rendering,
    pipeline defaults, etc.) so adding a new section is just
    appending another ``QGroupBox`` to the outer layout.

    On accept, every changed value is written back to QSettings and
    the host MainWindow is told to re-apply (so an in-flight
    playback timer picks up the new interval immediately).
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(380)

        outer = QVBoxLayout(self)

        # --- Playback section -------------------------------------------------
        playback_box = QGroupBox("Frame playback")
        playback_layout = QVBoxLayout(playback_box)

        hint = QLabel(
            "<i>Controls the speed of the Display-dock Play button.</i>"
        )
        hint.setWordWrap(True)
        playback_layout.addWidget(hint)

        # Two mutually-exclusive modes. The active radio's spinbox is
        # the one that takes effect; the other stays editable so the
        # user can flip between modes without losing their values.
        mode_box = QButtonGroup(self)
        mode_box.setExclusive(True)
        self._rb_frame = QRadioButton("Time per frame")
        self._rb_total = QRadioButton("Total play time")
        mode_box.addButton(self._rb_frame)
        mode_box.addButton(self._rb_total)
        self._rb_frame.toggled.connect(self._refresh_enabled)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._spin_frame_ms = QSpinBox()
        self._spin_frame_ms.setRange(
            PLAYBACK_FRAME_MS_MIN, PLAYBACK_FRAME_MS_MAX
        )
        self._spin_frame_ms.setSingleStep(10)
        self._spin_frame_ms.setSuffix(" ms")
        self._spin_frame_ms.setToolTip(
            "Time spent on each frame. Lower = faster playback. The 10 ms "
            "lower bound caps playback at 100 fps; cold-cache disk reads "
            "may stutter below ~50 ms on large files."
        )

        self._spin_total_s = QDoubleSpinBox()
        self._spin_total_s.setRange(
            PLAYBACK_TOTAL_S_MIN, PLAYBACK_TOTAL_S_MAX
        )
        self._spin_total_s.setSingleStep(0.5)
        self._spin_total_s.setDecimals(2)
        self._spin_total_s.setSuffix(" s")
        self._spin_total_s.setToolTip(
            "Total time to traverse the whole stack (first frame → last "
            "frame). The per-frame interval is computed at play-start "
            "from the active entry's frame count, so swapping entries "
            "automatically adjusts the speed."
        )

        form.addRow(self._rb_frame, self._spin_frame_ms)
        form.addRow(self._rb_total, self._spin_total_s)
        playback_layout.addLayout(form)
        outer.addWidget(playback_box)

        # --- Buttons + outer wiring -------------------------------------------
        outer.addStretch(1)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

        # Load current values from QSettings (with defaults).
        settings = QSettings()
        mode = settings.value(
            MainWindow._PLAYBACK_MODE_KEY, PLAYBACK_MODE_FRAME
        )
        # QSettings returns strings on Linux but raw types on macOS;
        # coerce defensively.
        try:
            frame_ms = int(settings.value(
                MainWindow._PLAYBACK_FRAME_MS_KEY, DEFAULT_PLAYBACK_FRAME_MS
            ))
        except (TypeError, ValueError):
            frame_ms = DEFAULT_PLAYBACK_FRAME_MS
        try:
            total_s = float(settings.value(
                MainWindow._PLAYBACK_TOTAL_S_KEY, DEFAULT_PLAYBACK_TOTAL_S
            ))
        except (TypeError, ValueError):
            total_s = DEFAULT_PLAYBACK_TOTAL_S
        # Clamp into the spinbox range so out-of-bounds stored values
        # don't silently revert to the spinbox minimum.
        frame_ms = max(PLAYBACK_FRAME_MS_MIN,
                       min(PLAYBACK_FRAME_MS_MAX, frame_ms))
        total_s = max(PLAYBACK_TOTAL_S_MIN,
                      min(PLAYBACK_TOTAL_S_MAX, total_s))
        self._spin_frame_ms.setValue(frame_ms)
        self._spin_total_s.setValue(total_s)
        if mode == PLAYBACK_MODE_TOTAL:
            self._rb_total.setChecked(True)
        else:
            self._rb_frame.setChecked(True)
        self._refresh_enabled()

    def _refresh_enabled(self) -> None:
        frame_active = self._rb_frame.isChecked()
        self._spin_frame_ms.setEnabled(frame_active)
        self._spin_total_s.setEnabled(not frame_active)

    def save_to_qsettings(self) -> None:
        """Write the dialog's current values to QSettings.

        Called by the host on accept. Stores both spinbox values so a
        later mode-flip preserves the user's last value in each mode.
        """
        settings = QSettings()
        mode = PLAYBACK_MODE_FRAME if self._rb_frame.isChecked() else PLAYBACK_MODE_TOTAL
        settings.setValue(MainWindow._PLAYBACK_MODE_KEY, mode)
        settings.setValue(
            MainWindow._PLAYBACK_FRAME_MS_KEY, int(self._spin_frame_ms.value())
        )
        settings.setValue(
            MainWindow._PLAYBACK_TOTAL_S_KEY, float(self._spin_total_s.value())
        )


class _UpdateCheckWorker(QObject):
    """Runs the blocking GitHub release fetch off the GUI thread.

    Emits ``finished(result)`` where result is ``(tag, url)`` or ``None``
    (offline / no newer release). Lives on a ``QThread``; the host connects
    ``finished`` with a queued connection to marshal back to the GUI thread.
    """

    finished = Signal(object)

    @Slot()
    def run(self) -> None:
        try:
            result = update_check.latest_release()
        except Exception:  # pragma: no cover - defensive; latest_release is safe
            logger.debug("update check worker failed", exc_info=True)
            result = None
        self.finished.emit(result)


class _UpdateInstallWorker(QObject):
    """Runs the blocking ``pip install --upgrade`` off the GUI thread.

    Emits ``finished(returncode, output)`` where ``output`` is the combined
    stdout+stderr of pip (for the success/failure dialog). Lives on a
    ``QThread`` like ``_UpdateCheckWorker``; the host marshals ``finished``
    back to the GUI thread with a queued connection.
    """

    finished = Signal(int, str)

    def __init__(self, command: list[str]) -> None:
        super().__init__()
        self._command = command

    @Slot()
    def run(self) -> None:
        try:
            proc = subprocess.run(
                self._command,
                capture_output=True,
                text=True,
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            self.finished.emit(proc.returncode, output)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("update install worker failed", exc_info=True)
            self.finished.emit(1, f"Failed to launch pip: {exc}")


class _UpdateBanner(QFrame):
    """Dismissible 'a newer version is available' strip above the tabs.

    Carries an "Update now" button that self-installs the release (emitted
    as ``installRequested``); the host shows it only when the install is a
    normal pip-managed one (see ``update_check.self_update_supported``).
    """

    installRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("UpdateBanner")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            "#UpdateBanner { background: #2d5a88; }"
            "#UpdateBanner QLabel { color: #f5f7fa; }"
        )
        self._url = update_check.RELEASES_PAGE
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 6, 8, 6)
        row.setSpacing(10)
        self._label = QLabel(self)
        self._label.setWordWrap(True)
        row.addWidget(self._label, 1)
        self._update_btn = QToolButton(self)
        self._update_btn.setText("Update now")
        self._update_btn.setToolTip(
            "Download and install this release into the current environment."
        )
        self._update_btn.clicked.connect(self.installRequested)
        self._update_btn.hide()
        row.addWidget(self._update_btn)
        view_btn = QToolButton(self)
        view_btn.setText("View release")
        view_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(self._url))
        )
        row.addWidget(view_btn)
        close_btn = QToolButton(self)
        close_btn.setText("✕")
        close_btn.setToolTip("Dismiss")
        close_btn.clicked.connect(self.hide)
        row.addWidget(close_btn)

    def show_update(
        self, current: str, latest: str, url: str, *, can_install: bool = False
    ) -> None:
        self._url = url or update_check.RELEASES_PAGE
        self._label.setText(
            f"A newer version of mlgidLAB is available: "
            f"<b>{latest}</b> (you have {current})."
        )
        self._update_btn.setVisible(can_install)
        self.show()


class MainWindow(QMainWindow):
    # Cross-thread invocation signals for the prefetch worker (queued
    # auto-connection to slots on the worker's own QThread). Emitting
    # is the safe cross-thread equivalent of calling the worker's
    # methods directly; the queued delivery serialises with the
    # worker's other queued slots and its internal QTimer ticks.
    _prefetchConfigure = Signal(str, str, int, int)
    _prefetchUpdate = Signal(int, bool, int)
    _prefetchRelease = Signal()
    # Queued request to the persistent EntryLoadWorker: (file_path, entry,
    # request_id). The worker opens + warms that entry's first frame off
    # the GUI thread and emits ``loaded`` back; stale request_ids (rapid
    # switching) are dropped on arrival. See ``_load_entry_async``.
    _entryLoadRequest = Signal(str, str, int)
    # Raw-mode counterpart: (RawEntry, request_id) → ``raw_loaded`` with a
    # ready LazyRawStack. Shares the same request-id counter so raw and
    # NeXus switches supersede each other. See ``_load_raw_entry_into_viewer``.
    _rawLoadRequest = Signal(object, int)

    def __init__(self) -> None:
        super().__init__()
        # Multiple files can be open at once — each as its own Session in the
        # file browser. The "active" one drives entry_combo, the image viewer,
        # and per-file actions (save, save-as, close, pipeline). Switching is
        # automatic when the user clicks a node from a different file.
        self._sessions: list[BaseSession] = []
        self._active_session: BaseSession | None = None
        # Opens run serially through the existing single-thread CopyWorker
        # plumbing; extra paths from a multi-select dialog wait here.
        self._open_queue: list[Path] = []
        # Classification is now done in the worker (off the GUI thread), so
        # raw files and unclassifiable files are reported back as each
        # worker finishes and collected here; the batch is finalized when
        # the queue drains (raw files bundle into one RawSession).
        self._pending_raw_paths: list[Path] = []
        self._pending_rejected: list[Path] = []
        # RawEntry lists CopyWorker found while classifying, keyed by
        # str(path). Handed to the RawSession in ``_finalize_open_batch``
        # so ``_populate_raw_entries`` never re-walks the file's metadata
        # on the GUI thread (the walk takes seconds on a big beamtime
        # file and was the raw-open freeze).
        self._pending_raw_entry_cache: dict[str, list] = {}
        self._thread: QThread | None = None
        self._worker: CopyWorker | None = None
        self._pipe_thread: QThread | None = None
        self._pipe_worker: PipelineWorker | None = None
        # Phase-tracking (mlgidBASE track_peaks) results. Per-entry and
        # referencing FITTED rows, so they invalidate on entry switch
        # and on any fitted mutation — see _invalidate_scan_tracks.
        self._scan_payload: phase_tracking.TrackingPayload | None = None
        self._scan_member_ids: list | None = None
        self._scan_track_entry: str | None = None
        # Fitted tables read during member-id reconstruction, kept for
        # the only-tracked display options (gap interpolation anchors,
        # GUI-side ring tracking). {frame: PeakTable | None}.
        self._scan_fitted_tables: dict[int, object] = {}
        # {track_index: [cif, ...]} — the matched crystal phases each
        # track's tracked peaks belong to (dominant first), for the
        # phase-views q-map "Color by matched phase" overlay. Empty
        # until Matching has run; cleared with the rest of the results.
        self._scan_track_phases: dict = {}
        # Track indices that are rings — the q-map draws them as dashed
        # quarter-circle arcs rather than trajectory points.
        self._scan_ring_tracks: set = set()
        # Interpolate-track (gap fill) run in flight: the fill records
        # returned by the interpolate_tracks op (each {"track",
        # "frame", "detected_id", "fitted_id", "is_ring"}), applied as
        # new track members once the LAST chained re-match finishes;
        # and the chain bookkeeping that drives the loading dialog with
        # the workers' REAL per-frame progress:
        #   {"commands": [cmd, ...],          # keeps ids alive
        #    "ticks": {id(cmd): n_frames},    # per-command bar weight
        #    "total": N, "base": 0}           # overall progress
        self._interp_fill_result: list | None = None
        self._interp_chain: dict | None = None
        # Phase-tracking views window: lazily built on first "Show
        # views…", kept alive so its settings and computed caches
        # persist across re-opens (same pattern as _figure_export_window).
        self._phase_views_window: PhaseViewsWindow | None = None
        # Modal progress dialog shown during a track_peaks run. The
        # tracking button lives in the Scan-tracking dock but the
        # pipeline panel's bar is in a (usually tabbed-away) dock, so a
        # centred dialog is the only reliably-visible feedback. Its bar
        # is advanced by a GUI-thread timer (the op has no clean
        # per-frame progress — see the busy-bar note in pipeline_panel).
        self._track_progress_dialog: QProgressDialog | None = None
        self._track_progress_timer: QTimer | None = None
        # pygid normalization is lazy (deferred off the open path). This
        # records which (temp-file, entry) pairs have already been
        # normalized so ``_ensure_entry_normalized`` runs the scoped
        # ``normalize_for_pygid`` at most once per entry. Keyed by the
        # temp_path string (matches the file_path snapshotted into the
        # pipeline queue); pruned in ``_close_session``.
        self._normalized_entries: dict[str, set[str]] = {}
        # Lazy per-frame peak loading (see ``_load_frame_peaks``). Peaks
        # for a frame are pulled into the viewer the first time that frame
        # is shown, not all up front. Reset on each entry (re)load.
        self._loaded_peaks_entry: str | None = None
        self._loaded_peak_frames: set[int] = set()
        # A tree node clicked while the Data tab was hidden, held until the
        # user switches to that tab (see ``_set_or_defer_data_node``). This
        # keeps a single Image-tab click from eagerly resolving a huge
        # external-linked dataset into a widget the user can't see.
        self._pending_data_node = None
        # Tools → Export figure window. Built lazily on first open
        # (see ``_action_export_figure``); kept alive across re-opens
        # so settings persist. None until the user invokes Tools →
        # Export figure… for the first time.
        self._figure_export_window = None  # type: ignore[var-annotated]
        # Queue of (file_path, PipelineCommand) tuples waiting to run
        # sequentially. The file_path is **snapshotted at enqueue time**
        # so a mid-queue active-session switch (user clicks the other
        # loaded file in the tree, etc.) can't cause later commands to
        # dispatch against the wrong file. The "All entries" option in
        # the pipeline panel expands one runRequested into one command
        # per entry — all sharing the path captured at expansion time.
        self._pipeline_queue: list[tuple[Path, PipelineCommand]] = []
        # Entry-level progress tracking — the depth of the current
        # "All entries" expansion plus the 1-indexed position we are
        # at. Reset to (0, 0) when the queue drains; an "Active entry"
        # run sets total=1 (no entry bar shown). Surfaced to the panel
        # via ``on_queue_progress`` and folded into the status-bar tail.
        self._entry_queue_total: int = 0
        self._entry_queue_pos: int = 0
        # CIF-parse worker thread. CifPattern construction is slow for
        # raw CIFs so we run it off the GUI thread; only one parse runs
        # at a time (the panel's button stays disabled while it's in
        # flight).
        self._cif_parse_thread: QThread | None = None
        self._cif_parse_worker: CifParseWorker | None = None
        # Conversion worker thread (raw → NeXus). Kept separate from
        # the pipeline worker because conversion runs on raw inputs,
        # while pipeline runs on converted NeXus files; the two never
        # need to share a worker.
        self._conv_thread: QThread | None = None
        self._conv_worker: ConversionWorker | None = None
        self._conv_progress: QProgressDialog | None = None

        # Background prefetch worker. Lives on its own QThread so it
        # can read frames + compute polar resamples without ever
        # blocking the GUI. Spawned lazily on first entry load (no
        # cost on cold startup) and survives across entry switches —
        # the worker is reconfigured per-entry rather than rebuilt.
        # See ``_ensure_prefetch_worker`` and the ``_prefetch*``
        # signals on the class for the cross-thread wiring.
        self._prefetch_thread: QThread | None = None
        self._prefetch_worker: PrefetchWorker | None = None

        # Persistent EntryLoadWorker: opens + warms an entry's first frame
        # off the GUI thread so switching entries (combo or file browser)
        # never blocks on a slow network read. Lazily spawned on the first
        # interactive switch. ``_entry_req_id`` is bumped per switch so a
        # late result from a superseded switch is dropped, not rendered.
        self._entry_load_thread: QThread | None = None
        self._entry_load_worker: EntryLoadWorker | None = None
        self._entry_req_id: int = 0

        # Frame step per play-tick. Stays at 1 unless the requested
        # per-frame interval drops below ``PLAYBACK_TICK_FLOOR_MS``,
        # in which case ``_compute_play_schedule`` bumps it so the
        # play-head jumps multiple frames per tick to honour the
        # total-time target without overrunning the 20 fps practical
        # ceiling. Refreshed on every Play press + every settings
        # change while playing.
        self._play_step: int = 1

        # Stash of the manual peak's geometry captured the moment the
        # "Save fitted as ring" toggle goes ON. Set to a tuple of
        # (peak_ref, radius, angle, radius_width, angle_width, is_ring)
        # while ring is active; cleared on the toggle's OFF transition
        # after the box has been restored. Allows the auto-uncheck
        # that follows a successful Add-to-fitted to revert the box
        # to its pre-ring shape without the host needing to track a
        # commit/cancel distinction.
        self._ring_pre_geom: tuple[
            ManualPeak, float, float, float, float, bool
        ] | None = None

        # 2D-preview cache: fingerprint -> ManualFitResult | None.
        # Fingerprint is the user-controlled inputs to the live
        # pygidfit call; identical fingerprint → reuse the cached
        # result instead of rerunning the (slow) 2D fit. A dict
        # (not just one slot) so the multi-selection cyan-preview
        # path can show N boxes without recomputing N peaks every
        # selection-tick. ``None`` value means pygidfit ran but
        # didn't converge — cached so we don't retry the same fail.
        self._2d_preview_cache: dict[tuple, object] = {}

        self.setWindowTitle(APP_NAME)
        self.resize(1400, 900)

        self._build_menu()
        self._build_central()
        self._build_docks()
        # View menu is built after docks because it pulls
        # toggleViewAction()s from them. Settings is built next.
        # Help comes last so it sits at the right end of the menu
        # bar — the conventional rightmost-menu placement.
        self._build_view_menu()
        self._build_settings_menu()
        self._build_help_menu()
        # Frame-navigation shortcuts. Installed last so the viewer +
        # entry combo exist. Window-context QActions; text inputs
        # (QLineEdit, QSpinBox) consume Left/Right/Home/End for
        # caret nav before the shortcut fires, so the bindings only
        # trigger when focus is on a non-text widget (viewer, dock
        # frame, menu bar). J/K give a Vim-style fallback that
        # works even when the viewer has unconventional focus
        # handling.
        self._install_frame_shortcuts()
        # Make undo/redo fire even when a third-party widget registers the
        # same standard shortcuts (see ``eventFilter``). Installed on the
        # application so it sees key events regardless of focus.
        QApplication.instance().installEventFilter(self)
        # Status bar depends on the viewer + entry combo existing; build
        # after central + docks.
        self._build_status_bar()
        self._update_title()
        self._update_actions()
        # Accept dropped files anywhere on the main window so the user
        # can drag NeXus / raw paths in from a file manager. The drop
        # handler classifies each file by content and dispatches.
        self.setAcceptDrops(True)
        # Snapshot the default dock arrangement now that everything
        # has been built and the initial ``_apply_session_mode(None)``
        # has tabified the right-side stack. View → Reset layout
        # restores from this snapshot — see ``_reset_layout``.
        self._default_layout_state = self.saveState()

    @property
    def session(self) -> Session | None:
        """The currently-active session — drives viewer + entry_combo + save.

        Most callers were written before multi-file support and reach for
        ``self.session``; making it a property of the active session keeps
        those call sites working without per-call refactors.
        """
        return self._active_session

    def _build_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        self._build_file_menu(file_menu)
        self._build_edit_menu(bar)
        self._build_tools_menu(bar)

    def _build_edit_menu(self, bar) -> None:
        edit_menu = bar.addMenu("&Edit")
        self.action_undo = QAction("&Undo", self)
        self.action_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.action_undo.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.action_undo.triggered.connect(self._action_undo)
        edit_menu.addAction(self.action_undo)

        self.action_redo = QAction("&Redo", self)
        # Bind both Ctrl+Y (Win/Linux default) and Ctrl+Shift+Z so muscle
        # memory from either platform works.
        self.action_redo.setShortcuts([
            QKeySequence(QKeySequence.StandardKey.Redo),
            QKeySequence("Ctrl+Shift+Z"),
        ])
        self.action_redo.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.action_redo.triggered.connect(self._action_redo)
        edit_menu.addAction(self.action_redo)

        edit_menu.addSeparator()
        # Ctrl+C / Ctrl+V: copy + paste detected peaks across frames in
        # the same entry. Scoped to detected in this iteration; fitted
        # / matched / manual copy is deferred. Text widgets (QLineEdit,
        # QSpinBox) intercept these for their own clipboard semantics
        # before window-level actions see them, so typing in inputs
        # works as expected.
        self.action_copy_peaks = QAction("&Copy detected peak(s)", self)
        self.action_copy_peaks.setShortcut(QKeySequence.StandardKey.Copy)
        self.action_copy_peaks.setShortcutContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self.action_copy_peaks.setToolTip(
            "Copy the selected detected peak(s) to the in-app clipboard. "
            "Paste on another frame in the same entry."
        )
        self.action_copy_peaks.triggered.connect(self._on_copy_peaks)
        edit_menu.addAction(self.action_copy_peaks)

        self.action_paste_peaks = QAction("&Paste detected peak(s)", self)
        self.action_paste_peaks.setShortcut(QKeySequence.StandardKey.Paste)
        self.action_paste_peaks.setShortcutContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self.action_paste_peaks.setToolTip(
            "Paste copied detected peak(s) onto the current frame. "
            "Same-entry only; cross-entry paste is intentionally blocked."
        )
        self.action_paste_peaks.triggered.connect(self._on_paste_peaks)
        edit_menu.addAction(self.action_paste_peaks)

        # Ctrl+Shift+V: paste the clipboard to every frame in a typed
        # range (e.g. "0-34,37"). One grouped undo reverses the whole
        # range. Out-of-range frames are filtered with a confirmation
        # dialog; duplicates in the input dedup silently.
        self.action_paste_peaks_to_range = QAction(
            "Paste detected peak(s) to &frames…", self,
        )
        self.action_paste_peaks_to_range.setShortcut(
            QKeySequence("Ctrl+Shift+V")
        )
        self.action_paste_peaks_to_range.setShortcutContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self.action_paste_peaks_to_range.setToolTip(
            "Paste copied detected peak(s) to a typed frame range "
            "(e.g. 0-34,37). Same-entry only."
        )
        self.action_paste_peaks_to_range.triggered.connect(
            self._on_paste_peaks_to_range
        )
        edit_menu.addAction(self.action_paste_peaks_to_range)

        edit_menu.addSeparator()
        # Find peak by ID. Opens a modal asking for the kind + numeric
        # ID, then selects that peak in the viewer (jumping to its
        # frame if needed) and switches the Peaks dock to the
        # matching tab via the existing selection-sync wiring.
        self.action_find_peak = QAction("&Find peak by ID…", self)
        self.action_find_peak.setShortcut(QKeySequence("Ctrl+F"))
        self.action_find_peak.setToolTip(
            "Find a Detected or Fitted peak by its numeric ID and "
            "select it in the viewer."
        )
        self.action_find_peak.triggered.connect(self._action_find_peak)
        edit_menu.addAction(self.action_find_peak)

    def _build_tools_menu(self, bar) -> None:
        """Bulk-edit operations that don't fit the per-peak ROI workflow.

        Currently scoped to "clear all of one kind for the active entry".
        Future additions (export, copy peaks across frames, statistics,
        symmetry ops, etc.) will land here too — see the README for the
        full roadmap.
        """
        tools_menu = bar.addMenu("&Tools")
        # The three per-kind clear-* entries each expand into a scope
        # sub-submenu (Active frame / Active entry / All entries) so the
        # user can wipe exactly the slice they mean. The Reset submenu
        # below still offers a one-click "everything on this scope"
        # for when the user does not care about the kind split.
        clear_menu = tools_menu.addMenu("&Clear peaks")
        clear_menu.setToolTipsVisible(True)

        self._clear_detected_menu = self._build_clear_kind_submenu(
            clear_menu, "Detected", "detected"
        )

        # "Fitted and Matched": clearing fitted necessarily invalidates
        # matched, because matched solutions reference fitted ids and an
        # orphaned matched_* group would render against missing rows.
        # The cascade is one-way (fitted -> matched).
        self._clear_fitted_menu = self._build_clear_kind_submenu(
            clear_menu, "Fitted and Matched", "fitted"
        )

        # "Matched" clears only the matched_* solutions; detected and
        # fitted are left intact (re-match without re-fitting).
        self._clear_matched_menu = self._build_clear_kind_submenu(
            clear_menu, "Matched", "matched"
        )

        # Re-evaluate "Active frame" gates across every per-kind submenu
        # just before the Clear peaks parent is shown. Cheap, and avoids
        # plumbing a frame-count signal into every action.
        clear_menu.aboutToShow.connect(self._refresh_clear_menu_state)

        # Reset submenu — full wipe of det + fit + match (and manual,
        # in-memory) at three scopes. "Active frame" is greyed out
        # when fewer than two frames are loaded since on a single-
        # frame file it would just duplicate "Active entry".
        clear_menu.addSeparator()
        reset_menu = clear_menu.addMenu("&Reset all peaks")
        reset_menu.setToolTipsVisible(True)
        self._reset_menu = reset_menu

        self.action_reset_all = QAction("All entries", self)
        self.action_reset_all.setToolTip(
            "Clear detected, fitted, matched, and manual peaks on every "
            "entry in the active file."
        )
        self.action_reset_all.triggered.connect(
            lambda: self._action_reset_analysis("all")
        )
        reset_menu.addAction(self.action_reset_all)

        self.action_reset_entry = QAction("Active entry (all frames)", self)
        self.action_reset_entry.setToolTip(
            "Clear detected, fitted, matched, and manual peaks on the "
            "currently displayed entry."
        )
        self.action_reset_entry.triggered.connect(
            lambda: self._action_reset_analysis("entry")
        )
        reset_menu.addAction(self.action_reset_entry)

        self.action_reset_frame = QAction("Active frame", self)
        self.action_reset_frame.setToolTip(
            "Clear detected, fitted, and matched peaks on just the "
            "currently displayed frame of the active entry. Manual "
            "peaks are wiped (they live in memory across frames)."
        )
        self.action_reset_frame.triggered.connect(
            lambda: self._action_reset_analysis("frame")
        )
        reset_menu.addAction(self.action_reset_frame)
        # Re-evaluate the per-scope enabled states right before the
        # submenu is shown — n_frames / session state can change between
        # menu opens, and aboutToShow keeps the gate cheap (no signal
        # plumbing on every viewer event).
        reset_menu.aboutToShow.connect(self._refresh_reset_menu_state)

        # Figure export. Replaces the previous pyqtgraph
        # ImageExporter-based PNG capture with a non-modal window
        # built around ``mlgidbase.plot_analysis_results``. Lives at
        # ``mlgidlab.figure_export_window.FigureExportWindow``;
        # imported lazily inside the handler so a missing pipeline
        # dep doesn't break menu construction.
        tools_menu.addSeparator()
        self.action_export_figure = QAction("Export figure…", self)
        self.action_export_figure.triggered.connect(self._action_export_figure)
        tools_menu.addAction(self.action_export_figure)

        # CSV export of detected/fitted/matched peaks. NeXus-only.
        self.action_export_csv = QAction("Export peaks as CSV…", self)
        self.action_export_csv.triggered.connect(self._action_export_csv)
        tools_menu.addAction(self.action_export_csv)

    def _build_clear_kind_submenu(self, parent_menu, label: str, kind: str):
        """Build a per-kind Clear-peaks submenu with three scope choices.

        ``kind`` is one of ``detected``/``fitted``/``matched`` and is
        forwarded to ``_action_clear_file_peaks`` together with the
        chosen scope. Keeps the per-action references on ``self`` so
        ``_refresh_clear_menu_state`` can flip "Active frame" enabled
        when the active entry has a single frame (no per-frame scope
        is meaningful there — it would just duplicate Active entry).
        Returns the QMenu so the caller can stash it for raw-mode
        gating.
        """
        sub = parent_menu.addMenu(label)
        sub.setToolTipsVisible(True)

        act_all = QAction("All entries", self)
        act_all.setToolTip(
            f"Clear every {kind} peak on every entry in the active file."
        )
        act_all.triggered.connect(
            lambda _checked=False, k=kind: self._action_clear_file_peaks(k, "all")
        )
        sub.addAction(act_all)

        act_entry = QAction("Active entry (all frames)", self)
        act_entry.setToolTip(
            f"Clear every {kind} peak on the currently displayed entry."
        )
        act_entry.triggered.connect(
            lambda _checked=False, k=kind: self._action_clear_file_peaks(k, "entry")
        )
        sub.addAction(act_entry)

        act_frame = QAction("Active frame", self)
        act_frame.setToolTip(
            f"Clear {kind} peaks on just the currently displayed frame "
            "of the active entry."
        )
        act_frame.triggered.connect(
            lambda _checked=False, k=kind: self._action_clear_file_peaks(k, "frame")
        )
        sub.addAction(act_frame)

        # Stash per-kind frame action so _refresh_clear_menu_state can
        # gate it on the live frame count.
        setattr(self, f"_clear_{kind}_frame_action", act_frame)
        setattr(self, f"_clear_{kind}_entry_action", act_entry)
        setattr(self, f"_clear_{kind}_all_action", act_all)
        return sub

    def _refresh_clear_menu_state(self) -> None:
        """Gate the per-kind Clear-peaks scope actions.

        Mirrors ``_refresh_reset_menu_state``: "Active frame" is greyed
        out unless there's an open session and the current entry has
        more than one frame. Active-entry / All-entries need only an
        open session. Kept cheap (called on ``aboutToShow``) — no signal
        plumbing on every viewer event.
        """
        has_session = self.session is not None and self._pipe_thread is None
        n_frames = getattr(self.viewer, "n_frames", 0) if has_session else 0
        for kind in ("detected", "fitted", "matched"):
            entry_a = getattr(self, f"_clear_{kind}_entry_action", None)
            all_a = getattr(self, f"_clear_{kind}_all_action", None)
            frame_a = getattr(self, f"_clear_{kind}_frame_action", None)
            if entry_a is not None:
                entry_a.setEnabled(has_session)
            if all_a is not None:
                all_a.setEnabled(has_session)
            if frame_a is not None:
                frame_a.setEnabled(has_session and n_frames > 1)

    def _action_clear_file_peaks(self, kind: str, scope: str = "entry") -> None:
        """Empty every ``<kind>_peaks`` dataset at the requested scope.

        ``scope`` is one of:
        - ``"entry"`` — active entry, every frame in it.
        - ``"all"``   — every entry in the active file, every frame.
        - ``"frame"`` — active entry, just the active frame.

        Cascade rule (one-way):
        - clearing ``fitted`` also clears ``matched`` (matched rows
          reference fitted ids; orphaned matched_* groups can't render).
        - clearing ``matched`` clears matched only; detected and fitted
          are left intact (re-match without re-fitting). See the
          Tools-menu wiring above.

        Manual peaks are session-wide and live only in memory — they
        are deliberately *not* touched here; only ``Reset all peaks``
        wipes them.
        """
        if self.session is None or self._pipe_thread is not None:
            return
        active_entry = self.entry_combo.currentText()
        if scope in ("entry", "frame") and not active_entry:
            return
        if scope == "frame" and getattr(self.viewer, "n_frames", 0) <= 1:
            return

        # Build scope-specific (entry, frame|None) targets, same shape
        # as _action_reset_analysis.
        if scope == "all":
            try:
                targets = [
                    (e, None) for e in file_model.list_entries(self.session.temp_path)
                ]
            except Exception as exc:
                QMessageBox.critical(self, "Clear failed", f"Could not list entries: {exc}")
                return
            scope_label = f"all {len(targets)} entries"
        elif scope == "entry":
            targets = [(active_entry, None)]
            scope_label = f"entry {active_entry}"
        else:  # frame
            frame_idx = int(self.viewer.current_frame)
            targets = [(active_entry, frame_idx)]
            scope_label = f"frame {frame_idx} of {active_entry}"

        if not self._confirm_clear(kind, scope_label):
            return

        kinds_to_clear = [kind]
        if kind == "fitted":
            kinds_to_clear.append("matched")

        with self._detached_silx_tree():
            try:
                removed_total = 0
                for entry, frame in targets:
                    for k in kinds_to_clear:
                        removed_total += file_model.clear_peaks(
                            self.session.temp_path, entry, k, frame=frame
                        )
            except Exception as exc:
                QMessageBox.critical(self, "Clear failed", str(exc))
                return

        self.session.mark_dirty()
        self._update_title()
        # Bulk wipe invalidates every FileGeomAction and the selection.
        self.viewer.clear_history()
        self.viewer.clear_selection()
        # Clearing fitted rows orphans any phase-tracking results
        # (upstream tracks fitted peaks). Detected/matched-only clears
        # leave fitted intact so tracks stay valid.
        if kind == "fitted":
            self._invalidate_scan_tracks()
        if active_entry:
            self._load_entry_into_viewer(active_entry, preserve_view=True)
        self.pipeline_panel.append_log(
            f"Cleared {' + '.join(kinds_to_clear)} peaks "
            f"({removed_total} rows total) on {scope_label}"
        )

    def _refresh_reset_menu_state(self) -> None:
        """Gate Reset submenu actions on session + frame availability.

        Active-frame is greyed out with a single frame loaded since the
        clear would be identical to Active-entry. Active-entry / All
        entries need only an open session.
        """
        has_session = self.session is not None and self._pipe_thread is None
        n_frames = getattr(self.viewer, "n_frames", 0) if has_session else 0
        self.action_reset_entry.setEnabled(has_session)
        self.action_reset_all.setEnabled(has_session)
        self.action_reset_frame.setEnabled(has_session and n_frames > 1)

    def _action_reset_analysis(self, scope: str) -> None:
        """Wipe det + fit + match (and manual peaks) at the requested scope.

        ``scope`` is one of:
        - ``"entry"`` — active entry, every frame in it.
        - ``"all"``   — every entry in the active file, every frame.
        - ``"frame"`` — active entry, just the active frame.

        Manual peaks are session-wide and live in memory only; every
        scope clears them outright since the user asked for a true reset.
        """
        if self.session is None or self._pipe_thread is not None:
            return
        active_entry = self.entry_combo.currentText()
        if scope in ("entry", "frame") and not active_entry:
            return
        if scope == "frame" and getattr(self.viewer, "n_frames", 0) <= 1:
            return

        # Build the scope-specific list of (entry, frame|None) tuples
        # the inner h5 wipe loop iterates over.
        if scope == "all":
            try:
                targets = [(e, None) for e in file_model.list_entries(self.session.temp_path)]
            except Exception as exc:
                QMessageBox.critical(self, "Reset failed", f"Could not list entries: {exc}")
                return
            scope_label = f"all {len(targets)} entries"
        elif scope == "entry":
            targets = [(active_entry, None)]
            scope_label = f"entry {active_entry}"
        else:  # frame
            frame_idx = int(self.viewer.current_frame)
            targets = [(active_entry, frame_idx)]
            scope_label = f"frame {frame_idx} of {active_entry}"

        reply = QMessageBox.question(
            self,
            "Reset analysis",
            f"Remove every detected, fitted, matched, and manual peak "
            f"on {scope_label}?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Manual peaks are global session state — drop them once,
        # regardless of scope.
        self.viewer.clear_all_manual_peaks()

        with self._detached_silx_tree():
            try:
                removed_total = 0
                for entry, frame in targets:
                    for kind in ("detected", "fitted", "matched"):
                        removed_total += file_model.clear_peaks(
                            self.session.temp_path, entry, kind, frame=frame
                        )
            except Exception as exc:
                QMessageBox.critical(self, "Reset failed", str(exc))
                return

        self.session.mark_dirty()
        self._update_title()
        self.viewer.clear_history()
        self.viewer.clear_selection()
        # Reset wipes detected rows — scan-tracking results are orphaned.
        self._invalidate_scan_tracks()
        # Refresh the displayed entry — the cleared one if the user
        # was looking at it, otherwise the currently-active one.
        if active_entry:
            self._load_entry_into_viewer(active_entry, preserve_view=True)
        self.pipeline_panel.append_log(
            f"Reset analysis: cleared {removed_total} peak rows on {scope_label} "
            f"(plus all manual peaks)"
        )

    def _action_export_figure(self) -> None:
        """Open the non-modal Figure Export window.

        The window is built lazily so cold startup doesn't pay
        matplotlib / mlgidbase import cost. A single instance per
        main window is reused across re-opens so the user's
        settings persist for the GUI session.
        """
        if self.session is None:
            QMessageBox.information(
                self, "No file open",
                "Open a NeXus file before exporting a figure.",
            )
            return
        if not isinstance(self.session, NexusSession):
            QMessageBox.information(
                self, "Figure export needs a NeXus file",
                "The figure exporter renders detected, fitted, and "
                "matched peak overlays from a processed NeXus file. "
                "Run the conversion on your raw data first.",
            )
            return
        if self._figure_export_window is None:
            from mlgidlab.figure_export_window import FigureExportWindow
            self._figure_export_window = FigureExportWindow(self)
        else:
            # Window already exists — refresh its cached mlgidbase
            # handle in case the user swapped files since it was
            # last shown.
            self._figure_export_window.refresh_for_session()
        self._figure_export_window.show()
        self._figure_export_window.raise_()
        self._figure_export_window.activateWindow()

    def _action_export_csv(self) -> None:
        """Pop the kind/scope dialog and write peaks to a CSV.

        NeXus-only — raw sessions don't have peak datasets. The actual
        flatten + write lives in ``file_model.export_peaks_csv`` /
        ``export_matched_csv``; the GUI's job here is dialog wiring,
        scope resolution, and the silx detach/reattach that frees the
        file's HDF5 handle for r-mode reads.
        """
        if self.session is None or self.session.kind != "nexus":
            QMessageBox.information(
                self, "Export peaks",
                "Open a NeXus file first — raw files have no peak datasets.",
            )
            return
        active_entry = self.entry_combo.currentText()
        if not active_entry:
            QMessageBox.information(
                self, "Export peaks", "No active entry to export from."
            )
            return
        n_frames = getattr(self.viewer, "n_frames", 0)
        dlg = _ExportPeaksDialog(self, has_multiple_frames=n_frames > 1)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        kind = dlg.selected_kind()
        scope = dlg.selected_scope()

        # Suggest a filename rooted at the original-file basename so
        # batched exports from multiple opens don't collide on disk.
        base = self.session.original_path.stem
        suggest = f"{base}_{kind}_{scope}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export peaks as CSV", suggest,
            "CSV (*.csv);;All files (*)",
        )
        if not path:
            return

        # Resolve the scope into an (entry, frame|None) target list
        # consumed by the file_model exporters.
        if scope == "all":
            try:
                entries = file_model.list_entries(self.session.temp_path)
            except Exception as exc:
                QMessageBox.critical(self, "Export failed", f"Could not list entries: {exc}")
                return
            targets: list[tuple[str, int | None]] = [(e, None) for e in entries]
        elif scope == "entry":
            targets = [(active_entry, None)]
        else:
            targets = [(active_entry, int(self.viewer.current_frame))]

        # silx may hold a read handle on the temp file; detach so h5py
        # can open it without contention.
        with self._detached_silx_tree():
            try:
                if kind == "matched":
                    n = file_model.export_matched_csv(
                        self.session.temp_path, targets, Path(path)
                    )
                else:
                    n = file_model.export_peaks_csv(
                        self.session.temp_path, targets, kind, Path(path)
                    )
            except Exception as exc:
                QMessageBox.critical(self, "Export failed", str(exc))
                return

        self.statusBar().showMessage(
            f"Wrote {n} {kind} peak rows ({scope}) to {path}", 6000
        )
        self.pipeline_panel.append_log(
            f"Exported {n} {kind} peak rows ({scope}) to {path}"
        )

    def _confirm_clear(self, kind: str, scope_label: str = "") -> bool:
        descriptions = {
            "detected": ("detected peaks",
                         "every row of detected_peaks"),
            "fitted":   ("fitted + matched peaks",
                         "every row of fitted_peaks AND every matched_* "
                         "solution (matched references fitted, so it has "
                         "to go too)"),
            "matched":  ("matched peaks",
                         "every matched_* solution "
                         "(detected and fitted peaks are left intact)"),
        }
        title, body = descriptions.get(kind, (kind, kind))
        scope_suffix = f" on {scope_label}" if scope_label else ""
        reply = QMessageBox.question(
            self,
            f"Clear {title}",
            f"Remove {body}{scope_suffix}?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _build_view_menu(self) -> None:
        """Expose dock visibility toggles in a top-level View menu.

        Each dock already has a built-in ``toggleViewAction()`` whose label
        and check state stay in sync with the dock — reusing them keeps the
        menu correct without manual bookkeeping.
        """
        view_menu = self.menuBar().addMenu("&View")
        for dock in (
            self._tree_dock,
            self._display_dock,
            self._pipeline_dock,
            self._conversion_dock,
            self._logs_dock,
            self._peaks_dock,
            self._profile_dock,
            self._scan_tracking_dock,
        ):
            view_menu.addAction(dock.toggleViewAction())
        view_menu.addSeparator()
        # Toggle for the cursor-readout segment of the status bar — some
        # users find the per-pixel readout distracting; on by default.
        self.action_toggle_cursor_readout = QAction(
            "Show cursor readout", self
        )
        self.action_toggle_cursor_readout.setCheckable(True)
        self.action_toggle_cursor_readout.setChecked(True)
        self.action_toggle_cursor_readout.toggled.connect(
            self._set_cursor_readout_visible
        )
        view_menu.addAction(self.action_toggle_cursor_readout)

        # Reset layout — restores the dock arrangement captured at
        # cold startup (see ``_capture_default_layout``). Useful when
        # the user has drag-rearranged things and wants to start
        # over without restarting the app.
        view_menu.addSeparator()
        self.action_reset_layout = QAction("Reset layout", self)
        self.action_reset_layout.setToolTip(
            "Restore the default dock arrangement."
        )
        self.action_reset_layout.triggered.connect(self._reset_layout)
        view_menu.addAction(self.action_reset_layout)

        # F11 fullscreen — hides every dock so the image viewer
        # owns the whole window. Menu bar stays so F11 / View
        # remains reachable. Checkable so the menu reads its state
        # back; toggled() drives the same path as the F11 keypress.
        self.action_fullscreen = QAction("&Fullscreen image viewer", self)
        self.action_fullscreen.setShortcut(QKeySequence("F11"))
        self.action_fullscreen.setCheckable(True)
        self.action_fullscreen.setToolTip(
            "Maximise the image viewer by hiding every dock. F11 "
            "toggles back."
        )
        self.action_fullscreen.toggled.connect(self._set_fullscreen)
        view_menu.addAction(self.action_fullscreen)

        # Theme submenu — Dark (default) / Light. Both checkable +
        # mutually exclusive via QActionGroup so the menu reads as
        # a radio choice. Selection persists via QSettings; applied
        # at startup in ``__init__``.
        view_menu.addSeparator()
        theme_menu = view_menu.addMenu("&Theme")
        self.action_theme_dark = QAction("&Dark", self)
        self.action_theme_dark.setCheckable(True)
        self.action_theme_light = QAction("&Light", self)
        self.action_theme_light.setCheckable(True)
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        theme_group.addAction(self.action_theme_dark)
        theme_group.addAction(self.action_theme_light)
        theme_menu.addAction(self.action_theme_dark)
        theme_menu.addAction(self.action_theme_light)
        self.action_theme_dark.triggered.connect(lambda: self._set_theme("dark"))
        self.action_theme_light.triggered.connect(lambda: self._set_theme("light"))
        # Sync the menu's check state with whatever's persisted /
        # currently active. ``_apply_persisted_theme`` (called once
        # at startup) writes self._current_theme.
        current = getattr(self, "_current_theme", "dark")
        (self.action_theme_dark if current == "dark"
         else self.action_theme_light).setChecked(True)

    def _set_fullscreen(self, on: bool) -> None:
        """Enter / leave the image-viewer-only fullscreen mode.

        On enter: snapshot every dock's current visibility, then
        hide them. On exit: restore the snapshotted states. The
        menu bar is left alone so the user has a discoverable way
        out beyond the F11 shortcut.
        """
        docks = [
            self._tree_dock,
            self._display_dock,
            self._pipeline_dock,
            self._conversion_dock,
            self._logs_dock,
            self._peaks_dock,
            self._profile_dock,
        ]
        if on:
            self._dock_visibility_before_fullscreen = {
                id(d): d.isVisible() for d in docks
            }
            for d in docks:
                d.setVisible(False)
        else:
            saved = getattr(self, "_dock_visibility_before_fullscreen", None)
            if saved is None:
                # No snapshot (e.g. user toggled the action via the
                # menu before any fullscreen entry). Fall back to
                # the mode-driven defaults so the layout doesn't end
                # up empty.
                self._apply_session_mode(self._active_session)
            else:
                for d in docks:
                    d.setVisible(bool(saved.get(id(d), True)))
            self._dock_visibility_before_fullscreen = None

    def _set_theme(self, theme: str) -> None:
        """Apply ``"dark"`` or ``"light"`` immediately, then persist
        via QSettings so next launch starts the same way.

        Swaps the full qdarkstyle stylesheet on the live QApplication,
        then **forces a re-polish of every widget** so the change is
        visible right away (Qt does not reliably restyle already-built
        widgets when the app stylesheet is replaced — without this the
        chrome only updates on the next restart). Finally it recolours
        the already-constructed pyqtgraph plots, whose colours are baked
        in at creation from ``pg.setConfigOption`` and so would otherwise
        stay on the old theme until the next render.
        """
        if theme not in ("dark", "light"):
            theme = "dark"
        from mlgidlab.theme import apply_dark_theme, apply_light_theme, pg_colors
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            if theme == "light":
                apply_light_theme(app)
            else:
                apply_dark_theme(app)
            # Force every existing widget to re-evaluate the new stylesheet.
            for w in app.allWidgets():
                try:
                    w.style().unpolish(w)
                    w.style().polish(w)
                    w.update()
                except Exception:
                    logger.debug("suppressed exception repolishing widget", exc_info=True)
        # Recolour the live pyqtgraph plots (config options only affect
        # newly-created items, so existing axes/backgrounds need an
        # explicit push).
        background, foreground = pg_colors(theme)
        for w in (
            getattr(self, "viewer", None),
            getattr(self, "profile_viewer", None),
            getattr(self, "_phase_views_window", None),
        ):
            fn = getattr(w, "apply_theme_colors", None)
            if fn is not None:
                try:
                    fn(background, foreground)
                except Exception:
                    logger.debug("suppressed exception in apply_theme_colors", exc_info=True)
        self._current_theme = theme
        # Persist.
        try:
            QSettings().setValue(self._THEME_KEY, theme)
        except Exception:
            logger.debug("suppressed exception in MainWindow._set_theme", exc_info=True)
            pass

    _THEME_KEY = "theme"

    def _reset_layout(self) -> None:
        """Restore the dock arrangement captured at cold startup.

        Two steps:

        1. ``restoreState`` with the cached snapshot — pops every
           dock back to its original area, undoes user drags, and
           re-applies original sizes.
        2. Re-run ``_apply_session_mode`` on the active session so
           the mode-specific tabify order (Display | Pipeline |
           Peaks | Logs for NeXus, Display | Conversion | Peaks |
           Logs for raw) is reapplied. The snapshot only captures
           the cold-start layout (no session), so without this
           second step a raw session reset would leave the user
           looking at the Pipeline dock instead of Conversion.
        """
        state = getattr(self, "_default_layout_state", None)
        if state is not None:
            try:
                self.restoreState(state)
            except Exception:
                # restoreState raises on a malformed state blob; we
                # generated this one ourselves so it shouldn't, but
                # don't take down the GUI if it does.
                logger.debug("suppressed exception in MainWindow._reset_layout", exc_info=True)
                pass
        # Reapply the session-mode-specific tab order + show/hide
        # toggles so Conversion/Pipeline visibility lines up with
        # the active session.
        self._apply_session_mode(self._active_session)

    def _build_settings_menu(self) -> None:
        """Build the top-level Settings menu.

        Houses application-wide preferences that don't justify a
        dedicated dock or main-toolbar slot. Currently exposes the
        frame-playback settings; future entries (e.g. default
        colormap, default render quality, log-verbosity toggle) hang
        off the same menu.

        The menu is built after View so it sits at the rightmost
        position, which is where users instinctively reach for
        Settings in cross-platform apps.
        """
        settings_menu = self.menuBar().addMenu("&Settings")
        self.action_playback_settings = QAction(
            "&Playback settings…", self
        )
        self.action_playback_settings.setToolTip(
            "Configure how the Display-dock Play button drives frame "
            "advance — either fixed time per frame or fixed total "
            "duration regardless of frame count."
        )
        self.action_playback_settings.triggered.connect(
            self._action_playback_settings
        )
        settings_menu.addAction(self.action_playback_settings)

    def _action_playback_settings(self) -> None:
        """Open the playback-settings dialog.

        On accept, persist the dialog's values via QSettings and, if
        the play timer is currently running, re-apply the new
        interval mid-flight so the change is felt immediately. The
        next press of Play also re-reads via ``_compute_play_schedule``
        so a setting change applied while paused still takes effect.
        """
        dlg = _SettingsDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        dlg.save_to_qsettings()
        # If playback is currently running, push the new schedule onto
        # the timer right away. The next tick will use it.
        if self._play_timer.isActive():
            interval, step = self._compute_play_schedule()
            self._play_timer.setInterval(interval)
            self._play_step = step
            if self._prefetch_worker is not None:
                self._prefetchUpdate.emit(
                    self.viewer.current_frame, True, step,
                )

    # ------------------------------------------------------------------
    # Help menu
    # ------------------------------------------------------------------

    def _build_help_menu(self) -> None:
        """Build the rightmost top-level Help menu.

        Three entries:
        - **Controls & shortcuts…** — modal reference of every
          keyboard shortcut and image-viewer interaction.
        - **About mlgidLAB…** — modal "About" dialog with versions.
        - **Copy diagnostics** — clipboard-friendly env/session/log
          dump for bug reports.
        """
        help_menu = self.menuBar().addMenu("&Help")
        self.action_controls = QAction("&Controls && shortcuts…", self)
        self.action_controls.setShortcut(QKeySequence("F1"))
        self.action_controls.setToolTip(
            "Reference for every keyboard shortcut, mouse interaction, "
            "and the manual-peak workflow."
        )
        self.action_controls.triggered.connect(self._show_controls)
        help_menu.addAction(self.action_controls)
        self.action_about = QAction("&About mlgidLAB…", self)
        self.action_about.triggered.connect(self._show_about)
        help_menu.addAction(self.action_about)
        self.action_copy_diagnostics = QAction("&Copy diagnostics", self)
        self.action_copy_diagnostics.setToolTip(
            "Copy environment info + active session details + recent "
            "log lines to the clipboard. Useful for bug reports."
        )
        self.action_copy_diagnostics.triggered.connect(self._copy_diagnostics)
        help_menu.addAction(self.action_copy_diagnostics)
        help_menu.addSeparator()
        self.action_check_updates = QAction("Check for &updates…", self)
        self.action_check_updates.setToolTip(
            "Check GitHub for a newer mlgidLAB release."
        )
        self.action_check_updates.triggered.connect(
            lambda: self._start_update_check(notify_uptodate=True)
        )
        help_menu.addAction(self.action_check_updates)
        supported, reason = update_check.self_update_supported()
        self.action_update_now = QAction("Update &now…", self)
        self.action_update_now.setEnabled(supported)
        self.action_update_now.setToolTip(
            "Check GitHub and, if a newer release exists, install it into "
            "the current environment."
            if supported
            else reason
        )
        # Re-check + install-if-newer in one step (no need to wait for the
        # startup banner). Same off-thread check as "Check for updates…",
        # but on finding a newer release it goes straight to the confirm +
        # install flow instead of showing the banner.
        self.action_update_now.triggered.connect(
            lambda: self._start_update_check(
                notify_uptodate=True, install_when_found=True
            )
        )
        help_menu.addAction(self.action_update_now)
        self.action_auto_update = QAction(
            "Automatically install updates", self
        )
        self.action_auto_update.setCheckable(True)
        self.action_auto_update.setEnabled(supported)
        self.action_auto_update.setChecked(
            supported
            and str(QSettings().value(self._AUTO_UPDATE_KEY, "false")).lower()
            == "true"
        )
        self.action_auto_update.setToolTip(
            "When a newer release is found on launch, install it into the "
            "current environment (after a confirmation)."
            if supported
            else reason
        )
        self.action_auto_update.toggled.connect(self._set_auto_update)
        help_menu.addAction(self.action_auto_update)

    def _set_auto_update(self, enabled: bool) -> None:
        QSettings().setValue(self._AUTO_UPDATE_KEY, bool(enabled))

    def _show_controls(self) -> None:
        """Modal reference for keyboard shortcuts + mouse + workflow.

        Plain QMessageBox.about so we get the title bar, an OK
        button, and rich-text rendering of the HTML body for free.
        Three sections — keyboard, image interactions, workflow —
        each presented as a small HTML table.
        """
        kbd_rows = [
            ("←  /  →", "Previous / next frame"),
            ("J  /  K", "Previous / next frame (Vim-style)"),
            ("Home  /  End", "First / last frame"),
            ("Ctrl+Z  /  Ctrl+Shift+Z (or Ctrl+Y)",
             "Undo / redo manual + geometry edits"),
            ("Ctrl+F", "Find peak by ID…"),
            ("Delete", "Delete the selected peak (image viewer) / "
             "remove the selected file (file browser)"),
            ("Ctrl+W", "Close (remove) the active file from the browser"),
            ("Esc", "Dismiss an in-progress manual draw"),
            ("F1", "Show this Controls reference"),
        ]
        mouse_rows = [
            ("Ctrl+Alt-drag (polar mode)",
             "Draw a manual peak rectangle"),
            ("Click a peak overlay",
             "Select the peak (any kind: manual / detected / fitted / matched)"),
            ("Drag ROI edges",
             "Resize the selected manual / detected / fitted peak"),
            ("LMB double-click on the image",
             "Reset aspect to Default (per-mode shape) and refit the zoom"),
            ("Mouse wheel on image",
             "Zoom in / out"),
            ("Click a row in the Peaks table",
             "Select the corresponding peak on the image"),
        ]
        flow_rows = [
            ("Manual peak workflow",
             "Ctrl+Alt-drag to label a candidate. Commit via "
             "<b>Add to detected</b> (box) or <b>Add to fitted</b> "
             "(1D Gaussian fit). Click off the box to abandon it "
             "(Ctrl+Z restores)."),
            ("Save fitted as ring",
             "Tick before Add to fitted to widen the angular extent "
             "to the full sweep — only meaningful for ring peaks."),
            ("Aspect (toolbar)",
             "<b>Default</b> (startup) uses a per-mode shape (Cartesian "
             "1:1, polar 2:1); <b>Fit</b> fills the panel; <b>Custom</b> "
             "locks a width:height ratio (e.g. <code>2</code> = twice as "
             "wide as tall). Scrolling over an axis switches to Custom and "
             "adjusts the ratio live (x wider, y taller); double-click the "
             "image to snap back to Default."),
            ("Display dock filter",
             "Type a CIF substring above the matched-structures "
             "list to hide non-matching rows + their image overlays."),
            ("Tools → Export figure…",
             "Non-modal window that drives "
             "<code>mlgidbase.plot_analysis_results</code> with a "
             "live preview. <b>Render preview</b> updates the image; "
             "<b>Save figure</b> writes the PNG."),
            ("Tools → Clear peaks → Reset all peaks",
             "Wipe detected + fitted + matched at three scopes "
             "(active entry, all entries, active frame). Manual "
             "peaks dropped from memory."),
        ]

        def _table(rows: list[tuple[str, str]]) -> str:
            cells = "".join(
                f"<tr><td style='padding-right:12px;white-space:nowrap'>"
                f"<b>{k}</b></td><td>{v}</td></tr>"
                for k, v in rows
            )
            return f"<table>{cells}</table>"

        body = (
            "<h3>mlgidLAB — Controls &amp; shortcuts</h3>"
            "<h4 style='margin-top:14px'>Keyboard</h4>"
            + _table(kbd_rows) +
            "<h4 style='margin-top:14px'>Mouse / image-viewer interactions</h4>"
            + _table(mouse_rows) +
            "<h4 style='margin-top:14px'>Workflow notes</h4>"
            + _table(flow_rows)
        )
        QMessageBox.about(self, "Controls & shortcuts", body)

    def _gather_versions(self) -> dict[str, str]:
        """Return a name → version-string map covering the modules
        most likely to matter in a bug report. Each lookup is
        guarded so a missing/older module reports ``(unavailable)``
        instead of breaking the diagnostics dump."""
        import platform
        import sys

        def _v(modname: str, attr: str = "__version__") -> str:
            try:
                mod = __import__(modname)
                # Some packages spell the version attr differently
                # (pyFAI uses both ``version`` and ``__version__``
                # depending on release).
                if attr == "__version__" and not hasattr(mod, "__version__"):
                    if hasattr(mod, "version"):
                        return str(mod.version)
                return str(getattr(mod, attr))
            except Exception:
                logger.debug("suppressed exception in MainWindow._gather_versions._v", exc_info=True)
                return "(unavailable)"

        try:
            from mlgidlab import __version__ as mlgidlab_version
        except Exception:
            logger.debug("suppressed exception in MainWindow._gather_versions", exc_info=True)
            mlgidlab_version = "(unavailable)"

        return {
            "mlgidLAB": mlgidlab_version,
            "Python": sys.version.split()[0],
            "OS": f"{platform.system()} {platform.release()}",
            "PySide6": _v("PySide6"),
            "Qt": _v("PySide6.QtCore", "__version__"),
            "numpy": _v("numpy"),
            "h5py": _v("h5py"),
            "silx": _v("silx"),
            "pyFAI": _v("pyFAI"),
            "pyqtgraph": _v("pyqtgraph"),
            "matplotlib": _v("matplotlib"),
            "mlgidbase": _v("mlgidbase"),
        }

    def _show_about(self) -> None:
        """Modal About dialog. Pure version info; no external links
        embedded yet (the project doesn't have a canonical docs URL
        we'd want to hardcode here)."""
        versions = self._gather_versions()
        rows = "".join(
            f"<tr><td><b>{name}</b></td><td>{ver}</td></tr>"
            for name, ver in versions.items()
        )
        body = (
            f"<h3>mlgidLAB {versions['mlgidLAB']}</h3>"
            "<p>Graphical interface for the mlgidBASE GIWAXS "
            "analysis pipeline.</p>"
            "<p>Use <b>Help → Copy diagnostics</b> to copy this "
            "environment plus the recent log lines for bug "
            "reports.</p>"
            f"<table>{rows}</table>"
        )
        QMessageBox.about(self, "About mlgidLAB", body)

    def _copy_diagnostics(self) -> None:
        """Build a plain-text diagnostics blob and put it on the
        clipboard. Sections:

        1. Versions — same map as the About dialog.
        2. Active session — file path, mode, entry, frame.
        3. Recent log lines — last 50 lines from the shared Logs
           dock, in chronological order.

        Nothing is uploaded; it's just text the user can paste.
        """
        import datetime

        # Section 1: versions
        versions = self._gather_versions()
        ver_lines = [f"  {k}: {v}" for k, v in versions.items()]

        # Section 2: active session
        session_lines = []
        sess = self._active_session
        if sess is None:
            session_lines.append("  (no file open)")
        else:
            try:
                session_lines.append(f"  kind:         {sess.kind}")
                session_lines.append(f"  display_path: {sess.display_path}")
                if hasattr(sess, "temp_path"):
                    session_lines.append(f"  temp_path:    {sess.temp_path}")
                if hasattr(sess, "raw_paths"):
                    for p in sess.raw_paths:
                        session_lines.append(f"  raw_path:     {p}")
                entry = (
                    self.entry_combo.currentText()
                    if hasattr(self, "entry_combo") else ""
                )
                session_lines.append(f"  entry:        {entry!r}")
                session_lines.append(
                    f"  frame:        {self.viewer.current_frame} / "
                    f"{max(0, self.viewer.n_frames - 1)}"
                )
                session_lines.append(f"  viewer mode:  {self.viewer._mode}")
            except Exception as exc:
                logger.debug("suppressed exception in MainWindow._copy_diagnostics", exc_info=True)
                session_lines.append(f"  (error gathering session info: {exc})")

        # Section 3: recent log lines
        log_lines: list[str] = []
        if hasattr(self, "_log_view"):
            try:
                blob = self._log_view.toPlainText()
                log_lines = blob.splitlines()[-50:]
            except Exception:
                logger.debug("suppressed exception in MainWindow._copy_diagnostics", exc_info=True)
                pass
        if not log_lines:
            log_lines = ["(no log lines)"]

        diagnostics = (
            f"mlgidLAB diagnostics — {datetime.datetime.now().isoformat(timespec='seconds')}\n\n"
            "=== Versions ===\n"
            + "\n".join(ver_lines)
            + "\n\n=== Active session ===\n"
            + "\n".join(session_lines)
            + "\n\n=== Recent log lines (last 50) ===\n"
            + "\n".join(log_lines)
            + "\n"
        )

        QApplication.clipboard().setText(diagnostics)
        # Tell the user it landed — status-bar message rather than a
        # modal because copying is a low-friction action.
        self.statusBar().showMessage(
            f"Copied {len(diagnostics)} chars of diagnostics to clipboard",
            5000,
        )

    # -- Update check + post-update changelog --

    def _run_startup_update_checks(self) -> None:
        """Post-update changelog + async online update check.

        Called once from ``main()`` after the window is shown — NOT from
        ``__init__``, so the test fixture (which builds the window directly)
        never triggers a network call or a changelog dialog. Both parts are
        best-effort and silent on failure.
        """
        self._maybe_show_changelog()
        self._start_update_check(notify_uptodate=False)

    def _start_update_check(
        self,
        *,
        notify_uptodate: bool = False,
        install_when_found: bool = False,
    ) -> None:
        """Fetch the latest GitHub release off-thread; show the banner if a
        newer version exists. ``notify_uptodate`` posts a status-bar
        confirmation when nothing newer is found (Help -> Check for updates).
        ``install_when_found`` skips the banner and goes straight to the
        confirm + install flow when a newer release exists and self-update is
        supported (Help -> Update now…).
        """
        if getattr(self, "_update_thread", None) is not None:
            return  # a check is already in flight
        self._update_notify_uptodate = notify_uptodate
        self._update_install_when_found = install_when_found
        self._update_thread = QThread(self)
        self._update_worker = _UpdateCheckWorker()
        self._update_worker.moveToThread(self._update_thread)
        self._update_thread.started.connect(self._update_worker.run)
        self._update_worker.finished.connect(self._on_update_check_finished)
        self._update_worker.finished.connect(self._update_thread.quit)
        self._update_thread.finished.connect(self._update_worker.deleteLater)
        self._update_thread.finished.connect(self._clear_update_thread)
        self._update_thread.start()

    def _clear_update_thread(self) -> None:
        thread = getattr(self, "_update_thread", None)
        if thread is not None:
            thread.deleteLater()
        self._update_thread = None
        self._update_worker = None

    @Slot(object)
    def _on_update_check_finished(self, result) -> None:
        from mlgidlab import __version__ as current

        if result is not None:
            tag, url = result
            if update_check.is_outdated(current, tag):
                self._latest_tag = tag
                self._latest_url = url
                supported, _ = update_check.self_update_supported()
                # Go straight to the install (via its confirm dialog, never
                # silently) when the user asked for it — Help -> Update now…
                # this run, or the opt-in "Automatically install updates"
                # setting — and self-update is a normal pip install. Otherwise
                # just surface the banner.
                auto = (
                    str(QSettings().value(self._AUTO_UPDATE_KEY, "false")).lower()
                    == "true"
                )
                want_install = (
                    getattr(self, "_update_install_when_found", False) or auto
                )
                if supported and want_install:
                    self._on_install_update_requested()
                else:
                    self._update_banner.show_update(
                        current, tag, url, can_install=supported
                    )
                return
        if getattr(self, "_update_notify_uptodate", False):
            self.statusBar().showMessage(
                f"mlgidLAB is up to date (v{current}).", 5000
            )

    # -- In-app self-update (pip install --upgrade) --

    def _on_install_update_requested(self) -> None:
        """Confirm, then upgrade to the last-found release in place.

        Wired to the banner's "Update now" button. The confirmation names the
        target version and warns that a restart is needed (the running
        process keeps the old code until relaunched).
        """
        tag = getattr(self, "_latest_tag", None)
        if not tag:
            return
        reply = QMessageBox.question(
            self,
            "Update mlgidLAB",
            f"Install {tag} into the current environment?\n\n"
            "mlgidLAB will download and pip-install the new version, then "
            "offer to restart.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Ok,
        )
        if reply == QMessageBox.StandardButton.Ok:
            self._start_update_install(tag)

    def _start_update_install(self, tag: str) -> None:
        """Run ``pip install --upgrade`` for ``tag`` off the GUI thread.

        Shows a busy progress dialog while pip runs; ``_on_update_install_
        finished`` closes it and reports success (restart prompt) or failure.
        """
        if getattr(self, "_install_thread", None) is not None:
            return  # an install is already in flight
        command = update_check.build_upgrade_command(tag)
        self._install_progress = QProgressDialog(
            f"Updating mlgidLAB to {tag}…", "", 0, 0, self
        )
        self._install_progress.setWindowTitle("Updating mlgidLAB")
        self._install_progress.setWindowModality(
            Qt.WindowModality.WindowModal
        )
        # No cancel: interrupting pip mid-write can leave a half-installed
        # package. Hide the button rather than leave a no-op "Cancel".
        self._install_progress.setCancelButton(None)
        self._install_progress.setMinimumDuration(0)
        self._install_progress.show()

        self._install_thread = QThread(self)
        self._install_worker = _UpdateInstallWorker(command)
        self._install_worker.moveToThread(self._install_thread)
        self._install_thread.started.connect(self._install_worker.run)
        self._install_worker.finished.connect(self._on_update_install_finished)
        self._install_worker.finished.connect(self._install_thread.quit)
        self._install_thread.finished.connect(self._install_worker.deleteLater)
        self._install_thread.finished.connect(self._clear_install_thread)
        self._install_thread.start()

    def _clear_install_thread(self) -> None:
        thread = getattr(self, "_install_thread", None)
        if thread is not None:
            thread.deleteLater()
        self._install_thread = None
        self._install_worker = None

    @Slot(int, str)
    def _on_update_install_finished(self, returncode: int, output: str) -> None:
        progress = getattr(self, "_install_progress", None)
        if progress is not None:
            progress.close()
            self._install_progress = None
        if returncode == 0:
            self._update_banner.hide()
            self._prompt_restart()
            return
        # Failure: surface the tail of pip's output plus the manual fallback.
        tail = "\n".join(output.strip().splitlines()[-15:]) or "(no output)"
        tag = getattr(self, "_latest_tag", "") or ""
        command = " ".join(update_check.build_upgrade_command(tag))
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Update failed")
        box.setText(
            "The update could not be installed. You can try again, or "
            "update manually from a terminal:"
        )
        box.setInformativeText(command)
        box.setDetailedText(tail)
        box.exec()

    def _prompt_restart(self) -> None:
        """Offer to relaunch so the freshly installed version loads.

        The upgrade landed on disk but the running interpreter still holds the
        old modules; only a restart picks up the new code.
        """
        reply = QMessageBox.question(
            self,
            "Update installed",
            "mlgidLAB was updated. Restart now to use the new version?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        # Relaunch the module entry point with the same interpreter, then quit
        # this instance. startDetached returns before we exit so the new
        # process survives our shutdown.
        QProcess.startDetached(sys.executable, ["-m", "mlgidlab"])
        QApplication.quit()

    def _maybe_show_changelog(self) -> None:
        """First launch after an update: show what changed since the last
        version this profile ran (bundled CHANGELOG). Records the current
        version so the popup only appears once per update.
        """
        from mlgidlab import __version__ as current

        settings = QSettings()
        raw = settings.value(self._LAST_SEEN_VERSION_KEY, None)
        last_seen = str(raw) if raw else None
        try:
            sections = update_check.whats_new(current, last_seen)
            if sections:
                self._show_changelog_dialog(current, sections)
        finally:
            settings.setValue(self._LAST_SEEN_VERSION_KEY, current)

    def _show_changelog_dialog(
        self, current: str, sections: list[tuple[str, str]]
    ) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(f"What's new in mlgidLAB {current}")
        dlg.resize(560, 480)
        layout = QVBoxLayout(dlg)
        intro = QLabel("mlgidLAB was updated. Here's what changed:", dlg)
        layout.addWidget(intro)
        browser = QTextBrowser(dlg)
        browser.setOpenExternalLinks(True)
        browser.setMarkdown(
            "\n\n".join(f"## {head}\n\n{body}" for head, body in sections)
        )
        layout.addWidget(browser)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok, dlg)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)
        dlg.exec()

    def _action_undo(self) -> None:
        # Covers manual add/remove, manual geom edits, and detected/fitted
        # geom edits. File-level deletes (delete_peak) are not undoable —
        # see the confirmation dialog in _on_delete_peak_requested.
        if hasattr(self, "viewer"):
            self.viewer.undo_last_action()

    def _action_redo(self) -> None:
        if hasattr(self, "viewer"):
            self.viewer.redo_last_action()

    def eventFilter(self, obj, ev):  # type: ignore[override]
        """Drive undo / redo from the keyboard even when the chord is an
        "ambiguous shortcut".

        The embedded pyFAI calibration dialog pulls in silx's
        ``MaskToolsWidget`` and pyFAI's peak-picking task, each of which
        binds Undo/Redo to the standard sequences (``Ctrl+Z`` / ``Ctrl+Y``,
        and on some silx/pyFAI versions the full standard Redo set, which
        includes ``Ctrl+Shift+Z``). Those widgets outlive the dialog
        (pyFAI's ``CalibrationContext`` is a singleton), so afterwards Qt
        finds two actions claiming the redo chord, logs "Ambiguous
        shortcut overload", and fires neither — the Edit-menu item still
        works, but the keyboard doesn't.

        We intercept the chord at the ``ShortcutOverride`` stage, before
        Qt's shortcut map runs, and run our own handler. Guarded so the
        calibration dialog's own mask undo/redo and text-field undo are
        left alone: only act while this window holds focus and the focus
        is not a text-editing widget.
        """
        # Once closed (or mid-teardown), do nothing — the filter may still
        # be installed on the app and our child widgets may be gone. Pytest
        # creates many windows per process; a stale filter touching a
        # destroyed ``self`` would otherwise raise during shutdown.
        if getattr(self, "_closed", False):
            return super().eventFilter(obj, ev)
        et = ev.type()
        if et not in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress):
            return super().eventFilter(obj, ev)
        fw = QApplication.focusWidget()
        if fw is not None:
            if fw.window() is not self:
                return super().eventFilter(obj, ev)  # a child dialog owns the key
            if isinstance(fw, (QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit)):
                return super().eventFilter(obj, ev)  # native text undo/redo
        elif not self.isActiveWindow():
            return super().eventFilter(obj, ev)
        mods = ev.modifiers()
        if not (mods & Qt.KeyboardModifier.ControlModifier) or (
            mods & Qt.KeyboardModifier.AltModifier
        ):
            return super().eventFilter(obj, ev)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        key = ev.key()
        is_redo = (key == Qt.Key.Key_Z and shift) or (
            key == Qt.Key.Key_Y and not shift
        )
        is_undo = key == Qt.Key.Key_Z and not shift
        if not (is_redo or is_undo):
            return super().eventFilter(obj, ev)
        if et == QEvent.Type.ShortcutOverride:
            # Claim the key so Qt skips (ambiguous) shortcut resolution and
            # re-delivers it as a plain KeyPress, handled on the next pass.
            ev.accept()
            return True
        (self._action_redo if is_redo else self._action_undo)()
        return True

    def _action_find_peak(self) -> None:
        """Modal: pick Kind + ID, select the peak in the viewer.

        Searches the current frame first, then scans every other
        frame in the active entry; on a hit in another frame the
        viewer jumps to that frame before selecting. Matched peaks
        are excluded — matched IDs reference fitted peak ids so the
        Fitted kind covers that case too, and the per-structure
        selection model would need a separate UI.
        """
        if self.session is None:
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Find peak by ID")
        form = QFormLayout(dlg)
        kind_combo = QComboBox()
        kind_combo.addItems(["Detected", "Fitted"])
        form.addRow("Kind:", kind_combo)
        id_spin = QSpinBox()
        id_spin.setRange(0, 999999)
        # Sensible default: continue from whatever's currently
        # selected so repeated invocations step through IDs.
        cur_sel = self.viewer.selected_peak
        if cur_sel is not None and cur_sel.kind in ("detected", "fitted"):
            kind_combo.setCurrentText(cur_sel.kind.capitalize())
            id_spin.setValue(int(cur_sel.peak_id))
        form.addRow("ID:", id_spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        kind = kind_combo.currentText().lower()
        peak_id = int(id_spin.value())
        self._find_and_select_peak(entry, kind, peak_id)

    def _find_and_select_peak(self, entry: str, kind: str, peak_id: int) -> None:
        """Locate (entry, *, kind, peak_id) across all frames and
        select it. Tries the viewer's in-memory peak tables first
        (cheap) before falling back to per-frame disk reads."""
        current_frame = self.viewer.current_frame
        # In-memory current-frame lookup.
        peaks_now = self.viewer._frame_peaks.get(current_frame, {})
        table = peaks_now.get(kind)
        if table is not None:
            ids = [int(x) for x in table.ids]
            if peak_id in ids:
                self._select_table_row(current_frame, kind, table, ids.index(peak_id))
                return
        # Scan other frames via file_model.
        n_frames = self.viewer.n_frames
        for frame in range(n_frames):
            if frame == current_frame:
                continue
            try:
                peaks = file_model.load_peaks(
                    self.session.temp_path, entry, frame,
                )
            except Exception:
                logger.debug("suppressed exception in MainWindow._find_and_select_peak", exc_info=True)
                continue
            table = peaks.get(kind)
            if table is None or len(table) == 0:
                continue
            ids = [int(x) for x in table.ids]
            if peak_id in ids:
                # Jump to that frame, then select.
                self.viewer.set_frame(frame)
                # _frame_peaks for the new frame is populated lazily
                # by _load_entry_into_viewer at open time; re-read
                # in case the viewer's per-frame cache hasn't been
                # touched yet for this run.
                self._select_table_row(frame, kind, table, ids.index(peak_id))
                return
        QMessageBox.information(
            self,
            "Find peak",
            f"No {kind} peak with id={peak_id} found in entry {entry!r}.",
        )

    def _select_table_row(
        self, frame: int, kind: str, table, idx: int,
    ) -> None:
        """Build a ``SelectedPeak`` from row ``idx`` of ``table`` and
        push it into the viewer. Mirrors the construction inside
        ``GIWAXSImageViewer._on_select_at`` for detected/fitted hits."""
        try:
            score = float(table.score[idx])
        except Exception:
            logger.debug("suppressed exception in MainWindow._select_table_row", exc_info=True)
            score = None
        sel = SelectedPeak(
            kind=kind,
            frame=frame,
            peak_id=int(table.ids[idx]),
            radius=float(table.radius[idx]),
            angle=float(table.angle[idx]),
            radius_width=float(table.radius_width[idx]),
            angle_width=float(table.angle_width[idx]),
            is_ring=bool(table.is_ring[idx]),
            score=score,
        )
        self.viewer._set_selected(sel)

    def _build_file_menu(self, file_menu) -> None:

        # Single Open action — file content is auto-classified as
        # NeXus or raw inside ``_action_open`` so users don't have to
        # pick the right entry point. Raw files are bundled into one
        # ``RawSession`` in the same way the old "Open raw" action did.
        self.action_open = QAction("&Open…", self)
        self.action_open.setShortcut(QKeySequence.StandardKey.Open)
        self.action_open.triggered.connect(self._action_open)
        file_menu.addAction(self.action_open)

        # Recent-files submenu — populated lazily on aboutToShow so the
        # missing-file filter stays accurate across sessions.
        self._recent_menu = file_menu.addMenu("Open &recent")
        self._recent_menu.setToolTipsVisible(True)
        self._recent_menu.aboutToShow.connect(self._refresh_recent_files_menu)
        # Build once now so the menu shows real entries on first open
        # (aboutToShow only fires when the user actually opens the
        # submenu — but the parent File menu's expansion looks better
        # if the count is right from the start).
        self._refresh_recent_files_menu()

        self.action_save = QAction("&Save", self)
        self.action_save.setShortcut(QKeySequence.StandardKey.Save)
        self.action_save.triggered.connect(self._action_save)
        file_menu.addAction(self.action_save)

        self.action_save_as = QAction("Save &As…", self)
        self.action_save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.action_save_as.triggered.connect(self._action_save_as)
        file_menu.addAction(self.action_save_as)

        self.action_close_file = QAction("&Close", self)
        self.action_close_file.setShortcut(QKeySequence.StandardKey.Close)
        self.action_close_file.triggered.connect(self._action_close_file)
        file_menu.addAction(self.action_close_file)

        file_menu.addSeparator()

        action_exit = QAction("E&xit", self)
        action_exit.setShortcut(QKeySequence.StandardKey.Quit)
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_exit)

    # -- Recent files (QSettings-backed) --

    _RECENT_FILES_KEY = "recentFiles"
    _MAX_RECENT_FILES = 10

    # Playback settings (persisted via QSettings). See the module-level
    # PLAYBACK_* constants for defaults and bounds.
    _PLAYBACK_MODE_KEY = "playbackMode"
    _PLAYBACK_FRAME_MS_KEY = "playbackFrameIntervalMs"
    _PLAYBACK_TOTAL_S_KEY = "playbackTotalTimeS"
    # Last mlgidLAB version this profile ran, for the post-update changelog.
    _LAST_SEEN_VERSION_KEY = "lastSeenVersion"
    # Opt-in: install a newer release automatically on launch (Help menu).
    _AUTO_UPDATE_KEY = "autoUpdate"

    def _load_recent_files(self) -> list[dict]:
        """Return the persisted recent-files list as a list of dicts.

        Each entry is ``{"type": "nexus"|"raw", "path": str}``. The
        list is stored as a JSON string in QSettings to keep the
        serialization explicit and robust across PySide/Qt platforms
        (raw QStringList round-tripping has bitten us before).
        """
        settings = QSettings()
        blob = settings.value(self._RECENT_FILES_KEY, "[]")
        if not isinstance(blob, str):
            return []
        try:
            data = json.loads(blob)
        except Exception:
            logger.debug("suppressed exception in MainWindow._load_recent_files", exc_info=True)
            return []
        if not isinstance(data, list):
            return []
        return [
            d for d in data
            if isinstance(d, dict)
            and d.get("type") in ("nexus", "raw")
            and isinstance(d.get("path"), str)
        ]

    def _save_recent_files(self, items: list[dict]) -> None:
        QSettings().setValue(self._RECENT_FILES_KEY, json.dumps(items))

    def _add_recent_file(self, path: str | Path, kind: str) -> None:
        """Push ``path`` onto the front of the recent list.

        Move-to-front semantics: if ``path`` is already in the list it
        gets bubbled up to the top instead of duplicated. The list is
        capped at ``_MAX_RECENT_FILES``.
        """
        if kind not in ("nexus", "raw"):
            return
        path_str = str(path)
        items = self._load_recent_files()
        items = [i for i in items if i.get("path") != path_str]
        items.insert(0, {"type": kind, "path": path_str})
        items = items[: self._MAX_RECENT_FILES]
        self._save_recent_files(items)
        self._refresh_recent_files_menu()

    def _refresh_recent_files_menu(self) -> None:
        """Rebuild the submenu from the persisted list.

        Existence is intentionally NOT checked here: this runs on the
        menu's ``aboutToShow`` (GUI thread), and ``Path.exists()`` over a
        slow / network share froze the window just opening the submenu. A
        file that has since moved is handled when the user clicks it
        (``_open_recent`` warns + prunes), so stale rows cost nothing
        until then.
        """
        self._recent_menu.clear()
        items = self._load_recent_files()
        if not items:
            empty = QAction("(no recent files)", self)
            empty.setEnabled(False)
            self._recent_menu.addAction(empty)
            return
        for entry in items:
            path = entry["path"]
            kind = entry["type"]
            basename = Path(path).name
            # NeXus rows show plain basename; raw rows get a "[raw] "
            # prefix so the two are distinguishable without an icon.
            label = basename if kind == "nexus" else f"[raw]  {basename}"
            action = QAction(label, self)
            # Tooltip shows the full path so the user can disambiguate
            # files with the same basename living in different folders.
            action.setToolTip(path)
            action.triggered.connect(
                lambda checked=False, p=path, k=kind: self._open_recent(p, k)
            )
            self._recent_menu.addAction(action)
        self._recent_menu.addSeparator()
        clear_action = QAction("Clear recent files", self)
        clear_action.triggered.connect(self._clear_recent_files)
        self._recent_menu.addAction(clear_action)

    def _clear_recent_files(self) -> None:
        self._save_recent_files([])
        self._refresh_recent_files_menu()

    def _open_recent(self, path: str, kind: str) -> None:
        """Open a file from the Recent-files submenu.

        ``kind`` is the recorded classification — kept for the menu
        labels, but the open itself goes through the same worker queue
        for both kinds (the worker re-classifies by content). Drops the
        entry from the list if the file has gone missing since it was
        recorded.
        """
        p = Path(path)
        # Show the indicator BEFORE the existence stat. On an SFTP mount the
        # recent file may be on slow/cold storage, so ``p.exists()`` (a
        # network round-trip) and the load that follows can briefly freeze
        # the window; painting the indicator first means the click visibly
        # registers instead of looking like nothing happened.
        self._show_open_progress(p.name)
        if not p.exists():
            self._dismiss_open_progress()
            QMessageBox.warning(
                self,
                "Recent files",
                f"File no longer exists:\n{path}\n\n"
                "Removing from the recent list.",
            )
            items = [i for i in self._load_recent_files() if i.get("path") != path]
            self._save_recent_files(items)
            self._refresh_recent_files_menu()
            return
        # Both kinds route through the CopyWorker queue. Raw recents used
        # to take a synchronous shortcut here (RawSession.open + insertFile
        # + activate inline) — on a big beamtime file that froze the whole
        # window for the metadata walk and the first full-stack read. The
        # worker re-classifies (cheap relative to the work it moves off
        # this thread) and its raw-entry walk doubles as the combo /
        # Conversion-panel listing, so nothing is scanned twice.
        self._open_queue.append(p)
        self._process_open_queue()

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
        self._tree_refresh_btn.setText("⟳ Refresh")
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
        # Compact "idx / max" readout. The "Frame" word was dropped to
        # save toolbar space — its meaning is obvious from context
        # (next to play / prev / next icons and a slider).
        self.frame_label = QLabel("—")
        self.frame_label.setMinimumWidth(48)
        self.frame_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        # Play / pause toggle. Drives a QTimer that calls
        # viewer.set_frame(current + 1) on every tick; stops when the
        # last frame is reached. Standard-icon based so qdarkstyle
        # picks up the right colour automatically.
        self.play_button = QToolButton()
        self.play_button.setCheckable(True)
        self._icon_play = self.style().standardIcon(
            QStyle.StandardPixmap.SP_MediaPlay
        )
        self._icon_pause = self.style().standardIcon(
            QStyle.StandardPixmap.SP_MediaPause
        )
        self.play_button.setIcon(self._icon_play)
        self.play_button.setToolTip(
            "Play frames from the current position to the end.\n"
            "Stops at the last frame; click again to pause."
        )
        self.play_button.toggled.connect(self._on_play_toggled)
        # Previous / next single-step buttons. Step by one frame and
        # clamp at boundaries (the buttons disable themselves at
        # frame 0 / last via ``_refresh_frame_nav_enabled``).
        self.prev_frame_button = QToolButton()
        self.prev_frame_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowBack)
        )
        self.prev_frame_button.setToolTip("Previous frame")
        self.prev_frame_button.setAutoRepeat(True)
        self.prev_frame_button.setAutoRepeatDelay(300)
        self.prev_frame_button.setAutoRepeatInterval(80)
        self.prev_frame_button.clicked.connect(self._on_prev_frame_clicked)
        self.next_frame_button = QToolButton()
        self.next_frame_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowForward)
        )
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
        # Hand the controls to the image viewer's toolbar. Order
        # reads left-to-right: prev / play / next / slider / label.
        self.viewer.insert_frame_controls([
            self.prev_frame_button,
            self.play_button,
            self.next_frame_button,
            self.frame_slider,
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
        # Lives in its own field so we can find/remove the placeholder row.
        self._matched_empty_label: QLabel | None = None
        # Shown when a non-empty filter hides every row (distinct
        # from the "no matched solutions" empty-list label).
        self._matched_filter_empty_label: QLabel | None = None
        self._refresh_matched_panel(0, [])
        self.viewer.matchedStructuresChanged.connect(self._refresh_matched_panel)

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
        self.pipeline_panel.runRequested.connect(self._on_run_requested)
        self.pipeline_panel.parseCifsRequested.connect(self._on_parse_cifs_requested)
        self._pipeline_dock = QDockWidget("Pipeline", self)
        self._pipeline_dock.setWidget(self.pipeline_panel)
        self._pipeline_dock.setObjectName("PipelineDock")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._pipeline_dock)
        self.tabifyDockWidget(self._display_dock, self._pipeline_dock)

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
        self.tabifyDockWidget(self._pipeline_dock, self._conversion_dock)
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
        self._drag_pygidfit_timer = QTimer(self)
        self._drag_pygidfit_timer.setSingleShot(True)
        self._drag_pygidfit_timer.setInterval(150)
        self._drag_pygidfit_timer.timeout.connect(self._refresh_2d_preview)
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

    # -- Actions --

    def _action_open(self) -> None:
        """Unified open: pick HDF5 files and auto-classify each as NeXus or raw.

        Multi-select is supported. Each picked file is classified by
        content (not extension) inside ``_open_paths`` — NeXus files
        stream through the per-file copy worker queue, raw files are
        bundled into a single shared ``RawSession`` matching the old
        Open-raw bulk behaviour. Files that match neither classifier
        are reported in the log and skipped.
        """
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open file(s)", "", OPEN_FILTER
        )
        if not paths:
            return
        self._open_paths([Path(p) for p in paths])

    def _open_paths(self, paths: list[Path]) -> None:
        """Open a mixed batch of NeXus + raw files, classifying each in the
        background.

        The Open click is instant: existing paths are queued and the
        worker is started immediately (with the bottom-left loading bar).
        Each ``CopyWorker`` classifies its file off the GUI thread and, for
        NeXus, copies + pre-warms it. NeXus files install as they finish;
        raw and unclassifiable files are collected and handled once the
        queue drains (``_finalize_open_batch`` — raw bundles into one
        ``RawSession`` so the Conversion panel applies one config to the
        batch). Used by both File → Open and the drag-and-drop handler.
        """
        queued = False
        for p in paths:
            if p.is_dir():
                # A dropped/opened folder becomes ONE image stack: its fabio
                # images in natural filename order (so img_2 precedes img_10).
                # Folders without any image files are reported as rejected.
                imgs = sorted(
                    (
                        q
                        for q in p.iterdir()
                        if q.is_file() and file_model.is_fabio_image(q)
                    ),
                    key=_natural_key,
                )
                if imgs:
                    self._open_queue.extend(imgs)
                    queued = True
                else:
                    self._pending_rejected.append(p)
            elif p.is_file():
                self._open_queue.append(p)
                queued = True
            else:
                self._pending_rejected.append(p)
        if queued:
            self._process_open_queue()
        elif self._pending_rejected:
            # Nothing to load (no worker will run) — flush the rejects now.
            self._finalize_open_batch()

    def _refresh_tree_raw_paths(self) -> None:
        """Push the active set of raw filesystem paths into the tree model.

        Called whenever the session list changes so the file browser's
        custom raw-icon stays accurate. NeXus sessions don't need to be
        listed — anything the model hasn't been told about as raw
        renders with the default NeXus icon.
        """
        model = self.tree.findHdf5TreeModel()
        if not isinstance(model, _MlgidHdf5TreeModel):
            return
        raw_paths: set[str] = set()
        for s in self._sessions:
            if s.kind == "raw" and isinstance(s, RawSession):
                for p in s.raw_paths:
                    raw_paths.add(str(p))
        model.set_raw_paths(raw_paths)

    def _process_open_queue(self) -> None:
        """Kick off the next queued open if no copy is in flight.

        When the queue is exhausted the batch is finalized (raw files
        bundled, rejects logged, loading bar hidden).
        """
        if self._thread is not None:
            return
        if not self._open_queue:
            self._finalize_open_batch()
            return
        self._open_path(self._open_queue.pop(0))

    def _finalize_open_batch(self) -> None:
        """Once the open queue drains: bundle any raw files into one
        ``RawSession``, log unclassifiable files, and hide the loading bar.
        The detector-dataset lists CopyWorker found per file ride along on
        the session (``_raw_entries_cache``) so activation reads metadata
        the worker already has instead of re-walking the files."""
        if self._pending_rejected:
            self.pipeline_panel.append_log(
                "Could not classify (no q-signal entries, no raw 3D "
                "detector datasets): "
                + ", ".join(str(p) for p in self._pending_rejected)
            )
            self._pending_rejected = []
        # Dismiss BEFORE bundling: raw activation dispatches an async
        # first-entry load that re-shows the bar ("Loading <entry>…"),
        # and ``_on_raw_entry_loaded`` dismisses it when the frame lands.
        # Dismissing last would hide that in-flight indicator.
        self._dismiss_open_progress()
        if self._pending_raw_paths:
            raw_paths = self._pending_raw_paths
            self._pending_raw_paths = []
            entry_cache = self._pending_raw_entry_cache
            self._pending_raw_entry_cache = {}
            try:
                session = RawSession.open(raw_paths)
            except Exception as exc:
                QMessageBox.critical(self, "Open failed", str(exc))
            else:
                session._raw_entries_cache = entry_cache  # type: ignore[attr-defined]
                model = self.tree.findHdf5TreeModel()
                for raw_path in session.raw_paths:
                    model.insertFile(str(raw_path))
                self._sessions.append(session)
                self._set_active_session(session)
                for raw_path in session.raw_paths:
                    self._add_recent_file(raw_path, "raw")
                self._refresh_tree_raw_paths()

    def _show_open_progress(self, name: str) -> None:
        """Show the bottom-left open indicator and force it to paint NOW.

        The synchronous ``repaint`` matters on a high-latency (SFTP) mount:
        the file stat + worker startup that follow can briefly starve the
        event loop, so a plain ``show()`` (which only schedules a paint for
        the next event-loop pass) would leave the user staring at an
        unchanged window — "no progress bar". Painting the widgets here,
        while the GUI thread is still free, guarantees the indicator is
        visible the instant the open is requested.
        """
        self._sb_open_label.setText(f"Loading {name}…")
        self._sb_open_label.show()
        # Indeterminate until the worker's first real tick (a prior open
        # may have left the bar determinate at 100).
        self._sb_open_bar.setRange(0, 0)
        self._sb_open_bar.show()
        # Flush layout + paint events (NOT user input — ExcludeUserInputEvents
        # avoids re-entrancy from a stray click/key) so the just-shown
        # widgets are laid out and drawn NOW, before the blocking stat /
        # worker startup, rather than on the next event-loop pass.
        QApplication.processEvents(
            QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
        )

    def _dismiss_open_progress(self) -> None:
        """Hide the bottom-left open-progress bar once the queue drains."""
        self._sb_open_label.hide()
        self._sb_open_bar.hide()
        self._sb_open_label.setText("")

    def _action_save(self) -> None:
        self._save(confirm=True)

    def _save(self, confirm: bool, session: BaseSession | None = None) -> bool:
        """Overwrite the original from the temp. Returns True on success.

        Raw sessions have no writable temp copy — Save and Save As are
        no-ops for them. The action is also disabled in the menu, but
        guard here too in case a shortcut fires.
        """
        target = session if session is not None else self._active_session
        if target is None or target.kind != "nexus":
            return False
        assert isinstance(target, NexusSession)
        if confirm:
            reply = QMessageBox.question(
                self,
                "Save",
                f"Overwrite the original file?\n\n{target.original_path}",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Save:
                return False
        try:
            target.save()
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return False
        self._update_title()
        return True

    def _action_save_as(self) -> None:
        if self.session is None or self.session.kind != "nexus":
            return
        assert isinstance(self.session, NexusSession)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save As",
            str(self.session.original_path),
            NEXUS_FILTER,
        )
        if not path:
            return
        # Release the viewer's FrameSource handle *before* the rename
        # so the file isn't open when shutil.copy2 + rename runs (matters
        # on Windows; harmless on Linux). silx is also detached so the
        # tree can be rebuilt at the new basename.
        with self._detached_silx_tree():
            try:
                self.session.save_as(Path(path))
            except Exception as exc:
                QMessageBox.critical(self, "Save As failed", str(exc))
                # The context manager still reattaches on this early
                # return, so the viewer keeps reading rather than being
                # stranded with the file detached and nothing written.
                return
            # Save As renamed the temp file to match the new basename.
            # The FrameSource was created against the old basename and
            # would otherwise fail to reopen — point it at the new path
            # before the context manager's reattach reacquires it.
            if (
                self.viewer._frame_source is not None
                and isinstance(self.session, NexusSession)
            ):
                self.viewer._frame_source.relocate(self.session.temp_path)
        self._update_title()
        # The user just wrote a new file at ``path``; surface it in the
        # recent menu so it's reopenable from the next session.
        self._add_recent_file(path, "nexus")

    def _action_close_file(self) -> None:
        """Close just the currently-active file. Other files stay open."""
        active = self._active_session
        if active is None:
            return
        if not self._confirm_discard_changes(active):
            return
        self._close_session(active)

    # -- Session lifecycle --

    def _open_path(self, path: Path) -> None:
        # Additive: never tear down existing sessions. The new file simply
        # gets appended to the file browser when the copy completes.
        self._thread = QThread(self)
        self._worker = CopyWorker(path)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_open_finished)
        # Queued (cross-thread) stage ticks update the bar's LABEL while the
        # indeterminate bar marches; the worker phase is GUI-responsive so
        # the march stays smooth.
        self._worker.progress.connect(self._on_open_progress)

        # Non-modal progress: a small bar + label in the bottom-left status
        # bar (NOT a window-modal dialog). The window stays fully usable
        # while the file scans + copies and its first frame warms off the
        # GUI thread; the file appears in the browser only once the load
        # finishes (``_on_open_finished``). Replaces the old WindowModal
        # ``QProgressDialog`` that dimmed + froze the whole window — for a
        # huge external-link master that froze long enough to trip the OS
        # "force quit or wait" prompt.
        self._show_open_progress(path.name)

        self._thread.start()

    def _on_open_progress(self, percent: int, label: str) -> None:
        """Update the bottom-left open bar from a ``CopyWorker`` progress
        tick (queued from the worker thread).

        The bar starts indeterminate (``_show_open_progress``) and flips
        determinate on the first worker tick — the worker reports real
        fractions (copy bytes, raw-scan groups), so the user sees actual
        progress instead of an endless march.
        """
        if label:
            self._sb_open_label.setText(label)
        if 0 <= percent <= 100:
            if self._sb_open_bar.maximum() == 0:
                self._sb_open_bar.setRange(0, 100)
            self._sb_open_bar.setValue(percent)

    def _on_open_finished(self, result: dict) -> None:
        """Handle one ``CopyWorker`` result (classify + open done off the
        GUI thread). NeXus files install + render now (from the warm
        pre-load); raw and unclassifiable files are collected for
        ``_finalize_open_batch`` when the queue drains."""
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread.deleteLater()
            self._thread = None
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        # The loading bar spans the whole queue; it's hidden in
        # ``_finalize_open_batch`` once the queue empties.

        error = result.get("error")
        kind = result.get("kind")
        session = result.get("session")
        prewarm = result.get("prewarm")
        path = result.get("path")

        if error is not None:
            QMessageBox.critical(self, "Open failed", str(error))
            if prewarm is not None:  # opened-but-unused source would leak
                self._release_prewarm(prewarm)
        elif kind == "nexus" and session is not None:
            # ``prewarm`` is ``(first_entry, FrameSource)`` warmed off the
            # GUI thread by CopyWorker (or None). Stash it on the session so
            # the first ``_load_entry_into_viewer`` renders from the warm
            # handle instead of re-reading frame 0 on the GUI thread.
            session._prewarm = prewarm  # type: ignore[attr-defined]
            # The worker already listed the q-entries (off the GUI thread,
            # the same external-link scan that drives the bar). Stash them
            # so ``_populate_entries`` fills the combo without re-scanning
            # every external scan on the GUI thread — that GUI-thread
            # re-scan was the residual open freeze on big masters.
            entries = result.get("entries")
            if entries is not None:
                session._entries_cache = entries  # type: ignore[attr-defined]
            # First frame's peaks, read off-thread by the worker, so the
            # initial render installs overlays without an SFTP read on the
            # GUI thread (consumed in ``_load_entry_into_viewer``).
            session._prewarm_overlays = result.get("prewarm_overlays")  # type: ignore[attr-defined]
            # pygid normalization (0-D angle_of_incidence → 1-D, per-frame
            # analysis groups) is NOT done here — it's deferred and run
            # lazily for one entry just before its first pipeline run (see
            # ``_ensure_entry_normalized``), so opening never touches every
            # linked scan.
            self._sessions.append(session)
            self.tree.findHdf5TreeModel().insertFile(str(session.temp_path))
            # Re-opening a path that's already open REPLACES the old
            # instance: its temp copy predates the new bytes (the
            # conversion append/overwrite-then-auto-open case), and two
            # instances of one file only confuse. A dirty old instance
            # gets the usual save prompt first; cancelling keeps both.
            self._close_duplicate_sessions(session)
            # Newly-opened file becomes the active one — the user almost
            # always wants to inspect what they just opened.
            self._set_active_session(session)
            # Remember the original (not the temp) so the recent menu
            # reopens the file at its real location next session.
            self._add_recent_file(session.original_path, "nexus")
            self._refresh_tree_raw_paths()
            # If the pre-warm wasn't consumed (e.g. no q-entries to land
            # on), release its handle so it doesn't leak.
            leftover = getattr(session, "_prewarm", None)
            if leftover is not None:
                self._release_prewarm(leftover)
                session._prewarm = None  # type: ignore[attr-defined]
        elif kind == "raw" and path is not None:
            # Bundled into one RawSession when the batch finishes. The
            # worker already walked the file for its detector datasets —
            # keep that list so activation never re-walks on the GUI
            # thread (``_populate_raw_entries`` consumes it).
            self._pending_raw_paths.append(path)
            raw_entries = result.get("raw_entries")
            if raw_entries is not None:
                self._pending_raw_entry_cache[str(path)] = raw_entries
        elif path is not None:
            self._pending_rejected.append(path)

        # Keep draining the queue regardless of this open's outcome so a
        # single bad file in a batch doesn't strand the rest.
        self._process_open_queue()

    def _close_duplicate_sessions(self, session: BaseSession) -> None:
        """Close any OLDER NeXus session open on ``session``'s original path.

        Re-opening a file (typically: the conversion just appended an
        entry/frames to, or replaced, an output file that was already
        open, and the auto-open re-opened it) must not leave two
        instances — the old one's working copy is stale. Dirty old
        instances get the standard save/discard/cancel prompt; on
        cancel the old instance is kept alongside the new one.
        """
        try:
            target = session.original_path.resolve()
        except OSError:
            return
        for old in list(self._sessions):
            if old is session or old.kind != "nexus":
                continue
            try:
                old_path = old.display_path.resolve()
            except OSError:
                continue
            if old_path != target:
                continue
            if not self._confirm_discard_changes(old):
                continue  # user cancelled — keep the old instance too
            self._close_session(old)

    @staticmethod
    def _release_prewarm(prewarm: object) -> None:
        """Release a ``(entry, FrameSource)`` pre-warm that won't be used."""
        try:
            prewarm[1].release()  # type: ignore[index]
        except Exception:
            logger.debug("suppressed exception releasing prewarm", exc_info=True)

    def _close_session(self, session: BaseSession) -> None:
        """Remove ``session`` from the window: tear down its tree entry,
        delete its temp dir, and pick a new active if it was the active one.
        """
        was_active = session is self._active_session
        if session not in self._sessions:
            return
        # Stop playback before pulling the file out from under the
        # viewer — the timer's next tick would otherwise read from a
        # released FrameSource.
        if was_active:
            self._pause_playback()
        self._sessions.remove(session)
        # An inactive session may hold a stashed FrameSource (parked for
        # instant re-activation) — release it before the temp dir goes.
        stash = getattr(session, "_prewarm", None)
        if stash is not None:
            self._release_prewarm(stash)
            session._prewarm = None  # type: ignore[attr-defined]
        # Drop this file's lazy-normalization record (keyed by temp_path).
        self._normalized_entries.pop(str(session.temp_path), None)
        # silx exposes no "remove single file" API on Hdf5TreeModel, so we
        # rebuild the tree from the remaining sessions. Cheap — sessions
        # are typically <5 and the model just re-opens HDF5 files.
        with self._detached_silx_tree():
            if was_active:
                # Active state is tied to viewer/entry_combo content — drop
                # it before swapping so we don't leak the old session's
                # overlays.
                self.viewer.clear()
                self.viewer.clear_history()
                self.profile_viewer.clear()
                self.peaks_table_panel.clear()
                self.entry_combo.blockSignals(True)
                self.entry_combo.clear()
                self.entry_combo.blockSignals(False)
                self._active_session = None
            session.close()
        # Closed session may have been raw — rebuild the tree's raw set.
        self._refresh_tree_raw_paths()
        if was_active:
            new_active = self._sessions[-1] if self._sessions else None
            if new_active is not None:
                self._set_active_session(new_active)
            else:
                self._update_title()
                self._update_actions()

    def _refresh_file_tree(self) -> None:
        """Manually re-sync every open session with the filesystem (the
        file-browser Refresh button / F5).

        Per session: a DELETED original closes the session (temp copy
        cleaned up) unless it has unsaved changes — those stay open, with
        a warning, so the user can Save As to rescue the edits. An
        original that CHANGED on disk is reloaded into the temp copy when
        the session is clean; with unsaved changes it is left untouched
        and the conflict is reported. Raw sessions close when every one
        of their input files is gone (partially-missing batches are
        reported but kept). Blocked while a pipeline run is in flight —
        reloading the active temp file under the worker would corrupt
        the run.
        """
        if self._pipe_thread is not None:
            self.statusBar().showMessage(
                "Refresh skipped — a pipeline run is in progress.", 5000
            )
            return
        closed: list[str] = []
        reloaded: list[str] = []
        deleted_dirty: list[str] = []
        conflicts: list[str] = []
        partial_raw: list[str] = []
        for session in list(self._sessions):
            if session.kind == "raw":
                missing = [p for p in session.raw_paths if not p.exists()]  # type: ignore[attr-defined]
                if missing and len(missing) == len(session.raw_paths):  # type: ignore[attr-defined]
                    self._close_session(session)
                    closed.append(session.display_path.name)
                elif missing:
                    partial_raw.extend(p.name for p in missing)
                continue
            original = session.display_path
            if not original.exists():
                if session.dirty:
                    deleted_dirty.append(original.name)
                else:
                    self._close_session(session)
                    closed.append(original.name)
                continue
            if session.disk_changed():  # type: ignore[attr-defined]
                if session.dirty:
                    conflicts.append(original.name)
                else:
                    self._reload_session_from_disk(session)
                    reloaded.append(original.name)
        parts: list[str] = []
        if closed:
            parts.append(f"closed (deleted on disk): {', '.join(closed)}")
        if reloaded:
            parts.append(f"reloaded from disk: {', '.join(reloaded)}")
        if deleted_dirty or conflicts or partial_raw:
            lines: list[str] = []
            if deleted_dirty:
                lines.append(
                    "Deleted on disk but kept open (unsaved changes — "
                    "use Save As to keep them):\n  "
                    + "\n  ".join(deleted_dirty)
                )
            if conflicts:
                lines.append(
                    "Changed on disk but NOT reloaded (your unsaved "
                    "changes were kept):\n  " + "\n  ".join(conflicts)
                )
            if partial_raw:
                lines.append(
                    "Raw inputs missing from disk (session kept open):"
                    "\n  " + "\n  ".join(partial_raw)
                )
            if parts:
                lines.append("Also: " + "; ".join(parts) + ".")
            QMessageBox.warning(self, "Refresh", "\n\n".join(lines))
        elif parts:
            self.statusBar().showMessage(
                "Refresh: " + "; ".join(parts) + ".", 8000
            )
        else:
            self.statusBar().showMessage(
                "Refresh: all files are up to date.", 5000
            )

    def _reload_session_from_disk(self, session: BaseSession) -> None:
        """Re-copy a clean session's original over its temp working copy
        and rebuild every cache/handle that pointed at the old bytes."""
        assert isinstance(session, NexusSession)
        # Parked re-activation handle (inactive sessions) targets the temp
        # file we're about to overwrite — release before the copy.
        stash = getattr(session, "_prewarm", None)
        if stash is not None:
            self._release_prewarm(stash)
            session._prewarm = None  # type: ignore[attr-defined]
        session._prewarm_overlays = None  # type: ignore[attr-defined]
        session._entries_cache = None  # type: ignore[attr-defined]
        is_active = session is self._active_session
        if is_active:
            self._pause_playback()
            self._entry_req_id += 1  # drop in-flight async loads of old bytes
            self.viewer.clear()
            self.viewer.clear_history()
            self.profile_viewer.clear()
            self.peaks_table_panel.clear()
        # The detached scope releases the silx model + viewer/prefetch
        # handles on the temp file so the overwrite never races a reader;
        # reattach re-inserts the files.
        with self._detached_silx_tree():
            session.reload_from_disk()
        # The old normalization record described the overwritten bytes.
        self._normalized_entries.pop(str(session.temp_path), None)
        if is_active:
            self._populate_entries()
            self._update_title()

    def _set_active_session(self, session: BaseSession | None) -> None:
        """Make ``session`` the active one and reload viewer-side state.

        No-op when ``session`` is already active. Blocked while a pipeline
        run is in flight — the worker captured the active temp_path at run
        time and ``_on_pipeline_finished`` reaches for ``self.session``,
        so swapping mid-flight would corrupt that path.
        """
        if session is self._active_session:
            return
        if self._pipe_thread is not None:
            return
        # Invalidate any in-flight async entry load: its result targets the
        # session we're leaving, so a late arrival must not install into the
        # new one (``_on_entry_loaded`` drops superseded request ids).
        self._entry_req_id += 1
        # Stop playback before swapping so the timer doesn't tick into
        # the new session's viewer state mid-construction.
        self._pause_playback()
        # Stash the outgoing NeXus session's live FrameSource as that
        # session's prewarm (entry name + warm handle): switching back
        # reinstalls it from memory, with the user's entry preserved.
        # Releasing it here (the old behaviour) made every re-activation
        # re-open the file and re-read frame 0 + peaks on the GUI thread,
        # which froze the window on big/remote masters. The handle is
        # read-only and writes only ever target the ACTIVE session, so a
        # parked handle on an inactive session's file is safe.
        outgoing = self._active_session
        src = self.viewer._frame_source
        if (
            outgoing is not None
            and outgoing.kind == "nexus"
            and src is not None
            and src.is_open
        ):
            outgoing._prewarm = (src.entry, src)  # type: ignore[attr-defined]
            # Overlays belong to the frame shown at stash time; the restore
            # lands on frame 0 and reloads peaks through the open handle.
            outgoing._prewarm_overlays = None  # type: ignore[attr-defined]
            self.viewer._frame_source = None  # clear() must not release it
        # Tear down viewer-side state belonging to the prior active session
        # before swapping — the new session's overlays must replace, not
        # accumulate on top of, whatever was previously shown.
        self.viewer.clear()
        self.viewer.clear_history()
        self.profile_viewer.clear()
        self.entry_combo.blockSignals(True)
        self.entry_combo.clear()
        self.entry_combo.blockSignals(False)
        self._active_session = session
        # ExpParameters are derived from the active NeXus metadata, so a
        # CIF cache built against the prior session may be misleading
        # for the new one. Forget it; the user can re-Parse when ready.
        if hasattr(self, "pipeline_panel"):
            self.pipeline_panel.clear_cif_cache()
        if session is not None:
            self._populate_entries()
        else:
            # No active session → wipe the per-entry options on the
            # pipeline panel so they don't reference a closed file.
            self.pipeline_panel.set_available_entries([])
        self._apply_session_mode(session)
        self._update_title()
        self._update_actions()
        # Status bar reflects the active session's entry / frame; the
        # entry change handler will fire shortly when the entry combo
        # repopulates, but pushing the file label now keeps the bar
        # consistent with the title bar even before that fires.
        self._update_status_entry()
        self._update_status_frame()
        # If the Figure Export window is open, drop its cached
        # mlgidbase handle and re-seed its basics pane from the new
        # session so its next render targets the right file.
        if self._figure_export_window is not None and self._figure_export_window.isVisible():
            self._figure_export_window.refresh_for_session()

    def _apply_session_mode(self, session: BaseSession | None) -> None:
        """Toggle dock visibility + viewer affordances for the session kind.

        Pipeline and Conversion docks are mode-exclusive: only one is ever
        visible at a time. Switching between a NeXus and a Raw session
        flips them in lockstep. With no active session, default to the
        Pipeline dock visible — that matches the cold-start UI.

        Raw mode also hides everything that doesn't apply to a raw
        detector frame: peak overlays / matched structures / profile
        viewer / parameter panel / Cartesian-Polar radios / Tools >
        Clear-peaks submenu. The user gets a clean canvas focused on
        the conversion workflow.
        """
        is_raw = session is not None and session.kind == "raw"
        self._pipeline_dock.setVisible(not is_raw)
        self._conversion_dock.setVisible(is_raw)
        # Re-tabify the right-dock chain per mode so the tab bar
        # order matches the active workflow:
        #   Raw mode:   Display | Conversion | Logs
        #   NeXus mode: Display | Pipeline | Logs
        # Peaks is on the bottom (tabified with Profiles) and isn't
        # part of the right-side chain. ``tabifyDockWidget``
        # repositions an already-tabified dock, so calling these
        # every mode-switch is cheap and idempotent.
        if is_raw:
            self.tabifyDockWidget(self._display_dock, self._conversion_dock)
            self.tabifyDockWidget(self._conversion_dock, self._logs_dock)
            self._conversion_dock.raise_()
        else:
            self.tabifyDockWidget(self._display_dock, self._pipeline_dock)
            self.tabifyDockWidget(self._pipeline_dock, self._logs_dock)
            # Keep Display in front by default for NeXus sessions; users
            # who prefer Pipeline up-front can click its tab.
            self._display_dock.raise_()
        # Hide NeXus-mode-only widgets in raw mode. Peaks is hidden
        # alongside Profiles since both depend on peak tables that
        # only exist after conversion.
        self._profile_dock.setVisible(not is_raw)
        if hasattr(self, "_peaks_dock"):
            self._peaks_dock.setVisible(not is_raw)
        if hasattr(self, "_scan_tracking_dock"):
            self._scan_tracking_dock.setVisible(not is_raw)
        if hasattr(self, "parameter_panel"):
            self.parameter_panel.setVisible(not is_raw)
        # Cartesian / Polar radios — meaningless before conversion.
        self.viewer.set_mode_radios_visible(not is_raw)
        # Tools > Clear peaks submenu has nothing to clear in raw mode.
        # Each kind is now a scope submenu, so disable the menu via its
        # menuAction (greys out the whole hover-target).
        for kind_menu in (
            getattr(self, "_clear_detected_menu", None),
            getattr(self, "_clear_fitted_menu", None),
            getattr(self, "_clear_matched_menu", None),
        ):
            if kind_menu is not None:
                kind_menu.menuAction().setEnabled(not is_raw)
        self._hide_stale_dock_tab_bars()

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

    def showEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().showEvent(event)
        # Dock tab bars only materialise with the first layout pass —
        # sweep for ghosts right after it (see the method docstring).
        QTimer.singleShot(0, self._hide_stale_dock_tab_bars)

    def _confirm_discard_changes(self, session: BaseSession | None = None) -> bool:
        target = session if session is not None else self._active_session
        if target is None or not target.dirty:
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved changes",
            f"{target.original_path.name} has unsaved changes. "
            f"Save before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Save:
            return self._save(confirm=False, session=target)
        if reply == QMessageBox.StandardButton.Discard:
            return True
        return False

    # -- silx tree helpers --

    def _detach_silx_tree(self) -> None:
        """Release silx's read handles + the viewer's FrameSource handle.

        Required before any code path opens an HDF5 file ``r+`` (pipeline
        runs, direct h5py edits) since open read handles would otherwise
        block the writer. After the lazy-loading milestone the viewer
        also holds a long-lived h5py handle through its FrameSource —
        that handle must be released here in addition to silx's.

        Both calls are wrapped in try/except: silx's ``clear()`` walks
        every Hdf5Item to close its owned file via ``obj.filename``,
        and on a stale ``obj`` that raises ``ValueError: Not a file or
        file object``. We swallow such errors so a partial-clear
        doesn't strand the detach half-done — the next reattach
        rebuilds the model from scratch anyway.
        """
        try:
            self.tree.findHdf5TreeModel().clear()
        except Exception:
            # silx's clear() can blow up when an Hdf5Item references a
            # closed h5py file. Swallow it; the reattach rebuilds the
            # whole tree from the live session list.
            logger.debug("suppressed exception in MainWindow._detach_silx_tree", exc_info=True)
            pass
        try:
            self.data_viewer.setData(None)
        except Exception:
            logger.debug("suppressed exception in MainWindow._detach_silx_tree", exc_info=True)
            pass
        # Drop any deferred tree node too — it belongs to the tree we're
        # tearing down, so rendering it later would reference a stale /
        # closed silx item.
        self._pending_data_node = None
        self.viewer.release_frame_source()
        # Tell the background prefetch worker to drop its own h5py
        # handle too. mlgidbase opens the same file r+ in the worker
        # we're about to spawn; an outstanding read handle from the
        # prefetcher would either contend (Windows), trip HDF5 file
        # locking ("Unable to synchronously open file"), or silently
        # serve pre-write data into the LRU mid-pipeline (Linux).
        #
        # Must be synchronous: a queued emit returns before the
        # worker has actually closed its handle, so the immediate
        # r+ open downstream of every caller (clear_peaks, the
        # pipeline run, save-as, peak-CSV export) would race the
        # release. BlockingQueuedConnection blocks the GUI thread
        # until the worker's release() slot has returned and the
        # handle is provably closed.
        if self._prefetch_worker is not None:
            QMetaObject.invokeMethod(
                self._prefetch_worker,
                "release",
                Qt.ConnectionType.BlockingQueuedConnection,
            )

    def _reattach_silx_tree(self) -> None:
        """Re-insert every session's files + reopen the viewer's
        FrameSource handle.

        NeXus sessions contribute one file (the temp working copy); raw
        sessions contribute every selected raw input so the user can keep
        browsing all of them while configuring conversion. The custom
        per-file icon set is repushed afterwards because clear() emptied
        the model. The viewer's FrameSource is reopened so subsequent
        frame reads can stream from disk again.

        Each ``insertFile`` is independent — one bad path doesn't
        strand the rest. silx returns a node reference on success and
        raises ``OSError`` on a missing/corrupt file; either way we
        continue with the next session.
        """
        model = self.tree.findHdf5TreeModel()
        for s in self._sessions:
            try:
                if isinstance(s, RawSession):
                    for raw_path in s.raw_paths:
                        model.insertFile(str(raw_path))
                else:
                    model.insertFile(str(s.temp_path))
            except Exception:
                # One bad session shouldn't strand the rest. The user
                # will see the missing entry; rebuild at next detach.
                logger.debug("suppressed exception in MainWindow._reattach_silx_tree", exc_info=True)
                pass
        self._refresh_tree_raw_paths()
        self.viewer.acquire_frame_source()

    @contextmanager
    def _detached_silx_tree(self):
        """Scoped silx detach/reattach for a *synchronous* critical
        section that needs the HDF5 file free of read handles.

        Detaches on entry and guarantees re-attachment on exit via
        ``finally`` — whether the block falls off the end, ``return``s
        early, or raises. This replaces the hand-paired
        ``_detach_silx_tree()`` / ``_reattach_silx_tree()`` calls whose
        reattach had to be duplicated on the happy path *and* every
        except/early-return branch (easy to forget one half).

        Deliberately NOT used at three sites, which keep explicit
        calls because the scoped semantics don't fit:

        * the pipeline run: detach spans a worker thread and the
          reattach happens later in ``_on_pipeline_finished``;
        * ``_safe_selected_h5_nodes``: a 2-line tear-down + rebuild
          recovery, not a "do work while detached" scope;
        * ``closeEvent``: a one-way detach on teardown — reattaching
          would be wrong.
        """
        self._detach_silx_tree()
        try:
            yield
        finally:
            self._reattach_silx_tree()

    # -- Entry / viewer wiring --

    def _populate_entries(self) -> None:
        if self.session is None:
            return
        if self.session.kind == "raw":
            self._populate_raw_entries()
            return
        # Prefer the entry list the open worker already computed (consumed
        # once); otherwise read the names shallowly now. Both paths use the
        # master's link names only and never resolve the external scans —
        # resolving them is what froze the GUI when opening a big master
        # (it opens every linked scan and holds the GIL while doing so).
        entries = getattr(self.session, "_entries_cache", None)
        if entries is not None:
            self.session._entries_cache = None  # type: ignore[attr-defined]
        else:
            try:
                entries = file_model.list_entry_names(self.session.temp_path)
            except Exception as exc:
                QMessageBox.warning(
                    self, "Read failed", f"Could not list entries: {exc}"
                )
                return
        self.entry_combo.blockSignals(True)
        self.entry_combo.clear()
        self.entry_combo.addItems(entries)
        # Land on the prewarmed entry when one is queued: on first open
        # that is entries[0] (CopyWorker's warm frame 0); on RE-activation
        # it is whatever entry the user was on when they switched away —
        # ``_set_active_session`` stashes the live FrameSource as the
        # outgoing session's prewarm, so switching back restores their
        # place from memory instead of re-reading the file.
        target = entries[0] if entries else ""
        prewarm = getattr(self.session, "_prewarm", None)
        if prewarm is not None and prewarm[0] in entries:
            target = prewarm[0]
            self.entry_combo.setCurrentText(target)
        self.entry_combo.blockSignals(False)
        # Push the same entry list into the pipeline panel's per-section
        # scope dropdowns so the user can pick a specific entry instead
        # of being limited to ACTIVE / ALL.
        self.pipeline_panel.set_available_entries(entries)
        if entries:
            self._load_entry_into_viewer(target)
        else:
            # Empty entry combo isn't always "empty file" — it's much more
            # often "file has entries but none are 2D q-images". Tell the
            # user exactly what's in the file so they don't think the GUI
            # silently dropped a working file.
            self._warn_no_q_entries()

    def _populate_raw_entries(self) -> None:
        """Walk every raw file in the active session and populate the
        entry combo with its 3D detector-image candidates.

        Combo items are labeled ``filename::dataset/path`` so the user
        can disambiguate when the batch contains multiple files. Pipeline
        panel's per-entry scope dropdown is cleared — pipeline ops aren't
        meaningful in raw mode. The Conversion panel also receives the
        same set of (file, entries) tuples for its selection tree.
        """
        assert isinstance(self.session, RawSession)
        # Maintain a mapping from combo label → RawEntry so the change
        # handler can resolve a click without re-walking the HDF5 file.
        self._raw_entries: dict[str, file_model.RawEntry] = {}
        labels: list[str] = []
        panel_inputs: list[tuple[Path, list[file_model.RawEntry]]] = []
        # Per-file entry lists are cached on the session: CopyWorker fills
        # the cache when the file is opened, and a GUI-thread fallback walk
        # fills it once for sessions that arrived without one. Activation
        # is on the GUI thread, so this must not re-walk a big beamtime
        # file's metadata every time the user switches back to the session.
        cache: dict[str, list] = getattr(self.session, "_raw_entries_cache", None) or {}
        # Standalone image files (TIFF/CBF/EDF) are read via fabio, not the
        # h5 detector walk; they bundle into ONE stack entry below. Keep the
        # HDF5 raw-walk path exactly as before for the rest.
        fabio_paths = [p for p in self.session.raw_paths if file_model.is_fabio_image(p)]
        h5_paths = [p for p in self.session.raw_paths if not file_model.is_fabio_image(p)]
        for raw_path in h5_paths:
            entries = cache.get(str(raw_path))
            if entries is None:
                try:
                    entries = file_model.list_raw_entries(raw_path)
                except Exception as exc:
                    self.conversion_panel.append_log(
                        f"Could not read {raw_path.name}: {exc}"
                    )
                    panel_inputs.append((raw_path, []))
                    continue
                cache[str(raw_path)] = entries
            panel_inputs.append((raw_path, entries))
            for re in entries:
                self._raw_entries[re.label] = re
                labels.append(re.label)
        self.session._raw_entries_cache = cache  # type: ignore[attr-defined]
        # One entry PER image file (each a standalone 1-frame view), read via
        # LazyFabioStack — multiple images open separately, not as a stack.
        if fabio_paths:
            try:
                fabio_entries = file_model.list_fabio_entries(fabio_paths)
            except Exception as exc:
                self.conversion_panel.append_log(
                    f"Could not read image files: {exc}"
                )
            else:
                for entry in fabio_entries:
                    self._raw_entries[entry.label] = entry
                    labels.append(entry.label)
                    panel_inputs.append((entry.file_path, [entry]))
        # Push the same data into the Conversion panel for its selection
        # tree. Done before populating the combo so the panel paint
        # happens once on activation.
        self.conversion_panel.set_raw_inputs(panel_inputs)
        self.entry_combo.blockSignals(True)
        self.entry_combo.clear()
        self.entry_combo.addItems(labels)
        self.entry_combo.blockSignals(False)
        self.pipeline_panel.set_available_entries([])
        if labels:
            # Auto-load the first candidate so the user sees something
            # immediately. The change handler handles further picks.
            self._load_raw_entry_into_viewer(labels[0])
        else:
            QMessageBox.information(
                self,
                "No raw datasets found",
                "None of the selected raw files contain a 3D detector "
                "dataset (shape (N, H, W) with H, W ≥ 32). Check the "
                "files in the tree on the left to see their structure.",
            )

    def _on_raw_flips_changed(self, flip_lr: bool, flip_ud: bool) -> None:
        """Apply the conversion panel's fliplr/flipud to the raw preview."""
        self.viewer.set_raw_flips(flip_lr, flip_ud)

    def _load_raw_entry_into_viewer(self, label: str) -> None:
        """Load the picked raw entry into the viewer in pixel coords.

        ``label`` is the combo's display string (file::dataset/path).
        Resolved through ``self._raw_entries`` to a ``RawEntry`` and
        dispatched to the EntryLoadWorker, which opens the dataset and
        warms frame 0 off the GUI thread (``_on_raw_entry_loaded``
        installs the result). The old synchronous path materialized the
        WHOLE 3D dataset on this thread — a multi-GB read that froze
        the window for its duration on big beamtime files.
        """
        raw_entry = getattr(self, "_raw_entries", {}).get(label)
        if raw_entry is None:
            return
        self._entry_req_id += 1
        self._ensure_entry_load_worker()
        self._sb_open_label.setText(f"Loading {label}…")
        self._sb_open_label.show()
        self._sb_open_bar.setRange(0, 0)  # indeterminate: a single read, no stages
        self._sb_open_bar.show()
        self._rawLoadRequest.emit(raw_entry, self._entry_req_id)

    def _on_raw_entry_loaded(self, request_id: int, label: str, stack) -> None:
        """Install the raw stack the worker just opened, unless a newer
        switch superseded it (then its handle is released). Mirrors
        ``_on_entry_loaded``."""
        if request_id != self._entry_req_id:
            if stack is not None:
                try:
                    stack.release()
                except Exception:
                    logger.debug("suppressed stale raw-stack release", exc_info=True)
            return
        self._dismiss_open_progress()
        if stack is None:
            QMessageBox.warning(self, "Load failed", f"Could not load {label}")
            return
        # The active session may have changed between dispatch and arrival
        # (file closed / switched to a NeXus session) — never install an
        # orphaned stack.
        if self.session is None or self.session.kind != "raw":
            try:
                stack.release()
            except Exception:
                logger.debug("suppressed orphan raw-stack release", exc_info=True)
            return
        self.viewer.show_raw_stack(stack)
        self._refresh_frame_slider()
        self._update_status_frame()

    def _warn_no_q_entries(self) -> None:
        """Diagnose why the entry combo ended up empty.

        The viewer + pipeline only handle ``img_gid_q`` entries. Files
        with reduced-data entries (``horiz_cut_gid``, ``rad_cut_gid``,
        polar-only ``img_gid_pol``, etc.) load fine but produce no
        viewable entry, which previously looked like the GUI broke.
        """
        try:
            signals = file_model.list_entry_signals(self.session.temp_path)
        except Exception:
            logger.debug("suppressed exception in MainWindow._warn_no_q_entries", exc_info=True)
            signals = {}
        if not signals:
            # Truly empty — file genuinely has no entry_* groups.
            QMessageBox.information(
                self,
                "Nothing to show",
                f"{self.session.original_path.name} has no entry_* groups "
                "to display.",
            )
            return
        rows = "\n".join(
            f"  • {name} — signal = {signal!r}"
            for name, signal in signals.items()
        )
        QMessageBox.information(
            self,
            "No 2D q-image entries",
            f"{self.session.original_path.name} loaded successfully but "
            "contains no entries with the 2D q-image data the viewer + "
            "pipeline operate on (signal = 'img_gid_q').\n\n"
            f"Entries found:\n{rows}\n\n"
            "These are likely reduced data (1D cuts, polar grids, or "
            "post-processed outputs). To use the GUI's detection / "
            "fitting / matching tools, open a NeXus file produced by "
            "the pygid → mlgidDETECT pipeline that still carries the "
            "raw q-image stack.",
        )

    def _on_entry_changed(self, entry: str) -> None:
        if not entry or self.session is None:
            return
        if self.session.kind == "raw":
            self._load_raw_entry_into_viewer(entry)
            self._update_status_entry()
            self._update_status_frame()
        else:
            # Interactive NeXus switch (combo or file-browser click): read
            # the entry's frame OFF the GUI thread so a big detector frame
            # over a slow share never freezes the window. The status bar is
            # refreshed in ``_on_entry_loaded`` once the frame is in.
            self._load_entry_async(entry)
            self._update_status_entry()

    def _on_frame_slider_changed(self, value: int) -> None:
        """User dragged the Display-dock slider — push to the viewer.

        ``viewer.set_frame`` is a no-op when ``value`` already matches
        the current frame, so the bidirectional sync (slider→viewer→
        slider via _on_viewer_frame_changed) doesn't recurse.
        """
        self.viewer.set_frame(value)

    def _on_viewer_frame_changed(self, frame: int) -> None:
        """Viewer changed frame (timeline scrub, programmatic seek, etc.)
        — keep the Display-dock slider + label in sync without
        re-emitting valueChanged back into the viewer. Also pushes
        the new play-head into the background prefetch worker so
        its sliding window slides with the user.
        """
        # Lazily pull this frame's peaks into the viewer (entry switch
        # loads only the landed frame; navigating loads the rest here on
        # demand). Runs before the peaks-table / matched-panel frame
        # slots — they're connected after this one — so they see the
        # freshly-installed peaks.
        entry = self.entry_combo.currentText()
        if entry:
            self._load_frame_peaks(entry, int(frame))
        self.frame_label.setText(self._frame_label_text(frame))
        if self.frame_slider.value() != frame:
            self.frame_slider.blockSignals(True)
            try:
                self.frame_slider.setValue(int(frame))
            finally:
                self.frame_slider.blockSignals(False)
        self._refresh_frame_nav_enabled()
        self._update_status_frame()
        # Tell the prefetch worker where the play-head is now. The
        # ``active`` flag tracks the play-button state so a manual
        # scrub doesn't accidentally wake the worker. Pass the live
        # step so the worker walks the same stride as the player.
        if self._prefetch_worker is not None:
            active = self.play_button.isChecked()
            step = self._play_step if active else 1
            self._prefetchUpdate.emit(int(frame), active, step)
        # Reset the Detected min-score slider to the new frame's
        # min score. Matched is reseeded via _refresh_matched_panel,
        # which fires from the viewer's matchedStructuresChanged
        # signal on the same frame change.
        self._seed_detected_score_slider()

    def _on_play_toggled(self, checked: bool) -> None:
        """Start / stop the frame-playback timer.

        Press → if the current frame is already at the end, restart
        from frame 0; otherwise advance from the current frame. Press
        again to pause. The icon flips between Play and Pause.

        Refuses to start during a pipeline run (the viewer is gated
        ``busy`` during those, so frame edits would block anyway).

        Reads the current playback settings from QSettings on every
        press so a setting change applies on the next play without
        any restart machinery.
        """
        if checked:
            if self._pipe_thread is not None or self.viewer.n_frames <= 1:
                # Bail: the play button toggle was either an erroneous
                # programmatic click or fired while a pipeline run owns
                # the viewer. Unchecking re-fires this slot with
                # checked=False, which is a no-op.
                self.play_button.setChecked(False)
                return
            # Wrap around when at the end so the second click of Play
            # always plays the full sequence.
            if self.viewer.current_frame >= self.viewer.n_frames - 1:
                self.viewer.set_frame(0)
            interval, step = self._compute_play_schedule()
            self._play_timer.setInterval(interval)
            self._play_step = step
            self.play_button.setIcon(self._icon_pause)
            self.play_button.setToolTip("Pause playback")
            self._play_timer.start()
            # Activate the background prefetch worker — it'll start
            # warming frames just ahead of the play-head, stepping
            # the same way the player does so prefetched frames
            # actually match the ones we'll display.
            if self._prefetch_worker is not None:
                self._prefetchUpdate.emit(
                    self.viewer.current_frame, True, step,
                )
        else:
            self._play_timer.stop()
            self._play_step = 1
            self.play_button.setIcon(self._icon_play)
            self.play_button.setToolTip(
                "Play frames from the current position to the end.\n"
                "Stops at the last frame; click again to pause."
            )
            if self._prefetch_worker is not None:
                self._prefetchUpdate.emit(
                    self.viewer.current_frame, False, 1,
                )

    def _compute_play_schedule(self) -> tuple[int, int]:
        """Resolve ``(timer_interval_ms, frame_step)`` from QSettings.

        The user expresses a desired *per-frame* duration — either
        directly (Time-per-frame mode) or implicitly (Total-time mode
        ÷ n_frames). If that desired duration is at or above
        ``PLAYBACK_TICK_FLOOR_MS`` (≈ 20 fps), playback uses it
        directly with ``step=1``. If it's *below* that floor, the
        timer is held at the floor and ``step`` is bumped so the
        play-head jumps multiple frames per tick — i.e. we honour the
        target total time by skipping frames instead of asking Qt to
        fire faster than the display + disk can keep up. 20 fps is
        more than enough to perceive the time-series motion; the
        skipped frames are still reachable via the slider.

        Out-of-bounds / unparseable stored values fall back to the
        defaults so a corrupted QSettings entry can't soft-lock the
        Play button.
        """
        settings = QSettings()
        mode = settings.value(self._PLAYBACK_MODE_KEY, PLAYBACK_MODE_FRAME)
        if mode == PLAYBACK_MODE_TOTAL:
            try:
                total_s = float(settings.value(
                    self._PLAYBACK_TOTAL_S_KEY, DEFAULT_PLAYBACK_TOTAL_S
                ))
            except (TypeError, ValueError):
                total_s = DEFAULT_PLAYBACK_TOTAL_S
            total_s = max(PLAYBACK_TOTAL_S_MIN,
                          min(PLAYBACK_TOTAL_S_MAX, total_s))
            steps = max(self.viewer.n_frames - 1, 1)
            desired_ms = total_s * 1000.0 / steps
        else:
            try:
                frame_ms = int(settings.value(
                    self._PLAYBACK_FRAME_MS_KEY, DEFAULT_PLAYBACK_FRAME_MS
                ))
            except (TypeError, ValueError):
                frame_ms = DEFAULT_PLAYBACK_FRAME_MS
            desired_ms = float(max(PLAYBACK_FRAME_MS_MIN,
                                   min(PLAYBACK_FRAME_MS_MAX, frame_ms)))

        if desired_ms < PLAYBACK_TICK_FLOOR_MS:
            # Below the 20 fps ceiling — bunch frames together per tick.
            # step = ceil(floor / desired) so the per-frame time stays
            # ≤ desired (= we never play slower than asked). Interval
            # then = desired * step, which lands at or just above the
            # floor.
            step = max(1, int(math.ceil(PLAYBACK_TICK_FLOOR_MS / desired_ms)))
            interval_ms = max(PLAYBACK_TICK_FLOOR_MS,
                              int(round(desired_ms * step)))
        else:
            step = 1
            interval_ms = int(round(desired_ms))
        return interval_ms, step

    def _on_play_tick(self) -> None:
        """One step of frame playback.

        Stops at end-of-stack. Auto-pauses if the viewer becomes busy
        (pipeline run kicked off mid-playback) or the user closed the
        file. The slider's ``valueChanged`` connection routes the
        frame change through ``viewer.set_frame`` so the existing
        sync paths fire exactly once per step.
        """
        if (
            self.viewer.n_frames <= 1
            or self._pipe_thread is not None
            or self.session is None
        ):
            self.play_button.setChecked(False)
            return
        step = max(1, self._play_step)
        next_frame = self.viewer.current_frame + step
        if next_frame >= self.viewer.n_frames:
            # Snap to the last frame so the user always sees the end
            # of the sequence even when ``step`` would overshoot — then
            # pause. Click Play again to wrap to frame 0 (the toggle
            # handler handles the wrap).
            last = self.viewer.n_frames - 1
            if self.viewer.current_frame < last:
                self.viewer.set_frame(last)
            self.play_button.setChecked(False)
            return
        self.viewer.set_frame(next_frame)

    def _pause_playback(self) -> None:
        """Stop the playback timer if it's running.

        Called from session-swap / file-close / pipeline-start paths
        so playback doesn't tick into a torn-down viewer or contend
        with a pipeline write. Safe to call when playback is already
        stopped.
        """
        if self.play_button.isChecked():
            self.play_button.setChecked(False)

    # -- Background prefetch worker ---------------------------------------

    def _ensure_prefetch_worker(self) -> None:
        """Spawn the prefetch worker + thread on first use. Idempotent.

        Lazy spawn keeps startup fast for users who only ever view
        single-frame files (no playback, no prefetch worth running).
        Once spawned, the worker survives across entry switches —
        each new entry triggers ``configure()`` rather than a
        rebuild.
        """
        if self._prefetch_worker is not None:
            return
        self._prefetch_thread = QThread(self)
        self._prefetch_worker = PrefetchWorker()
        self._prefetch_worker.moveToThread(self._prefetch_thread)
        # Cross-thread wiring. configure / update_state / release run
        # on the worker's thread via queued connections; prefetched
        # signal delivers back to the GUI thread.
        self._prefetchConfigure.connect(
            self._prefetch_worker.configure, Qt.ConnectionType.QueuedConnection,
        )
        self._prefetchUpdate.connect(
            self._prefetch_worker.update_state, Qt.ConnectionType.QueuedConnection,
        )
        self._prefetchRelease.connect(
            self._prefetch_worker.release, Qt.ConnectionType.QueuedConnection,
        )
        self._prefetch_worker.prefetched.connect(
            self._on_prefetched, Qt.ConnectionType.QueuedConnection,
        )
        self._prefetch_thread.start()

    def _configure_prefetch_for_active_entry(self) -> None:
        """Tell the worker about the active entry's shape + LRU size.

        Called after every successful entry load (in
        ``_load_entry_into_viewer``) and after the silx-reattach
        path completes a pipeline run. No-op for single-frame
        stacks (nothing to prefetch) and for raw sessions
        (FrameSource isn't used).
        """
        if (
            self.session is None
            or self.session.kind != "nexus"
            or self.viewer._frame_source is None
            or self.viewer.n_frames <= 1
        ):
            # Release the worker if we have one — no work on idle.
            if self._prefetch_worker is not None:
                self._prefetchRelease.emit()
            return
        self._ensure_prefetch_worker()
        fs = self.viewer._frame_source
        # Sliding-window size = LRU - 1 so the prefetcher can never
        # evict frames the play-head still needs to reach.
        window = max(1, fs.cart_lru_size - 1)
        entry = self.entry_combo.currentText()
        if not entry:
            return
        self._prefetchConfigure.emit(
            str(self.session.temp_path), entry, fs.n_frames, window,
        )
        # Start in paused state — the worker only ticks during
        # active playback. The play-button toggle (and any frame
        # change while playing) will flip ``active=True`` via
        # _prefetchUpdate.
        self._prefetchUpdate.emit(self.viewer.current_frame, False, 1)

    @Slot(int, object, object, object, object)
    def _on_prefetched(
        self,
        idx: int,
        cart: object,
        polar: object,
        radius: object,
        angle: object,
    ) -> None:
        """Deposit a prefetched frame into the active FrameSource's LRU.

        Runs on the GUI thread (queued from the worker). The
        FrameSource's LRUs are touched only here and from the
        synchronous ``get_cartesian`` / ``get_polar`` paths, both
        of which live on the GUI thread — so no locking is needed.

        Drops the result silently if the FrameSource has been
        released (post-pipeline detach), since a stale signal in
        flight should not warm a closed cache.
        """
        fs = self.viewer._frame_source
        if fs is None or not fs.is_open:
            return
        try:
            fs.warm_cartesian(int(idx), cart)        # type: ignore[arg-type]
            fs.warm_polar(int(idx), polar, radius, angle)  # type: ignore[arg-type]
        except Exception:
            # Defensive — a stale signal during teardown shouldn't
            # propagate.
            logger.debug("suppressed exception in MainWindow._on_prefetched", exc_info=True)
            pass

    def _refresh_frame_slider(self) -> None:
        """Match the slider's range + value to the active stack's
        frame count. Called after every show_stack — covers entry
        switches, file opens, and pipeline-finished reloads.
        Single-frame stacks hide the whole nav cluster.
        """
        n = self.viewer.n_frames
        cur = self.viewer.current_frame
        self.frame_slider.blockSignals(True)
        try:
            if n <= 1:
                self.frame_slider.setMinimum(0)
                self.frame_slider.setMaximum(0)
                self.frame_slider.setValue(0)
            else:
                self.frame_slider.setMinimum(0)
                self.frame_slider.setMaximum(n - 1)
                self.frame_slider.setValue(int(cur))
        finally:
            self.frame_slider.blockSignals(False)
        self.frame_label.setText(self._frame_label_text(cur))
        self._set_frame_slider_visible(n > 1)
        self._refresh_frame_nav_enabled()

    def _set_frame_slider_visible(self, visible: bool) -> None:
        """Show or hide the toolbar's frame-navigation cluster.

        With the controls living in the image-viewer toolbar (no
        form / no row container), each widget is toggled directly.
        """
        for w in (
            self.prev_frame_button,
            self.play_button,
            self.next_frame_button,
            self.frame_slider,
            self.frame_label,
        ):
            w.setVisible(visible)

    def _refresh_frame_nav_enabled(self) -> None:
        """Disable prev/next at the boundaries so the user can see
        they've hit the start / end of the stack."""
        n = self.viewer.n_frames
        cur = self.viewer.current_frame
        self.prev_frame_button.setEnabled(n > 1 and cur > 0)
        self.next_frame_button.setEnabled(n > 1 and cur < n - 1)

    # Minimum gap between successive prev/next frame steps. The OS
    # keyboard auto-repeat fires at ~30 events/sec (~33 ms apart);
    # without throttling, set_frame calls pile up faster than the
    # viewer can render and a held arrow key leaves a backlog that
    # keeps advancing after the user releases. 80 ms matches the
    # existing toolbar prev/next button autoRepeatInterval — fast
    # enough to feel responsive, slow enough to give each frame
    # room to render.
    _FRAME_STEP_THROTTLE_S = 0.08

    def _frame_step_throttle_ok(self) -> bool:
        """Time-throttle: drop step requests that arrive within
        ``_FRAME_STEP_THROTTLE_S`` of the previous one.

        Single clicks are always >80 ms apart so they're never
        affected; only OS keyboard auto-repeat (~33 ms cadence) and
        Qt toolbar auto-repeat get suppressed below the throttle.
        """
        now = time.monotonic()
        last = getattr(self, "_last_frame_step_t", 0.0)
        if (now - last) < self._FRAME_STEP_THROTTLE_S:
            return False
        self._last_frame_step_t = now
        return True

    def _step_frame(self, direction: int) -> None:
        """Shared prev/next step path with both a time-throttle and
        a queue-drain gate.

        Two complementary mechanisms:

        1. ``_frame_step_throttle_ok`` enforces a minimum gap
           between accepted steps. Protects against fast renders +
           OS autorepeat (drops the bulk of repeat events).

        2. The ``_frame_step_in_flight`` flag + ``processEvents()``
           drain protects against **slow** renders, where the
           synchronous ``set_frame`` call blocks the event loop
           long enough for the OS to enqueue multiple keypress
           events. After the render completes we explicitly drain
           the queue while the flag is still set, so the queued
           auto-repeats hit the flag, see "busy", and drop. Without
           this drain, holding a key on a slow stack continues
           advancing for ~1 s after the user releases.
        """
        if getattr(self, "_frame_step_in_flight", False):
            return
        if not self._frame_step_throttle_ok():
            return
        self._frame_step_in_flight = True
        try:
            cur = self.viewer.current_frame
            target = cur + direction
            if 0 <= target < self.viewer.n_frames:
                self.viewer.set_frame(target)
            # Drain OS auto-repeats that piled up during the
            # synchronous render. They'll recurse into this method,
            # see the in-flight flag set, and drop. This is the
            # one place in the GUI where ``processEvents`` from
            # inside a slot is required for correctness — see the
            # docstring above.
            QApplication.processEvents()
        finally:
            self._frame_step_in_flight = False

    def _on_prev_frame_clicked(self) -> None:
        self._step_frame(-1)

    def _on_next_frame_clicked(self) -> None:
        self._step_frame(+1)

    def _on_first_frame_shortcut(self) -> None:
        """Jump to frame 0. Bound to Home. Not throttled — these
        are single-target jumps, not repeated steps."""
        if self.viewer.n_frames > 1 and self.viewer.current_frame != 0:
            self.viewer.set_frame(0)

    def _on_last_frame_shortcut(self) -> None:
        """Jump to the last frame. Bound to End. Not throttled."""
        n = self.viewer.n_frames
        last = n - 1
        if n > 1 and self.viewer.current_frame != last:
            self.viewer.set_frame(last)

    def _install_frame_shortcuts(self) -> None:
        """Register window-level keyboard shortcuts for frame navigation.

        Each binding is a hidden ``QAction`` on the main window with
        ``WindowShortcut`` context. Qt's normal key-event chain means
        text-input widgets (QLineEdit, QSpinBox, QDoubleSpinBox)
        consume Left / Right / Home / End for caret navigation before
        the shortcut fires — so typing in a dock field still works.
        J/K give a Vim-style fallback that doesn't collide with text
        input either (most fields are numeric).
        """
        bindings = [
            ("Prev frame", ["Left", "J"], self._on_prev_frame_clicked),
            ("Next frame", ["Right", "K"], self._on_next_frame_clicked),
            ("First frame", ["Home"], self._on_first_frame_shortcut),
            ("Last frame", ["End"], self._on_last_frame_shortcut),
        ]
        self._frame_shortcut_actions = []
        for name, keys, slot in bindings:
            action = QAction(name, self)
            action.setShortcuts([QKeySequence(k) for k in keys])
            action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            action.triggered.connect(slot)
            self.addAction(action)
            self._frame_shortcut_actions.append(action)

    def _frame_label_text(self, idx: int) -> str:
        n = self.viewer.n_frames
        if n <= 1:
            return "—"
        return f"{int(idx)} / {n - 1}"

    def _safe_selected_h5_nodes(self) -> list:
        """Return ``selectedH5Nodes`` results, swallowing silx model errors.

        Under certain races (mid-pipeline detach/reattach, freshly
        inserted file with not-yet-resolved h5py state), silx's tree
        model can raise on attribute lookup deep inside the proxy
        chain. Qt then re-fires the call, producing a stack-busting
        recursion that brings down the click handler. We catch
        anything from that path here so a single bad click can't
        wedge the GUI.
        """
        try:
            return list(self.tree.selectedH5Nodes())
        except (RecursionError, RuntimeError, KeyError, OSError) as exc:
            self.pipeline_panel.append_log(
                f"WARN — silx tree query failed ({type(exc).__name__}); "
                "rebuilding the file browser"
            )
            # Drastic but reliable: tear the tree down and rebuild it
            # from the live session list. Any orphan / half-loaded
            # silx items get dropped in the process.
            self._detach_silx_tree()
            self._reattach_silx_tree()
            return []

    def _on_tree_selection_changed(self, *_: object) -> None:
        nodes = self._safe_selected_h5_nodes()
        if not nodes:
            return
        node = nodes[0]
        # Switch active session + entry FIRST, before touching the Data
        # viewer. silx's ``DataViewerFrame.setData`` eagerly resolves the
        # clicked entry's external-linked NXdata signal to pick a view —
        # slow enough on a big multi-scan master to freeze the GUI, and
        # because it used to run first it aborted the entry switch
        # entirely (the file-browser click appeared to "do nothing").
        # Doing the cheap entry switch first makes a tree click behave
        # exactly like the Entry combo.
        #
        # Multiple files may be loaded — clicking into a different file's
        # subtree promotes that file to the active session so the entry
        # combo, image viewer, and per-file actions follow the user's
        # focus without an extra click.
        self._activate_session_for_node(node)
        # Click into entry_X anywhere → switch the image tab to that
        # entry. The entry-combo signal already triggers the viewer
        # reload, so we just push the new value here.
        self._activate_entry_for_node(node)
        # Feed the Data viewer only when its tab is actually visible;
        # otherwise stash the node and render it lazily on tab switch, so
        # a single click on the Image tab never pays the external-link
        # resolve for a dataset the user can't even see.
        self._set_or_defer_data_node(node)

    def _on_tree_activated(self, *_: object) -> None:
        nodes = self._safe_selected_h5_nodes()
        if not nodes:
            return
        node = nodes[0]
        self._activate_session_for_node(node)
        self._activate_entry_for_node(node)
        # Double-click means "inspect this in the Data tab": switch to it
        # and render now — the user explicitly asked to see the data, so
        # the resolve cost is expected here (and only here).
        self.tabs.setCurrentWidget(self.data_viewer)
        self._set_data_node(node)

    def _set_or_defer_data_node(self, node) -> None:
        """Render ``node`` in the Data viewer if that tab is showing, else
        remember it and render on the next switch to the Data tab."""
        if self.tabs.currentWidget() is self.data_viewer:
            self._set_data_node(node)
        else:
            self._pending_data_node = node

    def _set_data_node(self, node) -> None:
        """Push ``node`` into the silx Data viewer, guarded so a slow / bad
        external-link resolve can't propagate out of a tree click."""
        self._pending_data_node = None
        try:
            self.data_viewer.setData(node)
        except Exception:
            logger.debug(
                "suppressed exception in MainWindow._set_data_node", exc_info=True
            )

    def _on_main_tab_changed(self, _index: int) -> None:
        """When the user switches to the Data tab, render any node that was
        clicked while the tab was hidden (deferred to keep Image-tab clicks
        from paying the external-link resolve)."""
        if (
            self.tabs.currentWidget() is self.data_viewer
            and self._pending_data_node is not None
        ):
            self._set_data_node(self._pending_data_node)

    def _activate_entry_for_node(self, node) -> None:
        """If the clicked node is inside an ``entry_*`` group, switch the
        entry combo (and therefore the image viewer) to that entry.

        No-ops for clicks on the file root or on nodes outside any entry
        group (e.g. top-level metadata). Also no-ops if the entry isn't
        in the combo — that would mean it's a non-q entry filtered out
        by ``list_entries``, where the viewer can't render anything
        useful anyway.

        Raw sessions take their own resolver: their combo holds detector-
        dataset candidates (``file::dataset/path``), not ``entry_*``
        groups. ``_activate_session_for_node`` ran first, so the active
        session's kind matches the clicked file.
        """
        if self.session is not None and self.session.kind == "raw":
            self._activate_raw_entry_for_node(node)
            return
        entry = self._node_entry_name(node)
        if entry is None:
            return
        if self.entry_combo.findText(entry) < 0:
            return
        if self.entry_combo.currentText() == entry:
            return
        # Triggers _on_entry_changed → _load_entry_into_viewer.
        self.entry_combo.setCurrentText(entry)

    def _activate_raw_entry_for_node(self, node) -> None:
        """Map a clicked tree node to a raw detector dataset and select
        it in the entry combo — raw files browse from the file browser
        exactly like NeXus ``entry_*`` groups (the combo change triggers
        the async ``EntryLoadWorker.load_raw``).

        A node matches a candidate when it IS the candidate's dataset or
        an ancestor group of one (clicking a scan group like ``1.1``
        selects the first candidate inside it, mirroring how clicking
        anywhere inside an ``entry_*`` group selects that entry). Clicks
        on the file root or on non-candidate nodes (axes, metadata)
        no-op.
        """
        raw_entries = getattr(self, "_raw_entries", {})
        if not raw_entries:
            return
        fname = self._node_filename(node)
        rel = self._node_h5_path(node)
        if fname is None or not rel:
            return
        try:
            target = fname.resolve()
        except OSError:
            target = fname
        for label, raw_entry in raw_entries.items():
            if not (
                raw_entry.dataset_path == rel
                or raw_entry.dataset_path.startswith(rel + "/")
            ):
                continue
            try:
                entry_file = raw_entry.file_path.resolve()
            except OSError:
                entry_file = raw_entry.file_path
            if entry_file != target:
                continue
            if self.entry_combo.currentText() != label:
                # Triggers _on_entry_changed → _load_raw_entry_into_viewer.
                self.entry_combo.setCurrentText(label)
            return

    @staticmethod
    def _node_h5_path(node) -> str | None:
        """The node's path inside its HDF5 file ('' for the file root).

        Prefers silx's ``local_name`` (the master-side path — reading it
        never resolves an external link); falls back to the h5py
        object's name for silx versions without it.
        """
        for getter in (
            lambda n: getattr(n, "local_name", None),
            lambda n: n.h5py_object.name,
        ):
            try:
                p = getter(node)
            except Exception:
                logger.debug("suppressed exception in MainWindow._node_h5_path", exc_info=True)
                continue
            if p:
                return str(p).strip("/")
        return None

    @staticmethod
    def _node_entry_name(node) -> str | None:
        """Extract the ``entry_*`` group name from a node's HDF5 path.

        silx exposes the absolute path as ``local_name`` (e.g.
        ``/entry_0000/data/img_gid_q``); we take the first component if
        it begins with ``entry_``. Returns None for nodes outside any
        entry group.
        """
        for getter in (
            lambda n: getattr(n, "local_name", None),
            lambda n: n.h5py_object.name,
        ):
            try:
                p = getter(node)
            except Exception:
                logger.debug("suppressed exception in MainWindow._node_entry_name", exc_info=True)
                continue
            if p:
                parts = str(p).lstrip("/").split("/")
                if not parts:
                    return None
                if file_model.is_entry_group_name(parts[0]):
                    return parts[0]
                # Entry named off-convention (mlgidFIT names it <sample>):
                # confirm parts[0] is an NXentry group via the node's file.
                try:
                    if parts[0] in file_model.entry_group_names(node.h5py_object.file):
                        return parts[0]
                except Exception:
                    logger.debug("suppressed exception in MainWindow._node_entry_name (nxentry)", exc_info=True)
                return None
        return None

    def _session_for_node(self, node) -> BaseSession | None:
        """Return the session whose file ``node`` was loaded from, or None.

        silx normalizes paths through the OS, so a literal ``Path`` equality
        with ``session.temp_path`` can fail when one side has a symlink,
        dotfile component, or trailing slash that the other doesn't.
        ``Path.resolve()`` collapses both sides to a canonical absolute form
        before the comparison.
        """
        fname = self._node_filename(node)
        if fname is None:
            return None
        try:
            target = fname.resolve()
        except OSError:
            target = fname
        for s in self._sessions:
            # Raw sessions own multiple files in the tree; any of them
            # maps back to the same RawSession. NeXus sessions own
            # exactly one file (the temp working copy).
            if isinstance(s, RawSession):
                candidate_paths = list(s.raw_paths)
            else:
                candidate_paths = [s.temp_path]
            for candidate in candidate_paths:
                try:
                    candidate_resolved = candidate.resolve()
                except OSError:
                    candidate_resolved = candidate
                if candidate_resolved == target:
                    return s
        return None

    def _activate_session_for_node(self, node) -> None:
        """If ``node`` lives in a non-active session's file, swap active.

        Previously a path mismatch silently left the wrong session active
        and the pipeline ran on the most recently opened file regardless
        of which tree the user clicked. ``_session_for_node`` does the
        resolve-and-match so the swap follows the clicked tree.
        """
        s = self._session_for_node(node)
        if s is not None and s is not self._active_session:
            self._set_active_session(s)

    def _remove_selected_file_from_browser(self) -> None:
        """Close the file the current tree selection belongs to.

        Wired to the file browser's Delete key (``deleteFileRequested``).
        Resolves the selected node back to its session and routes through
        the same discard-confirm + ``_close_session`` path as File →
        Close (Ctrl+W), so unsaved-change handling and tree teardown are
        identical no matter which entry point removes the file. No-ops if
        the selection can't be mapped to a live session (e.g. a stale or
        orphan node), mirroring how a no-active-session Close no-ops.
        """
        nodes = self._safe_selected_h5_nodes()
        if not nodes:
            return
        session = self._session_for_node(nodes[0])
        if session is None:
            return
        if not self._confirm_discard_changes(session):
            return
        self._close_session(session)

    @staticmethod
    def _node_filename(node) -> Path | None:
        """Resolve the filesystem path of the file ``node`` was loaded from.

        silx exposes this differently across versions — fall through the
        known accessors and give up silently if nothing answers.
        """
        for getter in (
            lambda n: getattr(n, "local_filename", None),
            lambda n: n.h5py_object.file.filename,
        ):
            try:
                p = getter(node)
            except Exception:
                logger.debug("suppressed exception in MainWindow._node_filename", exc_info=True)
                continue
            if p:
                return Path(p)
        return None

    # -- Profile viewer adapters --

    def _forward_selection_to_profile(self, sel: SelectedPeak | None) -> None:
        # Profiles render for any kind of selection; the profile viewer
        # internally makes regions non-movable for non-manual peaks since
        # those are edited through the 2D ROI.
        self.profile_viewer.set_selected_peak(sel)

    def _forward_geom_to_profile(self, sel: SelectedPeak | None) -> None:
        if sel is None:
            return
        self.profile_viewer.sync_regions_from_peak(sel)

    def _on_detected_border_commit(self, sel: SelectedPeak) -> None:
        """Persist a detected-peak border drag from the profile viewer
        to ``detected_peaks`` on disk.

        Funnels through the same ``_on_peak_row_write_requested`` slot
        the image-side ROI drag uses — it owns the silx detach /
        update_peak_row / matched cascade dance. The profile-side
        drag merely fires this commit signal at drag-end; everything
        else (live overlay sync, in-memory PeakTable mutation,
        ``q_xy``/``q_z`` recompute) has already happened during the
        live drag via ``update_detected_geometry_external``.
        """
        polar = {
            "radius": float(sel.radius),
            "angle": float(sel.angle),
            "radius_width": float(sel.radius_width),
            "angle_width": float(sel.angle_width),
        }
        self._on_peak_row_write_requested(
            int(sel.frame), "detected", int(sel.peak_id), polar,
        )

    def _on_selection_for_preview(self, sel: SelectedPeak | None) -> None:
        """Drop the fitted-preview overlay when the active selection isn't
        a candidate-for-fitted peak. Manual + detected are both candidates
        — Add-to-fitted is enabled for either — so the preview is shown
        for both kinds. Fitted / matched already have a stored box, so a
        cyan refit overlay there would be visual noise.

        Also kicks ``_refresh_2d_preview`` so the pygidfit override
        on the profile viewer follows the new selection.
        """
        if sel is None or sel.kind not in ("manual", "detected"):
            self.viewer.set_fitted_preview(None, None, None, None)
        self._refresh_2d_preview()

    def _apply_fit_curve_visibility(self) -> None:
        """Decide whether to show the pink fit overlay on the profile plots.

        Hide only when the selection is a *source* peak about to be
        committed via Add-to-fitted (kind ``manual`` or ``detected``)
        AND the active fit mode is 2D — that is the case where the
        pink curve mismatch was misleading (pygidfit's projected 1D
        Gaussian doesn't perfectly match the integrated 1D profile,
        and the image-side cyan box already previews what gets
        saved).

        Show in every other case:
        * 1D mode (the pink curves are the scipy fits
          Add-to-fitted (1D) actually persists)
        * Fitted / matched selections (the curves represent the
          *already-saved* fit and stay informative regardless of
          the panel's mode radio — switching the radio shouldn't
          erase a fitted peak's overlay)

        Wired to ``parameter_panel.fitModeChanged``,
        ``parameter_panel.saveAsRingChanged``, and
        ``viewer.selectionChanged`` so the visibility re-evaluates
        whenever any of those three change.
        """
        sel = self.viewer.selected_peak
        is_source_kind = (
            sel is not None and sel.kind in ("manual", "detected")
        )
        is_2d = (
            self.parameter_panel.fit_mode() == ParameterPanel.FIT_MODE_2D
        )
        visible = not (is_source_kind and is_2d)
        self.profile_viewer.set_fit_curves_visible(visible)

    def _refresh_2d_preview(self) -> None:
        """Run pygidfit on every fittable selected peak (primary +
        Ctrl+click extras) and cache the results so
        ``_update_fitted_preview`` (primary) and
        ``set_fitted_preview_extras`` (extras) paint a cyan box for
        each.

        The profile viewer is intentionally NOT updated from here
        anymore: in 2D mode the pink fit curves are hidden (see
        ``_apply_fit_curve_visibility``) and the grey integrated
        trace stays sliced over the user's ROI, so the only
        consumers of the cached pygidfit result are the image-side
        cyan boxes.

        Wired to ``viewer.selectionChanged``,
        ``viewer.selectionsChanged``, ``viewer.frameChanged``,
        ``parameter_panel.fitModeChanged``,
        ``parameter_panel.saveAsRingChanged``, and
        ``viewer.peakGeometryDragFinished``. The cache is a dict
        keyed by ``(file, entry, frame, sel geometry)`` so a
        previously-fit peak that's re-selected (e.g. user toggles
        a Ctrl+click extra off and back on) reuses the cached fit
        instead of re-paying the ~100-500 ms pygidfit cost.
        """
        save_as_ring = self.parameter_panel.save_as_ring()
        is_2d = self.parameter_panel.fit_mode() == ParameterPanel.FIT_MODE_2D
        applies = (
            is_2d
            and not save_as_ring
            and self.session is not None
            and self._pipe_thread is None
        )
        if not applies:
            self._repaint_fitted_preview()
            return
        entry = self.entry_combo.currentText()
        if not entry:
            self._repaint_fitted_preview()
            return

        # Compute pygidfit for every fittable selection that isn't
        # already in the cache. The cache is preserved across calls
        # so unchanged peaks (e.g. extras whose ROI hasn't moved)
        # don't re-fit.
        for s in self.viewer.selected_peaks():
            if s.kind not in ("manual", "detected"):
                continue
            fp = self._preview_fingerprint(entry, s)
            if fp in self._2d_preview_cache:
                continue
            fit_2d, _err = self._run_pygidfit_for_selection(
                s, entry, int(s.frame),
            )
            # Cache the result either way (None on failure) so we
            # don't retry a failed fit on every selectionsChanged
            # tick.
            self._2d_preview_cache[fp] = fit_2d
        self._repaint_fitted_preview()

    def _preview_fingerprint(
        self, entry: str, sel: SelectedPeak,
    ) -> tuple:
        """Build the cache key for ``sel``'s pygidfit preview."""
        return (
            self.session.temp_path if self.session is not None else None,
            entry,
            int(sel.frame),
            sel.kind,
            int(sel.peak_id) if sel.peak_id is not None else -1,
            float(sel.radius), float(sel.radius_width),
            float(sel.angle), float(sel.angle_width),
        )

    def _schedule_drag_2d_preview(self, _sel) -> None:
        """Debounced ``_refresh_2d_preview`` while a manual / detected
        ROI is being dragged in 2D mode.

        Per-tick ``peakGeometryChanged`` callbacks fan in here; the
        single-shot timer collapses bursts so pygidfit runs at most
        once every ~150 ms during a drag. The drag-end firing of
        ``peakGeometryDragFinished`` triggers a final synchronous
        ``_refresh_2d_preview`` that overrides whatever the timer
        was about to do.
        """
        if not self.viewer.is_dragging:
            return
        if self.parameter_panel.fit_mode() != ParameterPanel.FIT_MODE_2D:
            return
        if self.parameter_panel.save_as_ring():
            return
        # Restart the single-shot timer; bursts collapse into one
        # fire 150 ms after the last drag tick (or the timer is
        # superseded by the drag-end refresh).
        self._drag_pygidfit_timer.start()

    def _repaint_fitted_preview(self) -> None:
        """Re-run the cyan-box painter and the parameter readout
        against the freshest fit state.

        Re-reads the profile viewer's cached scipy 1D fits and
        feeds them through ``_update_fitted_preview`` (which
        prefers pygidfit cache in 2D mode) and
        ``_dispatch_fit_params`` (which routes pygidfit values to
        the parameter panel in 2D mode for the same reason). Also
        pushes the cached pygidfit boxes for the Ctrl+click extras
        into ``viewer.set_fitted_preview_extras`` so every selected
        peak gets its own cyan preview.
        """
        fits = self.profile_viewer.last_fit_params()
        rfit = fits.get("radial")
        afit = fits.get("angular")
        self._update_fitted_preview(rfit, afit)
        self._dispatch_fit_params(rfit, afit)
        self._push_extras_preview()

    def _push_extras_preview(self) -> None:
        """Compute the cyan preview geometries for every Ctrl+click
        extra and push them to the viewer.

        In 2D mode + non-ring + session-active: every extra (always
        detected by construction) gets a cyan box at its cached
        pygidfit-refined ``(r, dr, a, da)``. Extras with no cache
        entry (still computing, or pygidfit failed) fall back to
        the extra's own drawn geometry — better to show *something*
        than to hide the box and let the user wonder if the
        selection registered. In 1D / ring mode the extras preview
        is cleared.
        """
        v = self.viewer
        extras = v._selected_extras
        if not extras:
            v.set_fitted_preview_extras([])
            return
        is_2d_mode = (
            self.parameter_panel.fit_mode() == ParameterPanel.FIT_MODE_2D
            and not self.parameter_panel.save_as_ring()
        )
        if not is_2d_mode:
            v.set_fitted_preview_extras([])
            return
        geoms: list[tuple[float, float, float, float]] = []
        for s in extras:
            if s.kind != "detected":
                continue
            box = self._current_pygidfit_box_for_selection(s)
            if box is None:
                # Fallback: show the user's drawn geometry so the
                # box is at least visible. pygidfit may have failed
                # or just not run yet for this peak.
                geoms.append((
                    float(s.radius), float(s.radius_width),
                    float(s.angle), float(s.angle_width),
                ))
            else:
                geoms.append(box)
        v.set_fitted_preview_extras(geoms)

    def _dispatch_fit_params(self, rfit, afit) -> None:
        """Pick the right fit source for the parameter-panel readout.

        In 2D mode with a manual / detected selection that has a
        cached pygidfit result, the panel should reflect pygidfit's
        refined values (the truth Add-to-fitted (2D) will save), not
        scipy's 1D fits. Outside that case, scipy values pass
        through unchanged.

        When no pygidfit cache exists yet for a 2D-mode source
        selection, the panel is blanked rather than backfilled
        with scipy values — showing scipy numbers in 2D mode would
        misrepresent what the next commit would save.
        """
        sel = self.viewer.selected_peak
        is_source_kind = (
            sel is not None and sel.kind in ("manual", "detected")
        )
        is_2d_mode = (
            self.parameter_panel.fit_mode() == ParameterPanel.FIT_MODE_2D
            and not self.parameter_panel.save_as_ring()
        )
        if not (is_source_kind and is_2d_mode):
            self.parameter_panel.set_fits(rfit, afit)
            return
        # 2D mode + source kind: feed pygidfit-derived values when
        # the cache has them; blank otherwise.
        box = self._current_pygidfit_box_for_selection(sel)
        if box is None:
            self.parameter_panel.set_fits(None, None)
            return
        fp = self._preview_fingerprint(
            self.entry_combo.currentText(), sel,
        )
        fit_2d = self._2d_preview_cache.get(fp)
        if fit_2d is None:
            # Drag-fallback path returned a box from a stale
            # fingerprint — find any cached fit for this peak's
            # identity (file/entry/frame/kind/peak_id).
            for cached_fp, cached_fit in self._2d_preview_cache.items():
                if cached_fp[:5] == fp[:5] and cached_fit is not None:
                    fit_2d = cached_fit
                    break
        if fit_2d is None:
            self.parameter_panel.set_fits(None, None)
            return
        pyg_rfit, pyg_afit = _pygidfit_to_fit_like(fit_2d)
        self.parameter_panel.set_fits(pyg_rfit, pyg_afit)

    def _update_fitted_preview(self, rfit, afit) -> None:
        """Sync the viewer's fitted-preview box to the latest fit params.

        Relevant for manual + detected selections — both feed
        Add-to-fitted. File-resident fitted / matched peaks already
        carry their stored box and aren't previewed here.

        Mode-dependent — cyan box always matches what the
        corresponding commit path will save:

        * **2D mode**: cyan paints at pygidfit's *exact* box
          (``radius, radius_width, angle, angle_width`` straight
          from the cached ``ManualFitResult``). The pink profile
          curve is allowed to drift slightly via the anchored fit
          for visual cleanliness; that drift must not leak into
          the cyan box, which the user expects to match the saved
          blue box exactly.
        * **1D mode** (or ring, which forces 1D): cyan paints at
          ``(scipy_centre, 2σ_scipy)`` per axis — same convention
          ``_build_fitted_row_1d`` saves (``radius_width = FWHM ×
          1/√(2 ln 2)``). Per-axis fallback to the selected peak's
          drawn box when scipy hasn't converged.

        The pygidfit cache outlives a mode toggle, so the 2D-mode
        branch is gated on the live ``fit_mode()`` reading — without
        that gate, switching from 2D to 1D would briefly paint cyan
        at pygidfit's box even though the next commit would use
        scipy's geometry.
        """
        sel = self.viewer.selected_peak
        if sel is None or sel.kind not in ("manual", "detected"):
            self.viewer.set_fitted_preview(None, None, None, None)
            return
        save_as_ring = self.parameter_panel.save_as_ring()
        # Ring forces 1D in ``fit_mode()``; defensive check here keeps
        # the 2D-cyan-box branch off when the user has ring on but the
        # 2D-preview cache hasn't yet been cleared for the new state.
        is_2d_mode = (
            self.parameter_panel.fit_mode() == ParameterPanel.FIT_MODE_2D
            and not save_as_ring
        )

        fwhm_to_2sigma = 1.0 / float(np.sqrt(2.0 * np.log(2.0)))

        # 2D mode + a cached pygidfit result on the current selection:
        # paint the cyan box at pygidfit's *exact* geometry instead of
        # the anchored-fit ``rfit``/``afit``. The pink profile curve
        # was deliberately allowed to drift (centre ±0.5σ, sigma ×1.2)
        # so it sits clean on the data; that drift must NOT leak into
        # the image-side cyan box, which the user expects to match
        # the saved blue box exactly. Skipped in 1D mode because the
        # pygidfit cache can persist after a mode switch — 1D commit
        # uses scipy's fit, so cyan must follow rfit/afit there.
        if is_2d_mode:
            pygidfit_box = self._current_pygidfit_box_for_selection(sel)
            if pygidfit_box is not None:
                fr, fdr, fa, fda = pygidfit_box
                self.viewer.set_fitted_preview(
                    fr, fdr, fa, fda, is_ring=False,
                )
                return

        if rfit is not None:
            center_r = float(rfit.center)
            width_r = float(rfit.fwhm) * fwhm_to_2sigma
        else:
            center_r = float(sel.radius)
            width_r = float(sel.radius_width)

        if save_as_ring:
            # Ring sentinel — the painter ignores the angular args.
            # Ring forces 1D so the cached pygidfit result is never
            # used here; the radial width comes from scipy's rfit.
            self.viewer.set_fitted_preview(
                center_r, width_r, None, None, is_ring=True,
            )
            return

        if afit is not None:
            center_a = float(afit.center)
            width_a = float(afit.fwhm) * fwhm_to_2sigma
        else:
            center_a = float(sel.angle)
            width_a = float(sel.angle_width)

        self.viewer.set_fitted_preview(
            center_r, width_r, center_a, width_a, is_ring=False,
        )

    def _current_pygidfit_box_for_selection(
        self, sel: SelectedPeak,
    ) -> tuple[float, float, float, float] | None:
        """Return pygidfit's refined box for ``sel`` if cached, else None.

        The 2D-preview cache keyed by ``(file, entry, frame, sel
        geometry)`` is built in ``_refresh_2d_preview``. If the cache
        entry matches the current ``sel`` and holds a non-None
        ``ManualFitResult``, return its ``(r, dr, a, da)``. Otherwise
        return None — caller falls back to the rfit/afit-driven cyan
        box (the 1D-mode / no-pygidfit-yet path).

        While the viewer reports an active ROI drag, the geometry
        fields of ``sel`` change per tick faster than pygidfit can
        keep up, so this method matches on the *peak identity*
        only (file / entry / frame / kind / peak_id) and returns
        the cached refined box from the last completed pygidfit
        run. That keeps the cyan preview tracking pygidfit's
        output through the drag instead of flickering into the
        scipy 1D fallback between debounced refits.
        """
        if not self._2d_preview_cache:
            return None
        if self.session is None:
            return None
        entry = self.entry_combo.currentText()
        if not entry:
            return None
        peak_id = int(sel.peak_id) if sel.peak_id is not None else -1
        identity = (
            self.session.temp_path, entry, int(sel.frame), sel.kind, peak_id,
        )
        full_fp = identity + (
            float(sel.radius), float(sel.radius_width),
            float(sel.angle), float(sel.angle_width),
        )
        if self.viewer.is_dragging:
            # During drag: match on identity only so the last
            # cached refined box survives geometry mismatches.
            # Pick the first non-None fit for the matching identity
            # (typically there's just one).
            fit_2d = None
            for cache_fp, cached_fit in self._2d_preview_cache.items():
                if cache_fp[:5] == identity and cached_fit is not None:
                    fit_2d = cached_fit
                    break
            if fit_2d is None:
                return None
        else:
            fit_2d = self._2d_preview_cache.get(full_fp)
            if fit_2d is None:
                return None
        return (
            float(fit_2d.radius), float(fit_2d.radius_width),
            float(fit_2d.angle), float(fit_2d.angle_width),
        )

    def _on_save_as_ring_changed(self, is_ring: bool) -> None:
        """Toggle between segment / ring preview.

        Three coordinated effects:

        1. The profile viewer skips the angular Gaussian fit while ring
           is active — that fit wouldn't be saved by Add-to-fitted.
        2. If a manual peak is selected, its angular sweep is widened
           to span the full polar plot height (so the radial profile
           integrates over the entire angular axis, matching what the
           ring fit will eventually represent). The pre-ring geometry
           is stashed so unticking the box — including the auto-uncheck
           that fires after Add-to-fitted commits — restores the box.
        3. The fitted-preview is recomputed against the new fit cache.
        """
        sel = self.viewer.selected_peak
        manual_ref = (
            sel.manual_ref if sel is not None and sel.kind == "manual" else None
        )

        if is_ring and manual_ref is not None:
            # Stash pre-ring geometry once. If the user ticks → unticks
            # → re-ticks without committing, we keep the original stash
            # so the eventual restore returns to the very first state,
            # not the intermediate ring state.
            if self._ring_pre_geom is None:
                self._ring_pre_geom = (
                    manual_ref,
                    manual_ref.radius,
                    manual_ref.angle,
                    manual_ref.radius_width,
                    manual_ref.angle_width,
                    manual_ref.is_ring,
                )
            extent = self.viewer.angular_extent()
            if extent is not None:
                a_lo, a_hi = extent
                ring_angle = 0.5 * (a_lo + a_hi)
                ring_width = abs(a_hi - a_lo)
                self.viewer.set_manual_geometry(
                    manual_ref,
                    radius=manual_ref.radius,
                    angle=ring_angle,
                    radius_width=manual_ref.radius_width,
                    angle_width=ring_width,
                    is_ring=True,
                )
        elif not is_ring and self._ring_pre_geom is not None:
            (
                stashed_peak,
                pre_r,
                pre_a,
                pre_dr,
                pre_da,
                pre_is_ring,
            ) = self._ring_pre_geom
            self._ring_pre_geom = None
            # Only restore if the stashed peak still exists — the user
            # may have drawn a replacement (which removes the original
            # via the single-box policy) while ring was active. In that
            # case the new peak inherited the ring geometry but has no
            # captured pre-state, so leave it alone.
            for peaks in self.viewer._manual_peaks.values():
                if stashed_peak in peaks:
                    self.viewer.set_manual_geometry(
                        stashed_peak,
                        radius=pre_r,
                        angle=pre_a,
                        radius_width=pre_dr,
                        angle_width=pre_da,
                        is_ring=pre_is_ring,
                    )
                    break

        # Drop the angular fit *before* recomputing the preview so the
        # cached afit is None when _update_fitted_preview reads it.
        self.profile_viewer.set_skip_angular_fit(is_ring)
        fits = self.profile_viewer.last_fit_params()
        self._update_fitted_preview(fits.get("radial"), fits.get("angular"))

    def _on_fit_mode_changed(self, _mode: str) -> None:
        """Re-render the cyan preview when the user flips 1D ↔ 2D.

        The preview is only painted in 1D mode (see
        ``_update_fitted_preview``); flipping to 2D hides it,
        flipping back shows it. Without this hook the on-screen
        state wouldn't update until the next ROI / frame /
        selection change.
        """
        fits = self.profile_viewer.last_fit_params()
        self._update_fitted_preview(fits.get("radial"), fits.get("angular"))

    def _on_manual_peak_added(self, _frame: int, peak: ManualPeak) -> None:
        """Apply the active ring expansion to a freshly added manual peak.

        When the user draws a new manual box while the ring checkbox is
        on, the single-box-replace removes the old (with its ring stash)
        and adds the new one. Without this slot, the new box would stay
        as drawn — confusing because the checkbox is still ticked. We
        mirror what ``_on_save_as_ring_changed(True)`` would do for the
        new peak: stash its pre-ring shape, then expand to the full
        angular sweep.
        """
        if not self.parameter_panel.save_as_ring():
            return
        # Stash pre-ring state for the new peak. Any earlier stash
        # already pointed at a peak that's been removed (which our
        # manualPeakRemoved slot has already cleared).
        self._ring_pre_geom = (
            peak,
            peak.radius,
            peak.angle,
            peak.radius_width,
            peak.angle_width,
            peak.is_ring,
        )
        extent = self.viewer.angular_extent()
        if extent is None:
            return
        a_lo, a_hi = extent
        self.viewer.set_manual_geometry(
            peak,
            radius=peak.radius,
            angle=0.5 * (a_lo + a_hi),
            radius_width=peak.radius_width,
            angle_width=abs(a_hi - a_lo),
            is_ring=True,
        )

    def _on_manual_peak_removed(self, _frame: int, peak: ManualPeak) -> None:
        """Invalidate ``_ring_pre_geom`` when the peak it references goes away.

        Without this, an Esc / Delete / Add-to-detected on a ring-
        expanded peak would leave a dangling stash; later unticking
        the ring checkbox would walk the manual list looking for that
        ghost and find nothing, but the stash stays set and could
        mis-fire on a later toggle cycle.
        """
        if (
            self._ring_pre_geom is not None
            and self._ring_pre_geom[0] is peak
        ):
            self._ring_pre_geom = None

    # -- Pipeline --

    def _on_parse_cifs_requested(self, cif_input: str) -> None:
        """Run CIF parsing on a worker thread + post the result back.

        CIF preprocessing simulates every CIF and can take several
        seconds; the worker keeps the GUI responsive. Only one parse
        runs at a time — the panel's button stays disabled until we
        post the result back via ``set_cif_pattern``.
        """
        if self._cif_parse_thread is not None:
            return
        if self.session is None:
            self.pipeline_panel.set_cif_pattern(
                None, RuntimeError("Open a NeXus file first.")
            )
            return
        nexus_file = self.session.temp_path
        # Pass the active entry through so CifPattern is simulated
        # against that entry's energy / angle of incidence — multi-
        # energy datasets need this to match correctly.
        active_entry = self.entry_combo.currentText() or None
        self._cif_parse_thread = QThread(self)
        self._cif_parse_worker = CifParseWorker(
            cif_input, nexus_file, active_entry
        )
        self._cif_parse_worker.moveToThread(self._cif_parse_thread)
        self._cif_parse_thread.started.connect(self._cif_parse_worker.run)
        self._cif_parse_worker.finished.connect(self._on_parse_cifs_finished)
        self._cif_parse_thread.start()

    def _on_parse_cifs_finished(
        self, result: object | None, error: Exception | None
    ) -> None:
        if self._cif_parse_thread is not None:
            self._cif_parse_thread.quit()
            self._cif_parse_thread.wait()
            self._cif_parse_thread.deleteLater()
            self._cif_parse_thread = None
        if self._cif_parse_worker is not None:
            self._cif_parse_worker.deleteLater()
            self._cif_parse_worker = None
        self.pipeline_panel.set_cif_pattern(result, error)
        if error is not None:
            self.pipeline_panel.append_log(f"CIF parse failed: {error}")
        elif result is not None:
            n = len(getattr(result, "cifs", []) or [])
            self.pipeline_panel.append_log(
                f"CIF cache loaded ({n} CIFs) — reused across matching runs"
            )

    def _on_run_requested(self, command: PipelineCommand) -> None:
        """Dispatch a runRequested command from the pipeline panel.

        "All entries" runs are expanded into one ``PipelineCommand`` per
        q-entry and queued sequentially — the user gets per-entry log
        lines and a single bad entry doesn't strand the others. A command
        that already names an explicit ``entry`` (or runs on a file with
        a single entry) goes straight through unchanged.

        ``add_peak`` and ``delete_peak`` always carry a specific
        ``entry`` already; only the run_* ops are subject to expansion.

        The file path is snapshotted at this entry point and travels
        with every enqueued tuple, so a mid-queue active-session
        switch can't dispatch later commands at a different file.
        """
        if self.session is None:
            return
        if self._view_worker_blocks_pipeline():
            return
        file_path = self.session.temp_path
        if (
            command.op_name in ("run_detection", "run_fitting", "run_matching")
            and "entry" not in command.kwargs
        ):
            try:
                entries = file_model.list_entries(file_path)
            except Exception as exc:
                QMessageBox.warning(
                    self, "Pipeline", f"Could not list entries: {exc}"
                )
                return
            if not entries:
                # No q-entries to run on — fall through and let mlgidbase
                # raise its usual "no entries" message in the log.
                self._entry_queue_total = 1
                self._entry_queue_pos = 0
                self._enqueue_pipeline(file_path, command)
                return
            # Multi-entry expansion: stash the queue depth + reset the
            # position counter so the entry progress bar starts at 0/N
            # and the first ``_on_pipeline_run`` advances to 1/N.
            self._entry_queue_total = len(entries)
            self._entry_queue_pos = 0
            for entry in entries:
                self._enqueue_pipeline(
                    file_path,
                    PipelineCommand(
                        command.op_name,
                        {**command.kwargs, "entry": entry},
                    ),
                )
        else:
            # Single-entry command (user picked one explicitly, or
            # add_peak / delete_peak with their fixed entry). Counter
            # collapses to 1 so the panel keeps the entry bar hidden.
            self._entry_queue_total = 1
            self._entry_queue_pos = 0
            self._enqueue_pipeline(file_path, command)

    def _enqueue_pipeline(self, file_path: Path, command: PipelineCommand) -> None:
        """Queue ``(file_path, command)`` and start it if no run is in flight."""
        self._pipeline_queue.append((file_path, command))
        if self._pipe_thread is None:
            self._run_next_pipeline_command()

    def _run_next_pipeline_command(self) -> None:
        """Pop the next queued (path, command) tuple and start it, if any."""
        if self._pipe_thread is not None or not self._pipeline_queue:
            return
        file_path, command = self._pipeline_queue.pop(0)
        self._on_pipeline_run(file_path, command)

    def _ensure_entry_normalized(self, file_path: Path, entry: str | None) -> None:
        """Run pygid normalization for ``entry`` once per (file, entry).

        Normalization is deferred off the open path (see
        ``_on_open_finished``); this runs it lazily, scoped to the one
        entry a pipeline command targets. **Must be called with the silx
        tree detached** (no read handle on ``file_path``) — it writes new
        groups, which can't race the viewer's reads. Idempotent on repeat
        calls; a failure is non-fatal (the pipeline surfaces its own
        error if the file is genuinely unusable).
        """
        if not entry:
            return
        done = self._normalized_entries.setdefault(str(file_path), set())
        if entry in done:
            return
        try:
            patched = file_model.normalize_for_pygid(file_path, entry=entry)
        except Exception:
            logger.debug("suppressed exception in MainWindow._ensure_entry_normalized", exc_info=True)
            patched = {"angle": [], "frames": []}
        if patched.get("angle"):
            self.pipeline_panel.append_log(
                "Normalized 0-D angle_of_incidence in: " + ", ".join(patched["angle"])
            )
        if patched.get("frames"):
            self.pipeline_panel.append_log(
                "Created missing per-frame analysis groups in: "
                + ", ".join(patched["frames"])
            )
        done.add(entry)

    def _on_pipeline_run(self, file_path: Path, command: PipelineCommand) -> None:
        if self.session is None or self._pipe_thread is not None:
            return

        # Stop frame playback if it's running — the pipeline owns the
        # file r+ for the duration of the run and ticking would either
        # contend on the silx detach or read post-write data
        # mid-render.
        self._pause_playback()
        self.pipeline_panel.set_running(True)
        self.parameter_panel.set_busy(True)
        self.scan_tracking_panel.set_busy(True)
        self.viewer.set_busy(True)
        self._update_status_pipeline(command, running=True)
        # Any pipeline op that reshuffles peak ids (everything except
        # add_peak, which only appends) invalidates pending FileGeomActions.
        # add_peak is handled by commit_manual_peak's targeted scrub.
        if command.op_name != "add_peak":
            self.viewer.clear_history()
            self.viewer.clear_selection()
        # Per-run header: include the entry scope when present so the
        # user can see which entry is being processed in a multi-entry
        # batch.
        entry_tag = command.kwargs.get("entry")
        if entry_tag:
            self.pipeline_panel.append_log(
                f"--- {command.op_name} on {entry_tag} ---"
            )
        else:
            self.pipeline_panel.append_log(
                f"--- {command.op_name} (all entries) ---"
            )

        # Release silx's read handles on every loaded temp file so mlgidbase
        # can open the active one r+. Sibling files are reattached on finish.
        self._detach_silx_tree()

        # Normalize this entry for pygid now — lazily, once per entry.
        # Done here (tree detached, before the worker's r+ open) so the
        # group-creating write can't race the viewer's reads. Replaces the
        # old normalize-every-entry-on-open that froze huge files.
        self._ensure_entry_normalized(file_path, command.kwargs.get("entry"))

        # Stash the active command so the status-bar progress mirror
        # (``_on_pipeline_frame_progress``) can rebuild the status text
        # without needing to plumb the command through every emit.
        self._pipe_command = command
        self._pipe_progress_tail = ""

        # Advance the entry-level queue position and drive the panel's
        # second progress row. Single-entry runs (total == 1) keep the
        # entry bar hidden via the panel's own guard.
        if self._entry_queue_total >= 1:
            self._entry_queue_pos += 1
        entry_for_progress = command.kwargs.get("entry") or ""
        self.pipeline_panel.on_queue_progress(
            self._entry_queue_pos,
            self._entry_queue_total,
            entry_for_progress,
            command.op_name,
        )

        self._pipe_thread = QThread(self)
        # Use the file_path snapshotted at enqueue time, NOT
        # ``self.session.temp_path`` — they can disagree if the user
        # switched the active session between clicking Run and the
        # queue actually dispatching this command. Surfaced as a
        # pre-flight failure in pipeline.execute that named the wrong
        # file's entries.
        self._pipe_worker = PipelineWorker(file_path, command)
        self._pipe_worker.moveToThread(self._pipe_thread)
        self._pipe_worker.log.connect(self.pipeline_panel.append_log)
        self._pipe_worker.frameProgress.connect(self.pipeline_panel.on_frame_progress)
        # Mirror the frame counter into the status-bar tail so the user
        # gets a glanceable counter without needing the pipeline panel
        # in view. Stored on ``self`` so ``_update_status_pipeline``
        # can fold it into the status string.
        self._pipe_worker.frameProgress.connect(self._on_pipeline_frame_progress)
        self._pipe_worker.finished.connect(self._on_pipeline_finished)
        self._pipe_thread.started.connect(self._pipe_worker.run)
        self._pipe_thread.start()

    # -- Conversion (raw → NeXus) --

    def _on_conversion_run(self, cfg, scans: list) -> None:
        """Spawn the ConversionWorker for a fresh run.

        ``cfg`` is a ``ConversionConfig``; ``scans`` is a list of
        ``RawScan``. We don't refuse on overlapping output paths here
        — pygid handles overwrite-or-append per scan via ``cfg``'s
        flags.
        """
        if self._conv_thread is not None:
            QMessageBox.information(
                self, "Conversion in progress",
                "A conversion run is already in flight; please wait for it "
                "to finish before starting another.",
            )
            return
        # Modal progress dialog — a long batch can run for minutes; the
        # user needs a way to see it's progressing without watching the
        # log pane scroll.
        self._conv_progress = QProgressDialog(
            "Converting…", "", 0, max(len(scans), 1), self
        )
        self._conv_progress.setWindowTitle(APP_NAME)
        self._conv_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._conv_progress.setCancelButton(None)
        self._conv_progress.setMinimumDuration(0)
        self._conv_progress.setLabelText(
            f"Running {len(scans)} scan(s)…"
        )
        self._conv_progress.show()

        self.conversion_panel.set_running(True)
        self.conversion_panel.clear_log()
        self.conversion_panel.append_log(
            f"Starting conversion: {len(scans)} scan(s) → {cfg.output_dir}"
        )

        self._conv_thread = QThread(self)
        self._conv_worker = ConversionWorker(scans, cfg)
        self._conv_worker.moveToThread(self._conv_thread)
        self._conv_thread.started.connect(self._conv_worker.run)
        self._conv_worker.log.connect(self.conversion_panel.append_log)
        self._conv_worker.progress.connect(self._on_conversion_progress)
        self._conv_worker.finished.connect(self._on_conversion_finished)
        self._conv_thread.start()

    def _on_conversion_progress(self, done: int, total: int) -> None:
        if self._conv_progress is None:
            return
        self._conv_progress.setMaximum(max(total, 1))
        self._conv_progress.setValue(done)

    def _on_conversion_finished(
        self, output_paths: list | None, error: Exception | None
    ) -> None:
        if self._conv_thread is not None:
            self._conv_thread.quit()
            self._conv_thread.wait()
            self._conv_thread.deleteLater()
            self._conv_thread = None
        if self._conv_worker is not None:
            self._conv_worker.deleteLater()
            self._conv_worker = None
        if self._conv_progress is not None:
            self._conv_progress.close()
            self._conv_progress.deleteLater()
            self._conv_progress = None

        self.conversion_panel.set_running(False)

        if error is not None:
            self.conversion_panel.append_log(f"ERROR - {error}")
            QMessageBox.critical(self, "Conversion failed", str(error))
            return

        outputs = list(output_paths or [])
        if not outputs:
            self.conversion_panel.append_log(
                "Conversion completed but produced no output paths."
            )
            return

        self.conversion_panel.append_log(
            "Conversion DONE. Output files:\n  " + "\n  ".join(str(p) for p in outputs)
        )

        # Auto-open: queue every produced file as a NeXus session. The
        # existing CopyWorker path normalizes pygid metadata and handles
        # silx-tree insertion; ``_set_active_session`` swaps focus once
        # the first file lands.
        for out_path in outputs:
            self._open_queue.append(Path(out_path))
        self._process_open_queue()

    def _on_pipeline_finished(self, result: object, error: Exception | None) -> None:
        if self._pipe_thread is not None:
            self._pipe_thread.quit()
            self._pipe_thread.wait()
            self._pipe_thread.deleteLater()
            self._pipe_thread = None
        if self._pipe_worker is not None:
            self._pipe_worker.deleteLater()
            self._pipe_worker = None

        # Close the tracking loading dialog as soon as its run finishes
        # (before any error modal), regardless of success/failure. The
        # Interpolate-track chain keeps it up across all its queued
        # commands; it closes below when the chain's LAST command lands.
        if (
            getattr(self, "_pipe_command", None) is not None
            and self._pipe_command.op_name == "track_peaks"
        ):
            self._close_tracking_progress()

        if error is not None:
            self.pipeline_panel.append_log(f"ERROR - {error}")
            # In a queued multi-entry batch, surface the error in the log
            # but only show the modal once at the *end* (otherwise the user
            # gets a dialog per entry and the run halts in front of every
            # one). For single-command runs (queue empty) keep the modal.
            if not self._pipeline_queue:
                QMessageBox.critical(self, "Pipeline error", str(error))
        else:
            self.pipeline_panel.append_log("DONE")

        if self.session is not None and error is None:
            self.session.mark_dirty()

        # A re-fit rewrites the entry's fitted_peaks wholesale, so any
        # phase-tracking results for that entry are stale. Invalidate
        # when the finished command was a fitting run touching the
        # tracked entry (kwargs without an entry ran on every entry).
        cmd = getattr(self, "_pipe_command", None)
        if (
            error is None
            and self._scan_track_entry is not None
            and cmd is not None
            and cmd.op_name == "run_fitting"
            and cmd.kwargs.get("entry") in (None, self._scan_track_entry)
        ):
            self._invalidate_scan_tracks()

        # track_peaks is the one pipeline op whose RESULT the GUI
        # consumes (the captured TrackingPayload) — every other op's
        # output is read back from the file.
        if error is None and cmd is not None and cmd.op_name == "track_peaks":
            self._on_phase_track_result(result, cmd)

        # interpolate_tracks also returns its result (the gap-fill
        # records) — stash it until the chained re-match lands, then
        # apply the new peaks as track members. Applied even when the
        # re-match itself errored: the fills are real fitted rows either
        # way, and the error modal above already told the user.
        if (
            error is None
            and cmd is not None
            and cmd.op_name == "interpolate_tracks"
        ):
            self._interp_fill_result = list(result) if result else None
            if not result:
                self.pipeline_panel.append_log(
                    "Interpolate track: no gap could be filled "
                    "(every fit failed or was skipped — see warnings "
                    "above)."
                )
        # Interpolate-track chain accounting: advance the dialog's
        # completed-work base, and when the chain's LAST command lands
        # (the final re-match — or the fill itself when every re-match
        # was skipped), apply the fills as track members and drop the
        # dialog. Applied even when a re-match errored: the fills are
        # real fitted rows either way, and the error modal above
        # already told the user.
        chain = self._interp_chain
        if chain is not None and cmd is not None and id(cmd) in chain["ticks"]:
            chain["base"] += chain["ticks"].pop(id(cmd))
            if not chain["ticks"]:
                self._interp_chain = None
                self._close_tracking_progress()
                self._apply_interp_fill_result()

        # If more commands are queued, run the next one without
        # tearing down the silx tree / viewer state for the user — keep
        # the busy gating active and chain straight into the next run.
        if self._pipeline_queue:
            self._run_next_pipeline_command()
            return

        # Queue drained — final cleanup. Reattach silx, refresh the
        # viewer for the active entry, lift busy gating. Reset the
        # entry-queue counters so the next run starts from a clean
        # slate (the panel's set_running(False) below also hides both
        # progress rows).
        self._entry_queue_total = 0
        self._entry_queue_pos = 0
        self.pipeline_panel.set_running(False)
        self.parameter_panel.set_busy(False)
        self.scan_tracking_panel.set_busy(False)
        self.viewer.set_busy(False)
        self._update_status_pipeline(running=False)
        self._reattach_silx_tree()
        if self.session is not None:
            entry = self.entry_combo.currentText()
            if entry:
                # Same entry, same axes — preserve the user's zoom and
                # frame across the overlay refresh.
                self._load_entry_into_viewer(entry, preserve_view=True)
            self._update_title()

    def _on_add_to_detected(self) -> None:
        """Commit the active manual box to detected_peaks at the chosen
        confidence score.

        Writes synchronously via ``file_model.add_detected_peak_row`` —
        the same path copy/paste uses — so the score picked in the
        Parameter panel's Score-row dropdown (High = 1.0 / Medium = 0.5 /
        Low = 0.1) is stored verbatim. ``mlgidBASE.add_peak`` takes no
        score argument, so it can't carry the choice; a hand-drawn box
        has no model provenance anyway, so the user's pick is the
        authority. The manual box is left in place so it can also be sent
        to fitted_peaks or tweaked further. One grouped Ctrl+Z removes the
        added row.
        """
        if self.session is None or self._pipe_thread is not None:
            return
        sel = self.viewer.selected_peak
        entry = self.entry_combo.currentText()
        if sel is None or sel.kind != "manual" or sel.manual_ref is None or not entry:
            return
        manual_peak = sel.manual_ref
        frame = int(self.viewer.current_frame)
        score = float(self.parameter_panel.confidence_score())
        geom = {
            "radius": float(manual_peak.radius),
            "angle": float(manual_peak.angle),
            "radius_width": float(manual_peak.radius_width),
            "angle_width": float(manual_peak.angle_width),
            "is_ring": bool(manual_peak.is_ring),
        }
        with self._detached_silx_tree():
            try:
                new_id = int(file_model.add_detected_peak_row(
                    self.session.temp_path, entry, frame,
                    score=score, **geom,
                ))
            except KeyError as exc:
                QMessageBox.warning(self, "Add to detected", str(exc))
                return
            except Exception as exc:
                QMessageBox.critical(self, "Add to detected", str(exc))
                return
        self.session.mark_dirty()
        self._update_title()
        self._load_entry_into_viewer(entry, preserve_view=True)
        self._add_detected_to_selection(entry, frame, [new_id])
        self._push_add_detected_undo(
            entry=entry, frame=frame, peak_id=new_id, score=score, geom=geom,
        )
        msg = (
            f"Added detected peak (score {score:.2f}) on "
            f"{entry}/frame{frame:05d}"
        )
        self.statusBar().showMessage(msg, 4000)
        self.pipeline_panel.append_log(msg)

    def _push_add_detected_undo(
        self, *, entry: str, frame: int, peak_id: int,
        score: float, geom: dict,
    ) -> None:
        """Undo / redo for one 'Add to detected' commit.

        ``undo`` deletes the added row; ``redo`` re-adds it with the same
        geometry + score (fresh id). ``state["id"]`` rotates with each
        re-add so a follow-on undo deletes the right row — same pattern as
        ``_push_paste_undo`` / ``_push_delete_undo``.
        """
        state = {"id": int(peak_id)}
        g = dict(geom)

        def _undo() -> None:
            if self.session is None or self._pipe_thread is not None:
                return
            if self.entry_combo.currentText() != entry:
                self.entry_combo.setCurrentText(entry)
            if int(self.viewer.current_frame) != int(frame):
                self.viewer.set_frame(int(frame))
            with self._detached_silx_tree():
                try:
                    file_model.delete_peak_row(
                        self.session.temp_path, entry,
                        frame=int(frame), kind="detected",
                        peak_id=int(state["id"]),
                    )
                except Exception as exc:
                    QMessageBox.critical(self, "Undo add to detected", str(exc))
                    return
            self.session.mark_dirty()
            self._update_title()
            self._load_entry_into_viewer(entry, preserve_view=True)
            self._drop_detected_from_selection(int(frame), [state["id"]])
            self.pipeline_panel.append_log(
                f"Undo add to detected: removed peak on "
                f"{entry}/frame{int(frame):05d}"
            )

        def _redo() -> None:
            if self.session is None or self._pipe_thread is not None:
                return
            if self.entry_combo.currentText() != entry:
                self.entry_combo.setCurrentText(entry)
            if int(self.viewer.current_frame) != int(frame):
                self.viewer.set_frame(int(frame))
            with self._detached_silx_tree():
                try:
                    new_id = int(file_model.add_detected_peak_row(
                        self.session.temp_path, entry, int(frame),
                        score=float(score), **g,
                    ))
                except Exception as exc:
                    QMessageBox.critical(self, "Redo add to detected", str(exc))
                    return
            state["id"] = new_id
            self.session.mark_dirty()
            self._update_title()
            self._load_entry_into_viewer(entry, preserve_view=True)
            self._add_detected_to_selection(entry, int(frame), [new_id])
            self.pipeline_panel.append_log(
                f"Redo add to detected: re-added peak on "
                f"{entry}/frame{int(frame):05d}"
            )

        self.viewer._push_undo(_CallbackAction(_undo, _redo))

    def _on_add_to_fitted(self) -> None:
        """Append a row to fitted_peaks for the active manual / detected box.

        Dispatches strictly on ``parameter_panel.fit_mode()``:

        * ``FIT_MODE_2D`` (pygidfit) → route through
          ``manual_fit.fit_one_peak``; the persisted row carries real
          A/B/C/theta shape coefficients — identical to what the
          pipeline ``run_fitting`` writes for the same box.
        * ``FIT_MODE_1D`` (legacy scipy) → use the profile viewer's
          cached 1D fits + zero-fill the 2D shape coefficients. Width
          convention matches the 2D path so saved boxes render
          identically: ``radius_width = angle_width = 2σ ≈ 0.849 ×
          FWHM`` (see ``_build_fitted_row_1d``).

        Ring storage forces 1D regardless of the radio (pygidfit's
        segment model can't fit rings — the ring saved row uses
        ``angle = 45°``, ``angle_width = ∞``).

        Strict failure: if the chosen mode can't produce a fit (1D
        without scipy convergence, or 2D with pygidfit raising / NaN /
        missing geometry), the user gets a clear error naming both
        the chosen mode and the reason — no silent fall-back to the
        other mode. Physics-audit finding F-06 closure.
        """
        if self.session is None or self._pipe_thread is not None:
            return
        sel = self.viewer.selected_peak
        entry = self.entry_combo.currentText()
        if sel is None or sel.kind not in ("manual", "detected") or not entry:
            return
        save_as_ring = self.parameter_panel.save_as_ring()
        frame = self.viewer.current_frame
        # ``fit_mode()`` already collapses to FIT_MODE_1D when ring is
        # checked, so the rest of this function just branches on the
        # token.
        mode = self.parameter_panel.fit_mode()

        if mode == ParameterPanel.FIT_MODE_2D:
            row = self._build_fitted_row_2d(sel, entry, frame)
        else:
            row = self._build_fitted_row_1d(sel, save_as_ring)
        if row is None:
            # The helper already showed a QMessageBox / log line.
            return

        with self._detached_silx_tree():
            try:
                new_id = file_model.add_fitted_peak_row(
                    self.session.temp_path, entry, frame,
                    radius=row["radius"],
                    radius_width=row["radius_width"],
                    angle=row["angle"],
                    angle_width=row["angle_width"],
                    amplitude=row["amplitude"],
                    is_ring=save_as_ring,
                    theta=row["theta"],
                    A=row["A"],
                    B=row["B"],
                    C=row["C"],
                )
            except KeyError as exc:
                QMessageBox.warning(self, "Add to fitted", str(exc))
                return
            except Exception as exc:
                QMessageBox.critical(self, "Add to fitted", str(exc))
                return

        self.session.mark_dirty()
        self._update_title()
        # Add-to-fitted only appends a fitted row (fresh id); no existing
        # id is invalidated, so prior undo history is preserved.
        # Pull the fresh fitted_peaks (and matched, which references it) back
        # into the viewer — same entry, so preserve the user's zoom + frame.
        self._load_entry_into_viewer(entry, preserve_view=True)
        # In 2D mode the user committed without seeing pygidfit's actual
        # result on the profile plots (the pink curves are hidden in
        # 2D mode — see ``_apply_fit_curve_visibility``). Auto-switch
        # the selection to the new fitted peak so the result is
        # immediately visible (cyan box + profile fits both update to
        # the fitted-kind selection). 1D mode is unchanged because the
        # pink curves already previewed what was saved.
        if mode == ParameterPanel.FIT_MODE_2D:
            fitted_sel = SelectedPeak(
                kind="fitted",
                frame=int(frame),
                peak_id=int(new_id),
                radius=float(row["radius"]),
                angle=float(row["angle"]),
                radius_width=float(row["radius_width"]),
                angle_width=float(row["angle_width"]),
                is_ring=bool(save_as_ring),
                amplitude=(
                    float(row["amplitude"])
                    if row.get("amplitude") is not None else None
                ),
            )
            # ``preserve_manual=True`` so the source manual peak
            # is not auto-removed by the away-from-manual side
            # effect on _set_selected. Without this, the silent
            # ManualRemoveAction it pushes would sit on the undo
            # stack alongside our FittedRowAction and the user
            # would need TWO Ctrl+Z presses to revert the commit.
            self.viewer._set_selected(fitted_sel, preserve_manual=True)
        # Ctrl+Z support for Add-to-fitted: the file mutation is
        # reversible (delete the just-added row + restore the source
        # selection). Pushed onto the existing history so the commit
        # stacks with any prior undoable edits.
        self._push_fitted_add_undo(
            entry=entry, frame=int(frame), new_id=int(new_id),
            add_kwargs={
                "radius": row["radius"],
                "radius_width": row["radius_width"],
                "angle": row["angle"],
                "angle_width": row["angle_width"],
                "amplitude": row["amplitude"],
                "is_ring": save_as_ring,
                "theta": row["theta"],
                "A": row["A"], "B": row["B"], "C": row["C"],
            },
            source_sel=sel,
        )
        # Ring toggle is sticky across selections but reset on commit so
        # the next Add-to-fitted defaults back to segment unless the user
        # explicitly opts in again.
        self.parameter_panel.reset_save_as_ring()
        self.pipeline_panel.append_log(
            f"Added fitted peak id={new_id} "
            f"({'ring' if save_as_ring else 'segment'} from {sel.kind}) on "
            f"{entry}/frame{frame:05d}"
        )

    def _push_fitted_add_undo(
        self, *, entry: str, frame: int, new_id: int,
        add_kwargs: dict, source_sel: SelectedPeak,
    ) -> None:
        """Push a Ctrl+Z entry for an Add-to-fitted commit.

        ``undo`` deletes the just-added fitted row (and cascades
        ``clear_peaks(matched)`` on the frame, mirroring the
        existing fitted-delete handler), then reselects the source
        peak so the user is back where they started visually. For
        a manual source, the manual peak still exists in
        ``viewer._manual_peaks`` (Add-to-fitted leaves it alone),
        so reselecting it surfaces the yellow ROI again — exactly
        the "let the manual selection reappear" behaviour from the
        feature spec. For a detected source, the detected row is
        located by id in the freshly-reloaded peaks table.

        ``redo`` re-adds the row with the original kwargs.
        ``file_model.add_fitted_peak_row`` assigns a fresh id on
        each call (``max(id)+1``), so the post-redo id may differ
        from the original; ``state["current_id"]`` is updated so
        a follow-on undo deletes the correct row.

        On either side, any file error surfaces a QMessageBox and
        leaves the stack alone — the caller's downstream slot
        (``undo_last_action`` / ``redo_last_action``) is tolerant
        of no-op actions.
        """
        # Closure state for the rotating fitted-row id across
        # repeated undo/redo cycles.
        state = {"current_id": int(new_id)}
        source_kind = source_sel.kind
        source_peak_id = (
            int(source_sel.peak_id)
            if source_sel.peak_id is not None else None
        )

        def _undo() -> None:
            if self.session is None or self._pipe_thread is not None:
                return
            if self.entry_combo.currentText() != entry:
                self.entry_combo.setCurrentText(entry)
            if int(self.viewer.current_frame) != int(frame):
                self.viewer.set_frame(int(frame))
            with self._detached_silx_tree():
                try:
                    removed = file_model.delete_peak_row(
                        self.session.temp_path, entry,
                        frame=int(frame), kind="fitted",
                        peak_id=int(state["current_id"]),
                    )
                    if removed > 0:
                        file_model.clear_peaks(
                            self.session.temp_path, entry,
                            kind="matched", frame=int(frame),
                        )
                except Exception as exc:
                    QMessageBox.critical(
                        self, "Undo Add to fitted", str(exc)
                    )
                    return
            if removed == 0:
                self.pipeline_panel.append_log(
                    f"Undo Add-to-fitted: fitted id="
                    f"{state['current_id']} already gone on "
                    f"{entry}/frame{int(frame):05d}"
                )
                return
            self.session.mark_dirty()
            self._update_title()
            self._load_entry_into_viewer(entry, preserve_view=True)
            # Reselect the source peak so the visual state matches
            # what the user had before Add-to-fitted.
            if source_kind == "manual":
                manuals = self.viewer._manual_peaks.get(int(frame), [])
                if manuals:
                    self.viewer._set_selected(
                        SelectedPeak.from_manual(manuals[0], int(frame))
                    )
            elif source_kind == "detected" and source_peak_id is not None:
                tables = self.viewer._frame_peaks.get(int(frame)) or {}
                det = tables.get("detected")
                if det is not None:
                    for i in range(len(det)):
                        if int(det.ids[i]) == source_peak_id:
                            self._select_table_row(
                                int(frame), "detected", det, i
                            )
                            break
            self.pipeline_panel.append_log(
                f"Undo Add-to-fitted: removed fitted id="
                f"{state['current_id']} on "
                f"{entry}/frame{int(frame):05d}"
            )

        def _redo() -> None:
            if self.session is None or self._pipe_thread is not None:
                return
            if self.entry_combo.currentText() != entry:
                self.entry_combo.setCurrentText(entry)
            if int(self.viewer.current_frame) != int(frame):
                self.viewer.set_frame(int(frame))
            with self._detached_silx_tree():
                try:
                    readded_id = file_model.add_fitted_peak_row(
                        self.session.temp_path, entry, int(frame),
                        **add_kwargs,
                    )
                except Exception as exc:
                    QMessageBox.critical(
                        self, "Redo Add to fitted", str(exc)
                    )
                    return
            state["current_id"] = int(readded_id)
            self.session.mark_dirty()
            self._update_title()
            self._load_entry_into_viewer(entry, preserve_view=True)
            # Re-select the re-added fitted peak if the user is
            # currently in 2D mode (mirrors the original commit's
            # auto-select). In 1D mode leave selection alone, also
            # mirroring the original.
            if self.parameter_panel.fit_mode() == ParameterPanel.FIT_MODE_2D:
                # Same ``preserve_manual=True`` rationale as the
                # original commit's auto-switch: keep the manual
                # source intact across the redo so a subsequent
                # Ctrl+Z still has the manual peak to reselect.
                self.viewer._set_selected(
                    SelectedPeak(
                        kind="fitted", frame=int(frame),
                        peak_id=int(readded_id),
                        radius=float(add_kwargs["radius"]),
                        angle=float(add_kwargs["angle"]),
                        radius_width=float(add_kwargs["radius_width"]),
                        angle_width=float(add_kwargs["angle_width"]),
                        is_ring=bool(add_kwargs["is_ring"]),
                        amplitude=(
                            float(add_kwargs["amplitude"])
                            if add_kwargs.get("amplitude") is not None
                            else None
                        ),
                    ),
                    preserve_manual=True,
                )
            self.pipeline_panel.append_log(
                f"Redo Add-to-fitted: re-added fitted id="
                f"{readded_id} on {entry}/frame{int(frame):05d}"
            )

        self.viewer._push_undo(_CallbackAction(_undo, _redo))

    # -- Copy / paste detected peaks ---------------------------------

    def _on_copy_peaks(self) -> None:
        """Snapshot the selected detected peaks into the in-app clipboard.

        Filters the multi-selection to ``kind="detected"`` — non-detected
        kinds are deferred per user scope. No-op + status-bar hint when
        nothing copyable is selected. The clipboard remembers the source
        entry so a same-entry paste check is enforced at paste time.
        """
        if self.session is None:
            return
        sels = [s for s in self.viewer.selected_peaks() if s.kind == "detected"]
        if not sels:
            self.statusBar().showMessage(
                "Copy: select one or more detected peaks first", 4000
            )
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        items = [
            peak_clipboard.ClipboardItem(
                radius=float(s.radius),
                angle=float(s.angle),
                radius_width=float(s.radius_width),
                angle_width=float(s.angle_width),
                is_ring=bool(s.is_ring),
                source_frame=int(s.frame),
                source_peak_id=int(s.peak_id) if s.peak_id is not None else -1,
                score=float(s.score) if s.score is not None else 0.0,
            )
            for s in sels
        ]
        peak_clipboard.set_items(items, entry=entry)
        self.statusBar().showMessage(
            f"Copied {len(items)} detected peak(s)", 4000
        )

    def _on_paste_peaks(self) -> None:
        """Paste clipboard contents onto the current frame as detected rows.

        Same-entry only: ``peak_clipboard.take_items`` returns ``[]`` for
        an entry mismatch. Writes synchronously via
        ``file_model.add_detected_peak_row`` so the new ids are known
        immediately and we can multi-select the pasted peaks after the
        reload. Pushes one grouped ``_CallbackAction`` so a single
        Ctrl+Z reverses the whole paste.
        """
        if self.session is None or self._pipe_thread is not None:
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        items = peak_clipboard.take_items(target_entry=entry)
        if not items:
            self.statusBar().showMessage(
                "Paste: clipboard empty or copied from a different entry",
                4000,
            )
            return
        frame = int(self.viewer.current_frame)
        added_ids: list[int] = []
        with self._detached_silx_tree():
            try:
                for item in items:
                    new_id = file_model.add_detected_peak_row(
                        self.session.temp_path, entry, frame,
                        radius=item.radius,
                        angle=item.angle,
                        radius_width=item.radius_width,
                        angle_width=item.angle_width,
                        is_ring=item.is_ring,
                        score=item.score,
                    )
                    added_ids.append(int(new_id))
            except KeyError as exc:
                QMessageBox.warning(self, "Paste detected peaks", str(exc))
                return
            except Exception as exc:
                QMessageBox.critical(self, "Paste detected peaks", str(exc))
                return
        if not added_ids:
            return
        self.session.mark_dirty()
        self._update_title()
        # Paste only appends rows (fresh ids); no existing id is
        # invalidated, so prior undo history is preserved.
        self._load_entry_into_viewer(entry, preserve_view=True)
        # Add the pasted rows to the existing multi-selection so the
        # user immediately sees what landed without losing whatever
        # they had selected before. If the prior primary wasn't a
        # detected peak (or there was no selection), the first pasted
        # row becomes the new primary.
        self._add_detected_to_selection(entry, frame, added_ids)
        self._push_paste_undo(
            entry=entry, frame=frame, ids=added_ids, items=items,
        )
        self.statusBar().showMessage(
            f"Pasted {len(added_ids)} detected peak(s) on "
            f"{entry}/frame{frame:05d}",
            4000,
        )
        self.pipeline_panel.append_log(
            f"Pasted {len(added_ids)} detected peak(s) on "
            f"{entry}/frame{frame:05d}"
        )

    def _build_detected_sels(
        self, frame: int, ids: list[int], kind: str = "detected",
    ) -> list[SelectedPeak]:
        """Look up the freshly-loaded ``kind`` rows with the given ids
        on ``frame`` and return them as ``SelectedPeak`` snapshots.

        ``kind`` is ``"detected"`` (default) or ``"fitted"``. Silently
        skips ids that aren't present in the table (e.g. after a
        follow-on pipeline rerun). Used by the paste / bulk-delete
        handlers and their undo/redo paths to (re)build a selection."""
        if not ids:
            return []
        tables = self.viewer._frame_peaks.get(int(frame)) or {}
        det = tables.get(kind)
        if det is None:
            return []
        id_set = {int(x) for x in ids}
        sels: list[SelectedPeak] = []
        for i in range(len(det)):
            if int(det.ids[i]) in id_set:
                sels.append(SelectedPeak(
                    kind=kind, frame=int(frame),
                    peak_id=int(det.ids[i]),
                    radius=float(det.radius[i]),
                    angle=float(det.angle[i]),
                    radius_width=float(det.radius_width[i]),
                    angle_width=float(det.angle_width[i]),
                    is_ring=bool(det.is_ring[i]),
                    score=float(det.score[i]),
                    amplitude=float(det.amplitude[i]),
                ))
        return sels

    def _add_detected_to_selection(
        self, entry: str, frame: int, ids: list[int], kind: str = "detected",
    ) -> None:
        """Add the given ``kind`` rows to the existing multi-selection.

        ``kind`` is ``"detected"`` (default) or ``"fitted"``. Behaviour
        mirrors a sequence of Ctrl+clicks: if the current primary is the
        same kind, all ``ids`` are appended to ``_selected_extras``;
        otherwise the first id becomes the new primary (so the
        multi-selection can grow) and the rest go to extras. Lets
        paste / delete-undo grow the user's selection instead of
        wiping it.
        """
        sels = self._build_detected_sels(frame, ids, kind=kind)
        if not sels:
            return
        v = self.viewer
        if v._selected is None or v._selected.kind != kind:
            # ``preserve_manual`` so a previously-selected manual
            # peak isn't auto-removed when we promote a peak to
            # primary (matches the post-Add-to-fitted auto-select
            # pattern at ``_on_add_to_fitted``).
            v._set_selected(sels[0], preserve_manual=True)
            v._selected_extras = list(sels[1:])
        else:
            # Avoid double-adding if any id is already in the
            # selection.
            existing = {
                (s.kind, int(s.frame), int(s.peak_id))
                for s in v.selected_peaks()
            }
            for s in sels:
                key = (s.kind, int(s.frame), int(s.peak_id))
                if key not in existing:
                    v._selected_extras.append(s)
        v._sync_roi()
        v._render_overlays(v.current_frame)
        v.selectionChanged.emit(v._selected)
        v.selectionsChanged.emit(v.selected_peaks())

    def _drop_detected_from_selection(
        self, frame: int, ids: list[int], kind: str = "detected",
    ) -> None:
        """Remove the given ``kind`` rows from the multi-selection.

        ``kind`` is ``"detected"`` (default) or ``"fitted"``. Used by
        the undo/redo paths so the highlight overlay tracks the on-disk
        state — without this, just-deleted rows would keep painting
        white selection boxes at their old positions (the SelectedPeak
        holds its own geometry independent of the peak table).
        """
        v = self.viewer
        if v._selected is None and not v._selected_extras:
            return
        id_set = {(int(frame), int(i)) for i in ids}

        def _hit(s: SelectedPeak) -> bool:
            return (
                s.kind == kind
                and (int(s.frame), int(s.peak_id)) in id_set
            )

        # Filter extras first, then handle the primary.
        v._selected_extras = [
            s for s in v._selected_extras if not _hit(s)
        ]
        if v._selected is not None and _hit(v._selected):
            if v._selected_extras:
                v._selected = v._selected_extras.pop(0)
            else:
                v._selected = None
        v._sync_roi()
        v._render_overlays(v.current_frame)
        v.selectionChanged.emit(v._selected)
        v.selectionsChanged.emit(v.selected_peaks())

    def _push_paste_undo(
        self, *, entry: str, frame: int,
        ids: list[int], items: list,
    ) -> None:
        """Undo / redo for a paste batch — one grouped action.

        ``state["ids"]`` is the rotating list of fitted ids; each redo
        re-adds rows and overwrites it with the freshly-assigned ids
        so a follow-on undo deletes the right rows. Same pattern as
        ``_push_fitted_add_undo`` extended for N rows.
        """
        state = {"ids": list(ids)}
        snapshot = list(items)

        def _undo() -> None:
            if self.session is None or self._pipe_thread is not None:
                return
            if self.entry_combo.currentText() != entry:
                self.entry_combo.setCurrentText(entry)
            if int(self.viewer.current_frame) != int(frame):
                self.viewer.set_frame(int(frame))
            with self._detached_silx_tree():
                try:
                    for peak_id in state["ids"]:
                        file_model.delete_peak_row(
                            self.session.temp_path, entry,
                            frame=int(frame), kind="detected",
                            peak_id=int(peak_id),
                        )
                except Exception as exc:
                    QMessageBox.critical(
                        self, "Undo paste", str(exc)
                    )
                    return
            self.session.mark_dirty()
            self._update_title()
            self._load_entry_into_viewer(entry, preserve_view=True)
            # Drop the now-deleted rows from the multi-selection so
            # the white highlight overlay tracks the on-disk state.
            self._drop_detected_from_selection(int(frame), state["ids"])
            self.pipeline_panel.append_log(
                f"Undo paste: removed {len(state['ids'])} detected "
                f"peak(s) on {entry}/frame{int(frame):05d}"
            )

        def _redo() -> None:
            if self.session is None or self._pipe_thread is not None:
                return
            if self.entry_combo.currentText() != entry:
                self.entry_combo.setCurrentText(entry)
            if int(self.viewer.current_frame) != int(frame):
                self.viewer.set_frame(int(frame))
            new_ids: list[int] = []
            with self._detached_silx_tree():
                try:
                    for item in snapshot:
                        new_id = file_model.add_detected_peak_row(
                            self.session.temp_path, entry, int(frame),
                            radius=item.radius,
                            angle=item.angle,
                            radius_width=item.radius_width,
                            angle_width=item.angle_width,
                            is_ring=item.is_ring,
                            score=item.score,
                        )
                        new_ids.append(int(new_id))
                except Exception as exc:
                    QMessageBox.critical(
                        self, "Redo paste", str(exc)
                    )
                    return
            state["ids"] = new_ids
            self.session.mark_dirty()
            self._update_title()
            self._load_entry_into_viewer(entry, preserve_view=True)
            # Re-add to the existing selection (additive, matching
            # the original paste).
            self._add_detected_to_selection(entry, int(frame), new_ids)
            self.pipeline_panel.append_log(
                f"Redo paste: re-added {len(new_ids)} detected "
                f"peak(s) on {entry}/frame{int(frame):05d}"
            )

        self.viewer._push_undo(_CallbackAction(_undo, _redo))

    def _on_paste_peaks_to_range(self) -> None:
        """Paste the clipboard onto every frame in a typed range.

        Ctrl+Shift+V entry point. Mirrors ``_on_paste_peaks`` but
        loops the frame set produced by ``frame_range.parse_frame_range``.
        ``take_items`` is non-draining, so the same snapshot is replayed
        on every target frame and remains available for further pastes.

        UX rules (confirmed with the user):
          * Out-of-range frames are filtered with a confirmation dialog
            naming the count + compact range list.
          * Duplicates in the input deduplicate silently.
          * After a successful paste, only the current frame's rows
            (if it landed in the range) get added to the multi-selection.
          * Cancel mid-loop keeps already-written frames.
          * One grouped ``_CallbackAction`` reverses the entire batch.
        """
        if self.session is None or self._pipe_thread is not None:
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        items = peak_clipboard.take_items(target_entry=entry)
        if not items:
            self.statusBar().showMessage(
                "Paste: clipboard empty or copied from a different entry",
                4000,
            )
            return
        n_frames = int(self.viewer.n_frames)
        if n_frames <= 0:
            return
        text, ok = QInputDialog.getText(
            self, "Paste to frames",
            f"Frames (e.g. 0-{max(n_frames - 1, 0)},37):",
            text=str(int(self.viewer.current_frame)),
        )
        if not ok:
            return
        try:
            valid, dropped = frame_range.parse_frame_range(
                text, n_frames=n_frames,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Paste to frames", str(exc))
            return
        if dropped:
            reply = QMessageBox.question(
                self, "Paste to frames",
                f"Pasting to {len(valid)} frame(s). "
                f"{len(dropped)} frame(s) outside the {n_frames}-frame "
                f"stack will be skipped: "
                f"{frame_range.compact_repr(dropped)}.\n\nContinue?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        if not valid:
            self.statusBar().showMessage(
                "Paste to frames: nothing to do — no frames in range",
                4000,
            )
            return

        n_frames_to_write = len(valid)
        dialog = QProgressDialog(
            f"Pasting to {n_frames_to_write} frame(s)…",
            "Cancel", 0, n_frames_to_write, self,
        )
        dialog.setWindowTitle("Paste to frames")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setValue(0)

        per_frame_ids: dict[int, list[int]] = {}
        cancelled = False
        with self._detached_silx_tree():
            try:
                for i, target_frame in enumerate(valid):
                    if dialog.wasCanceled():
                        cancelled = True
                        break
                    dialog.setLabelText(
                        f"Pasting frame {i + 1}/{n_frames_to_write} "
                        f"(frame {target_frame:05d})…"
                    )
                    QApplication.processEvents()
                    frame_ids: list[int] = []
                    for item in items:
                        new_id = file_model.add_detected_peak_row(
                            self.session.temp_path, entry,
                            int(target_frame),
                            radius=item.radius,
                            angle=item.angle,
                            radius_width=item.radius_width,
                            angle_width=item.angle_width,
                            is_ring=item.is_ring,
                            score=item.score,
                        )
                        frame_ids.append(int(new_id))
                    per_frame_ids[int(target_frame)] = frame_ids
                    dialog.setValue(i + 1)
                    QApplication.processEvents()
            except KeyError as exc:
                dialog.close()
                QMessageBox.warning(self, "Paste to frames", str(exc))
                return
            except Exception as exc:
                dialog.close()
                QMessageBox.critical(self, "Paste to frames", str(exc))
                return
        dialog.close()

        if not per_frame_ids:
            return

        self.session.mark_dirty()
        self._update_title()
        # Append-only across the range (fresh ids per frame); prior undo
        # history is preserved.
        self._load_entry_into_viewer(entry, preserve_view=True)
        # Only the current frame's pasted rows get added to the
        # selection (per confirmed UX). Off-frame selections would
        # be pruned on frame switch anyway.
        current_frame = int(self.viewer.current_frame)
        if current_frame in per_frame_ids:
            self._add_detected_to_selection(
                entry, current_frame, per_frame_ids[current_frame],
            )
        self._push_paste_to_range_undo(
            entry=entry, items=items, per_frame_ids=per_frame_ids,
        )

        n_rows = sum(len(v) for v in per_frame_ids.values())
        n_frames_written = len(per_frame_ids)
        summary = (
            f"Pasted {n_rows} detected peak(s) across "
            f"{n_frames_written} frame(s) on {entry}"
        )
        if dropped:
            summary += f" ({len(dropped)} out-of-range frame(s) skipped)"
        if cancelled:
            summary += " (cancelled)"
        self.statusBar().showMessage(summary, 6000)
        self.pipeline_panel.append_log(summary)

    def _push_paste_to_range_undo(
        self, *, entry: str, items: list,
        per_frame_ids: dict[int, list[int]],
    ) -> None:
        """Undo / redo for a range paste — one grouped action.

        ``state["per_frame_ids"]`` rotates on every redo so a follow-on
        undo deletes the correct (freshly-assigned) ids. Same idea as
        ``_push_paste_undo`` extended from a single frame to a frame
        dict; frames the original Cancel skipped never enter the dict
        so they're left alone on undo/redo.
        """
        state: dict[str, dict[int, list[int]]] = {
            "per_frame_ids": {
                int(f): [int(i) for i in ids]
                for f, ids in per_frame_ids.items()
            },
        }
        snapshot = list(items)

        def _undo() -> None:
            if self.session is None or self._pipe_thread is not None:
                return
            if self.entry_combo.currentText() != entry:
                self.entry_combo.setCurrentText(entry)
            current_frame = int(self.viewer.current_frame)
            with self._detached_silx_tree():
                try:
                    for f, ids in state["per_frame_ids"].items():
                        for peak_id in ids:
                            file_model.delete_peak_row(
                                self.session.temp_path, entry,
                                frame=int(f), kind="detected",
                                peak_id=int(peak_id),
                            )
                except Exception as exc:
                    QMessageBox.critical(
                        self, "Undo paste to frames", str(exc)
                    )
                    return
            self.session.mark_dirty()
            self._update_title()
            self._load_entry_into_viewer(entry, preserve_view=True)
            if current_frame in state["per_frame_ids"]:
                self._drop_detected_from_selection(
                    current_frame, state["per_frame_ids"][current_frame],
                )
            n_rows = sum(len(v) for v in state["per_frame_ids"].values())
            n_frames_written = len(state["per_frame_ids"])
            self.pipeline_panel.append_log(
                f"Undo paste to frames: removed {n_rows} detected "
                f"peak(s) across {n_frames_written} frame(s) on {entry}"
            )

        def _redo() -> None:
            if self.session is None or self._pipe_thread is not None:
                return
            if self.entry_combo.currentText() != entry:
                self.entry_combo.setCurrentText(entry)
            current_frame = int(self.viewer.current_frame)
            new_per_frame_ids: dict[int, list[int]] = {}
            with self._detached_silx_tree():
                try:
                    for f in sorted(state["per_frame_ids"].keys()):
                        frame_ids: list[int] = []
                        for item in snapshot:
                            new_id = file_model.add_detected_peak_row(
                                self.session.temp_path, entry, int(f),
                                radius=item.radius,
                                angle=item.angle,
                                radius_width=item.radius_width,
                                angle_width=item.angle_width,
                                is_ring=item.is_ring,
                                score=item.score,
                            )
                            frame_ids.append(int(new_id))
                        new_per_frame_ids[int(f)] = frame_ids
                except Exception as exc:
                    QMessageBox.critical(
                        self, "Redo paste to frames", str(exc)
                    )
                    return
            state["per_frame_ids"] = new_per_frame_ids
            self.session.mark_dirty()
            self._update_title()
            self._load_entry_into_viewer(entry, preserve_view=True)
            if current_frame in new_per_frame_ids:
                self._add_detected_to_selection(
                    entry, current_frame, new_per_frame_ids[current_frame],
                )
            n_rows = sum(len(v) for v in new_per_frame_ids.values())
            n_frames_written = len(new_per_frame_ids)
            self.pipeline_panel.append_log(
                f"Redo paste to frames: re-added {n_rows} detected "
                f"peak(s) across {n_frames_written} frame(s) on {entry}"
            )

        self.viewer._push_undo(_CallbackAction(_undo, _redo))

    # -- Batch 2D fit ------------------------------------------------

    def _refresh_fit_buttons(self) -> None:
        """Re-evaluate visibility + enable state for the fit buttons.

        Visibility rule (per user request — only show the button that
        makes sense, don't grey them out):

        * Exactly one fittable peak selected (manual or detected) →
          'Add to fitted' visible, 'Fit selected (2D)' hidden.
        * Two or more detected peaks selected → 'Fit selected (2D)'
          visible, 'Add to fitted' hidden.
        * Otherwise (no selection, only fitted / matched, or the
          mixed-kind edge case of 1 detected + 1 manual which neither
          button handles cleanly) → both hidden.

        'Add to detected' stays in place — its enable state is driven
        by ``ParameterPanel._update_actions_enabled`` (greyed out for
        non-manual selections).

        Enable state for 'Fit selected (2D)' is still gated on
        save-as-ring being OFF (ring forces 1D).
        """
        sels = self.viewer.selected_peaks()
        n_fittable = sum(1 for s in sels if s.kind in ("manual", "detected"))
        n_detected = sum(1 for s in sels if s.kind == "detected")
        ring_on = self.parameter_panel.save_as_ring()
        show_add_fitted = (n_fittable == 1)
        show_batch_fit = (n_detected >= 2 and not ring_on)
        self.parameter_panel.set_fit_button_visibility(
            add_fitted=show_add_fitted,
            fit_selected_2d=show_batch_fit,
        )
        # Keep the enable state coherent so a click on the visible
        # batch-fit button always proceeds (otherwise a stale enable=False
        # from before the visibility change could leak through).
        self.parameter_panel.set_batch_fit_enabled(show_batch_fit)
        # ≥2 fittable selected → batch fit is the only sensible
        # action, and batch fit is 2D-only. Grey out the 1D radio
        # to make that constraint visible.
        self.parameter_panel.set_multi_select_active(n_fittable >= 2)

    def _on_batch_fit_2d(self) -> None:
        """Run pygidfit on every selected detected peak, in one batch.

        Two-phase to keep the viewer's FrameSource open while pygidfit
        reads frame data:

        1. **Fit phase** (no silx detach): loop the selections,
           calling ``_run_pygidfit_for_selection`` which needs
           ``viewer._frame_source.is_open``. The QProgressDialog
           ticks per fit and respects Cancel.
        2. **Write phase** (single ``_detached_silx_tree`` scope):
           append a ``fitted_peaks`` row for every successful fit.
           h5py detach is expensive; doing it once for the whole
           batch keeps the run snappy.
        """
        if self.session is None or self._pipe_thread is not None:
            return
        sels = [
            s for s in self.viewer.selected_peaks() if s.kind == "detected"
        ]
        if not sels:
            self.statusBar().showMessage(
                "Fit selected (2D): select detected peaks first", 4000
            )
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        n = len(sels)
        dialog = QProgressDialog(
            f"Fitting {n} peak(s) (2D)…", "Cancel", 0, n, self,
        )
        dialog.setWindowTitle("Fit selected (2D)")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setValue(0)

        # Phase 1: fits. Each ``_run_pygidfit_for_selection`` call
        # reads ``viewer._frame_source.get_cartesian(frame)``, so the
        # silx tree must stay attached here (the detach in phase 2
        # closes the frame source).
        fit_results: list[tuple[SelectedPeak, object, str | None]] = []
        cancelled = False
        for i, sel in enumerate(sels):
            if dialog.wasCanceled():
                cancelled = True
                break
            dialog.setLabelText(
                f"Fitting peak {i + 1}/{n} (id={sel.peak_id})…"
            )
            QApplication.processEvents()
            fit_2d, err = self._run_pygidfit_for_selection(
                sel, entry, int(sel.frame),
            )
            fit_results.append((sel, fit_2d, err))
            dialog.setValue(i + 1)
            QApplication.processEvents()

        # Phase 2: writes. One ``_detached_silx_tree`` scope for the
        # whole batch — h5py detach is the expensive bit.
        added_ids: list[int] = []
        added_kwargs: list[dict] = []
        failures: list[tuple[int, str]] = []
        with self._detached_silx_tree():
            for sel, fit_2d, err in fit_results:
                if fit_2d is None:
                    failures.append((int(sel.peak_id), err or "fit failed"))
                    continue
                row = {
                    "radius": float(fit_2d.radius),
                    "radius_width": float(fit_2d.radius_width),
                    "angle": float(fit_2d.angle),
                    "angle_width": float(fit_2d.angle_width),
                    "amplitude": float(fit_2d.amplitude),
                    "is_ring": False,
                    "theta": float(fit_2d.theta),
                    "A": float(fit_2d.A),
                    "B": float(fit_2d.B),
                    "C": float(fit_2d.C),
                }
                try:
                    new_id = file_model.add_fitted_peak_row(
                        self.session.temp_path, entry, int(sel.frame),
                        **row,
                    )
                except Exception as exc:
                    failures.append((int(sel.peak_id), f"write failed: {exc}"))
                    continue
                added_ids.append(int(new_id))
                added_kwargs.append({**row, "frame": int(sel.frame)})
        dialog.close()

        if not added_ids:
            QMessageBox.warning(
                self, "Fit selected (2D)",
                "No peaks were fitted successfully.\n\n"
                + "\n".join(f"id={pid}: {msg}" for pid, msg in failures),
            )
            return

        self.session.mark_dirty()
        self._update_title()
        # Batch fit only appends fitted rows (fresh ids); the detected
        # ids it fit from are untouched, so prior undo history stays valid.
        self._load_entry_into_viewer(entry, preserve_view=True)
        # Multi-select the newly-added fitted rows. They may span
        # multiple frames if the original detected selection did
        # (rare — multi-select today is per-frame — but kept correct
        # for future when cross-frame multi-select lands).
        frames_touched = sorted({kw["frame"] for kw in added_kwargs})
        if frames_touched:
            # Select on the first touched frame; the user can navigate.
            self._select_fitted_ids(
                entry, frames_touched[0],
                [
                    new_id for new_id, kw in zip(added_ids, added_kwargs)
                    if kw["frame"] == frames_touched[0]
                ],
            )
        self._push_batch_fit_undo(
            entry=entry, added_ids=added_ids, added_kwargs=added_kwargs,
        )

        summary = (
            f"Fit {len(added_ids)}/{n} selected peaks "
            f"({len(failures)} failed"
            + (", cancelled" if cancelled else "")
            + ")"
        )
        self.statusBar().showMessage(summary, 6000)
        self.pipeline_panel.append_log(summary)
        for pid, msg in failures:
            self.pipeline_panel.append_log(f"  fit failed: id={pid} — {msg}")

    def _select_fitted_ids(
        self, entry: str, frame: int, ids: list[int],
    ) -> None:
        """Multi-select the fitted rows with the given ids on ``frame``."""
        if not ids:
            return
        tables = self.viewer._frame_peaks.get(int(frame)) or {}
        fit = tables.get("fitted")
        if fit is None:
            return
        id_set = {int(x) for x in ids}
        sels = []
        for i in range(len(fit)):
            if int(fit.ids[i]) in id_set:
                sels.append(SelectedPeak(
                    kind="fitted", frame=int(frame),
                    peak_id=int(fit.ids[i]),
                    radius=float(fit.radius[i]),
                    angle=float(fit.angle[i]),
                    radius_width=float(fit.radius_width[i]),
                    angle_width=float(fit.angle_width[i]),
                    is_ring=bool(fit.is_ring[i]),
                    score=float(fit.score[i]),
                    amplitude=float(fit.amplitude[i]),
                ))
        if not sels:
            return
        # Fitted is not currently in the extras allowlist (extras are
        # detected-only this iteration), so set the primary alone here.
        # Multi-selection of fitted peaks lands when fitted copy/paste
        # is enabled in a follow-up.
        self.viewer._set_selected(sels[0], preserve_manual=True)

    def _push_batch_fit_undo(
        self, *, entry: str, added_ids: list[int], added_kwargs: list[dict],
    ) -> None:
        """Undo / redo for a batch-fit run — one grouped action.

        Mirrors ``_push_fitted_add_undo`` extended to N rows. The
        rotating ``state["ids"]`` keeps undo correct after a redo
        re-assigns ids (one per redo'd row).
        """
        state = {"ids": list(added_ids)}
        snapshot_kwargs = [dict(kw) for kw in added_kwargs]

        def _undo() -> None:
            if self.session is None or self._pipe_thread is not None:
                return
            if self.entry_combo.currentText() != entry:
                self.entry_combo.setCurrentText(entry)
            with self._detached_silx_tree():
                try:
                    for kw, peak_id in zip(snapshot_kwargs, state["ids"]):
                        file_model.delete_peak_row(
                            self.session.temp_path, entry,
                            frame=int(kw["frame"]), kind="fitted",
                            peak_id=int(peak_id),
                        )
                    # Cascading clear on every touched frame mirrors
                    # the single-fitted-delete behaviour.
                    for f in sorted({int(kw["frame"]) for kw in snapshot_kwargs}):
                        file_model.clear_peaks(
                            self.session.temp_path, entry,
                            kind="matched", frame=f,
                        )
                except Exception as exc:
                    QMessageBox.critical(
                        self, "Undo batch fit", str(exc)
                    )
                    return
            self.session.mark_dirty()
            self._update_title()
            self._load_entry_into_viewer(entry, preserve_view=True)
            self.pipeline_panel.append_log(
                f"Undo batch fit: removed {len(state['ids'])} fitted "
                f"row(s) on {entry}"
            )

        def _redo() -> None:
            if self.session is None or self._pipe_thread is not None:
                return
            if self.entry_combo.currentText() != entry:
                self.entry_combo.setCurrentText(entry)
            new_ids: list[int] = []
            with self._detached_silx_tree():
                try:
                    for kw in snapshot_kwargs:
                        write_kw = {k: v for k, v in kw.items() if k != "frame"}
                        new_id = file_model.add_fitted_peak_row(
                            self.session.temp_path, entry, int(kw["frame"]),
                            **write_kw,
                        )
                        new_ids.append(int(new_id))
                except Exception as exc:
                    QMessageBox.critical(
                        self, "Redo batch fit", str(exc)
                    )
                    return
            state["ids"] = new_ids
            self.session.mark_dirty()
            self._update_title()
            self._load_entry_into_viewer(entry, preserve_view=True)
            self.pipeline_panel.append_log(
                f"Redo batch fit: re-added {len(new_ids)} fitted "
                f"row(s) on {entry}"
            )

        self.viewer._push_undo(_CallbackAction(_undo, _redo))

    def _run_pygidfit_for_selection(
        self, sel: SelectedPeak, entry: str, frame: int,
    ) -> tuple["ManualFitResult | None", str | None]:
        """Run pygidfit on ``sel`` and return ``(result, error)``.

        Silent — no QMessageBox, no logging at warning level. Used by
        both the commit path (``_build_fitted_row_2d``) and the live
        preview path (``_refresh_2d_preview``). When ``result`` is
        None, ``error`` is a short human-readable string the caller
        can surface or ignore.

        Replicates ``pygidfit.ProcessDataFromFile.process_single_frame``
        — same polar resolution (``512 × 1024``), ``img_preprocessing``
        masking, ``crit_angle`` / ``theta_fixed`` / ``clustering_*``
        values pulled from the Pipeline panel — so the preview's box
        equals what the commit would save.
        """
        from mlgidlab.manual_fit import ManualFitError, fit_one_peak

        fs = getattr(self.viewer, "_frame_source", None)
        stack = getattr(self.viewer, "_stack", None)
        if fs is None or not fs.is_open or stack is None:
            return None, (
                "The active frame isn't currently available "
                "(viewer mid-pipeline or released)."
            )
        try:
            cartesian = np.asarray(fs.get_cartesian(int(frame)))
        except Exception as exc:
            return None, f"Could not load the cartesian frame: {exc}"
        q_xy = np.asarray(stack.q_xy)
        q_z = np.asarray(stack.q_z)

        if self.session is None:
            return None, "No active session."
        geom = file_model.read_geometry_for_entry(
            self.session.temp_path, entry, frame=int(frame),
        )
        if geom is None:
            return None, (
                "Instrument metadata (wavelength / angle of incidence "
                "/ q ranges) is missing or malformed for this entry."
            )
        geom = dict(geom)
        geom.pop("q_z_axis", None)

        panel = self.pipeline_panel
        try:
            crit_angle = float(panel.fit_crit_angle.value())
            cdp = float(panel.fit_dist_peaks.value())
            cdr = float(panel.fit_dist_rings.value())
            ce = int(panel.fit_cluster_extend.value())
            tf = bool(panel.fit_theta_fixed.isChecked())
        except Exception:
            crit_angle, cdp, cdr, ce, tf = 0.0, 10.0, 10.0, 2, True

        try:
            fit_2d = fit_one_peak(
                cartesian, q_xy, q_z,
                radius=float(sel.radius),
                radius_width=float(sel.radius_width),
                angle=float(sel.angle),
                angle_width=float(sel.angle_width),
                crit_angle=crit_angle,
                theta_fixed=tf,
                clustering_distance_peaks=cdp,
                clustering_distance_rings=cdr,
                clustering_extend=ce,
                **geom,
            )
        except ManualFitError as exc:
            return None, str(exc)
        except Exception as exc:
            return None, f"Unexpected 2D fit error: {exc!r}"
        return fit_2d, None

    def _build_fitted_row_2d(
        self, sel: SelectedPeak, entry: str, frame: int,
    ) -> dict | None:
        """Run pygidfit on ``sel`` and shape the result into an
        ``add_fitted_peak_row`` kwarg dict. Surfaces a QMessageBox
        on failure — strict, no silent fall-back to the 1D path.
        Delegates the actual fit to ``_run_pygidfit_for_selection``
        so the live-preview path can reuse the same code.
        """
        fit_2d, err = self._run_pygidfit_for_selection(sel, entry, frame)
        if fit_2d is None:
            QMessageBox.warning(
                self, "Add to fitted (2D)",
                f"pygidfit could not fit this box: {err}\n\n"
                "Try widening the box, or switch to '1D fit (scipy)' "
                "mode to commit with the legacy 1D Gaussian fit.",
            )
            return None
        return {
            "radius": fit_2d.radius,
            "radius_width": fit_2d.radius_width,
            "angle": fit_2d.angle,
            "angle_width": fit_2d.angle_width,
            "amplitude": fit_2d.amplitude,
            "theta": fit_2d.theta,
            "A": fit_2d.A, "B": fit_2d.B, "C": fit_2d.C,
        }

    def _build_fitted_row_1d(
        self, sel: SelectedPeak, save_as_ring: bool,
    ) -> dict | None:
        """Build the row from the profile viewer's cached 1D scipy fits.

        Width convention matches the 2D path so the saved blue box for
        the same physical Gaussian renders identically regardless of
        which mode produced it: ``radius_width = angle_width = 2σ =
        FWHM / sqrt(2 ln 2) ≈ 0.849 × FWHM``. This is pygidfit's
        convention (see ``manual_fit.fit_one_peak``) and what the
        pipeline ``run_fitting`` writes. The only difference between
        modes is whether 2D shape coefficients (A/B/C/theta) carry
        real values — 1D zero-fills them.

        Ring storage keeps ``angle = 45°``, ``angle_width = inf`` as
        the sentinel — not a Gaussian width, so the unified
        convention doesn't apply.

        Returns ``None`` and shows a QMessageBox when the required 1D
        fit isn't available (typical cause: narrow detected box where
        scipy didn't converge). Strict — no fall-back to pygidfit.
        """
        fits = self.profile_viewer.last_fit_params()
        rfit = fits.get("radial")
        afit = fits.get("angular")
        if rfit is None:
            QMessageBox.warning(
                self, "Add to fitted (1D)",
                "No radial Gaussian fit is available. Drag the box "
                "until the pink fit curve appears in the radial "
                "profile, or switch to '2D fit (pygidfit)' mode.",
            )
            return None
        if not save_as_ring and afit is None:
            QMessageBox.warning(
                self, "Add to fitted (1D)",
                "No angular Gaussian fit is available — required for "
                "segment peaks in 1D mode. Drag the box until the "
                "pink fit curve appears in the angular profile, "
                "check 'Save fitted as ring' if this is a ring, or "
                "switch to '2D fit (pygidfit)' mode.",
            )
            return None
        fwhm_to_2sigma = 1.0 / float(np.sqrt(2.0 * np.log(2.0)))
        if save_as_ring:
            angle_to_save = 45.0
            angle_width_to_save = float("inf")
        else:
            angle_to_save = float(afit.center)
            angle_width_to_save = float(afit.fwhm) * fwhm_to_2sigma
        return {
            "radius": float(rfit.center),
            "radius_width": float(rfit.fwhm) * fwhm_to_2sigma,
            "angle": angle_to_save,
            "angle_width": angle_width_to_save,
            "amplitude": float(rfit.amplitude),
            "theta": 0.0, "A": 0.0, "B": 0.0, "C": 0.0,
        }

    def _on_delete_peak_requested(self, sel: SelectedPeak | None) -> None:
        """Confirm + delete a non-manual peak — kind-scoped, no cascade.

        Replaces the previous ``mlgidbase.delete_peak`` dispatch
        (which removed the row from detected + fitted + matched in
        one shot, leaving the user unable to delete just the fitted
        prediction without also wiping its detected source). Now:

        * ``sel.kind == "detected"`` → delete only the detected row.
          Leaves fitted / matched alone.
        * ``sel.kind == "fitted"`` → delete only the fitted row, then
          reindex ``matched_*`` ``peak_list`` so each surviving
          structure keeps its other peaks (the deleted peak is dropped
          from any structure that referenced it; structures that didn't
          are untouched). Replaces the old blunt "clear every
          ``matched_*`` on the frame" cascade.
        * ``sel.kind == "matched"`` → still routes through
          ``mlgidbase.delete_peak`` for now — matched live across
          per-solution datasets and a dedicated single-solution
          deleter isn't built yet. Leaves detected / fitted intact
          via mlgidbase's own per-kind missing-id tolerance.
        """
        if (
            sel is None or sel.kind == "manual"
            or self.session is None or self._pipe_thread is not None
        ):
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return

        if sel.kind in ("detected", "fitted"):
            return self._delete_file_peak_scoped(sel, entry)

        # matched fallthrough — keep the cascade path until a
        # dedicated single-matched-row deleter exists.
        if not is_mlgidbase_available():
            QMessageBox.information(
                self, "Delete peak",
                "mlgidbase is not installed; cannot delete peaks.",
            )
            return
        reply = QMessageBox.question(
            self,
            "Delete matched peak",
            (
                f"Delete matched peak id={sel.peak_id} on frame {sel.frame}?\n\n"
                "It will be removed from every matched solution that "
                "references it. Detected and fitted rows are left "
                "intact. This cannot be undone."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        cmd = PipelineCommand(
            "delete_peak",
            {
                "entry": entry,
                "frame_num": int(sel.frame),
                "peak_id": int(sel.peak_id),
            },
        )
        self._on_pipeline_run(self.session.temp_path, cmd)

    def _delete_file_peak_scoped(
        self, sel: SelectedPeak, entry: str,
    ) -> None:
        """Delete one detected/fitted row — kind-scoped, undoable.

        Fitted deletes also reindex ``matched_*`` ``peak_list`` on the
        frame (positions into ``fitted_peaks`` shift when a row is
        removed), keeping the surviving structures. Delegates to the
        shared ``_delete_peaks_scoped`` so single and bulk delete share
        one undoable path.
        """
        self._delete_peaks_scoped([sel], entry)

    def _on_delete_peaks_requested(self, sels: list[SelectedPeak]) -> None:
        """Bulk-delete an all-detected OR all-fitted multi-selection.

        Triggered by Delete when >= 2 same-kind peaks are selected
        (``deletePeaksRequested``). One confirmation, one grouped,
        undoable action — see ``_delete_peaks_scoped``.
        """
        kinds = {s.kind for s in sels}
        if len(sels) < 2 or len(kinds) != 1:
            return
        if next(iter(kinds)) not in ("detected", "fitted"):
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        self._delete_peaks_scoped(list(sels), entry)

    def _delete_peaks_scoped(
        self, sels: list[SelectedPeak], entry: str,
    ) -> None:
        """Shared, undoable delete for 1..N same-kind detected/fitted peaks.

        Confirms once, snapshots the full rows (so undo can re-create
        them verbatim — including the 2D-fit params PeakTable doesn't
        surface), deletes them in one ``_detached_silx_tree`` scope,
        reindexes ``matched_*`` ``peak_list`` on the frame for fitted
        deletes (dropping the deleted peaks from any structure that
        referenced them, keeping every structure), and pushes one
        grouped ``_CallbackAction`` so a single Ctrl+Z restores every
        row. Undo re-adds the fitted rows with fresh ids appended at the
        end; it does **not** re-add them to the matched structures they
        were in (the structures already reference the surviving peaks
        correctly — re-run Matching if a peak's membership matters).
        """
        if self.session is None or self._pipe_thread is not None:
            return
        if not sels or not entry:
            return
        kind = sels[0].kind
        if kind not in ("detected", "fitted"):
            return
        frame = int(self.viewer.current_frame)
        on_frame = [
            s for s in sels if s.kind == kind and int(s.frame) == frame
        ]
        if not on_frame:
            return
        ids = [int(s.peak_id) for s in on_frame]
        n = len(ids)
        kind_human = "detected" if kind == "detected" else "fitted"
        if kind == "detected":
            siblings_msg = (
                "Fitted and matched rows stay intact."
            )
        else:
            siblings_msg = (
                "Detected rows stay intact. Matched structures that "
                "include the deleted peak(s) lose just those peaks; "
                "structures that don't reference them are left unchanged."
            )
        noun = f"{kind_human} peak" + ("s" if n > 1 else "")
        target = (
            f"id={ids[0]} on frame {frame}" if n == 1
            else f"{n} {kind_human} peaks on frame {frame}"
        )
        reply = QMessageBox.question(
            self,
            f"Delete {kind_human} peak" + ("s" if n > 1 else ""),
            f"Delete {target}?\n\n{siblings_msg} This can be undone "
            "with Ctrl+Z.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        removed = 0
        snapshot: list[dict] = []
        matched_changed = 0
        with self._detached_silx_tree():
            try:
                # Snapshot full rows (incl. theta/A/B/C for fitted) so
                # undo can re-create them. Read inside the detached
                # scope so no handle contends with silx.
                snapshot = file_model.read_peak_rows(
                    self.session.temp_path, entry, frame, kind, ids,
                )
                # matched_* peak_list stores positions into fitted_peaks,
                # not ids, and deleting compacts the array — capture the
                # rows' positions BEFORE deleting so matched can be
                # reindexed afterwards.
                deleted_positions = (
                    file_model.fitted_positions_for_ids(
                        self.session.temp_path, entry, frame, ids,
                    )
                    if kind == "fitted" else []
                )
                for peak_id in ids:
                    removed += file_model.delete_peak_row(
                        self.session.temp_path, entry,
                        frame=frame, kind=kind, peak_id=peak_id,
                    )
                if kind == "fitted" and removed > 0:
                    # Drop the deleted peaks from any matched structure
                    # that referenced them and shift the surviving
                    # indices down; structures are kept (the old code
                    # wiped every matched_* on the frame instead).
                    matched_changed = file_model.remap_matched_peak_lists(
                        self.session.temp_path, entry, frame,
                        deleted_positions,
                    )
            except Exception as exc:
                QMessageBox.critical(
                    self, f"Delete {kind_human} peak", str(exc),
                )
                return
        if removed == 0:
            self.pipeline_panel.append_log(
                f"Delete {kind_human}: no matching rows on "
                f"{entry}/frame{frame:05d} (already gone?)"
            )
            return

        self.session.mark_dirty()
        self._update_title()
        # Drop only the deleted ids' pending geom edits (they would key
        # off rows that no longer exist); prior undoable ops stay on the
        # stack so a run of deletes remains fully undoable / redoable.
        self.viewer.prune_file_geom_history(frame, kind, ids)
        self.viewer.clear_selection()
        self._load_entry_into_viewer(entry, preserve_view=True)
        self._push_delete_undo(
            entry=entry, frame=frame, kind=kind, snapshot=snapshot,
            cascade_matched=(kind == "fitted"),
        )
        msg = (
            f"Deleted {removed} {kind_human} peak(s) on "
            f"{entry}/frame{frame:05d}"
            + (
                f" ({matched_changed} matched structure(s) updated)"
                if matched_changed else ""
            )
        )
        self.statusBar().showMessage(msg, 4000)
        self.pipeline_panel.append_log(msg)

    def _readd_peak_row(self, entry: str, frame: int, kind: str, d: dict) -> int:
        """Re-create one detected/fitted row from a ``read_peak_rows``
        snapshot dict. Returns the new (freshly-assigned) id."""
        if kind == "detected":
            return file_model.add_detected_peak_row(
                self.session.temp_path, entry, int(frame),
                radius=d["radius"], angle=d["angle"],
                radius_width=d["radius_width"], angle_width=d["angle_width"],
                score=d.get("score", 0.0), is_ring=d.get("is_ring", False),
            )
        return file_model.add_fitted_peak_row(
            self.session.temp_path, entry, int(frame),
            angle=d["angle"], angle_width=d["angle_width"],
            radius=d["radius"], radius_width=d["radius_width"],
            amplitude=d.get("amplitude", 0.0), is_ring=d.get("is_ring", False),
            theta=d.get("theta", 0.0), score=d.get("score", 0.0),
            A=d.get("A", 0.0), B=d.get("B", 0.0), C=d.get("C", 0.0),
            is_cut_qz=d.get("is_cut_qz", False),
            is_cut_qxy=d.get("is_cut_qxy", False),
            visibility=d.get("visibility", 0),
        )

    def _push_delete_undo(
        self, *, entry: str, frame: int, kind: str,
        snapshot: list[dict], cascade_matched: bool,
    ) -> None:
        """Undo / redo for a (single or bulk) detected/fitted delete.

        The delete already happened, so ``undo`` re-adds the rows (fresh
        ids) and ``redo`` deletes them again. ``state["ids"]`` rotates:
        populated by ``undo`` with the re-assigned ids so a following
        ``redo`` deletes the right rows. For fitted, ``redo`` re-applies
        the ``matched_*`` reindex (recomputing the deleted rows'
        positions from the current ids first, since the re-added rows
        sit at fresh positions); ``undo`` leaves ``matched_*`` as-is —
        see ``_delete_peaks_scoped``.
        """
        snap = [dict(d) for d in snapshot]
        state: dict[str, list[int]] = {"ids": []}

        def _undo() -> None:
            if self.session is None or self._pipe_thread is not None:
                return
            if self.entry_combo.currentText() != entry:
                self.entry_combo.setCurrentText(entry)
            if int(self.viewer.current_frame) != int(frame):
                self.viewer.set_frame(int(frame))
            new_ids: list[int] = []
            with self._detached_silx_tree():
                try:
                    for d in snap:
                        new_ids.append(
                            int(self._readd_peak_row(entry, frame, kind, d))
                        )
                except Exception as exc:
                    QMessageBox.critical(self, "Undo delete", str(exc))
                    return
            state["ids"] = new_ids
            self.session.mark_dirty()
            self._update_title()
            self._load_entry_into_viewer(entry, preserve_view=True)
            self._add_detected_to_selection(
                entry, int(frame), new_ids, kind=kind,
            )
            self.pipeline_panel.append_log(
                f"Undo delete: restored {len(new_ids)} {kind} "
                f"peak(s) on {entry}/frame{int(frame):05d}"
            )

        def _redo() -> None:
            if self.session is None or self._pipe_thread is not None:
                return
            if self.entry_combo.currentText() != entry:
                self.entry_combo.setCurrentText(entry)
            if int(self.viewer.current_frame) != int(frame):
                self.viewer.set_frame(int(frame))
            with self._detached_silx_tree():
                try:
                    deleted_positions = (
                        file_model.fitted_positions_for_ids(
                            self.session.temp_path, entry, int(frame),
                            [int(p) for p in state["ids"]],
                        )
                        if cascade_matched else []
                    )
                    for peak_id in state["ids"]:
                        file_model.delete_peak_row(
                            self.session.temp_path, entry,
                            frame=int(frame), kind=kind,
                            peak_id=int(peak_id),
                        )
                    if cascade_matched and state["ids"]:
                        file_model.remap_matched_peak_lists(
                            self.session.temp_path, entry, int(frame),
                            deleted_positions,
                        )
                except Exception as exc:
                    QMessageBox.critical(self, "Redo delete", str(exc))
                    return
            self.session.mark_dirty()
            self._update_title()
            self._load_entry_into_viewer(entry, preserve_view=True)
            self._drop_detected_from_selection(
                int(frame), state["ids"], kind=kind,
            )
            self.pipeline_panel.append_log(
                f"Redo delete: removed {len(state['ids'])} {kind} "
                f"peak(s) on {entry}/frame{int(frame):05d}"
            )

        self.viewer._push_undo(_CallbackAction(_undo, _redo))

    # -- Phase tracking (mlgidBASE track_peaks) --

    def _on_track_scan_requested(self, threshold: float, length: int) -> None:
        """Enqueue an mlgidBASE ``track_peaks`` run for the active entry.

        Rides the ordinary pipeline queue (silx detach, busy gating,
        logging, entry snapshot all come for free); the captured
        ``TrackingPayload`` comes back through ``_on_pipeline_finished``
        -> ``_on_phase_track_result``.
        """
        if self.session is None:
            return
        if self._view_worker_blocks_pipeline():
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        self._entry_queue_total = 1
        self._entry_queue_pos = 0
        self._show_tracking_progress()
        self._enqueue_pipeline(
            self.session.temp_path,
            PipelineCommand("track_peaks", {
                "entry": entry,
                "threshold": float(threshold),
                "length": int(length),
            }),
        )

    def _show_tracking_progress(
        self, label: str = "Tracking peaks across the scan…",
        manual: bool = False,
    ) -> None:
        """Show a centred, always-visible loading dialog for the run.

        track_peaks is one opaque mlgidBASE call with no clean per-frame
        progress (its cost is the file open + IoU/graph, not the reads),
        so a GUI-thread ``QTimer`` advances the bar asymptotically toward
        ~95% while the worker runs; ``_close_tracking_progress`` snaps it
        to 100% on finish. ``setMinimumDuration(0)`` + explicit ``show``
        defeat QProgressDialog's default 4 s show delay (longer than the
        whole run).

        ``manual=True`` skips the fake timer entirely: the caller drives
        the bar with REAL progress values (the Interpolate-track chain
        feeds it the workers' per-frame ``frameProgress`` ticks via
        ``_on_pipeline_frame_progress``, so the dialog and the Pipeline
        panel's bar agree instead of the dialog idling at ~95%)."""
        self._close_tracking_progress()
        dlg = QProgressDialog(self)
        dlg.setWindowTitle("Peak tracking")
        dlg.setLabelText(label)
        dlg.setCancelButton(None)          # no clean way to interrupt mlgidBASE
        dlg.setRange(0, 100)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)
        dlg.show()
        self._track_progress_dialog = dlg
        if manual:
            return

        timer = QTimer(self)
        timer.setInterval(80)

        def _tick() -> None:
            v = dlg.value()
            if v < 95:
                dlg.setValue(v + max(1, int((95 - v) * 0.06)))

        timer.timeout.connect(_tick)
        timer.start()
        self._track_progress_timer = timer

    def _close_tracking_progress(self) -> None:
        if self._track_progress_timer is not None:
            self._track_progress_timer.stop()
            self._track_progress_timer.deleteLater()
            self._track_progress_timer = None
        # Detach the attribute FIRST: setValue on a window-modal
        # QProgressDialog runs processEvents(), so re-entrant handlers
        # (frameProgress, opFinished) must already see "no dialog".
        dlg = self._track_progress_dialog
        self._track_progress_dialog = None
        if dlg is not None:
            dlg.setValue(100)
            dlg.close()
            dlg.deleteLater()

    def _on_phase_track_result(self, payload: object, command) -> None:
        """Install a finished ``track_peaks`` payload (panel + views)."""
        if self.session is None or not isinstance(
            payload, phase_tracking.TrackingPayload
        ):
            return
        entry = command.kwargs.get("entry") or payload.entry
        if self.entry_combo.currentText() != entry:
            self.statusBar().showMessage(
                "Peak tracking finished for a no-longer-active entry; "
                "results discarded.",
                4000,
            )
            return
        tables = self._read_fitted_tables(payload, entry)
        ids = (
            phase_tracking.member_ids(payload, tables)
            if tables else [None] * payload.n_members
        )
        # Upstream cannot track rings (infinite angular width -> NaN
        # IoU -> zeroed -> single-member components cut by `length`), so
        # track them GUI-side by 1-D radial IoU and append the ring
        # tracks to the payload. They then flow through the table, the
        # views, and the only-tracked filter exactly like spot tracks.
        if tables:
            ring_comps = phase_tracking.track_rings(
                payload, ids, tables, payload.threshold, payload.length
            )
            if ring_comps:
                payload.components = payload.components + ring_comps
        self._scan_payload = payload
        self._scan_member_ids = ids
        self._scan_fitted_tables = tables
        self._scan_track_entry = entry
        # Which tracks are rings (any ring member; rings and spots never
        # mix in a component). The q-map draws these as dashed arcs.
        self._scan_ring_tracks = self._compute_ring_tracks(payload, ids, tables)
        # Map each track to the matched crystal phases its tracked peaks
        # belong to (for the q-map phase-identity overlay). No-op / empty
        # when Matching hasn't run.
        matched_tables = self._read_matched_tables(payload, entry, tables)
        self._scan_track_phases = phase_tracking.match_tracks_to_structures(
            payload, ids, matched_tables
        )
        # Pre-assign overlay colours for every matched identity (sorted,
        # so the assignment is deterministic no matter which frame is
        # rendered first).
        all_keys = sorted({
            s.color_key
            for structs in matched_tables.values() if structs
            for s in structs
        })
        self.viewer.seed_matched_colors(all_keys)
        self.scan_tracking_panel.set_payload(payload, ids)
        # Re-apply the show-only-tracked overlay filter against the
        # fresh membership (or lift it if the box is unticked).
        self._push_tracked_filter()
        if self._phase_views_window is not None:
            self._phase_views_window.set_context(
                self.session.temp_path, entry, self.viewer.n_frames
            )
            self._phase_views_window.set_payload(payload)
            self._phase_views_window.set_ring_tracks(self._scan_ring_tracks)
            self._phase_views_window.set_track_phases(self._scan_track_phases)

    def _compute_ring_tracks(self, payload, ids, tables) -> set:
        """Set of track indices whose members are rings (checked via the
        reconstructed ids against the fitted tables' ``is_ring``)."""
        rings: set = set()
        for k, comp in enumerate(payload.components):
            for i in comp:
                tagged = ids[int(i)]
                if tagged is None:
                    continue
                t = tables.get(int(tagged[0]))
                if t is None or len(t) == 0:
                    continue
                hits = np.flatnonzero(
                    np.asarray(t.ids, dtype=int) == int(tagged[1])
                )
                if hits.size == 1 and bool(t.is_ring[int(hits[0])]):
                    rings.add(k)
                    break
        return rings

    def _read_fitted_tables(self, payload, entry: str) -> dict:
        """One-handle read of every payload frame's fitted table.

        Feeds ``phase_tracking.member_ids`` (upstream's capture hook
        carries no peak ids) and is kept on ``_scan_fitted_tables``
        for the only-tracked display options (gap interpolation
        anchors, GUI-side ring tracking). The per-frame tables are tiny.
        """
        tables: dict[int, object] = {}
        try:
            import h5py

            with h5py.File(self.session.temp_path, "r") as f:
                for frame in sorted({int(x) for x in payload.frame_num}):
                    tables[frame] = file_model.read_peaks(
                        f, entry, frame
                    )["fitted"]
        except Exception:
            logger.debug(
                "suppressed exception in MainWindow._read_fitted_tables",
                exc_info=True,
            )
            return {}
        return tables

    def _read_matched_tables(self, payload, entry: str, fitted_tables: dict) -> dict:
        """One-handle read of every payload frame's matched structures.

        ``read_matched_peaks`` needs the frame's fitted table (already
        read for member-id reconstruction), so this reuses
        ``fitted_tables``. Feeds ``match_tracks_to_structures`` for the
        q-map phase-identity overlay; empty on any error or when
        Matching hasn't run.
        """
        out: dict = {}
        try:
            import h5py

            with h5py.File(self.session.temp_path, "r") as f:
                for frame in sorted({int(x) for x in payload.frame_num}):
                    out[frame] = file_model.read_matched_peaks(
                        f, entry, frame, fitted_tables.get(frame)
                    )
        except Exception:
            logger.debug(
                "suppressed exception in MainWindow._read_matched_tables",
                exc_info=True,
            )
            return {}
        return out

    def _invalidate_scan_tracks(self) -> None:
        """Drop phase-tracking results and clear panel + views.

        Called whenever the fitted tables the tracks reference may have
        changed identity: entry switch, ``run_fitting`` finish,
        Tools > Clear fitted / Reset.
        """
        self._scan_payload = None
        self._scan_member_ids = None
        self._scan_fitted_tables = {}
        self._scan_track_phases = {}
        self._scan_ring_tracks = set()
        self._scan_track_entry = None
        # A gap-fill run in flight references the payload being dropped
        # — its result must not be applied against fresh tracks.
        self._interp_fill_result = None
        self._interp_chain = None
        # panel.clear() unticks "Show only tracked peaks", whose toggled
        # emit routes back through _push_tracked_filter and lifts the
        # viewer filter; the explicit lift below is belt-and-braces for
        # the box-already-unticked path.
        self.scan_tracking_panel.clear()
        self.viewer.set_fitted_visible_only(None)
        if self._phase_views_window is not None:
            self._phase_views_window.clear()

    def _on_show_phase_views(self) -> None:
        """Open (lazily building) and raise the phase-tracking views."""
        if self._phase_views_window is None:
            self._phase_views_window = PhaseViewsWindow(
                self,
                busy_probe=lambda: self._pipe_thread is not None,
            )
            self._phase_views_window.frameJumpRequested.connect(
                self._on_phase_view_frame_jump
            )
            self._phase_views_window.saveFiguresRequested.connect(
                self._on_save_official_figures
            )
        if self.session is not None:
            entry = self.entry_combo.currentText()
            if entry:
                self._phase_views_window.set_context(
                    self.session.temp_path, entry, self.viewer.n_frames
                )
        self._phase_views_window.set_payload(self._scan_payload)
        self._phase_views_window.set_ring_tracks(self._scan_ring_tracks)
        self._phase_views_window.set_track_phases(self._scan_track_phases)
        self._phase_views_window.show()
        self._phase_views_window.raise_()
        self._phase_views_window.activateWindow()

    def _view_worker_blocks_pipeline(self) -> bool:
        """True (+ status message) while a phase-views worker holds a
        read handle on the file.

        pygid opens the NeXus file in r+ mode even for reads
        (nexus_reader.get_dataset), and HDF5 refuses an r+ open while
        another same-process handle has the file open read-only — so a
        pipeline run starting during a waterfall / mean-image
        computation dies with "Unable to synchronously open file (file
        is already open for read-only)". The silx detach dance cannot
        release that worker mid-loop; refusing to start is the honest
        fix (the views window refuses in the opposite direction via
        its busy_probe).
        """
        w = self._phase_views_window
        if w is not None and w.worker_active:
            self.statusBar().showMessage(
                "Wait for the phase-views computation (waterfall / mean "
                "image) to finish before running pipeline commands — it "
                "holds a read handle on the file.",
                6000,
            )
            return True
        return False

    def _on_only_tracked_toggled(self, _checked: bool) -> None:
        self._push_tracked_filter()

    def _push_tracked_filter(self) -> None:
        """Apply (or lift) the show-only-tracked overlay filter.

        Builds ``{frame: {fitted_id, ...}}`` from the surviving tracks'
        members via the reconstructed ids (members whose id could not
        be matched back are unavoidably hidden — best-effort, flagged
        in the checkbox tooltip) and pushes it to
        ``viewer.set_fitted_visible_only``. Anything else lifts it.

        Rings are tracked too: ``_on_phase_track_result`` appended the
        GUI-side ring tracks (``phase_tracking.track_rings``) to
        ``payload.components``, so tracked rings are whitelisted here
        like any spot track and untracked rings are hidden — no
        special-casing needed at this layer. Gap frames filled by
        "Interpolate track" join the whitelist the same way: their new
        fitted rows are appended as track members.
        """
        panel = self.scan_tracking_panel
        payload = self._scan_payload
        ids = self._scan_member_ids
        if (
            not panel.only_tracked_checked
            or payload is None
            or ids is None
        ):
            self.viewer.set_fitted_visible_only(None)
        else:
            allowed: dict[int, set[int]] = {}
            for comp in payload.components:
                for i in comp:
                    tagged = ids[int(i)]
                    if tagged is not None:
                        allowed.setdefault(int(tagged[0]), set()).add(
                            int(tagged[1])
                        )
            self.viewer.set_fitted_visible_only(allowed)
        # The "Unmatched fitted peaks" dock row exists only while the
        # frame has unmatched rows the filter lets through — rebuild the
        # matched panel so the row appears/disappears with the filter.
        self._refresh_matched_panel(
            self.viewer.current_frame,
            self.viewer.matched_structures(self.viewer.current_frame),
        )

    def _on_interpolate_tracks_requested(self) -> None:
        """Fill each track's gap frames with REAL peaks (spots + rings).

        Plans the injections (``phase_tracking.plan_gap_fills``), then
        enqueues an ``interpolate_tracks`` command (inject a detected
        box/ring per gap at the interpolated position, 2D-fit it,
        persist the fitted row) followed by ``run_matching`` command(s)
        scoped to the affected frames — segments and rings separately,
        and CIF-restricted to the structures the filled tracks were
        tracked in (see ``_build_interp_match_commands``). The fill
        records come back through ``_on_pipeline_finished`` and are
        applied as new track members in ``_apply_interp_fill_result``
        once the LAST re-match lands. The loading dialog is driven by
        the workers' real per-frame progress across the whole chain.
        """
        if self.session is None or self._view_worker_blocks_pipeline():
            return
        payload = self._scan_payload
        ids = self._scan_member_ids
        entry = self.entry_combo.currentText()
        if (
            payload is None
            or ids is None
            or not entry
            or entry != self._scan_track_entry
        ):
            return
        plan = phase_tracking.plan_gap_fills(
            payload, ids, self._scan_fitted_tables, self._scan_ring_tracks
        )
        if not plan:
            self.statusBar().showMessage(
                "Interpolate track: every track already has a fitted "
                "peak on each frame of its span — nothing to fill.",
                6000,
            )
            return
        match_kwargs = self.pipeline_panel.matching_kwargs()
        if match_kwargs is None:
            QMessageBox.warning(
                self, "Interpolate track",
                "Gap filling ends with a re-match of the affected "
                "frames, but no CIF source is configured. Set the CIF "
                "file/folder (or pickle) in the Pipeline panel's "
                "Matching section first.",
            )
            return
        # Same fit config the pipeline panel would use for run_fitting,
        # so filled boxes are fitted like every other peak.
        panel = self.pipeline_panel
        try:
            fit_params = {
                "crit_angle": float(panel.fit_crit_angle.value()),
                "clustering_distance_peaks": float(
                    panel.fit_dist_peaks.value()
                ),
                "clustering_distance_rings": float(
                    panel.fit_dist_rings.value()
                ),
                "clustering_extend": int(panel.fit_cluster_extend.value()),
                "theta_fixed": bool(panel.fit_theta_fixed.isChecked()),
            }
        except Exception:
            logger.debug(
                "suppressed exception reading fit params for "
                "interpolate_tracks", exc_info=True,
            )
            fit_params = {}
        n_boxes = sum(len(v) for v in plan.values())
        self.pipeline_panel.append_log(
            f"Interpolate track: filling {n_boxes} gap box(es) across "
            f"{len(plan)} frame(s) of {entry}…"
        )
        fill_cmd = PipelineCommand("interpolate_tracks", {
            "entry": entry,
            "plan": plan,
            "fit_params": fit_params,
        })
        match_cmds = self._build_interp_match_commands(
            plan, entry, match_kwargs
        )
        self._interp_fill_result = None
        self._entry_queue_total = 1
        self._entry_queue_pos = 0
        # Chain bookkeeping: one bar tick per gap frame (fill) + per
        # matched frame (each matching command), driven by the REAL
        # frameProgress signals in _on_pipeline_frame_progress.
        commands = [fill_cmd] + match_cmds
        ticks = {id(fill_cmd): len(plan)}
        for cmd in match_cmds:
            ticks[id(cmd)] = len(cmd.kwargs.get("frame_num") or []) or 1
        self._interp_chain = {
            "commands": commands,
            "ticks": ticks,
            "total": max(1, sum(ticks.values())),
            "base": 0,
        }
        self._show_tracking_progress(
            "Interpolate track: fitting injected boxes…", manual=True,
        )
        for cmd in commands:
            self._enqueue_pipeline(self.session.temp_path, cmd)

    def _build_interp_match_commands(
        self, plan: dict, entry: str, match_kwargs: dict
    ) -> list:
        """Build the re-match command(s) for the gap-fill chain.

        Segments and rings are matched SEPARATELY (mlgidmatch matches
        one ``peaks_type`` per run), each command pinned to the frames
        that gained fills of that type.

        The CIF source is ALWAYS restricted — no new structures can
        appear from a gap fill. Per frame + type the set is the union
        of (a) the structures the filled tracks were tracked in
        (``_scan_track_phases`` — a track matched to one structure
        re-matches against only that CIF), (b) for a never-matched
        filled track, every structure identified anywhere in this
        scan for that type (the peak may belong to one of them; the
        full panel folder is never used), and (c) the CIFs already
        matched on the frame for that type — matching REWRITES the
        frame's ``matched_<type>_*`` datasets wholesale, so dropping
        an already-identified CIF from the set would erase its
        existing solutions there. An empty set (nothing identified in
        the scan) or an unsubsettable source (pickle / missing .cif
        files) SKIPS that matching pass with a log line — the filled
        peaks stay unmatched rather than brute-forcing every loaded
        structure. Frames sharing the same (type, CIF-set) collapse
        into one command with a ``frame_num`` list.
        """
        matched_tables = self._read_matched_tables(
            self._scan_payload, entry, self._scan_fitted_tables
        )
        phases = self._scan_track_phases
        # Everything identified anywhere in this scan, split by type —
        # the fallback set for filled tracks that were never matched.
        scan_cifs = {"matched_segments": set(), "matched_rings": set()}
        for structs in matched_tables.values():
            for st in structs:
                for prefix in scan_cifs:
                    if st.solution_field.startswith(prefix):
                        scan_cifs[prefix].add(str(st.cif))
        # The raw .cif source is preferred for restriction: after a
        # panel "Parse CIFs" click, matching_kwargs() carries the
        # cached CifPattern OBJECT, which cannot be subset — the raw
        # text can.
        raw_source = self.pipeline_panel.cif_source_text()
        if not raw_source and isinstance(match_kwargs.get("cif_prepr"), str):
            raw_source = match_kwargs["cif_prepr"]

        groups: dict = {}   # (peaks_type, frozenset) -> [frames]
        skipped: list = []
        for frame, specs in plan.items():
            for ring_flag, peaks_type, prefix in (
                (False, "segments", "matched_segments"),
                (True, "rings", "matched_rings"),
            ):
                type_specs = [
                    s for s in specs
                    if bool(s.get("is_ring", False)) == ring_flag
                ]
                if not type_specs:
                    continue
                cifs: set = set()
                for s in type_specs:
                    track_cifs = phases.get(int(s["track"]))
                    if track_cifs:
                        cifs.update(str(c) for c in track_cifs)
                    else:
                        cifs.update(scan_cifs[prefix])
                for st in matched_tables.get(int(frame), []):
                    if st.solution_field.startswith(prefix):
                        cifs.add(str(st.cif))
                if not cifs:
                    skipped.append((peaks_type, int(frame), "no structures"))
                    continue
                groups.setdefault(
                    (peaks_type, frozenset(cifs)), []
                ).append(int(frame))
        cmds: list = []
        for (peaks_type, cif_key), frames in sorted(
            groups.items(), key=lambda kv: (kv[0][0], min(kv[1]))
        ):
            restricted = self._restrict_cif_source(raw_source, set(cif_key))
            if restricted is None:
                for f in frames:
                    skipped.append((peaks_type, f, "source not subsettable"))
                continue
            cmds.append(PipelineCommand("run_matching", {
                **match_kwargs,
                "entry": entry,
                "frame_num": sorted(set(frames)),
                "peaks_type": peaks_type,
                "cif_prepr": restricted,
            }))
        if skipped:
            by_reason: dict = {}
            for peaks_type, f, reason in skipped:
                by_reason.setdefault(reason, set()).add(f)
            for reason, frames in sorted(by_reason.items()):
                self.pipeline_panel.append_log(
                    "Interpolate track: skipping the re-match on frame(s) "
                    f"{sorted(frames)} ({reason}) — filled peaks there "
                    "stay unmatched; no new structures are introduced. "
                    "Use a .cif folder/file source (not a pickle) and "
                    "keep the identified structures' files in place to "
                    "enable restricted re-matching."
                )
        return cmds

    def _restrict_cif_source(self, base, cifs: set):
        """Semicolon-joined ``.cif`` paths covering exactly ``cifs``.

        ``base`` is the panel's raw ``cif_prepr`` string (a folder, a
        single ``.cif``, or a ``;``-separated list — the forms
        ``pipeline._build_cif_pattern_from_raw`` accepts). Returns
        ``None`` when the source cannot be subset: a pickle path, a
        pre-parsed ``CifPattern`` object, or any tracked CIF whose file
        is missing from the source (restricting would silently drop
        that structure — the caller falls back to the full source).
        CIF names are matched case-insensitively against file stems
        (``MatchedStructure.cif`` is the extensionless basename).
        """
        import os

        if not cifs or not isinstance(base, str) or not base.strip():
            return None
        paths = [p.strip() for p in base.split(";") if p.strip()]
        if len(paths) == 1 and paths[0].lower().endswith((".pickle", ".pkl")):
            return None
        wanted = {str(c).lower() for c in cifs}
        try:
            if len(paths) == 1 and os.path.isdir(paths[0]):
                folder = paths[0]
                hits = [
                    os.path.join(folder, f)
                    for f in sorted(os.listdir(folder))
                    if f.lower().endswith(".cif")
                    and os.path.splitext(f)[0].lower() in wanted
                ]
            else:
                hits = [
                    p for p in paths
                    if p.lower().endswith(".cif")
                    and os.path.splitext(os.path.basename(p))[0].lower()
                    in wanted
                ]
        except Exception:
            logger.debug(
                "suppressed exception restricting the CIF source",
                exc_info=True,
            )
            return None
        found = {
            os.path.splitext(os.path.basename(p))[0].lower() for p in hits
        }
        if not hits or found != wanted:
            return None
        return ";".join(hits)

    def _apply_interp_fill_result(self) -> None:
        """Append the gap-fill peaks as members of their origin tracks.

        Runs after the chained re-match finishes. Each fill record's
        fitted row is re-read from the file (fresh table → exact
        coordinates), appended to the payload's member arrays, and its
        index added to ``components[track]`` — so the track's span/count
        update everywhere (panel table, views, only-tracked whitelist)
        without a re-run of ``track_peaks``. Member ids and the q-map
        phase mapping are then rebuilt against the fresh tables.
        """
        result = self._interp_fill_result
        self._interp_fill_result = None
        payload = self._scan_payload
        entry = self._scan_track_entry
        if not result or payload is None or entry is None or self.session is None:
            return
        # Fresh fitted tables for the affected frames (they just gained
        # rows; ids beyond the old table are unknown to the cache).
        affected = sorted({int(rec["frame"]) for rec in result})
        try:
            import h5py

            with h5py.File(self.session.temp_path, "r") as f:
                for frame in affected:
                    self._scan_fitted_tables[frame] = file_model.read_peaks(
                        f, entry, frame
                    )["fitted"]
        except Exception:
            logger.debug(
                "suppressed exception re-reading fitted tables after "
                "interpolate_tracks", exc_info=True,
            )
            return
        appended = 0
        for rec in result:
            frame = int(rec["frame"])
            fid = int(rec["fitted_id"])
            track = int(rec["track"])
            table = self._scan_fitted_tables.get(frame)
            if table is None or not (0 <= track < len(payload.components)):
                continue
            hits = np.flatnonzero(np.asarray(table.ids, dtype=int) == fid)
            if hits.size != 1:
                continue
            j = int(hits[0])
            new_index = payload.n_members
            payload.q_xy = np.append(payload.q_xy, float(table.q_xy[j]))
            payload.q_z = np.append(payload.q_z, float(table.q_z[j]))
            payload.frame_num = np.append(payload.frame_num, frame)
            payload.amplitude = np.append(
                payload.amplitude, float(table.amplitude[j])
            )
            comp = list(payload.components[track])
            comp.append(new_index)
            payload.components[track] = comp
            appended += 1
        if not appended:
            return
        # Rebuild the derived state against the fresh membership.
        self._scan_member_ids = phase_tracking.member_ids(
            payload, self._scan_fitted_tables
        )
        matched_tables = self._read_matched_tables(
            payload, entry, self._scan_fitted_tables
        )
        self._scan_track_phases = phase_tracking.match_tracks_to_structures(
            payload, self._scan_member_ids, matched_tables
        )
        self.scan_tracking_panel.set_payload(
            payload, self._scan_member_ids
        )
        self._push_tracked_filter()
        if self._phase_views_window is not None:
            self._phase_views_window.set_payload(payload)
            self._phase_views_window.set_ring_tracks(self._scan_ring_tracks)
            self._phase_views_window.set_track_phases(self._scan_track_phases)
        self.statusBar().showMessage(
            f"Interpolate track: filled {appended} gap peak(s) "
            f"(fitted + matched) across {len(affected)} frame(s).",
            8000,
        )

    def _on_phase_view_frame_jump(self, frame: int) -> None:
        if self.viewer.n_frames <= 0:
            return
        self.viewer.set_frame(
            max(0, min(int(frame), self.viewer.n_frames - 1))
        )

    def _on_save_official_figures(self, path: str, axis: str) -> None:
        """Run upstream's real matplotlib export (save_fig=True).

        GUI thread on purpose: the run is quick (fitted-table reads +
        two Agg savefigs) and matplotlib's pyplot state machine is not
        worker-thread-safe under a Qt backend. The silx tree is
        detached around mlgidBASE's file open, same as pipeline runs.
        """
        if self.session is None or self._pipe_thread is not None:
            return
        payload = self._scan_payload
        entry = self._scan_track_entry
        if payload is None or not entry:
            return
        try:
            with self._detached_silx_tree():
                from mlgidbase import mlgidBASE  # noqa: N814

                analysis = mlgidBASE(filename=str(self.session.temp_path))
                analysis.track_peaks(
                    entry=entry,
                    threshold=payload.threshold,
                    length=payload.length,
                    axis=axis,
                    plot_params={
                        "plot_result": False,
                        "save_fig": True,
                        "path_to_save_fig": path,
                    },
                )
        except Exception as exc:
            QMessageBox.critical(self, "Save figures", str(exc))
            return
        self.statusBar().showMessage(
            f"Saved mlgidBASE tracking figures next to {path}", 6000
        )

    def _forward_selection_to_scan_tracking(self, sel) -> None:
        """Mirror a FITTED-peak selection onto the Scan-tracking table.

        The reverse of ``_on_track_row_selected``: clicking ANY member
        peak of a track (any frame, image or Peaks dock) highlights the
        track's row — whose columns carry the first/last/frames span —
        and echoes "track N: frames x-y" to the status bar. Non-fitted
        or trackless selections just clear the row highlight. The panel
        applies the highlight under its ``_applying_external`` guard,
        so this never bounces the viewer to another frame.
        """
        if self._scan_payload is None or self._scan_track_entry is None:
            return
        if self.entry_combo.currentText() != self._scan_track_entry:
            return
        if sel is None or getattr(sel, "kind", None) != "fitted":
            self.scan_tracking_panel.clear_external_selection()
            return
        track = self.scan_tracking_panel.set_external_peak(
            int(sel.frame), int(sel.peak_id)
        )
        if track is not None:
            first, last, n_frames, _members = self._scan_payload.track_span(
                track
            )
            span = (
                f"frame {first}" if first == last
                else f"frames {first}-{last}"
            )
            self.statusBar().showMessage(
                f"Fitted peak {int(sel.peak_id)} (frame {int(sel.frame)}) "
                f"belongs to track {track}: {span}, present in "
                f"{n_frames} frame(s)",
                6000,
            )

    def _on_track_row_selected(self, frame: int, peak_id: int) -> None:
        """Jump to a clicked track's representative member (+select).

        ``peak_id`` is -1 when no fitted id could be reconstructed for
        the track — then the click only jumps to the frame.
        """
        if self.session is None or self._scan_track_entry is None:
            return
        entry = self.entry_combo.currentText()
        if entry != self._scan_track_entry:
            return
        frame = int(frame)
        if int(self.viewer.current_frame) != frame:
            # frameChanged is wired synchronously to _load_frame_peaks,
            # so the frame's tables are in _frame_peaks afterwards.
            self.viewer.set_frame(frame)
        if int(peak_id) < 0:
            return
        table = self.viewer._frame_peaks.get(frame, {}).get("fitted")
        if table is None:
            try:
                table = file_model.load_peaks(
                    self.session.temp_path, entry, frame
                ).get("fitted")
            except Exception:
                logger.debug(
                    "suppressed exception in MainWindow._on_track_row_selected",
                    exc_info=True,
                )
                return
        if table is None or len(table) == 0:
            return
        ids = [int(x) for x in table.ids]
        if int(peak_id) in ids:
            self._select_table_row(
                frame, "fitted", table, ids.index(int(peak_id))
            )

    def _on_peak_row_write_requested(
        self, frame: int, kind: str, peak_id: int, polar: dict
    ) -> None:
        """Persist a detected/fitted box edit straight to the NeXus file.

        Drops silx's read handle for the duration of the write (matching the
        pipeline-run dance in ``_on_pipeline_run``) so h5py can open r+, then
        re-attaches. On KeyError (peak vanished), the undo/redo stacks are
        cleared since they're keyed on stale ids.
        """
        if self.session is None:
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        with self._detached_silx_tree():
            try:
                file_model.update_peak_row(
                    self.session.temp_path, entry, frame, kind, peak_id, **polar
                )
            except KeyError:
                QMessageBox.warning(
                    self, "Edit failed",
                    f"Peak id={peak_id} no longer exists in the file. "
                    "Undo history has been cleared.",
                )
                self.viewer.clear_history()
            except Exception as exc:
                QMessageBox.critical(self, "Edit failed", str(exc))
                self.viewer.clear_history()
            else:
                self.session.mark_dirty()
                self._update_title()

    def _load_frame_peaks(self, entry: str, frame: int) -> None:
        """Load + install one frame's detected/fitted/matched peaks into
        the viewer, at most once per frame per entry load.

        Peak datasets are tiny, so loading the frame the user is actually
        looking at on demand is cheap — far cheaper than the old
        load-every-frame-up-front loop, which dominated entry-switch lag.
        Reset via ``_loaded_peak_frames`` on each entry (re)load.
        """
        if self.session is None or frame < 0:
            return
        if entry != getattr(self, "_loaded_peaks_entry", None):
            return
        if frame in self._loaded_peak_frames:
            return
        # Read through the viewer's already-open FrameSource handle when it
        # matches this entry: its master handle has the entry's external
        # link resolved, so reading the tiny peak tables is an in-file
        # navigation, NOT a fresh multi-second network ``h5py.File`` open.
        # That synchronous open per frame/entry was the freeze on big
        # external-link masters (the off-thread frame read alone wasn't
        # enough — this peak read still ran on the GUI thread). Fall back to
        # open-by-path only when no live handle is available.
        src = self.viewer._frame_source
        try:
            if src is not None and src.is_open and src.entry == entry:
                peaks, matched = src.read_frame_overlays(frame)
            else:
                peaks = file_model.load_peaks(self.session.temp_path, entry, frame)
                matched = file_model.load_matched_peaks(
                    self.session.temp_path, entry, frame, peaks.get("fitted")
                )
        except Exception:
            logger.debug("suppressed exception in MainWindow._load_frame_peaks", exc_info=True)
            peaks = {kind: None for kind in OVERLAY_KINDS}
            matched = []
        self.viewer.set_peaks(frame, peaks)
        self.viewer.set_matched_structures(frame, matched)
        self._loaded_peak_frames.add(frame)

    def _load_entry_into_viewer(
        self, entry: str, *, preserve_view: bool = False
    ) -> None:
        """Load ``entry`` into the image viewer.

        ``preserve_view``: when True, the viewer keeps its current zoom and
        frame index across the reload. Used after pipeline ops and direct
        h5py edits (the underlying stack is unchanged — only peak overlays
        are different). Switching to a different entry passes False so the
        viewer auto-ranges to the new axes.
        """
        assert self.session is not None
        # Reuse a pre-warmed FrameSource (opened + frame-0 read off the GUI
        # thread by CopyWorker) when one is queued for this entry, so the
        # first render never blocks on a slow read. Otherwise load
        # synchronously — the path tests and entry switches use.
        overlays = None
        prewarm = getattr(self.session, "_prewarm", None)
        if prewarm is not None and prewarm[0] == entry:
            self.session._prewarm = None  # type: ignore[attr-defined]
            stack = file_model.stack_from_source(prewarm[1])
            # Consume the worker-read first-frame peaks alongside the frame.
            overlays = getattr(self.session, "_prewarm_overlays", None)
            self.session._prewarm_overlays = None  # type: ignore[attr-defined]
        else:
            try:
                stack = file_model.load_entry(self.session.temp_path, entry)
            except Exception as exc:
                QMessageBox.warning(self, "Load failed", f"Could not load {entry}: {exc}")
                return
        self._install_stack_into_viewer(
            entry, stack, preserve_view=preserve_view, overlays=overlays
        )

    def _install_stack_into_viewer(
        self, entry: str, stack, *, preserve_view: bool = False, overlays=None
    ) -> None:
        """Render an already-built ``EntryStack`` and wire up the per-entry
        viewer state (slider, peaks, polar profile, prefetch, peaks dock).

        Split out of ``_load_entry_into_viewer`` so the slow disk read (the
        ``FrameSource`` open + frame-0 read) can be done off the GUI thread
        — by ``CopyWorker`` (prewarm) or ``EntryLoadWorker`` (interactive
        switch) — and only this fast, GUI-thread install runs here. The
        viewer's ``show_stack`` releases the previous entry's FrameSource
        before adopting the new one (image_viewer.py), so this also frees
        the old handle.
        """
        self.viewer.show_stack(stack, preserve_view=preserve_view)
        # Match the slider to the new stack's frame range. preserve_view
        # already restores the prior frame index inside show_stack; we
        # only need to repopulate the slider's bounds + label here.
        self._refresh_frame_slider()
        # Peaks load lazily, per frame, not all up front. The old
        # all-frames loop did N HDF5 opens per entry switch (hundreds on a
        # many-frame entry) — a big part of the switch lag. Load only the
        # frame the viewer landed on; other frames load on demand in
        # ``_on_viewer_frame_changed`` via ``_load_frame_peaks``. The
        # ``_peaks_loaded`` set is reset here so a post-pipeline reload
        # (preserve_view=True) re-reads fresh peaks as frames are revisited.
        self._loaded_peaks_entry = entry
        self._loaded_peak_frames = set()
        # Phase-tracking results are per-entry: another entry's tracks
        # reference different frames/ids. Same-entry overlay refreshes
        # (peak edits, pipeline reattach) keep them.
        if self._scan_track_entry is not None and entry != self._scan_track_entry:
            self._invalidate_scan_tracks()
        if not is_mlgidbase_available():
            self.scan_tracking_panel.set_scan_available(
                False,
                reason="Peak tracking needs the [pipeline] backends "
                       "(mlgidbase) installed.",
            )
        else:
            self.scan_tracking_panel.set_scan_available(
                self.viewer.n_frames > 1,
                reason="Peak tracking needs a multi-frame entry.",
            )
        if self._phase_views_window is not None and self.session is not None:
            self._phase_views_window.set_context(
                self.session.temp_path, entry, self.viewer.n_frames
            )
        cur = self.viewer.current_frame
        if overlays is not None and overlays[0] == cur:
            # Peaks for the landed frame were read off-thread (worker) — install
            # them directly so the GUI does no SFTP read here. Other frames
            # still load on demand via ``_load_frame_peaks``.
            self.viewer.set_peaks(cur, overlays[1])
            self.viewer.set_matched_structures(cur, overlays[2])
            self._loaded_peak_frames.add(cur)
        else:
            self._load_frame_peaks(entry, cur)
        # Initial panel state for whichever frame the viewer is showing now.
        self._refresh_matched_panel(
            self.viewer.current_frame,
            self.viewer.matched_structures(self.viewer.current_frame),
        )
        # Hand the polar transform to the profile viewer. After the
        # lazy-loading milestone ``viewer.polar_data()`` returns a
        # ``(_LazyPolarStack, radius, angle)`` tuple — frames are
        # resampled on demand inside the FrameSource so this no longer
        # forces an eager precompute of the full polar stack. The
        # profile viewer indexes the wrapper by frame; the cursor
        # readout uses tuple indexing for single-pixel lookups.
        polar = self.viewer.polar_data()
        if polar is not None:
            self.profile_viewer.set_polar_stack(*polar)
        # Configure the prefetch worker for the new entry. Single-
        # frame stacks short-circuit inside the helper; multi-frame
        # stacks spawn the worker (if not already) and reset its
        # _done set so the next Play press starts filling from
        # frame current+1 onward.
        self._configure_prefetch_for_active_entry()
        # Repopulate the Peaks dock with the new entry's peaks. The
        # viewer's frameChanged path also drives this slot, but the
        # initial load may finish on the *same* frame index as the
        # previous session — in which case no frameChanged fires and
        # the panel would otherwise keep stale rows.
        self._refresh_peaks_table()

    # -- Async entry switching (off-GUI-thread frame read) --

    def _ensure_entry_load_worker(self) -> None:
        """Spawn the persistent EntryLoadWorker + thread on first use.

        Idempotent; mirrors ``_ensure_prefetch_worker``. The worker reads
        an entry's first frame off the GUI thread and hands back a ready
        ``FrameSource`` so a switch never blocks on a slow network read.
        """
        if self._entry_load_worker is not None:
            return
        self._entry_load_thread = QThread(self)
        self._entry_load_worker = EntryLoadWorker()
        self._entry_load_worker.moveToThread(self._entry_load_thread)
        self._entryLoadRequest.connect(
            self._entry_load_worker.load, Qt.ConnectionType.QueuedConnection,
        )
        self._entry_load_worker.loaded.connect(
            self._on_entry_loaded, Qt.ConnectionType.QueuedConnection,
        )
        # Raw-mode pair: same worker/thread, separate slot + result signal
        # (a raw load hands back a LazyRawStack, not a FrameSource).
        self._rawLoadRequest.connect(
            self._entry_load_worker.load_raw, Qt.ConnectionType.QueuedConnection,
        )
        self._entry_load_worker.raw_loaded.connect(
            self._on_raw_entry_loaded, Qt.ConnectionType.QueuedConnection,
        )
        self._entry_load_thread.start()

    def _load_entry_async(self, entry: str) -> None:
        """Switch to ``entry`` without blocking the GUI.

        A pre-warm queued for this entry (the just-opened first entry) is
        consumed synchronously — it is already in memory, so there is
        nothing to wait for. Otherwise the read is dispatched to the
        EntryLoadWorker and the bottom-left bar marches until it returns;
        ``_entry_req_id`` is bumped so a superseded switch's late result is
        dropped.
        """
        if self.session is None:
            return
        prewarm = getattr(self.session, "_prewarm", None)
        if prewarm is not None and prewarm[0] == entry:
            # Already warmed (open path) — install immediately, no worker.
            self._load_entry_into_viewer(entry)
            self._update_status_frame()
            return
        self._entry_req_id += 1
        self._ensure_entry_load_worker()
        self._sb_open_label.setText(f"Loading {entry}…")
        self._sb_open_label.show()
        self._sb_open_bar.setRange(0, 0)  # indeterminate: one read, no stages
        self._sb_open_bar.show()
        self._entryLoadRequest.emit(
            str(self.session.temp_path), entry, self._entry_req_id,
        )

    def _on_entry_loaded(self, request_id: int, entry: str, source, overlays=None) -> None:
        """Install the entry whose frame + peaks the worker just read, unless
        a newer switch has superseded it (then the source is released)."""
        if request_id != self._entry_req_id:
            # Superseded by a later switch — drop the stale source.
            if source is not None:
                try:
                    source.release()
                except Exception:
                    logger.debug("suppressed stale-source release", exc_info=True)
            return
        self._dismiss_open_progress()
        if source is None:
            QMessageBox.warning(self, "Load failed", f"Could not load {entry}")
            return
        # The active session may have changed between dispatch and arrival
        # (file closed / switched); guard so we never install a source from
        # a stale file. ``_set_active_session`` bumps the id, so a mismatch
        # here means this result is orphaned.
        if self.session is None or self.session.kind != "nexus":
            try:
                source.release()
            except Exception:
                logger.debug("suppressed orphan-source release", exc_info=True)
            return
        stack = file_model.stack_from_source(source)
        self._install_stack_into_viewer(
            entry, stack, preserve_view=False, overlays=overlays
        )
        self._update_status_frame()

    # -- UI state --

    def _refresh_peaks_table(self) -> None:
        """Repopulate the Peaks dock from the viewer's current state.

        Called on entry-load completion, post-pipeline reload (via
        ``_load_entry_into_viewer``), and on every frame change (via
        ``_refresh_peaks_table_on_frame``). Empties all three tabs
        when no session is active.
        """
        if self.session is None or self.session.kind != "nexus":
            self.peaks_table_panel.clear()
            return
        frame = self.viewer.current_frame
        peaks_for_frame = self.viewer._frame_peaks.get(frame) or {}
        matched = self.viewer.matched_structures(frame)
        self.peaks_table_panel.set_frame_peaks(
            frame,
            peaks_for_frame.get("detected"),
            peaks_for_frame.get("fitted"),
            matched,
        )

    def _refresh_peaks_table_on_frame(self, _frame: int) -> None:
        """Slot for ``viewer.frameChanged`` — drops the unused frame
        index since ``_refresh_peaks_table`` re-reads it from the
        viewer."""
        self._refresh_peaks_table()

    def _on_peak_selected_from_table(self, sel: SelectedPeak | None) -> None:
        """Route a table-row click back into the image viewer.

        The panel builds the SelectedPeak with ``frame=0`` since it
        has no viewer context; stamp the real current_frame here
        before handing it to ``_set_selected``. The viewer's
        equality guard short-circuits the round-trip if the
        selection didn't actually change.
        """
        if sel is None:
            return
        sel.frame = self.viewer.current_frame
        self.viewer._set_selected(sel)

    def _refresh_matched_panel(self, _frame: int, structures: list) -> None:
        """Rebuild the per-structure rows under the Matched-peaks master.

        Called on every frame change and after a fresh entry load. We blow
        away the old QCheckBox widgets and create new ones — the structure
        list is small (1-N rows) so this is cheap.

        The active filter (see ``_apply_matched_filter``) is re-applied
        at the end so newly-built rows respect the current search
        text without the user needing to retype.
        """
        # Clear children of the dynamic container.
        while self._matched_struct_layout.count():
            item = self._matched_struct_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._matched_empty_label = None
        self._matched_filter_empty_label = None
        self._matched_struct_checkboxes.clear()
        self._matched_struct_rows.clear()
        self._matched_struct_probs.clear()

        # Fitted rows no loaded structure claims (tracked filter
        # applied) get their own sub-row below the structures.
        has_unmatched = self.viewer.has_unmatched_fitted(_frame)

        if not structures and not has_unmatched:
            self._matched_empty_label = QLabel("<i>No matched solutions for this frame.</i>")
            self._matched_empty_label.setWordWrap(True)
            self._matched_struct_layout.addWidget(self._matched_empty_label)
            return

        for i, s in enumerate(structures):
            # Stable per-identity pen (CIF + hkl) so the swatch matches
            # the overlay and stays the same colour across frames.
            pen = self.viewer.matched_pen(s)
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            swatch = QLabel()
            # Mirror the *exact* pen used to render the structure on the
            # image so the user can map a row to its overlay shape even
            # when colour repeats — the dashed/dotted swatch flags it.
            swatch.setPixmap(_make_pen_swatch(pen))
            row.addWidget(swatch)
            chk = QCheckBox(s.label)
            chk.setChecked(self.viewer.matched_visibility(_frame, s.unique_id))
            chk.toggled.connect(
                lambda v, uid=s.unique_id: self._on_matched_structure_toggled(uid, v)
            )
            row.addWidget(chk)
            row.addStretch(1)
            row_widget = QWidget()
            row_widget.setLayout(row)
            self._matched_struct_layout.addWidget(row_widget)
            self._matched_struct_checkboxes[s.unique_id] = chk
            self._matched_struct_rows[s.unique_id] = row_widget
            try:
                self._matched_struct_probs[s.unique_id] = float(s.probability)
            except Exception:
                logger.debug("suppressed exception in MainWindow._refresh_matched_panel", exc_info=True)
                self._matched_struct_probs[s.unique_id] = 0.0

        # Reset the min-probability slider so each frame's first
        # render shows every structure; the user can drag up to
        # filter weak matches.
        if structures:
            self._seed_matched_prob_slider(structures)
            self._apply_matched_filter()

        # "Unmatched fitted peaks" sub-row: the grey pseudo-group of
        # fitted rows no loaded structure claims, rendered in the
        # matched display style (markers/boxes). Off by default (the
        # fitted overlay already draws these rows); the checkbox state
        # persists across frames. Deliberately NOT registered in
        # _matched_struct_checkboxes/rows so the CIF-substring and
        # min-probability filters never touch it.
        if has_unmatched:
            pen = {"color": UNMATCHED_COLOR, **MATCHED_STYLE}
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            swatch = QLabel()
            swatch.setPixmap(_make_pen_swatch(pen))
            row.addWidget(swatch)
            chk = QCheckBox("Unmatched fitted peaks")
            chk.setToolTip(
                "Show every fitted peak that is NOT part of any matched "
                "structure, in neutral grey, using the matched display "
                "style (boxes or markers). Follows the Matched master "
                "toggle and the \"Show only tracked peaks\" filter — "
                "handy for viewing tracked-but-unmatched peaks as "
                "markers alongside the structures."
            )
            chk.setChecked(self.viewer.unmatched_visible())
            chk.toggled.connect(self.viewer.set_unmatched_visible)
            row.addWidget(chk)
            row.addStretch(1)
            row_widget = QWidget()
            row_widget.setLayout(row)
            self._matched_struct_layout.addWidget(row_widget)

    def _on_matched_prob_changed(self, value: int) -> None:
        """Slider 0–100 → readable 0.00–1.00 in the side label, then
        re-apply the composite filter."""
        if hasattr(self, "_matched_prob_value_label"):
            self._matched_prob_value_label.setText(f"{value / 100.0:.2f}")
        self._apply_matched_filter()

    def _on_detected_score_changed(self, value: int) -> None:
        """Slider 0–100 → 0.00–1.00 cutoff forwarded to the viewer.

        Updates the side-label readout, then asks the viewer to
        re-render the detected overlay with the new threshold. The
        filter is applied in ``GIWAXSImageViewer._render_overlays``
        via a row-subset of the detected ``PeakTable``.
        """
        if hasattr(self, "_detected_score_value_label"):
            self._detected_score_value_label.setText(f"{value / 100.0:.2f}")
        if hasattr(self, "viewer"):
            self.viewer.set_detected_score_cutoff(value / 100.0)

    def _seed_detected_score_slider(self) -> None:
        """Reset the Detected min-score slider to the lowest score
        on the current frame so the default shows every detection.

        Called on every frame change and after entry load. Uses
        ``blockSignals`` so the seed doesn't trigger a redundant
        viewer re-render (we're already rendering this frame).
        """
        if not hasattr(self, "_detected_score_slider"):
            return
        frame = self.viewer.current_frame
        peaks = self.viewer._frame_peaks.get(frame, {})
        det = peaks.get("detected")
        try:
            if det is not None and len(det) > 0:
                scores = np.asarray(det.score, dtype=float)
                if scores.size and np.all(np.isfinite(scores)):
                    lo = float(scores.min())
                else:
                    lo = 0.0
            else:
                lo = 0.0
        except Exception:
            logger.debug("suppressed exception in MainWindow._seed_detected_score_slider", exc_info=True)
            lo = 0.0
        lo = max(0.0, min(1.0, lo))
        slider_val = int(round(lo * 100))
        self._detected_score_slider.blockSignals(True)
        try:
            self._detected_score_slider.setValue(slider_val)
        finally:
            self._detected_score_slider.blockSignals(False)
        self._detected_score_value_label.setText(f"{slider_val / 100.0:.2f}")
        # Also apply the new cutoff to the viewer so its render
        # state matches the slider after the silent setValue.
        if hasattr(self, "viewer"):
            self.viewer.set_detected_score_cutoff(slider_val / 100.0)

    def _seed_matched_prob_slider(self, structures: list) -> None:
        """Reset the matched min-probability slider to the lowest
        probability on the current frame so the default shows every
        structure. Mirrors ``_seed_detected_score_slider``."""
        if not hasattr(self, "_matched_prob_slider"):
            return
        try:
            probs = [float(s.probability) for s in structures]
        except Exception:
            logger.debug("suppressed exception in MainWindow._seed_matched_prob_slider", exc_info=True)
            probs = []
        lo = min(probs) if probs else 0.0
        lo = max(0.0, min(1.0, lo))
        slider_val = int(round(lo * 100))
        self._matched_prob_slider.blockSignals(True)
        try:
            self._matched_prob_slider.setValue(slider_val)
        finally:
            self._matched_prob_slider.blockSignals(False)
        self._matched_prob_value_label.setText(f"{slider_val / 100.0:.2f}")

    def _apply_matched_filter(self, *_args) -> None:
        """Hide per-structure rows that fail either:
        (a) the substring filter — label must contain
        ``_matched_filter_edit``'s text (case-insensitive), or
        (b) the min-probability slider — structure probability
        must be ≥ ``_matched_prob_slider`` value / 100.

        Empty substring + zero cutoff = show everything. When all
        rows are hidden by the active filter a "No matches" hint
        replaces them so the empty pane doesn't look like a bug.
        """
        text = ""
        if hasattr(self, "_matched_filter_edit"):
            text = self._matched_filter_edit.text().strip().lower()
        prob_cutoff = 0.0
        if hasattr(self, "_matched_prob_slider"):
            prob_cutoff = self._matched_prob_slider.value() / 100.0

        # Drop a leftover "no filter matches" hint before recomputing.
        if self._matched_filter_empty_label is not None:
            self._matched_filter_empty_label.deleteLater()
            self._matched_filter_empty_label = None

        any_visible = False
        hidden_uids: set[str] = set()
        for uid, row_widget in self._matched_struct_rows.items():
            chk = self._matched_struct_checkboxes.get(uid)
            label = chk.text().lower() if chk is not None else ""
            substring_ok = (text == "") or (text in label)
            prob = self._matched_struct_probs.get(uid, 0.0)
            # Epsilon = half the slider's natural step (0.01 → 0.005)
            # so a structure with p=1.00 passes when the slider is at
            # 1.00, regardless of FP roundoff in the stored value.
            prob_ok = prob >= prob_cutoff - 0.005
            visible = substring_ok and prob_ok
            row_widget.setVisible(visible)
            if visible:
                any_visible = True
            else:
                hidden_uids.add(uid)

        # Forward the hidden-uid set to the viewer so filtered-out
        # structures also drop their overlay on the image. Independent
        # of the per-structure checkbox state — see
        # ``GIWAXSImageViewer.set_matched_filter_hidden``.
        if hasattr(self, "viewer"):
            self.viewer.set_matched_filter_hidden(hidden_uids)

        # Only show the "no matches" hint when an active filter has
        # zeroed the list — pure "no matched solutions for this
        # frame" is handled by ``_matched_empty_label`` separately.
        filter_active = bool(text) or prob_cutoff > 0.0
        if filter_active and self._matched_struct_rows and not any_visible:
            reasons = []
            if text:
                reasons.append(
                    f"CIF substring '{self._matched_filter_edit.text()}'"
                )
            if prob_cutoff > 0.0:
                reasons.append(f"p ≥ {prob_cutoff:.2f}")
            self._matched_filter_empty_label = QLabel(
                f"<i>No structures match {' and '.join(reasons)}.</i>"
            )
            self._matched_filter_empty_label.setWordWrap(True)
            self._matched_struct_layout.addWidget(self._matched_filter_empty_label)

    def _on_matched_master_toggled(self, checked: bool) -> None:
        """Master toggles cascade to every per-structure row.

        Unchecking the master now also unchecks every structure
        checkbox; checking it back rechecks them all. The viewer's
        own master flag is updated either way so its hit-test gating
        stays in sync. Per-checkbox ``setChecked`` calls are blocked
        from re-emitting ``toggled`` so the structure-toggled slot
        doesn't interpret the cascade as a user-driven single-show.
        """
        self.viewer.set_matched_master_visible(checked)
        for uid, chk in self._matched_struct_checkboxes.items():
            with QSignalBlocker(chk):
                chk.setChecked(checked)
            self.viewer.set_matched_structure_visible(uid, checked)

    def _on_matched_style_changed(self, _index: int) -> None:
        """Switch the matched overlay between boxes and q-map markers."""
        self.viewer.set_matched_display_style(
            self._matched_style_combo.currentData()
        )

    def _on_matched_structure_toggled(self, uid: str, checked: bool) -> None:
        """Per-structure toggle. Promotes a ``check while master is off``
        click into a "show only this one" view: every other structure is
        unchecked, the master is auto-ticked (without re-cascading), and
        only the freshly-checked structure ends up visible.
        """
        self.viewer.set_matched_structure_visible(uid, checked)
        if not checked:
            return
        if self._matched_master_check.isChecked():
            return
        # Master was off → user wants to see this single structure.
        # Force the others off (both UI + viewer state) before flipping
        # the master ON, since the master toggle would otherwise
        # cascade and re-show every structure.
        for other_uid, chk in self._matched_struct_checkboxes.items():
            if other_uid == uid:
                continue
            with QSignalBlocker(chk):
                chk.setChecked(False)
            self.viewer.set_matched_structure_visible(other_uid, False)
        with QSignalBlocker(self._matched_master_check):
            self._matched_master_check.setChecked(True)
        # blockSignals suppressed _on_matched_master_toggled, so call
        # the viewer's master flag directly.
        self.viewer.set_matched_master_visible(True)

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
            # Light separation so the eye can scan the row.
            w.setStyleSheet("padding: 0 8px; border-left: 1px solid #444;")
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
        self._sb_open_bar = QProgressBar()
        self._sb_open_bar.setRange(0, 0)          # indeterminate "busy" march
        self._sb_open_bar.setMaximumWidth(110)
        self._sb_open_bar.setTextVisible(False)
        self._sb_open_label.hide()
        self._sb_open_bar.hide()
        sb.addWidget(self._sb_open_label)
        sb.addWidget(self._sb_open_bar)

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
            if op_name == "interpolate_tracks":
                dlg.setLabelText(
                    "Interpolate track: fitting injected boxes… "
                    f"({min(int(done), ticks)}/{ticks} frames)"
                )
            elif op_name == "run_matching":
                dlg.setLabelText(
                    "Interpolate track: matching "
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

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept drops carrying local file URLs.

        Acceptance is loose at enter time — content classification
        happens in ``dropEvent`` so the cursor reflects "yes you can
        drop" while the user drags over the window. Non-file payloads
        (text, internal Qt drags) are ignored.
        """
        mime = event.mimeData()
        if mime.hasUrls() and any(u.isLocalFile() for u in mime.urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragEnterEvent) -> None:
        # Same gate as dragEnter — Qt fires move events repeatedly while
        # the drag is in flight and the proposed-action state has to
        # stay accepted across them or the drop won't fire.
        mime = event.mimeData()
        if mime.hasUrls() and any(u.isLocalFile() for u in mime.urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        """Open every dropped file via the unified _open_paths classifier."""
        urls = event.mimeData().urls()
        paths: list[Path] = []
        for u in urls:
            if not u.isLocalFile():
                continue
            local = u.toLocalFile()
            if not local:
                continue
            paths.append(Path(local))
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        self._open_paths(paths)

    def closeEvent(self, event: QCloseEvent) -> None:
        # Idempotent: close() can be delivered more than once (Qt can
        # re-emit a close event, and pytest-qt's teardown calls close()
        # again on every registered widget after our own close). Running
        # the teardown twice would re-clear already-destroyed pyqtgraph
        # widgets — on some PySide6 builds that raises
        # "Internal C++ object already deleted" during shutdown. Skip a
        # second pass.
        if getattr(self, "_closed", False):
            event.accept()
            return
        # Each loaded file may have unsaved changes — prompt per dirty
        # session in load order so the user gets the same per-file save
        # dialog they would on _action_close_file.
        for s in list(self._sessions):
            if not self._confirm_discard_changes(s):
                event.ignore()
                return
        # Past the point of no return — mark closed so a re-entrant or
        # pytest-qt second close() short-circuits above.
        self._closed = True
        # Drop the app-wide undo/redo event filter so a closed window
        # (which may linger under the test session's gc.disable) stops
        # receiving key events.
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        # Stop frame playback so the timer doesn't fire one last tick
        # against a torn-down viewer during shutdown.
        self._pause_playback()
        # All clear — tear everything down. silx must release its handles
        # before we delete the temp files. One-way detach: the app is
        # closing, so there is no reattach (hence not _detached_silx_tree).
        self._detach_silx_tree()
        self.viewer.clear()
        self.profile_viewer.clear()
        # Drop the shared open-progress dialog if it's still up (rare —
        # the modal would normally have blocked the close). Otherwise its
        # leftover modal overlay would dim the next opened window too.
        self._dismiss_open_progress()
        # Stop a CIF parse if one is running — closing the window while
        # CifPattern construction is in flight otherwise drops the worker
        # thread on the floor and Qt complains at exit.
        if self._cif_parse_thread is not None:
            self._cif_parse_thread.quit()
            self._cif_parse_thread.wait()
        # Stop a conversion run if one is in flight — pygid + h5py do
        # their own cleanup on a clean thread exit.
        if self._conv_thread is not None:
            self._conv_thread.quit()
            self._conv_thread.wait()
        # Close the phase-tracking views window (its closeEvent waits
        # for any in-flight view-computation worker).
        if self._phase_views_window is not None:
            self._phase_views_window.close()
        self._close_tracking_progress()
        # Shut the background prefetch worker down cleanly. Release
        # its h5py handle first (so the worker stops trying to read
        # frames), then quit + wait the thread so its event loop
        # exits before we delete its Q objects.
        if self._prefetch_worker is not None:
            self._prefetchRelease.emit()
            # Process the queued release so it lands on the worker's
            # thread before we quit it; otherwise the worker would
            # try to read on a destroyed h5py.File during shutdown.
            QCoreApplication.processEvents()
            self._prefetch_thread.quit()
            self._prefetch_thread.wait()
            self._prefetch_worker.deleteLater()
            self._prefetch_worker = None
            self._prefetch_thread.deleteLater()
            self._prefetch_thread = None
        # Shut the async entry-load worker down. Bump the request id first
        # so any result still in the queue is treated as stale (its source
        # released) rather than installed into a tearing-down viewer.
        if self._entry_load_worker is not None:
            self._entry_req_id += 1
            self._entry_load_thread.quit()
            self._entry_load_thread.wait()
            self._entry_load_worker.deleteLater()
            self._entry_load_worker = None
            self._entry_load_thread.deleteLater()
            self._entry_load_thread = None
        for s in list(self._sessions):
            # Inactive sessions may hold a stashed (parked) FrameSource —
            # release the handle before the temp file is deleted.
            stash = getattr(s, "_prewarm", None)
            if stash is not None:
                self._release_prewarm(stash)
            s.close()
        self._sessions.clear()
        self._active_session = None
        event.accept()
