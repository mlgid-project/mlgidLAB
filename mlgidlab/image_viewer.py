from __future__ import annotations

import os

# Pin pyqtgraph to PySide6 before it auto-detects.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QRectF,
    QSettings,
    QSignalBlocker,
    Qt,
    Signal,
)
from PySide6.QtGui import QAction, QColor, QPainterPath
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from mlgidlab.file_model import (
    EntryStack,
    FrameSource,
    MatchedStructure,
    PeakTable,
    _LazyImageStack,
    _LazyPolarStack,
)
from mlgidlab.polar import polar_to_qxyz, stack_to_polar  # noqa: F401 (stack_to_polar retained as a reference impl)

import logging
logger = logging.getLogger(__name__)

OVERLAY_KINDS = ("detected", "fitted", "manual")
MODE_CARTESIAN = "cartesian"
MODE_POLAR = "polar"
# Raw detector data preview — pixel coordinates, no overlays. Reached only
# when a RawSession is active; converted-NeXus sessions never visit this
# mode and their existing Cartesian / Polar paths are unchanged.
MODE_RAW = "raw"

LABEL_MODIFIERS = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier

# Subdivisions along the angular edge for the full 0–90° range; narrower
# segments scale down proportionally (with a small minimum for sharp corners).
ANGULAR_SUBDIV_FULL = 90
ANGULAR_SUBDIV_MIN = 4

# Outer bounds for clipping a peak's angular extent before drawing it as
# a polygon. Set to atan2's full range so peaks produced by converted
# images that span multiple quadrants (e.g. ``vert_positive=False`` →
# angles in [0°, 180°]) still draw correctly. Peaks whose angle / width
# are non-finite are treated as "ring" — their polygon spans the full
# range below.
ANGLE_MIN_DEG = -180.0
ANGLE_MAX_DEG = 180.0

# Visual style for each overlay kind. Dashed for "raw" detection output,
# solid for the refined fit, dotted yellow for user-drawn manual labels.
OVERLAY_STYLE: dict[str, dict] = {
    "detected": {"color": "#ff5c5c", "style": Qt.PenStyle.DashLine, "width": 1.2},
    "fitted":   {"color": "#26d0ce", "style": Qt.PenStyle.SolidLine, "width": 1.2},
    "manual":   {"color": "#ffeb3b", "style": Qt.PenStyle.SolidLine, "width": 1.6},
}

SELECTION_STYLE = {"color": "#ffffff", "style": Qt.PenStyle.SolidLine, "width": 2.5}

# Faint preview of the would-be fitted_peaks box for the currently selected
# manual peak. Same hue as the fitted overlay so the user reads the
# relationship at a glance, but dashed + reduced opacity so it's clearly a
# preview rather than a stored peak.
FITTED_PREVIEW_STYLE = {
    "color": OVERLAY_STYLE["fitted"]["color"],
    "style": Qt.PenStyle.DashLine,
    "width": 1.4,
}
FITTED_PREVIEW_OPACITY = 0.45

# Distinct, dark-mode-legible palette for matched structures. Cycled by
# insertion order so multiple structures in one frame are easy to tell apart.
# Avoids the existing detected/fitted/manual hues to prevent confusion.
MATCHED_PALETTE: tuple[str, ...] = (
    "#3b82f6",  # blue
    "#84cc16",  # lime
    "#ef4444",  # red
    "#c0c0c0",  # silver
    "#eab308",  # yellow
    "#ec4899",  # pink
    "#a855f7",  # purple
    "#06b6d4",  # cyan
    "#22c55e",  # green
    "#f97316",  # orange
)
# Ordered farthest-point-first (each hue is the most perceptually distant
# from all already-assigned ones), so the first N structures pick up the
# most mutually distinguishable colours; every pair is >=34 dE (CIE76)
# apart. The previous palette had two near-identical cyans (~5 dE), which
# read as "the same colour" for adjacent structures.
# Line styles cycled after the palette wraps. Combined with
# MATCHED_PALETTE this yields ``len(palette) * len(styles)`` unique
# pens before any (colour, style) pair repeats — enough headroom for
# the 28-row deduped solutions on real datasets without resorting to
# colour-only disambiguation.
MATCHED_LINE_STYLES: tuple[Qt.PenStyle, ...] = (
    Qt.PenStyle.SolidLine,
    Qt.PenStyle.DashLine,
    Qt.PenStyle.DashDotLine,
    Qt.PenStyle.DotLine,
)
MATCHED_LINE_WIDTH = 1.6
# Backwards-compat: callers that still want the default line style
# can keep using this dict. New code should prefer ``matched_pen_for``
# which combines the palette + line-style cycle.
MATCHED_STYLE = {"style": MATCHED_LINE_STYLES[0], "width": MATCHED_LINE_WIDTH}

# Screen-fixed diameter (px) of the circular markers used by the
# "Markers" matched-display style (peaks as hollow circles, rings as
# dashed arcs — the q-map look, easier on the eye during scan playback
# than boxes whose size varies with each peak's width).
MATCHED_MARKER_SIZE = 11

# Pseudo-group for fitted peaks NOT claimed by any matched structure.
# Rendered through the matched overlay machinery (same display style,
# master toggle, tracked-peak filter) so unmatched peaks can be shown
# as markers alongside the structures — neutral grey, matching the hue
# the phase views use for unmatched tracks. Off by default: the fitted
# overlay already draws every fitted row, so this group is an opt-in
# view. The sentinel uid/key ride the existing per-item and
# per-identity bookkeeping without colliding with real structures.
UNMATCHED_COLOR = "#969696"
UNMATCHED_UID = "__unmatched__"
UNMATCHED_KEY = ("__unmatched__", 0, 0, 0)


def matched_pen_for(index: int) -> dict:
    """Return ``{color, style, width}`` for the ``index``-th structure.

    Colour cycles first so adjacent rows pick up a different hue at
    the same line style — the palette gives the strongest visual
    contrast and is enough for files with up to ``len(MATCHED_PALETTE)``
    matched structures. Once the palette wraps, the line style steps
    to the next (dashed → dash-dot → dotted) so the next 10 rows are
    still distinguishable from the first 10 even when their colours
    repeat. With 10 colours × 4 styles the palette runs out only past
    40 simultaneous structures.
    """
    n_colors = len(MATCHED_PALETTE)
    n_styles = len(MATCHED_LINE_STYLES)
    color = MATCHED_PALETTE[index % n_colors]
    style = MATCHED_LINE_STYLES[(index // n_colors) % n_styles]
    return {"color": color, "style": style, "width": MATCHED_LINE_WIDTH}

# Curated list of colormaps. Names are matplotlib's; pg.colormap.get falls
# back to matplotlib's registry, which is always available since matplotlib
# is a transitive dep via silx.
COLORMAPS = ("viridis", "inferno", "plasma", "magma", "cividis", "gray")
DEFAULT_COLORMAP = "magma"


def _disable_viewport_scroll(widget) -> None:
    """Disable QAbstractScrollArea-level scrolling on a pyqtgraph widget.

    Pyqtgraph's GraphicsView / PlotWidget inherit from QAbstractScrollArea,
    so even with the scrollbars hidden the viewport can still slide
    when the scene rect is slightly bigger than the visible area
    (typically by a few pixels of axis-label padding). Overriding
    ``scrollContentsBy`` to a no-op blocks every scroll path —
    scrollbar drag, wheel-on-bar, two-finger gesture, programmatic
    `setValue` — without touching the inner ViewBox's pan / zoom,
    which lives one Qt level deeper as a graphics-item event.

    Implemented by reparenting the instance to a dynamically-created
    subclass; safer than instance-level monkey-patching since Qt
    dispatches virtual methods through the C++ vtable.
    """
    cls = type(widget)
    if cls.__name__.endswith("_NoScroll"):
        return
    new_cls = type(
        cls.__name__ + "_NoScroll",
        (cls,),
        {"scrollContentsBy": lambda self, dx, dy: None},
    )
    widget.__class__ = new_cls


def _bin_index(axis: np.ndarray, value: float) -> int:
    """Floor-based bin index for an evenly-spaced axis.

    The image-display routines call ``setImage(pos=axis[0], scale=step)``,
    so axis[0] is the LOWER edge of pixel 0 and pixel ``i`` covers
    ``[axis[0] + i*step, axis[0] + (i+1)*step)``. Returning the bin
    index by ``floor`` instead of ``argmin(|axis - v|)`` keeps the
    cursor readout constant within a displayed pixel — argmin
    transitions at axis-midpoints, which is half a pixel off from
    where the user sees the boundary.
    """
    n = len(axis)
    if n == 0:
        return 0
    if n == 1:
        return 0
    step = (float(axis[-1]) - float(axis[0])) / (n - 1)
    if step == 0.0:
        return 0
    idx = int(np.floor((float(value) - float(axis[0])) / step))
    if idx < 0:
        return 0
    if idx >= n:
        return n - 1
    return idx


def _robust_levels(frame: np.ndarray) -> tuple[float, float]:
    finite = frame[np.isfinite(frame)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, (1.0, 99.5))
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _apply_raw_flips(frame: np.ndarray, flip_lr: bool, flip_ud: bool) -> np.ndarray:
    """Orient a file-order (H, W) raw detector frame for preview.

    Mirrors pygid's ``process_image`` (flipud then fliplr) so the preview
    matches the orientation the conversion will produce. Returns the frame
    unchanged when neither flip is set.
    """
    if flip_ud:
        frame = np.flipud(frame)
    if flip_lr:
        frame = np.fliplr(frame)
    return frame


@dataclass
class _DisplayParams:
    """Per-mode display state. ``image_pg`` is the *single 2D frame*
    for the current frame index — we never feed pyqtgraph a 3D stack
    anymore (see lazy-loading milestone). The viewer keeps the most
    recent ``_DisplayParams`` so per-frame scrubs can re-use ``pos`` /
    ``scale`` / ``levels`` without re-deriving them.
    """
    image_pg: np.ndarray
    pos: tuple[float, float]
    scale: tuple[float, float]
    levels: tuple[float, float]
    x_label: tuple[str, str]
    y_label: tuple[str, str]


@dataclass
class ManualPeak:
    """A user-drawn polar peak box. In-memory only; phase 4c persists these."""

    radius: float
    angle: float
    radius_width: float
    angle_width: float
    is_ring: bool = False
    temp_id: int = 0


@dataclass
class SelectedPeak:
    """Snapshot of the currently-selected peak, regardless of source.

    Carries enough geometry for the ROI and parameter panel without forcing
    callers to know which overlay holds the underlying data. ``manual_ref``
    is set only when ``kind == "manual"`` and is the same instance held by
    ``_manual_peaks`` — mutating it propagates to the manual overlay.
    """

    kind: str  # "manual" | "detected" | "fitted" | "matched"
    frame: int
    peak_id: int
    radius: float
    angle: float
    radius_width: float
    angle_width: float
    is_ring: bool = False
    structure_uid: str | None = None
    # Human-readable structure label + overlay color, populated only for
    # matched selections so the parameter panel can render the source row
    # without re-deriving these from the viewer's matched bookkeeping.
    structure_label: str | None = None
    structure_color: str | None = None
    manual_ref: ManualPeak | None = None
    # mlgidDETECT confidence score for the underlying row. Populated for
    # detected / fitted / matched selections; left as None for manual
    # peaks (which have no model provenance) so the parameter panel can
    # skip the row instead of showing a misleading zero.
    score: float | None = None
    # Peak amplitude (2D-Gaussian peak height). Populated for detected /
    # fitted / matched selections; the profile viewer uses it to render
    # the projection of the persisted 2D Gaussian onto the radial /
    # angular axis (physics-audit F-06: "profile must reflect 2D fit").
    # None for manual peaks (no fit yet), which fall back to the live
    # 1D scipy fit on the integrated profile data.
    amplitude: float | None = None
    # When non-None, the selection represents a *matched structure*
    # rather than a single peak. The list holds every fitted-peak id
    # that belongs to the structure; the overlay highlight is drawn
    # around all of them at once. The single peak_id field still
    # carries a representative id (typically the first one) so the
    # ParameterPanel can render the structure_label + first-peak
    # geometry. Always None for detected / fitted / manual selections.
    multi_peak_ids: list[int] | None = None

    @classmethod
    def from_manual(cls, peak: ManualPeak, frame: int) -> SelectedPeak:
        return cls(
            kind="manual",
            frame=frame,
            peak_id=peak.temp_id,
            radius=peak.radius,
            angle=peak.angle,
            radius_width=peak.radius_width,
            angle_width=peak.angle_width,
            is_ring=peak.is_ring,
            manual_ref=peak,
        )

    def polar_tuple(self) -> tuple[float, float, float, float]:
        return (self.radius, self.angle, self.radius_width, self.angle_width)


# -- Undo/redo actions -----------------------------------------------------
#
# Each action carries the data needed to flip the viewer + (for FileGeom)
# the file. ``undo`` and ``redo`` mirror each other so we can move back and
# forth on the stack without special-casing.


class _Action(Protocol):
    def undo(self, viewer: "GIWAXSImageViewer") -> None: ...
    def redo(self, viewer: "GIWAXSImageViewer") -> None: ...


@dataclass
class ManualAddAction:
    frame: int
    peak: ManualPeak

    def undo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._undoable_remove_manual(self.frame, self.peak)

    def redo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._undoable_add_manual(self.frame, self.peak)


@dataclass
class ManualRemoveAction:
    frame: int
    peak: ManualPeak

    def undo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._undoable_add_manual(self.frame, self.peak)

    def redo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._undoable_remove_manual(self.frame, self.peak)


@dataclass
class ManualGeomAction:
    frame: int
    peak: ManualPeak
    before: tuple[float, float, float, float]  # (r, a, dr, da)
    after: tuple[float, float, float, float]

    def undo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._apply_manual_geom(self.frame, self.peak, self.before)

    def redo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._apply_manual_geom(self.frame, self.peak, self.after)


@dataclass
class ManualReplaceAction:
    """Atomic swap of the single manual peak on a frame.

    With the new "at most one manual box per frame" policy, drawing a
    new box replaces any existing one. We model that as a single undo
    entry (instead of separate add + remove entries) so a single
    Ctrl+Z rewinds the whole replace cleanly. ``old_peak`` may be None
    when the user drew the very first manual peak on this frame —
    redo then just adds the new one without removing anything.
    """

    frame: int
    old_peak: ManualPeak | None
    new_peak: ManualPeak

    def undo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._undoable_remove_manual(self.frame, self.new_peak)
        if self.old_peak is not None:
            viewer._undoable_add_manual(self.frame, self.old_peak)

    def redo(self, viewer: "GIWAXSImageViewer") -> None:
        if self.old_peak is not None:
            viewer._undoable_remove_manual(self.frame, self.old_peak)
        viewer._undoable_add_manual(self.frame, self.new_peak)


@dataclass
class FileGeomAction:
    frame: int
    kind: str  # "detected" | "fitted"
    peak_id: int
    before: tuple[float, float, float, float]
    after: tuple[float, float, float, float]

    def undo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._apply_file_geom(self.frame, self.kind, self.peak_id, self.before)

    def redo(self, viewer: "GIWAXSImageViewer") -> None:
        viewer._apply_file_geom(self.frame, self.kind, self.peak_id, self.after)


def _action_targets_manual(action: _Action, peak: ManualPeak) -> bool:
    """True when ``action`` references the ManualPeak ``peak`` by identity.

    Used by ``commit_manual_peak`` to scrub any stale stack entries that
    point at a peak we just persisted to the file.
    """
    target = getattr(action, "peak", None)
    return target is peak


def _peaks_subset(table: PeakTable, ids: list[int]) -> PeakTable:
    """Return a row-subset of ``table`` keyed by ``ids``.

    Used to build a multi-peak selection overlay from a matched
    structure's full id list. Rows whose id isn't in ``table.ids``
    are silently skipped (stale matched references — same tolerance
    applied in ``file_model.list_matched_structures``).
    """
    if len(table) == 0 or not ids:
        empty = np.zeros(0, dtype=float)
        return PeakTable(
            q_xy=empty, q_z=empty, angle=empty, radius=empty,
            angle_width=empty, radius_width=empty,
            is_ring=np.zeros(0, dtype=bool),
            ids=np.zeros(0, dtype=int),
            score=empty, amplitude=empty,
        )
    want = set(int(x) for x in ids)
    idx = np.array(
        [i for i in range(len(table)) if int(table.ids[i]) in want],
        dtype=int,
    )
    if idx.size == 0:
        return _peaks_subset(table, [])
    return PeakTable(
        q_xy=table.q_xy[idx],
        q_z=table.q_z[idx],
        angle=table.angle[idx],
        radius=table.radius[idx],
        angle_width=table.angle_width[idx],
        radius_width=table.radius_width[idx],
        is_ring=table.is_ring[idx],
        ids=table.ids[idx],
        score=table.score[idx],
        amplitude=table.amplitude[idx],
    )


def _peaks_from_manual(manual: list[ManualPeak]) -> PeakTable:
    """Adapt a list of ManualPeak to the PeakTable shape so the existing
    rendering helpers can draw them without special-casing."""
    if not manual:
        empty = np.zeros(0, dtype=float)
        return PeakTable(
            q_xy=empty, q_z=empty, angle=empty, radius=empty,
            angle_width=empty, radius_width=empty,
            is_ring=np.zeros(0, dtype=bool),
            ids=np.zeros(0, dtype=int),
            score=empty,
            amplitude=empty,
        )
    _radius = np.array([m.radius for m in manual], dtype=float)
    _angle = np.array([m.angle for m in manual], dtype=float)
    _qxy, _qz = polar_to_qxyz(_radius, _angle)
    return PeakTable(
        q_xy=_qxy,
        q_z=_qz,
        angle=_angle,
        radius=_radius,
        angle_width=np.array([m.angle_width for m in manual], dtype=float),
        radius_width=np.array([m.radius_width for m in manual], dtype=float),
        is_ring=np.array([m.is_ring for m in manual], dtype=bool),
        ids=np.array([m.temp_id for m in manual], dtype=int),
        score=np.zeros(len(manual), dtype=float),
        amplitude=np.zeros(len(manual), dtype=float),
    )


class _LabelEventFilter(QObject):
    """Qt event filter that emits high-level labelling signals from raw mouse
    events on a graphics-view's viewport.

    Installed instead of subclassing ``pg.ViewBox`` because pyqtgraph's drag
    dispatch only fires ``mouseDragEvent`` when the press was accepted at the
    QGraphicsItem layer — which it isn't for plain LMB on the image area, so a
    ViewBox subclass never sees the drag.
    """

    drawStarted = Signal(QPointF)
    drawUpdated = Signal(QPointF, QPointF)
    drawFinished = Signal(QPointF, QPointF)
    # LMB click (no drag). Carries the data-space click point + the
    # keyboard modifiers that were held at press time. Ctrl+click is
    # multi-select; bare click replaces; Ctrl+Alt+click never gets
    # here because the press branch routes it to ``drawStarted``.
    selectAt = Signal(QPointF, object)  # (QPointF, Qt.KeyboardModifiers)
    # Bare LMB double-click (no modifiers, no drag) — wired to reset zoom.
    doubleClicked = Signal()
    # Hover-aware cursor tracking — fires on every mouse move (with or
    # without a button held). The viewer translates the data-space point
    # into the public ``cursorMoved`` payload consumed by the status bar.
    cursorPos = Signal(QPointF)
    cursorLeft = Signal()

    # Pixel tolerance below which a press+release counts as a click, not a drag.
    CLICK_TOLERANCE_PX = 4

    def __init__(
        self, graphics_view, viewbox: pg.ViewBox, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._gv = graphics_view
        self._vb = viewbox
        self._drawing = False
        self._origin: QPointF | None = None
        self._press_pos: QPoint | None = None
        self._press_mods: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier

    def install(self) -> None:
        self._gv.viewport().installEventFilter(self)
        # MouseMove only fires with a button held unless tracking is on.
        # The status-bar cursor readout needs hover updates, so force it.
        self._gv.viewport().setMouseTracking(True)

    def eventFilter(self, _obj: QObject, ev: QEvent) -> bool:  # type: ignore[override]
        et = ev.type()
        if (
            et == QEvent.Type.MouseButtonDblClick
            and ev.button() == Qt.MouseButton.LeftButton
            and ev.modifiers() == Qt.KeyboardModifier.NoModifier
        ):
            # Bare LMB double-click anywhere on the image resets the zoom.
            # Modifier+double-click and other-button double-click fall
            # through so pyqtgraph's default handlers (e.g. ROI editing)
            # still see the event.
            self.doubleClicked.emit()
            return True
        if et == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
            mods = ev.modifiers()
            if _has_label_modifiers(mods):
                pos = self._viewport_to_data(ev.position().toPoint())
                self._origin = pos
                self._drawing = True
                self.drawStarted.emit(pos)
                return True  # consume so pan doesn't engage
            self._press_pos = ev.position().toPoint()
            self._press_mods = mods
            # Ctrl-only LMB press: consume so pyqtgraph's ViewBox
            # doesn't take it as a zoom-rect / pan-modifier drag and
            # swallow the matching release before our click handler
            # sees it. Bare LMB still falls through so pyqtgraph can
            # pan / zoom as before.
            if (
                bool(mods & Qt.KeyboardModifier.ControlModifier)
                and not bool(mods & Qt.KeyboardModifier.AltModifier)
            ):
                return True
            return False
        if et == QEvent.Type.MouseMove:
            # Always emit cursor position for the status-bar readout —
            # independent of whether a draw drag is in progress.
            data_pos = self._viewport_to_data(ev.position().toPoint())
            self.cursorPos.emit(data_pos)
            if self._drawing and self._origin is not None:
                self.drawUpdated.emit(self._origin, data_pos)
                return True
        if et == QEvent.Type.Leave:
            self.cursorLeft.emit()
        if et == QEvent.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton:
            if self._drawing and self._origin is not None:
                end = self._viewport_to_data(ev.position().toPoint())
                self.drawFinished.emit(self._origin, end)
                self._drawing = False
                self._origin = None
                return True
            if self._press_pos is not None:
                delta = ev.position().toPoint() - self._press_pos
                is_click = (
                    delta.manhattanLength() <= self.CLICK_TOLERANCE_PX
                )
                press_mods = self._press_mods
                self._press_pos = None
                self._press_mods = Qt.KeyboardModifier.NoModifier
                ctrl_only_press = (
                    bool(press_mods & Qt.KeyboardModifier.ControlModifier)
                    and not bool(press_mods & Qt.KeyboardModifier.AltModifier)
                )
                if is_click:
                    # Emit for any single-modifier combo (Ctrl alone for
                    # multi-select toggle, no modifier for replace). Ctrl+Alt
                    # never reaches here because the press branch routed
                    # to ``drawStarted`` and consumed the event. Other
                    # combinations (e.g. Shift) fall through to the same
                    # slot, which can decide whether to act.
                    pos = self._viewport_to_data(ev.position().toPoint())
                    self.selectAt.emit(pos, press_mods)
                    # Bare clicks don't consume — pyqtgraph still emits
                    # a click for menus, etc. Ctrl+LMB release IS
                    # consumed because we consumed the matching press;
                    # pyqtgraph would otherwise see an orphan release.
                    if ctrl_only_press:
                        return True
        return False

    def _viewport_to_data(self, viewport_pt: QPoint) -> QPointF:
        scene_pt = self._gv.mapToScene(viewport_pt)
        return self._vb.mapSceneToView(scene_pt)


def _has_label_modifiers(mods: Qt.KeyboardModifier) -> bool:
    return bool(
        mods & Qt.KeyboardModifier.ControlModifier
        and mods & Qt.KeyboardModifier.AltModifier
    )


class _PeakShapeItem(pg.GraphicsObject):
    """Draws a collection of peak shapes from a single QPainterPath.

    In polar mode every peak is an axis-aligned rectangle. In Cartesian mode,
    rings become quarter-circle arcs at the central radius and segments become
    polygons formed by tessellating the polar rectangle's angular edges.
    """

    def __init__(self, color: str, style: Qt.PenStyle, width: float) -> None:
        super().__init__()
        pen = pg.mkPen(QColor(color), width=width)
        pen.setStyle(style)
        pen.setCosmetic(True)  # constant pixel width regardless of zoom
        self._pen = pen
        self._path = QPainterPath()
        self._bounding = QRectF()

    def set_polar(
        self,
        peaks: PeakTable | None,
        extent: tuple[float, float] | None = None,
    ) -> None:
        path = QPainterPath()
        if peaks is not None and len(peaks) > 0:
            for i in range(len(peaks)):
                clip = _clip_angle(
                    float(peaks.angle[i]), float(peaks.angle_width[i]),
                    extent=extent,
                )
                if clip is None:
                    continue
                a_lo, a_hi = clip
                r = float(peaks.radius[i])
                dr = float(peaks.radius_width[i])
                path.addRect(QRectF(r - dr / 2, a_lo, dr, a_hi - a_lo))
        self._update_path(path)

    def set_cartesian(
        self,
        peaks: PeakTable | None,
        extent: tuple[float, float] | None = None,
    ) -> None:
        path = QPainterPath()
        if peaks is not None and len(peaks) > 0:
            for i in range(len(peaks)):
                clip = _clip_angle(
                    float(peaks.angle[i]), float(peaks.angle_width[i]),
                    extent=extent,
                )
                if clip is None:
                    continue
                a_lo, a_hi = clip
                path.addPath(
                    _polar_rect_polygon(
                        float(peaks.radius[i]),
                        float(peaks.radius_width[i]),
                        a_lo,
                        a_hi,
                    )
                )
        self._update_path(path)

    def clear_path(self) -> None:
        self._update_path(QPainterPath())

    def _update_path(self, path: QPainterPath) -> None:
        self.prepareGeometryChange()
        self._path = path
        self._bounding = path.boundingRect()
        self.update()

    def boundingRect(self) -> QRectF:
        return self._bounding

    def paint(self, painter, *_args) -> None:
        painter.setPen(self._pen)
        painter.drawPath(self._path)


def _polar_box_contains(peak: ManualPeak, x: float, y: float) -> bool:
    """Hit-test the polar bounding box of a ManualPeak."""
    r_lo = peak.radius - peak.radius_width / 2.0
    r_hi = peak.radius + peak.radius_width / 2.0
    a_lo = peak.angle - peak.angle_width / 2.0
    a_hi = peak.angle + peak.angle_width / 2.0
    return r_lo <= x <= r_hi and a_lo <= y <= a_hi


def _polar_table_row_contains(table: PeakTable, i: int, x: float, y: float) -> bool:
    """Hit-test row ``i`` of a PeakTable in polar coordinates."""
    r = float(table.radius[i])
    dr = float(table.radius_width[i])
    a = float(table.angle[i])
    da = float(table.angle_width[i])
    r_lo, r_hi = r - dr / 2.0, r + dr / 2.0
    a_lo, a_hi = a - da / 2.0, a + da / 2.0
    return r_lo <= x <= r_hi and a_lo <= y <= a_hi


def _cart_to_polar(q_xy: float, q_z: float) -> tuple[float, float]:
    """Map a Cartesian click ``(q_xy, q_z)`` back to ``(radius, angle_deg)``.

    Matches the convention used everywhere else in the pipeline: the
    Cartesian projections of a polar peak are written as
    ``q_xy = r * cos(a)``, ``q_z = r * sin(a)`` (see
    ``file_model.add_fitted_peak_row``), so inverting with
    ``hypot`` + ``atan2(q_z, q_xy)`` reproduces the original
    ``(radius, angle)``. The polar containment checks downstream
    can then run unchanged.
    """
    r = float(np.hypot(q_xy, q_z))
    a = float(np.degrees(np.arctan2(q_z, q_xy)))
    return r, a


def _cart_box_contains(peak: ManualPeak, q_xy: float, q_z: float) -> bool:
    """Cartesian hit-test for a ManualPeak's polar box."""
    r, a = _cart_to_polar(q_xy, q_z)
    return _polar_box_contains(peak, r, a)


def _cart_table_row_contains(
    table: PeakTable, i: int, q_xy: float, q_z: float,
) -> bool:
    """Cartesian hit-test for row ``i`` of a PeakTable."""
    r, a = _cart_to_polar(q_xy, q_z)
    return _polar_table_row_contains(table, i, r, a)


def _clip_angle(
    a_deg: float,
    da_deg: float,
    extent: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    """Clip a polar angular box to the viewer's visible range.

    Treats infinite or non-finite angle_width as 'spans the whole quadrant',
    so rings (whose angle_width is sometimes inf) still draw correctly.
    ``extent`` is the actual displayed angular axis of the active polar
    stack — pass it so ring overlays stop at the image edge instead of
    extending to the global ``[-180°, 180°]`` clipping bounds. When
    ``extent`` is None the global bounds are used as a fallback (raw-
    mode renders, unit tests).

    Returns (lo, hi) in degrees, or None if the box is empty/invalid.
    """
    if extent is None:
        ext_lo, ext_hi = ANGLE_MIN_DEG, ANGLE_MAX_DEG
    else:
        ext_lo, ext_hi = float(extent[0]), float(extent[1])
        if ext_hi < ext_lo:
            ext_lo, ext_hi = ext_hi, ext_lo
    if not np.isfinite(a_deg) or not np.isfinite(da_deg):
        a_lo, a_hi = ext_lo, ext_hi
    else:
        a_lo = a_deg - da_deg / 2.0
        a_hi = a_deg + da_deg / 2.0
    a_lo = max(a_lo, ext_lo)
    a_hi = min(a_hi, ext_hi)
    if a_hi <= a_lo:
        return None
    return a_lo, a_hi


def _polar_rect_polygon(
    radius: float, dr: float, a_lo_deg: float, a_hi_deg: float
) -> QPainterPath:
    """Render a polar rectangle (already clipped) as a closed polygon in q-space.

    For full-quadrant rings this becomes a proper quarter-annulus that closes
    along the q_xy and q_z axes. For narrow segments, a thin curved trapezoid.
    """
    a_lo = np.deg2rad(a_lo_deg)
    a_hi = np.deg2rad(a_hi_deg)
    inner = max(radius - dr / 2.0, 0.0)
    outer = radius + dr / 2.0

    span = a_hi_deg - a_lo_deg
    n_sub = max(
        int(np.ceil(span / 90.0 * ANGULAR_SUBDIV_FULL)),
        ANGULAR_SUBDIV_MIN,
    )
    angs = np.linspace(a_lo, a_hi, n_sub)

    path = QPainterPath()
    path.moveTo(QPointF(float(outer * np.cos(angs[0])), float(outer * np.sin(angs[0]))))
    for ang in angs[1:]:
        path.lineTo(QPointF(float(outer * np.cos(ang)), float(outer * np.sin(ang))))
    for ang in angs[::-1]:
        path.lineTo(QPointF(float(inner * np.cos(ang)), float(inner * np.sin(ang))))
    path.closeSubpath()
    return path


class GIWAXSImageViewer(QWidget):
    """Image viewer with Cartesian ↔ polar mode toggle + peak-box overlays."""

    # QSettings key for the last Custom aspect ratio (the on-screen
    # width:height). The mode itself is not persisted — the viewer always
    # starts in Fit; only the ratio is remembered so re-entering Custom is
    # sensible. Shares the global store with the theme / playback settings.
    _ASPECT_RATIO_KEY = "viewerAspectRatio"

    frameChanged = Signal(int)
    modeChanged = Signal(str)
    # Cursor readout — emits a dict describing the data point under the
    # cursor (q-mode vs pixel-mode), or None when the pointer leaves
    # the viewport. Consumers (status bar) format the dict for display.
    cursorMoved = Signal(object)
    manualPeakAdded = Signal(int, object)     # frame, ManualPeak
    manualPeakRemoved = Signal(int, object)   # frame, ManualPeak
    selectionChanged = Signal(object)         # SelectedPeak | None
    peakGeometryChanged = Signal(object)      # SelectedPeak whose r/dr/a/da changed
    # Drag-end variant of ``peakGeometryChanged``. Fires once when the
    # user releases the ROI handle (or the user-side undo/redo path
    # settles a geometry mutation). Consumers that want to react only
    # to the final position — not every drag tick — subscribe here.
    # MainWindow's 2D-preview refresh listens to this one because each
    # pygidfit call is ~100-500 ms and per-tick refits would freeze
    # the drag.
    peakGeometryDragFinished = Signal(object)  # SelectedPeak (post-drag)
    # Emitted on drag-end for non-manual peaks: (frame, kind, peak_id,
    # polar_kwargs). MainWindow drives the actual h5py mutation since it
    # owns the silx tree handle that needs releasing first.
    peakRowWriteRequested = Signal(int, str, int, dict)
    # Emitted when the user presses Delete on a single non-manual peak.
    deletePeakRequested = Signal(object)      # SelectedPeak
    # Emitted when Delete is pressed with >= 2 detected peaks selected:
    # carries the full multi-selection so the host can bulk-delete them
    # in one grouped, undoable action.
    deletePeaksRequested = Signal(list)       # list[SelectedPeak]
    # Emitted whenever the *current* frame's matched-structure list might be
    # different from what the UI showed last (frame change, fresh load,
    # re-render after pipeline run). Args: (frame, list[MatchedStructure]).
    matchedStructuresChanged = Signal(int, list)
    # Emitted alongside ``selectionChanged`` with the full multi-selection
    # list (primary + extras). Single-peak consumers stay on the legacy
    # ``selectionChanged(SelectedPeak | None)``; multi-aware consumers
    # (copy handler, batch-fit button enable) subscribe here.
    selectionsChanged = Signal(list)          # list[SelectedPeak]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        bar = QHBoxLayout()
        bar.setContentsMargins(8, 4, 8, 4)
        bar.addWidget(QLabel("View:"))
        self._radio_cart = QRadioButton("Cartesian")
        self._radio_polar = QRadioButton("Polar")
        self._radio_polar.setChecked(True)
        self._radio_group = QButtonGroup(self)
        self._radio_group.addButton(self._radio_cart)
        self._radio_group.addButton(self._radio_polar)
        self._radio_cart.toggled.connect(self._on_radio_toggled)
        bar.addWidget(self._radio_cart)
        bar.addWidget(self._radio_polar)
        bar.addSpacing(16)
        bar.addWidget(QLabel("Colormap:"))
        self._cmap_combo = QComboBox()
        for name in COLORMAPS:
            self._cmap_combo.addItem(name)
        self._cmap_combo.setCurrentText(DEFAULT_COLORMAP)
        self._cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        bar.addWidget(self._cmap_combo)
        bar.addSpacing(16)
        # Log/linear contrast toggle. When checked, the displayed image
        # is log10(clip(data, floor, inf)) and the histogram levels are
        # recomputed on the transformed array so the LUT stays sensible.
        # Coordinates and overlays are unaffected — only the intensity
        # mapping changes.
        self._log_check = QCheckBox("Log scale")
        self._log_check.setChecked(False)
        self._log_check.setToolTip(
            "Display log10(intensity) instead of linear intensity. "
            "Useful for GIWAXS data with wide dynamic range; coordinate "
            "axes and overlays are unchanged."
        )
        self._log_check.toggled.connect(self._on_log_toggled)
        bar.addWidget(self._log_check)
        bar.addSpacing(16)
        # Aspect ratio of the shown image. "Fit" (default) stretches the
        # frame to fill the panel (setAspectLocked(False)); "Custom" locks
        # the y-to-x data-unit scale to the spin value — 1.00 is the
        # undistorted view (equal Å⁻¹ per axis, so q-space rings stay
        # round in Cartesian mode). The lock lives on the PlotItem and so
        # survives mode switches and re-renders. See _on_aspect_changed.
        bar.addWidget(QLabel("Aspect:"))
        self._aspect_combo = QComboBox()
        self._aspect_combo.addItems(["Fit", "Default", "Custom"])
        self._aspect_combo.setToolTip(
            "Fit fills the panel; Default uses a per-mode shape "
            "(Cartesian 1:1, polar 2:1); Custom locks the width:height to "
            "the ratio box. Scrolling over an axis switches to Custom."
        )
        self._aspect_combo.currentIndexChanged.connect(self._on_aspect_changed)
        bar.addWidget(self._aspect_combo)
        self._aspect_spin = QDoubleSpinBox()
        self._aspect_spin.setRange(0.05, 20.0)
        self._aspect_spin.setSingleStep(0.05)
        self._aspect_spin.setDecimals(2)
        self._aspect_spin.setValue(1.0)
        self._aspect_spin.setToolTip(
            "Custom on-screen width:height of the image: >1 wider than "
            "tall, <1 taller than wide."
        )
        self._aspect_spin.valueChanged.connect(self._on_aspect_changed)
        bar.addWidget(self._aspect_spin)
        # The frame-navigation controls (prev / play / next / slider /
        # label) used to live in the Display dock's Frame row. They
        # now sit here in the toolbar so the user can scrub frames
        # regardless of which right-dock tab is in front. The host
        # injects them via ``insert_frame_controls`` after the
        # Display dock builds them — see MainWindow.
        bar.addStretch(1)
        # Keep a handle on the layout so the host can splice the
        # frame-navigation widgets in just before the trailing stretch.
        self._toolbar_layout = bar
        bar_widget = QWidget(self)
        bar_widget.setLayout(bar)
        outer.addWidget(bar_widget)

        self._plot = pg.PlotItem()
        self._view = pg.ImageView(self, view=self._plot)
        self._view.ui.roiBtn.hide()
        self._view.ui.menuBtn.hide()
        # Hide pyqtgraph's bottom timeline strip — redundant with the
        # Display-dock frame slider. ``setImage`` re-shows it via
        # ``roiClicked`` for any multi-frame stack, so we re-apply our
        # toggle state in ``_apply_params`` after every render.
        self._view.ui.roiPlot.hide()
        # Set the splitter handle width to 0 so even when the strip is
        # hidden there's no grey separator line eating into the image's
        # x-axis label area.
        self._view.ui.splitter.setHandleWidth(0)
        self._view.ui.splitter.setSizes([1, 0])
        # Pyqtgraph's GraphicsView occasionally lets the scene scroll
        # by a few pixels when its sceneRect has drifted from the
        # viewport size — kill scrollbars + frame so the plot is
        # unconditionally pinned inside its tab.
        gv = self._view.ui.graphicsView
        gv.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        gv.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        gv.setFrameStyle(QFrame.Shape.NoFrame)
        # Block QAbstractScrollArea-level scrolling without touching
        # ViewBox pan / zoom — see _disable_viewport_scroll docstring.
        _disable_viewport_scroll(gv)
        outer.addWidget(self._view)
        # Remember the user's contrast when they finish dragging the
        # histogram region, so it survives operation re-renders. Connect
        # to the LUT *item*; the widget forwards attribute access but not
        # the Qt signal object itself.
        self._view.getHistogramWidget().item.sigLevelChangeFinished.connect(
            self._on_contrast_changed
        )

        self._plot.invertY(False)
        self._plot.setAspectLocked(False)
        # PyQtGraph occasionally underestimates the bottom axis cell so
        # the axis label ("radius" / "q_xy") gets clipped by the
        # viewport's lower edge — and that clip is what creates the
        # small scrollable region. A small bottom layout margin gives
        # the label guaranteed clearance and keeps the plot fitted
        # inside its tab.
        self._plot.layout.setContentsMargins(0, 0, 0, 12)
        # Restore the persisted aspect choice now that both the toolbar
        # widgets and the plot exist. Starts in Fit, leaving the
        # setAspectLocked(False) baseline above untouched.
        self._restore_aspect_from_settings()
        # Route single-axis (axis-region) wheel events through our handler
        # so scrolling one axis adjusts the Custom ratio. Must come after
        # the viewbox exists.
        self._install_axis_wheel_handler()

        self._detected = _PeakShapeItem(**OVERLAY_STYLE["detected"])
        self._fitted = _PeakShapeItem(**OVERLAY_STYLE["fitted"])
        self._manual = _PeakShapeItem(**OVERLAY_STYLE["manual"])
        self._selection = _PeakShapeItem(**SELECTION_STYLE)
        self._fitted_preview = _PeakShapeItem(**FITTED_PREVIEW_STYLE)
        self._fitted_preview.setOpacity(FITTED_PREVIEW_OPACITY)
        vb = self._plot.getViewBox()
        vb.addItem(self._detected, ignoreBounds=True)
        vb.addItem(self._fitted, ignoreBounds=True)
        vb.addItem(self._manual, ignoreBounds=True)
        vb.addItem(self._selection, ignoreBounds=True)
        vb.addItem(self._fitted_preview, ignoreBounds=True)

        self._preview_item = pg.QtWidgets.QGraphicsRectItem()
        preview_pen = pg.mkPen(QColor("#ffeb3b"), width=1.0)
        preview_pen.setStyle(Qt.PenStyle.DashLine)
        preview_pen.setCosmetic(True)
        self._preview_item.setPen(preview_pen)
        self._preview_item.setBrush(QColor(255, 235, 59, 40))
        self._preview_item.setZValue(50)
        self._preview_item.setVisible(False)
        vb.addItem(self._preview_item, ignoreBounds=True)

        # Mouse handling lives in a Qt event filter on the graphics-view's
        # viewport — see _LabelEventFilter for why a ViewBox subclass doesn't work.
        self._label_filter = _LabelEventFilter(self._view.ui.graphicsView, vb, self)
        self._label_filter.install()
        self._label_filter.drawStarted.connect(self._on_draw_started)
        self._label_filter.drawUpdated.connect(self._on_draw_updated)
        self._label_filter.drawFinished.connect(self._on_draw_finished)
        self._label_filter.selectAt.connect(self._on_select_at)
        self._label_filter.doubleClicked.connect(self._reset_view_to_default)
        self._label_filter.cursorPos.connect(self._on_cursor_pos)
        self._label_filter.cursorLeft.connect(self._on_cursor_left)

        # Apply default colormap immediately.
        self._apply_cmap(DEFAULT_COLORMAP)

        # Need keyboard focus for the Delete shortcut to fire.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._frame_peaks: dict[int, dict[str, PeakTable | None]] = {}
        self._manual_peaks: dict[int, list[ManualPeak]] = {}
        self._visibility: dict[str, bool] = {kind: True for kind in OVERLAY_KINDS}

        # Matched-structure overlays. Variable count per frame (one per row in
        # each matched_* dataset), each with its own color and visibility.
        # Items in ``_matched_items`` belong to the *currently rendered* frame
        # only — they're torn down and rebuilt on frame change.
        self._matched_per_frame: dict[int, list[MatchedStructure]] = {}
        # Show/hide state keyed by a structure's CROSS-FRAME identity
        # (``MatchedStructure.color_key`` = CIF + hkl), so unchecking a
        # structure on one frame hides it on every frame it appears.
        # Defaults to True on first sight.
        self._matched_visibility: dict[tuple, bool] = {}
        # Stable colour index per structure identity, assigned first-seen
        # so a given (CIF, hkl) keeps its colour across the whole scan
        # (previously coloured by per-frame position, which shuffled as
        # the structure set changed frame to frame).
        self._matched_color_index: dict[tuple, int] = {}
        self._matched_master_visible: bool = True
        # How matched structures are drawn: "boxes" (default, per-peak
        # boxes) or "markers" (hollow circles for peaks + dashed arcs for
        # rings, like the q-map). A display preference, not reset on
        # file close.
        self._matched_display_style: str = "boxes"
        # ``unique_id``s currently hidden by the Display-dock search
        # filter. Independent of ``_matched_visibility`` (which
        # tracks checkbox state); the filter set overrides checkbox
        # state but doesn't mutate it, so clearing the filter
        # restores the previous user selection.
        self._matched_filter_hidden: set[str] = set()
        # Minimum detection score to render on the Detected overlay.
        # Driven by the Display dock's Min-score slider. 0.0 (the
        # default) lets every detected peak through; raising it
        # hides weak detections without mutating the file.
        self._detected_score_cutoff: float = 0.0
        # When not None, ONLY these fitted rows render, per frame
        # (``{frame: {fitted_id, ...}}``; frames absent from the mapping
        # show no fitted rows at all). Driven by the Scan-tracking
        # dock's "Show only tracked peaks" toggle, so the user can
        # verify track consistency by scrubbing. None = filter off.
        self._fitted_visible_only: dict[int, set[int]] | None = None
        self._matched_items: list[tuple[str, _PeakShapeItem]] = []
        # Identities whose tracked-peak subset is empty on the rendered
        # frame (kept hidden against later visibility refreshes).
        self._matched_empty_uids: set = set()

        self._mode = MODE_POLAR
        self._log_scale: bool = False
        # Raw-preview orientation flips, driven by the Conversion panel's
        # fliplr/flipud checkboxes. Default False (show the frame exactly as
        # stored); when set, the preview is flipped to match what pygid's
        # conversion will produce. Only the raw branch of _render_frame reads
        # these; the converted display is unaffected (its flips are already
        # baked in by pygid). See set_raw_flips and _apply_raw_flips.
        self._raw_flip_lr: bool = False
        self._raw_flip_ud: bool = False
        # The contrast (histogram) levels the user has dialed in with the
        # slider, or None to auto-contrast from the data. When set, it is
        # reused across operation re-renders (add-peak, pipeline reattach,
        # frame scrubs) so editing or running the pipeline doesn't snap the
        # contrast back to the robust default. Reset to None whenever the
        # underlying data changes (new stack/entry, log/linear toggle) so
        # those genuinely re-auto-contrast. Captured from the histogram's
        # sigLevelChangeFinished signal; ``setImage(levels=...)`` does not
        # fire that signal, so renders never masquerade as a user change.
        self._user_levels: tuple[float, float] | None = None
        # Guard: True while we push an image via ``setImage``, so the
        # histogram's ``sigLevelChangeFinished`` (which can fire when the
        # data range shifts, e.g. log toggle / new frame) is ignored and
        # only genuine user drags update ``_user_levels``.
        self._suppress_level_capture: bool = False
        self._stack: EntryStack | None = None
        # The FrameSource backing the active EntryStack. Owns the live
        # h5py handle + per-frame LRU; released across the silx detach
        # / reattach dance via release_frame_source / acquire_frame_source.
        self._frame_source: "FrameSource | None" = None  # type: ignore[name-defined]
        # The viewer drives frame indexing itself (the Display-dock slider
        # is the single source of truth). pyqtgraph used to track this via
        # ImageView.currentIndex, but we stopped feeding it 3D stacks once
        # large-file support landed — see the lazy-loading milestone.
        self._frame_index: int = 0
        # Last applied _DisplayParams so per-frame scrubs reuse pos/scale/
        # levels without rebuilding them.
        self._display_params: _DisplayParams | None = None
        # Raw-mode preview state. Held separately from ``_stack`` because raw
        # detector frames have no q-axes — they're rendered in pixel
        # coordinates and carry no overlays. Either an ndarray (small
        # stacks, tests) or a lazily-read file_model.LazyRawStack.
        self._raw_image_stack: "np.ndarray | object | None" = None
        # Polar grid axes derived from the FrameSource. Stored as a
        # (lazy_polar_stack, radius, angle) tuple so consumers
        # (profile viewer, cursor readout) get a single-frame index
        # without precomputing the full polar stack.
        self._polar_cache: "tuple[_LazyPolarStack, np.ndarray, np.ndarray] | None" = None  # type: ignore[name-defined]
        self._next_manual_id = -1  # negative IDs distinguish manual from detected
        self._selected: SelectedPeak | None = None
        # Secondary selections for the multi-select (Ctrl+click / Ctrl+A)
        # workflow. Restricted to ``kind == "detected"`` in this iteration
        # (the only kind in scope for copy/paste + batch fit). The primary
        # ``_selected`` can be any kind; extras are only ever appended via
        # ``_toggle_selected`` which gates on detected.
        self._selected_extras: list[SelectedPeak] = []
        self._roi_item: pg.ROI | None = None
        # Stacks of `_Action` objects. Pushing to undo clears redo. ROI drags
        # populate _roi_drag_before on sigRegionChangeStarted and consume it
        # on sigRegionChangeFinished — partial state never lands on the stack.
        self._undo_stack: list[_Action] = []
        self._redo_stack: list[_Action] = []
        self._roi_drag_before: tuple[float, float, float, float] | None = None
        # Set during a pipeline run so we don't allow concurrent ROI edits or
        # Delete keypresses while mlgidbase has the file open for writes.
        self._busy: bool = False

        # Geometry of the fitted-preview box for the current selection
        # (radius_center, fwhm_radial, angle_center, fwhm_angular). Cleared
        # whenever the selection isn't a manual / detected peak with valid
        # 1D fits.
        self._fitted_preview_geom: tuple[float, float, float, float] | None = None
        # When True, the preview is rendered as a ring (full angular sweep
        # at angle = 45°, angle_width = ∞) regardless of the angular fit —
        # mirrors what Add-to-fitted will write when the "Save fitted as
        # ring" toggle is on.
        self._fitted_preview_is_ring: bool = False
        # Cyan dashed previews for the Ctrl+click extras. Same shape
        # as ``_fitted_preview_geom`` but a list — one entry per extra
        # whose pygidfit result the host has computed. Always
        # segments (extras are detected, no ring sentinel).
        self._fitted_preview_extras_geoms: list[
            tuple[float, float, float, float]
        ] = []

        # NB: pyqtgraph's sigTimeChanged is no longer connected — we
        # stopped feeding ImageView 3D stacks once lazy frame loading
        # landed (see the lazy-loading milestone in
        # Documentation/). Frame changes now flow exclusively through
        # MainWindow → viewer.set_frame, and ``frameChanged`` is emitted
        # from there.

        # Add a "Reset zoom" action to the viewbox right-click menu so the
        # user can undo a manual zoom without leaving the keyboard / mouse.
        # pyqtgraph's default "View All" does the same thing under a less
        # discoverable label; this just adds the explicitly-named entry.
        self._install_reset_zoom_action()

    # -- Public API --

    def show_stack(self, stack: EntryStack, *, preserve_view: bool = False) -> None:
        """Render ``stack`` in the active mode.

        ``preserve_view`` keeps the current viewbox range and frame
        index across the re-render — used after pipeline ops and
        direct h5py edits where the underlying stack is identical and
        only the peak overlays changed. Default ``False`` (autorange)
        for the entry-switch / file-open paths, where the new stack
        typically has different axes.

        ``stack.image_stack`` must be a ``_LazyImageStack`` (i.e. the
        return value of ``file_model.load_entry``). The viewer extracts
        its backing ``FrameSource`` and stashes it on
        ``self._frame_source`` so frame reads can stream from disk
        without the full stack ever entering RAM.
        """
        # Capture before resetting cached state so the saved range is the
        # one the user is actually looking at right now.
        saved_xrange: tuple[float, float] | None = None
        saved_yrange: tuple[float, float] | None = None
        saved_frame: int | None = None
        if preserve_view and self._stack is not None:
            try:
                xr, yr = self._plot.getViewBox().viewRange()
                saved_xrange = (float(xr[0]), float(xr[1]))
                saved_yrange = (float(yr[0]), float(yr[1]))
            except Exception:
                logger.debug("suppressed exception in GIWAXSImageViewer.show_stack", exc_info=True)
                pass
            saved_frame = self.current_frame

        # Release any prior FrameSource before adopting the new one —
        # an open h5py handle on the previous temp file would block its
        # cleanup during session close on Windows.
        if (
            self._frame_source is not None
            and not isinstance(stack.image_stack, _LazyImageStack)
        ):
            self._frame_source.release()
            self._frame_source = None
        elif (
            self._frame_source is not None
            and isinstance(stack.image_stack, _LazyImageStack)
            and stack.image_stack.source is not self._frame_source
        ):
            self._frame_source.release()
            self._frame_source = None

        self._stack = stack
        if isinstance(stack.image_stack, _LazyImageStack):
            self._frame_source = stack.image_stack.source
        else:
            # Defensive fallback for tests that hand-build an EntryStack
            # with a raw ndarray. The viewer's hot path always goes
            # through _frame_source, so reject this case loudly.
            raise TypeError(
                "show_stack expects an EntryStack whose image_stack is "
                "a _LazyImageStack (i.e. produced by load_entry). Got: "
                f"{type(stack.image_stack).__name__}"
            )
        # ``_raw_image_stack`` belongs to a prior RawSession; clearing it
        # here ensures _render_active_mode never tries to re-render raw
        # pixel data over a NeXus stack.
        self._drop_raw_stack()
        # If the previous session was raw, the mode flag is still
        # ``MODE_RAW`` even though raw rendering ignored the radios.
        # Snap back to whichever Cartesian / Polar radio is checked
        # (Polar is the default at startup) so ``_render_active_mode``
        # takes the converted-data branch.
        if self._mode == MODE_RAW:
            self._mode = (
                MODE_CARTESIAN if self._radio_cart.isChecked() else MODE_POLAR
            )
        self._polar_cache = None
        self._frame_peaks.clear()
        # A fresh stack (entry change, file open) should re-auto-contrast;
        # a preserve_view re-render (peak edits, pipeline reattach) keeps
        # the user's slider contrast.
        if not preserve_view:
            self._user_levels = None
        # Clamp frame index to the new stack's range. Preserved when
        # the caller asked for it and the prior frame is still valid;
        # otherwise default to 0 so a fresh entry starts at the first
        # frame.
        if preserve_view and saved_frame is not None:
            self._frame_index = max(
                0, min(int(saved_frame), self._stack.n_frames - 1)
            )
        else:
            self._frame_index = 0
        # Skip the setImage autoRange entirely when we're preserving
        # the user's zoom — otherwise the viewbox flashes to the
        # full extent before the post-render setRange snaps it back,
        # which is visually jarring (and breaks if Qt schedules the
        # autoRange via a deferred signal that fires after setRange).
        # The post-render setRange below is the belt to this braces.
        self._render_active_mode(auto_range=not preserve_view)
        if preserve_view and saved_xrange is not None and saved_yrange is not None:
            self._plot.getViewBox().setRange(
                xRange=saved_xrange, yRange=saved_yrange, padding=0
            )

    def set_mode_radios_visible(self, visible: bool) -> None:
        """Show / hide the Cartesian / Polar radios in the top toolbar.

        Used by the host to remove mode controls when a raw session is
        active — raw frames don't carry q-axes, so the toggles would be
        nonsensical. The toolbar's "Colormap" + "Timeline" widgets stay
        visible because they apply equally to raw and converted data.
        """
        # Find the leading "View:" label by walking up from the radio's
        # parent layout — the label was added directly before the radios.
        for w in (self._radio_cart, self._radio_polar):
            w.setVisible(visible)
        # Hide the "View:" prefix label too. It lives in the same toolbar
        # row built by __init__; locate it by text rather than caching a
        # reference at construction time so existing layout code stays put.
        for label in self.findChildren(QLabel):
            if label.text() == "View:":
                label.setVisible(visible)
                break

    def show_raw_stack(self, arr_3d) -> None:
        """Render a raw detector stack in pixel coordinates.

        Used only for raw-mode (pre-conversion) preview. Wipes any prior
        NeXus-mode state — overlays, peaks, undo history — because none
        of it applies to a raw detector frame. The viewer's frame slider
        and timeline still drive frame navigation across the stack.

        Accepts an ndarray (small stacks, tests) or a
        ``file_model.LazyRawStack`` (the GUI's path — frames are read on
        demand, so a tens-of-GB beamtime stack never gets materialized;
        only ndarrays are copied contiguous here, a lazy stack would be
        materialized by ``ascontiguousarray``).
        """
        if getattr(arr_3d, "ndim", None) != 3:
            raise ValueError(
                f"show_raw_stack expects a 3D (N, H, W) array, "
                f"got shape {getattr(arr_3d, 'shape', None)}"
            )
        # Drop NeXus-mode state (peaks / matched / undo / cached polar).
        # ``clear()`` already covers everything except the raw-stack field
        # itself (and releases any prior lazy raw stack's handle).
        self.clear()
        self._mode = MODE_RAW
        if isinstance(arr_3d, np.ndarray):
            arr_3d = np.ascontiguousarray(arr_3d)
        self._raw_image_stack = arr_3d
        self._render_active_mode()

    def set_raw_flips(self, flip_lr: bool, flip_ud: bool) -> None:
        """Set the raw-preview orientation flips and re-render.

        Driven by the Conversion panel's fliplr/flipud checkboxes so the
        preview shows the raw image in the orientation the conversion will
        produce (see ``_apply_raw_flips``). The flags persist, so a raw
        stack loaded later is rendered with the same flips. Re-renders with
        ``auto_range=False`` so the user's zoom/pan survives the toggle.
        """
        flip_lr, flip_ud = bool(flip_lr), bool(flip_ud)
        if (flip_lr, flip_ud) == (self._raw_flip_lr, self._raw_flip_ud):
            return
        self._raw_flip_lr, self._raw_flip_ud = flip_lr, flip_ud
        if self._mode == MODE_RAW and self._raw_image_stack is not None:
            self._render_active_mode(auto_range=False)

    def _drop_raw_stack(self) -> None:
        """Forget the raw stack, closing a lazy stack's h5py handle.

        ndarray stacks have nothing to release; ``LazyRawStack`` holds a
        read handle on the raw file that must close before the session
        is dropped (Windows can't delete an open file; on any OS a
        leaked handle pins the LRU frames in memory).
        """
        stack = self._raw_image_stack
        self._raw_image_stack = None
        release = getattr(stack, "release", None)
        if release is not None:
            try:
                release()
            except Exception:
                logger.debug("suppressed release in _drop_raw_stack", exc_info=True)

    def reset_zoom(self) -> None:
        """Auto-fit the viewbox to the current image."""
        try:
            self._plot.getViewBox().autoRange()
        except Exception:
            logger.debug("suppressed exception in GIWAXSImageViewer.reset_zoom", exc_info=True)
            pass

    def apply_theme_colors(self, background, foreground) -> None:
        """Recolour the plot background + axes live for a theme switch.

        pyqtgraph bakes the configured background/foreground into items at
        creation (``pg.setConfigOption``), so an already-built plot keeps
        the old theme's colours until told otherwise — this pushes them
        immediately so a dark→light switch doesn't leave white axis text
        on a white background.
        """
        try:
            self._view.ui.graphicsView.setBackground(background)
        except Exception:
            logger.debug("suppressed exception setting viewer background", exc_info=True)
        pen = pg.mkPen(foreground)
        for name in ("left", "bottom", "right", "top"):
            try:
                ax = self._plot.getAxis(name)
            except Exception:
                ax = None
            if ax is None:
                continue
            try:
                ax.setPen(pen)
                ax.setTextPen(pen)
            except Exception:
                logger.debug("suppressed exception recolouring axis", exc_info=True)
        # The contrast/LUT histogram is its own GraphicsView (the
        # "contrast slider"), so it keeps the construction-time background
        # until told otherwise — recolour it and its level axis too.
        try:
            hist = self._view.getHistogramWidget()
            hist.setBackground(background)
            hist_axis = getattr(hist.item, "axis", None)
            if hist_axis is not None:
                hist_axis.setPen(pen)
                hist_axis.setTextPen(pen)
        except Exception:
            logger.debug("suppressed exception recolouring histogram", exc_info=True)

    # -- Aspect ratio (toolbar "Aspect:" Fit / Default / Custom) --

    _ASPECT_RATIO_MIN = 0.05
    _ASPECT_RATIO_MAX = 20.0
    # Per-mode width:height used by the "Default" preset.
    _ASPECT_DEFAULT_POLAR = 2.0
    _ASPECT_DEFAULT_OTHER = 1.0   # Cartesian + raw

    def aspect(self) -> tuple[str, float]:
        """Return the current aspect choice as ``(mode, ratio)``.

        ``mode`` is ``"fit"``, ``"default"`` or ``"custom"``; ``ratio`` is
        the target on-screen **width:height** of the image (``2`` = twice
        as wide as tall, the natural shape of a polar radius×angle map).
        For Default the ratio reported is the per-mode preset value.
        """
        txt = self._aspect_combo.currentText()
        mode = {"Default": "default", "Custom": "custom"}.get(txt, "fit")
        if mode == "default":
            return mode, self._default_ratio_for_mode()
        return mode, float(self._aspect_spin.value())

    def _default_ratio_for_mode(self) -> float:
        """Width:height the "Default" preset uses for the active mode:
        2:1 for polar (radius×angle), 1:1 for Cartesian / raw. Uses a
        getattr fallback because the aspect control is restored in
        ``__init__`` before ``_mode`` is assigned (the viewer starts
        polar)."""
        if getattr(self, "_mode", MODE_POLAR) == MODE_POLAR:
            return self._ASPECT_DEFAULT_POLAR
        return self._ASPECT_DEFAULT_OTHER

    def set_aspect(self, mode: str, ratio: float | None = None) -> None:
        """Programmatically set the aspect choice (tests / future wiring).
        Seeds the toolbar widgets without retriggering their signals, then
        applies + persists once via the shared path."""
        mode = str(mode).lower()
        label = {"default": "Default", "custom": "Custom"}.get(mode, "Fit")
        with QSignalBlocker(self._aspect_combo), QSignalBlocker(self._aspect_spin):
            self._aspect_combo.setCurrentText(label)
            if ratio is not None:
                self._aspect_spin.setValue(float(ratio))
        self._on_aspect_changed()

    def _data_extent(self) -> tuple[float, float] | None:
        """Width/height of the current image in *data* units (x, y), from
        the cached ``_DisplayParams``. None when no image is shown yet.

        This is what makes the ratio pixel-based rather than q-based: the
        lock value handed to pyqtgraph is derived from the image's actual
        extent, so a polar map (radius ~0-3 Å⁻¹ vs angle ~0-360°) is not
        collapsed into a sliver — see ``_apply_aspect``.
        """
        p = self._display_params
        if p is None:
            return None
        img = getattr(p, "image_pg", None)
        if img is None or getattr(img, "ndim", 0) < 2:
            return None
        nx, ny = img.shape[0], img.shape[1]   # pyqtgraph col-major: (x, y)
        sx, sy = p.scale
        dx = abs(nx * sx)
        dy = abs(ny * sy)
        if dx <= 0 or dy <= 0:
            return None
        return dx, dy

    def _live_box_ratio(self) -> float | None:
        """The image's *current* on-screen width:height, whatever the
        view state. ViewBox.getAspectRatio() returns
        ``a = (px per x-unit)/(px per y-unit)``; the image box is
        ``(Dx/Dy) * a``. Used to seed Custom when the user starts
        scrolling an axis so the number continues from what is shown."""
        ext = self._data_extent()
        if ext is None:
            return None
        try:
            a = float(self._plot.getViewBox().getAspectRatio())
        except Exception:
            return None
        if a <= 0:
            return None
        return (ext[0] / ext[1]) * a

    def _lock_to_ratio(self, ratio: float) -> None:
        """Lock the plot so the image box width:height == ``ratio``.

        pyqtgraph's ``setAspectLocked(True, a)`` locks ``a = (px per
        x-unit)/(px per y-unit)`` (see ViewBox.updateViewRange: "aspect is
        (widget w/h) / (view range w/h)"); the box is ``(Dx/Dy) * a``, so
        ``a = ratio * Dy / Dx`` from the current extent. Re-derived per
        call because Dx/Dy differ between Cartesian and polar, so the same
        ratio means the same on-screen shape in both. (The reciprocal
        collapsed polar — where Dy >> Dx — to a sliver.)
        """
        ext = self._data_extent()
        if ext is not None and ratio > 0:
            dx, dy = ext
            self._plot.setAspectLocked(True, ratio * dy / dx)
        else:
            # No image yet — defer; re-applied after the first render.
            self._plot.setAspectLocked(False)

    def _apply_aspect(self, refit: bool = False) -> None:
        """Push the current toolbar choice onto the plot. Fit frees the
        lock; Default and Custom lock to a width:height ratio (Default's is
        per-mode, reflected read-only in the spin)."""
        mode, ratio = self.aspect()
        try:
            if mode == "fit":
                self._plot.setAspectLocked(False)
            else:
                if mode == "default":
                    # Mirror the per-mode preset into the (disabled) spin.
                    with QSignalBlocker(self._aspect_spin):
                        self._aspect_spin.setValue(ratio)
                self._lock_to_ratio(ratio)
        except Exception:
            logger.debug("suppressed exception in GIWAXSImageViewer._apply_aspect", exc_info=True)
        if refit:
            self.reset_zoom()

    def _on_aspect_changed(self, *_: object) -> None:
        """Toolbar slot: enable the ratio spin only in Custom mode, apply
        the choice (refitting so the whole image is shown at the new
        shape), and persist the Custom ratio."""
        mode, ratio = self.aspect()
        self._aspect_spin.setEnabled(mode == "custom")
        self._apply_aspect(refit=True)
        if mode == "custom":
            QSettings().setValue(self._ASPECT_RATIO_KEY, ratio)

    def _reset_view_to_default(self) -> None:
        """Bare LMB double-click handler: snap the aspect to the per-mode
        **Default** preset. ``set_aspect`` refits, so this also restores the
        zoom to the full extent (replacing the old plain reset-zoom)."""
        self.set_aspect("default")

    def _restore_aspect_from_settings(self) -> None:
        """Seed the toolbar widgets in ``__init__``. The mode starts at
        **Default** (the per-mode preset); only the last Custom ratio is
        remembered so re-entering Custom is sensible."""
        s = QSettings()
        try:
            ratio = float(s.value(self._ASPECT_RATIO_KEY, 1.0))
        except (TypeError, ValueError):
            ratio = 1.0
        if not (self._ASPECT_RATIO_MIN <= ratio <= self._ASPECT_RATIO_MAX):
            ratio = 1.0
        with QSignalBlocker(self._aspect_combo), QSignalBlocker(self._aspect_spin):
            self._aspect_combo.setCurrentText("Default")
            self._aspect_spin.setValue(ratio)
            self._aspect_spin.setEnabled(False)
        self._apply_aspect(refit=False)

    def _install_axis_wheel_handler(self) -> None:
        """Route single-axis (axis-region) wheels through our handler.

        pyqtgraph forwards an axis-region wheel as
        ``viewbox.wheelEvent(ev, axis=0|1)`` (see ``AxisItem.wheelEvent``);
        a centre-of-image wheel reaches the ViewBox through Qt's C++ vtable
        and is untouched here (it zooms both axes, honouring any lock).
        """
        vb = self._plot.getViewBox()
        vb.wheelEvent = self._axis_wheel_event  # bound method; see docstring

    def _axis_wheel_event(self, ev, axis=None) -> None:
        """Scrolling over a single axis adjusts the aspect ratio and
        switches the control to **Custom**, seeded with the live shown
        ratio so the view continues smoothly. x widens (ratio up), y
        heightens (ratio down); the spin tracks it live."""
        vb = self._plot.getViewBox()
        if axis in (0, 1):
            base = self._live_box_ratio()
            if base is None:
                _, base = self.aspect()
            try:
                s = 1.02 ** (ev.delta() * vb.state["wheelScaleFactor"])
            except Exception:
                s = 1.0
            new_ratio = base / s if axis == 0 else base * s
            new_ratio = min(self._ASPECT_RATIO_MAX, max(self._ASPECT_RATIO_MIN, new_ratio))
            with QSignalBlocker(self._aspect_combo), QSignalBlocker(self._aspect_spin):
                self._aspect_combo.setCurrentText("Custom")
                self._aspect_spin.setValue(new_ratio)
                self._aspect_spin.setEnabled(True)
            self._apply_aspect(refit=False)   # restretch around current view
            QSettings().setValue(self._ASPECT_RATIO_KEY, new_ratio)
            ev.accept()
            return
        # Centre wheel routed here (axis is None): default both-axis zoom.
        pg.ViewBox.wheelEvent(vb, ev, axis)

    def _install_reset_zoom_action(self) -> None:
        vb = self._plot.getViewBox()
        menu = getattr(vb, "menu", None)
        if menu is None:
            return
        action = QAction("Reset zoom", menu)
        action.triggered.connect(self.reset_zoom)
        # Insert at the top so it lands above pyqtgraph's default entries.
        first = menu.actions()[0] if menu.actions() else None
        if first is not None:
            menu.insertAction(first, action)
            menu.insertSeparator(first)
        else:
            menu.addAction(action)
        self._reset_zoom_action = action  # keep a reference

    def set_peaks(self, frame: int, peaks: dict[str, PeakTable | None]) -> None:
        self._frame_peaks[frame] = peaks
        if frame == self.current_frame:
            self._render_overlays(frame)

    def insert_frame_controls(self, widgets: list[QWidget]) -> None:
        """Splice ``widgets`` into the toolbar just before the
        trailing stretch.

        The host (MainWindow) owns the frame-navigation widgets
        (previous, play, next, slider, label) so it can keep their
        signal wiring intact when re-parenting them. The trailing
        stretch added in ``__init__`` is **preserved** here — when
        the slider is hidden (single-frame stack) the stretch is
        the only expanding item left, which keeps the other toolbar
        controls clustered to the left instead of being spread out
        across the image width. When the slider is visible it
        carries a much larger stretch factor so it dominates and
        consumes the leftover horizontal space; the trailing
        stretch only kicks in when the slider's stretch
        contribution is zero (hidden widget).
        """
        bar = self._toolbar_layout
        # Insert in front of the trailing stretch (always the last
        # item from __init__). ``insert_at`` is updated after each
        # call so widgets land in the order given.
        insert_at = bar.count() - 1
        if insert_at < 0:
            insert_at = 0
        bar.insertSpacing(insert_at, 16)
        insert_at += 1
        for w in widgets:
            if isinstance(w, QSlider):
                # Slider gets a much larger stretch factor than the
                # trailing stretch so it eats the leftover width
                # when visible. When the slider is hidden, the
                # trailing stretch (factor 1) absorbs the space
                # alone and the other toolbar items stay packed
                # against the left edge.
                bar.insertWidget(insert_at, w, 100)
            else:
                bar.insertWidget(insert_at, w)
            insert_at += 1

    def set_overlay_visible(self, kind: str, visible: bool) -> None:
        if kind not in OVERLAY_KINDS:
            return
        self._visibility[kind] = visible
        item = self._overlay_item(kind)
        if item is not None:
            item.setVisible(visible)
        # Hiding the overlay that owns the current selection also clears the
        # selection highlight so it doesn't dangle.
        if (
            not visible
            and self._selected is not None
            and self._selected.kind == kind
        ):
            self.clear_selection()

    # -- Matched-structure API --

    def set_matched_structures(
        self, frame: int, structures: list[MatchedStructure]
    ) -> None:
        """Replace the list of matched structures for ``frame``.

        Visibility flags for previously-seen ``unique_id``s on this frame are
        preserved; new structures default to visible. Re-renders if ``frame``
        is the one currently shown.
        """
        self._matched_per_frame[frame] = list(structures)
        # Visibility is identity-keyed and persists across frames, so
        # there is nothing to prune here; new identities default to
        # visible via the ``.get(..., True)`` lookups. Assign each a
        # stable colour index up front so the swatch/overlay agree.
        for s in structures:
            self._color_index_for(s)
        if frame == self.current_frame:
            self._render_overlays(frame)
            self.matchedStructuresChanged.emit(frame, list(structures))

    def matched_structures(self, frame: int) -> list[MatchedStructure]:
        return list(self._matched_per_frame.get(frame, []))

    def matched_color(self, structure: MatchedStructure) -> str:
        """Return the hex color assigned to a structure on the current frame.
        Deterministic per insertion order within the frame so the Display
        panel and the overlay agree without extra plumbing.
        """
        return self.matched_pen(structure)["color"]

    def _color_index_for_key(self, key: tuple) -> int:
        """Stable palette index for a structure identity (``color_key``),
        assigned on first sight and reused thereafter so the colour is
        constant across frames and shared by the real overlay, the
        interpolated overlay, and the Display-dock swatch."""
        return self._matched_color_index.setdefault(
            key, len(self._matched_color_index)
        )

    def _color_index_for(self, structure: MatchedStructure) -> int:
        return self._color_index_for_key(structure.color_key)

    def seed_matched_colors(self, keys) -> None:
        """Pre-assign palette indices for ``keys`` (an ordered iterable of
        ``color_key`` tuples) so colours are deterministic regardless of
        which frame is rendered first. Idempotent: keys already assigned
        keep their index. The host seeds this from the full set of matched
        identities (sorted) when a scan is tracked, so a structure seen
        only via interpolation gets the same colour as on its real frames.
        """
        for key in keys:
            self._color_index_for_key(tuple(key))

    def matched_pen(self, structure: MatchedStructure) -> dict:
        """Return the full ``{color, style, width}`` pen for ``structure``.

        Keyed by the structure's cross-frame identity so the panel swatch
        and the overlay share one stable colour on every frame.
        """
        return matched_pen_for(self._color_index_for(structure))

    def matched_visibility(self, frame: int, unique_id: str) -> bool:
        """Show/hide state for the structure with ``unique_id`` on
        ``frame`` — resolved to its cross-frame identity so the state
        follows the structure between frames."""
        for s in self._matched_per_frame.get(frame, []):
            if s.unique_id == unique_id:
                return self._matched_visibility.get(s.color_key, True)
        return True

    def unmatched_visible(self) -> bool:
        """Checkbox state of the unmatched-fitted pseudo-group (defaults
        OFF — the fitted overlay already draws every fitted row)."""
        return bool(self._matched_visibility.get(UNMATCHED_KEY, False))

    def set_unmatched_visible(self, visible: bool) -> None:
        """Show/hide the unmatched-fitted pseudo-group. Persists across
        frames like a structure checkbox (identity-keyed)."""
        self._matched_visibility[UNMATCHED_KEY] = bool(visible)
        self._apply_matched_item_visibility()

    def has_unmatched_fitted(self, frame: int) -> bool:
        """True when ``frame`` has fitted rows not claimed by any
        matched structure (tracked-peak filter applied) — drives the
        Display dock's "Unmatched fitted peaks" row."""
        return self._unmatched_fitted_subset(frame) is not None

    def _unmatched_fitted_subset(self, frame: int):
        """Fitted rows on ``frame`` outside every matched structure's
        ``peaks.ids`` (minus rows the tracked-peak filter hides), or
        ``None`` when there are none."""
        fit = self._frame_peaks.get(frame, {}).get("fitted")
        if fit is None or len(fit) == 0:
            return None
        claimed: set = set()
        for s in self._matched_per_frame.get(frame, []):
            claimed.update(
                int(p) for p in np.asarray(s.peaks.ids, dtype=int)
            )
        kept = [
            int(pid) for pid in fit.ids
            if int(pid) not in claimed
            and not self._fitted_row_hidden(frame, int(pid))
        ]
        if not kept:
            return None
        if len(kept) == len(fit):
            return fit
        return _peaks_subset(fit, kept)

    def set_matched_display_style(self, style: str) -> None:
        """Switch the matched overlay between ``"boxes"`` and
        ``"markers"`` (hollow circles for peaks, dashed arcs/lines for
        rings — the q-map look). Re-renders the current frame; no other
        overlay is affected."""
        style = "markers" if style == "markers" else "boxes"
        if style == self._matched_display_style:
            return
        self._matched_display_style = style
        self._render_overlays(self.current_frame)

    def _matched_marker_items(self, peaks, color, extent) -> list:
        """Build q-map-style items for one structure's peaks: a hollow
        circle scatter for the spots and a dashed arc (Cartesian) /
        constant-radius line (polar) per ring. Returns pg items to add
        and track alongside the box items."""
        items: list = []
        if peaks is None or len(peaks) == 0:
            return items
        is_ring = np.asarray(peaks.is_ring, dtype=bool)
        spot = np.flatnonzero(~is_ring)
        if spot.size:
            r = np.asarray(peaks.radius, dtype=float)[spot]
            ang = np.asarray(peaks.angle, dtype=float)[spot]
            if self._mode == MODE_POLAR:
                xs, ys = r, ang
            else:
                a = np.deg2rad(ang)
                xs, ys = r * np.cos(a), r * np.sin(a)
            items.append(pg.ScatterPlotItem(
                x=xs, y=ys, symbol="o", size=MATCHED_MARKER_SIZE,
                pen=pg.mkPen(QColor(color), width=1.5), brush=None,
            ))
        # Ring and peak share the structure's colour; the SHAPE tells
        # them apart (a big dashed quarter-circle arc vs small circles),
        # so one well-distinguishable colour per structure is enough and
        # the eye isn't asked to separate two shades of the same hue.
        for i in np.flatnonzero(is_ring):
            clip = _clip_angle(
                float(peaks.angle[i]), float(peaks.angle_width[i]),
                extent=extent,
            )
            if clip is None:
                continue
            a_lo, a_hi = clip
            r = float(peaks.radius[i])
            pen = pg.mkPen(QColor(color), width=1.5, style=Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            if self._mode == MODE_POLAR:
                curve = pg.PlotCurveItem(
                    x=np.array([r, r]), y=np.array([a_lo, a_hi]), pen=pen,
                )
            else:
                th = np.linspace(np.deg2rad(a_lo), np.deg2rad(a_hi), 100)
                curve = pg.PlotCurveItem(
                    x=r * np.cos(th), y=r * np.sin(th), pen=pen,
                )
            items.append(curve)
        return items

    def _apply_matched_item_visibility(self) -> None:
        """Refresh visibility of every current-frame matched item
        (gated by ``_is_matched_item_visible``)."""
        for uid, item in self._matched_items:
            item.setVisible(self._is_matched_item_visible(uid))

    def set_matched_master_visible(self, visible: bool) -> None:
        self._matched_master_visible = visible
        self._apply_matched_item_visibility()

    def set_matched_structure_visible(self, unique_id: str, visible: bool) -> None:
        # Store against the structure's cross-frame identity so the
        # choice persists as the user scrubs frames.
        key = None
        for s in self._matched_per_frame.get(self.current_frame, []):
            if s.unique_id == unique_id:
                key = s.color_key
                break
        if key is None:
            return
        self._matched_visibility[key] = visible
        # Refresh every current-frame item (cheap; small list) so any
        # item sharing this identity updates too.
        self._apply_matched_item_visibility()

    def set_detected_score_cutoff(self, value: float) -> None:
        """Hide detected-overlay rows whose score is below ``value``.

        Re-renders overlays for the current frame so the change
        takes effect immediately. The cutoff is applied with a
        small float-epsilon so a slider at exactly the maximum
        score still shows that row (otherwise FP roundoff on stored
        scores can make ``score >= cutoff`` spuriously False).
        """
        self._detected_score_cutoff = float(value)
        self._render_overlays(self.current_frame)

    def set_fitted_visible_only(
        self, mapping: dict[int, set[int]] | None
    ) -> None:
        """Render ONLY the given fitted rows (``{frame: {id, ...}}``).

        Driven by the Scan-tracking dock's "Show only tracked peaks"
        toggle: pass the per-frame member ids of the surviving tracks
        and every other fitted row disappears from the overlay (frames
        not in the mapping show none), so scrubbing the scan reveals
        whether the tracks are visually consistent. Pass ``None`` to
        disable the filter entirely. Non-destructive render-time
        subset, like the Display dock's min-score cutoff; re-renders
        the current frame immediately.
        """
        if mapping is None:
            self._fitted_visible_only = None
        else:
            self._fitted_visible_only = {
                int(f): {int(i) for i in ids}
                for f, ids in mapping.items()
            }
        self._render_overlays(self.current_frame)

    def _fitted_row_hidden(self, frame: int, peak_id: int) -> bool:
        """True when the "show only tracked peaks" filter is active and
        this fitted row is not in the visible whitelist for ``frame``.

        Such a row is not rendered, so it must not be click- or
        Ctrl+A-selectable either — what you can't see, you can't select.
        Used by every fitted hit-test path in ``_on_select_at`` and by
        ``_select_all_of_kind_on_frame``.
        """
        vo = self._fitted_visible_only
        return vo is not None and int(peak_id) not in vo.get(int(frame), set())

    def set_matched_filter_hidden(self, hidden_ids) -> None:
        """Hide overlays for structures filtered out of the
        Display-dock search.

        Independent of ``set_matched_structure_visible`` (the
        checkbox-driven path): an empty filter set restores
        whatever the user's per-structure checkboxes say, while a
        non-empty set forces those ``unique_id``s off regardless of
        checkbox state. This lets the user search by CIF substring
        without losing their checkbox selections when the filter
        clears.
        """
        self._matched_filter_hidden = set(hidden_ids)
        for uid, item in self._matched_items:
            item.setVisible(self._is_matched_item_visible(uid))

    def _is_matched_item_visible(self, unique_id: str) -> bool:
        if not self._matched_master_visible:
            return False
        # The unmatched-fitted pseudo-group has no per-frame structure
        # to resolve; its own identity-keyed checkbox (default OFF)
        # decides, under the master toggle above.
        if unique_id == UNMATCHED_UID:
            return bool(self._matched_visibility.get(UNMATCHED_KEY, False))
        # Structures whose tracked-peak subset is empty on this frame were
        # rendered with nothing to draw; they must stay hidden even when a
        # later visibility refresh (e.g. the Display-dock filter) re-runs.
        if unique_id in self._matched_empty_uids:
            return False
        if unique_id in self._matched_filter_hidden:
            return False
        # Resolve the current-frame item to its cross-frame identity and
        # read the identity-keyed checkbox state.
        for s in self._matched_per_frame.get(self.current_frame, []):
            if s.unique_id == unique_id:
                return self._matched_visibility.get(s.color_key, True)
        return True

    def _overlay_item(self, kind: str) -> _PeakShapeItem | None:
        return {
            "detected": self._detected,
            "fitted":   self._fitted,
            "manual":   self._manual,
        }.get(kind)

    def set_mode(self, mode: str) -> None:
        if mode not in (MODE_CARTESIAN, MODE_POLAR) or mode == self._mode:
            return
        self._mode = mode
        if mode == MODE_POLAR:
            self._radio_polar.setChecked(True)
        else:
            self._radio_cart.setChecked(True)
        self._sync_roi()  # ROI exists only in polar mode
        self._render_active_mode()
        self.modeChanged.emit(mode)

    @property
    def mode(self) -> str:
        return self._mode

    def clear(self) -> None:
        self._view.clear()
        self._detected.clear_path()
        self._fitted.clear_path()
        self._manual.clear_path()
        self._selection.clear_path()
        self._fitted_preview.clear_path()
        self._fitted_preview_geom = None
        self._fitted_preview_extras_geoms = []
        self._frame_peaks.clear()
        self._manual_peaks.clear()
        self._fitted_visible_only = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        # Tear down all matched items and forget per-frame state.
        self._teardown_matched_items()
        self._matched_empty_uids = set()
        self._matched_per_frame.clear()
        self._matched_visibility.clear()
        self._matched_color_index.clear()
        had_selection = self._selected is not None
        self._selected = None
        self._selected_extras = []
        self._roi_drag_before = None
        self._sync_roi()
        # Release the long-lived h5py read handle before dropping the
        # stack reference — temp-dir cleanup at session close otherwise
        # fails on Windows because the file is still open.
        if self._frame_source is not None:
            self._frame_source.release()
            self._frame_source = None
        self._stack = None
        self._drop_raw_stack()
        self._polar_cache = None
        self._display_params = None
        self._user_levels = None
        self._frame_index = 0
        if had_selection:
            self.selectionChanged.emit(None)

    # -- File-handle coordination with the silx detach/reattach dance --

    def release_frame_source(self) -> None:
        """Close the FrameSource's h5py handle + clear its LRUs.

        Called by MainWindow._detach_silx_tree before any path that
        opens the same temp file ``r+`` (pipeline runs, ROI commit
        write-throughs, Add-to-fitted, clear-peaks, save-as).
        Idempotent — safe to call when no FrameSource is active.

        Also clears ``self._polar_cache`` so the cursor-readout
        handler's ``self._polar_cache is None`` guard fires while
        the FrameSource is closed. Without this drop, the cached
        ``_LazyPolarStack`` would still satisfy that None-check but
        its ``__getitem__`` would call into a released
        ``FrameSource`` and raise ``RuntimeError("FrameSource not
        acquired")`` every time the user moved the mouse across the
        polar plot during a pipeline run.
        """
        if self._frame_source is not None:
            self._frame_source.release()
        self._polar_cache = None

    def acquire_frame_source(self) -> None:
        """Reopen the FrameSource's h5py handle after a write completes.

        Pairs with ``release_frame_source`` via the silx reattach path.
        The polar cache is invalidated because per-frame polar arrays
        depend on the underlying Cartesian frames — and after a pipeline
        run those may have been overwritten. The polar grid axes
        (radius / angle) stay valid since they're functions of q_xy /
        q_z only, which don't change.

        Re-render runs with ``auto_range=False`` so the user's zoom
        survives the detach/reattach dance. This is the only call
        site for ``acquire_frame_source`` and it's always paired
        with ``release_frame_source`` for an op that the user
        triggered while looking at a specific area (pipeline ops,
        Add-to-fitted, clear-peaks…). Resetting the viewbox there
        is precisely the "zoom jumps to full extent after every
        commit" bug.
        """
        if self._frame_source is not None and not self._frame_source.is_open:
            self._frame_source.acquire()
            # Force a refresh of the polar cache wrapper so consumers
            # holding the previous tuple drop their reference.
            self._polar_cache = None
            # Re-render the current frame so the on-screen image
            # reflects any post-write changes to the underlying data.
            # _render_active_mode rebuilds the display params + LUT
            # but skips setImage's autoRange so the zoom survives.
            if self._stack is not None:
                self._render_active_mode(auto_range=False)

    # -- Manual peaks --

    def manual_peaks(self, frame: int) -> list[ManualPeak]:
        return list(self._manual_peaks.get(frame, []))

    def add_manual_peak(self, frame: int, peak: ManualPeak) -> None:
        self._undoable_add_manual(frame, peak)
        self._push_undo(ManualAddAction(frame=frame, peak=peak))

    def remove_manual_peak(self, frame: int, peak: ManualPeak) -> None:
        if peak not in self._manual_peaks.get(frame, []):
            return
        self._undoable_remove_manual(frame, peak)
        self._push_undo(ManualRemoveAction(frame=frame, peak=peak))

    def angular_extent(self) -> tuple[float, float] | None:
        """Return ``(angle_min_deg, angle_max_deg)`` for the active polar
        stack. Used by the host to size ring-mode expansions to the
        actual displayed angular range — converted files vary
        (``[0, 90]`` for the upper-right quadrant; ``[-180, 180]`` for
        full-quadrant data). Returns None when no polar stack is
        currently rendered (raw mode, no file open).
        """
        if self._polar_cache is None:
            return None
        _, _, angle = self._polar_cache
        if angle.size == 0:
            return None
        return float(angle[0]), float(angle[-1])

    def set_manual_geometry(
        self,
        peak: ManualPeak,
        radius: float,
        angle: float,
        radius_width: float,
        angle_width: float,
        is_ring: bool,
    ) -> None:
        """Mutate every geometry field on ``peak`` (including ``is_ring``)
        and trigger the standard refresh path. Skips the undo stack —
        this is for transient state changes driven by UI toggles
        (e.g. the ring checkbox), not user-initiated edits that should
        be reversible via Ctrl+Z. The host stashes its own pre-state
        when it needs to revert.

        Mirrors `_apply_manual_geom` but adds `is_ring` so we can flip
        the ring/segment kind in lockstep with the angular sweep.
        """
        peak.radius = radius
        peak.angle = angle
        peak.radius_width = radius_width
        peak.angle_width = angle_width
        peak.is_ring = is_ring
        if (
            self._selected is not None
            and self._selected.kind == "manual"
            and self._selected.manual_ref is peak
        ):
            self._selected.radius = radius
            self._selected.angle = angle
            self._selected.radius_width = radius_width
            self._selected.angle_width = angle_width
            self._selected.is_ring = is_ring
            self._sync_roi_geometry()
        # Find the frame this peak lives on so the overlay refreshes
        # against the right bucket.
        for fr, peaks in self._manual_peaks.items():
            if peak in peaks:
                if fr == self.current_frame:
                    self._render_overlays(fr)
                break
        if (
            self._selected is not None
            and self._selected.kind == "manual"
            and self._selected.manual_ref is peak
        ):
            self.peakGeometryChanged.emit(self._selected)

    def commit_manual_peak(self, frame: int, peak: ManualPeak) -> None:
        """Drop a manual peak that has been persisted to the NeXus file.

        Like ``remove_manual_peak`` but does not push to the undo stack — the
        peak now lives in the detected/fitted overlay, so undoing back to its
        manual state would resurrect a duplicate. Any existing undo/redo
        entries referencing this peak are scrubbed for the same reason.
        """
        peaks = self._manual_peaks.get(frame, [])
        if peak in peaks:
            peaks.remove(peak)
        self._undo_stack = [a for a in self._undo_stack if not _action_targets_manual(a, peak)]
        self._redo_stack = [a for a in self._redo_stack if not _action_targets_manual(a, peak)]
        was_selected = (
            self._selected is not None
            and self._selected.kind == "manual"
            and self._selected.manual_ref is peak
        )
        if was_selected:
            self._selected = None
            self._sync_roi()
        if frame == self.current_frame:
            self._render_overlays(frame)
        if was_selected:
            self.selectionChanged.emit(None)
        self.manualPeakRemoved.emit(frame, peak)

    def undo_last_action(self) -> None:
        """Reverse the most recent action. No-ops if empty."""
        if self._busy or not self._undo_stack:
            return
        action = self._undo_stack.pop()
        action.undo(self)
        self._redo_stack.append(action)

    def redo_last_action(self) -> None:
        """Re-apply the most recently undone action."""
        if self._busy or not self._redo_stack:
            return
        action = self._redo_stack.pop()
        action.redo(self)
        self._undo_stack.append(action)

    def clear_history(self) -> None:
        """Drop both undo and redo stacks. Called after pipeline ops that
        regenerate peak ids — pending FileGeomActions would key off stale ids.

        For ordinary peak edits (delete / paste / fit), prefer
        ``prune_file_geom_history`` instead: file-resident ids are stable
        (``add_*`` appends ``max(id)+1``, ``delete`` filters by id and
        leaves the rest), so only the *removed* ids' geometry edits go
        stale — wiping the whole history would needlessly break
        multi-level undo/redo.
        """
        self._undo_stack.clear()
        self._redo_stack.clear()

    def prune_file_geom_history(
        self, frame: int, kind: str, peak_ids
    ) -> None:
        """Drop only the FileGeomActions that target ``(frame, kind, id)``
        for ``id in peak_ids`` from both stacks, keeping everything else.

        Used after a delete: the removed ids' pending geometry edits would
        key off rows that no longer exist (and would otherwise raise a
        stale-id KeyError on the deferred write). Unlike ``clear_history``
        this preserves prior undoable operations, so a run of deletes (or
        deletes interleaved with pastes / fits) stays fully undoable and
        redoable.
        """
        targets = {int(p) for p in peak_ids}

        def _stale(a: _Action) -> bool:
            return (
                isinstance(a, FileGeomAction)
                and int(a.frame) == int(frame)
                and a.kind == kind
                and int(a.peak_id) in targets
            )

        self._undo_stack = [a for a in self._undo_stack if not _stale(a)]
        self._redo_stack = [a for a in self._redo_stack if not _stale(a)]

    def clear_selection(self) -> None:
        if self._selected is None and not self._selected_extras:
            return
        self._selected = None
        self._selected_extras = []
        self._fitted_preview_geom = None
        self._fitted_preview_is_ring = False
        self._fitted_preview_extras_geoms = []
        self._sync_roi()
        self._render_overlays(self.current_frame)
        self.selectionChanged.emit(None)
        self.selectionsChanged.emit([])

    def clear_all_manual_peaks(self) -> None:
        """Drop every manual peak across all frames + the undo history.

        Matches Tools → Clear all manual peaks. Manual peaks are
        in-memory only, so no file write is involved. The selection is
        also cleared if it pointed at a manual peak.
        """
        if not self._manual_peaks:
            # Still clear undo history of any orphaned ManualGeomActions
            # and refresh in case overlays drift.
            self._undo_stack.clear()
            self._redo_stack.clear()
            return
        self._manual_peaks.clear()
        if self._selected is not None and self._selected.kind == "manual":
            self._selected = None
            self._fitted_preview_geom = None
            self._fitted_preview_is_ring = False
            self._sync_roi()
            self.selectionChanged.emit(None)
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._render_overlays(self.current_frame)

    def set_fitted_preview(
        self,
        center_r: float | None,
        width_r: float | None,
        center_a: float | None,
        width_a: float | None,
        *,
        is_ring: bool = False,
    ) -> None:
        """Paint the dashed cyan preview of the would-be fitted_peaks box.

        Pure painter: the box is drawn at ``(width_r × width_a)``
        verbatim, centred at ``(center_r, center_a)``. All convention
        math (``2σ_r × 2σ_a`` for both modes; fallback to the user's
        drawn box when scipy's 1D fit hasn't converged) lives in the
        caller (``MainWindow._update_fitted_preview``) so a single
        place owns the rules.

        Pass any of ``center_r`` / ``width_r`` as ``None`` to hide.
        Pass ``is_ring=True`` to draw a full-angular-sweep ring at the
        canonical ``angle = 45°, angle_width = ∞`` — the caller still
        passes a sentinel for ``center_a`` / ``width_a`` (they're
        ignored).
        """
        if (
            center_r is None or width_r is None
            or not (np.isfinite(center_r) and np.isfinite(width_r) and width_r > 0)
        ):
            self._fitted_preview_geom = None
            self._fitted_preview_is_ring = False
        elif is_ring:
            # Ring path: angular fit isn't required (or even meaningful)
            # — store sentinel angular values that _render_overlays
            # rewrites to (45°, ∞) when it builds the preview row.
            self._fitted_preview_geom = (
                float(center_r), float(width_r), 45.0, 0.0,
            )
            self._fitted_preview_is_ring = True
        elif (
            center_a is None or width_a is None
            or not (np.isfinite(center_a) and np.isfinite(width_a) and width_a > 0)
        ):
            self._fitted_preview_geom = None
            self._fitted_preview_is_ring = False
        else:
            self._fitted_preview_geom = (
                float(center_r), float(width_r),
                float(center_a), float(width_a),
            )
            self._fitted_preview_is_ring = False
        self._render_overlays(self.current_frame)

    def set_fitted_preview_extras(
        self,
        geoms: list[tuple[float, float, float, float]],
    ) -> None:
        """Paint cyan dashed previews for every Ctrl+click extra.

        ``geoms`` is a list of ``(radius, radius_width, angle,
        angle_width)`` tuples — one per extra peak whose pygidfit
        result the host has already computed (or fallen back from).
        Pass ``[]`` to clear. Each tuple paints the same way the
        primary preview does (no ring support for extras: extras are
        detected peaks, which never use the ring sentinel).

        Stored alongside ``_fitted_preview_geom`` (which still owns
        the primary peak's preview) so the renderer can paint N
        boxes from a single ``_PeakShapeItem``.
        """
        cleaned: list[tuple[float, float, float, float]] = []
        for g in geoms:
            try:
                cr, wr, ca, wa = (float(g[0]), float(g[1]), float(g[2]), float(g[3]))
            except (TypeError, ValueError, IndexError):
                continue
            if not (
                np.isfinite(cr) and np.isfinite(wr) and wr > 0
                and np.isfinite(ca) and np.isfinite(wa) and wa > 0
            ):
                continue
            cleaned.append((cr, wr, ca, wa))
        if cleaned == self._fitted_preview_extras_geoms:
            return
        self._fitted_preview_extras_geoms = cleaned
        self._render_overlays(self.current_frame)

    def set_busy(self, busy: bool) -> None:
        """Disable interactive editing while a pipeline run is in flight."""
        self._busy = busy
        self._sync_roi()

    @property
    def current_frame(self) -> int:
        return self._frame_index

    @property
    def n_frames(self) -> int:
        """Number of frames in the active stack (0 if no stack loaded)."""
        if self._mode == MODE_RAW and self._raw_image_stack is not None:
            return int(self._raw_image_stack.shape[0])
        return 0 if self._stack is None else int(self._stack.n_frames)

    def set_frame(self, frame: int) -> None:
        """Seek to ``frame``.

        The single source of truth for frame changes. Reads the new
        frame from the active source (FrameSource for NeXus, the
        in-memory raw stack for raw mode), pushes the 2D image to
        pyqtgraph via ``setImage`` (with the cached pos / scale /
        levels so the LUT and viewbox stay stable), updates the
        per-frame overlays, and emits ``frameChanged`` plus
        ``matchedStructuresChanged`` exactly once.

        Also prunes any selected peaks that belong to a different
        frame so the white highlight overlay doesn't paint stale
        rectangles at coordinates that don't correspond to a real
        peak on the new frame (typical after a copy/paste workflow
        where the source-frame selection lingers).

        No-op when ``frame`` is already current or out of range.
        """
        n = self.n_frames
        if n == 0:
            return
        idx = max(0, min(int(frame), n - 1))
        if idx == self._frame_index:
            return
        self._frame_index = idx
        self._prune_off_frame_selections(idx)
        self._render_frame(idx, auto_range=False)
        self._render_overlays(idx)
        self.frameChanged.emit(idx)
        # The Display dock rebuilds its matched-structure rows from this
        # signal — different frames can have different solutions.
        self.matchedStructuresChanged.emit(idx, self.matched_structures(idx))

    def _prune_off_frame_selections(self, frame: int) -> None:
        """Drop any selected peaks whose ``.frame`` is not ``frame``.

        Called when the user navigates between frames so the white
        highlight overlay only paints peaks that actually exist on
        the visible frame. Manual peaks are exempt because their
        ``.frame`` is the frame they were drawn on and the manual-
        peak rendering already gates on ``self.current_frame``;
        pruning them here would erase their selection across
        navigation which is the wrong default. Detected / fitted /
        matched selections are frame-bound and get pruned.

        Emits ``selectionChanged`` and ``selectionsChanged`` once
        if anything actually changed.
        """
        def _on_frame(s: SelectedPeak) -> bool:
            # Manual peaks survive a frame switch — they have a
            # frame field but their renderer is frame-aware and the
            # user's draw intent is "this one specific peak".
            if s.kind == "manual":
                return True
            return int(s.frame) == int(frame)

        prev_primary = self._selected
        prev_extras = list(self._selected_extras)
        # Filter extras first.
        self._selected_extras = [s for s in self._selected_extras if _on_frame(s)]
        # If the primary doesn't belong here, promote the first
        # surviving extra (or clear).
        if self._selected is not None and not _on_frame(self._selected):
            if self._selected_extras:
                self._selected = self._selected_extras.pop(0)
            else:
                self._selected = None
        if (
            self._selected is prev_primary
            and self._selected_extras == prev_extras
        ):
            return
        self._sync_roi()
        self.selectionChanged.emit(self._selected)
        self.selectionsChanged.emit(self.selected_peaks())

    @property
    def selected_peak(self) -> SelectedPeak | None:
        return self._selected

    def selected_peaks(self) -> list[SelectedPeak]:
        """Full multi-selection list (primary + extras), or empty when
        nothing is selected.

        Backs the Ctrl+C copy handler and the batch-fit button enable
        check on the host. Order is stable: primary first, extras in
        the order they were Ctrl+click-added.
        """
        if self._selected is None:
            return []
        return [self._selected, *self._selected_extras]

    @property
    def is_dragging(self) -> bool:
        """True while an image-side ROI handle drag is in progress.

        Exposed so the host can decide whether per-tick
        ``peakGeometryChanged`` signals come from a live drag (and
        deserve debounced expensive recompute) or a settled
        programmatic update.
        """
        return self._roi_drag_before is not None

    # -- Action helpers (used by both public API and undo/redo) --

    def _push_undo(self, action: _Action) -> None:
        self._undo_stack.append(action)
        self._redo_stack.clear()

    def _undoable_add_manual(self, frame: int, peak: ManualPeak) -> None:
        """Insert a manual peak without touching the undo stack.

        Single-box policy invariant: whenever a manual peak is on
        screen it is also the active selection. This applies to every
        add path — the user-draw flow already auto-selected before;
        now undo of a remove (which restores the manual box) and redo
        of an add do too. Skipped when the peak is added on a non-
        current frame because the user can't see / interact with it.
        """
        bucket = self._manual_peaks.setdefault(frame, [])
        if peak not in bucket:
            bucket.append(peak)
        if frame == self.current_frame:
            self._render_overlays(frame)
        self.manualPeakAdded.emit(frame, peak)
        if frame == self.current_frame:
            self._set_selected(SelectedPeak.from_manual(peak, frame))

    def _undoable_remove_manual(self, frame: int, peak: ManualPeak) -> None:
        """Remove a manual peak without touching the undo stack."""
        bucket = self._manual_peaks.get(frame, [])
        if peak in bucket:
            bucket.remove(peak)
        was_selected = (
            self._selected is not None
            and self._selected.kind == "manual"
            and self._selected.manual_ref is peak
        )
        if was_selected:
            self._selected = None
            self._sync_roi()
        if frame == self.current_frame:
            self._render_overlays(frame)
        if was_selected:
            self.selectionChanged.emit(None)
        self.manualPeakRemoved.emit(frame, peak)

    def _apply_manual_geom(
        self, frame: int, peak: ManualPeak,
        polar: tuple[float, float, float, float],
    ) -> None:
        r, a, dr, da = polar
        peak.radius = r
        peak.angle = a
        peak.radius_width = dr
        peak.angle_width = da
        # If this peak is the active selection, mirror it on the SelectedPeak
        # snapshot and refresh the ROI without retriggering its signals.
        if (
            self._selected is not None
            and self._selected.kind == "manual"
            and self._selected.manual_ref is peak
        ):
            self._selected.radius = r
            self._selected.angle = a
            self._selected.radius_width = dr
            self._selected.angle_width = da
            self._sync_roi_geometry()
        if frame == self.current_frame:
            self._render_overlays(frame)
        if (
            self._selected is not None
            and self._selected.kind == "manual"
            and self._selected.manual_ref is peak
        ):
            self.peakGeometryChanged.emit(self._selected)

    def _apply_file_geom(
        self, frame: int, kind: str, peak_id: int,
        polar: tuple[float, float, float, float],
    ) -> None:
        r, a, dr, da = polar
        # Update the in-memory PeakTable so overlays paint the new box right
        # away; the disk write is fired separately via peakRowWriteRequested.
        peaks_for_frame = self._frame_peaks.get(frame) or {}
        table = peaks_for_frame.get(kind)
        if table is not None and len(table) > 0:
            matches = np.where(table.ids == peak_id)[0]
            if matches.size > 0:
                idx = int(matches[0])
                table.radius[idx] = r
                table.angle[idx] = a
                table.radius_width[idx] = dr
                table.angle_width[idx] = da
                table.q_xy[idx], table.q_z[idx] = polar_to_qxyz(r, a)
        # If the user edited a fitted peak, every matched solution that
        # references it must re-slice from the updated fitted table.
        if kind == "fitted":
            self._refresh_matched_for(frame)
        # Reflect the change on the SelectedPeak if it's the active selection.
        if (
            self._selected is not None
            and self._selected.kind in (kind, "matched")
            and self._selected.frame == frame
            and self._selected.peak_id == peak_id
        ):
            self._selected.radius = r
            self._selected.angle = a
            self._selected.radius_width = dr
            self._selected.angle_width = da
            self._sync_roi_geometry()
        if frame == self.current_frame:
            self._render_overlays(frame)
        # Fire the file-write so undo/redo also persist.
        self.peakRowWriteRequested.emit(
            frame, kind, int(peak_id),
            {"radius": r, "angle": a, "radius_width": dr, "angle_width": da},
        )
        if (
            self._selected is not None
            and self._selected.peak_id == peak_id
            and self._selected.frame == frame
        ):
            self.peakGeometryChanged.emit(self._selected)

    def _refresh_matched_for(self, frame: int) -> None:
        """Re-slice the frame's fitted PeakTable into each MatchedStructure
        using its cached ``peak_list`` indices. Cheap (numpy fancy index).
        """
        peaks_for_frame = self._frame_peaks.get(frame) or {}
        fitted = peaks_for_frame.get("fitted")
        structures = self._matched_per_frame.get(frame, [])
        if fitted is None or not structures:
            return
        n_fit = len(fitted)
        for s in structures:
            idx = s.peak_list
            idx = idx[(idx >= 0) & (idx < n_fit)]
            s.peaks = PeakTable(
                q_xy=fitted.q_xy[idx],
                q_z=fitted.q_z[idx],
                angle=fitted.angle[idx],
                radius=fitted.radius[idx],
                angle_width=fitted.angle_width[idx],
                radius_width=fitted.radius_width[idx],
                is_ring=fitted.is_ring[idx],
                ids=fitted.ids[idx],
                score=fitted.score[idx],
                amplitude=fitted.amplitude[idx],
            )

    def polar_data(
        self,
    ) -> "tuple[_LazyPolarStack, np.ndarray, np.ndarray] | None":
        """Return ``(polar_stack, radius, angle)`` for the active entry.

        ``polar_stack`` is a ``_LazyPolarStack`` — supports
        ``polar_stack[i]`` (one polar frame) and
        ``polar_stack[frame, r, a]`` (single-pixel cursor lookup).
        Per-frame resampling happens on demand inside the
        ``FrameSource``; nothing is precomputed eagerly.

        Returns None if no stack is currently loaded or the
        FrameSource is currently released (e.g. mid-pipeline silx
        detach). Callers should re-call after the reattach.
        """
        if self._stack is None or self._frame_source is None:
            return None
        if not self._frame_source.is_open:
            return None
        if self._polar_cache is None:
            radius, angle = self._frame_source.polar_axes()
            self._polar_cache = (
                _LazyPolarStack(self._frame_source), radius, angle,
            )
        return self._polar_cache

    # -- Rendering --

    def _current_levels(self) -> tuple[float, float] | None:
        """The contrast levels currently on the histogram, or None when
        unreadable / degenerate. Used to snapshot the user's slider."""
        try:
            lo, hi = self._view.getHistogramWidget().getLevels()
        except Exception:
            logger.debug("suppressed exception in GIWAXSImageViewer._current_levels", exc_info=True)
            return None
        if lo is None or hi is None:
            return None
        lo, hi = float(lo), float(hi)
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            return None
        return lo, hi

    def _on_contrast_changed(self, *args) -> None:
        """Histogram drag finished — remember the user's contrast so it
        sticks across operation re-renders and frame scrubs.

        ``setImage(levels=...)`` (every render) does NOT emit
        ``sigLevelChangeFinished``, so this only fires on a genuine user
        drag, never on our own renders — no suppression flag needed.
        """
        if self._suppress_level_capture:
            return
        lv = self._current_levels()
        if lv is not None:
            self._user_levels = lv

    def _render_active_mode(self, *, auto_range: bool = True) -> None:
        """Build per-mode axis state + first-frame levels, then render.

        Called on stack swap (entry change, file open), mode toggle
        (Cartesian ↔ polar), and log/linear toggle. The actual per-
        frame pixel push happens in ``_render_frame`` which is also
        the path ``set_frame`` takes on every scrub.

        ``auto_range`` defaults to True so cold-open / mode-swap
        renders fit the new image into the viewbox. Callers that
        preserve the user's prior zoom (``show_stack(preserve_view=True)``,
        the log-scale toggle) pass ``False`` so ``setImage`` never
        clobbers the viewbox in the first place.
        """
        if self._mode == MODE_RAW:
            if self._raw_image_stack is None:
                return
            self._display_params = self._build_raw_params()
            # Re-derive the aspect lock for this mode's extents before the
            # render's autoRange so the locked shape is honoured.
            self._apply_aspect(refit=False)
            self._render_frame(self._frame_index, auto_range=auto_range)
            # No overlays in raw mode — nothing to render past _render_frame.
            return
        # ``_build_*_params`` read frame 0 through the FrameSource to
        # compute robust levels, so bail when the source is released by
        # the silx detach/reattach dance. Without this, toggling the
        # Cartesian/Polar radio during a write (pipeline run, ROI
        # commit, Add-to-fitted, clear-peaks, save-as) drives
        # ``get_cartesian(0)`` / ``get_polar(0)`` into a closed handle
        # and raises ``RuntimeError("FrameSource not acquired")``. The
        # new mode is still recorded; ``acquire_frame_source`` re-renders
        # it once the handle reopens.
        if (
            self._stack is None
            or self._frame_source is None
            or not self._frame_source.is_open
        ):
            return
        if self._mode == MODE_POLAR:
            self._display_params = self._build_polar_params()
        else:
            self._display_params = self._build_cartesian_params()
        # Re-derive the aspect lock for this mode's extents before the
        # render's autoRange so the locked shape is honoured (the same
        # ratio means the same on-screen shape in Cartesian and polar).
        self._apply_aspect(refit=False)
        self._render_frame(self._frame_index, auto_range=auto_range)
        self._render_overlays(self.current_frame)

    def _build_raw_params(self) -> _DisplayParams:
        """Pixel-coordinate axis state for a raw detector stack.

        File order is (frames, H, W). For pyqtgraph we want each frame
        displayed as (W, H) — see ``_render_frame``'s raw branch for
        the per-frame transpose. Axes are labelled in pixels; q
        coordinates aren't meaningful before conversion.
        """
        assert self._raw_image_stack is not None
        # ``image_pg`` here is just the *first* frame, used for level
        # computation. Per-frame pushes happen in ``_render_frame``.
        first = np.asarray(self._raw_image_stack[0]).T  # (W, H)
        levels = _robust_levels(first)
        return _DisplayParams(
            image_pg=first,
            pos=(0.0, 0.0),
            scale=(1.0, 1.0),
            levels=levels,
            x_label=("x", "px"),
            y_label=("y", "px"),
        )

    def _build_cartesian_params(self) -> _DisplayParams:
        """Cartesian axis state. Reads frame 0 via the FrameSource to
        compute robust intensity levels; per-frame rendering is lazy.
        """
        assert self._stack is not None and self._frame_source is not None
        q_xy = self._frame_source.q_xy
        q_z = self._frame_source.q_z
        x0 = float(q_xy[0]); y0 = float(q_z[0])
        sx = float(q_xy[-1] - q_xy[0]) / max(len(q_xy) - 1, 1)
        sy = float(q_z[-1] - q_z[0]) / max(len(q_z) - 1, 1)
        first = self._frame_source.get_cartesian(0).T  # (n_qxy, n_qz)
        levels = _robust_levels(first)
        return _DisplayParams(
            image_pg=first,
            pos=(x0, y0),
            scale=(sx, sy),
            levels=levels,
            x_label=("q_xy", "Å⁻¹"),
            y_label=("q_z", "Å⁻¹"),
        )

    def _build_polar_params(self) -> _DisplayParams:
        """Polar axis state + lazy polar wrapper.

        The polar grid (``radius`` / ``angle``) is derived once from
        ``q_xy`` / ``q_z`` and reused across frames; per-frame polar
        resampling happens on demand in ``FrameSource.get_polar``.
        """
        assert self._stack is not None and self._frame_source is not None
        radius, angle = self._frame_source.polar_axes()
        # Cache a (lazy_polar_stack, radius, angle) tuple so consumers
        # (profile viewer, cursor readout) share the same FrameSource-
        # backed view without re-creating wrappers.
        if self._polar_cache is None:
            self._polar_cache = (
                _LazyPolarStack(self._frame_source), radius, angle,
            )
        x0 = float(radius[0]); y0 = float(angle[0])
        sx = float(radius[-1] - radius[0]) / max(len(radius) - 1, 1)
        sy = float(angle[-1] - angle[0]) / max(len(angle) - 1, 1)
        # polar_stack frame layout is (n_radius, n_angle); pyqtgraph
        # wants (x=radius, y=angle) per frame, which already matches.
        first = self._frame_source.get_polar(0)
        levels = _robust_levels(first)
        return _DisplayParams(
            image_pg=first,
            pos=(x0, y0),
            scale=(sx, sy),
            levels=levels,
            x_label=("radius", "Å⁻¹"),
            y_label=("angle", "deg"),
        )

    def _render_frame(self, idx: int, *, auto_range: bool) -> None:
        """Push the 2D frame at ``idx`` to pyqtgraph.

        Uses the cached ``_display_params`` (built once per stack/mode
        swap) so per-frame scrubs don't rebuild axes. ``auto_range`` is
        True only on the initial render after a stack/mode change;
        every subsequent frame change passes False to preserve the
        user's zoom/pan.
        """
        if self._display_params is None:
            return
        p = self._display_params
        # Fetch the actual 2D frame for ``idx``. Wrapped in
        # try/except because the FrameSource can be in a briefly-
        # released state during the silx detach/reattach dance —
        # if a play-tick or scrub fires in that window, get_polar
        # / get_cartesian raise ``RuntimeError("FrameSource not
        # acquired")``. We swallow that quietly; the reattach path
        # re-renders the active frame once acquire completes.
        try:
            if self._mode == MODE_RAW:
                if self._raw_image_stack is None:
                    return
                frame = _apply_raw_flips(
                    np.asarray(self._raw_image_stack[idx]),
                    self._raw_flip_lr, self._raw_flip_ud,
                ).T
            elif self._mode == MODE_POLAR:
                if self._frame_source is None or not self._frame_source.is_open:
                    return
                frame = self._frame_source.get_polar(idx)
            else:  # MODE_CARTESIAN
                if self._frame_source is None or not self._frame_source.is_open:
                    return
                frame = self._frame_source.get_cartesian(idx).T
        except (RuntimeError, ValueError, OSError, KeyError):
            return

        self._plot.setLabel("bottom", p.x_label[0], units=p.x_label[1])
        self._plot.setLabel("left", p.y_label[0], units=p.y_label[1])
        # Prefer the user's dialled-in contrast over the per-build robust
        # default so it survives re-renders; falls back to the cached
        # robust levels when the user hasn't touched the slider.
        base_levels = (
            self._user_levels if self._user_levels is not None else p.levels
        )
        image, levels = self._maybe_apply_log(frame, base_levels)
        # The histogram can emit sigLevelChangeFinished while setImage
        # re-fits the LUT to a new data range; suppress capture so that
        # render doesn't overwrite the user's sticky contrast.
        self._suppress_level_capture = True
        try:
            self._view.setImage(
                image,
                autoRange=auto_range,
                autoLevels=False,
                levels=levels,
                pos=p.pos,
                scale=p.scale,
            )
        finally:
            self._suppress_level_capture = False
        # pyqtgraph's setImage internally calls roiClicked() which
        # force-shows the bottom timeline strip whenever the image has
        # a time axis. With single 2D frames there is no time axis so
        # the strip stays hidden naturally — no extra action needed
        # since the Timeline toggle was removed in the lazy-loading
        # milestone.
        self._hide_pyqtgraph_timeline()

    def _maybe_apply_log(
        self, image: np.ndarray, levels: tuple[float, float]
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """If log-scale is on, return (log10(clip(image, floor)), levels')
        where levels' are the user's sticky contrast when set, else
        recomputed robustly on the transformed first frame.

        Floor is the 1st percentile of strictly-positive finite values
        (or 1e-6 fallback) so the log transform is well-defined for
        zero / negative pixels (background, masked regions). The
        original ``image`` array is not modified.
        """
        if not self._log_scale:
            return image, levels
        finite = image[np.isfinite(image)]
        pos = finite[finite > 0]
        if pos.size > 0:
            floor = float(np.percentile(pos, 1.0))
        else:
            floor = 1e-6
        if floor <= 0:
            floor = 1e-6
        transformed = np.log10(np.clip(image, floor, None))
        # Keep the user's contrast (already in the log domain — the
        # histogram shows the transformed image) when they've set one;
        # otherwise auto-contrast the transformed frame.
        if self._user_levels is not None:
            return transformed, levels
        ref = transformed[0] if transformed.ndim == 3 else transformed
        return transformed, _robust_levels(ref)

    def _on_log_toggled(self, checked: bool) -> None:
        """Re-render in the active mode with log/linear contrast.

        Saves and restores the viewbox range so toggling contrast
        doesn't reset the user's zoom or pan. Frame index is preserved
        too — pyqtgraph's setImage keeps the time-axis position when
        the stack shape is unchanged.
        """
        self._log_scale = bool(checked)
        # Linear and log levels live in different domains, so a sticky
        # linear contrast is meaningless once log is on (and vice versa);
        # drop it and let the toggle re-auto-contrast.
        self._user_levels = None
        saved: tuple[tuple[float, float], tuple[float, float]] | None = None
        try:
            xr, yr = self._plot.getViewBox().viewRange()
            saved = ((float(xr[0]), float(xr[1])), (float(yr[0]), float(yr[1])))
        except Exception:
            logger.debug("suppressed exception in GIWAXSImageViewer._on_log_toggled", exc_info=True)
            saved = None
        self._render_active_mode()
        if saved is not None:
            try:
                self._plot.getViewBox().setRange(
                    xRange=saved[0], yRange=saved[1], padding=0
                )
            except Exception:
                logger.debug("suppressed exception in GIWAXSImageViewer._on_log_toggled", exc_info=True)
                pass

    def _hide_pyqtgraph_timeline(self) -> None:
        """Keep pyqtgraph's bottom timeline strip hidden.

        pyqtgraph's ``setImage`` internally calls ``roiClicked()``
        which re-shows the strip whenever the image has a time axis.
        After the lazy-loading milestone we only ever pass 2D frames
        (no time axis), so the strip stays hidden naturally — but
        we re-apply hidden state explicitly to defend against future
        pyqtgraph versions that might force it back on. The splitter
        handle is set to 0-width in ``__init__`` so there's no
        residual line clipping the image's x-axis label.
        """
        ui = self._view.ui
        ui.roiPlot.setVisible(False)
        ui.splitter.setSizes([1, 0])

    # NB: _on_time_changed (the old pyqtgraph sigTimeChanged slot) is
    # removed. Frame changes flow through set_frame; emissions of
    # frameChanged + matchedStructuresChanged happen there. Left as a
    # comment so future code reviews see the deliberate removal.

    def _render_overlays(self, frame: int) -> None:
        # Raw mode has no peak data to draw — return before touching any
        # overlay path. The four _PeakShapeItems were already cleared by
        # show_raw_stack via clear(), so they have no leftover geometry.
        if self._mode == MODE_RAW:
            return
        peaks = self._frame_peaks.get(frame, {})
        det = peaks.get("detected")
        fit = peaks.get("fitted")

        # Apply the Display-dock min-score filter to the detected
        # overlay. Epsilon is half the slider's natural resolution
        # (0.01 per step → 0.005 tolerance) so a peak whose score
        # the slider was just seeded from is guaranteed to pass,
        # regardless of FP roundoff in the underlying value.
        if det is not None and self._detected_score_cutoff > 0.0 and len(det) > 0:
            try:
                cutoff = self._detected_score_cutoff - 0.005
                mask = np.asarray(det.score) >= cutoff
                if mask.sum() < len(det):
                    keep_ids = [int(det.ids[i]) for i in np.where(mask)[0]]
                    det = _peaks_subset(det, keep_ids)
            except Exception:
                # If the table doesn't have a usable score column
                # (older files), silently skip the filter rather
                # than break the overlay.
                logger.debug("suppressed exception in GIWAXSImageViewer._render_overlays", exc_info=True)
                pass

        # "Show only tracked peaks" (Scan-tracking dock): when active,
        # render only the fitted rows that belong to a surviving track
        # on this frame; frames without members show none. Render-time
        # subset only — the in-memory tables and the file are untouched.
        if self._fitted_visible_only is not None and fit is not None and len(fit) > 0:
            allowed = self._fitted_visible_only.get(frame, set())
            keep_ids = [int(i) for i in fit.ids if int(i) in allowed]
            if len(keep_ids) < len(fit):
                fit = _peaks_subset(fit, keep_ids)

        manual_list = list(self._manual_peaks.get(frame, []))
        # Manual boxes are scratch labels — keep them invisible
        # unless they are the active selection. The ManualPeak
        # instances stay in ``_manual_peaks`` so things like the
        # post-Add-to-fitted Ctrl+Z (FittedRowAction.undo) can
        # reselect the source manual and bring its yellow ROI
        # back. Hit-testing in ``_on_select_at`` still operates on
        # ``_manual_peaks`` directly, so the user can click the
        # invisible region to re-select / reveal the peak.
        selected_is_manual = (
            self._selected is not None and self._selected.kind == "manual"
        )
        if not selected_is_manual:
            manual_list = []
        # When an ROI is active the selected peak is shown via the ROI handles —
        # exclude the manual peak from the manual-overlay path so it doesn't
        # render twice. (Detected/fitted overlays still draw the underlying
        # row; the white SELECTION_STYLE highlight is what's suppressed.)
        roi_active = self._roi_item is not None and self._selected is not None
        if roi_active and self._selected.kind == "manual":
            manual_list = [
                m for m in manual_list if m is not self._selected.manual_ref
            ]
        manual_table = _peaks_from_manual(manual_list)
        # Selection highlight: a PeakTable holding one row per selected
        # peak (primary + Ctrl+click extras). The primary peak gets
        # painted *both* the white outline and (for editable kinds)
        # the resize-ROI handles — the redundant outline is a small
        # price for the visual consistency of "every selected peak
        # looks selected", which matters in multi-select where the
        # user is scanning to confirm what's in the set.
        #
        # Matched-structure primary selections (multi_peak_ids set)
        # expand to every peak in the structure — pull the geometry
        # from the frame's fitted table so the boxes track ROI edits
        # / re-fits. Matched-multi can't coexist with extras (extras
        # only accept detected), so the two branches stay mutually
        # exclusive.
        sel_table: PeakTable | None = None
        if self._selected is not None:
            if self._selected.multi_peak_ids and fit is not None:
                sel_table = _peaks_subset(fit, self._selected.multi_peak_ids)
                # Fallback to the representative peak if every id has
                # gone stale (e.g. user re-ran fitting since the
                # selection was set).
                if len(sel_table) == 0:
                    sel_table = _peaks_from_manual([
                        ManualPeak(
                            radius=self._selected.radius,
                            angle=self._selected.angle,
                            radius_width=self._selected.radius_width,
                            angle_width=self._selected.angle_width,
                            is_ring=self._selected.is_ring,
                            temp_id=self._selected.peak_id,
                        )
                    ])
            else:
                sel_table = _peaks_from_manual([
                    ManualPeak(
                        radius=s.radius,
                        angle=s.angle,
                        radius_width=s.radius_width,
                        angle_width=s.angle_width,
                        is_ring=s.is_ring,
                        temp_id=s.peak_id,
                    )
                    for s in self.selected_peaks()
                ])

        # Fitted-preview overlay: one cyan dashed box per fittable
        # selected peak (primary + Ctrl+click extras). Only meaningful
        # for manual / detected selections — fitted / matched are
        # already on file with their own boxes.
        #
        # Layered painter contract: primary geometry comes from
        # ``_fitted_preview_geom`` (with optional ring mode);
        # extras come from ``_fitted_preview_extras_geoms`` (always
        # segments). The host populates both before invoking
        # ``_render_overlays``; this function just builds the table.
        preview_peaks: list[ManualPeak] = []
        primary_eligible = (
            self._selected is not None
            and self._selected.kind in ("manual", "detected")
        )
        if primary_eligible and self._fitted_preview_geom is not None:
            cr, wr, ca, wa = self._fitted_preview_geom
            if self._fitted_preview_is_ring:
                preview_peaks.append(ManualPeak(
                    radius=cr, angle=45.0,
                    radius_width=wr,
                    angle_width=float("inf"),
                    is_ring=True, temp_id=0,
                ))
            else:
                preview_peaks.append(ManualPeak(
                    radius=cr, angle=ca,
                    radius_width=wr,
                    angle_width=wa,
                    is_ring=False, temp_id=0,
                ))
        # Extras get a preview whenever the primary is fittable
        # (extras only carry detected, which is always fittable).
        if primary_eligible:
            for cr, wr, ca, wa in self._fitted_preview_extras_geoms:
                preview_peaks.append(ManualPeak(
                    radius=cr, angle=ca,
                    radius_width=wr,
                    angle_width=wa,
                    is_ring=False, temp_id=0,
                ))
        preview_table: PeakTable | None = (
            _peaks_from_manual(preview_peaks) if preview_peaks else None
        )

        # Live angular extent of the displayed polar stack so ring
        # overlays (and any segment whose stored angle_width spills
        # past the image bounds) clip to the data instead of the
        # global ±180° fallback.
        extent = self.angular_extent()
        if self._mode == MODE_POLAR:
            self._detected.set_polar(det, extent=extent)
            self._fitted.set_polar(fit, extent=extent)
            self._manual.set_polar(manual_table, extent=extent)
            if sel_table is not None:
                self._selection.set_polar(sel_table, extent=extent)
            else:
                self._selection.clear_path()
            if preview_table is not None:
                self._fitted_preview.set_polar(preview_table, extent=extent)
            else:
                self._fitted_preview.clear_path()
        else:
            self._detected.set_cartesian(det, extent=extent)
            self._fitted.set_cartesian(fit, extent=extent)
            self._manual.set_cartesian(manual_table, extent=extent)
            if sel_table is not None:
                self._selection.set_cartesian(sel_table, extent=extent)
            else:
                self._selection.clear_path()
            if preview_table is not None:
                self._fitted_preview.set_cartesian(preview_table, extent=extent)
            else:
                self._fitted_preview.clear_path()

        self._detected.setVisible(self._visibility["detected"])
        self._fitted.setVisible(self._visibility["fitted"])
        self._manual.setVisible(self._visibility["manual"])

        # Matched overlays: rebuild items for whatever the current frame has.
        self._render_matched_overlays(frame)

    def _render_matched_overlays(self, frame: int) -> None:
        """Tear down the previous frame's matched items and rebuild for this
        frame. Each structure becomes one ``_PeakShapeItem`` in its assigned
        color, painted in the current display mode (polar / Cartesian).
        """
        self._teardown_matched_items()
        # Identities rendered with an empty tracked subset this frame —
        # kept hidden by _is_matched_item_visible against later refreshes.
        self._matched_empty_uids = set()
        structures = self._matched_per_frame.get(frame, [])
        extent = self.angular_extent()
        vb = self._plot.getViewBox()
        for i, s in enumerate(structures):
            # "Show only tracked peaks" applies per matched peak: draw
            # only the boxes around this structure's TRACKED fitted
            # peaks. A structure whose peaks are all untracked ends up
            # with an empty subset and paints nothing (hidden below).
            peaks = s.peaks
            has_peaks = len(peaks) > 0
            if self._fitted_visible_only is not None and has_peaks:
                kept = [
                    int(pid) for pid in s.peaks.ids
                    if not self._fitted_row_hidden(frame, int(pid))
                ]
                peaks = _peaks_subset(s.peaks, kept)
                has_peaks = len(peaks) > 0
            if not has_peaks:
                self._matched_empty_uids.add(s.unique_id)
            pen_info = matched_pen_for(self._color_index_for(s))
            visible = has_peaks and self._is_matched_item_visible(s.unique_id)
            if self._matched_display_style == "markers":
                # q-map look: circles for peaks, dashed arcs for rings.
                # A structure may map to several items (one scatter + N
                # ring curves); all share the structure's unique_id so
                # visibility toggling reaches every one.
                items = self._matched_marker_items(
                    peaks, pen_info["color"], extent
                )
            else:
                box = _PeakShapeItem(**pen_info)
                if self._mode == MODE_POLAR:
                    box.set_polar(peaks, extent=extent)
                else:
                    box.set_cartesian(peaks, extent=extent)
                items = [box]
            for it in items:
                it.setVisible(visible)
                vb.addItem(it, ignoreBounds=True)
                self._matched_items.append((s.unique_id, it))

        # Unmatched-fitted pseudo-group: every fitted row no structure
        # claims (tracked filter applied), drawn in neutral grey with
        # the same display style — rendered even when the frame has no
        # matched structures at all (then EVERY fitted row is
        # unmatched). Built hidden unless its checkbox is on.
        un = self._unmatched_fitted_subset(frame)
        if un is not None:
            visible = self._is_matched_item_visible(UNMATCHED_UID)
            if self._matched_display_style == "markers":
                items = self._matched_marker_items(
                    un, UNMATCHED_COLOR, extent
                )
            else:
                box = _PeakShapeItem(
                    color=UNMATCHED_COLOR,
                    style=MATCHED_STYLE["style"],
                    width=MATCHED_LINE_WIDTH,
                )
                if self._mode == MODE_POLAR:
                    box.set_polar(un, extent=extent)
                else:
                    box.set_cartesian(un, extent=extent)
                items = [box]
            for it in items:
                it.setVisible(visible)
                vb.addItem(it, ignoreBounds=True)
                self._matched_items.append((UNMATCHED_UID, it))

    def _teardown_matched_items(self) -> None:
        if not self._matched_items:
            return
        vb = self._plot.getViewBox()
        for _uid, item in self._matched_items:
            vb.removeItem(item)
        self._matched_items.clear()

    # -- Internals --

    def _on_radio_toggled(self) -> None:
        # The Cartesian / Polar radios are meaningless in RAW mode (raw
        # frames carry no q-axes), so swallow the toggle. Step 7 hides
        # the radios entirely in raw sessions; this guard is the
        # belt-and-braces backup if the radios are still reachable.
        if self._mode == MODE_RAW:
            return
        new = MODE_CARTESIAN if self._radio_cart.isChecked() else MODE_POLAR
        if new != self._mode:
            self.set_mode(new)

    def _on_cmap_changed(self, name: str) -> None:
        self._apply_cmap(name)

    def _apply_cmap(self, name: str) -> None:
        # Try matplotlib first (always present via silx); fall back to the
        # internal pyqtgraph maps if the user picked something not in mpl.
        cmap = None
        for source in ("matplotlib", None):
            try:
                cmap = pg.colormap.get(name, source=source) if source else pg.colormap.get(name)
            except Exception:
                logger.debug("suppressed exception in GIWAXSImageViewer._apply_cmap", exc_info=True)
                cmap = None
            if cmap is not None:
                break
        if cmap is not None:
            self._view.setColorMap(cmap)

    # -- Cursor readout (status bar) --

    def _on_cursor_pos(self, pt: QPointF) -> None:
        info = self._compute_cursor_info(pt)
        self.cursorMoved.emit(info)

    def _on_cursor_left(self) -> None:
        self.cursorMoved.emit(None)

    def _compute_cursor_info(self, pt: QPointF) -> dict | None:
        """Translate a data-space cursor point into a status-bar payload.

        Returns one of three shapes, distinguished by the ``mode`` key:

        - ``"pixel"`` — raw mode: ``row, col, intensity``.
        - ``"cartesian"`` — q-cartesian view: ``q_xy, q_z, intensity``.
        - ``"polar"`` — q-polar view: ``r, theta, intensity``.

        Polar-view axes are **x = radius, y = angle** (matches what
        ``_polar_params`` puts on the plot, NOT the math convention).
        Intensity is looked up against the polar cache when the polar
        view is active; if that returns NaN (uncovered region of the
        polar transform), we fall back to the cartesian grid via the
        derived ``(q_xy, q_z)`` so users see something useful at the
        rim of the polar image.
        """
        x, y = pt.x(), pt.y()
        frame = self.current_frame
        if self._mode == MODE_RAW and self._raw_image_stack is not None:
            stack = self._raw_image_stack
            n_fr, n_rows, n_cols = stack.shape
            col = int(round(x))
            row = int(round(y))
            if (
                0 <= frame < n_fr
                and 0 <= row < n_rows
                and 0 <= col < n_cols
            ):
                intensity = float(stack[frame, row, col])
            else:
                intensity = float("nan")
            return {
                "mode": "pixel",
                "row": row,
                "col": col,
                "intensity": intensity,
            }
        if self._stack is None:
            return None
        if self._mode == MODE_CARTESIAN:
            q_xy_val = float(x)
            q_z_val = float(y)
            intensity = self._lookup_cartesian_intensity(
                frame, q_xy_val, q_z_val
            )
            return {
                "mode": "cartesian",
                "q_xy": q_xy_val,
                "q_z": q_z_val,
                "intensity": intensity,
            }
        # MODE_POLAR — viewer's polar image is laid out with radius on
        # the x axis and angle on the y axis (see _polar_params).
        r_val = float(x)
        theta_deg = float(y)
        intensity = float("nan")
        # The ``_polar_cache is None`` guard fires once
        # ``release_frame_source`` runs, but the cursor can also move in
        # the window where the cache tuple still exists yet the
        # underlying FrameSource is released (e.g. ``clear()`` drops the
        # source without nulling the cache first). Indexing the stack
        # then delegates into ``FrameSource.get_polar`` and raises
        # ``RuntimeError("FrameSource not acquired")``. Require the
        # source to be open so the lookup falls through to the cartesian
        # fallback (which returns NaN safely) instead. Mirror of the
        # guard in ``_lookup_cartesian_intensity``.
        if (
            self._polar_cache is not None
            and self._frame_source is not None
            and self._frame_source.is_open
        ):
            polar_stack, radius_axis, angle_axis = self._polar_cache
            if (
                0 <= frame < polar_stack.shape[0]
                and len(radius_axis) > 0
                and len(angle_axis) > 0
            ):
                r_idx = _bin_index(radius_axis, r_val)
                a_idx = _bin_index(angle_axis, theta_deg)
                intensity = float(polar_stack[frame, r_idx, a_idx])
        # Polar transform leaves NaN in uncovered regions; fall back
        # to the cartesian grid so the readout still shows a real
        # intensity near the edge.
        if intensity != intensity:  # NaN
            q_xy_val, q_z_val = polar_to_qxyz(r_val, theta_deg)
            intensity = self._lookup_cartesian_intensity(
                frame, q_xy_val, q_z_val
            )
        return {
            "mode": "polar",
            "r": r_val,
            "theta": theta_deg,
            "intensity": intensity,
        }

    def _lookup_cartesian_intensity(
        self, frame: int, q_xy_val: float, q_z_val: float
    ) -> float:
        """Pixel-bin intensity lookup against the cartesian stack.

        Uses floor-based binning (not nearest-neighbour) so the
        returned intensity stays constant while the cursor is inside
        the same displayed pixel — matches pyqtgraph's ``pos=axis[0]``
        / ``scale=step`` image transform exactly.

        Returns NaN while the FrameSource is released by the silx
        detach/reattach dance (pipeline runs, Add-to-fitted, clear-
        peaks, save-as). Without this guard, ``stack3d[...]`` delegates
        into ``FrameSource.get_cartesian`` which raises
        ``RuntimeError("FrameSource not acquired")`` every time the
        cursor moves over the cartesian plot during a write. Polar
        mode uses the ``self._polar_cache is None`` guard for the
        same reason; this is the cartesian-side mirror.
        """
        if self._stack is None:
            return float("nan")
        if self._frame_source is None or not self._frame_source.is_open:
            return float("nan")
        stack3d = self._stack.image_stack
        qxy_axis = self._stack.q_xy
        qz_axis = self._stack.q_z
        if not (
            0 <= frame < stack3d.shape[0]
            and len(qxy_axis) > 0
            and len(qz_axis) > 0
        ):
            return float("nan")
        qxy_idx = _bin_index(qxy_axis, q_xy_val)
        qz_idx = _bin_index(qz_axis, q_z_val)
        return float(stack3d[frame, qz_idx, qxy_idx])

    # -- Labelling event handlers (polar mode only for now) --

    def _on_draw_started(self, origin: QPointF) -> None:
        if self._mode != MODE_POLAR:
            return
        rect = QRectF(origin, origin)
        self._preview_item.setRect(rect.normalized())
        self._preview_item.setVisible(True)

    def _on_draw_updated(self, origin: QPointF, current: QPointF) -> None:
        if self._mode != MODE_POLAR or not self._preview_item.isVisible():
            return
        self._preview_item.setRect(QRectF(origin, current).normalized())

    def _on_draw_finished(self, origin: QPointF, end: QPointF) -> None:
        self._preview_item.setVisible(False)
        if self._mode != MODE_POLAR or self._busy:
            return
        # In polar mode: x = radius (Å⁻¹), y = angle (deg).
        rect = QRectF(origin, end).normalized()
        if rect.width() <= 0.0 or rect.height() <= 0.0:
            return
        peak = ManualPeak(
            radius=float(rect.center().x()),
            angle=float(rect.center().y()),
            radius_width=float(rect.width()),
            angle_width=float(rect.height()),
            is_ring=False,
            temp_id=self._next_manual_id,
        )
        self._next_manual_id -= 1
        # Single-manual-box policy: any pre-existing manual peak on this
        # frame is replaced atomically. Modelled as one undo entry so
        # Ctrl+Z rewinds the whole swap rather than two staged steps.
        frame = self.current_frame
        existing = self._manual_peaks.get(frame, [])
        old_peak = existing[0] if existing else None
        if old_peak is not None:
            self._undoable_remove_manual(frame, old_peak)
        # _undoable_add_manual auto-selects on the current frame, so no
        # explicit selection call is needed here.
        self._undoable_add_manual(frame, peak)
        self._push_undo(
            ManualReplaceAction(frame=frame, old_peak=old_peak, new_peak=peak)
        )

    def _on_select_at(self, pos: QPointF, mods=Qt.KeyboardModifier.NoModifier) -> None:
        # Raw mode has no q-space overlays to hit-test. Polar and
        # Cartesian both run the same hit-test pipeline; the click
        # coordinates arrive in whatever space the viewbox is currently
        # showing (polar = (r, a), cartesian = (q_xy, q_z)), and the
        # mode-specific helpers below normalise to polar before
        # checking containment.
        #
        # ``mods``: keyboard modifiers held at press time.
        #
        # * Bare click (no modifiers): walks the full priority list
        #   ``manual > fitted > detected > matched`` and routes the
        #   first hit through ``_set_selected``.
        # * Ctrl+click: multi-select. Hit-tests the detected and fitted
        #   overlays (only — manual / matched stay single-select) and
        #   routes through ``_toggle_selected``. The search prefers the
        #   *current* multi-selection's kind so the user keeps extending
        #   one kind: e.g. a detected multi-select still reaches detected
        #   peaks that have a fitted box drawn on top (post-Run Fitting),
        #   and a fitted multi-select reaches fitted. With no multi-kind
        #   primary yet, fitted wins over detected (same priority as a
        #   bare click). No hit under the cursor → no-op.
        # * Ctrl+Alt never reaches here (press branch consumed it
        #   as a draw gesture).
        if self._mode not in (MODE_POLAR, MODE_CARTESIAN) or self._busy:
            return
        x, y = float(pos.x()), float(pos.y())
        cart = self._mode == MODE_CARTESIAN
        frame = self.current_frame
        peaks_for_frame = self._frame_peaks.get(frame) or {}
        ctrl_only = (
            bool(mods & Qt.KeyboardModifier.ControlModifier)
            and not bool(mods & Qt.KeyboardModifier.AltModifier)
        )

        def hit_manual(peak: ManualPeak) -> bool:
            return _cart_box_contains(peak, x, y) if cart else _polar_box_contains(peak, x, y)

        def hit_table(tbl: PeakTable, i: int) -> bool:
            return (
                _cart_table_row_contains(tbl, i, x, y) if cart
                else _polar_table_row_contains(tbl, i, x, y)
            )

        # Ctrl+click multi-selects detected OR fitted peaks. Search the
        # current multi-selection's kind first (so the user keeps
        # building one kind even where detected/fitted overlap), then
        # the other; with no multi-kind primary, fitted wins over
        # detected (bare-click priority). Manual / matched are never
        # multi-selected.
        if ctrl_only:
            primary = self._selected
            if primary is not None and primary.kind in ("detected", "fitted"):
                order = (
                    ["detected", "fitted"] if primary.kind == "detected"
                    else ["fitted", "detected"]
                )
            else:
                order = ["fitted", "detected"]
            for kind in order:
                if not self._visibility.get(kind, True):
                    continue
                table = peaks_for_frame.get(kind)
                if table is None or len(table) == 0:
                    continue
                for i in reversed(range(len(table))):
                    if kind == "fitted" and self._fitted_row_hidden(
                        frame, int(table.ids[i])
                    ):
                        continue
                    if hit_table(table, i):
                        self._toggle_selected(SelectedPeak(
                            kind=kind,
                            frame=frame,
                            peak_id=int(table.ids[i]),
                            radius=float(table.radius[i]),
                            angle=float(table.angle[i]),
                            radius_width=float(table.radius_width[i]),
                            angle_width=float(table.angle_width[i]),
                            is_ring=bool(table.is_ring[i]),
                            score=float(table.score[i]),
                            amplitude=float(table.amplitude[i]),
                        ))
                        return
            # Ctrl+click on empty space (or only manual/matched under
            # the cursor) = no-op; the existing selection stays put.
            return

        # Priority order: manual > fitted > detected > matched. Matched is
        # last because it's a subset of fitted; the rare case where the user
        # wants the matched-context selection is still reachable by hiding
        # the fitted overlay.
        # 1) manual
        if self._visibility.get("manual", True):
            for peak in reversed(self._manual_peaks.get(frame, [])):
                if hit_manual(peak):
                    self._set_selected(SelectedPeak.from_manual(peak, frame))
                    return

        # 2) fitted, 3) detected — same hit-test against the PeakTable rows.
        for kind in ("fitted", "detected"):
            if not self._visibility.get(kind, True):
                continue
            table = peaks_for_frame.get(kind)
            if table is None or len(table) == 0:
                continue
            for i in reversed(range(len(table))):
                if kind == "fitted" and self._fitted_row_hidden(
                    frame, int(table.ids[i])
                ):
                    continue
                if hit_table(table, i):
                    self._set_selected(SelectedPeak(
                        kind=kind,
                        frame=frame,
                        peak_id=int(table.ids[i]),
                        radius=float(table.radius[i]),
                        angle=float(table.angle[i]),
                        radius_width=float(table.radius_width[i]),
                        angle_width=float(table.angle_width[i]),
                        is_ring=bool(table.is_ring[i]),
                        score=float(table.score[i]),
                        amplitude=float(table.amplitude[i]),
                    ))
                    return

        # 4) matched — only when the master toggle is on. The hit's peak_id
        # is the underlying fitted id (which is what delete_peak consumes).
        if self._matched_master_visible:
            structures = self._matched_per_frame.get(frame, [])
            for s_idx, s in reversed(list(enumerate(structures))):
                if not self._is_matched_item_visible(s.unique_id):
                    continue
                tbl = s.peaks
                color = matched_pen_for(self._color_index_for(s))["color"]
                # Under "show only tracked peaks", only this structure's
                # tracked peaks are drawn — so only they are clickable,
                # and the structure-level highlight covers just them.
                visible_ids = [
                    int(x) for x in tbl.ids
                    if not self._fitted_row_hidden(frame, int(x))
                ]
                for i in reversed(range(len(tbl))):
                    if self._fitted_row_hidden(frame, int(tbl.ids[i])):
                        continue
                    if hit_table(tbl, i):
                        self._set_selected(SelectedPeak(
                            kind="matched",
                            frame=frame,
                            peak_id=int(tbl.ids[i]),
                            radius=float(tbl.radius[i]),
                            angle=float(tbl.angle[i]),
                            radius_width=float(tbl.radius_width[i]),
                            angle_width=float(tbl.angle_width[i]),
                            is_ring=bool(tbl.is_ring[i]),
                            structure_uid=s.unique_id,
                            structure_label=s.label,
                            structure_color=color,
                            score=float(tbl.score[i]),
                            amplitude=float(tbl.amplitude[i]),
                            # Clicking any peak of the structure
                            # promotes the whole (visible) structure
                            # into the selection — overlay highlights
                            # every visible peak in it, table syncs the
                            # structure row.
                            multi_peak_ids=visible_ids,
                        ))
                        return

        # Click on empty space → deselect (the Ctrl branch already
        # early-returned above, so this only runs on bare clicks).
        if self._selected is not None:
            self._set_selected(None)

    def _set_selected(
        self, sel: SelectedPeak | None, *, preserve_manual: bool = False,
    ) -> None:
        """Update the selection and sync the ROI + emit selectionChanged once.

        Side effect: if we're transitioning **away** from a
        manual-peak selection to anything else (a different peak, or
        nothing), drop the previous manual peak via
        ``remove_manual_peak``. This makes manual boxes truly
        transient — clicking off them abandons the draw. A
        ``ManualRemoveAction`` is pushed so ``Ctrl+Z`` brings the box
        back. The new-box draw path already removes the old peak via
        ``ManualReplaceAction`` *before* this method runs, so no
        double-remove happens there.

        Programmatic deselects from ``clear_selection`` bypass this
        method (they set ``self._selected = None`` directly), which
        preserves the manual peak across pipeline resets and other
        non-user-driven selection clears.

        ``preserve_manual``: when True, the manual-removal side
        effect is suppressed. Used by host flows that intentionally
        keep the manual peak around across a programmatic
        switch — currently ``MainWindow._on_add_to_fitted`` (so the
        manual source survives the auto-switch to the new fitted
        peak; without this, the silent ``ManualRemoveAction`` push
        would force the user to press Ctrl+Z twice to fully revert
        an Add-to-fitted commit) and the matching redo closure in
        ``_push_fitted_add_undo``.
        """
        if sel is None and self._selected is None and not self._selected_extras:
            return
        primary_unchanged = (
            sel is not None
            and self._selected is not None
            and sel.kind == self._selected.kind
            and sel.frame == self._selected.frame
            and sel.peak_id == self._selected.peak_id
            and sel.structure_uid == self._selected.structure_uid
        )
        # When the primary doesn't change but extras exist, this call
        # still has work to do (collapse multi-selection back to one).
        if primary_unchanged and not self._selected_extras:
            return
        prev = self._selected
        transitioning_away_from_manual = (
            not preserve_manual
            and prev is not None
            and prev.kind == "manual"
            and prev.manual_ref is not None
            and (sel is None or sel.manual_ref is not prev.manual_ref)
        )
        if transitioning_away_from_manual:
            bucket = self._manual_peaks.get(prev.frame, [])
            if prev.manual_ref in bucket:
                # ``remove_manual_peak`` clears ``self._selected`` to
                # None internally and emits ``selectionChanged(None)``
                # since the removed peak was the active selection. We
                # then re-apply the actual new ``sel`` below, which
                # emits ``selectionChanged(sel)`` a second time. The
                # transient None emit is harmless — listeners that
                # store state will end up with the right value.
                self.remove_manual_peak(prev.frame, prev.manual_ref)
        self._selected = sel
        # _set_selected is the "replace the selection" path; extras
        # always clear here. ``_toggle_selected`` is the alternative
        # entry that appends instead.
        self._selected_extras = []
        self._sync_roi()
        self._render_overlays(self.current_frame)
        self.selectionChanged.emit(sel)
        self.selectionsChanged.emit(self.selected_peaks())

    def _toggle_selected(self, sel: SelectedPeak) -> None:
        """Add / remove ``sel`` to/from the multi-selection (Ctrl+click).

        Multi-select covers ``detected`` and ``fitted`` peaks, but only
        within a single kind: a multi-selection is all-detected or
        all-fitted (the bulk ops downstream — delete cascade, batch
        fit — are kind-specific). If ``sel`` is manual / matched, or it
        is a different kind than the current primary, fall back to
        ``_set_selected`` (single-select replace), which starts a fresh
        selection of the new kind.

        Toggle semantics:
          * No primary yet → ``sel`` becomes the primary.
          * ``sel`` matches the primary → demote: promote
            ``_selected_extras[0]`` to primary if any, else clear.
          * ``sel`` matches an extra → drop from extras.
          * Otherwise → append to extras.
        """
        if sel.kind not in ("detected", "fitted"):
            self._set_selected(sel)
            return
        if self._selected is not None and self._selected.kind != sel.kind:
            self._set_selected(sel)
            return
        if self._selected is None:
            self._selected = sel
            self._sync_roi()
            self._render_overlays(self.current_frame)
            self.selectionChanged.emit(sel)
            self.selectionsChanged.emit(self.selected_peaks())
            return
        # Match by (kind, frame, peak_id) — geometry can shift via ROI
        # drag without changing identity.
        def _same(a: SelectedPeak, b: SelectedPeak) -> bool:
            return (
                a.kind == b.kind
                and a.frame == b.frame
                and a.peak_id == b.peak_id
            )
        if _same(sel, self._selected):
            if self._selected_extras:
                self._selected = self._selected_extras.pop(0)
            else:
                self._selected = None
        elif any(_same(sel, e) for e in self._selected_extras):
            self._selected_extras = [
                e for e in self._selected_extras if not _same(sel, e)
            ]
        else:
            self._selected_extras.append(sel)
        self._sync_roi()
        self._render_overlays(self.current_frame)
        self.selectionChanged.emit(self._selected)
        self.selectionsChanged.emit(self.selected_peaks())

    def keyPressEvent(self, ev) -> None:  # type: ignore[override]
        if (
            ev.key() == Qt.Key.Key_Delete
            and self._selected is not None
            and not self._busy
        ):
            sels = self.selected_peaks()
            kinds = {s.kind for s in sels}
            if (
                len(sels) >= 2
                and len(kinds) == 1
                and next(iter(kinds)) in ("detected", "fitted")
            ):
                # All-detected OR all-fitted multi-selection → bulk
                # delete in one grouped, undoable action on the host.
                self.deletePeaksRequested.emit(list(sels))
                ev.accept()
                return
            sel = self._selected
            if sel.kind == "manual" and sel.manual_ref is not None:
                self.remove_manual_peak(self.current_frame, sel.manual_ref)
            else:
                # File-resident peaks go through MainWindow → mlgidbase
                # delete_peak (cascading + with confirmation).
                self.deletePeakRequested.emit(sel)
            ev.accept()
            return
        # Esc on a selected manual peak removes it. Manual boxes are an
        # in-memory scratchpad — no file write, no confirmation.
        # File-resident selections fall through (Esc is meaningless for
        # them; Delete is the documented binding).
        if (
            ev.key() == Qt.Key.Key_Escape
            and self._selected is not None
            and self._selected.kind == "manual"
            and self._selected.manual_ref is not None
            and not self._busy
        ):
            self.remove_manual_peak(self.current_frame, self._selected.manual_ref)
            ev.accept()
            return
        # Ctrl+A: select every peak of the *current kind* on the current
        # frame. If a fitted peak is the primary, select all fitted;
        # otherwise (detected primary, or nothing selected) select all
        # detected. First row becomes primary; the rest become extras.
        if (
            ev.key() == Qt.Key.Key_A
            and ev.modifiers() == Qt.KeyboardModifier.ControlModifier
            and not self._busy
        ):
            kind = (
                self._selected.kind
                if self._selected is not None
                and self._selected.kind in ("detected", "fitted")
                else "detected"
            )
            self._select_all_of_kind_on_frame(kind)
            ev.accept()
            return
        super().keyPressEvent(ev)

    def _select_all_detected_on_frame(self) -> None:
        """Back-compat shim: select all detected peaks on the frame."""
        self._select_all_of_kind_on_frame("detected")

    def _select_all_of_kind_on_frame(self, kind: str) -> None:
        """Replace the selection with every ``kind`` peak on the current frame.

        ``kind`` is ``"detected"`` or ``"fitted"``. No-op if the frame
        has no such table or zero rows. Emits both ``selectionChanged``
        (primary) and ``selectionsChanged`` (full list) once.
        """
        frame = self.current_frame
        peaks_for_frame = self._frame_peaks.get(frame) or {}
        table = peaks_for_frame.get(kind)
        if table is None or len(table) == 0:
            return

        def _sel(i: int) -> SelectedPeak:
            return SelectedPeak(
                kind=kind,
                frame=frame,
                peak_id=int(table.ids[i]),
                radius=float(table.radius[i]),
                angle=float(table.angle[i]),
                radius_width=float(table.radius_width[i]),
                angle_width=float(table.angle_width[i]),
                is_ring=bool(table.is_ring[i]),
                score=float(table.score[i]),
                amplitude=float(table.amplitude[i]),
            )

        # Skip fitted rows hidden by the "show only tracked peaks"
        # filter — Ctrl+A must not grab invisible untracked peaks.
        indices = [
            i for i in range(len(table))
            if not (kind == "fitted"
                    and self._fitted_row_hidden(frame, int(table.ids[i])))
        ]
        if not indices:
            return
        sels = [_sel(i) for i in indices]
        # Bypass _set_selected (which would clear extras) and write
        # the multi-selection in one shot.
        self._selected = sels[0]
        self._selected_extras = sels[1:]
        self._sync_roi()
        self._render_overlays(self.current_frame)
        self.selectionChanged.emit(self._selected)
        self.selectionsChanged.emit(self.selected_peaks())

    # -- Resizable ROI on the selected peak --

    def _sync_roi(self) -> None:
        """Create / update / destroy the resize ROI to match the selection.

        Polar mode only, and only for editable kinds (manual / detected).
        Fitted and matched selections show the box but no ROI — fitted
        boxes encode the ``2σ`` convention so dragging their bounds
        would misrepresent the underlying Gaussian; matched is a
        derived view of fitted_peaks. Both edit through Add-to-fitted
        / delete instead.

        Handles ring peaks (``is_ring`` true or non-finite ``angle_width``)
        and peaks whose box edges fall outside the visible polar range:
        the ROI is clamped to the data axes so the handles are reachable,
        and ring peaks get only the radial (left/right) handles since their
        angular extent is the whole quadrant by definition.
        """
        self._teardown_roi()

        if (
            self._selected is None
            or self._mode != MODE_POLAR
            or self._busy
            or self._selected.kind in ("fitted", "matched")
        ):
            return

        # Need the polar axes to clamp against; bail if not yet computed.
        if self._polar_cache is None:
            return
        _, radius_axis, angle_axis = self._polar_cache
        if radius_axis.size == 0 or angle_axis.size == 0:
            return
        r_min, r_max = float(radius_axis[0]), float(radius_axis[-1])
        a_min, a_max = float(angle_axis[0]), float(angle_axis[-1])

        sel = self._selected
        is_ring_box = sel.is_ring or not np.isfinite(sel.angle_width)

        if is_ring_box:
            a_lo, a_hi = a_min, a_max
        else:
            a_lo = max(sel.angle - sel.angle_width / 2.0, a_min)
            a_hi = min(sel.angle + sel.angle_width / 2.0, a_max)
            if a_hi <= a_lo:
                return  # peak entirely outside visible angular range
        r_lo = max(sel.radius - sel.radius_width / 2.0, r_min)
        r_hi = min(sel.radius + sel.radius_width / 2.0, r_max)
        if r_hi <= r_lo:
            return

        pos = (r_lo, a_lo)
        size = (r_hi - r_lo, a_hi - a_lo)

        # ROI pen uses the selection white so the resize handles
        # match the selection-highlight outline that wraps every
        # selected peak. Source-kind colouring is preserved by the
        # static overlay items (red dashed for detected, yellow
        # solid for manual) — those stay visible behind the white
        # outline + ROI handles.
        pen = pg.mkPen(
            QColor(SELECTION_STYLE["color"]),
            width=SELECTION_STYLE["width"],
        )
        pen.setStyle(SELECTION_STYLE["style"])
        pen.setCosmetic(True)
        # Hover keeps the same white pen so there's no flicker on
        # mouseover.
        hover_pen = pg.mkPen(
            QColor(SELECTION_STYLE["color"]),
            width=SELECTION_STYLE["width"],
        )
        hover_pen.setCosmetic(True)

        roi = pg.ROI(pos=pos, size=size, pen=pen, hoverPen=hover_pen, movable=True)
        # Edge-only handles (no corners): each handle drags one edge while the
        # opposite edge stays anchored. Rings have only radial handles — the
        # angular bounds are the whole quadrant by construction.
        roi.addScaleHandle([1.0, 0.5], [0.0, 0.5])  # right
        roi.addScaleHandle([0.0, 0.5], [1.0, 0.5])  # left
        if not is_ring_box:
            roi.addScaleHandle([0.5, 1.0], [0.5, 0.0])  # top
            roi.addScaleHandle([0.5, 0.0], [0.5, 1.0])  # bottom
        roi.setZValue(60)
        # Track which dimensions are user-editable for _on_roi_changed.
        roi._mlgid_ring_box = is_ring_box  # type: ignore[attr-defined]
        roi.sigRegionChangeStarted.connect(self._on_roi_drag_started)
        roi.sigRegionChanged.connect(self._on_roi_changed)
        roi.sigRegionChangeFinished.connect(self._on_roi_drag_finished)

        self._plot.getViewBox().addItem(roi, ignoreBounds=True)
        self._roi_item = roi

    def _teardown_roi(self) -> None:
        if self._roi_item is None:
            return
        roi = self._roi_item
        for sig_name in (
            "sigRegionChangeStarted", "sigRegionChanged", "sigRegionChangeFinished",
        ):
            try:
                getattr(roi, sig_name).disconnect()
            except (RuntimeError, TypeError):
                pass
        self._plot.getViewBox().removeItem(roi)
        self._roi_item = None

    def _sync_roi_geometry(self) -> None:
        """Adjust the existing ROI to match the SelectedPeak without rebuilding.

        Used by undo/redo and external geometry updates — blocks signals so we
        don't recursively re-enter ``_on_roi_changed``.
        """
        if self._roi_item is None or self._selected is None:
            return
        roi = self._roi_item
        roi.blockSignals(True)
        try:
            roi.setPos(
                [
                    self._selected.radius - self._selected.radius_width / 2.0,
                    self._selected.angle - self._selected.angle_width / 2.0,
                ],
                update=False,
            )
            roi.setSize([self._selected.radius_width, self._selected.angle_width])
        finally:
            roi.blockSignals(False)

    def _on_roi_drag_started(self) -> None:
        if self._selected is None:
            return
        self._roi_drag_before = self._selected.polar_tuple()

    def _on_roi_changed(self) -> None:
        if self._selected is None or self._roi_item is None:
            return
        roi = self._roi_item
        pos = roi.pos()
        size = roi.size()
        w = abs(float(size[0]))
        h = abs(float(size[1]))
        # ROI sizes can go negative if dragged past the opposite edge — take abs
        # then derive the new center from the (possibly flipped) bottom-left.
        x0 = float(pos[0]) + min(float(size[0]), 0.0)
        y0 = float(pos[1]) + min(float(size[1]), 0.0)
        new_r = x0 + w / 2.0
        new_a = y0 + h / 2.0

        sel = self._selected
        # Ring peaks have only radial handles — the angular extent is fixed
        # at the whole visible quadrant (and the underlying angle_width is
        # often inf). Don't propagate the ROI's angular pos/size into the
        # peak's geometry, or we'd corrupt the ring on every drag.
        is_ring_box = bool(getattr(roi, "_mlgid_ring_box", False))

        sel.radius = new_r
        sel.radius_width = w
        if not is_ring_box:
            sel.angle = new_a
            sel.angle_width = h

        if sel.kind == "manual" and sel.manual_ref is not None:
            sel.manual_ref.radius = new_r
            sel.manual_ref.radius_width = w
            if not is_ring_box:
                sel.manual_ref.angle = new_a
                sel.manual_ref.angle_width = h
        else:
            # Mutate the in-memory PeakTable so the colored detected/fitted
            # outline tracks the drag live. Disk + matched-resync happen on
            # drag-end via _on_roi_drag_finished. For ring boxes we don't
            # touch angle / angle_width since those handles aren't shown.
            peaks_for_frame = self._frame_peaks.get(sel.frame) or {}
            table = peaks_for_frame.get(sel.kind)
            if table is not None and len(table) > 0:
                matches = np.where(table.ids == sel.peak_id)[0]
                if matches.size > 0:
                    idx = int(matches[0])
                    table.radius[idx] = new_r
                    table.radius_width[idx] = w
                    if not is_ring_box:
                        table.angle[idx] = new_a
                        table.angle_width[idx] = h
                    cur_a = float(table.angle[idx])
                    table.q_xy[idx], table.q_z[idx] = polar_to_qxyz(new_r, cur_a)

        self._render_overlays(self.current_frame)
        self.peakGeometryChanged.emit(sel)

    def _on_roi_drag_finished(self) -> None:
        if self._selected is None or self._roi_drag_before is None:
            return
        before = self._roi_drag_before
        after = self._selected.polar_tuple()
        self._roi_drag_before = None
        if before == after:
            return  # idle release — nothing to record

        sel = self._selected
        if sel.kind == "manual" and sel.manual_ref is not None:
            self._push_undo(ManualGeomAction(
                frame=sel.frame, peak=sel.manual_ref,
                before=before, after=after,
            ))
        elif sel.kind in ("detected", "fitted"):
            self._push_undo(FileGeomAction(
                frame=sel.frame, kind=sel.kind, peak_id=sel.peak_id,
                before=before, after=after,
            ))
            # Re-derive matched overlays if a fitted edit changed an
            # underlying row used by any matched solution.
            if sel.kind == "fitted":
                self._refresh_matched_for(sel.frame)
                self._render_overlays(self.current_frame)
            # Persist to the file via MainWindow (see peakRowWriteRequested).
            self.peakRowWriteRequested.emit(
                sel.frame, sel.kind, int(sel.peak_id),
                {"radius": after[0], "angle": after[1],
                 "radius_width": after[2], "angle_width": after[3]},
            )
        # Drag-end notification (all kinds) — subscribers that want the
        # final geometry without per-tick refits hook here.
        self.peakGeometryDragFinished.emit(sel)

    def update_peak_geometry_external(self, peak: ManualPeak) -> None:
        """Sync the ROI to a peak whose geometry was changed elsewhere
        (e.g. by dragging a profile region). Suppresses ROI signals so this
        doesn't loop back into ``_on_roi_changed``.
        """
        if (
            self._selected is None
            or self._selected.kind != "manual"
            or self._selected.manual_ref is not peak
            or self._roi_item is None
        ):
            return
        # Mirror the new geometry onto the SelectedPeak snapshot.
        self._selected.radius = peak.radius
        self._selected.angle = peak.angle
        self._selected.radius_width = peak.radius_width
        self._selected.angle_width = peak.angle_width
        self._sync_roi_geometry()
        self._render_overlays(self.current_frame)
        self.peakGeometryChanged.emit(self._selected)

    def update_detected_geometry_external(self, sel: "SelectedPeak") -> None:
        """Sync the detected-peak overlay to a SelectedPeak whose geometry
        was changed elsewhere — currently the profile region drag (see
        ``profile_viewer.detectedPeakGeometryChanged``).

        Mirrors the in-memory mutation that ``_on_roi_changed`` does
        for detected/fitted peaks during an image-side ROI drag:
        updates the cached ``_frame_peaks`` PeakTable so the colored
        overlay tracks the drag live; recomputes the row's Cartesian
        ``q_xy`` / ``q_z`` from the new polar coordinates; re-renders
        and re-syncs the image-space ROI. Disk persistence happens
        separately on drag-end via the host's
        ``_on_peak_row_write_requested`` slot.
        """
        if (
            self._selected is None
            or self._selected.kind != "detected"
            or sel is not self._selected
            or self._roi_item is None
        ):
            return
        peaks_for_frame = self._frame_peaks.get(sel.frame) or {}
        table = peaks_for_frame.get("detected")
        if table is None or len(table) == 0:
            return
        matches = np.where(table.ids == sel.peak_id)[0]
        if matches.size == 0:
            return
        idx = int(matches[0])
        table.radius[idx] = sel.radius
        table.radius_width[idx] = sel.radius_width
        table.angle[idx] = sel.angle
        table.angle_width[idx] = sel.angle_width
        table.q_xy[idx], table.q_z[idx] = polar_to_qxyz(sel.radius, sel.angle)
        self._sync_roi_geometry()
        self._render_overlays(self.current_frame)
        self.peakGeometryChanged.emit(self._selected)
