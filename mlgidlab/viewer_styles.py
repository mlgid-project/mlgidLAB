"""Overlay, matched-structure and simulation palettes and pen helpers.
Moved out of ``image_viewer`` in the 2026 source split; ``image_viewer``
re-exports every name so older references resolve unchanged.
"""
from __future__ import annotations

import json

import numpy as np
from PySide6.QtCore import QSettings, Qt
from mlgidlab import theme_tokens


OVERLAY_KINDS = ("detected", "fitted", "manual")
MODE_CARTESIAN = "cartesian"
MODE_POLAR = "polar"
# Raw detector data preview — pixel coordinates, no overlays. Reached only
# when a RawSession is active; converted-NeXus sessions never visit this
# mode and their existing Cartesian / Polar paths are unchanged.
MODE_RAW = "raw"

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


def selection_style(theme: str | None = None) -> dict:
    """``SELECTION_STYLE`` with a colour visible on this theme's plot.

    The constant keeps its dark value so every existing import still
    resolves; only the colour is swapped, because a white highlight on
    the light theme's #fafafa plot ground is invisible.
    """
    return {**SELECTION_STYLE,
            "color": theme_tokens.color("overlay_selection", theme)}

# Pre-selection preview: the box a bare click would take, outlined
# while the cursor is over it. Same colour as the selection highlight
# (it is the same answer, one step earlier) but thinner, dashed and
# half-opaque so it never reads as an actual selection.
HOVER_STYLE = {"color": SELECTION_STYLE["color"],
               "style": Qt.PenStyle.DashLine,
               "width": 1.8}
HOVER_OPACITY = 0.55


def hover_style(theme: str | None = None) -> dict:
    """``HOVER_STYLE`` with a colour visible on this theme's plot."""
    return {**HOVER_STYLE,
            "color": theme_tokens.color("overlay_selection", theme)}


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

# "Expected pattern" simulation overlay: forward-simulated reflections
# of a parsed CIF drawn as faint hollow diamonds (spots) and dashed
# arcs (rings). Orange = predicted but not explained by any matched
# peak, green = explained by a matched box, white = selected for
# injection. Diamonds (vs the matched style's circles) so simulated
# and matched markers stay distinguishable when they coincide.
SIM_MISSED_COLOR = "#ff9f43"
SIM_EXPLAINED_COLOR = "#4ade80"
SIM_SELECTED_COLOR = "#ffffff"
# Render-state precedence: selected > explained > missed. Iteration
# order also fixes z-order (later buckets paint on top), so selected
# reflections always read above the rest.
_SIM_STATE_COLORS = {
    "missed": SIM_MISSED_COLOR,
    "explained": SIM_EXPLAINED_COLOR,
    "selected": SIM_SELECTED_COLOR,
}


def sim_state_colors(theme: str | None = None) -> dict:
    """``_SIM_STATE_COLORS`` with a theme-visible "selected" entry.

    Missed (orange) and explained (green) read on both grounds and are
    left alone; only the white selection highlight needs flipping.
    """
    return {**_SIM_STATE_COLORS,
            "selected": theme_tokens.color("sim_selected", theme)}
SIM_OVERLAY_OPACITY = 0.55
# Marker diameter encodes relative simulated intensity, log-scaled
# across three decades (rel=1 -> MAX, rel<=1e-3 -> MIN).
SIM_MARKER_MIN_PX = 5.0
SIM_MARKER_MAX_PX = 14.0


def _sim_intensity_scale(rel_intensity: float) -> float:
    """Map a (0, 1] relative intensity to [0, 1] over three decades."""
    t = 1.0 + np.log10(max(float(rel_intensity), 1e-12)) / 3.0
    return min(1.0, max(0.0, t))


def _sim_marker_size(rel_intensity: float) -> float:
    t = _sim_intensity_scale(rel_intensity)
    return SIM_MARKER_MIN_PX + t * (SIM_MARKER_MAX_PX - SIM_MARKER_MIN_PX)


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


# User-picked colours per structure identity (CIF + hkl), overriding the
# automatic palette. Persisted app-wide (same JSON-in-QSettings idiom as
# the recent-files list) so a structure keeps its chosen colour across
# sessions and files.
_MATCHED_COLORS_KEY = "matchedColors"


def _load_matched_color_overrides() -> dict[tuple, str]:
    raw = QSettings().value(_MATCHED_COLORS_KEY, "")
    try:
        pairs = json.loads(str(raw)) if raw else []
        return {
            (str(c), int(h), int(k), int(l)): str(color)
            for (c, h, k, l), color in pairs
        }
    except (ValueError, TypeError):
        return {}


def _save_matched_color_overrides(overrides: dict[tuple, str]) -> None:
    pairs = [[list(key), color] for key, color in overrides.items()]
    QSettings().setValue(_MATCHED_COLORS_KEY, json.dumps(pairs))

# Curated list of colormaps. Names are matplotlib's; pg.colormap.get falls
# back to matplotlib's registry, which is always available since matplotlib
# is a transitive dep via silx.
COLORMAPS = ("viridis", "inferno", "plasma", "magma", "cividis", "gray")
DEFAULT_COLORMAP = "magma"


def resolve_colormap(name: str):
    """The pyqtgraph ``ColorMap`` for ``name``, or None.

    matplotlib first (always present via silx), then pyqtgraph's own
    registry for anything matplotlib does not know. Shared by the render
    path and the dropdown swatches so the strip cannot advertise a ramp
    the image does not use.
    """
    import pyqtgraph as pg

    for source in ("matplotlib", None):
        try:
            return (pg.colormap.get(name, source=source) if source
                    else pg.colormap.get(name))
        except Exception:
            continue
    return None


def colormap_swatch(name: str, size: int = 32):
    """A square gradient chip for ``name`` as a QPixmap (null if unknown).

    The dropdown used to list six words; a GIWAXS user picks a colormap
    by how it ramps, not by its name.

    Square on purpose. A wide strip is the nicer picture, but Qt sizes
    item icons from a single length — ``iconSize`` and the QSS
    ``icon-size`` property are both one number — so a 56x12 strip is
    either squeezed to the style's 16 px box or forces 58 px rows. A
    square chip renders correctly at whatever size the style asks for.
    """
    from PySide6.QtGui import QColor, QImage, QPixmap

    cmap = resolve_colormap(name)
    if cmap is None:
        return QPixmap()
    lut = cmap.getLookupTable(nPts=size, alpha=False)
    strip = QImage(size, 1, QImage.Format.Format_RGB32)
    for x in range(size):
        r, g, b = (int(v) for v in lut[x][:3])
        strip.setPixelColor(x, 0, QColor(r, g, b))
    return QPixmap.fromImage(strip).scaled(size, size)
