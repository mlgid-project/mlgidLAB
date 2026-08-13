#!/usr/bin/env python
"""Regenerate the mlgidLAB brand assets (app icon + wordmark).

    python tools/render_brand.py

Writes:

    mlgidlab/assets/app/mlgidlab_<size>.png   window / taskbar icon
    mlgidlab/assets/app/mlgidlab.ico          Windows shortcuts, PyInstaller
    docs/images/mlgid_logo_mlgidlab.png       README header (on white)
    docs/images/mlgid_logo_mlgidlab_dark.png  About dialog (on the dark theme)

Everything is drawn here rather than stored as hand-edited SVG so the two
marks cannot drift apart: they share one description of the family mark.

**The geometry is measured, not eyeballed.** The mlgid family logo
(``mlgidBASE/docs/images/mlgid_logo.png``, 1920x1440) was analysed by
fitting a circle to its main arc — centre (887.9, 503.5), r = 397.2 px —
and every constant below is expressed in units of that radius: six nodes
with their interior radii and stroke weight, nine edges with their widths
and colours, and the two concentric arcs. So mlgidLAB's wordmark carries
the same graph as the rest of mlgid-project, and the app icon is that
same graph recoloured to the detector colormap.

Dev-only: needs PySide6 (already a runtime dependency). A multi-size
.ico additionally needs Pillow; without it the .ico is written at a
single size, which Windows scales.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QApplication

REPO = Path(__file__).resolve().parents[1]
APP_DIR = REPO / "mlgidlab" / "assets" / "app"
DOCS_DIR = REPO / "docs" / "images"
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)

# --- mlgid family palette (sampled from the shipped family logos) -------
NAVY = "#183c52"     # the arcs
DEEP = "#012037"     # tagline, darkest nodes
CHAR = "#323c3b"     # the "mlgid" stem of the wordmark
STEEL = "#5b7585"    # the package suffix
LSTEEL = "#8d9faa"   # lighter nodes / edges
PALE = "#e5e9eb"     # the faintest edge

# --- magma accents: the app's default detector colormap -----------------
MAGENTA = "#b73779"
ORANGE = "#fc8961"
CREAM = "#fcfdbf"
GROUND = "#19232d"   # matches the dark theme's panel colour

# --- the traced family mark, in units of the main arc radius ------------
ARC_MAIN = (1.000, -37.1, 269.4, 0.149)      # radius, start deg, span, width
ARC_FRAGMENT = (1.362, -27.0, 60.5, 0.157)

# (dx, dy, interior radius, colour)
FAMILY_NODES = [
    (-0.1194, -0.5565, 0.1448, STEEL),
    (-0.5386, -0.2028, 0.1410, NAVY),
    (+0.5364, -0.2330, 0.1410, DEEP),
    (+0.0228, -0.1398, 0.1422, LSTEEL),
    (-0.3523, +0.3637, 0.1410, NAVY),
    (+0.2129, +0.4228, 0.1448, LSTEEL),
]
NODE_STROKE = 0.0378

# (a, b, colour, width)
FAMILY_EDGES = [
    (0, 1, STEEL, 0.0403), (0, 2, STEEL, 0.0378), (0, 4, NAVY, 0.0378),
    (1, 3, LSTEEL, 0.0378), (1, 5, LSTEEL, 0.0378), (2, 3, LSTEEL, 0.0403),
    (2, 5, LSTEEL, 0.0378), (3, 5, NAVY, 0.0403), (4, 5, PALE, 0.0352),
]

# Family colour -> magma / on-dark equivalents. Both preserve the value
# order: what reads as most prominent on white (darkest navy) must read
# as most prominent on a dark tile (brightest cream).
MAGMA_MAP = {DEEP: CREAM, NAVY: ORANGE, STEEL: "#e05780",
             LSTEEL: MAGENTA, PALE: "#8c2981"}
DARK_MAP = {DEEP: "#e8f0f5", NAVY: "#a9c9dc", STEEL: "#7fa2ba",
            LSTEEL: "#5d7d95", PALE: "#41586b"}

WORDMARK_FONT = "Noto Sans"
TAGLINE = ("Interactive Analysis Workbench",
           "for Grazing Incidence Diffraction")

# The package motif sits where mlgidGUI puts its cursor: lower right,
# overlapping the graph, drawn on an opaque backing so it reads as on top.
MOTIF_CENTRE = (0.22, 0.26)
MOTIF_SCALE = 0.52


def _c(colour: str, palette: dict[str, str] | None) -> QColor:
    return QColor(palette.get(colour, colour) if palette else colour)


def draw_graph(p: QPainter, cx: float, cy: float, r: float, *,
               palette: dict[str, str] | None = None,
               fill: str = "#ffffff") -> None:
    """The family node graph. ``r`` is the main-arc radius."""
    pts = [(cx + dx * r, cy + dy * r) for dx, dy, _, _ in FAMILY_NODES]
    for a, b, colour, width in FAMILY_EDGES:
        p.setPen(QPen(_c(colour, palette), max(1.0, width * r),
                      Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(*pts[a], *pts[b])
    for (x, y), (_, _, rad, colour) in zip(pts, FAMILY_NODES):
        stroke = max(1.0, NODE_STROKE * r)
        rr = rad * r + stroke / 2.0          # traced radius is the interior
        p.setPen(QPen(_c(colour, palette), stroke))
        p.setBrush(QBrush(QColor(fill)))
        p.drawEllipse(QRectF(x - rr, y - rr, 2 * rr, 2 * rr))
    p.setBrush(Qt.BrushStyle.NoBrush)


def draw_detector_motif(p: QPainter, cx: float, cy: float, r: float,
                        colour: str, bg: str) -> None:
    """Detector frame with three peak boxes — mlgidLAB's package motif."""
    scale = MOTIF_SCALE * r
    x = cx + MOTIF_CENTRE[0] * r + scale * 0.10
    y = cy + MOTIF_CENTRE[1] * r + scale * 0.16
    w, h = scale * 0.66, scale * 0.56
    pad = scale * 0.10
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor(bg)))
    p.drawRoundedRect(QRectF(x - pad, y - pad, w + 2 * pad, h + 2 * pad),
                      scale * 0.12, scale * 0.12)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(QColor(colour), max(2.0, scale * 0.075),
                  Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                  Qt.PenJoinStyle.RoundJoin))
    p.drawRoundedRect(QRectF(x, y, w, h), scale * 0.07, scale * 0.07)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor(colour)))
    for fx, fy, fw, fh in ((0.12, 0.16, 0.22, 0.16),
                           (0.48, 0.30, 0.28, 0.20),
                           (0.20, 0.58, 0.18, 0.14)):
        p.drawRect(QRectF(x + fx * w, y + fy * h, fw * w, fh * h))
    p.setBrush(Qt.BrushStyle.NoBrush)


def draw_mark(p: QPainter, cx: float, cy: float, r: float, *,
              palette: dict[str, str] | None = None,
              fill: str = "#ffffff", motif_colour: str = DEEP) -> None:
    """Arcs + graph + motif."""
    for radius, start, span, width in (ARC_MAIN, ARC_FRAGMENT):
        p.setPen(QPen(_c(NAVY, palette), width * r, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.FlatCap))
        rr = radius * r
        p.drawArc(QRectF(cx - rr, cy - rr, 2 * rr, 2 * rr),
                  int(start * 16), int(span * 16))
    draw_graph(p, cx, cy, r, palette=palette, fill=fill)
    draw_detector_motif(p, cx, cy, r, motif_colour, fill)


def _text_path(text: str, font: QFont, x: float, baseline: float) -> QPainterPath:
    path = QPainterPath()
    path.addText(x, baseline, font, text)
    return path


def render_wordmark(path: Path, *, width: int = 1200, dark: bool = False) -> Path:
    """The README / About wordmark: mark over "mlgidLAB" over the tagline.

    Text is converted to outlines, so the PNG carries no font dependency.
    """
    height = int(width * 0.42)
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent if dark else QColor("#ffffff"))
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    fill = GROUND if dark else "#ffffff"
    draw_mark(p, width / 2.0, height * 0.265, height * 0.205,
              palette=DARK_MAP if dark else None, fill=fill,
              motif_colour="#d7e3ea" if dark else DEEP)

    font = QFont(WORDMARK_FONT)
    font.setPixelSize(int(height * 0.215))
    fm = QFontMetricsF(font)
    stem_w = fm.horizontalAdvance("mlgid")
    suffix_w = fm.horizontalAdvance("LAB")
    x = width / 2.0 - (stem_w + suffix_w) / 2.0
    baseline = height * 0.725
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor("#e6e9ea" if dark else CHAR)))
    p.drawPath(_text_path("mlgid", font, x, baseline))
    p.setBrush(QBrush(QColor("#8fb0c4" if dark else STEEL)))
    p.drawPath(_text_path("LAB", font, x + stem_w, baseline))

    tag_font = QFont(WORDMARK_FONT)
    tag_font.setPixelSize(int(height * 0.066))
    tag_fm = QFontMetricsF(tag_font)
    p.setBrush(QBrush(QColor("#c2ced6" if dark else DEEP)))
    for i, line in enumerate(TAGLINE):
        p.drawPath(_text_path(
            line, tag_font,
            width / 2.0 - tag_fm.horizontalAdvance(line) / 2.0,
            height * 0.845 + i * height * 0.066 * 1.32))
    p.end()

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path), "PNG")
    return path


def _tight_bbox(image: QImage):
    import numpy as np

    w, h = image.width(), image.height()
    arr = np.frombuffer(image.constBits(), dtype=np.uint8)
    arr = arr.reshape(h, image.bytesPerLine() // 4, 4)[:, :w, :]
    alpha = arr[..., 3]
    rows = np.flatnonzero(alpha.any(axis=1))
    cols = np.flatnonzero(alpha.any(axis=0))
    if not rows.size or not cols.size:
        return None
    return (int(cols[0]), int(rows[0]),
            int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1))


def render_app_icon(path: Path, size: int) -> Path:
    """The window/taskbar icon: the family graph alone, in magma.

    The arcs are dropped because a thin broken ring is the first thing to
    turn to mush at 16 px; the graph is measured, then centred by its own
    ink rather than by hand, so every size sits square in the tile.
    """
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

    margin = size * 0.078
    tile = QPainterPath()
    tile.addRoundedRect(QRectF(margin, margin, size - 2 * margin,
                               size - 2 * margin), size * 0.18, size * 0.18)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QBrush(QColor(GROUND)))
    p.drawPath(tile)
    p.setBrush(Qt.BrushStyle.NoBrush)

    hi = 512
    scratch = QImage(hi, hi, QImage.Format.Format_ARGB32_Premultiplied)
    scratch.fill(Qt.GlobalColor.transparent)
    q = QPainter(scratch)
    q.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    draw_graph(q, hi * 0.5, hi * 0.5, hi * 0.42, palette=MAGMA_MAP, fill=GROUND)
    q.end()

    box = _tight_bbox(scratch)
    if box is not None:
        x, y, w, h = box
        scale = (size * 0.72) / max(w, h)
        dw, dh = w * scale, h * scale
        p.drawImage(QRectF(size / 2 - dw / 2, size / 2 - dh / 2, dw, dh),
                    scratch, QRectF(x, y, w, h))
    p.end()

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path), "PNG")
    return path


def render_ico(path: Path, pngs: list[Path]) -> Path:
    """Multi-size .ico. Falls back to a single size without Pillow."""
    try:
        from PIL import Image

        frames = [Image.open(png) for png in pngs]
        frames[0].save(path, format="ICO",
                       sizes=[(f.width, f.height) for f in frames])
    except ImportError:
        QImage(str(pngs[-1])).save(str(path), "ICO")
        print("  (Pillow not installed: .ico written at one size)")
    return path


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    _ = app

    pngs = [render_app_icon(APP_DIR / f"mlgidlab_{s}.png", s)
            for s in ICON_SIZES]
    written = [*pngs, render_ico(APP_DIR / "mlgidlab.ico", pngs)]
    for directory in (DOCS_DIR, APP_DIR):
        # docs/ feeds the README header; the package copy is what the
        # About dialog loads at runtime.
        written.append(render_wordmark(directory / "mlgid_logo_mlgidlab.png"))
        written.append(render_wordmark(directory / "mlgid_logo_mlgidlab_dark.png",
                                       dark=True))
    for path in written:
        print(path.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
