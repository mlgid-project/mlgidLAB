"""How wide the side columns open, and why it is measured.

The right-hand docks share one column, and the panels in them sit in
resizable scroll areas — so a column that is too narrow does not scroll,
it compresses and elides. The pinned 350 px squeezed the Pipeline form
(its "Config (yaml)" field came out as a stub), which is what "most of
the content is obstructed" looked like.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea

from mlgidlab.main_window import MainWindow
from mlgidlab.widgets import Card

pytestmark = pytest.mark.gui


def _cap(window) -> int:
    """The widest the column is allowed to open for this window.

    Mirrors the clamps in ``_preferred_right_dock_width``: a share of
    the window, an absolute cap, and a floor.
    """
    return min(
        MainWindow._RIGHT_DOCK_MAX_PX,
        max(window.width() // MainWindow._RIGHT_DOCK_WINDOW_SHARE,
            MainWindow._RIGHT_DOCK_MIN_PX),
    )


def _panel_content(dock):
    inner = dock.widget()
    scroll = inner.findChild(QScrollArea)
    if scroll is not None and scroll.widget() is not None:
        return scroll.widget()
    return inner


def test_the_right_column_is_measured_from_its_panels(main_window):
    wanted = main_window._preferred_right_dock_width()
    for dock in (main_window._display_dock, main_window._pipeline_dock,
                 main_window._sim_dock, main_window._logs_dock):
        assert wanted >= min(_panel_content(dock).sizeHint().width(),
                             _cap(main_window))


def test_it_counts_the_sections_that_start_closed(main_window):
    """Pipeline's Fitting section wants ~465 px and starts collapsed, so
    measuring only what is open reports far too little — and the column
    would be too narrow the moment the user opens one.

    Written against whichever right-hand panel has a collapsed section:
    without the analysis backend the Pipeline panel builds a stub with
    no sections at all, and the Conversion dock carries the point just
    as well."""
    widest_closed = 0
    for dock in (main_window._pipeline_dock, main_window._conversion_dock,
                 main_window._display_dock, main_window._sim_dock):
        content = _panel_content(dock)
        for card in content.findChildren(Card):
            if not card.is_expanded():
                widest_closed = max(widest_closed, card.open_width_hint())
    if widest_closed <= 0:
        pytest.skip("no collapsed sections in this build")
    assert main_window._preferred_right_dock_width() >= min(
        widest_closed, _cap(main_window))


def test_the_column_stays_inside_its_bounds(main_window):
    wanted = main_window._preferred_right_dock_width()
    assert MainWindow._RIGHT_DOCK_MIN_PX <= wanted <= MainWindow._RIGHT_DOCK_MAX_PX
    main_window.resize(900, 800)
    assert main_window._preferred_right_dock_width() >= MainWindow._RIGHT_DOCK_MIN_PX


def test_the_request_lands_once_the_window_has_room(qtbot, main_window):
    """A resizeDocks call made during construction is advisory — the
    window has no geometry yet and QMainWindow scales it down. It is
    re-applied on the first show, and this is the check that the number
    survives."""
    main_window.resize(1600, 950)
    main_window.show()
    qtbot.waitExposed(main_window)
    qtbot.wait(100)
    main_window._apply_default_dock_widths()
    qtbot.wait(150)
    # Within a few pixels rather than to the pixel: QMainWindow trims
    # for separators, and the offscreen platform's idea of a screen
    # varies with what else the suite has built before this test.
    assert abs(main_window._tree_dock.width() - MainWindow._TREE_DOCK_PX) <= 20
    assert abs(main_window._display_dock.width()
               - main_window._preferred_right_dock_width()) <= 20
    assert main_window._display_dock.width() > 350, (
        "wider than the pinned width it replaced")


def test_the_default_window_grows_with_the_screen():
    width, height = MainWindow._default_window_size()
    assert 1400 <= width <= 1600
    assert 900 <= height <= 950
