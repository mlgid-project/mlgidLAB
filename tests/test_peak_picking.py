"""Which box a click takes when several are stacked under the cursor.

Overlay boxes overlap constantly: a fitted box sits on the detection it
refines, a wide detection contains a narrow one, and a ring's box spans
every angle at its radius, so it contains every spot peak in that band.
The old rule resolved that by table order, which is invisible on screen
because each overlay kind is drawn as one path with one pen. These tests
pin the rules that replaced it: kind priority first, smallest box inside
a kind, rings only on their radial edges, and a repeat click walking
down the stack.
"""

from __future__ import annotations

import math

import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication

from mlgidlab import peak_picking
from mlgidlab.image_viewer import GIWAXSImageViewer
from mlgidlab.peak_picking import Box, box_area, contains, rank_hits
from mlgidlab.theme import apply_dark_theme
from mlgidlab.viewer_items import ManualPeak, _peaks_from_manual

pytestmark = pytest.mark.gui

RING = Box(radius=2.0, radius_width=0.2, angle=45.0,
           angle_width=math.inf, is_ring=True)
SPOT = Box(radius=2.0, radius_width=0.05, angle=30.0, angle_width=4.0)
WIDE = Box(radius=2.0, radius_width=0.50, angle=30.0, angle_width=20.0)


# -- the rules, without Qt ------------------------------------------------

def test_a_ring_is_taken_only_near_its_radial_edges():
    """The whole point: a ring's interior must not swallow the peaks
    sitting inside its band."""
    assert not contains(RING, 2.0, 30.0, ring_edge_tol=0.03)
    assert contains(RING, 1.9, 30.0, ring_edge_tol=0.03)   # inner edge
    assert contains(RING, 2.1, 30.0, ring_edge_tol=0.03)   # outer edge
    assert not contains(RING, 1.5, 30.0, ring_edge_tol=0.03)


def test_without_a_tolerance_a_ring_keeps_its_whole_band():
    """Non-picking callers (geometry maths, the old helpers) still want
    plain containment, so the edge rule is opt-in."""
    assert contains(RING, 2.0, 30.0)


def test_a_non_finite_angular_width_counts_as_a_ring():
    """Rows from the pipeline carry ``angle_width = inf`` instead of the
    GUI's ``is_ring`` flag; both mean the same shape."""
    implied = Box(radius=2.0, radius_width=0.2, angle=45.0,
                  angle_width=math.inf, is_ring=False)
    assert peak_picking.is_ring_box(implied)
    assert not contains(implied, 2.0, 10.0, ring_edge_tol=0.03)


def test_a_ring_with_a_clamped_angular_width_still_checks_the_angle():
    """mlgidbase 0.1.5 clamps a ring's infinite width to a finite one,
    and a finite width is a real angular bound."""
    clamped = Box(radius=2.0, radius_width=0.2, angle=45.0,
                  angle_width=45.0, is_ring=True)
    assert contains(clamped, 1.9, 45.0, ring_edge_tol=0.03)
    assert not contains(clamped, 1.9, 170.0, ring_edge_tol=0.03)


def test_the_smallest_box_ranks_first_and_rings_rank_last():
    assert box_area(SPOT) < box_area(WIDE) < box_area(RING)
    assert rank_hits([("wide", WIDE), ("ring", RING), ("spot", SPOT)]) == [
        "spot", "wide", "ring"]


def test_equal_boxes_keep_the_order_they_were_given():
    """Ties fall back to the caller's order, and the caller feeds rows in
    reverse table order — so same-sized boxes resolve exactly as they did
    before ranking existed."""
    twin = Box(radius=SPOT.radius, radius_width=SPOT.radius_width,
               angle=SPOT.angle, angle_width=SPOT.angle_width)
    assert rank_hits([("second", twin), ("first", SPOT)]) == ["second", "first"]


# -- the viewer ------------------------------------------------------------

@pytest.fixture
def picker(qtbot):
    """A viewer with a known zoom, so pixel-sized tolerances are known."""
    apply_dark_theme(QApplication.instance())
    viewer = GIWAXSImageViewer()
    qtbot.addWidget(viewer)
    viewer.resize(900, 600)
    viewer.show()
    qtbot.waitExposed(viewer)
    viewer._plot.getViewBox().setRange(
        xRange=(0.0, 5.0), yRange=(-90.0, 90.0), padding=0)
    return viewer


@pytest.fixture
def instant_double_click():
    """Make every re-click count as deliberate.

    Cycling ignores clicks that arrive within the double-click interval,
    because a bare double-click (which resets the zoom) delivers a single
    click on its way through. Setting the interval to zero lets the tests
    click as fast as they like; the guard itself is tested separately.
    """
    app = QApplication.instance()
    original = app.doubleClickInterval()
    app.setDoubleClickInterval(0)
    yield
    app.setDoubleClickInterval(original)


def _peak(radius, angle, dr, da, temp_id, is_ring=False):
    return ManualPeak(radius=radius, angle=angle, radius_width=dr,
                      angle_width=da, is_ring=is_ring, temp_id=temp_id)


def _install(viewer, detected=(), fitted=()):
    viewer.set_peaks(0, {
        "detected": _peaks_from_manual(list(detected)) if detected else None,
        "fitted": _peaks_from_manual(list(fitted)) if fitted else None,
        "manual": None,
    })


def _click(viewer, r, a, mods=Qt.KeyboardModifier.NoModifier):
    viewer._on_select_at(QPointF(r, a), mods)


def test_the_inner_box_wins_where_two_are_nested(picker):
    """A box fully inside another used to be unreachable when it came
    first in the table; now it is the one you get."""
    _install(picker, detected=[_peak(2.0, 30.0, 0.5, 20.0, 1),
                               _peak(2.0, 30.0, 0.05, 4.0, 2)])
    _click(picker, 2.0, 30.0)
    assert picker.selected_peak.peak_id == 2

    # ... and the outer box is still reachable everywhere the inner is not
    _click(picker, 2.2, 30.0)
    assert picker.selected_peak.peak_id == 1


def test_kind_priority_still_outranks_size(picker):
    """A fitted box wins over a detected one even when it is the larger
    of the two: the refined result is what the user works with."""
    _install(picker,
             detected=[_peak(2.0, 30.0, 0.05, 4.0, 7)],
             fitted=[_peak(2.0, 30.0, 0.5, 20.0, 9)])
    _click(picker, 2.0, 30.0)
    assert picker.selected_peak.kind == "fitted"
    assert picker.selected_peak.peak_id == 9


def test_a_ring_no_longer_swallows_the_peaks_in_its_band(picker):
    _install(picker, detected=[_peak(2.0, 45.0, 0.2, math.inf, 1, is_ring=True),
                               _peak(2.0, 30.0, 0.04, 4.0, 2)])
    _click(picker, 2.0, 30.0)
    assert picker.selected_peak.peak_id == 2, "the spot, not the ring"

    # Empty part of the ring's band: nothing at all, rather than the ring
    _click(picker, 2.0, 80.0)
    assert picker.selected_peak is None


def test_a_ring_is_selected_by_clicking_its_edge(picker):
    _install(picker, detected=[_peak(2.0, 45.0, 0.2, math.inf, 1, is_ring=True)])
    tol = picker._ring_edge_tol()
    assert tol > 0
    _click(picker, 2.0 - 0.2 / 2 + tol / 2, 80.0)
    assert picker.selected_peak is not None
    assert picker.selected_peak.peak_id == 1


def test_clicking_again_walks_down_the_stack(picker, instant_double_click):
    _install(picker, detected=[_peak(2.0, 30.0, 0.5, 20.0, 1),
                               _peak(2.0, 30.0, 0.05, 4.0, 2)])
    seen = []
    for _ in range(3):
        _click(picker, 2.0, 30.0)
        seen.append(picker.selected_peak.peak_id)
    assert seen == [2, 1, 2], "smallest first, then out, then round again"


def test_a_click_somewhere_else_starts_the_stack_over(picker,
                                                      instant_double_click):
    _install(picker, detected=[_peak(2.0, 30.0, 0.5, 20.0, 1),
                               _peak(2.0, 30.0, 0.05, 4.0, 2)])
    _click(picker, 2.0, 30.0)
    _click(picker, 2.0, 30.0)
    assert picker.selected_peak.peak_id == 1
    # Same boxes, different spot inside the outer one: not a repeat click
    _click(picker, 2.01, 31.0)
    assert picker.selected_peak.peak_id == 2


def test_a_rapid_second_click_does_not_cycle(picker):
    """A bare double-click resets the zoom, and Qt delivers a single
    click first — it must not change the selection on its way through."""
    QApplication.instance().setDoubleClickInterval(10_000)
    try:
        _install(picker, detected=[_peak(2.0, 30.0, 0.5, 20.0, 1),
                                   _peak(2.0, 30.0, 0.05, 4.0, 2)])
        _click(picker, 2.0, 30.0)
        _click(picker, 2.0, 30.0)
        assert picker.selected_peak.peak_id == 2
    finally:
        QApplication.instance().setDoubleClickInterval(400)


def test_ctrl_click_keeps_its_kind_and_takes_the_smaller_box(picker,
                                                             instant_double_click):
    """Ctrl+click still extends one kind (so a detected multi-select is
    not broken by the fitted box drawn over it), and inside that kind it
    follows the same smallest-box rule."""
    _install(picker,
             detected=[_peak(2.0, 30.0, 0.5, 20.0, 1),
                       _peak(2.0, 30.0, 0.05, 4.0, 2)],
             fitted=[_peak(2.0, 30.0, 0.06, 5.0, 5)])
    _click(picker, 2.0, 30.0, Qt.KeyboardModifier.ControlModifier)
    assert picker.selected_peak.kind == "fitted"

    picker.clear_selection()
    _click(picker, 2.2, 30.0)                       # a detected primary
    assert picker.selected_peak.kind == "detected"
    _click(picker, 2.0, 30.0, Qt.KeyboardModifier.ControlModifier)
    ids = [s.peak_id for s in picker.selected_peaks()]
    assert 2 in ids, "extended detected, and took the inner detected box"


# -- hover preview ---------------------------------------------------------

def test_the_hover_outline_marks_what_a_click_would_take(picker):
    _install(picker, detected=[_peak(2.0, 30.0, 0.5, 20.0, 1),
                               _peak(2.0, 30.0, 0.05, 4.0, 2)])
    depth = picker._update_hover(2.0, 30.0)
    assert depth == 2, "two boxes stacked here"
    assert picker._hover_key == ("detected", 2, None)
    assert not picker._hover.boundingRect().isNull()


def test_the_hover_outline_clears_on_empty_space(picker):
    _install(picker, detected=[_peak(2.0, 30.0, 0.05, 4.0, 2)])
    picker._update_hover(2.0, 30.0)
    assert picker._update_hover(4.5, -80.0) == 0
    assert picker._hover_key is None
    assert picker._hover.boundingRect().isNull()


def test_the_hover_outline_steps_aside_for_the_selection(picker):
    """The selected box already carries the solid highlight; drawing the
    preview on top of it would just thicken the line."""
    _install(picker, detected=[_peak(2.0, 30.0, 0.05, 4.0, 2)])
    _click(picker, 2.0, 30.0)
    assert picker._update_hover(2.0, 30.0) == 1
    assert picker._hover_key is None


def test_leaving_the_image_drops_the_outline(picker):
    _install(picker, detected=[_peak(2.0, 30.0, 0.05, 4.0, 2)])
    picker._update_hover(2.0, 30.0)
    assert picker._hover_key is not None
    picker._on_cursor_left()
    assert picker._hover_key is None
    assert picker._hover_pos is None


def test_the_readout_reports_a_stacked_cursor(main_window):
    """The count is what makes the repeat click discoverable."""
    main_window._on_status_cursor_moved(
        {"mode": "polar", "r": 2.0, "theta": 30.0, "intensity": 5.0,
         "overlapping": 3})
    assert "3 boxes here" in main_window._sb_cursor.text()
    main_window._on_status_cursor_moved(
        {"mode": "polar", "r": 2.0, "theta": 30.0, "intensity": 5.0})
    assert "boxes here" not in main_window._sb_cursor.text()
