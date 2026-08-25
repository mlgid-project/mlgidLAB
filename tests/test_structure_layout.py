"""What the Structure tab does to the window around it.

Two behaviours asked for after the first manual pass: the side and
bottom docks fold away while the tab is up, because nothing in them
applies to editing a file's structure; and the workflow strip above the
image can be folded away for good, because not everyone works from it.

The restoring half is what these tests really guard. "Hide them" is
easy; "put back exactly what was open, and nothing else" is where this
kind of feature usually goes wrong.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QSettings

from mlgidlab.main_window_build import RAIL_EXPANDED_KEY
from mlgidlab.session import NexusSession
from mlgidlab.workflow_rail import WorkflowRail

pytestmark = pytest.mark.gui

FOLDED = (
    "_display_dock", "_pipeline_dock", "_sim_dock", "_conversion_dock",
    "_logs_dock", "_profile_dock", "_peaks_dock", "_scan_tracking_dock",
)


@pytest.fixture
def opened(main_window, synthetic_nexus):
    session = NexusSession.open(synthetic_nexus)
    main_window._sessions.append(session)
    main_window._set_active_session(session)
    main_window._confirm_discard_changes = lambda session=None: True
    main_window.show()
    yield main_window
    # Entering the Structure tab now takes the write handle, and the
    # temp dir is removed by the session-cleanup fixture.
    if main_window._h5_edit_handle is not None:
        main_window._h5_edit_handle.release()


def _visible(window):
    return {
        name for name in FOLDED
        if getattr(window, name, None) is not None
        and getattr(window, name).isVisible()
    }


# -- the docks -------------------------------------------------------------


def test_the_structure_tab_folds_the_side_and_bottom_docks(opened):
    before = _visible(opened)
    assert before, "the fixture must start with docks open"
    opened.tabs.setCurrentWidget(opened.structure_panel)
    assert _visible(opened) == set()


def test_the_file_browser_stays_open(opened):
    """It is this tab's navigator; hiding it would leave nothing to edit."""
    opened.tabs.setCurrentWidget(opened.structure_panel)
    assert opened._tree_dock.isVisible()


def test_leaving_the_tab_puts_them_back(opened):
    before = _visible(opened)
    opened.tabs.setCurrentWidget(opened.structure_panel)
    opened.tabs.setCurrentWidget(opened.viewer)
    assert _visible(opened) == before


def test_a_dock_the_user_had_closed_stays_closed(opened):
    """Restoring must not open something the user deliberately shut."""
    opened._logs_dock.hide()
    before = _visible(opened)
    assert "_logs_dock" not in before
    opened.tabs.setCurrentWidget(opened.structure_panel)
    opened.tabs.setCurrentWidget(opened.data_viewer)
    assert _visible(opened) == before
    assert not opened._logs_dock.isVisible()


def test_switching_between_image_and_data_changes_nothing(opened):
    before = _visible(opened)
    opened.tabs.setCurrentWidget(opened.data_viewer)
    assert _visible(opened) == before
    opened.tabs.setCurrentWidget(opened.viewer)
    assert _visible(opened) == before


def test_a_mode_change_cannot_pop_the_docks_back_up(opened):
    """_apply_session_mode decides visibility from scratch."""
    opened.tabs.setCurrentWidget(opened.structure_panel)
    assert _visible(opened) == set()
    opened._apply_session_mode(opened.session)
    assert _visible(opened) == set()


def test_the_docks_survive_a_round_trip_through_a_mode_change(opened):
    before = _visible(opened)
    opened.tabs.setCurrentWidget(opened.structure_panel)
    opened._apply_session_mode(opened.session)
    opened.tabs.setCurrentWidget(opened.viewer)
    assert _visible(opened) == before


def test_folding_is_a_no_op_with_no_session(main_window):
    """The welcome page is up and the docks are already hidden."""
    main_window.tabs.setCurrentWidget(main_window.structure_panel)
    main_window.tabs.setCurrentWidget(main_window.viewer)


# -- the workflow strip ----------------------------------------------------


@pytest.fixture
def rail(qtbot) -> WorkflowRail:
    widget = WorkflowRail()
    qtbot.addWidget(widget)
    widget.show()
    return widget


def test_the_strip_starts_open(rail):
    assert rail.is_expanded()
    assert all(block.isVisible() for block in rail._blocks)
    assert not rail._folded_label.isVisible()


def test_folding_hides_the_chips_and_names_itself(rail):
    rail.set_expanded(False)
    assert not any(block.isVisible() for block in rail._blocks)
    assert rail._folded_label.isVisible()
    assert rail._folded_label.text() == "Workflow"


def test_folding_reclaims_the_height(rail):
    open_height = rail.sizeHint().height()
    rail.set_expanded(False)
    assert rail.sizeHint().height() < open_height


def test_the_toggle_emits_once_per_change(rail, qtbot):
    with qtbot.waitSignal(rail.expandedChanged, timeout=500) as caught:
        rail.set_expanded(False)
    assert caught.args == [False]
    # Setting the same value again is not a change and must not re-emit.
    seen = []
    rail.expandedChanged.connect(seen.append)
    rail.set_expanded(False)
    assert seen == []


def test_the_chips_still_answer_while_folded(rail):
    """Folding is presentation only — the state underneath keeps updating."""
    rail.set_expanded(False)
    rail.set_state("detect", "12 this frame", "ok")
    rail.set_expanded(True)
    assert rail.chips["detect"].state.text() == "12 this frame"


def test_the_choice_is_remembered(main_window):
    settings = QSettings()
    previous = settings.value(RAIL_EXPANDED_KEY, None)
    try:
        main_window.workflow_rail.set_expanded(False)
        assert str(settings.value(RAIL_EXPANDED_KEY)).lower() in ("false", "0")
        main_window.workflow_rail.set_expanded(True)
        assert str(settings.value(RAIL_EXPANDED_KEY)).lower() in ("true", "1")
    finally:
        if previous is None:
            settings.remove(RAIL_EXPANDED_KEY)
        else:
            settings.setValue(RAIL_EXPANDED_KEY, previous)


# -- what folding the docks must not leave behind -------------------------
#
# Reported from a manual pass: controls above the image were frequently
# unclickable. Hiding and re-showing docks that share a tab group is a
# re-tabify as far as Qt is concerned, and this window carries two fixes
# for what that leaves behind — a stale QTabBar painted into a corner,
# and fresh tabs with no glyph. The fold has to run both.


def test_folding_runs_the_dock_chrome_cleanup(opened, monkeypatch):
    calls = []
    monkeypatch.setattr(
        opened, "_hide_stale_dock_tab_bars", lambda: calls.append("bars"))
    monkeypatch.setattr(
        opened, "_apply_dock_tab_icons", lambda: calls.append("icons"))
    opened.tabs.setCurrentWidget(opened.structure_panel)
    assert calls == ["bars", "icons"]
    calls.clear()
    opened.tabs.setCurrentWidget(opened.viewer)
    assert calls == ["bars", "icons"]


def test_the_cleanup_is_skipped_when_nothing_was_folded(opened, monkeypatch):
    """No dock changed, so there is nothing for Qt to have mangled."""
    opened.tabs.setCurrentWidget(opened.structure_panel)
    calls = []
    monkeypatch.setattr(
        opened, "_hide_stale_dock_tab_bars", lambda: calls.append("bars"))
    opened.tabs.setCurrentWidget(opened.structure_panel)
    assert calls == []


def test_a_failing_cleanup_does_not_break_the_switch(opened, monkeypatch):
    def _boom():
        raise RuntimeError("Qt said no")

    monkeypatch.setattr(opened, "_hide_stale_dock_tab_bars", _boom)
    opened.tabs.setCurrentWidget(opened.structure_panel)
    assert _visible(opened) == set()


def test_no_dock_tab_bar_is_left_showing_an_empty_tab_set(opened):
    """A visible bar with no tabs is a ghost with nothing to click."""
    from PySide6.QtWidgets import QTabBar

    opened.tabs.setCurrentWidget(opened.structure_panel)
    opened.tabs.setCurrentWidget(opened.viewer)
    ghosts = [
        tb for tb in opened.findChildren(QTabBar)
        if tb.parent() is opened and tb.isVisible() and tb.count() == 0
    ]
    assert ghosts == []


def test_the_rail_has_no_widget_outside_its_layout(qtbot):
    """An unmanaged child sits at (0, 0) on top of whatever is there."""
    from PySide6.QtWidgets import QWidget

    rail = WorkflowRail()
    qtbot.addWidget(rail)
    rail.show()
    managed = set()
    layout = rail.layout()
    for i in range(layout.count()):
        widget = layout.itemAt(i).widget()
        if widget is not None:
            managed.add(id(widget))
    stray = [c for c in rail.children()
             if isinstance(c, QWidget) and id(c) not in managed]
    assert stray == []


# -- taking the write handle up front -------------------------------------


def test_opening_the_tab_takes_the_write_handle(opened):
    """So the first edit is as immediate as every edit after it."""
    assert opened._h5_edit_handle is None
    opened.tabs.setCurrentWidget(opened.structure_panel)
    assert opened._h5_edit_handle is not None
    assert opened._h5_edit_handle.is_open


def test_a_raw_session_is_not_opened_for_writing(main_window, synthetic_raw):
    from mlgidlab.session import RawSession

    session = RawSession.open([synthetic_raw])
    main_window._sessions.append(session)
    main_window._set_active_session(session)
    main_window.tabs.setCurrentWidget(main_window.structure_panel)
    assert main_window._h5_edit_handle is None


def test_a_failure_to_take_the_handle_is_silent(opened, monkeypatch):
    """The user has asked for nothing yet; the next edit reports it."""
    from mlgidlab.h5_edit import EditError

    def _boom(_path):
        raise EditError("nope")

    monkeypatch.setattr(opened, "_acquire_edit_handle", _boom)
    opened.tabs.setCurrentWidget(opened.structure_panel)
    assert opened.structure_panel is opened.tabs.currentWidget()
