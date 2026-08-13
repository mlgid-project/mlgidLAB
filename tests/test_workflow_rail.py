"""The stage strip: convert, detect, fit, match, track.

The rail's whole value is that it agrees with the rest of the window, so
the tests are about agreement rather than appearance: it holds no
pipeline logic (its run glyph clicks the panel's own Run button, so the
kwargs and the queueing stay in one place), it cannot offer to run
something that panel would refuse, and its counts describe the frame the
user is looking at — not the scan, which would need a per-frame sweep
this stage deliberately does not do.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlgidlab import skin
from mlgidlab.file_model import PeakTable
from mlgidlab.workflow_rail import STAGES, WorkflowRail

pytestmark = pytest.mark.gui


def _table(n: int) -> PeakTable:
    z = np.zeros(n)
    return PeakTable(
        q_xy=z.copy(), q_z=z.copy(), angle=z.copy(), radius=z.copy(),
        angle_width=z.copy(), radius_width=z.copy(),
        is_ring=np.zeros(n, dtype=bool), ids=np.arange(n),
        score=z.copy(), amplitude=z.copy(),
    )


@pytest.fixture
def rail(qtbot):
    widget = WorkflowRail()
    qtbot.addWidget(widget)
    return widget


def test_the_rail_lists_the_workflow_in_order(rail):
    assert list(rail.chips) == [key for key, _, _ in STAGES]
    assert [c.button.text() for c in rail.chips.values()] == [
        "Convert", "Detect", "Fit", "Match", "Track"]


def test_a_raw_session_offers_only_conversion(rail):
    rail.set_mode("raw")
    assert rail.chips["convert"].button.isEnabled()
    for key in ("detect", "fit", "match", "track"):
        assert not rail.chips[key].button.isEnabled(), key
        assert "Convert this raw scan first" in rail.chips[key].button.toolTip()


def test_a_converted_session_is_past_conversion(rail):
    rail.set_mode("nexus")
    assert not rail.chips["convert"].button.isEnabled()
    for key in ("detect", "fit", "match", "track"):
        assert rail.chips[key].button.isEnabled(), key


def test_state_tags_the_chip_so_the_skin_can_paint_it(rail):
    rail.set_state("detect", "12 this frame", "ok")
    chip = rail.chips["detect"]
    assert chip.state.text() == "12 this frame"
    assert chip.state.property("status") == "ok"
    assert chip.button.property("state") == "ok"


def test_the_rail_is_hidden_with_no_session(main_window):
    main_window._refresh_workflow_rail()
    assert main_window.workflow_rail.isHidden()


def test_the_rail_appears_and_reports_the_frame(main_window, synthetic_nexus):
    from mlgidlab.session import NexusSession

    main_window._set_active_session(NexusSession.open(synthetic_nexus))
    rail = main_window.workflow_rail
    assert not rail.isHidden()
    assert rail.chips["convert"].state.text() == "done"

    frame = int(main_window.viewer.current_frame)
    main_window.viewer.set_peaks(frame, {"detected": _table(7),
                                         "fitted": None, "manual": None})
    main_window._refresh_workflow_rail()
    assert rail.chips["detect"].state.text() == "7 this frame"
    assert rail.chips["detect"].button.property("state") == "ok"
    assert rail.chips["fit"].state.text() == "not run"


def test_the_counts_are_per_frame_and_move_with_it(main_window,
                                                   synthetic_nexus):
    """"done" here means "this frame has rows", which is why the chip
    says so — an entry-wide claim would need a sweep over every frame."""
    from mlgidlab.session import NexusSession

    main_window._set_active_session(NexusSession.open(synthetic_nexus))
    viewer = main_window.viewer
    viewer.set_peaks(0, {"detected": _table(3), "fitted": None, "manual": None})
    viewer.set_peaks(1, {"detected": None, "fitted": None, "manual": None})

    viewer.set_frame(0)
    main_window._refresh_workflow_rail()
    assert "3 this frame" in main_window.workflow_rail.chips["detect"].state.text()

    viewer.set_frame(1)
    main_window._refresh_workflow_rail()
    assert main_window.workflow_rail.chips["detect"].state.text() == "not run"


def test_clicking_a_stage_raises_its_dock(main_window, synthetic_nexus):
    from mlgidlab.session import NexusSession

    main_window._set_active_session(NexusSession.open(synthetic_nexus))
    main_window._pipeline_dock.hide()
    main_window.workflow_rail.chips["detect"].button.click()
    assert not main_window._pipeline_dock.isHidden()

    main_window._scan_tracking_dock.hide()
    main_window.workflow_rail.chips["track"].button.click()
    assert not main_window._scan_tracking_dock.isHidden()


def test_the_run_glyph_clicks_the_panel_s_own_button(main_window, monkeypatch):
    """One place builds a PipelineCommand, and it is the panel. The rail
    presses that panel's button rather than assembling kwargs itself.

    Driven through "track", whose button exists in every environment:
    backend-less, the Pipeline panel builds a stub with no run buttons
    at all, which is a legitimate absence rather than a wiring error.
    """
    clicked = []
    button = main_window._rail_run_button("track")
    assert button is not None
    monkeypatch.setattr(button, "isEnabled", lambda: True)
    monkeypatch.setattr(button, "click", lambda: clicked.append("track"))

    main_window._on_rail_stage_run("track")
    assert clicked == ["track"]


def test_the_rail_cannot_run_what_the_panel_refuses(main_window, monkeypatch):
    clicked = []
    button = main_window._rail_run_button("track")
    monkeypatch.setattr(button, "isEnabled", lambda: False)
    monkeypatch.setattr(button, "click", lambda: clicked.append("track"))

    main_window._on_rail_stage_run("track")
    assert clicked == []


def test_a_stage_with_no_run_button_is_inert_not_a_crash(main_window):
    """Backend-less, the Pipeline panel has no run buttons; the rail must
    simply do nothing rather than raise."""
    main_window._on_rail_stage_run("nonexistent-stage")


def test_every_stage_maps_to_a_real_dock_and_button(main_window):
    """A typo in the map would leave a chip inert with no error."""
    for key in main_window._RAIL_TARGETS:
        dock_attr, (panel_attr, button_attr) = main_window._RAIL_TARGETS[key]
        assert getattr(main_window, dock_attr, None) is not None, dock_attr
        panel = getattr(main_window, panel_attr, None)
        assert panel is not None, panel_attr
        # The pipeline panel drops its run buttons without the backend,
        # which is a legitimate absence rather than a mapping error.
        from mlgidlab.pipeline import is_mlgidbase_available
        if is_mlgidbase_available() or panel_attr != "pipeline_panel":
            assert getattr(panel, button_attr, None) is not None, button_attr


def test_the_skin_defines_the_rail_rules():
    for theme in ("dark", "light"):
        qss = skin.build_qss(theme)
        assert 'QWidget[mlgid="rail"]' in qss
        assert 'QToolButton[stage="chip"]' in qss
        assert 'QToolButton[stage="run"]' in qss
