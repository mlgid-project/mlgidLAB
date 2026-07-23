"""GUI wiring of the "Expected pattern" simulation overlay (its own
right-side tabbed dock since the workflow outgrew the Display dock).

Backend-less-safe: the fake CifPattern is a SimpleNamespace seeded
through ``PipelinePanel.set_cif_pattern`` (public API, guarded to work
with or without mlgidbase). Enable-state assertions branch on backend
availability; overlay behaviour is driven programmatically so it runs
identically in both environments.

Source: main_window.py "Expected pattern" section + ``_apply_sim_pattern``
etc.; image_viewer.py ``set_simulation_pattern`` / ``_render_simulation_
overlays``; pipeline_panel.py ``set_cif_pattern`` / ``cifCacheChanged``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPointF, Qt

from mlgidlab.file_model import MatchedStructure, PeakTable
from mlgidlab.image_viewer import _PeakShapeItem
from mlgidlab.pipeline import is_mlgidbase_available
from mlgidlab.session import NexusSession

pytestmark = pytest.mark.gui

BACKEND = is_mlgidbase_available()


def _fake_cifpattern():
    """Two CIFs; the first has two precomputed orientations."""
    return SimpleNamespace(
        cifs=["In2O3.cif", "PbI2.cif"],
        pattern_3d=SimpleNamespace(orientations=[
            np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 2.0]], dtype=np.float32),
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        ]),
        all_patterns_q2d=[
            [
                np.array([[0.5, 0.5], [1.0, 0.2], [0.0, 1.2]]),
                np.array([[0.3, 0.1]]),
            ],
            [np.array([[0.7, 0.7]])],
        ],
        all_patterns_int2d=[
            [np.array([40.0, 100.0, 5.0]), np.array([7.0])],
            [np.array([9.0])],
        ],
        all_patterns_q1d=[
            np.array([0.8, 0.4, 1.6]), np.array([1.1]),
        ],
        all_patterns_int1d=[
            np.array([2.0, 8.0, 4.0]), np.array([3.0]),
        ],
    )


def _open(window, path) -> NexusSession:
    session = NexusSession.open(path)
    window._set_active_session(session)
    return session


def _peak_table(rows) -> PeakTable:
    """rows: (radius, angle, radius_width, angle_width, is_ring)."""
    rows = list(rows)
    radius = np.array([r[0] for r in rows], dtype=float)
    angle = np.array([r[1] for r in rows], dtype=float)
    a = np.deg2rad(angle)
    return PeakTable(
        q_xy=radius * np.cos(a), q_z=radius * np.sin(a),
        angle=angle, radius=radius,
        angle_width=np.array([r[3] for r in rows], dtype=float),
        radius_width=np.array([r[2] for r in rows], dtype=float),
        is_ring=np.array([r[4] for r in rows], dtype=bool),
        ids=np.arange(len(rows)),
        score=np.zeros(len(rows)), amplitude=np.zeros(len(rows)),
    )


def _structure(
    cif: str, hkl: tuple, prob: float = 0.94, boxes=None, local_idx: int = 0,
) -> MatchedStructure:
    peaks = _peak_table(boxes or [(0.0, 0.0, 0.0, 0.0, False)])
    return MatchedStructure(
        solution_field="matched_segments_0000", local_idx=local_idx,
        cif=cif, h=hkl[0], k=hkl[1], l=hkl[2],
        probability=prob, peaks=peaks,
        peak_list=np.arange(len(peaks)),
    )


def test_wheel_does_not_change_combo_selection(main_window):
    """The app-wide _ComboWheelBlocker: scrolling over a CLOSED
    combobox must not step through its items (hover-scrolling a dock
    kept changing settings by accident)."""
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication, QComboBox

    combo = QComboBox(main_window)
    combo.addItems(["a", "b", "c"])
    combo.setCurrentIndex(1)
    for step in (-120, 120):
        event = QWheelEvent(
            QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, step),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase, False,
        )
        QApplication.sendEvent(combo, event)
        assert combo.currentIndex() == 1


def test_section_disabled_until_cache_and_session(main_window, synthetic_nexus):
    mw = main_window
    assert not mw._sim_section_container.isEnabled()
    tip = mw._sim_section_container.toolTip()
    if BACKEND:
        assert "Parse CIFs" in tip
    else:
        assert "mlgidbase" in tip

    _open(mw, synthetic_nexus)
    mw.pipeline_panel.set_cif_pattern(_fake_cifpattern(), None)
    if BACKEND:
        assert mw._sim_section_container.isEnabled()
        assert mw._sim_section_container.toolTip() == ""
    else:
        # Permanently disabled without the backend, but the cache and
        # combos still function for programmatic use.
        assert not mw._sim_section_container.isEnabled()
        assert "mlgidbase" in mw._sim_section_container.toolTip()
    assert mw._sim_cif_combo.count() == 2


def test_cache_signal_and_combo_population(main_window, synthetic_nexus, qtbot):
    mw = main_window
    _open(mw, synthetic_nexus)
    seen: list = []
    mw.pipeline_panel.cifCacheChanged.connect(seen.append)
    fake = _fake_cifpattern()
    mw.pipeline_panel.set_cif_pattern(fake, None)
    assert seen == [fake]
    assert mw.pipeline_panel.cached_cif_pattern() is fake

    combo = mw._sim_cif_combo
    assert [combo.itemData(i) for i in range(combo.count())] == [
        "in2o3", "pbi2"
    ]
    orient = mw._sim_orient_combo
    # No matches on the frame: powder first (and selected), then the
    # precomputed orientations in upstream order.
    assert orient.itemData(0) == (0, 0, 0)
    assert orient.itemText(0) == "Powder (rings)"
    assert [orient.itemData(i) for i in range(orient.count())] == [
        (0, 0, 0), (1, 1, 1), (0, 0, 2)
    ]
    assert orient.currentData() == (0, 0, 0)


def test_master_toggle_installs_and_clears_pattern(main_window, synthetic_nexus):
    mw = main_window
    _open(mw, synthetic_nexus)
    mw.pipeline_panel.set_cif_pattern(_fake_cifpattern(), None)
    assert mw.viewer.simulation_pattern() is None
    assert mw.viewer._sim_items == []

    mw._sim_master_check.setChecked(True)
    pattern = mw.viewer.simulation_pattern()
    assert pattern is not None and pattern.is_powder
    # Default matched style is "boxes": one _PeakShapeItem carries the
    # whole missed bucket (all three rings as radial bands).
    assert len(mw.viewer._sim_items) == 1
    assert isinstance(mw.viewer._sim_items[0], _PeakShapeItem)

    mw._sim_master_check.setChecked(False)
    assert mw.viewer.simulation_pattern() is None
    assert mw.viewer._sim_items == []


def test_matched_entries_listed_first_and_default(main_window, synthetic_nexus):
    mw = main_window
    _open(mw, synthetic_nexus)
    # Matched BEFORE the cache is parsed: the fresh combo defaults to
    # the first matched entry instead of powder.
    mw.viewer.set_matched_structures(0, [_structure("In2O3", (1, 1, 1))])
    mw.pipeline_panel.set_cif_pattern(_fake_cifpattern(), None)

    cif_combo = mw._sim_cif_combo
    assert cif_combo.itemText(0).endswith("— matched")
    assert cif_combo.itemText(1) == "pbi2"

    orient = mw._sim_orient_combo
    assert orient.itemData(0) == (1, 1, 1)
    assert "matched p=0.94" in orient.itemText(0)
    assert orient.currentData() == (1, 1, 1)
    # The matched hkl is not duplicated in the plain orientation list.
    datas = [orient.itemData(i) for i in range(orient.count())]
    assert datas == [(1, 1, 1), (0, 0, 0), (0, 0, 2)]


def _markers_style(mw) -> None:
    idx = mw._matched_style_combo.findData("markers")
    assert idx >= 0
    mw._matched_style_combo.setCurrentIndex(idx)


def test_orientation_switch_and_intensity_cutoff(main_window, synthetic_nexus):
    mw = main_window
    _open(mw, synthetic_nexus)
    mw.pipeline_panel.set_cif_pattern(_fake_cifpattern(), None)
    mw._sim_master_check.setChecked(True)
    _markers_style(mw)

    orient = mw._sim_orient_combo
    idx = mw._combo_data_index(orient, (1, 1, 1))
    assert idx >= 0
    orient.setCurrentIndex(idx)
    pattern = mw.viewer.simulation_pattern()
    assert pattern is not None and pattern.hkl == (1, 1, 1)
    # One scatter item carrying all three spots (rel 0.4, 1.0, 0.05).
    assert len(mw.viewer._sim_items) == 1
    xs, _ys = mw.viewer._sim_items[0].getData()
    assert len(xs) == 3

    # 30% cutoff hides the rel=0.05 reflection.
    mw._sim_int_spin.setValue(30.0)
    xs, _ys = mw.viewer._sim_items[0].getData()
    assert len(xs) == 2

    # 100% keeps only the strongest — twice: the scatter item must
    # exist with a single point.
    mw._sim_int_spin.setValue(100.0)
    xs, _ys = mw.viewer._sim_items[0].getData()
    assert len(xs) == 1

    # Sub-percent precision: 4.99% keeps the rel=0.05 reflection.
    mw._sim_int_spin.setValue(4.99)
    assert mw.viewer._sim_min_intensity == pytest.approx(0.0499)
    xs, _ys = mw.viewer._sim_items[0].getData()
    assert len(xs) == 3

    # The spinbox steps adaptively (about a decade below the current
    # magnitude) so decades-spanning cutoffs stay reachable by arrows.
    mw._sim_int_spin.setValue(0.01)
    mw._sim_int_spin.stepUp()
    assert mw._sim_int_spin.value() == pytest.approx(0.011)


def test_refresh_keeps_installed_pattern_identity(main_window, synthetic_nexus):
    """Frame-change-driven combo refreshes must not reinstall an
    unchanged pattern (a reinstall would drop the reflection
    selection later, in the selection stage)."""
    mw = main_window
    _open(mw, synthetic_nexus)
    mw.pipeline_panel.set_cif_pattern(_fake_cifpattern(), None)
    mw._sim_master_check.setChecked(True)
    before = mw.viewer.simulation_pattern()
    assert before is not None

    mw._refresh_sim_matched_entries(0, [])
    assert mw.viewer.simulation_pattern() is before


def test_cache_invalidation_clears_overlay_and_disables(main_window, synthetic_nexus):
    mw = main_window
    _open(mw, synthetic_nexus)
    mw.pipeline_panel.set_cif_pattern(_fake_cifpattern(), None)
    mw._sim_master_check.setChecked(True)
    assert mw.viewer.simulation_pattern() is not None

    mw.pipeline_panel.clear_cif_cache()
    assert mw.pipeline_panel.cached_cif_pattern() is None
    assert not mw._sim_section_container.isEnabled()
    assert mw.viewer.simulation_pattern() is None
    assert mw.viewer._sim_items == []
    assert mw._sim_cif_combo.count() == 0
    assert mw._sim_orient_combo.count() == 0


def test_viewer_clear_drops_sim_items(main_window, synthetic_nexus):
    mw = main_window
    _open(mw, synthetic_nexus)
    mw.pipeline_panel.set_cif_pattern(_fake_cifpattern(), None)
    mw._sim_master_check.setChecked(True)
    assert mw.viewer._sim_items

    mw.viewer.clear()
    assert mw.viewer._sim_items == []
    assert mw.viewer.simulation_pattern() is None


def test_sim_overlay_follows_matched_style(main_window, synthetic_nexus):
    """The simulated overlay renders boxes or markers in lockstep with
    the matched 'style:' combo."""
    mw = main_window
    _open(mw, synthetic_nexus)
    mw.pipeline_panel.set_cif_pattern(_fake_cifpattern(), None)
    mw._sim_master_check.setChecked(True)
    orient = mw._sim_orient_combo
    orient.setCurrentIndex(mw._combo_data_index(orient, (1, 1, 1)))

    # Default: boxes — one dashed _PeakShapeItem for the missed bucket.
    assert [type(it) for it in mw.viewer._sim_items] == [_PeakShapeItem]

    _markers_style(mw)
    assert len(mw.viewer._sim_items) == 1
    assert isinstance(mw.viewer._sim_items[0], pg.ScatterPlotItem)

    idx = mw._matched_style_combo.findData("boxes")
    mw._matched_style_combo.setCurrentIndex(idx)
    assert [type(it) for it in mw.viewer._sim_items] == [_PeakShapeItem]


def _select_oriented(mw) -> None:
    """Enable the overlay on the (1,1,1) spot pattern of in2o3 with a
    deterministic pixel size for hit-testing (the offscreen viewbox has
    no meaningful on-screen geometry)."""
    mw.pipeline_panel.set_cif_pattern(_fake_cifpattern(), None)
    mw._sim_master_check.setChecked(True)
    orient = mw._sim_orient_combo
    orient.setCurrentIndex(mw._combo_data_index(orient, (1, 1, 1)))
    vb = mw.viewer._plot.getViewBox()
    vb.viewPixelSize = lambda: (0.01, 0.5)  # (Å⁻¹, deg) per pixel


def test_click_toggles_simulated_reflection(main_window, synthetic_nexus, qtbot):
    mw = main_window
    _open(mw, synthetic_nexus)
    _select_oriented(mw)
    refl = mw.viewer.simulated_visible_reflections()
    r0 = refl[0]  # the (0.5, 0.5) spot: radius ~0.707, angle 45

    seen: list = []
    mw.viewer.simulationSelectionChanged.connect(seen.append)
    # Polar mode is the default: click at (radius, angle).
    mw.viewer._on_select_at(
        QPointF(r0.radius, r0.angle), Qt.KeyboardModifier.NoModifier
    )
    assert mw.viewer.simulation_selected() == [r0.index]
    assert seen[-1] == [r0.index]
    assert mw._sim_sel_count_label.text() == "1 selected"

    # Second click on the same reflection toggles it off (Ctrl too).
    mw.viewer._on_select_at(
        QPointF(r0.radius, r0.angle), Qt.KeyboardModifier.ControlModifier
    )
    assert mw.viewer.simulation_selected() == []
    assert mw._sim_sel_count_label.text() == "0 selected"

    # A click far from every reflection changes nothing.
    mw.viewer._on_select_at(
        QPointF(2.5, 10.0), Qt.KeyboardModifier.NoModifier
    )
    assert mw.viewer.simulation_selected() == []


def test_click_miss_falls_through_to_peak_selection(main_window, synthetic_nexus):
    """A click that misses every simulated reflection still selects the
    underlying fitted peak through the normal flow."""
    mw = main_window
    _open(mw, synthetic_nexus)
    _select_oriented(mw)
    # A fitted peak well away from all three simulated spots.
    mw.viewer._frame_peaks[0] = {
        "detected": None,
        "fitted": _peak_table([(2.0, 20.0, 0.2, 10.0, False)]),
    }
    mw.viewer._on_select_at(
        QPointF(2.0, 20.0), Qt.KeyboardModifier.NoModifier
    )
    assert mw.viewer.simulation_selected() == []
    sel = mw.viewer._selected
    assert sel is not None and sel.kind == "fitted"


def test_select_missed_targets_selected_structure_only(
    main_window, synthetic_nexus,
):
    """Select missed = everything not matched to the SELECTED phase:
    reflections sitting on fitted-but-unmatched peaks (or on peaks
    matched to a different CIF) are selected — the injection planner
    handles duplicate avoidance, not the selection."""
    mw = main_window
    _open(mw, synthetic_nexus)
    _select_oriented(mw)
    # Reflections (sorted by radius): 0 = (0.5, 0.5) r~0.707/a45,
    # 1 = (1.0, 0.2) r~1.020/a11.3, 2 = (0.0, 1.2) r1.2/a90.
    # A matched box of the selected CIF explains reflection 1; a plain
    # fitted box (not attributed to any structure) covers reflection 0
    # — it must still be selected and render "missed".
    mw.viewer.set_matched_structures(0, [_structure(
        "In2O3", (1, 1, 1), boxes=[(1.02, 11.3, 0.2, 10.0, False)],
    )])
    mw.viewer._frame_peaks[0] = {
        "detected": None,
        "fitted": _peak_table([
            (1.02, 11.3, 0.2, 10.0, False),   # the matched one
            (0.707, 45.0, 0.2, 10.0, False),  # unmatched but real
        ]),
    }
    assert mw.viewer.simulation_explained_mask(0).tolist() == [
        False, True, False
    ]

    got = mw.viewer.select_missed_simulated(0)
    assert got == [0, 2]
    assert mw._sim_sel_count_label.text() == "2 selected"

    mw.viewer.clear_simulation_selection()
    assert mw.viewer.simulation_selected() == []
    assert mw._sim_sel_count_label.text() == "0 selected"

    # Re-attribute the same box to a DIFFERENT CIF: it no longer
    # explains reflection 1 for the selected In2O3 pattern.
    mw.viewer.set_matched_structures(0, [_structure(
        "PbI2", (0, 0, 0), boxes=[(1.02, 11.3, 0.2, 10.0, False)],
    )])
    assert mw.viewer.simulation_explained_mask(0).tolist() == [
        False, False, False
    ]
    assert mw.viewer.select_missed_simulated(0) == [0, 1, 2]


def test_intensity_prune_drops_hidden_selection(main_window, synthetic_nexus):
    mw = main_window
    _open(mw, synthetic_nexus)
    _select_oriented(mw)
    assert mw.viewer.select_missed_simulated(0) == [0, 1, 2]

    # 30% cutoff hides reflection 2 (rel 0.05) — it leaves the
    # selection too, so the queued set always matches what is visible.
    mw._sim_int_spin.setValue(30.0)
    assert mw.viewer.simulation_selected() == [0, 1]
    assert mw._sim_sel_count_label.text() == "2 selected"


def test_select_missed_recognizes_tight_refits(main_window, synthetic_nexus):
    """Post-add regression: the flow's own fitted rows are tight 2σ
    boxes slightly off the predicted positions. Once such a peak is
    MATCHED to the selected structure, the IoM recognition must accept
    it as coverage (the old centre-in-box rule kept re-selecting
    freshly added peaks)."""
    mw = main_window
    _open(mw, synthetic_nexus)
    _select_oriented(mw)
    mw.viewer._frame_peaks[0] = {
        "detected": None,
        "fitted": _peak_table([
            (0.72, 46.0, 0.04, 1.0, False),  # tight refit near refl 0
            (2.0, 20.0, 0.3, 8.0, False),    # normal rows set the
            (2.2, 30.0, 0.3, 8.0, False),    # median seed size
        ]),
    }
    # Not matched yet: the tight fitted row does NOT hide reflection 0
    # from Select missed anymore — an unclaimed peak still needs the
    # re-match pass.
    assert mw.viewer.select_missed_simulated(0) == [0, 1, 2]
    # Matched to the selected structure: the tight box counts as
    # coverage for both the explained colouring and Select missed.
    mw.viewer.set_matched_structures(0, [_structure(
        "In2O3", (1, 1, 1), boxes=[(0.72, 46.0, 0.04, 1.0, False)],
    )])
    assert mw.viewer.simulation_explained_mask(0).tolist() == [
        True, False, False
    ]
    assert mw.viewer.select_missed_simulated(0) == [1, 2]


def test_matched_legend_mirrored_and_synced(main_window, synthetic_nexus):
    """The Expected-pattern dock carries a twin of the Display dock's
    matched-structures legend: same rows, and toggling a checkbox on
    either side drives the viewer AND the counterpart checkbox."""
    mw = main_window
    _open(mw, synthetic_nexus)
    v = mw.viewer
    s1 = _structure("In2O3", (1, 1, 1), local_idx=0)
    s2 = _structure("PbI2", (0, 0, 0), prob=0.80, local_idx=1)
    v.set_matched_structures(0, [s1, s2])

    assert set(mw._sim_legend_struct_checkboxes) == set(
        mw._matched_struct_checkboxes
    ) == {s1.unique_id, s2.unique_id}
    for uid, chk in mw._matched_struct_checkboxes.items():
        twin = mw._sim_legend_struct_checkboxes[uid]
        assert twin.text() == chk.text()
        assert twin.isChecked() == chk.isChecked()

    # Expected-pattern side → viewer + Display side.
    mw._sim_legend_struct_checkboxes[s1.unique_id].setChecked(False)
    assert not v.matched_visibility(0, s1.unique_id)
    assert not mw._matched_struct_checkboxes[s1.unique_id].isChecked()
    # Display side → viewer + Expected-pattern side.
    mw._matched_struct_checkboxes[s1.unique_id].setChecked(True)
    assert v.matched_visibility(0, s1.unique_id)
    assert mw._sim_legend_struct_checkboxes[s1.unique_id].isChecked()

    # The Display filter also hides the twin rows (their overlays are
    # filter-hidden, so a live checkbox there would be a no-op).
    mw._matched_filter_edit.setText("In2O3")
    assert not mw._matched_struct_rows[s1.unique_id].isHidden()
    assert not mw._sim_legend_struct_rows[s1.unique_id].isHidden()
    assert mw._matched_struct_rows[s2.unique_id].isHidden()
    assert mw._sim_legend_struct_rows[s2.unique_id].isHidden()


def test_matched_legend_master_and_promotion_sync(
    main_window, synthetic_nexus,
):
    """The twin master cascades exactly like the Display master, and
    the "check one while master is off" promotion updates both legends."""
    mw = main_window
    _open(mw, synthetic_nexus)
    v = mw.viewer
    s1 = _structure("In2O3", (1, 1, 1), local_idx=0)
    s2 = _structure("PbI2", (0, 0, 0), prob=0.80, local_idx=1)
    v.set_matched_structures(0, [s1, s2])

    mw._sim_legend_master_check.setChecked(False)
    assert not mw._matched_master_check.isChecked()
    assert not any(
        c.isChecked() for c in mw._matched_struct_checkboxes.values()
    )
    assert not any(
        c.isChecked() for c in mw._sim_legend_struct_checkboxes.values()
    )

    # Promotion from the twin legend: only s1 visible, masters on.
    mw._sim_legend_struct_checkboxes[s1.unique_id].setChecked(True)
    assert mw._matched_master_check.isChecked()
    assert mw._sim_legend_master_check.isChecked()
    for legend in (
        mw._matched_struct_checkboxes, mw._sim_legend_struct_checkboxes,
    ):
        assert legend[s1.unique_id].isChecked()
        assert not legend[s2.unique_id].isChecked()
    assert v.matched_visibility(0, s1.unique_id)
    assert not v.matched_visibility(0, s2.unique_id)


def test_unmatched_row_mirrored_and_synced(main_window, synthetic_nexus):
    """The grey "Unmatched fitted peaks" pseudo-row appears in both
    legends and the two checkboxes stay in sync."""
    mw = main_window
    _open(mw, synthetic_nexus)
    v = mw.viewer
    v._frame_peaks[0] = {
        "detected": None,
        "fitted": _peak_table([(1.0, 30.0, 0.2, 8.0, False)]),
    }
    v.set_matched_structures(0, [])
    # No structures, but unmatched fitted rows exist → one pseudo-row
    # per legend.
    assert len(mw._unmatched_row_checks) == 2
    first, second = mw._unmatched_row_checks
    first.setChecked(True)
    assert second.isChecked()
    assert v.unmatched_visible()
    second.setChecked(False)
    assert not first.isChecked()
    assert not v.unmatched_visible()


def test_selection_survives_frame_relabel_refresh(main_window, synthetic_nexus):
    """The matched-entries refresh that fires on frame changes must not
    drop the reflection selection (the pattern is not reinstalled)."""
    mw = main_window
    _open(mw, synthetic_nexus)
    _select_oriented(mw)
    assert mw.viewer.select_missed_simulated(0) == [0, 1, 2]

    mw._refresh_sim_matched_entries(1, [])
    assert mw.viewer.simulation_selected() == [0, 1, 2]
