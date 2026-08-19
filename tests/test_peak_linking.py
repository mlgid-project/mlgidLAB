"""Linking a fitted peak to the detected peak it came from, by id.

detected_peaks and fitted_peaks are independent tables: each assigns its
own ``max(id) + 1`` and nothing records which fit belongs to which
detection. With the link setting on (the default), the two are paired by
carrying the *same* id — one fitted peak per detected peak, refitting
replaces rather than appends, and deleting the detection takes its fit
with it.

This file starts at the storage layer, which is where the pairing has to
be expressible at all: ``file_model``'s add helpers had no way to write a
row under a chosen id, and the snapshot helper had no way to report the
id a row had. The re-key exists because a pipeline fitting run assigns
fitted ids positionally and would otherwise break every pair.

No Qt in this section — it runs in the backend-less CI environment too.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from mlgidlab import file_model

ENTRY = "entry_0000"

_FIT_DTYPE = np.dtype(
    [
        ("amplitude", "f4"), ("angle", "f4"), ("angle_width", "f4"),
        ("radius", "f4"), ("radius_width", "f4"), ("q_z", "f4"),
        ("q_xy", "f4"), ("theta", "f4"), ("score", "f4"),
        ("A", "f4"), ("B", "f4"), ("C", "f4"),
        ("is_ring", "bool"), ("is_cut_qz", "bool"), ("is_cut_qxy", "bool"),
        ("visibility", "i4"), ("id", "i4"),
    ]
)
_VLEN_INT = h5py.vlen_dtype(np.int32)
_MATCHED_DTYPE = np.dtype(
    [
        ("CIF", "S64"), ("h", "i4"), ("k", "i4"), ("l", "i4"),
        ("probability", "f4"), ("peak_list", _VLEN_INT),
    ]
)


def _rows(ids: list[int]) -> np.ndarray:
    arr = np.zeros(len(ids), dtype=_FIT_DTYPE)
    arr["id"] = ids
    # radius doubles as a position marker so a row stays identifiable
    # after its id changes.
    arr["radius"] = [float(i + 1) for i in range(len(ids))]
    arr["angle"] = 45.0
    arr["angle_width"] = 5.0
    arr["radius_width"] = 0.2
    return arr


def _build(
    tmp_path: Path,
    detected_ids: list[int],
    fitted_ids: list[int],
    *,
    peak_list: list[int] | None = None,
    frames: int = 1,
) -> Path:
    """One-entry NeXus file carrying the given detected / fitted ids on
    every frame, optionally with one matched solution row."""
    path = tmp_path / "linked.h5"
    with h5py.File(path, "w", track_order=True) as f:
        data = f.create_group(f"{ENTRY}/data", track_order=True)
        data.attrs["signal"] = "img_gid_q"
        data.create_dataset("img_gid_q", data=np.zeros((frames, 8, 8), np.float32))
        data.create_dataset("q_xy", data=np.linspace(-1, 3, 8, dtype=np.float32))
        data.create_dataset("q_z", data=np.linspace(0, 4, 8, dtype=np.float32))
        for fr in range(frames):
            g = data.create_group(f"analysis/frame{fr:05d}", track_order=True)
            g.create_dataset("detected_peaks", data=_rows(detected_ids))
            g.create_dataset("fitted_peaks", data=_rows(fitted_ids))
            g.create_dataset("fitted_peaks_errors", data=_rows([]))
            if peak_list is not None:
                m = np.zeros(1, dtype=_MATCHED_DTYPE)
                m["CIF"] = b"phase.cif"
                m["probability"] = 0.9
                m["peak_list"][0] = np.asarray(peak_list, dtype=np.int32)
                g.create_dataset("matched_segments_0000", data=m)
    return path


def _ids(path: Path, kind: str, frame: int = 0) -> list[int]:
    ds = (
        f"{ENTRY}/data/analysis/frame{frame:05d}/{kind}_peaks"
    )
    with h5py.File(path, "r") as f:
        return [int(v) for v in f[ds][()]["id"]]


# -- Writing a row under a chosen id ------------------------------------

def test_a_fitted_row_can_be_written_under_a_given_id(tmp_path):
    path = _build(tmp_path, [0, 1, 2], [0, 1])
    new_id = file_model.add_fitted_peak_row(
        path, ENTRY, 0,
        angle=30.0, angle_width=4.0, radius=2.0, radius_width=0.2,
        amplitude=5.0, peak_id=7,
    )
    assert new_id == 7
    assert _ids(path, "fitted") == [0, 1, 7]


def test_a_detected_row_can_be_written_under_a_given_id(tmp_path):
    path = _build(tmp_path, [0, 1, 2], [0, 1])
    new_id = file_model.add_detected_peak_row(
        path, ENTRY, 0,
        angle=30.0, angle_width=4.0, radius=2.0, radius_width=0.2,
        peak_id=9,
    )
    assert new_id == 9
    assert _ids(path, "detected") == [0, 1, 2, 9]


def test_without_an_explicit_id_the_old_rule_still_applies(tmp_path):
    """max(id) + 1, unchanged — the link is opt-out, and everything that
    does not ask for an id must behave exactly as before."""
    path = _build(tmp_path, [0, 1, 2], [3, 8])
    new_id = file_model.add_fitted_peak_row(
        path, ENTRY, 0,
        angle=30.0, angle_width=4.0, radius=2.0, radius_width=0.2,
        amplitude=5.0,
    )
    assert new_id == 9


def test_the_snapshot_can_carry_the_id_it_had(tmp_path):
    """Undo of a linked delete has to put the row back under its old id,
    or the restored pair is no longer a pair."""
    path = _build(tmp_path, [0, 1, 2], [4, 5])
    plain = file_model.read_peak_rows(path, ENTRY, 0, "fitted", [5])
    assert plain and "id" not in plain[0]
    with_ids = file_model.read_peak_rows(
        path, ENTRY, 0, "fitted", [5], with_ids=True,
    )
    assert with_ids[0]["id"] == 5
    # The rest of the dict is unchanged, so it still splats into the add
    # helper once the id is popped off.
    stripped = dict(with_ids[0])
    stripped.pop("id")
    assert stripped == plain[0]


# -- Re-keying fitted ids after a pipeline fitting run -------------------

def test_the_rekey_maps_fitted_onto_detected_by_position(tmp_path):
    """A pipeline run numbers fitted rows 0..N-1 by position over the
    detected table, so once detected has gaps the two disagree."""
    path = _build(tmp_path, [3, 7, 11], [0, 1, 2])
    assert file_model.rekey_fitted_ids_to_detected(path, ENTRY) == 1
    assert _ids(path, "fitted") == [3, 7, 11]
    # Row order and content are untouched; only the id column moved.
    with h5py.File(path, "r") as f:
        arr = f[f"{ENTRY}/data/analysis/frame00000/fitted_peaks"][()]
    assert [float(v) for v in arr["radius"]] == [1.0, 2.0, 3.0]


def test_the_rekey_is_a_no_op_when_the_ids_already_agree(tmp_path):
    path = _build(tmp_path, [0, 1, 2], [0, 1, 2])
    assert file_model.rekey_fitted_ids_to_detected(path, ENTRY) == 0
    assert _ids(path, "fitted") == [0, 1, 2]


def test_the_rekey_skips_a_frame_whose_counts_disagree(tmp_path):
    """Two fits against three detections cannot be paired positionally.
    Guessing would silently mislabel peaks, so the frame is left alone."""
    path = _build(tmp_path, [3, 7, 11], [0, 1])
    assert file_model.rekey_fitted_ids_to_detected(path, ENTRY) == 0
    assert _ids(path, "fitted") == [0, 1]


def test_the_rekey_walks_every_frame_or_just_the_one_asked_for(tmp_path):
    path = _build(tmp_path, [3, 7, 11], [0, 1, 2], frames=3)
    assert file_model.rekey_fitted_ids_to_detected(path, ENTRY, frame=1) == 1
    assert _ids(path, "fitted", 0) == [0, 1, 2]
    assert _ids(path, "fitted", 1) == [3, 7, 11]
    assert file_model.rekey_fitted_ids_to_detected(path, ENTRY) == 2


def test_the_rekey_leaves_matched_peak_lists_alone(tmp_path):
    """``peak_list`` holds *positions* into fitted_peaks, not ids. The
    re-key rewrites one column in place without touching row order, so
    matched structures cannot be disturbed by it."""
    path = _build(tmp_path, [3, 7, 11], [0, 1, 2], peak_list=[0, 2])
    file_model.rekey_fitted_ids_to_detected(path, ENTRY)
    with h5py.File(path, "r") as f:
        row = f[f"{ENTRY}/data/analysis/frame00000/matched_segments_0000"][()][0]
    assert [int(v) for v in row["peak_list"]] == [0, 2]


@pytest.mark.parametrize("missing", ["detected_peaks", "fitted_peaks"])
def test_the_rekey_tolerates_a_frame_without_both_tables(tmp_path, missing):
    path = _build(tmp_path, [0, 1], [0, 1])
    with h5py.File(path, "r+") as f:
        del f[f"{ENTRY}/data/analysis/frame00000/{missing}"]
    assert file_model.rekey_fitted_ids_to_detected(path, ENTRY) == 0


# -- The two settings ---------------------------------------------------

@pytest.fixture
def clean_link_settings():
    """Remove both keys before and after, so one test's choice cannot
    leak into another's default (the whole point of these tests is what
    happens when nothing is stored)."""
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        app.setOrganizationName("mlgidLAB")
        app.setApplicationName("mlgidLAB")
    keys = ("peakLinkFittedToDetected", "peakLinkFittedDeleteRemovesDetected")
    s = QSettings()
    for k in keys:
        s.remove(k)
    s.sync()
    yield s
    for k in keys:
        s.remove(k)
    s.sync()


@pytest.mark.gui
def test_the_link_is_on_and_the_reverse_cascade_off_by_default(
    clean_link_settings,
):
    from mlgidlab import peak_link

    assert peak_link.link_enabled() is True
    assert peak_link.reverse_delete_enabled() is False


@pytest.mark.gui
def test_the_settings_round_trip(clean_link_settings):
    from mlgidlab import peak_link

    s = clean_link_settings
    s.setValue("peakLinkFittedToDetected", False)
    s.sync()
    assert peak_link.link_enabled() is False
    s.setValue("peakLinkFittedToDetected", True)
    s.setValue("peakLinkFittedDeleteRemovesDetected", True)
    s.sync()
    assert peak_link.reverse_delete_enabled() is True


@pytest.mark.gui
def test_the_reverse_cascade_needs_the_link(clean_link_settings):
    """Without paired ids there is no partner to delete, so the reverse
    setting cannot act on its own however it is stored."""
    from mlgidlab import peak_link

    s = clean_link_settings
    s.setValue("peakLinkFittedToDetected", False)
    s.setValue("peakLinkFittedDeleteRemovesDetected", True)
    s.sync()
    assert peak_link.reverse_delete_enabled() is False


@pytest.mark.gui
@pytest.mark.parametrize(
    "stored,expected",
    [("true", True), ("false", False), (True, True), (False, False)],
)
def test_a_default_on_setting_survives_the_ini_backend(
    clean_link_settings, stored, expected,
):
    """QSettings hands back the string "false" on Linux and a real bool
    on macOS. The repo's older idiom only reads back a default-off
    setting correctly, which would have made this one unturnoffable."""
    from mlgidlab import peak_link

    s = clean_link_settings
    s.setValue("peakLinkFittedToDetected", stored)
    s.sync()
    assert peak_link.link_enabled() is expected


@pytest.mark.gui
def test_the_settings_dialog_shows_and_saves_both(main_window, clean_link_settings):
    from mlgidlab.main_window_dialogs import _SettingsDialog

    dlg = _SettingsDialog(main_window)
    assert dlg._chk_peak_link.isChecked()
    assert not dlg._chk_peak_link_reverse.isChecked()
    # The dependent box greys out with the link rather than lying about
    # what it would do.
    assert dlg._chk_peak_link_reverse.isEnabled()
    dlg._chk_peak_link.setChecked(False)
    assert not dlg._chk_peak_link_reverse.isEnabled()

    dlg._chk_peak_link.setChecked(True)
    dlg._chk_peak_link_reverse.setChecked(True)
    dlg.save_to_qsettings()
    clean_link_settings.sync()

    from mlgidlab import peak_link
    assert peak_link.link_enabled() is True
    assert peak_link.reverse_delete_enabled() is True
    # A fresh dialog reflects what was stored.
    assert _SettingsDialog(main_window)._chk_peak_link_reverse.isChecked()


# -- Fitting: the fit takes its detected peak's id ----------------------
#
# The fixture file carries detected ids [0, 1, 2] and fitted ids [0, 1]
# on frame 0, so detected 2 is unfitted and detected 0/1 already have a
# fit — exactly the two cases the link has to tell apart.

from mlgidlab.image_viewer import ManualPeak, SelectedPeak  # noqa: E402
from mlgidlab.session import NexusSession  # noqa: E402

_FIT_ROW = {
    "radius": 2.75, "radius_width": 0.11,
    "angle": 33.0, "angle_width": 6.0,
    "amplitude": 42.0, "theta": 0.0, "A": 1.0, "B": 0.0, "C": 1.0,
}


@pytest.fixture(autouse=True)
def _no_blocking_modals(monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    def _boom(*a, **k):
        raise AssertionError(f"unexpected blocking QMessageBox: {a[1:3]!r}")

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_boom))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_boom))


def _confirm(monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )


def _open(window, path):
    window._set_active_session(NexusSession.open(path))
    return window.session.temp_path


def _table(path, kind, frame=0):
    return file_model.load_peaks(path, ENTRY, frame)[kind]


def _table_ids(path, kind, frame=0) -> list[int]:
    t = _table(path, kind, frame)
    return [] if t is None else [int(v) for v in t.ids]


def _select_detected(window, index: int, frame: int = 0) -> SelectedPeak:
    det = (window.viewer._frame_peaks.get(frame) or {})["detected"]
    sel = SelectedPeak(
        kind="detected", frame=frame, peak_id=int(det.ids[index]),
        radius=float(det.radius[index]), angle=float(det.angle[index]),
        radius_width=float(det.radius_width[index]),
        angle_width=float(det.angle_width[index]),
        is_ring=bool(det.is_ring[index]),
        score=float(det.score[index]),
        amplitude=float(det.amplitude[index]),
    )
    window.viewer._set_selected(sel)
    return sel


def _stub_fit(window, monkeypatch, row=None):
    """Skip pygidfit; the link is about which id the row lands under."""
    monkeypatch.setattr(
        window, "_build_fitted_row_2d",
        lambda sel, entry, frame, **_kw: dict(row or _FIT_ROW),
    )


@pytest.mark.gui
def test_a_fit_is_stored_under_its_detected_peaks_id(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    _stub_fit(main_window, monkeypatch)
    _select_detected(main_window, 2)  # detected id 2, not yet fitted
    main_window._on_add_to_fitted()
    assert _table_ids(path, "fitted") == [0, 1, 2]


@pytest.mark.gui
def test_refitting_replaces_the_previous_fit(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    """The whole point: one fitted peak per detected peak, however many
    times the user refits it with different borders."""
    path = _open(main_window, synthetic_nexus_with_peaks)
    _stub_fit(main_window, monkeypatch)
    _select_detected(main_window, 0)  # detected id 0, already fitted
    main_window._on_add_to_fitted()
    assert _table_ids(path, "fitted") == [1, 0]  # 0 rewritten at the end
    fit = _table(path, "fitted")
    row = {int(fit.ids[i]): float(fit.radius[i]) for i in range(len(fit))}
    assert row[0] == pytest.approx(_FIT_ROW["radius"])
    # And again, with different borders — still one row for peak 0.
    _select_detected(main_window, 0)
    _stub_fit(main_window, monkeypatch, {**_FIT_ROW, "radius": 2.9})
    main_window._on_add_to_fitted()
    assert sorted(_table_ids(path, "fitted")) == [0, 1]
    fit = _table(path, "fitted")
    row = {int(fit.ids[i]): float(fit.radius[i]) for i in range(len(fit))}
    assert row[0] == pytest.approx(2.9)


@pytest.mark.gui
def test_with_the_link_off_a_refit_appends_as_before(
    main_window, synthetic_nexus_with_peaks, monkeypatch, peak_link_off,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    _stub_fit(main_window, monkeypatch)
    _select_detected(main_window, 0)
    main_window._on_add_to_fitted()
    assert _table_ids(path, "fitted") == [0, 1, 2]


@pytest.mark.gui
def test_undoing_a_refit_brings_the_previous_fit_back(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    before = _table(path, "fitted")
    old_radius = float(before.radius[0])
    _stub_fit(main_window, monkeypatch)
    _select_detected(main_window, 0)
    main_window._on_add_to_fitted()

    main_window.viewer.undo_last_action()
    fit = _table(path, "fitted")
    assert sorted(int(v) for v in fit.ids) == [0, 1]
    row = {int(fit.ids[i]): float(fit.radius[i]) for i in range(len(fit))}
    assert row[0] == pytest.approx(old_radius)


@pytest.mark.gui
def test_a_hand_drawn_box_gets_a_detected_partner(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    """A manual box has no detected peak behind it, so one is written
    first and the fit pairs to it — every fitted peak has a partner."""
    path = _open(main_window, synthetic_nexus_with_peaks)
    _stub_fit(main_window, monkeypatch)
    manual = ManualPeak(
        radius=2.0, angle=45.0, radius_width=0.3, angle_width=8.0, temp_id=1,
    )
    main_window.viewer.add_manual_peak(0, manual)
    main_window.viewer._set_selected(SelectedPeak.from_manual(manual, 0))
    main_window._on_add_to_fitted()

    assert _table_ids(path, "detected") == [0, 1, 2, 3]
    assert _table_ids(path, "fitted") == [0, 1, 3]

    # One Ctrl+Z removes both halves.
    main_window.viewer.undo_last_action()
    assert _table_ids(path, "detected") == [0, 1, 2]
    assert _table_ids(path, "fitted") == [0, 1]


@pytest.mark.gui
def test_with_the_link_off_a_hand_drawn_box_adds_no_detected_row(
    main_window, synthetic_nexus_with_peaks, monkeypatch, peak_link_off,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    _stub_fit(main_window, monkeypatch)
    manual = ManualPeak(
        radius=2.0, angle=45.0, radius_width=0.3, angle_width=8.0, temp_id=1,
    )
    main_window.viewer.add_manual_peak(0, manual)
    main_window.viewer._set_selected(SelectedPeak.from_manual(manual, 0))
    main_window._on_add_to_fitted()
    assert _table_ids(path, "detected") == [0, 1, 2]
    assert _table_ids(path, "fitted") == [0, 1, 2]


@pytest.mark.gui
def test_a_batch_fit_pairs_every_fit_with_its_detection(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    """Three detected peaks, two of them already fitted → three fits,
    one per detection, not five rows."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _Stub:
        radius: float = 2.75
        radius_width: float = 0.11
        angle: float = 33.0
        angle_width: float = 6.0
        amplitude: float = 42.0
        A: float = 1.0
        B: float = 0.0
        C: float = 1.0
        theta: float = 0.0

    path = _open(main_window, synthetic_nexus_with_peaks)
    monkeypatch.setattr(
        main_window, "_run_pygidfit_for_selection",
        lambda sel, entry, frame: (_Stub(), None),
    )
    main_window.viewer._select_all_of_kind_on_frame("detected")
    main_window._on_batch_fit_2d()
    assert sorted(_table_ids(path, "fitted")) == [0, 1, 2]

    main_window.viewer.undo_last_action()
    assert sorted(_table_ids(path, "fitted")) == [0, 1]


# -- Deleting one side of the pair --------------------------------------

@pytest.mark.gui
def test_deleting_a_detected_peak_takes_its_fit(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    _confirm(monkeypatch)
    main_window._on_delete_peak_requested(_select_detected(main_window, 0))
    assert _table_ids(path, "detected") == [1, 2]
    assert _table_ids(path, "fitted") == [1]


@pytest.mark.gui
def test_deleting_a_fitted_peak_keeps_the_detection_by_default(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    """Discarding a fit says the prediction was wrong, not the
    detection — so the reverse cascade is opt-in."""
    path = _open(main_window, synthetic_nexus_with_peaks)
    _confirm(monkeypatch)
    fit = _table(path, "fitted")
    sel = SelectedPeak(
        kind="fitted", frame=0, peak_id=int(fit.ids[0]),
        radius=float(fit.radius[0]), angle=float(fit.angle[0]),
        radius_width=float(fit.radius_width[0]),
        angle_width=float(fit.angle_width[0]),
        is_ring=False, score=0.0, amplitude=0.0,
    )
    main_window.viewer._set_selected(sel)
    main_window._on_delete_peak_requested(sel)
    assert _table_ids(path, "fitted") == [1]
    assert _table_ids(path, "detected") == [0, 1, 2]

    # With the second setting on, the detection goes too.
    clean_link_settings.setValue("peakLinkFittedDeleteRemovesDetected", True)
    clean_link_settings.sync()
    fit = _table(path, "fitted")
    sel = SelectedPeak(
        kind="fitted", frame=0, peak_id=int(fit.ids[0]),
        radius=float(fit.radius[0]), angle=float(fit.angle[0]),
        radius_width=float(fit.radius_width[0]),
        angle_width=float(fit.angle_width[0]),
        is_ring=False, score=0.0, amplitude=0.0,
    )
    main_window.viewer._set_selected(sel)
    main_window._on_delete_peak_requested(sel)
    assert _table_ids(path, "fitted") == []
    assert _table_ids(path, "detected") == [0, 2]


@pytest.mark.gui
def test_undo_of_a_linked_delete_restores_both_under_their_ids(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    """Restoring under fresh ids would hand back two rows that are no
    longer a pair, so the undo would not have undone."""
    path = _open(main_window, synthetic_nexus_with_peaks)
    _confirm(monkeypatch)
    main_window._on_delete_peak_requested(_select_detected(main_window, 0))
    main_window.viewer.undo_last_action()
    assert sorted(_table_ids(path, "detected")) == [0, 1, 2]
    assert sorted(_table_ids(path, "fitted")) == [0, 1]

    main_window.viewer.redo_last_action()
    assert sorted(_table_ids(path, "detected")) == [1, 2]
    assert sorted(_table_ids(path, "fitted")) == [1]


@pytest.mark.gui
def test_with_the_link_off_a_detected_delete_leaves_the_fit(
    main_window, synthetic_nexus_with_peaks, monkeypatch, peak_link_off,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    _confirm(monkeypatch)
    main_window._on_delete_peak_requested(_select_detected(main_window, 0))
    assert _table_ids(path, "detected") == [1, 2]
    assert _table_ids(path, "fitted") == [0, 1]


# -- Clearing ------------------------------------------------------------

@pytest.mark.gui
def test_clearing_detected_also_clears_fitted_when_linked(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    _confirm(monkeypatch)
    main_window._action_clear_file_peaks("detected", "entry")
    assert _table_ids(path, "detected") == []
    assert _table_ids(path, "fitted") == []


@pytest.mark.gui
def test_clearing_detected_leaves_fitted_when_unlinked(
    main_window, synthetic_nexus_with_peaks, monkeypatch, peak_link_off,
):
    path = _open(main_window, synthetic_nexus_with_peaks)
    _confirm(monkeypatch)
    main_window._action_clear_file_peaks("detected", "entry")
    assert _table_ids(path, "detected") == []
    assert _table_ids(path, "fitted") == [0, 1]


# -- Matched, and the pipeline ------------------------------------------

@pytest.mark.gui
def test_a_linked_delete_remaps_matched_the_way_a_fitted_delete_does(
    main_window, synthetic_nexus_with_peaks, monkeypatch, clean_link_settings,
):
    """The link gives matched no new semantics. Deleting a detected peak
    removes its fit, and that removal goes through the same
    ``remap_matched_peak_lists`` a plain fitted delete has always used:
    the structure loses exactly that peak and keeps the rest."""
    path = _open(main_window, synthetic_nexus_with_peaks)
    ds_path = f"{ENTRY}/data/analysis/frame00000/matched_segments_0000"
    with main_window._detached_silx_tree():
        with h5py.File(path, "r+") as f:
            m = np.zeros(1, dtype=_MATCHED_DTYPE)
            m["CIF"] = b"phase.cif"
            m["probability"] = 0.9
            m["peak_list"][0] = np.asarray([0, 1], dtype=np.int32)
            f.create_dataset(ds_path, data=m)

    _confirm(monkeypatch)
    # Detected 0 pairs with fitted 0, which sits at position 0.
    main_window._on_delete_peak_requested(_select_detected(main_window, 0))
    assert _table_ids(path, "fitted") == [1]
    with h5py.File(path, "r") as f:
        row = f[ds_path][()][0]
    # Position 0 dropped, position 1 shifted down into it. The structure
    # itself survives with its remaining peak.
    assert [int(v) for v in row["peak_list"]] == [0]


@pytest.mark.gui
def test_a_fitting_run_relinks_the_fits_it_wrote(
    main_window, synthetic_nexus_with_peaks, clean_link_settings,
):
    """A pipeline run numbers fitted rows positionally. The completion
    handler re-keys them onto their detected peaks so the pairing the
    rest of the feature depends on survives the run."""
    from mlgidlab.pipeline import PipelineCommand

    path = _open(main_window, synthetic_nexus_with_peaks)
    # Stand in for what a run leaves behind: three fits, ids 0..N-1,
    # against detected ids with a gap.
    with main_window._detached_silx_tree():
        with h5py.File(path, "r+") as f:
            g = f[f"{ENTRY}/data/analysis/frame00000"]
            det = g["detected_peaks"][()]
            det["id"] = [3, 7, 11]
            g["detected_peaks"][...] = det
            fit = np.zeros(3, dtype=g["fitted_peaks"].dtype)
            fit["id"] = [0, 1, 2]
            fit["radius"] = [1.0, 2.0, 3.0]
            fit["angle"] = 45.0
            fit["angle_width"] = 5.0
            fit["radius_width"] = 0.2
            del g["fitted_peaks"]
            g.create_dataset("fitted_peaks", data=fit)

    # The real caller runs while the silx tree is still detached (it
    # reattaches only when the pipeline queue drains), so the helper
    # writes without detaching itself.
    with main_window._detached_silx_tree():
        main_window._rekey_fitted_after_fit(
            PipelineCommand("run_fitting", {"entry": ENTRY, "frame_num": 0})
        )
    assert _table_ids(path, "fitted") == [3, 7, 11]


@pytest.mark.gui
def test_a_fitting_run_leaves_the_ids_alone_when_unlinked(
    main_window, synthetic_nexus_with_peaks, peak_link_off,
):
    from mlgidlab.pipeline import PipelineCommand

    path = _open(main_window, synthetic_nexus_with_peaks)
    with main_window._detached_silx_tree():
        with h5py.File(path, "r+") as f:
            g = f[f"{ENTRY}/data/analysis/frame00000"]
            det = g["detected_peaks"][()]
            det["id"] = [3, 7, 11]
            g["detected_peaks"][...] = det
    with main_window._detached_silx_tree():
        main_window._rekey_fitted_after_fit(
            PipelineCommand("run_fitting", {"entry": ENTRY})
        )
    assert _table_ids(path, "fitted") == [0, 1]
