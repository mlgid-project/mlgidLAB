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
