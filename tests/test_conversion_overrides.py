"""Conversion panel: orientation flips, value-driven overrides, PONI
autofill, and the append-frames output mode.

The flips moved OUT of the Manual-overrides subsection (routine
per-beamline settings, always honoured when checked); the remaining
override fields are value-driven — "(unset)" fields are skipped, set
fields are forwarded, no enable-checkbox gate. Loading a PONI pre-fills
the override fields with the values pygid would derive from it
(``parse_poni_overrides`` mirrors ``pygid.ExpParams`` math). Append
mode extends an existing entry instead of creating ``entry_NNNN``.
Source: conversion_panel.py ``_build_exp_params_section`` /
``_collect_config`` / ``parse_poni_overrides`` /
``_autofill_overrides_from_poni`` / ``_on_append_frames_toggled`` /
``_refresh_append_entries``; conversion.py ``_validate_append_target``.
"""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import pytest

from mlgidlab import conversion
from mlgidlab.conversion_panel import (
    OUTPUT_SEPARATE_DATASETS,
    ConversionConfig,
    parse_poni_overrides,
)

pytestmark = pytest.mark.gui


PONI_V2 = """\
# Calibration converted at ...
poni_version: 2
Detector: Detector
Detector_config: {{"pixel1": 7.5e-05, "pixel2": 7.5e-05, "max_shape": [1679, 1475]}}
Distance: {dist}
Poni1: {poni1}
Poni2: {poni2}
Rot1: {rot1}
Rot2: {rot2}
Rot3: 0.0
Wavelength: {wl}
"""


def _write_poni(tmp_path, **kw) -> Path:
    defaults = dict(dist=0.2871, poni1=0.084, poni2=0.0405, rot1=0.0, rot2=0.0, wl=9.6e-11)
    defaults.update(kw)
    path = tmp_path / "test.poni"
    path.write_text(PONI_V2.format(**defaults))
    return path


# -- parse_poni_overrides ----------------------------------------------


def test_parse_poni_no_rotation(tmp_path):
    vals = parse_poni_overrides(_write_poni(tmp_path))
    assert vals["SDD"] == pytest.approx(0.2871)
    assert vals["wavelength"] == pytest.approx(0.96)  # m → Å
    # rot = 0 → centers are just poni/pixel.
    assert vals["centerX"] == pytest.approx(0.0405 / 7.5e-05)
    assert vals["centerY"] == pytest.approx(0.084 / 7.5e-05)


def test_parse_poni_rotation_matches_pygid_formula(tmp_path):
    rot1, rot2 = 0.01, -0.02
    vals = parse_poni_overrides(_write_poni(tmp_path, rot1=rot1, rot2=rot2))
    sdd, px = 0.2871, 7.5e-05
    assert vals["centerX"] == pytest.approx((-sdd * math.tan(rot1) + 0.0405) / px)
    assert vals["centerY"] == pytest.approx(
        (sdd * math.tan(rot2) / math.cos(rot1) + 0.084) / px
    )


def test_parse_poni_without_pixel_size_yields_sdd_and_wavelength(tmp_path):
    path = tmp_path / "nopx.poni"
    path.write_text("Distance: 0.5\nWavelength: 1e-10\nPoni1: 0.1\nPoni2: 0.1\n")
    vals = parse_poni_overrides(path)
    assert set(vals) == {"SDD", "wavelength"}


# -- panel: flips + value-driven overrides + autofill -------------------


def test_flips_collected_without_override_fields(main_window):
    panel = main_window.conversion_panel
    panel.flip_lr.setChecked(True)
    panel.flip_ud.setChecked(True)

    cfg = panel._collect_config()

    assert cfg.expmeta_overrides == {"fliplr": True, "flipud": True}
    # The old enable-gate is gone — no checkable box to forget.
    assert not hasattr(panel, "_override_box")


def test_overrides_are_value_driven(main_window):
    panel = main_window.conversion_panel
    panel.over_centerX.setValue(737.5)
    panel.over_transp.setChecked(True)

    cfg = panel._collect_config()

    assert cfg.expmeta_overrides == {"centerX": 737.5, "transp": True}
    # Unset numeric fields stay absent.
    assert "SDD" not in cfg.expmeta_overrides


def test_poni_autofill_fills_override_fields(main_window, tmp_path):
    panel = main_window.conversion_panel
    poni = _write_poni(tmp_path)
    panel.poni_path.setText(str(poni))

    panel._autofill_overrides_from_poni()

    assert panel.over_SDD.value() == pytest.approx(0.2871)
    assert panel.over_wavelength.value() == pytest.approx(0.96)
    assert panel.over_centerX.value() == pytest.approx(540.0)
    assert panel.over_centerY.value() == pytest.approx(1120.0)


# -- panel: append-frames UI -------------------------------------------


def _existing_output(tmp_path, n_entries=2) -> Path:
    out = tmp_path / "converted.h5"
    with h5py.File(out, "w", track_order=True) as f:
        for i in range(n_entries):
            f.create_group(f"entry_{i:04d}/data")
    return out


def test_append_toggle_locks_overwrite_and_lists_entries(main_window, tmp_path):
    panel = main_window.conversion_panel
    _existing_output(tmp_path)
    panel.output_dir.setText(str(tmp_path))
    panel.output_mode_combo.setCurrentText(OUTPUT_SEPARATE_DATASETS)

    panel.append_frames_chk.setChecked(True)

    assert not panel.overwrite_file_chk.isEnabled()
    assert not panel.overwrite_file_chk.isChecked()
    assert not panel.overwrite_dataset_chk.isEnabled()
    items = [panel.append_entry_combo.itemText(i)
             for i in range(panel.append_entry_combo.count())]
    assert items == ["entry_0000", "entry_0001"]
    assert panel.append_entry_combo.currentText() == "entry_0001"  # last

    panel.append_frames_chk.setChecked(False)
    assert panel.overwrite_file_chk.isEnabled()


def test_append_config_collection_and_missing_entry_error(main_window, tmp_path):
    panel = main_window.conversion_panel
    _existing_output(tmp_path)
    panel.output_dir.setText(str(tmp_path))
    panel.output_mode_combo.setCurrentText(OUTPUT_SEPARATE_DATASETS)
    panel.append_frames_chk.setChecked(True)

    cfg = panel._collect_config()
    assert cfg.append_frames is True
    assert cfg.append_entry == "entry_0001"
    assert cfg.overwrite_file is False and cfg.overwrite_dataset is False

    # No target entry (empty combo) → collection refuses loudly.
    panel.append_entry_combo.clear()
    with pytest.raises(ValueError):
        panel._collect_config()


# -- engine: append-target validation (pure h5py, no pygid) -------------


def test_validate_append_target(tmp_path):
    out = _existing_output(tmp_path)
    cfg = ConversionConfig(append_frames=True, append_entry="entry_0001")

    # Happy path — single target, file + entry exist.
    conversion._validate_append_target(cfg, {Path("raw.h5"): out})

    # Multiple output files are ambiguous.
    with pytest.raises(ValueError, match="single output file"):
        conversion._validate_append_target(
            cfg, {Path("a.h5"): out, Path("b.h5"): tmp_path / "other.h5"}
        )

    # Output file missing.
    with pytest.raises(ValueError, match="does not exist"):
        conversion._validate_append_target(
            cfg, {Path("raw.h5"): tmp_path / "missing.h5"}
        )

    # Entry missing.
    cfg.append_entry = "entry_9999"
    with pytest.raises(ValueError, match="not found"):
        conversion._validate_append_target(cfg, {Path("raw.h5"): out})


# -- engine: single-entry output mode ------------------------------------


class _RecordingPygid:
    """Minimal pygid stand-in recording each conversion call's routing
    kwargs — lets the single-entry group/overwrite logic run in CI
    without the pipeline extra installed."""

    calls: list[dict]

    class ExpParams:
        def __init__(self, **kw):
            pass

    class CoordMaps:
        def __init__(self, params, **kw):
            pass

    class SampleMetadata:
        def __init__(self, **kw):
            pass

    class ExpMetadata:
        def __init__(self, **kw):
            pass

    def __init__(self):
        _RecordingPygid.calls = []
        outer = self

        class Conversion:
            def __init__(self, matrix=None, path=None, dataset=None,
                         frame_num=None):
                self._path = path

            def det2q_gid(self, **kwargs):
                outer.calls.append({"path": self._path, **kwargs})

        self.Conversion = Conversion


def test_single_entry_mode_routes_all_scans_into_one_group(
    tmp_path, monkeypatch
):
    """Every scan targets the SAME fresh entry; only the first call may
    overwrite (a later overwrite would truncate the growing stack)."""
    import sys

    from mlgidlab.conversion_panel import OUTPUT_SINGLE_ENTRY, RawScan

    fake = _RecordingPygid()
    monkeypatch.setitem(sys.modules, "pygid", fake)
    poni = _write_poni(tmp_path)
    scans = [
        RawScan(file_path=tmp_path / f"i{i}.tif", entry="", frame_num=None)
        for i in range(3)
    ]
    cfg = ConversionConfig(
        poni_path=poni,
        ai=0.1,
        output_mode=OUTPUT_SINGLE_ENTRY,
        output_dir=tmp_path,
        overwrite_file=True,
        overwrite_dataset=True,
    )
    written = conversion.execute(scans, cfg)

    assert written == [tmp_path / "converted.h5"]
    calls = _RecordingPygid.calls
    assert len(calls) == 3
    assert {c["h5_group"] for c in calls} == {"entry_0000"}
    assert [c["overwrite_file"] for c in calls] == [True, False, False]
    assert [c["overwrite_group"] for c in calls] == [True, False, False]

    # Append + single-entry are mutually exclusive.
    cfg.append_frames = True
    cfg.append_entry = "entry_0000"
    with pytest.raises(ValueError, match="mutually exclusive"):
        conversion.execute(scans, cfg)


def test_single_entry_mode_end_to_end_with_pygid(tmp_path):
    """Real pygid run: 3 TIFFs land as one 3-frame entry with per-frame
    frame_ind / angle_of_incidence / analysis groups, and an overwrite
    re-run replaces the stack instead of growing it."""
    pytest.importorskip("pygid")
    import fabio.tifimage
    import numpy as np

    from mlgidlab.conversion_panel import (
        CONV_DET2Q_GID,
        OUTPUT_SINGLE_ENTRY,
        RawScan,
    )

    rng = np.random.default_rng(0)
    tifs = []
    for i in range(3):
        p = tmp_path / f"img_{i}.tif"
        fabio.tifimage.TifImage(
            data=(rng.random((220, 180)) * 1000).astype(np.uint32)
        ).write(str(p))
        tifs.append(p)
    poni = tmp_path / "small.poni"
    poni.write_text(PONI_V2.format(
        dist=0.2871, poni1=0.0084, poni2=0.00675, rot1=0.0, rot2=0.0,
        wl=9.6e-11,
    ).replace("[1679, 1475]", "[220, 180]"))

    cfg = ConversionConfig(
        conv_type=CONV_DET2Q_GID,
        poni_path=poni,
        ai=0.12,
        output_mode=OUTPUT_SINGLE_ENTRY,
        output_dir=tmp_path,
        output_filename="single.h5",
        overwrite_file=True,
        # The fixture poni carries the generic "Detector" name, which
        # pygid cannot map to a pixel size — forward it explicitly via
        # the same override channel the panel uses.
        expmeta_overrides={"px_size": 7.5e-05},
    )
    scans = [RawScan(file_path=p, entry="", frame_num=None) for p in tifs]
    written = conversion.execute(scans, cfg)

    assert written == [tmp_path / "single.h5"]
    with h5py.File(written[0], "r") as f:
        entries = [k for k in f.keys() if k.startswith("entry")]
        assert entries == ["entry_0000"]
        g = f["entry_0000"]
        assert g["data/img_gid_q"].shape[0] == 3
        assert list(g["data/frame_ind"][()]) == [0.0, 1.0, 2.0]
        assert list(g["instrument/angle_of_incidence"][()]) == [0.12] * 3
        assert sorted(g["data/analysis"].keys()) == [
            "frame00000", "frame00001", "frame00002"
        ]

    # Overwrite re-run replaces (no growth to 6 frames).
    conversion.execute(scans, cfg)
    with h5py.File(written[0], "r") as f:
        assert f["entry_0000/data/img_gid_q"].shape[0] == 3
        assert [k for k in f.keys() if k.startswith("entry")] == [
            "entry_0000"
        ]
