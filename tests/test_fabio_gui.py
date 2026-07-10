"""Fabio images through the GUI: open -> separate raw entries -> view + convert.

GUI-side counterpart of ``test_fabio_stack``: a TIFF classifies as raw in the
CopyWorker, a RawSession of image files activates with one entry PER file
(built by ``_populate_raw_entries`` via ``list_fabio_entries``), the viewer
renders the selected image through a ``LazyFabioStack``, and the Conversion
panel yields one ``RawScan`` per checked image file.

Source: workers.py ``CopyWorker`` (fabio branch); main_window.py
``_populate_raw_entries`` (fabio branch); conversion_panel.py
``_refresh_selection_tree`` / ``_collect_scans`` / ``_expand_fabio_scans``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt

from mlgidlab import file_model
from mlgidlab.conversion_panel import FRAME_ALL
from mlgidlab.session import RawSession
from mlgidlab.workers import CopyWorker

pytestmark = pytest.mark.gui


def _write_tiffs(tmp_path, n=3, h=48, w=40) -> list[Path]:
    import fabio

    paths: list[Path] = []
    for i in range(n):
        p = tmp_path / f"scan_{i:04d}.tif"
        fabio.tifimage.TifImage(
            data=np.full((h, w), i + 1, dtype=np.int32)
        ).write(str(p))
        paths.append(p)
    return paths


def _activate_fabio(window, paths):
    """Install a fabio RawSession (no pre-cached walk — the fabio branch of
    ``_populate_raw_entries`` builds the per-file entries itself)."""
    session = RawSession.open(paths)
    window._sessions.append(session)
    window._set_active_session(session)
    return session


def test_copyworker_classifies_tiff_as_raw(qtbot, tmp_path):
    """A standalone TIFF classifies as raw with no pre-walked entries (the
    entry is built lazily at activation)."""
    (tiff,) = _write_tiffs(tmp_path, n=1)
    worker = CopyWorker(tiff)
    got: dict = {}
    worker.finished.connect(got.update)
    worker.run()

    assert got["error"] is None
    assert got["kind"] == "raw"
    assert got["session"] is None
    assert got["raw_entries"] is None


def test_fabio_files_open_as_separate_entries(main_window, qtbot, tmp_path):
    tiffs = _write_tiffs(tmp_path, n=3)
    session = _activate_fabio(main_window, tiffs)

    assert session.kind == "raw"
    # One entry per file (separate, NOT a combined stack).
    assert main_window.entry_combo.count() == 3
    labels = [main_window.entry_combo.itemText(i) for i in range(3)]
    assert labels == [p.name for p in tiffs]
    qtbot.waitUntil(
        lambda: main_window.viewer._raw_image_stack is not None, timeout=5000
    )
    stack = main_window.viewer._raw_image_stack
    assert isinstance(stack, file_model.LazyFabioStack)
    assert stack.shape == (1, 48, 40)
    assert main_window.viewer.n_frames == 1
    assert float(stack[0].mean()) == 1.0  # first file auto-loaded


def test_fabio_conversion_collects_one_scan_per_file(main_window, qtbot, tmp_path):
    tiffs = _write_tiffs(tmp_path, n=3)
    _activate_fabio(main_window, tiffs)
    qtbot.waitUntil(
        lambda: main_window.viewer._raw_image_stack is not None, timeout=5000
    )

    panel = main_window.conversion_panel
    tree = panel.selection_tree
    # One top-level file group per image, each with a single "image (1 frame)".
    assert tree.topLevelItemCount() == 3
    for i in range(3):
        file_item = tree.topLevelItem(i)
        assert file_item.childCount() == 1
        child = file_item.child(0)
        assert "image (1 frame)" in child.text(0)
        child.setCheckState(0, Qt.CheckState.Checked)

    panel.frame_mode.setCurrentText(FRAME_ALL)
    scans = panel._collect_scans()
    assert [s.file_path for s in scans] == tiffs
    assert all(s.entry == "" and s.frame_num is None for s in scans)

    # Checking just one file yields only that file's scan.
    for i in range(3):
        tree.topLevelItem(i).child(0).setCheckState(
            0, Qt.CheckState.Checked if i == 1 else Qt.CheckState.Unchecked
        )
    one = panel._collect_scans()
    assert [s.file_path for s in one] == [tiffs[1]]
