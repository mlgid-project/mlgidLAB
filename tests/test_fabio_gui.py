"""Fabio images through the GUI: open -> separate raw entries -> view + convert.

GUI-side counterpart of ``test_fabio_stack``: a TIFF classifies as raw in the
CopyWorker, a RawSession of image files activates with one entry PER file
(built by ``_populate_raw_entries`` via ``list_fabio_entries``), the viewer
renders the selected image through a ``LazyFabioStack``, and the Conversion
panel yields one ``RawScan`` per checked image file.

Source: workers.py ``CopyWorker`` (fabio branch); main_window.py
``_populate_raw_entries`` (fabio branch); conversion_panel.py
``_refresh_selection_tree`` / ``_collect_scans`` / ``_expand_fabio_scans``.

Batch-open behavior (the 1000-image freeze fix): image paths classify
inline in ``_process_open_queue`` (no CopyWorker churn), browser rows
arrive via the chunked ``_queue_tree_inserts`` timer as display-only
``_ImageFileNode`` rows (no decode — silx's real rows pin the full
pixel data per file), clicking a row selects that image in the entry
combo, and the recent-files list updates in one write via
``_add_recent_files``.
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
    # Generous timeout: the first image loads on a worker thread and
    # CI's shared 2-core runners have blown a 5 s budget before.
    qtbot.waitUntil(
        lambda: main_window.viewer._raw_image_stack is not None,
        timeout=30000,
    )
    stack = main_window.viewer._raw_image_stack
    assert isinstance(stack, file_model.LazyFabioStack)
    assert stack.shape == (1, 48, 40)
    assert main_window.viewer.n_frames == 1
    assert float(stack[0].mean()) == 1.0  # first file auto-loaded


def test_fabio_conversion_collects_one_scan_per_file(main_window, qtbot, tmp_path):
    tiffs = _write_tiffs(tmp_path, n=3)
    _activate_fabio(main_window, tiffs)
    # Same slow-runner allowance as test_fabio_files_open_as_separate_entries.
    qtbot.waitUntil(
        lambda: main_window.viewer._raw_image_stack is not None,
        timeout=30000,
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


def test_select_all_checkbox_round_trip(main_window, qtbot, tmp_path):
    """The Selection section's select-all box: click checks every file
    (one `_collect_scans` covers the batch), click again unchecks, and
    manual per-item edits mirror back as checked/partial/unchecked."""
    tiffs = _write_tiffs(tmp_path, n=3)
    _activate_fabio(main_window, tiffs)
    qtbot.waitUntil(
        lambda: main_window.viewer._raw_image_stack is not None,
        timeout=30000,
    )
    panel = main_window.conversion_panel
    tree = panel.selection_tree
    box = panel.select_all_box

    assert box.isEnabled()
    assert box.checkState() == Qt.CheckState.Unchecked

    box.click()
    assert box.checkState() == Qt.CheckState.Checked
    assert all(
        tree.topLevelItem(i).checkState(0) == Qt.CheckState.Checked
        and tree.topLevelItem(i).child(0).checkState(0)
        == Qt.CheckState.Checked
        for i in range(3)
    )
    panel.frame_mode.setCurrentText(FRAME_ALL)
    assert [s.file_path for s in panel._collect_scans()] == tiffs

    box.click()
    assert box.checkState() == Qt.CheckState.Unchecked
    assert panel._collect_scans() == []

    # Manual edits reflect back on the box.
    tree.topLevelItem(0).child(0).setCheckState(0, Qt.CheckState.Checked)
    assert box.checkState() == Qt.CheckState.PartiallyChecked
    for i in (1, 2):
        tree.topLevelItem(i).child(0).setCheckState(
            0, Qt.CheckState.Checked
        )
    assert box.checkState() == Qt.CheckState.Checked
    tree.topLevelItem(1).child(0).setCheckState(0, Qt.CheckState.Unchecked)
    assert box.checkState() == Qt.CheckState.PartiallyChecked

    # A click from partial selects EVERYTHING (never lands on partial).
    box.click()
    assert box.checkState() == Qt.CheckState.Checked
    assert len(panel._collect_scans()) == 3


def test_batch_open_images_skips_worker_queue(main_window, qtbot, tmp_path):
    """``_open_paths`` with image files classifies inline on the GUI
    thread: no CopyWorker thread per file, the batch finalizes
    synchronously, browser rows arrive via the chunked insert queue,
    and the recents list records the batch."""
    tiffs = _write_tiffs(tmp_path, n=3)
    main_window._open_paths(list(tiffs))

    # Finalized synchronously — no worker thread was ever spawned.
    assert main_window._thread is None
    assert main_window._open_queue == []
    session = main_window.session
    assert session is not None and session.kind == "raw"
    assert session.raw_paths == [p.resolve() for p in tiffs]
    assert main_window.entry_combo.count() == 3

    # Tree rows land chunk by chunk off the insert timer.
    qtbot.waitUntil(lambda: not main_window._tree_insert_queue, timeout=10000)
    model = main_window.tree.findHdf5TreeModel()
    qtbot.waitUntil(lambda: model.rowCount() == 3, timeout=10000)

    # One batched recents write: last-opened first.
    items = main_window._load_recent_files()
    assert [i["path"] for i in items[:3]] == [
        str(p) for p in reversed(tiffs)
    ]


def test_recent_files_batch_matches_sequential_semantics(main_window, tmp_path):
    """``_add_recent_files`` produces the exact list sequential
    ``_add_recent_file`` pushes would: later items higher, capped at
    ``_MAX_RECENT_FILES``, duplicates bubble to their last occurrence."""
    window = main_window
    window._save_recent_files(
        [{"type": "raw", "path": f"/old/{i}.h5"} for i in range(4)]
    )
    batch = [tmp_path / f"b_{i:03d}.tif" for i in range(12)]
    window._add_recent_files(batch, "raw")
    items = window._load_recent_files()
    expected = [str(p) for p in reversed(batch)][: window._MAX_RECENT_FILES]
    assert [i["path"] for i in items] == expected

    window._save_recent_files([])
    a, b = tmp_path / "a.tif", tmp_path / "b.tif"
    window._add_recent_files([a, b, a], "raw")
    assert [i["path"] for i in window._load_recent_files()] == [str(a), str(b)]


def test_image_rows_are_lightweight_and_click_to_view(
    main_window, qtbot, tmp_path
):
    """Browser rows for image files are display-only ``_ImageFileNode``s
    (name + icon, never a decoded silx file), and selecting one switches
    the entry combo — and therefore the viewer — to that image."""
    from PySide6.QtCore import QItemSelectionModel, QModelIndex
    from silx.gui.hdf5 import Hdf5TreeModel as _SilxModel

    from mlgidlab.main_window import _ImageFileNode

    tiffs = _write_tiffs(tmp_path, n=3)
    main_window._open_paths(list(tiffs))
    qtbot.waitUntil(lambda: not main_window._tree_insert_queue, timeout=10000)
    model = main_window.tree.findHdf5TreeModel()
    qtbot.waitUntil(lambda: model.rowCount() == 3, timeout=10000)

    items = [
        model.data(
            model.index(r, 0, QModelIndex()), _SilxModel.H5PY_ITEM_ROLE
        )
        for r in range(3)
    ]
    assert all(isinstance(i, _ImageFileNode) for i in items)
    assert [i._display_name for i in items] == [p.name for p in tiffs]

    # Click the middle row (through the view's proxy): the combo — the
    # actual navigation surface — follows.
    view = main_window.tree
    proxy = view.model()
    view.selectionModel().select(
        proxy.index(1, 0), QItemSelectionModel.SelectionFlag.ClearAndSelect
    )
    assert main_window.entry_combo.currentText() == tiffs[1].name


def test_action_open_uses_fast_widget_dialog(
    main_window, qtbot, tmp_path, monkeypatch
):
    """File -> Open runs Qt's widget dialog (non-native, constant-time
    icon provider — native pickers thumbnail image files and crawl on
    big detector dirs) and feeds the picked files to ``_open_paths``."""
    from mlgidlab import main_window as mw_mod

    tiffs = _write_tiffs(tmp_path, n=3)
    seen: dict = {}

    def fake_exec(dlg):
        seen["non_native"] = dlg.testOption(
            mw_mod.QFileDialog.Option.DontUseNativeDialog
        )
        seen["provider"] = dlg.iconProvider()
        dlg.selectFile(str(tiffs[0]))
        return 1

    monkeypatch.setattr(mw_mod.QFileDialog, "exec", fake_exec)
    main_window._action_open()

    assert seen["non_native"] is True
    assert isinstance(seen["provider"], mw_mod._FastFileIconProvider)
    session = main_window.session
    assert session is not None and session.kind == "raw"
    assert session.raw_paths == [tiffs[0].resolve()]

    # The provider itself answers without touching file contents.
    provider = main_window._file_dialog_icons
    from PySide6.QtCore import QFileInfo

    assert not provider.icon(QFileInfo(str(tiffs[0]))).isNull()
    assert not provider.icon(QFileInfo(str(tmp_path))).isNull()


def test_browser_fill_progress_indicator(
    main_window, qtbot, tmp_path, monkeypatch
):
    """Image batches at or above ``_TREE_PROGRESS_MIN`` drive the
    status-bar "Browser: n/N files" indicator while rows stream in;
    it clears when the fill completes, and small batches never show
    it."""
    from mlgidlab.main_window import MainWindow

    monkeypatch.setattr(MainWindow, "_TREE_PROGRESS_MIN", 2)
    tiffs = _write_tiffs(tmp_path, n=3)
    main_window._open_paths(list(tiffs))
    # The finalize queued the rows synchronously; the drain timer has
    # not fired yet, so the indicator is up at 0/3.
    assert main_window._tree_insert_queue
    assert not main_window._sb_tree_bar.isHidden()
    assert main_window._sb_tree_label.text() == "Browser: 0/3 files"

    qtbot.waitUntil(lambda: not main_window._tree_insert_queue, timeout=10000)
    assert main_window._sb_tree_bar.isHidden()
    assert main_window._sb_tree_label.text() == ""

    # Below the threshold the indicator never appears.
    monkeypatch.setattr(MainWindow, "_TREE_PROGRESS_MIN", 10)
    sub = tmp_path / "b"
    sub.mkdir()
    more = _write_tiffs(sub, n=3)
    main_window._open_paths(list(more))
    assert main_window._tree_insert_queue
    assert main_window._sb_tree_bar.isHidden()
    qtbot.waitUntil(lambda: not main_window._tree_insert_queue, timeout=10000)


def test_detach_drops_pending_tree_inserts(main_window, qtbot, tmp_path):
    """Detaching the silx tree mid-fill drops the queued rows (they
    target the model being cleared); the reattach re-queues every
    session's files and the rows come back."""
    tiffs = _write_tiffs(tmp_path, n=3)
    main_window._open_paths(list(tiffs))

    main_window._detach_silx_tree()
    assert main_window._tree_insert_queue == []
    assert not main_window._tree_insert_timer.isActive()

    main_window._reattach_silx_tree()
    qtbot.waitUntil(lambda: not main_window._tree_insert_queue, timeout=10000)
    model = main_window.tree.findHdf5TreeModel()
    qtbot.waitUntil(lambda: model.rowCount() == 3, timeout=10000)
