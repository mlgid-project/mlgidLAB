"""Interactive raw-preview flip: the Conversion panel's fliplr/flipud
checkboxes flip the live raw preview to match the orientation pygid's
conversion produces.

Covers:
  - the pure orientation helper ``_apply_raw_flips`` (mirrors pygid's
    ``process_image`` order: flipud then fliplr on the file-order frame);
  - ``GIWAXSImageViewer.set_raw_flips`` re-rendering the displayed frame;
  - ``ConversionPanel.rawFlipsChanged`` emission on toggle;
  - the MainWindow panel -> viewer wiring.

Source: image_viewer.py ``_apply_raw_flips`` / ``set_raw_flips`` /
``_render_frame`` raw branch; conversion_panel.py ``_emit_raw_flips``;
main_window.py ``_on_raw_flips_changed``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlgidlab.image_viewer import _apply_raw_flips

pytestmark = pytest.mark.gui


def test_apply_raw_flips_matches_numpy_for_all_combinations():
    arr = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)  # (H, W)
    assert np.array_equal(_apply_raw_flips(arr, False, False), arr)
    assert np.array_equal(_apply_raw_flips(arr, True, False), np.fliplr(arr))
    assert np.array_equal(_apply_raw_flips(arr, False, True), np.flipud(arr))
    # both -> flipud then fliplr == 180 deg rotation (order-independent)
    assert np.array_equal(
        _apply_raw_flips(arr, True, True), np.fliplr(np.flipud(arr))
    )
    # input is not mutated
    assert np.array_equal(arr, np.arange(3 * 4, dtype=np.float32).reshape(3, 4))


def test_viewer_set_raw_flips_reorients_preview(qtbot):
    from mlgidlab.image_viewer import GIWAXSImageViewer

    viewer = GIWAXSImageViewer()
    qtbot.addWidget(viewer)
    # distinctive (N, H, W) so every orientation is distinguishable
    stack = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    viewer.show_raw_stack(stack)

    def displayed():
        return np.asarray(viewer._view.getImageItem().image)

    frame = stack[0]
    # default: shown as stored (transpose only), no flip
    assert not viewer._raw_flip_lr and not viewer._raw_flip_ud
    assert np.array_equal(displayed(), frame.T)
    # fliplr only
    viewer.set_raw_flips(True, False)
    assert np.array_equal(displayed(), np.fliplr(frame).T)
    # flipud only
    viewer.set_raw_flips(False, True)
    assert np.array_equal(displayed(), np.flipud(frame).T)
    # both
    viewer.set_raw_flips(True, True)
    assert viewer._raw_flip_lr and viewer._raw_flip_ud
    assert np.array_equal(displayed(), np.fliplr(np.flipud(frame)).T)
    # revert
    viewer.set_raw_flips(False, False)
    assert np.array_equal(displayed(), frame.T)


def test_conversion_panel_emits_raw_flips(qtbot):
    from mlgidlab.conversion_panel import ConversionPanel

    panel = ConversionPanel()
    qtbot.addWidget(panel)
    seen: list[tuple[bool, bool]] = []
    panel.rawFlipsChanged.connect(lambda lr, ud: seen.append((lr, ud)))
    panel.flip_lr.setChecked(True)
    panel.flip_ud.setChecked(True)
    panel.flip_lr.setChecked(False)
    assert seen == [(True, False), (True, True), (False, True)]


def test_main_window_wires_flips_to_viewer(main_window):
    panel = main_window.conversion_panel
    viewer = main_window.viewer
    panel.flip_lr.setChecked(True)
    assert viewer._raw_flip_lr is True
    panel.flip_ud.setChecked(True)
    assert viewer._raw_flip_ud is True
    panel.flip_lr.setChecked(False)
    panel.flip_ud.setChecked(False)
    assert viewer._raw_flip_lr is False and viewer._raw_flip_ud is False
