"""The colour-grid popup used by the matched-peaks legend swatches:
preset grid contents, and the pick / Automatic signals."""
from __future__ import annotations

import pytest

from mlgidlab.color_picker import ColorGridPopup, grid_colors
from mlgidlab.image_viewer import MATCHED_PALETTE

pytestmark = pytest.mark.gui


def test_grid_colors_cover_palette_and_shades():
    """40 unique presets, with every automatic palette colour among
    them so a reset-equivalent pick is always one click away."""
    colors = grid_colors()
    assert len(colors) == 4 * len(MATCHED_PALETTE)
    assert len(set(colors)) == len(colors)
    for base in MATCHED_PALETTE:
        assert base in colors


def test_swatch_click_emits_its_color(qtbot):
    popup = ColorGridPopup()
    qtbot.addWidget(popup)
    picked: list[str] = []
    popup.colorPicked.connect(picked.append)
    target = popup._swatch_buttons[7]
    target.click()
    assert picked == [grid_colors()[7]]


def test_automatic_emits_reset(qtbot):
    popup = ColorGridPopup()
    qtbot.addWidget(popup)
    picked: list[str] = []
    reset: list[bool] = []
    popup.colorPicked.connect(picked.append)
    popup.resetPicked.connect(lambda: reset.append(True))
    popup._auto_button.click()
    assert reset == [True]
    assert picked == []
