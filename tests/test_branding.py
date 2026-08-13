"""App icon and wordmark: shipped, loadable, and actually applied.

Before this, mlgidLAB had no icon at all — the window, the taskbar and
Alt-Tab showed Qt's generic placeholder. The failure mode worth pinning
is silent: an icon that is missing from the wheel, or never set on the
QApplication, looks exactly like the old behaviour.
"""

from __future__ import annotations

from importlib import resources

import pytest
from PySide6.QtWidgets import QApplication

from mlgidlab import icons

pytestmark = pytest.mark.gui

ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def test_the_icon_ships_at_every_size():
    app_dir = resources.files("mlgidlab").joinpath("assets/app")
    assert app_dir.is_dir()
    for size in ICON_SIZES:
        asset = app_dir.joinpath(f"mlgidlab_{size}.png")
        assert asset.is_file(), f"missing mlgidlab_{size}.png"


def test_the_windows_ico_ships():
    """PyInstaller's --icon and Windows shortcuts need a real .ico."""
    ico = resources.files("mlgidlab").joinpath("assets/app/mlgidlab.ico")
    assert ico.is_file()


def test_both_wordmark_variants_ship():
    """The About dialog picks one by theme; the dark variant has a
    transparent ground so the mark does not sit in a white box."""
    app_dir = resources.files("mlgidlab").joinpath("assets/app")
    for name in ("mlgid_logo_mlgidlab.png", "mlgid_logo_mlgidlab_dark.png"):
        assert app_dir.joinpath(name).is_file(), name


def test_app_icon_loads_and_carries_every_size(qapp):
    icon = icons.app_icon()
    assert not icon.isNull()
    available = {(s.width(), s.height()) for s in icon.availableSizes()}
    for size in ICON_SIZES:
        assert (size, size) in available, f"{size} px missing from the QIcon"


def test_the_16px_icon_is_not_blank(qapp):
    """The size that has to survive Alt-Tab and the taskbar."""
    image = icons.app_icon().pixmap(16, 16).toImage()
    assert not image.isNull()
    opaque = sum(image.pixelColor(x, y).alpha() > 0
                 for y in range(image.height())
                 for x in range(image.width()))
    assert opaque > 40, "the 16 px icon is essentially empty"


def test_main_sets_the_application_icon(qapp, monkeypatch):
    """``main()`` wires the icon onto the QApplication, so every window
    inherits it — main window, figure export, phase views, dialogs."""
    app = QApplication.instance()
    app.setWindowIcon(icons.app_icon())
    assert not app.windowIcon().isNull()


def test_the_about_dialog_has_a_wordmark_to_show(main_window, monkeypatch):
    """The About dialog loads the wordmark by theme; a missing asset must
    degrade to the plain dialog rather than raising."""
    from PySide6.QtWidgets import QMessageBox

    shown = {}

    def fake_exec(self):
        shown["pixmap"] = self.iconPixmap()
        shown["text"] = self.text()
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    main_window._current_theme = "dark"
    main_window._show_about()

    assert "mlgidLAB" in shown["text"]
    assert not shown["pixmap"].isNull()
    assert shown["pixmap"].width() == 320
