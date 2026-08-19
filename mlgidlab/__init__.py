from __future__ import annotations

import logging
import os
import sys

# Disable HDF5's SWMR file locking. silx's Hdf5TreeModel opens each
# loaded file as ``r`` and pygid (via mlgidbase) reopens the same
# path as ``r+`` for the Figure Export window's renderer; with
# default locking on, the second open fails with
# "file is already open for read-only". Setting this before any
# ``h5py`` import means neither side acquires the OS-level lock that
# would block the other. Must be set before the first ``h5py``
# import — keep it at the very top of the package init.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from PySide6.QtWidgets import QApplication

__version__ = "0.1.0a16"


def main() -> int:
    from PySide6.QtCore import QSettings, QTimer
    from pathlib import Path
    from mlgidlab.main_window import MainWindow
    from mlgidlab.theme import apply_dark_theme, apply_light_theme

    app = QApplication(sys.argv)
    # Reclaim working-copy temp dirs leaked by a previous run that was killed
    # before its graceful close ran (see mlgidlab.session). Only removes dirs
    # whose owning PID is dead, so a second running instance is left alone.
    try:
        import logging
        from mlgidlab import session as _session
        _n = _session.sweep_stale_temp_dirs()
        if _n:
            logging.getLogger("mlgidlab").info(
                "Reclaimed %d stale temp working-copy dir(s) from a prior run.", _n
            )
    except Exception:
        pass
    # Set both org + app names so QSettings has a stable key path on
    # every platform (used by the Recent files menu, may grow other
    # persisted preferences over time).
    app.setOrganizationName("mlgidLAB")
    app.setApplicationName("mlgidLAB")
    # Window, taskbar and Alt-Tab icon. Set on the application so every
    # window inherits it (main window, figure export, phase views, the
    # dialogs). Guarded: a packaging slip should cost an icon, not a
    # startup.
    try:
        from mlgidlab.icons import app_icon
        app.setWindowIcon(app_icon())
    except Exception:
        logging.getLogger("mlgidlab").debug(
            "could not set the application icon", exc_info=True)
    if sys.platform == "win32":
        # Without an explicit AppUserModelID, Windows groups the app
        # under python.exe's icon in the taskbar no matter what
        # setWindowIcon says.
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "mlgidLAB.mlgidLAB")
        except Exception:
            logging.getLogger("mlgidlab").debug(
                "could not set the Windows AppUserModelID", exc_info=True)
    # Honor the persisted theme choice (View → Theme). Defaults to
    # dark; the menu sync inside MainWindow reads the same key.
    theme = str(QSettings().value("theme", "dark")).lower()
    if theme == "light":
        apply_light_theme(app)
    else:
        apply_dark_theme(app)
    window = MainWindow()
    # Tell MainWindow which theme is now live so its View → Theme
    # menu shows the right checked entry on first open.
    window._current_theme = theme if theme in ("dark", "light") else "dark"
    # Re-sync menu checkmark (the menu was built before _current_theme
    # was set on this path; the default check goes to Dark, so update
    # if necessary).
    if window._current_theme == "light":
        window.action_theme_light.setChecked(True)
    else:
        window.action_theme_dark.setChecked(True)
    # The welcome page was seeded during __init__, before the theme above
    # was known; re-seed so a light-theme user gets the light wordmark.
    window._refresh_welcome_view()
    window.show()
    # Post-update changelog (offline) + async "newer release available"
    # check. Deferred into the event loop so it runs after the window is up
    # and never blocks show(). Not started from MainWindow.__init__ so the
    # headless test fixture never performs a network call.
    QTimer.singleShot(0, window._run_startup_update_checks)
    # argv[0] is the program; treat any extra args as files to open.
    paths = [Path(a) for a in sys.argv[1:] if Path(a).exists()]
    if paths:
        # Defer into the event loop: _open_paths spawns the async
        # CopyWorker and touches widgets, which is only safe once
        # exec() is running. A 0 ms single-shot timer runs the
        # callback on the first loop iteration, after exec() starts.
        QTimer.singleShot(0, lambda: window._open_paths(paths))
    return app.exec()
