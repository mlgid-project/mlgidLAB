"""App-level constants shared by the main window and its dialogs."""
from __future__ import annotations



APP_NAME = "mlgidLAB"
NEXUS_FILTER = "HDF5 / NeXus (*.h5 *.hdf5 *.nxs);;All files (*)"
# Open dialog auto-classifies NeXus vs raw HDF5 vs fabio image (TIFF/CBF/EDF);
# one combined filter does for all, with per-family filters below it.
OPEN_FILTER = (
    "All supported (*.h5 *.hdf5 *.nxs *.tif *.tiff *.cbf *.edf);;"
    "HDF5 / NeXus (*.h5 *.hdf5 *.nxs);;"
    "Detector images (*.tif *.tiff *.cbf *.edf);;"
    "All files (*)"
)

PLAYBACK_MODE_FRAME = "frame_interval_ms"
PLAYBACK_MODE_TOTAL = "total_time_s"
DEFAULT_PLAYBACK_FRAME_MS = 50          # 20 fps — was 100 ms / 10 fps
DEFAULT_PLAYBACK_TOTAL_S = 3.0
PLAYBACK_FRAME_MS_MIN = 10              # 100 fps requested ceiling
PLAYBACK_FRAME_MS_MAX = 2000            # 0.5 fps floor
PLAYBACK_TOTAL_S_MIN = 0.5
PLAYBACK_TOTAL_S_MAX = 600.0            # 10 minutes max

# Real tick cap. The eye stops perceiving extra frames much above
# ~20 fps, and large frames cannot be painted faster than ~50 ms
# regardless. When the user requests a faster per-frame rate (e.g.
# 3 s total over 300 frames = 10 ms / frame) we keep the timer at
# 50 ms and skip frames instead — see ``_compute_play_schedule``.
PLAYBACK_TICK_FLOOR_MS = 50

# QSettings keys for the playback preferences. Shared by MainWindow
# (class-attr aliases preserved there) and the Settings dialog.
PLAYBACK_MODE_KEY = "playbackMode"
PLAYBACK_FRAME_MS_KEY = "playbackFrameIntervalMs"
PLAYBACK_TOTAL_S_KEY = "playbackTotalTimeS"
