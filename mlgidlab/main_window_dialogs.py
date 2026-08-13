"""Small dialogs owned by the main window: peak export and app settings.
Moved out of ``main_window`` in the 2026 source split.
"""
from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)
from mlgidlab.main_window_constants import (
    PLAYBACK_FRAME_MS_KEY,
    PLAYBACK_MODE_KEY,
    PLAYBACK_TOTAL_S_KEY,
    DEFAULT_PLAYBACK_FRAME_MS,
    DEFAULT_PLAYBACK_TOTAL_S,
    PLAYBACK_FRAME_MS_MAX,
    PLAYBACK_FRAME_MS_MIN,
    PLAYBACK_MODE_FRAME,
    PLAYBACK_MODE_TOTAL,
    PLAYBACK_TOTAL_S_MAX,
    PLAYBACK_TOTAL_S_MIN,
)


class _ExportPeaksDialog(QDialog):
    """Modal kind/scope picker for Tools → Export peaks as CSV.

    Two QButtonGroups hold the kind (Detected/Fitted/Matched) and the
    scope (Active frame / Active entry / All entries) respectively;
    Active-frame is greyed when the active stack has only one frame
    so the option doesn't masquerade as different from Active-entry.
    """

    def __init__(self, parent: QWidget, *, has_multiple_frames: bool) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export peaks as CSV")
        layout = QVBoxLayout(self)

        kind_box = QGroupBox("Peak kind")
        kind_layout = QVBoxLayout(kind_box)
        self._rb_detected = QRadioButton("Detected")
        self._rb_fitted = QRadioButton("Fitted (with fit errors)")
        self._rb_matched = QRadioButton("Matched (flattened: one row per peak)")
        self._rb_fitted.setChecked(True)
        for rb in (self._rb_detected, self._rb_fitted, self._rb_matched):
            kind_layout.addWidget(rb)
        layout.addWidget(kind_box)

        scope_box = QGroupBox("Scope")
        scope_layout = QVBoxLayout(scope_box)
        self._rb_frame = QRadioButton("Active frame")
        self._rb_entry = QRadioButton("Active entry (all frames)")
        self._rb_all = QRadioButton("All entries (one combined CSV)")
        self._rb_entry.setChecked(True)
        if not has_multiple_frames:
            self._rb_frame.setEnabled(False)
            self._rb_frame.setToolTip(
                "Available only when the active entry has more than one frame."
            )
        for rb in (self._rb_frame, self._rb_entry, self._rb_all):
            scope_layout.addWidget(rb)
        layout.addWidget(scope_box)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def selected_kind(self) -> str:
        if self._rb_detected.isChecked():
            return "detected"
        if self._rb_matched.isChecked():
            return "matched"
        return "fitted"

    def selected_scope(self) -> str:
        if self._rb_frame.isChecked():
            return "frame"
        if self._rb_all.isChecked():
            return "all"
        return "entry"


# Playback settings persisted via QSettings under the keys below. The
# defaults give a 2× speed-up over the previous fixed 100 ms interval
# while still leaving headroom for cold-cache disk reads (~70-100 ms
# per fresh frame on local SSD). Users who want true frame-by-frame
# stepping can dial Frame interval up; users who want a fixed total
# duration (e.g. 5 s overview regardless of frame count) can flip to
# Total play time.


class _SettingsDialog(QDialog):
    """Application-wide settings dialog.

    Currently only carries the frame-playback section, but its
    layout reserves room for future settings groups (rendering,
    pipeline defaults, etc.) so adding a new section is just
    appending another ``QGroupBox`` to the outer layout.

    On accept, every changed value is written back to QSettings and
    the host MainWindow is told to re-apply (so an in-flight
    playback timer picks up the new interval immediately).
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(380)

        outer = QVBoxLayout(self)

        # --- Playback section -------------------------------------------------
        playback_box = QGroupBox("Frame playback")
        playback_layout = QVBoxLayout(playback_box)

        hint = QLabel(
            "<i>Controls the speed of the Display-dock Play button.</i>"
        )
        hint.setWordWrap(True)
        playback_layout.addWidget(hint)

        # Two mutually-exclusive modes. The active radio's spinbox is
        # the one that takes effect; the other stays editable so the
        # user can flip between modes without losing their values.
        mode_box = QButtonGroup(self)
        mode_box.setExclusive(True)
        self._rb_frame = QRadioButton("Time per frame")
        self._rb_total = QRadioButton("Total play time")
        mode_box.addButton(self._rb_frame)
        mode_box.addButton(self._rb_total)
        self._rb_frame.toggled.connect(self._refresh_enabled)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._spin_frame_ms = QSpinBox()
        self._spin_frame_ms.setRange(
            PLAYBACK_FRAME_MS_MIN, PLAYBACK_FRAME_MS_MAX
        )
        self._spin_frame_ms.setSingleStep(10)
        self._spin_frame_ms.setSuffix(" ms")
        self._spin_frame_ms.setToolTip(
            "Time spent on each frame. Lower = faster playback. The 10 ms "
            "lower bound caps playback at 100 fps; cold-cache disk reads "
            "may stutter below ~50 ms on large files."
        )

        self._spin_total_s = QDoubleSpinBox()
        self._spin_total_s.setRange(
            PLAYBACK_TOTAL_S_MIN, PLAYBACK_TOTAL_S_MAX
        )
        self._spin_total_s.setSingleStep(0.5)
        self._spin_total_s.setDecimals(2)
        self._spin_total_s.setSuffix(" s")
        self._spin_total_s.setToolTip(
            "Total time to traverse the whole stack (first frame → last "
            "frame). The per-frame interval is computed at play-start "
            "from the active entry's frame count, so swapping entries "
            "automatically adjusts the speed."
        )

        form.addRow(self._rb_frame, self._spin_frame_ms)
        form.addRow(self._rb_total, self._spin_total_s)
        playback_layout.addLayout(form)
        outer.addWidget(playback_box)

        # --- Buttons + outer wiring -------------------------------------------
        outer.addStretch(1)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

        # Load current values from QSettings (with defaults).
        settings = QSettings()
        mode = settings.value(
            PLAYBACK_MODE_KEY, PLAYBACK_MODE_FRAME
        )
        # QSettings returns strings on Linux but raw types on macOS;
        # coerce defensively.
        try:
            frame_ms = int(settings.value(
                PLAYBACK_FRAME_MS_KEY, DEFAULT_PLAYBACK_FRAME_MS
            ))
        except (TypeError, ValueError):
            frame_ms = DEFAULT_PLAYBACK_FRAME_MS
        try:
            total_s = float(settings.value(
                PLAYBACK_TOTAL_S_KEY, DEFAULT_PLAYBACK_TOTAL_S
            ))
        except (TypeError, ValueError):
            total_s = DEFAULT_PLAYBACK_TOTAL_S
        # Clamp into the spinbox range so out-of-bounds stored values
        # don't silently revert to the spinbox minimum.
        frame_ms = max(PLAYBACK_FRAME_MS_MIN,
                       min(PLAYBACK_FRAME_MS_MAX, frame_ms))
        total_s = max(PLAYBACK_TOTAL_S_MIN,
                      min(PLAYBACK_TOTAL_S_MAX, total_s))
        self._spin_frame_ms.setValue(frame_ms)
        self._spin_total_s.setValue(total_s)
        if mode == PLAYBACK_MODE_TOTAL:
            self._rb_total.setChecked(True)
        else:
            self._rb_frame.setChecked(True)
        self._refresh_enabled()

    def _refresh_enabled(self) -> None:
        frame_active = self._rb_frame.isChecked()
        self._spin_frame_ms.setEnabled(frame_active)
        self._spin_total_s.setEnabled(not frame_active)

    def save_to_qsettings(self) -> None:
        """Write the dialog's current values to QSettings.

        Called by the host on accept. Stores both spinbox values so a
        later mode-flip preserves the user's last value in each mode.
        """
        settings = QSettings()
        mode = PLAYBACK_MODE_FRAME if self._rb_frame.isChecked() else PLAYBACK_MODE_TOTAL
        settings.setValue(PLAYBACK_MODE_KEY, mode)
        settings.setValue(
            PLAYBACK_FRAME_MS_KEY, int(self._spin_frame_ms.value())
        )
        settings.setValue(
            PLAYBACK_TOTAL_S_KEY, float(self._spin_total_s.value())
        )
