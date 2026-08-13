"""Viewer interaction: manual peaks, undo/redo, selection state, the shared action helpers, mouse/label event handling and the resizable ROI.

Plain mixin over ``GIWAXSImageViewer``: no __init__, no Signals; all state
lives on the combined class. Split out of ``image_viewer`` in the 2026
source split.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor
from mlgidlab.file_model import PeakTable, _LazyPolarStack
from mlgidlab.polar import polar_to_qxyz
from mlgidlab.viewer_items import (
    FileGeomAction,
    ManualAddAction,
    ManualGeomAction,
    ManualPeak,
    ManualRemoveAction,
    ManualReplaceAction,
    SelectedPeak,
    _cart_box_contains,
    _cart_table_row_contains,
    _polar_box_contains,
    _polar_table_row_contains,
)
from mlgidlab.viewer_styles import (
    MODE_CARTESIAN,
    MODE_POLAR,
    MODE_RAW,
    SELECTION_STYLE,
    selection_style,
)


class ViewerInteractMixin:
    # -- Manual peaks --

    def manual_peaks(self, frame: int) -> list[ManualPeak]:
        return list(self._manual_peaks.get(frame, []))

    def add_manual_peak(self, frame: int, peak: ManualPeak) -> None:
        self._undoable_add_manual(frame, peak)
        self._push_undo(ManualAddAction(frame=frame, peak=peak))

    def remove_manual_peak(self, frame: int, peak: ManualPeak) -> None:
        if peak not in self._manual_peaks.get(frame, []):
            return
        self._undoable_remove_manual(frame, peak)
        self._push_undo(ManualRemoveAction(frame=frame, peak=peak))

    def angular_extent(self) -> tuple[float, float] | None:
        """Return ``(angle_min_deg, angle_max_deg)`` for the active polar
        stack. Used by the host to size ring-mode expansions to the
        actual displayed angular range — converted files vary
        (``[0, 90]`` for the upper-right quadrant; ``[-180, 180]`` for
        full-quadrant data). Returns None when no polar stack is
        currently rendered (raw mode, no file open).
        """
        if self._polar_cache is None:
            return None
        _, _, angle = self._polar_cache
        if angle.size == 0:
            return None
        return float(angle[0]), float(angle[-1])

    def set_manual_geometry(
        self,
        peak: ManualPeak,
        radius: float,
        angle: float,
        radius_width: float,
        angle_width: float,
        is_ring: bool,
    ) -> None:
        """Mutate every geometry field on ``peak`` (including ``is_ring``)
        and trigger the standard refresh path. Skips the undo stack —
        this is for transient state changes driven by UI toggles
        (e.g. the ring checkbox), not user-initiated edits that should
        be reversible via Ctrl+Z. The host stashes its own pre-state
        when it needs to revert.

        Mirrors `_apply_manual_geom` but adds `is_ring` so we can flip
        the ring/segment kind in lockstep with the angular sweep.
        """
        peak.radius = radius
        peak.angle = angle
        peak.radius_width = radius_width
        peak.angle_width = angle_width
        peak.is_ring = is_ring
        if (
            self._selected is not None
            and self._selected.kind == "manual"
            and self._selected.manual_ref is peak
        ):
            self._selected.radius = radius
            self._selected.angle = angle
            self._selected.radius_width = radius_width
            self._selected.angle_width = angle_width
            self._selected.is_ring = is_ring
            self._sync_roi_geometry()
        # Find the frame this peak lives on so the overlay refreshes
        # against the right bucket.
        for fr, peaks in self._manual_peaks.items():
            if peak in peaks:
                if fr == self.current_frame:
                    self._render_overlays(fr)
                break
        if (
            self._selected is not None
            and self._selected.kind == "manual"
            and self._selected.manual_ref is peak
        ):
            self.peakGeometryChanged.emit(self._selected)

    def undo_last_action(self) -> None:
        """Reverse the most recent action. No-ops if empty."""
        if self._busy or not self._undo_stack:
            return
        action = self._undo_stack.pop()
        action.undo(self)
        self._redo_stack.append(action)

    def redo_last_action(self) -> None:
        """Re-apply the most recently undone action."""
        if self._busy or not self._redo_stack:
            return
        action = self._redo_stack.pop()
        action.redo(self)
        self._undo_stack.append(action)

    def clear_history(self) -> None:
        """Drop both undo and redo stacks. Called after pipeline ops that
        regenerate peak ids — pending FileGeomActions would key off stale ids.

        For ordinary peak edits (delete / paste / fit), prefer
        ``prune_file_geom_history`` instead: file-resident ids are stable
        (``add_*`` appends ``max(id)+1``, ``delete`` filters by id and
        leaves the rest), so only the *removed* ids' geometry edits go
        stale — wiping the whole history would needlessly break
        multi-level undo/redo.
        """
        self._undo_stack.clear()
        self._redo_stack.clear()

    def prune_file_geom_history(
        self, frame: int, kind: str, peak_ids
    ) -> None:
        """Drop only the FileGeomActions that target ``(frame, kind, id)``
        for ``id in peak_ids`` from both stacks, keeping everything else.

        Used after a delete: the removed ids' pending geometry edits would
        key off rows that no longer exist (and would otherwise raise a
        stale-id KeyError on the deferred write). Unlike ``clear_history``
        this preserves prior undoable operations, so a run of deletes (or
        deletes interleaved with pastes / fits) stays fully undoable and
        redoable.
        """
        targets = {int(p) for p in peak_ids}

        def _stale(a: _Action) -> bool:
            return (
                isinstance(a, FileGeomAction)
                and int(a.frame) == int(frame)
                and a.kind == kind
                and int(a.peak_id) in targets
            )

        self._undo_stack = [a for a in self._undo_stack if not _stale(a)]
        self._redo_stack = [a for a in self._redo_stack if not _stale(a)]

    def clear_selection(self) -> None:
        if self._selected is None and not self._selected_extras:
            return
        self._selected = None
        self._selected_extras = []
        self._fitted_preview_geom = None
        self._fitted_preview_is_ring = False
        self._fitted_preview_extras_geoms = []
        self._sync_roi()
        self._render_overlays(self.current_frame)
        self.selectionChanged.emit(None)
        self.selectionsChanged.emit([])

    def clear_all_manual_peaks(self) -> None:
        """Drop every manual peak across all frames + the undo history.

        Matches Tools → Clear all manual peaks. Manual peaks are
        in-memory only, so no file write is involved. The selection is
        also cleared if it pointed at a manual peak.
        """
        if not self._manual_peaks:
            # Still clear undo history of any orphaned ManualGeomActions
            # and refresh in case overlays drift.
            self._undo_stack.clear()
            self._redo_stack.clear()
            return
        self._manual_peaks.clear()
        if self._selected is not None and self._selected.kind == "manual":
            self._selected = None
            self._fitted_preview_geom = None
            self._fitted_preview_is_ring = False
            self._sync_roi()
            self.selectionChanged.emit(None)
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._render_overlays(self.current_frame)

    def set_fitted_preview(
        self,
        center_r: float | None,
        width_r: float | None,
        center_a: float | None,
        width_a: float | None,
        *,
        is_ring: bool = False,
    ) -> None:
        """Paint the dashed cyan preview of the would-be fitted_peaks box.

        Pure painter: the box is drawn at ``(width_r × width_a)``
        verbatim, centred at ``(center_r, center_a)``. All convention
        math (``2σ_r × 2σ_a`` for both modes; fallback to the user's
        drawn box when scipy's 1D fit hasn't converged) lives in the
        caller (``MainWindow._update_fitted_preview``) so a single
        place owns the rules.

        Pass any of ``center_r`` / ``width_r`` as ``None`` to hide.
        Pass ``is_ring=True`` to draw a full-angular-sweep ring at the
        canonical ``angle = 45°, angle_width = ∞`` — the caller still
        passes a sentinel for ``center_a`` / ``width_a`` (they're
        ignored).
        """
        if (
            center_r is None or width_r is None
            or not (np.isfinite(center_r) and np.isfinite(width_r) and width_r > 0)
        ):
            self._fitted_preview_geom = None
            self._fitted_preview_is_ring = False
        elif is_ring:
            # Ring path: angular fit isn't required (or even meaningful)
            # — store sentinel angular values that _render_overlays
            # rewrites to (45°, ∞) when it builds the preview row.
            self._fitted_preview_geom = (
                float(center_r), float(width_r), 45.0, 0.0,
            )
            self._fitted_preview_is_ring = True
        elif (
            center_a is None or width_a is None
            or not (np.isfinite(center_a) and np.isfinite(width_a) and width_a > 0)
        ):
            self._fitted_preview_geom = None
            self._fitted_preview_is_ring = False
        else:
            self._fitted_preview_geom = (
                float(center_r), float(width_r),
                float(center_a), float(width_a),
            )
            self._fitted_preview_is_ring = False
        self._render_overlays(self.current_frame)

    def set_fitted_preview_extras(
        self,
        geoms: list[tuple[float, float, float, float]],
    ) -> None:
        """Paint cyan dashed previews for every Ctrl+click extra.

        ``geoms`` is a list of ``(radius, radius_width, angle,
        angle_width)`` tuples — one per extra peak whose pygidfit
        result the host has already computed (or fallen back from).
        Pass ``[]`` to clear. Each tuple paints the same way the
        primary preview does (no ring support for extras: extras are
        detected peaks, which never use the ring sentinel).

        Stored alongside ``_fitted_preview_geom`` (which still owns
        the primary peak's preview) so the renderer can paint N
        boxes from a single ``_PeakShapeItem``.
        """
        cleaned: list[tuple[float, float, float, float]] = []
        for g in geoms:
            try:
                cr, wr, ca, wa = (float(g[0]), float(g[1]), float(g[2]), float(g[3]))
            except (TypeError, ValueError, IndexError):
                continue
            if not (
                np.isfinite(cr) and np.isfinite(wr) and wr > 0
                and np.isfinite(ca) and np.isfinite(wa) and wa > 0
            ):
                continue
            cleaned.append((cr, wr, ca, wa))
        if cleaned == self._fitted_preview_extras_geoms:
            return
        self._fitted_preview_extras_geoms = cleaned
        self._render_overlays(self.current_frame)

    def set_busy(self, busy: bool) -> None:
        """Disable interactive editing while a pipeline run is in flight."""
        self._busy = busy
        self._sync_roi()

    @property
    def current_frame(self) -> int:
        return self._frame_index

    @property
    def n_frames(self) -> int:
        """Number of frames in the active stack (0 if no stack loaded)."""
        if self._mode == MODE_RAW and self._raw_image_stack is not None:
            return int(self._raw_image_stack.shape[0])
        return 0 if self._stack is None else int(self._stack.n_frames)

    def set_frame(self, frame: int) -> None:
        """Seek to ``frame``.

        The single source of truth for frame changes. Reads the new
        frame from the active source (FrameSource for NeXus, the
        in-memory raw stack for raw mode), pushes the 2D image to
        pyqtgraph via ``setImage`` (with the cached pos / scale /
        levels so the LUT and viewbox stay stable), updates the
        per-frame overlays, and emits ``frameChanged`` plus
        ``matchedStructuresChanged`` exactly once.

        Also prunes any selected peaks that belong to a different
        frame so the white highlight overlay doesn't paint stale
        rectangles at coordinates that don't correspond to a real
        peak on the new frame (typical after a copy/paste workflow
        where the source-frame selection lingers).

        No-op when ``frame`` is already current or out of range.
        """
        n = self.n_frames
        if n == 0:
            return
        idx = max(0, min(int(frame), n - 1))
        if idx == self._frame_index:
            return
        self._frame_index = idx
        self._prune_off_frame_selections(idx)
        self._render_frame(idx, auto_range=False)
        self._render_overlays(idx)
        self.frameChanged.emit(idx)
        # The Display dock rebuilds its matched-structure rows from this
        # signal — different frames can have different solutions.
        self.matchedStructuresChanged.emit(idx, self.matched_structures(idx))

    def _prune_off_frame_selections(self, frame: int) -> None:
        """Drop any selected peaks whose ``.frame`` is not ``frame``.

        Called when the user navigates between frames so the white
        highlight overlay only paints peaks that actually exist on
        the visible frame. Manual peaks are exempt because their
        ``.frame`` is the frame they were drawn on and the manual-
        peak rendering already gates on ``self.current_frame``;
        pruning them here would erase their selection across
        navigation which is the wrong default. Detected / fitted /
        matched selections are frame-bound and get pruned.

        Emits ``selectionChanged`` and ``selectionsChanged`` once
        if anything actually changed.
        """
        def _on_frame(s: SelectedPeak) -> bool:
            # Manual peaks survive a frame switch — they have a
            # frame field but their renderer is frame-aware and the
            # user's draw intent is "this one specific peak".
            if s.kind == "manual":
                return True
            return int(s.frame) == int(frame)

        prev_primary = self._selected
        prev_extras = list(self._selected_extras)
        # Filter extras first.
        self._selected_extras = [s for s in self._selected_extras if _on_frame(s)]
        # If the primary doesn't belong here, promote the first
        # surviving extra (or clear).
        if self._selected is not None and not _on_frame(self._selected):
            if self._selected_extras:
                self._selected = self._selected_extras.pop(0)
            else:
                self._selected = None
        if (
            self._selected is prev_primary
            and self._selected_extras == prev_extras
        ):
            return
        self._sync_roi()
        self.selectionChanged.emit(self._selected)
        self.selectionsChanged.emit(self.selected_peaks())

    @property
    def selected_peak(self) -> SelectedPeak | None:
        return self._selected

    def selected_peaks(self) -> list[SelectedPeak]:
        """Full multi-selection list (primary + extras), or empty when
        nothing is selected.

        Backs the Ctrl+C copy handler and the batch-fit button enable
        check on the host. Order is stable: primary first, extras in
        the order they were Ctrl+click-added.
        """
        if self._selected is None:
            return []
        return [self._selected, *self._selected_extras]

    @property
    def is_dragging(self) -> bool:
        """True while an image-side ROI handle drag is in progress.

        Exposed so the host can decide whether per-tick
        ``peakGeometryChanged`` signals come from a live drag (and
        deserve debounced expensive recompute) or a settled
        programmatic update.
        """
        return self._roi_drag_before is not None

    # -- Action helpers (used by both public API and undo/redo) --

    def _push_undo(self, action: _Action) -> None:
        self._undo_stack.append(action)
        self._redo_stack.clear()

    def _undoable_add_manual(self, frame: int, peak: ManualPeak) -> None:
        """Insert a manual peak without touching the undo stack.

        Single-box policy invariant: whenever a manual peak is on
        screen it is also the active selection. This applies to every
        add path — the user-draw flow already auto-selected before;
        now undo of a remove (which restores the manual box) and redo
        of an add do too. Skipped when the peak is added on a non-
        current frame because the user can't see / interact with it.
        """
        bucket = self._manual_peaks.setdefault(frame, [])
        if peak not in bucket:
            bucket.append(peak)
        if frame == self.current_frame:
            self._render_overlays(frame)
        self.manualPeakAdded.emit(frame, peak)
        if frame == self.current_frame:
            self._set_selected(SelectedPeak.from_manual(peak, frame))

    def _undoable_remove_manual(self, frame: int, peak: ManualPeak) -> None:
        """Remove a manual peak without touching the undo stack."""
        bucket = self._manual_peaks.get(frame, [])
        if peak in bucket:
            bucket.remove(peak)
        was_selected = (
            self._selected is not None
            and self._selected.kind == "manual"
            and self._selected.manual_ref is peak
        )
        if was_selected:
            self._selected = None
            self._sync_roi()
        if frame == self.current_frame:
            self._render_overlays(frame)
        if was_selected:
            self.selectionChanged.emit(None)
        self.manualPeakRemoved.emit(frame, peak)

    def _apply_manual_geom(
        self, frame: int, peak: ManualPeak,
        polar: tuple[float, float, float, float],
    ) -> None:
        r, a, dr, da = polar
        peak.radius = r
        peak.angle = a
        peak.radius_width = dr
        peak.angle_width = da
        # If this peak is the active selection, mirror it on the SelectedPeak
        # snapshot and refresh the ROI without retriggering its signals.
        if (
            self._selected is not None
            and self._selected.kind == "manual"
            and self._selected.manual_ref is peak
        ):
            self._selected.radius = r
            self._selected.angle = a
            self._selected.radius_width = dr
            self._selected.angle_width = da
            self._sync_roi_geometry()
        if frame == self.current_frame:
            self._render_overlays(frame)
        if (
            self._selected is not None
            and self._selected.kind == "manual"
            and self._selected.manual_ref is peak
        ):
            self.peakGeometryChanged.emit(self._selected)

    def _apply_file_geom(
        self, frame: int, kind: str, peak_id: int,
        polar: tuple[float, float, float, float],
    ) -> None:
        r, a, dr, da = polar
        # Update the in-memory PeakTable so overlays paint the new box right
        # away; the disk write is fired separately via peakRowWriteRequested.
        peaks_for_frame = self._frame_peaks.get(frame) or {}
        table = peaks_for_frame.get(kind)
        if table is not None and len(table) > 0:
            matches = np.where(table.ids == peak_id)[0]
            if matches.size > 0:
                idx = int(matches[0])
                table.radius[idx] = r
                table.angle[idx] = a
                table.radius_width[idx] = dr
                table.angle_width[idx] = da
                table.q_xy[idx], table.q_z[idx] = polar_to_qxyz(r, a)
        # If the user edited a fitted peak, every matched solution that
        # references it must re-slice from the updated fitted table.
        if kind == "fitted":
            self._refresh_matched_for(frame)
        # Reflect the change on the SelectedPeak if it's the active selection.
        if (
            self._selected is not None
            and self._selected.kind in (kind, "matched")
            and self._selected.frame == frame
            and self._selected.peak_id == peak_id
        ):
            self._selected.radius = r
            self._selected.angle = a
            self._selected.radius_width = dr
            self._selected.angle_width = da
            self._sync_roi_geometry()
        if frame == self.current_frame:
            self._render_overlays(frame)
        # Fire the file-write so undo/redo also persist.
        self.peakRowWriteRequested.emit(
            frame, kind, int(peak_id),
            {"radius": r, "angle": a, "radius_width": dr, "angle_width": da},
        )
        if (
            self._selected is not None
            and self._selected.peak_id == peak_id
            and self._selected.frame == frame
        ):
            self.peakGeometryChanged.emit(self._selected)

    def _refresh_matched_for(self, frame: int) -> None:
        """Re-slice the frame's fitted PeakTable into each MatchedStructure
        using its cached ``peak_list`` indices. Cheap (numpy fancy index).
        """
        peaks_for_frame = self._frame_peaks.get(frame) or {}
        fitted = peaks_for_frame.get("fitted")
        structures = self._matched_per_frame.get(frame, [])
        if fitted is None or not structures:
            return
        n_fit = len(fitted)
        for s in structures:
            idx = s.peak_list
            idx = idx[(idx >= 0) & (idx < n_fit)]
            s.peaks = PeakTable(
                q_xy=fitted.q_xy[idx],
                q_z=fitted.q_z[idx],
                angle=fitted.angle[idx],
                radius=fitted.radius[idx],
                angle_width=fitted.angle_width[idx],
                radius_width=fitted.radius_width[idx],
                is_ring=fitted.is_ring[idx],
                ids=fitted.ids[idx],
                score=fitted.score[idx],
                amplitude=fitted.amplitude[idx],
            )

    def polar_data(
        self,
    ) -> "tuple[_LazyPolarStack, np.ndarray, np.ndarray] | None":
        """Return ``(polar_stack, radius, angle)`` for the active entry.

        ``polar_stack`` is a ``_LazyPolarStack`` — supports
        ``polar_stack[i]`` (one polar frame) and
        ``polar_stack[frame, r, a]`` (single-pixel cursor lookup).
        Per-frame resampling happens on demand inside the
        ``FrameSource``; nothing is precomputed eagerly.

        Returns None if no stack is currently loaded or the
        FrameSource is currently released (e.g. mid-pipeline silx
        detach). Callers should re-call after the reattach.
        """
        if self._stack is None or self._frame_source is None:
            return None
        if not self._frame_source.is_open:
            return None
        if self._polar_cache is None:
            radius, angle = self._frame_source.polar_axes()
            self._polar_cache = (
                _LazyPolarStack(self._frame_source), radius, angle,
            )
        return self._polar_cache

    # -- Labelling event handlers (polar mode only for now) --

    def _on_draw_started(self, origin: QPointF) -> None:
        if self._mode != MODE_POLAR:
            return
        rect = QRectF(origin, origin)
        self._preview_item.setRect(rect.normalized())
        self._preview_item.setVisible(True)

    def _on_draw_updated(self, origin: QPointF, current: QPointF) -> None:
        if self._mode != MODE_POLAR or not self._preview_item.isVisible():
            return
        self._preview_item.setRect(QRectF(origin, current).normalized())

    def _on_draw_finished(self, origin: QPointF, end: QPointF) -> None:
        self._preview_item.setVisible(False)
        if self._mode != MODE_POLAR or self._busy:
            return
        # In polar mode: x = radius (Å⁻¹), y = angle (deg).
        rect = QRectF(origin, end).normalized()
        if rect.width() <= 0.0 or rect.height() <= 0.0:
            return
        peak = ManualPeak(
            radius=float(rect.center().x()),
            angle=float(rect.center().y()),
            radius_width=float(rect.width()),
            angle_width=float(rect.height()),
            is_ring=False,
            temp_id=self._next_manual_id,
        )
        self._next_manual_id -= 1
        # Single-manual-box policy: any pre-existing manual peak on this
        # frame is replaced atomically. Modelled as one undo entry so
        # Ctrl+Z rewinds the whole swap rather than two staged steps.
        frame = self.current_frame
        existing = self._manual_peaks.get(frame, [])
        old_peak = existing[0] if existing else None
        if old_peak is not None:
            self._undoable_remove_manual(frame, old_peak)
        # _undoable_add_manual auto-selects on the current frame, so no
        # explicit selection call is needed here.
        self._undoable_add_manual(frame, peak)
        self._push_undo(
            ManualReplaceAction(frame=frame, old_peak=old_peak, new_peak=peak)
        )

    def _on_select_at(self, pos: QPointF, mods=Qt.KeyboardModifier.NoModifier) -> None:
        # Raw mode has no q-space overlays to hit-test. Polar and
        # Cartesian both run the same hit-test pipeline; the click
        # coordinates arrive in whatever space the viewbox is currently
        # showing (polar = (r, a), cartesian = (q_xy, q_z)), and the
        # mode-specific helpers below normalise to polar before
        # checking containment.
        #
        # ``mods``: keyboard modifiers held at press time.
        #
        # * Bare click (no modifiers): walks the full priority list
        #   ``manual > fitted > detected > matched`` and routes the
        #   first hit through ``_set_selected``.
        # * Ctrl+click: multi-select. Hit-tests the detected and fitted
        #   overlays (only — manual / matched stay single-select) and
        #   routes through ``_toggle_selected``. The search prefers the
        #   *current* multi-selection's kind so the user keeps extending
        #   one kind: e.g. a detected multi-select still reaches detected
        #   peaks that have a fitted box drawn on top (post-Run Fitting),
        #   and a fitted multi-select reaches fitted. With no multi-kind
        #   primary yet, fitted wins over detected (same priority as a
        #   bare click). No hit under the cursor → no-op.
        # * Ctrl+Alt never reaches here (press branch consumed it
        #   as a draw gesture).
        if self._mode not in (MODE_POLAR, MODE_CARTESIAN) or self._busy:
            return
        x, y = float(pos.x()), float(pos.y())
        # Simulated-reflection selection preempts peak selection while
        # the Expected-pattern overlay is visible: a click landing on a
        # simulated marker/box centre toggles it (bare and Ctrl clicks
        # alike) and stops here; a miss falls through to the normal
        # manual > fitted > detected > matched flow untouched. A sim
        # marker can occlude a peak under it — unticking "Expected
        # pattern" restores plain selection.
        if (
            self._sim_pattern is not None
            and self._sim_visible
            and mods in (
                Qt.KeyboardModifier.NoModifier,
                Qt.KeyboardModifier.ControlModifier,
            )
            and self._sim_toggle_at(x, y)
        ):
            return
        cart = self._mode == MODE_CARTESIAN
        frame = self.current_frame
        peaks_for_frame = self._frame_peaks.get(frame) or {}
        ctrl_only = (
            bool(mods & Qt.KeyboardModifier.ControlModifier)
            and not bool(mods & Qt.KeyboardModifier.AltModifier)
        )

        def hit_manual(peak: ManualPeak) -> bool:
            return _cart_box_contains(peak, x, y) if cart else _polar_box_contains(peak, x, y)

        def hit_table(tbl: PeakTable, i: int) -> bool:
            return (
                _cart_table_row_contains(tbl, i, x, y) if cart
                else _polar_table_row_contains(tbl, i, x, y)
            )

        # Ctrl+click multi-selects detected OR fitted peaks. Search the
        # current multi-selection's kind first (so the user keeps
        # building one kind even where detected/fitted overlap), then
        # the other; with no multi-kind primary, fitted wins over
        # detected (bare-click priority). Manual / matched are never
        # multi-selected.
        if ctrl_only:
            primary = self._selected
            if primary is not None and primary.kind in ("detected", "fitted"):
                order = (
                    ["detected", "fitted"] if primary.kind == "detected"
                    else ["fitted", "detected"]
                )
            else:
                order = ["fitted", "detected"]
            for kind in order:
                if not self._visibility.get(kind, True):
                    continue
                table = peaks_for_frame.get(kind)
                if table is None or len(table) == 0:
                    continue
                for i in reversed(range(len(table))):
                    if kind == "fitted" and self._fitted_row_hidden(
                        frame, int(table.ids[i])
                    ):
                        continue
                    if hit_table(table, i):
                        self._toggle_selected(SelectedPeak(
                            kind=kind,
                            frame=frame,
                            peak_id=int(table.ids[i]),
                            radius=float(table.radius[i]),
                            angle=float(table.angle[i]),
                            radius_width=float(table.radius_width[i]),
                            angle_width=float(table.angle_width[i]),
                            is_ring=bool(table.is_ring[i]),
                            score=float(table.score[i]),
                            amplitude=float(table.amplitude[i]),
                        ))
                        return
            # Ctrl+click on empty space (or only manual/matched under
            # the cursor) = no-op; the existing selection stays put.
            return

        # Priority order: manual > fitted > detected > matched. Matched is
        # last because it's a subset of fitted; the rare case where the user
        # wants the matched-context selection is still reachable by hiding
        # the fitted overlay.
        # 1) manual
        if self._visibility.get("manual", True):
            for peak in reversed(self._manual_peaks.get(frame, [])):
                if hit_manual(peak):
                    self._set_selected(SelectedPeak.from_manual(peak, frame))
                    return

        # 2) fitted, 3) detected — same hit-test against the PeakTable rows.
        for kind in ("fitted", "detected"):
            if not self._visibility.get(kind, True):
                continue
            table = peaks_for_frame.get(kind)
            if table is None or len(table) == 0:
                continue
            for i in reversed(range(len(table))):
                if kind == "fitted" and self._fitted_row_hidden(
                    frame, int(table.ids[i])
                ):
                    continue
                if hit_table(table, i):
                    self._set_selected(SelectedPeak(
                        kind=kind,
                        frame=frame,
                        peak_id=int(table.ids[i]),
                        radius=float(table.radius[i]),
                        angle=float(table.angle[i]),
                        radius_width=float(table.radius_width[i]),
                        angle_width=float(table.angle_width[i]),
                        is_ring=bool(table.is_ring[i]),
                        score=float(table.score[i]),
                        amplitude=float(table.amplitude[i]),
                    ))
                    return

        # 4) matched — only when the master toggle is on. The hit's peak_id
        # is the underlying fitted id (which is what delete_peak consumes).
        if self._matched_master_visible:
            structures = self._matched_per_frame.get(frame, [])
            for s in reversed(structures):
                if not self._is_matched_item_visible(s.unique_id):
                    continue
                tbl = s.peaks
                color = self._pen_for_key(s.color_key)["color"]
                # Under "show only tracked peaks", only this structure's
                # tracked peaks are drawn — so only they are clickable,
                # and the structure-level highlight covers just them.
                visible_ids = [
                    int(x) for x in tbl.ids
                    if not self._fitted_row_hidden(frame, int(x))
                ]
                for i in reversed(range(len(tbl))):
                    if self._fitted_row_hidden(frame, int(tbl.ids[i])):
                        continue
                    if hit_table(tbl, i):
                        self._set_selected(SelectedPeak(
                            kind="matched",
                            frame=frame,
                            peak_id=int(tbl.ids[i]),
                            radius=float(tbl.radius[i]),
                            angle=float(tbl.angle[i]),
                            radius_width=float(tbl.radius_width[i]),
                            angle_width=float(tbl.angle_width[i]),
                            is_ring=bool(tbl.is_ring[i]),
                            structure_uid=s.unique_id,
                            structure_label=s.label,
                            structure_color=color,
                            score=float(tbl.score[i]),
                            amplitude=float(tbl.amplitude[i]),
                            # Clicking any peak of the structure
                            # promotes the whole (visible) structure
                            # into the selection — overlay highlights
                            # every visible peak in it, table syncs the
                            # structure row.
                            multi_peak_ids=visible_ids,
                        ))
                        return

        # Click on empty space → deselect (the Ctrl branch already
        # early-returned above, so this only runs on bare clicks).
        if self._selected is not None:
            self._set_selected(None)

    def _set_selected(
        self, sel: SelectedPeak | None, *, preserve_manual: bool = False,
    ) -> None:
        """Update the selection and sync the ROI + emit selectionChanged once.

        Side effect: if we're transitioning **away** from a
        manual-peak selection to anything else (a different peak, or
        nothing), drop the previous manual peak via
        ``remove_manual_peak``. This makes manual boxes truly
        transient — clicking off them abandons the draw. A
        ``ManualRemoveAction`` is pushed so ``Ctrl+Z`` brings the box
        back. The new-box draw path already removes the old peak via
        ``ManualReplaceAction`` *before* this method runs, so no
        double-remove happens there.

        Programmatic deselects from ``clear_selection`` bypass this
        method (they set ``self._selected = None`` directly), which
        preserves the manual peak across pipeline resets and other
        non-user-driven selection clears.

        ``preserve_manual``: when True, the manual-removal side
        effect is suppressed. Used by host flows that intentionally
        keep the manual peak around across a programmatic
        switch — currently ``MainWindow._on_add_to_fitted`` (so the
        manual source survives the auto-switch to the new fitted
        peak; without this, the silent ``ManualRemoveAction`` push
        would force the user to press Ctrl+Z twice to fully revert
        an Add-to-fitted commit) and the matching redo closure in
        ``_push_fitted_add_undo``.
        """
        if sel is None and self._selected is None and not self._selected_extras:
            return
        primary_unchanged = (
            sel is not None
            and self._selected is not None
            and sel.kind == self._selected.kind
            and sel.frame == self._selected.frame
            and sel.peak_id == self._selected.peak_id
            and sel.structure_uid == self._selected.structure_uid
        )
        # When the primary doesn't change but extras exist, this call
        # still has work to do (collapse multi-selection back to one).
        if primary_unchanged and not self._selected_extras:
            return
        prev = self._selected
        transitioning_away_from_manual = (
            not preserve_manual
            and prev is not None
            and prev.kind == "manual"
            and prev.manual_ref is not None
            and (sel is None or sel.manual_ref is not prev.manual_ref)
        )
        if transitioning_away_from_manual:
            bucket = self._manual_peaks.get(prev.frame, [])
            if prev.manual_ref in bucket:
                # ``remove_manual_peak`` clears ``self._selected`` to
                # None internally and emits ``selectionChanged(None)``
                # since the removed peak was the active selection. We
                # then re-apply the actual new ``sel`` below, which
                # emits ``selectionChanged(sel)`` a second time. The
                # transient None emit is harmless — listeners that
                # store state will end up with the right value.
                self.remove_manual_peak(prev.frame, prev.manual_ref)
        self._selected = sel
        # _set_selected is the "replace the selection" path; extras
        # always clear here. ``_toggle_selected`` is the alternative
        # entry that appends instead.
        self._selected_extras = []
        self._sync_roi()
        self._render_overlays(self.current_frame)
        self.selectionChanged.emit(sel)
        self.selectionsChanged.emit(self.selected_peaks())

    def _toggle_selected(self, sel: SelectedPeak) -> None:
        """Add / remove ``sel`` to/from the multi-selection (Ctrl+click).

        Multi-select covers ``detected`` and ``fitted`` peaks, but only
        within a single kind: a multi-selection is all-detected or
        all-fitted (the bulk ops downstream — delete cascade, batch
        fit — are kind-specific). If ``sel`` is manual / matched, or it
        is a different kind than the current primary, fall back to
        ``_set_selected`` (single-select replace), which starts a fresh
        selection of the new kind.

        Toggle semantics:
          * No primary yet → ``sel`` becomes the primary.
          * ``sel`` matches the primary → demote: promote
            ``_selected_extras[0]`` to primary if any, else clear.
          * ``sel`` matches an extra → drop from extras.
          * Otherwise → append to extras.
        """
        if sel.kind not in ("detected", "fitted"):
            self._set_selected(sel)
            return
        if self._selected is not None and self._selected.kind != sel.kind:
            self._set_selected(sel)
            return
        if self._selected is None:
            self._selected = sel
            self._sync_roi()
            self._render_overlays(self.current_frame)
            self.selectionChanged.emit(sel)
            self.selectionsChanged.emit(self.selected_peaks())
            return
        # Match by (kind, frame, peak_id) — geometry can shift via ROI
        # drag without changing identity.
        def _same(a: SelectedPeak, b: SelectedPeak) -> bool:
            return (
                a.kind == b.kind
                and a.frame == b.frame
                and a.peak_id == b.peak_id
            )
        if _same(sel, self._selected):
            if self._selected_extras:
                self._selected = self._selected_extras.pop(0)
            else:
                self._selected = None
        elif any(_same(sel, e) for e in self._selected_extras):
            self._selected_extras = [
                e for e in self._selected_extras if not _same(sel, e)
            ]
        else:
            self._selected_extras.append(sel)
        self._sync_roi()
        self._render_overlays(self.current_frame)
        self.selectionChanged.emit(self._selected)
        self.selectionsChanged.emit(self.selected_peaks())

    def keyPressEvent(self, ev) -> None:  # type: ignore[override]
        if (
            ev.key() == Qt.Key.Key_Delete
            and self._selected is not None
            and not self._busy
        ):
            sels = self.selected_peaks()
            kinds = {s.kind for s in sels}
            if (
                len(sels) >= 2
                and len(kinds) == 1
                and next(iter(kinds)) in ("detected", "fitted")
            ):
                # All-detected OR all-fitted multi-selection → bulk
                # delete in one grouped, undoable action on the host.
                self.deletePeaksRequested.emit(list(sels))
                ev.accept()
                return
            sel = self._selected
            if sel.kind == "manual" and sel.manual_ref is not None:
                self.remove_manual_peak(self.current_frame, sel.manual_ref)
            else:
                # File-resident peaks go through MainWindow → mlgidbase
                # delete_peak (cascading + with confirmation).
                self.deletePeakRequested.emit(sel)
            ev.accept()
            return
        # Esc on a selected manual peak removes it. Manual boxes are an
        # in-memory scratchpad — no file write, no confirmation.
        # File-resident selections fall through (Esc is meaningless for
        # them; Delete is the documented binding).
        if (
            ev.key() == Qt.Key.Key_Escape
            and self._selected is not None
            and self._selected.kind == "manual"
            and self._selected.manual_ref is not None
            and not self._busy
        ):
            self.remove_manual_peak(self.current_frame, self._selected.manual_ref)
            ev.accept()
            return
        # Ctrl+A: select every peak of the *current kind* on the current
        # frame. If a fitted peak is the primary, select all fitted;
        # otherwise (detected primary, or nothing selected) select all
        # detected. First row becomes primary; the rest become extras.
        if (
            ev.key() == Qt.Key.Key_A
            and ev.modifiers() == Qt.KeyboardModifier.ControlModifier
            and not self._busy
        ):
            kind = (
                self._selected.kind
                if self._selected is not None
                and self._selected.kind in ("detected", "fitted")
                else "detected"
            )
            self._select_all_of_kind_on_frame(kind)
            ev.accept()
            return
        super().keyPressEvent(ev)

    def _select_all_of_kind_on_frame(self, kind: str) -> None:
        """Replace the selection with every ``kind`` peak on the current frame.

        ``kind`` is ``"detected"`` or ``"fitted"``. No-op if the frame
        has no such table or zero rows. Emits both ``selectionChanged``
        (primary) and ``selectionsChanged`` (full list) once.
        """
        frame = self.current_frame
        peaks_for_frame = self._frame_peaks.get(frame) or {}
        table = peaks_for_frame.get(kind)
        if table is None or len(table) == 0:
            return

        def _sel(i: int) -> SelectedPeak:
            return SelectedPeak(
                kind=kind,
                frame=frame,
                peak_id=int(table.ids[i]),
                radius=float(table.radius[i]),
                angle=float(table.angle[i]),
                radius_width=float(table.radius_width[i]),
                angle_width=float(table.angle_width[i]),
                is_ring=bool(table.is_ring[i]),
                score=float(table.score[i]),
                amplitude=float(table.amplitude[i]),
            )

        # Skip fitted rows hidden by the "show only tracked peaks"
        # filter — Ctrl+A must not grab invisible untracked peaks.
        indices = [
            i for i in range(len(table))
            if not (kind == "fitted"
                    and self._fitted_row_hidden(frame, int(table.ids[i])))
        ]
        if not indices:
            return
        sels = [_sel(i) for i in indices]
        # Bypass _set_selected (which would clear extras) and write
        # the multi-selection in one shot.
        self._selected = sels[0]
        self._selected_extras = sels[1:]
        self._sync_roi()
        self._render_overlays(self.current_frame)
        self.selectionChanged.emit(self._selected)
        self.selectionsChanged.emit(self.selected_peaks())

    # -- Resizable ROI on the selected peak --

    def _sync_roi(self) -> None:
        """Create / update / destroy the resize ROI to match the selection.

        Polar mode only, and only for editable kinds (manual / detected).
        Fitted and matched selections show the box but no ROI — fitted
        boxes encode the ``2σ`` convention so dragging their bounds
        would misrepresent the underlying Gaussian; matched is a
        derived view of fitted_peaks. Both edit through Add-to-fitted
        / delete instead.

        Handles ring peaks (``is_ring`` true or non-finite ``angle_width``)
        and peaks whose box edges fall outside the visible polar range:
        the ROI is clamped to the data axes so the handles are reachable,
        and ring peaks get only the radial (left/right) handles since their
        angular extent is the whole quadrant by definition.
        """
        self._teardown_roi()

        if (
            self._selected is None
            or self._mode != MODE_POLAR
            or self._busy
            or self._selected.kind in ("fitted", "matched")
        ):
            return

        # Need the polar axes to clamp against; bail if not yet computed.
        if self._polar_cache is None:
            return
        _, radius_axis, angle_axis = self._polar_cache
        if radius_axis.size == 0 or angle_axis.size == 0:
            return
        r_min, r_max = float(radius_axis[0]), float(radius_axis[-1])
        a_min, a_max = float(angle_axis[0]), float(angle_axis[-1])

        sel = self._selected
        is_ring_box = sel.is_ring or not np.isfinite(sel.angle_width)

        if is_ring_box:
            a_lo, a_hi = a_min, a_max
        else:
            a_lo = max(sel.angle - sel.angle_width / 2.0, a_min)
            a_hi = min(sel.angle + sel.angle_width / 2.0, a_max)
            if a_hi <= a_lo:
                return  # peak entirely outside visible angular range
        r_lo = max(sel.radius - sel.radius_width / 2.0, r_min)
        r_hi = min(sel.radius + sel.radius_width / 2.0, r_max)
        if r_hi <= r_lo:
            return

        pos = (r_lo, a_lo)
        size = (r_hi - r_lo, a_hi - a_lo)

        # ROI pen uses the selection white so the resize handles
        # match the selection-highlight outline that wraps every
        # selected peak. Source-kind colouring is preserved by the
        # static overlay items (red dashed for detected, yellow
        # solid for manual) — those stay visible behind the white
        # outline + ROI handles.
        # Asked for per draw: the highlight colour follows the theme
        # (white on the dark plot ground, near-black on the light one).
        style = selection_style()
        pen = pg.mkPen(
            QColor(style["color"]),
            width=style["width"],
        )
        pen.setStyle(style["style"])
        pen.setCosmetic(True)
        # Hover keeps the same pen so there's no flicker on mouseover.
        hover_pen = pg.mkPen(
            QColor(style["color"]),
            width=style["width"],
        )
        hover_pen.setCosmetic(True)

        roi = pg.ROI(pos=pos, size=size, pen=pen, hoverPen=hover_pen, movable=True)
        # Edge-only handles (no corners): each handle drags one edge while the
        # opposite edge stays anchored. Rings have only radial handles — the
        # angular bounds are the whole quadrant by construction.
        roi.addScaleHandle([1.0, 0.5], [0.0, 0.5])  # right
        roi.addScaleHandle([0.0, 0.5], [1.0, 0.5])  # left
        if not is_ring_box:
            roi.addScaleHandle([0.5, 1.0], [0.5, 0.0])  # top
            roi.addScaleHandle([0.5, 0.0], [0.5, 1.0])  # bottom
        roi.setZValue(60)
        # Track which dimensions are user-editable for _on_roi_changed.
        roi._mlgid_ring_box = is_ring_box  # type: ignore[attr-defined]
        roi.sigRegionChangeStarted.connect(self._on_roi_drag_started)
        roi.sigRegionChanged.connect(self._on_roi_changed)
        roi.sigRegionChangeFinished.connect(self._on_roi_drag_finished)

        self._plot.getViewBox().addItem(roi, ignoreBounds=True)
        self._roi_item = roi

    def _teardown_roi(self) -> None:
        if self._roi_item is None:
            return
        roi = self._roi_item
        for sig_name in (
            "sigRegionChangeStarted", "sigRegionChanged", "sigRegionChangeFinished",
        ):
            try:
                getattr(roi, sig_name).disconnect()
            except (RuntimeError, TypeError):
                pass
        self._plot.getViewBox().removeItem(roi)
        self._roi_item = None

    def _sync_roi_geometry(self) -> None:
        """Adjust the existing ROI to match the SelectedPeak without rebuilding.

        Used by undo/redo and external geometry updates — blocks signals so we
        don't recursively re-enter ``_on_roi_changed``.
        """
        if self._roi_item is None or self._selected is None:
            return
        roi = self._roi_item
        roi.blockSignals(True)
        try:
            roi.setPos(
                [
                    self._selected.radius - self._selected.radius_width / 2.0,
                    self._selected.angle - self._selected.angle_width / 2.0,
                ],
                update=False,
            )
            roi.setSize([self._selected.radius_width, self._selected.angle_width])
        finally:
            roi.blockSignals(False)

    def _on_roi_drag_started(self) -> None:
        if self._selected is None:
            return
        self._roi_drag_before = self._selected.polar_tuple()

    def _on_roi_changed(self) -> None:
        if self._selected is None or self._roi_item is None:
            return
        roi = self._roi_item
        pos = roi.pos()
        size = roi.size()
        w = abs(float(size[0]))
        h = abs(float(size[1]))
        # ROI sizes can go negative if dragged past the opposite edge — take abs
        # then derive the new center from the (possibly flipped) bottom-left.
        x0 = float(pos[0]) + min(float(size[0]), 0.0)
        y0 = float(pos[1]) + min(float(size[1]), 0.0)
        new_r = x0 + w / 2.0
        new_a = y0 + h / 2.0

        sel = self._selected
        # Ring peaks have only radial handles — the angular extent is fixed
        # at the whole visible quadrant (and the underlying angle_width is
        # often inf). Don't propagate the ROI's angular pos/size into the
        # peak's geometry, or we'd corrupt the ring on every drag.
        is_ring_box = bool(getattr(roi, "_mlgid_ring_box", False))

        sel.radius = new_r
        sel.radius_width = w
        if not is_ring_box:
            sel.angle = new_a
            sel.angle_width = h

        if sel.kind == "manual" and sel.manual_ref is not None:
            sel.manual_ref.radius = new_r
            sel.manual_ref.radius_width = w
            if not is_ring_box:
                sel.manual_ref.angle = new_a
                sel.manual_ref.angle_width = h
        else:
            # Mutate the in-memory PeakTable so the colored detected/fitted
            # outline tracks the drag live. Disk + matched-resync happen on
            # drag-end via _on_roi_drag_finished. For ring boxes we don't
            # touch angle / angle_width since those handles aren't shown.
            peaks_for_frame = self._frame_peaks.get(sel.frame) or {}
            table = peaks_for_frame.get(sel.kind)
            if table is not None and len(table) > 0:
                matches = np.where(table.ids == sel.peak_id)[0]
                if matches.size > 0:
                    idx = int(matches[0])
                    table.radius[idx] = new_r
                    table.radius_width[idx] = w
                    if not is_ring_box:
                        table.angle[idx] = new_a
                        table.angle_width[idx] = h
                    cur_a = float(table.angle[idx])
                    table.q_xy[idx], table.q_z[idx] = polar_to_qxyz(new_r, cur_a)

        self._render_overlays(self.current_frame)
        self.peakGeometryChanged.emit(sel)

    def _on_roi_drag_finished(self) -> None:
        if self._selected is None or self._roi_drag_before is None:
            return
        before = self._roi_drag_before
        after = self._selected.polar_tuple()
        self._roi_drag_before = None
        if before == after:
            return  # idle release — nothing to record

        sel = self._selected
        if sel.kind == "manual" and sel.manual_ref is not None:
            self._push_undo(ManualGeomAction(
                frame=sel.frame, peak=sel.manual_ref,
                before=before, after=after,
            ))
        elif sel.kind in ("detected", "fitted"):
            self._push_undo(FileGeomAction(
                frame=sel.frame, kind=sel.kind, peak_id=sel.peak_id,
                before=before, after=after,
            ))
            # Re-derive matched overlays if a fitted edit changed an
            # underlying row used by any matched solution.
            if sel.kind == "fitted":
                self._refresh_matched_for(sel.frame)
                self._render_overlays(self.current_frame)
            # Persist to the file via MainWindow (see peakRowWriteRequested).
            self.peakRowWriteRequested.emit(
                sel.frame, sel.kind, int(sel.peak_id),
                {"radius": after[0], "angle": after[1],
                 "radius_width": after[2], "angle_width": after[3]},
            )
        # Drag-end notification (all kinds) — subscribers that want the
        # final geometry without per-tick refits hook here.
        self.peakGeometryDragFinished.emit(sel)

    def update_peak_geometry_external(self, peak: ManualPeak) -> None:
        """Sync the ROI to a peak whose geometry was changed elsewhere
        (e.g. by dragging a profile region). Suppresses ROI signals so this
        doesn't loop back into ``_on_roi_changed``.
        """
        if (
            self._selected is None
            or self._selected.kind != "manual"
            or self._selected.manual_ref is not peak
            or self._roi_item is None
        ):
            return
        # Mirror the new geometry onto the SelectedPeak snapshot.
        self._selected.radius = peak.radius
        self._selected.angle = peak.angle
        self._selected.radius_width = peak.radius_width
        self._selected.angle_width = peak.angle_width
        self._sync_roi_geometry()
        self._render_overlays(self.current_frame)
        self.peakGeometryChanged.emit(self._selected)

    def update_detected_geometry_external(self, sel: "SelectedPeak") -> None:
        """Sync the detected-peak overlay to a SelectedPeak whose geometry
        was changed elsewhere — currently the profile region drag (see
        ``profile_viewer.detectedPeakGeometryChanged``).

        Mirrors the in-memory mutation that ``_on_roi_changed`` does
        for detected/fitted peaks during an image-side ROI drag:
        updates the cached ``_frame_peaks`` PeakTable so the colored
        overlay tracks the drag live; recomputes the row's Cartesian
        ``q_xy`` / ``q_z`` from the new polar coordinates; re-renders
        and re-syncs the image-space ROI. Disk persistence happens
        separately on drag-end via the host's
        ``_on_peak_row_write_requested`` slot.
        """
        if (
            self._selected is None
            or self._selected.kind != "detected"
            or sel is not self._selected
            or self._roi_item is None
        ):
            return
        peaks_for_frame = self._frame_peaks.get(sel.frame) or {}
        table = peaks_for_frame.get("detected")
        if table is None or len(table) == 0:
            return
        matches = np.where(table.ids == sel.peak_id)[0]
        if matches.size == 0:
            return
        idx = int(matches[0])
        table.radius[idx] = sel.radius
        table.radius_width[idx] = sel.radius_width
        table.angle[idx] = sel.angle
        table.angle_width[idx] = sel.angle_width
        table.q_xy[idx], table.q_z[idx] = polar_to_qxyz(sel.radius, sel.angle)
        self._sync_roi_geometry()
        self._render_overlays(self.current_frame)
        self.peakGeometryChanged.emit(self._selected)
