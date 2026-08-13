"""Peak editing: add to detected/fitted, copy/paste (single frame and to-range), batch 2D fit, deletes, the undo-push plumbing and the Peaks-table sync.

Plain mixin over ``MainWindow``: no __init__, no Signals; all state
lives on the combined class. Split out of ``main_window`` in the 2026
source split.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QInputDialog,
    QMessageBox,
    QProgressDialog,
)
from mlgidlab import (
    file_model,
    frame_range,
    peak_clipboard,
    pipeline,
)
from mlgidlab.image_viewer import SelectedPeak
from mlgidlab.parameter_panel import ParameterPanel
from mlgidlab.pipeline import PipelineCommand
from mlgidlab.widgets import skin_progress


class _CallbackAction:
    """Generic ``_Action`` (per the image_viewer protocol) that
    delegates ``undo`` / ``redo`` to caller-supplied callables.

    Lets host-level operations participate in the viewer's undo
    stack without dragging file-mutation code across the viewer
    abstraction. Used by ``_on_add_to_fitted`` so Ctrl+Z reverts a
    just-committed fitted row.
    """

    def __init__(self, undo_fn, redo_fn) -> None:
        self._undo_fn = undo_fn
        self._redo_fn = redo_fn

    def undo(self, _viewer) -> None:
        self._undo_fn()

    def redo(self, _viewer) -> None:
        self._redo_fn()


class PeaksMixin:
    def _on_add_to_detected(self) -> None:
        """Commit the active manual box to detected_peaks at the chosen
        confidence score.

        Writes synchronously via ``file_model.add_detected_peak_row`` —
        the same path copy/paste uses — so the score picked in the
        Parameter panel's Score-row dropdown (High = 1.0 / Medium = 0.5 /
        Low = 0.1) is stored verbatim. ``mlgidBASE.add_peak`` takes no
        score argument, so it can't carry the choice; a hand-drawn box
        has no model provenance anyway, so the user's pick is the
        authority. The manual box is left in place so it can also be sent
        to fitted_peaks or tweaked further. One grouped Ctrl+Z removes the
        added row.
        """
        if self._is_busy():
            return
        sel = self.viewer.selected_peak
        entry = self.entry_combo.currentText()
        if sel is None or sel.kind != "manual" or sel.manual_ref is None or not entry:
            return
        manual_peak = sel.manual_ref
        frame = int(self.viewer.current_frame)
        score = float(self.parameter_panel.confidence_score())
        geom = {
            "radius": float(manual_peak.radius),
            "angle": float(manual_peak.angle),
            "radius_width": float(manual_peak.radius_width),
            "angle_width": float(manual_peak.angle_width),
            "is_ring": bool(manual_peak.is_ring),
        }
        with self._detached_silx_tree():
            try:
                new_id = int(file_model.add_detected_peak_row(
                    self.session.temp_path, entry, frame,
                    score=score, **geom,
                ))
            except KeyError as exc:
                QMessageBox.warning(self, "Add to detected", str(exc))
                return
            except Exception as exc:
                QMessageBox.critical(self, "Add to detected", str(exc))
                return
        self.session.mark_dirty()
        self._update_title()
        self._load_entry_into_viewer(entry, preserve_view=True)
        self._add_detected_to_selection(entry, frame, [new_id])
        self._push_add_detected_undo(
            entry=entry, frame=frame, peak_id=new_id, score=score, geom=geom,
        )
        msg = (
            f"Added detected peak (score {score:.2f}) on "
            f"{entry}/frame{frame:05d}"
        )
        self.statusBar().showMessage(msg, 4000)
        self.pipeline_panel.append_log(msg)

    def _undoable_write_step(
        self, *, entry: str, frame: int | None, title: str,
        write, after=None,
    ) -> None:
        """The shared body of the undo/redo closures below.

        Bail while busy, navigate to the written entry (and frame, when
        given), run ``write`` inside the detached silx tree with
        ``QMessageBox.critical(title)`` on failure, then mark dirty,
        refresh title + viewer and hand ``write``'s return value to
        ``after`` (selection fixes, id rotation, logging). Closures
        whose sequence deviates (``_push_fitted_add_undo``'s
        removed-row bail-out, ``_push_paste_to_range_undo``'s
        no-navigation frame capture) keep their own bodies.
        """
        if self._is_busy():
            return
        if self.entry_combo.currentText() != entry:
            self.entry_combo.setCurrentText(entry)
        if frame is not None and int(self.viewer.current_frame) != int(frame):
            self.viewer.set_frame(int(frame))
        with self._detached_silx_tree():
            try:
                result = write()
            except Exception as exc:
                QMessageBox.critical(self, title, str(exc))
                return
        self.session.mark_dirty()
        self._update_title()
        self._load_entry_into_viewer(entry, preserve_view=True)
        if after is not None:
            after(result)

    def _push_add_detected_undo(
        self, *, entry: str, frame: int, peak_id: int,
        score: float, geom: dict,
    ) -> None:
        """Undo / redo for one 'Add to detected' commit.

        ``undo`` deletes the added row; ``redo`` re-adds it with the same
        geometry + score (fresh id). ``state["id"]`` rotates with each
        re-add so a follow-on undo deletes the right row — same pattern as
        ``_push_paste_undo`` / ``_push_delete_undo``.
        """
        state = {"id": int(peak_id)}
        g = dict(geom)

        def _undo() -> None:
            def write():
                file_model.delete_peak_row(
                    self.session.temp_path, entry,
                    frame=int(frame), kind="detected",
                    peak_id=int(state["id"]),
                )

            def after(_result) -> None:
                self._drop_detected_from_selection(int(frame), [state["id"]])
                self.pipeline_panel.append_log(
                    f"Undo add to detected: removed peak on "
                    f"{entry}/frame{int(frame):05d}"
                )

            self._undoable_write_step(
                entry=entry, frame=frame,
                title="Undo add to detected", write=write, after=after,
            )

        def _redo() -> None:
            def write():
                return int(file_model.add_detected_peak_row(
                    self.session.temp_path, entry, int(frame),
                    score=float(score), **g,
                ))

            def after(new_id) -> None:
                state["id"] = new_id
                self._add_detected_to_selection(entry, int(frame), [new_id])
                self.pipeline_panel.append_log(
                    f"Redo add to detected: re-added peak on "
                    f"{entry}/frame{int(frame):05d}"
                )

            self._undoable_write_step(
                entry=entry, frame=frame,
                title="Redo add to detected", write=write, after=after,
            )

        self.viewer._push_undo(_CallbackAction(_undo, _redo))

    def _on_add_to_fitted(self) -> None:
        """Append a row to fitted_peaks for the active manual / detected box.

        Dispatches strictly on ``parameter_panel.fit_mode()``:

        * ``FIT_MODE_2D`` (pygidfit) → route through
          ``manual_fit.fit_one_peak``; the persisted row carries real
          A/B/C/theta shape coefficients — identical to what the
          pipeline ``run_fitting`` writes for the same box.
        * ``FIT_MODE_1D`` (legacy scipy) → use the profile viewer's
          cached 1D fits + zero-fill the 2D shape coefficients. Width
          convention matches the 2D path so saved boxes render
          identically: ``radius_width = angle_width = 2σ ≈ 0.849 ×
          FWHM`` (see ``_build_fitted_row_1d``).

        Ring storage forces 1D regardless of the radio (pygidfit's
        segment model can't fit rings — the ring saved row uses
        ``angle = 45°``, ``angle_width = ∞``).

        Strict failure: if the chosen mode can't produce a fit (1D
        without scipy convergence, or 2D with pygidfit raising / NaN /
        missing geometry), the user gets a clear error naming both
        the chosen mode and the reason — no silent fall-back to the
        other mode. Physics-audit finding F-06 closure.
        """
        if self._is_busy():
            return
        sel = self.viewer.selected_peak
        entry = self.entry_combo.currentText()
        if sel is None or sel.kind not in ("manual", "detected") or not entry:
            return
        save_as_ring = self.parameter_panel.save_as_ring()
        frame = self.viewer.current_frame
        # ``fit_mode()`` already collapses to FIT_MODE_1D when ring is
        # checked, so the rest of this function just branches on the
        # token.
        mode = self.parameter_panel.fit_mode()

        if mode == ParameterPanel.FIT_MODE_2D:
            row = self._build_fitted_row_2d(sel, entry, frame)
        else:
            row = self._build_fitted_row_1d(sel, save_as_ring)
        if row is None:
            # The helper already showed a QMessageBox / log line.
            return

        with self._detached_silx_tree():
            try:
                new_id = file_model.add_fitted_peak_row(
                    self.session.temp_path, entry, frame,
                    radius=row["radius"],
                    radius_width=row["radius_width"],
                    angle=row["angle"],
                    angle_width=row["angle_width"],
                    amplitude=row["amplitude"],
                    is_ring=save_as_ring,
                    theta=row["theta"],
                    A=row["A"],
                    B=row["B"],
                    C=row["C"],
                )
            except KeyError as exc:
                QMessageBox.warning(self, "Add to fitted", str(exc))
                return
            except Exception as exc:
                QMessageBox.critical(self, "Add to fitted", str(exc))
                return

        self.session.mark_dirty()
        self._update_title()
        # Add-to-fitted only appends a fitted row (fresh id); no existing
        # id is invalidated, so prior undo history is preserved.
        # Pull the fresh fitted_peaks (and matched, which references it) back
        # into the viewer — same entry, so preserve the user's zoom + frame.
        self._load_entry_into_viewer(entry, preserve_view=True)
        # In 2D mode the user committed without seeing pygidfit's actual
        # result on the profile plots (the pink curves are hidden in
        # 2D mode — see ``_apply_fit_curve_visibility``). Auto-switch
        # the selection to the new fitted peak so the result is
        # immediately visible (cyan box + profile fits both update to
        # the fitted-kind selection). 1D mode is unchanged because the
        # pink curves already previewed what was saved.
        if mode == ParameterPanel.FIT_MODE_2D:
            fitted_sel = SelectedPeak(
                kind="fitted",
                frame=int(frame),
                peak_id=int(new_id),
                radius=float(row["radius"]),
                angle=float(row["angle"]),
                radius_width=float(row["radius_width"]),
                angle_width=float(row["angle_width"]),
                is_ring=bool(save_as_ring),
                amplitude=(
                    float(row["amplitude"])
                    if row.get("amplitude") is not None else None
                ),
            )
            # ``preserve_manual=True`` so the source manual peak
            # is not auto-removed by the away-from-manual side
            # effect on _set_selected. Without this, the silent
            # ManualRemoveAction it pushes would sit on the undo
            # stack alongside our FittedRowAction and the user
            # would need TWO Ctrl+Z presses to revert the commit.
            self.viewer._set_selected(fitted_sel, preserve_manual=True)
        # Ctrl+Z support for Add-to-fitted: the file mutation is
        # reversible (delete the just-added row + restore the source
        # selection). Pushed onto the existing history so the commit
        # stacks with any prior undoable edits.
        self._push_fitted_add_undo(
            entry=entry, frame=int(frame), new_id=int(new_id),
            add_kwargs={
                "radius": row["radius"],
                "radius_width": row["radius_width"],
                "angle": row["angle"],
                "angle_width": row["angle_width"],
                "amplitude": row["amplitude"],
                "is_ring": save_as_ring,
                "theta": row["theta"],
                "A": row["A"], "B": row["B"], "C": row["C"],
            },
            source_sel=sel,
        )
        # Ring toggle is sticky across selections but reset on commit so
        # the next Add-to-fitted defaults back to segment unless the user
        # explicitly opts in again.
        self.parameter_panel.reset_save_as_ring()
        self.pipeline_panel.append_log(
            f"Added fitted peak id={new_id} "
            f"({'ring' if save_as_ring else 'segment'} from {sel.kind}) on "
            f"{entry}/frame{frame:05d}"
        )

    def _push_fitted_add_undo(
        self, *, entry: str, frame: int, new_id: int,
        add_kwargs: dict, source_sel: SelectedPeak,
    ) -> None:
        """Push a Ctrl+Z entry for an Add-to-fitted commit.

        ``undo`` deletes the just-added fitted row (and cascades
        ``clear_peaks(matched)`` on the frame, mirroring the
        existing fitted-delete handler), then reselects the source
        peak so the user is back where they started visually. For
        a manual source, the manual peak still exists in
        ``viewer._manual_peaks`` (Add-to-fitted leaves it alone),
        so reselecting it surfaces the yellow ROI again — exactly
        the "let the manual selection reappear" behaviour from the
        feature spec. For a detected source, the detected row is
        located by id in the freshly-reloaded peaks table.

        ``redo`` re-adds the row with the original kwargs.
        ``file_model.add_fitted_peak_row`` assigns a fresh id on
        each call (``max(id)+1``), so the post-redo id may differ
        from the original; ``state["current_id"]`` is updated so
        a follow-on undo deletes the correct row.

        On either side, any file error surfaces a QMessageBox and
        leaves the stack alone — the caller's downstream slot
        (``undo_last_action`` / ``redo_last_action``) is tolerant
        of no-op actions.
        """
        # Closure state for the rotating fitted-row id across
        # repeated undo/redo cycles.
        state = {"current_id": int(new_id)}
        source_kind = source_sel.kind
        source_peak_id = (
            int(source_sel.peak_id)
            if source_sel.peak_id is not None else None
        )

        def _undo() -> None:
            if self._is_busy():
                return
            if self.entry_combo.currentText() != entry:
                self.entry_combo.setCurrentText(entry)
            if int(self.viewer.current_frame) != int(frame):
                self.viewer.set_frame(int(frame))
            with self._detached_silx_tree():
                try:
                    removed = file_model.delete_peak_row(
                        self.session.temp_path, entry,
                        frame=int(frame), kind="fitted",
                        peak_id=int(state["current_id"]),
                    )
                    if removed > 0:
                        file_model.clear_peaks(
                            self.session.temp_path, entry,
                            kind="matched", frame=int(frame),
                        )
                except Exception as exc:
                    QMessageBox.critical(
                        self, "Undo Add to fitted", str(exc)
                    )
                    return
            if removed == 0:
                self.pipeline_panel.append_log(
                    f"Undo Add-to-fitted: fitted id="
                    f"{state['current_id']} already gone on "
                    f"{entry}/frame{int(frame):05d}"
                )
                return
            self.session.mark_dirty()
            self._update_title()
            self._load_entry_into_viewer(entry, preserve_view=True)
            # Reselect the source peak so the visual state matches
            # what the user had before Add-to-fitted.
            if source_kind == "manual":
                manuals = self.viewer._manual_peaks.get(int(frame), [])
                if manuals:
                    self.viewer._set_selected(
                        SelectedPeak.from_manual(manuals[0], int(frame))
                    )
            elif source_kind == "detected" and source_peak_id is not None:
                tables = self.viewer._frame_peaks.get(int(frame)) or {}
                det = tables.get("detected")
                if det is not None:
                    for i in range(len(det)):
                        if int(det.ids[i]) == source_peak_id:
                            self._select_table_row(
                                int(frame), "detected", det, i
                            )
                            break
            self.pipeline_panel.append_log(
                f"Undo Add-to-fitted: removed fitted id="
                f"{state['current_id']} on "
                f"{entry}/frame{int(frame):05d}"
            )

        def _redo() -> None:
            if self._is_busy():
                return
            if self.entry_combo.currentText() != entry:
                self.entry_combo.setCurrentText(entry)
            if int(self.viewer.current_frame) != int(frame):
                self.viewer.set_frame(int(frame))
            with self._detached_silx_tree():
                try:
                    readded_id = file_model.add_fitted_peak_row(
                        self.session.temp_path, entry, int(frame),
                        **add_kwargs,
                    )
                except Exception as exc:
                    QMessageBox.critical(
                        self, "Redo Add to fitted", str(exc)
                    )
                    return
            state["current_id"] = int(readded_id)
            self.session.mark_dirty()
            self._update_title()
            self._load_entry_into_viewer(entry, preserve_view=True)
            # Re-select the re-added fitted peak if the user is
            # currently in 2D mode (mirrors the original commit's
            # auto-select). In 1D mode leave selection alone, also
            # mirroring the original.
            if self.parameter_panel.fit_mode() == ParameterPanel.FIT_MODE_2D:
                # Same ``preserve_manual=True`` rationale as the
                # original commit's auto-switch: keep the manual
                # source intact across the redo so a subsequent
                # Ctrl+Z still has the manual peak to reselect.
                self.viewer._set_selected(
                    SelectedPeak(
                        kind="fitted", frame=int(frame),
                        peak_id=int(readded_id),
                        radius=float(add_kwargs["radius"]),
                        angle=float(add_kwargs["angle"]),
                        radius_width=float(add_kwargs["radius_width"]),
                        angle_width=float(add_kwargs["angle_width"]),
                        is_ring=bool(add_kwargs["is_ring"]),
                        amplitude=(
                            float(add_kwargs["amplitude"])
                            if add_kwargs.get("amplitude") is not None
                            else None
                        ),
                    ),
                    preserve_manual=True,
                )
            self.pipeline_panel.append_log(
                f"Redo Add-to-fitted: re-added fitted id="
                f"{readded_id} on {entry}/frame{int(frame):05d}"
            )

        self.viewer._push_undo(_CallbackAction(_undo, _redo))

    # -- Copy / paste detected peaks ---------------------------------

    def _on_copy_peaks(self) -> None:
        """Snapshot the selected detected peaks into the in-app clipboard.

        Filters the multi-selection to ``kind="detected"`` — non-detected
        kinds are deferred per user scope. No-op + status-bar hint when
        nothing copyable is selected. The clipboard remembers the source
        entry so a same-entry paste check is enforced at paste time.
        """
        if self.session is None:
            return
        sels = [s for s in self.viewer.selected_peaks() if s.kind == "detected"]
        if not sels:
            self.statusBar().showMessage(
                "Copy: select one or more detected peaks first", 4000
            )
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        items = [
            peak_clipboard.ClipboardItem(
                radius=float(s.radius),
                angle=float(s.angle),
                radius_width=float(s.radius_width),
                angle_width=float(s.angle_width),
                is_ring=bool(s.is_ring),
                source_frame=int(s.frame),
                source_peak_id=int(s.peak_id) if s.peak_id is not None else -1,
                score=float(s.score) if s.score is not None else 0.0,
            )
            for s in sels
        ]
        peak_clipboard.set_items(items, entry=entry)
        self.statusBar().showMessage(
            f"Copied {len(items)} detected peak(s)", 4000
        )

    def _on_paste_peaks(self) -> None:
        """Paste clipboard contents onto the current frame as detected rows.

        Same-entry only: ``peak_clipboard.take_items`` returns ``[]`` for
        an entry mismatch. Writes synchronously via
        ``file_model.add_detected_peak_row`` so the new ids are known
        immediately and we can multi-select the pasted peaks after the
        reload. Pushes one grouped ``_CallbackAction`` so a single
        Ctrl+Z reverses the whole paste.
        """
        if self._is_busy():
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        items = peak_clipboard.take_items(target_entry=entry)
        if not items:
            self.statusBar().showMessage(
                "Paste: clipboard empty or copied from a different entry",
                4000,
            )
            return
        frame = int(self.viewer.current_frame)
        added_ids: list[int] = []
        with self._detached_silx_tree():
            try:
                for item in items:
                    new_id = file_model.add_detected_peak_row(
                        self.session.temp_path, entry, frame,
                        radius=item.radius,
                        angle=item.angle,
                        radius_width=item.radius_width,
                        angle_width=item.angle_width,
                        is_ring=item.is_ring,
                        score=item.score,
                    )
                    added_ids.append(int(new_id))
            except KeyError as exc:
                QMessageBox.warning(self, "Paste detected peaks", str(exc))
                return
            except Exception as exc:
                QMessageBox.critical(self, "Paste detected peaks", str(exc))
                return
        if not added_ids:
            return
        self.session.mark_dirty()
        self._update_title()
        # Paste only appends rows (fresh ids); no existing id is
        # invalidated, so prior undo history is preserved.
        self._load_entry_into_viewer(entry, preserve_view=True)
        # Add the pasted rows to the existing multi-selection so the
        # user immediately sees what landed without losing whatever
        # they had selected before. If the prior primary wasn't a
        # detected peak (or there was no selection), the first pasted
        # row becomes the new primary.
        self._add_detected_to_selection(entry, frame, added_ids)
        self._push_paste_undo(
            entry=entry, frame=frame, ids=added_ids, items=items,
        )
        self.statusBar().showMessage(
            f"Pasted {len(added_ids)} detected peak(s) on "
            f"{entry}/frame{frame:05d}",
            4000,
        )
        self.pipeline_panel.append_log(
            f"Pasted {len(added_ids)} detected peak(s) on "
            f"{entry}/frame{frame:05d}"
        )

    def _build_detected_sels(
        self, frame: int, ids: list[int], kind: str = "detected",
    ) -> list[SelectedPeak]:
        """Look up the freshly-loaded ``kind`` rows with the given ids
        on ``frame`` and return them as ``SelectedPeak`` snapshots.

        ``kind`` is ``"detected"`` (default) or ``"fitted"``. Silently
        skips ids that aren't present in the table (e.g. after a
        follow-on pipeline rerun). Used by the paste / bulk-delete
        handlers and their undo/redo paths to (re)build a selection."""
        if not ids:
            return []
        tables = self.viewer._frame_peaks.get(int(frame)) or {}
        det = tables.get(kind)
        if det is None:
            return []
        id_set = {int(x) for x in ids}
        sels: list[SelectedPeak] = []
        for i in range(len(det)):
            if int(det.ids[i]) in id_set:
                sels.append(SelectedPeak(
                    kind=kind, frame=int(frame),
                    peak_id=int(det.ids[i]),
                    radius=float(det.radius[i]),
                    angle=float(det.angle[i]),
                    radius_width=float(det.radius_width[i]),
                    angle_width=float(det.angle_width[i]),
                    is_ring=bool(det.is_ring[i]),
                    score=float(det.score[i]),
                    amplitude=float(det.amplitude[i]),
                ))
        return sels

    def _add_detected_to_selection(
        self, entry: str, frame: int, ids: list[int], kind: str = "detected",
    ) -> None:
        """Add the given ``kind`` rows to the existing multi-selection.

        ``kind`` is ``"detected"`` (default) or ``"fitted"``. Behaviour
        mirrors a sequence of Ctrl+clicks: if the current primary is the
        same kind, all ``ids`` are appended to ``_selected_extras``;
        otherwise the first id becomes the new primary (so the
        multi-selection can grow) and the rest go to extras. Lets
        paste / delete-undo grow the user's selection instead of
        wiping it.
        """
        sels = self._build_detected_sels(frame, ids, kind=kind)
        if not sels:
            return
        v = self.viewer
        if v._selected is None or v._selected.kind != kind:
            # ``preserve_manual`` so a previously-selected manual
            # peak isn't auto-removed when we promote a peak to
            # primary (matches the post-Add-to-fitted auto-select
            # pattern at ``_on_add_to_fitted``).
            v._set_selected(sels[0], preserve_manual=True)
            v._selected_extras = list(sels[1:])
        else:
            # Avoid double-adding if any id is already in the
            # selection.
            existing = {
                (s.kind, int(s.frame), int(s.peak_id))
                for s in v.selected_peaks()
            }
            for s in sels:
                key = (s.kind, int(s.frame), int(s.peak_id))
                if key not in existing:
                    v._selected_extras.append(s)
        v._sync_roi()
        v._render_overlays(v.current_frame)
        v.selectionChanged.emit(v._selected)
        v.selectionsChanged.emit(v.selected_peaks())

    def _drop_detected_from_selection(
        self, frame: int, ids: list[int], kind: str = "detected",
    ) -> None:
        """Remove the given ``kind`` rows from the multi-selection.

        ``kind`` is ``"detected"`` (default) or ``"fitted"``. Used by
        the undo/redo paths so the highlight overlay tracks the on-disk
        state — without this, just-deleted rows would keep painting
        white selection boxes at their old positions (the SelectedPeak
        holds its own geometry independent of the peak table).
        """
        v = self.viewer
        if v._selected is None and not v._selected_extras:
            return
        id_set = {(int(frame), int(i)) for i in ids}

        def _hit(s: SelectedPeak) -> bool:
            return (
                s.kind == kind
                and (int(s.frame), int(s.peak_id)) in id_set
            )

        # Filter extras first, then handle the primary.
        v._selected_extras = [
            s for s in v._selected_extras if not _hit(s)
        ]
        if v._selected is not None and _hit(v._selected):
            if v._selected_extras:
                v._selected = v._selected_extras.pop(0)
            else:
                v._selected = None
        v._sync_roi()
        v._render_overlays(v.current_frame)
        v.selectionChanged.emit(v._selected)
        v.selectionsChanged.emit(v.selected_peaks())

    def _push_paste_undo(
        self, *, entry: str, frame: int,
        ids: list[int], items: list,
    ) -> None:
        """Undo / redo for a paste batch — one grouped action.

        ``state["ids"]`` is the rotating list of fitted ids; each redo
        re-adds rows and overwrites it with the freshly-assigned ids
        so a follow-on undo deletes the right rows. Same pattern as
        ``_push_fitted_add_undo`` extended for N rows.
        """
        state = {"ids": list(ids)}
        snapshot = list(items)

        def _undo() -> None:
            def write():
                for peak_id in state["ids"]:
                    file_model.delete_peak_row(
                        self.session.temp_path, entry,
                        frame=int(frame), kind="detected",
                        peak_id=int(peak_id),
                    )

            def after(_result) -> None:
                # Drop the now-deleted rows from the multi-selection so
                # the white highlight overlay tracks the on-disk state.
                self._drop_detected_from_selection(int(frame), state["ids"])
                self.pipeline_panel.append_log(
                    f"Undo paste: removed {len(state['ids'])} detected "
                    f"peak(s) on {entry}/frame{int(frame):05d}"
                )

            self._undoable_write_step(
                entry=entry, frame=frame,
                title="Undo paste", write=write, after=after,
            )

        def _redo() -> None:
            def write():
                new_ids: list[int] = []
                for item in snapshot:
                    new_id = file_model.add_detected_peak_row(
                        self.session.temp_path, entry, int(frame),
                        radius=item.radius,
                        angle=item.angle,
                        radius_width=item.radius_width,
                        angle_width=item.angle_width,
                        is_ring=item.is_ring,
                        score=item.score,
                    )
                    new_ids.append(int(new_id))
                return new_ids

            def after(new_ids) -> None:
                state["ids"] = new_ids
                # Re-add to the existing selection (additive, matching
                # the original paste).
                self._add_detected_to_selection(entry, int(frame), new_ids)
                self.pipeline_panel.append_log(
                    f"Redo paste: re-added {len(new_ids)} detected "
                    f"peak(s) on {entry}/frame{int(frame):05d}"
                )

            self._undoable_write_step(
                entry=entry, frame=frame,
                title="Redo paste", write=write, after=after,
            )

        self.viewer._push_undo(_CallbackAction(_undo, _redo))

    def _on_paste_peaks_to_range(self) -> None:
        """Paste the clipboard onto every frame in a typed range.

        Ctrl+Shift+V entry point. Mirrors ``_on_paste_peaks`` but
        loops the frame set produced by ``frame_range.parse_frame_range``.
        ``take_items`` is non-draining, so the same snapshot is replayed
        on every target frame and remains available for further pastes.

        UX rules (confirmed with the user):
          * Out-of-range frames are filtered with a confirmation dialog
            naming the count + compact range list.
          * Duplicates in the input deduplicate silently.
          * After a successful paste, only the current frame's rows
            (if it landed in the range) get added to the multi-selection.
          * Cancel mid-loop keeps already-written frames.
          * One grouped ``_CallbackAction`` reverses the entire batch.
        """
        if self._is_busy():
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        items = peak_clipboard.take_items(target_entry=entry)
        if not items:
            self.statusBar().showMessage(
                "Paste: clipboard empty or copied from a different entry",
                4000,
            )
            return
        n_frames = int(self.viewer.n_frames)
        if n_frames <= 0:
            return
        text, ok = QInputDialog.getText(
            self, "Paste to frames",
            f"Frames (e.g. 0-{max(n_frames - 1, 0)},37):",
            text=str(int(self.viewer.current_frame)),
        )
        if not ok:
            return
        try:
            valid, dropped = frame_range.parse_frame_range(
                text, n_frames=n_frames,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Paste to frames", str(exc))
            return
        if dropped:
            reply = QMessageBox.question(
                self, "Paste to frames",
                f"Pasting to {len(valid)} frame(s). "
                f"{len(dropped)} frame(s) outside the {n_frames}-frame "
                f"stack will be skipped: "
                f"{frame_range.compact_repr(dropped)}.\n\nContinue?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        if not valid:
            self.statusBar().showMessage(
                "Paste to frames: nothing to do — no frames in range",
                4000,
            )
            return

        n_frames_to_write = len(valid)
        dialog = skin_progress(QProgressDialog(
            f"Pasting to {n_frames_to_write} frame(s)…",
            "Cancel", 0, n_frames_to_write, self,
        ))
        dialog.setWindowTitle("Paste to frames")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setValue(0)

        per_frame_ids: dict[int, list[int]] = {}
        cancelled = False
        with self._detached_silx_tree():
            try:
                for i, target_frame in enumerate(valid):
                    if dialog.wasCanceled():
                        cancelled = True
                        break
                    dialog.setLabelText(
                        f"Pasting frame {i + 1}/{n_frames_to_write} "
                        f"(frame {target_frame:05d})…"
                    )
                    QApplication.processEvents()
                    frame_ids: list[int] = []
                    for item in items:
                        new_id = file_model.add_detected_peak_row(
                            self.session.temp_path, entry,
                            int(target_frame),
                            radius=item.radius,
                            angle=item.angle,
                            radius_width=item.radius_width,
                            angle_width=item.angle_width,
                            is_ring=item.is_ring,
                            score=item.score,
                        )
                        frame_ids.append(int(new_id))
                    per_frame_ids[int(target_frame)] = frame_ids
                    dialog.setValue(i + 1)
                    QApplication.processEvents()
            except KeyError as exc:
                dialog.close()
                QMessageBox.warning(self, "Paste to frames", str(exc))
                return
            except Exception as exc:
                dialog.close()
                QMessageBox.critical(self, "Paste to frames", str(exc))
                return
        dialog.close()

        if not per_frame_ids:
            return

        self.session.mark_dirty()
        self._update_title()
        # Append-only across the range (fresh ids per frame); prior undo
        # history is preserved.
        self._load_entry_into_viewer(entry, preserve_view=True)
        # Only the current frame's pasted rows get added to the
        # selection (per confirmed UX). Off-frame selections would
        # be pruned on frame switch anyway.
        current_frame = int(self.viewer.current_frame)
        if current_frame in per_frame_ids:
            self._add_detected_to_selection(
                entry, current_frame, per_frame_ids[current_frame],
            )
        self._push_paste_to_range_undo(
            entry=entry, items=items, per_frame_ids=per_frame_ids,
        )

        n_rows = sum(len(v) for v in per_frame_ids.values())
        n_frames_written = len(per_frame_ids)
        summary = (
            f"Pasted {n_rows} detected peak(s) across "
            f"{n_frames_written} frame(s) on {entry}"
        )
        if dropped:
            summary += f" ({len(dropped)} out-of-range frame(s) skipped)"
        if cancelled:
            summary += " (cancelled)"
        self.statusBar().showMessage(summary, 6000)
        self.pipeline_panel.append_log(summary)

    def _push_paste_to_range_undo(
        self, *, entry: str, items: list,
        per_frame_ids: dict[int, list[int]],
    ) -> None:
        """Undo / redo for a range paste — one grouped action.

        ``state["per_frame_ids"]`` rotates on every redo so a follow-on
        undo deletes the correct (freshly-assigned) ids. Same idea as
        ``_push_paste_undo`` extended from a single frame to a frame
        dict; frames the original Cancel skipped never enter the dict
        so they're left alone on undo/redo.
        """
        state: dict[str, dict[int, list[int]]] = {
            "per_frame_ids": {
                int(f): [int(i) for i in ids]
                for f, ids in per_frame_ids.items()
            },
        }
        snapshot = list(items)

        def _undo() -> None:
            if self._is_busy():
                return
            if self.entry_combo.currentText() != entry:
                self.entry_combo.setCurrentText(entry)
            current_frame = int(self.viewer.current_frame)
            with self._detached_silx_tree():
                try:
                    for f, ids in state["per_frame_ids"].items():
                        for peak_id in ids:
                            file_model.delete_peak_row(
                                self.session.temp_path, entry,
                                frame=int(f), kind="detected",
                                peak_id=int(peak_id),
                            )
                except Exception as exc:
                    QMessageBox.critical(
                        self, "Undo paste to frames", str(exc)
                    )
                    return
            self.session.mark_dirty()
            self._update_title()
            self._load_entry_into_viewer(entry, preserve_view=True)
            if current_frame in state["per_frame_ids"]:
                self._drop_detected_from_selection(
                    current_frame, state["per_frame_ids"][current_frame],
                )
            n_rows = sum(len(v) for v in state["per_frame_ids"].values())
            n_frames_written = len(state["per_frame_ids"])
            self.pipeline_panel.append_log(
                f"Undo paste to frames: removed {n_rows} detected "
                f"peak(s) across {n_frames_written} frame(s) on {entry}"
            )

        def _redo() -> None:
            if self._is_busy():
                return
            if self.entry_combo.currentText() != entry:
                self.entry_combo.setCurrentText(entry)
            current_frame = int(self.viewer.current_frame)
            new_per_frame_ids: dict[int, list[int]] = {}
            with self._detached_silx_tree():
                try:
                    for f in sorted(state["per_frame_ids"].keys()):
                        frame_ids: list[int] = []
                        for item in snapshot:
                            new_id = file_model.add_detected_peak_row(
                                self.session.temp_path, entry, int(f),
                                radius=item.radius,
                                angle=item.angle,
                                radius_width=item.radius_width,
                                angle_width=item.angle_width,
                                is_ring=item.is_ring,
                                score=item.score,
                            )
                            frame_ids.append(int(new_id))
                        new_per_frame_ids[int(f)] = frame_ids
                except Exception as exc:
                    QMessageBox.critical(
                        self, "Redo paste to frames", str(exc)
                    )
                    return
            state["per_frame_ids"] = new_per_frame_ids
            self.session.mark_dirty()
            self._update_title()
            self._load_entry_into_viewer(entry, preserve_view=True)
            if current_frame in new_per_frame_ids:
                self._add_detected_to_selection(
                    entry, current_frame, new_per_frame_ids[current_frame],
                )
            n_rows = sum(len(v) for v in new_per_frame_ids.values())
            n_frames_written = len(new_per_frame_ids)
            self.pipeline_panel.append_log(
                f"Redo paste to frames: re-added {n_rows} detected "
                f"peak(s) across {n_frames_written} frame(s) on {entry}"
            )

        self.viewer._push_undo(_CallbackAction(_undo, _redo))

    # -- Batch 2D fit ------------------------------------------------

    def _refresh_fit_buttons(self) -> None:
        """Re-evaluate visibility + enable state for the fit buttons.

        Visibility rule (per user request — only show the button that
        makes sense, don't grey them out):

        * Exactly one fittable peak selected (manual or detected) →
          'Add to fitted' visible, 'Fit selected (2D)' hidden.
        * Two or more detected peaks selected → 'Fit selected (2D)'
          visible, 'Add to fitted' hidden.
        * Otherwise (no selection, only fitted / matched, or the
          mixed-kind edge case of 1 detected + 1 manual which neither
          button handles cleanly) → both hidden.

        'Add to detected' stays in place — its enable state is driven
        by ``ParameterPanel._update_actions_enabled`` (greyed out for
        non-manual selections).

        Enable state for 'Fit selected (2D)' is still gated on
        save-as-ring being OFF (ring forces 1D).
        """
        sels = self.viewer.selected_peaks()
        n_fittable = sum(1 for s in sels if s.kind in ("manual", "detected"))
        n_detected = sum(1 for s in sels if s.kind == "detected")
        ring_on = self.parameter_panel.save_as_ring()
        show_add_fitted = (n_fittable == 1)
        show_batch_fit = (n_detected >= 2 and not ring_on)
        self.parameter_panel.set_fit_button_visibility(
            add_fitted=show_add_fitted,
            fit_selected_2d=show_batch_fit,
        )
        # Keep the enable state coherent so a click on the visible
        # batch-fit button always proceeds (otherwise a stale enable=False
        # from before the visibility change could leak through).
        self.parameter_panel.set_batch_fit_enabled(show_batch_fit)
        # ≥2 fittable selected → batch fit is the only sensible
        # action, and batch fit is 2D-only. Grey out the 1D radio
        # to make that constraint visible.
        self.parameter_panel.set_multi_select_active(n_fittable >= 2)

    def _on_batch_fit_2d(self) -> None:
        """Run pygidfit on every selected detected peak, in one batch.

        Two-phase to keep the viewer's FrameSource open while pygidfit
        reads frame data:

        1. **Fit phase** (no silx detach): loop the selections,
           calling ``_run_pygidfit_for_selection`` which needs
           ``viewer._frame_source.is_open``. The QProgressDialog
           ticks per fit and respects Cancel.
        2. **Write phase** (single ``_detached_silx_tree`` scope):
           append a ``fitted_peaks`` row for every successful fit.
           h5py detach is expensive; doing it once for the whole
           batch keeps the run snappy.
        """
        if self._is_busy():
            return
        sels = [
            s for s in self.viewer.selected_peaks() if s.kind == "detected"
        ]
        if not sels:
            self.statusBar().showMessage(
                "Fit selected (2D): select detected peaks first", 4000
            )
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        n = len(sels)
        dialog = skin_progress(QProgressDialog(
            f"Fitting {n} peak(s) (2D)…", "Cancel", 0, n, self,
        ))
        dialog.setWindowTitle("Fit selected (2D)")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setValue(0)

        # Phase 1: fits. Each ``_run_pygidfit_for_selection`` call
        # reads ``viewer._frame_source.get_cartesian(frame)``, so the
        # silx tree must stay attached here (the detach in phase 2
        # closes the frame source).
        fit_results: list[tuple[SelectedPeak, object, str | None]] = []
        cancelled = False
        for i, sel in enumerate(sels):
            if dialog.wasCanceled():
                cancelled = True
                break
            dialog.setLabelText(
                f"Fitting peak {i + 1}/{n} (id={sel.peak_id})…"
            )
            QApplication.processEvents()
            fit_2d, err = self._run_pygidfit_for_selection(
                sel, entry, int(sel.frame),
            )
            fit_results.append((sel, fit_2d, err))
            dialog.setValue(i + 1)
            QApplication.processEvents()

        # Phase 2: writes. One ``_detached_silx_tree`` scope for the
        # whole batch — h5py detach is the expensive bit.
        added_ids: list[int] = []
        added_kwargs: list[dict] = []
        failures: list[tuple[int, str]] = []
        with self._detached_silx_tree():
            for sel, fit_2d, err in fit_results:
                if fit_2d is None:
                    failures.append((int(sel.peak_id), err or "fit failed"))
                    continue
                row = {
                    "radius": float(fit_2d.radius),
                    "radius_width": float(fit_2d.radius_width),
                    "angle": float(fit_2d.angle),
                    "angle_width": float(fit_2d.angle_width),
                    "amplitude": float(fit_2d.amplitude),
                    "is_ring": False,
                    "theta": float(fit_2d.theta),
                    "A": float(fit_2d.A),
                    "B": float(fit_2d.B),
                    "C": float(fit_2d.C),
                }
                try:
                    new_id = file_model.add_fitted_peak_row(
                        self.session.temp_path, entry, int(sel.frame),
                        **row,
                    )
                except Exception as exc:
                    failures.append((int(sel.peak_id), f"write failed: {exc}"))
                    continue
                added_ids.append(int(new_id))
                added_kwargs.append({**row, "frame": int(sel.frame)})
        dialog.close()

        if not added_ids:
            QMessageBox.warning(
                self, "Fit selected (2D)",
                "No peaks were fitted successfully.\n\n"
                + "\n".join(f"id={pid}: {msg}" for pid, msg in failures),
            )
            return

        self.session.mark_dirty()
        self._update_title()
        # Batch fit only appends fitted rows (fresh ids); the detected
        # ids it fit from are untouched, so prior undo history stays valid.
        self._load_entry_into_viewer(entry, preserve_view=True)
        # Multi-select the newly-added fitted rows. They may span
        # multiple frames if the original detected selection did
        # (rare — multi-select today is per-frame — but kept correct
        # for future when cross-frame multi-select lands).
        frames_touched = sorted({kw["frame"] for kw in added_kwargs})
        if frames_touched:
            # Select on the first touched frame; the user can navigate.
            self._select_fitted_ids(
                entry, frames_touched[0],
                [
                    new_id for new_id, kw in zip(added_ids, added_kwargs)
                    if kw["frame"] == frames_touched[0]
                ],
            )
        self._push_batch_fit_undo(
            entry=entry, added_ids=added_ids, added_kwargs=added_kwargs,
        )

        summary = (
            f"Fit {len(added_ids)}/{n} selected peaks "
            f"({len(failures)} failed"
            + (", cancelled" if cancelled else "")
            + ")"
        )
        self.statusBar().showMessage(summary, 6000)
        self.pipeline_panel.append_log(summary)
        for pid, msg in failures:
            self.pipeline_panel.append_log(f"  fit failed: id={pid} — {msg}")

    def _select_fitted_ids(
        self, entry: str, frame: int, ids: list[int],
    ) -> None:
        """Multi-select the fitted rows with the given ids on ``frame``."""
        if not ids:
            return
        tables = self.viewer._frame_peaks.get(int(frame)) or {}
        fit = tables.get("fitted")
        if fit is None:
            return
        id_set = {int(x) for x in ids}
        sels = []
        for i in range(len(fit)):
            if int(fit.ids[i]) in id_set:
                sels.append(SelectedPeak(
                    kind="fitted", frame=int(frame),
                    peak_id=int(fit.ids[i]),
                    radius=float(fit.radius[i]),
                    angle=float(fit.angle[i]),
                    radius_width=float(fit.radius_width[i]),
                    angle_width=float(fit.angle_width[i]),
                    is_ring=bool(fit.is_ring[i]),
                    score=float(fit.score[i]),
                    amplitude=float(fit.amplitude[i]),
                ))
        if not sels:
            return
        # Fitted is not currently in the extras allowlist (extras are
        # detected-only this iteration), so set the primary alone here.
        # Multi-selection of fitted peaks lands when fitted copy/paste
        # is enabled in a follow-up.
        self.viewer._set_selected(sels[0], preserve_manual=True)

    def _push_batch_fit_undo(
        self, *, entry: str, added_ids: list[int], added_kwargs: list[dict],
    ) -> None:
        """Undo / redo for a batch-fit run — one grouped action.

        Mirrors ``_push_fitted_add_undo`` extended to N rows. The
        rotating ``state["ids"]`` keeps undo correct after a redo
        re-assigns ids (one per redo'd row).
        """
        state = {"ids": list(added_ids)}
        snapshot_kwargs = [dict(kw) for kw in added_kwargs]

        def _undo() -> None:
            def write():
                for kw, peak_id in zip(snapshot_kwargs, state["ids"]):
                    file_model.delete_peak_row(
                        self.session.temp_path, entry,
                        frame=int(kw["frame"]), kind="fitted",
                        peak_id=int(peak_id),
                    )
                # Cascading clear on every touched frame mirrors
                # the single-fitted-delete behaviour.
                for f in sorted({int(kw["frame"]) for kw in snapshot_kwargs}):
                    file_model.clear_peaks(
                        self.session.temp_path, entry,
                        kind="matched", frame=f,
                    )

            def after(_result) -> None:
                self.pipeline_panel.append_log(
                    f"Undo batch fit: removed {len(state['ids'])} fitted "
                    f"row(s) on {entry}"
                )

            self._undoable_write_step(
                entry=entry, frame=None,
                title="Undo batch fit", write=write, after=after,
            )

        def _redo() -> None:
            def write():
                new_ids: list[int] = []
                for kw in snapshot_kwargs:
                    write_kw = {k: v for k, v in kw.items() if k != "frame"}
                    new_id = file_model.add_fitted_peak_row(
                        self.session.temp_path, entry, int(kw["frame"]),
                        **write_kw,
                    )
                    new_ids.append(int(new_id))
                return new_ids

            def after(new_ids) -> None:
                state["ids"] = new_ids
                self.pipeline_panel.append_log(
                    f"Redo batch fit: re-added {len(new_ids)} fitted "
                    f"row(s) on {entry}"
                )

            self._undoable_write_step(
                entry=entry, frame=None,
                title="Redo batch fit", write=write, after=after,
            )

        self.viewer._push_undo(_CallbackAction(_undo, _redo))

    def _run_pygidfit_for_selection(
        self, sel: SelectedPeak, entry: str, frame: int,
    ) -> tuple["manual_fit.ManualFitResult | None", str | None]:
        """Run pygidfit on ``sel`` and return ``(result, error)``.

        Silent — no QMessageBox, no logging at warning level. Used by
        both the commit path (``_build_fitted_row_2d``) and the live
        preview path (``_refresh_2d_preview``). When ``result`` is
        None, ``error`` is a short human-readable string the caller
        can surface or ignore.

        Replicates ``pygidfit.ProcessDataFromFile.process_single_frame``
        — same polar resolution (``512 × 1024``), ``img_preprocessing``
        masking, ``crit_angle`` / ``theta_fixed`` / ``clustering_*``
        values pulled from the Pipeline panel — so the preview's box
        equals what the commit would save.
        """
        from mlgidlab.manual_fit import ManualFitError, fit_one_peak

        fs = getattr(self.viewer, "_frame_source", None)
        stack = getattr(self.viewer, "_stack", None)
        if fs is None or not fs.is_open or stack is None:
            return None, (
                "The active frame isn't currently available "
                "(viewer mid-pipeline or released)."
            )
        try:
            cartesian = np.asarray(fs.get_cartesian(int(frame)))
        except Exception as exc:
            return None, f"Could not load the cartesian frame: {exc}"
        q_xy = np.asarray(stack.q_xy)
        q_z = np.asarray(stack.q_z)

        if self.session is None:
            return None, "No active session."
        geom = file_model.read_geometry_for_entry(
            self.session.temp_path, entry, frame=int(frame),
        )
        if geom is None:
            return None, (
                "Instrument metadata (wavelength / angle of incidence "
                "/ q ranges) is missing or malformed for this entry."
            )
        geom = dict(geom)
        geom.pop("q_z_axis", None)

        panel = self.pipeline_panel
        try:
            crit_angle = float(panel.fit_crit_angle.value())
            cdp = float(panel.fit_dist_peaks.value())
            cdr = float(panel.fit_dist_rings.value())
            ce = int(panel.fit_cluster_extend.value())
            tf = bool(panel.fit_theta_fixed.isChecked())
        except Exception:
            crit_angle, cdp, cdr, ce, tf = 0.0, 10.0, 10.0, 2, True

        try:
            fit_2d = fit_one_peak(
                cartesian, q_xy, q_z,
                radius=float(sel.radius),
                radius_width=float(sel.radius_width),
                angle=float(sel.angle),
                angle_width=float(sel.angle_width),
                crit_angle=crit_angle,
                theta_fixed=tf,
                clustering_distance_peaks=cdp,
                clustering_distance_rings=cdr,
                clustering_extend=ce,
                **geom,
            )
        except ManualFitError as exc:
            return None, str(exc)
        except Exception as exc:
            return None, f"Unexpected 2D fit error: {exc!r}"
        return fit_2d, None

    def _build_fitted_row_2d(
        self, sel: SelectedPeak, entry: str, frame: int,
    ) -> dict | None:
        """Run pygidfit on ``sel`` and shape the result into an
        ``add_fitted_peak_row`` kwarg dict. Surfaces a QMessageBox
        on failure — strict, no silent fall-back to the 1D path.
        Delegates the actual fit to ``_run_pygidfit_for_selection``
        so the live-preview path can reuse the same code.
        """
        fit_2d, err = self._run_pygidfit_for_selection(sel, entry, frame)
        if fit_2d is None:
            QMessageBox.warning(
                self, "Add to fitted (2D)",
                f"pygidfit could not fit this box: {err}\n\n"
                "Try widening the box, or switch to '1D fit (scipy)' "
                "mode to commit with the legacy 1D Gaussian fit.",
            )
            return None
        return {
            "radius": fit_2d.radius,
            "radius_width": fit_2d.radius_width,
            "angle": fit_2d.angle,
            "angle_width": fit_2d.angle_width,
            "amplitude": fit_2d.amplitude,
            "theta": fit_2d.theta,
            "A": fit_2d.A, "B": fit_2d.B, "C": fit_2d.C,
        }

    def _build_fitted_row_1d(
        self, sel: SelectedPeak, save_as_ring: bool,
    ) -> dict | None:
        """Build the row from the profile viewer's cached 1D scipy fits.

        Width convention matches the 2D path so the saved blue box for
        the same physical Gaussian renders identically regardless of
        which mode produced it: ``radius_width = angle_width = 2σ =
        FWHM / sqrt(2 ln 2) ≈ 0.849 × FWHM``. This is pygidfit's
        convention (see ``manual_fit.fit_one_peak``) and what the
        pipeline ``run_fitting`` writes. The only difference between
        modes is whether 2D shape coefficients (A/B/C/theta) carry
        real values — 1D zero-fills them.

        Ring storage keeps ``angle = 45°``, ``angle_width = inf`` as
        the sentinel — not a Gaussian width, so the unified
        convention doesn't apply.

        Returns ``None`` and shows a QMessageBox when the required 1D
        fit isn't available (typical cause: narrow detected box where
        scipy didn't converge). Strict — no fall-back to pygidfit.
        """
        fits = self.profile_viewer.last_fit_params()
        rfit = fits.get("radial")
        afit = fits.get("angular")
        if rfit is None:
            QMessageBox.warning(
                self, "Add to fitted (1D)",
                "No radial Gaussian fit is available. Drag the box "
                "until the pink fit curve appears in the radial "
                "profile, or switch to '2D fit (pygidfit)' mode.",
            )
            return None
        if not save_as_ring and afit is None:
            QMessageBox.warning(
                self, "Add to fitted (1D)",
                "No angular Gaussian fit is available — required for "
                "segment peaks in 1D mode. Drag the box until the "
                "pink fit curve appears in the angular profile, "
                "check 'Save fitted as ring' if this is a ring, or "
                "switch to '2D fit (pygidfit)' mode.",
            )
            return None
        fwhm_to_2sigma = 1.0 / float(np.sqrt(2.0 * np.log(2.0)))
        if save_as_ring:
            angle_to_save = 45.0
            angle_width_to_save = float("inf")
        else:
            angle_to_save = float(afit.center)
            angle_width_to_save = float(afit.fwhm) * fwhm_to_2sigma
        return {
            "radius": float(rfit.center),
            "radius_width": float(rfit.fwhm) * fwhm_to_2sigma,
            "angle": angle_to_save,
            "angle_width": angle_width_to_save,
            "amplitude": float(rfit.amplitude),
            "theta": 0.0, "A": 0.0, "B": 0.0, "C": 0.0,
        }

    def _on_delete_peak_requested(self, sel: SelectedPeak | None) -> None:
        """Confirm + delete a non-manual peak — kind-scoped, no cascade.

        Replaces the previous ``mlgidbase.delete_peak`` dispatch
        (which removed the row from detected + fitted + matched in
        one shot, leaving the user unable to delete just the fitted
        prediction without also wiping its detected source). Now:

        * ``sel.kind == "detected"`` → delete only the detected row.
          Leaves fitted / matched alone.
        * ``sel.kind == "fitted"`` → delete only the fitted row, then
          reindex ``matched_*`` ``peak_list`` so each surviving
          structure keeps its other peaks (the deleted peak is dropped
          from any structure that referenced it; structures that didn't
          are untouched). Replaces the old blunt "clear every
          ``matched_*`` on the frame" cascade.
        * ``sel.kind == "matched"`` → still routes through
          ``mlgidbase.delete_peak`` for now — matched live across
          per-solution datasets and a dedicated single-solution
          deleter isn't built yet. Leaves detected / fitted intact
          via mlgidbase's own per-kind missing-id tolerance.
        """
        if (
            sel is None or sel.kind == "manual"
            or self.session is None or self._pipe_thread is not None
        ):
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return

        if sel.kind in ("detected", "fitted"):
            return self._delete_file_peak_scoped(sel, entry)

        # matched fallthrough — keep the cascade path until a
        # dedicated single-matched-row deleter exists.
        if not pipeline.is_mlgidbase_available():
            QMessageBox.information(
                self, "Delete peak",
                "mlgidbase is not installed; cannot delete peaks.",
            )
            return
        reply = QMessageBox.question(
            self,
            "Delete matched peak",
            (
                f"Delete matched peak id={sel.peak_id} on frame {sel.frame}?\n\n"
                "It will be removed from every matched solution that "
                "references it. Detected and fitted rows are left "
                "intact. This cannot be undone."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        cmd = PipelineCommand(
            "delete_peak",
            {
                "entry": entry,
                "frame_num": int(sel.frame),
                "peak_id": int(sel.peak_id),
            },
        )
        self._on_pipeline_run(self.session.temp_path, cmd)

    def _delete_file_peak_scoped(
        self, sel: SelectedPeak, entry: str,
    ) -> None:
        """Delete one detected/fitted row — kind-scoped, undoable.

        Fitted deletes also reindex ``matched_*`` ``peak_list`` on the
        frame (positions into ``fitted_peaks`` shift when a row is
        removed), keeping the surviving structures. Delegates to the
        shared ``_delete_peaks_scoped`` so single and bulk delete share
        one undoable path.
        """
        self._delete_peaks_scoped([sel], entry)

    def _on_delete_peaks_requested(self, sels: list[SelectedPeak]) -> None:
        """Bulk-delete an all-detected OR all-fitted multi-selection.

        Triggered by Delete when >= 2 same-kind peaks are selected
        (``deletePeaksRequested``). One confirmation, one grouped,
        undoable action — see ``_delete_peaks_scoped``.
        """
        kinds = {s.kind for s in sels}
        if len(sels) < 2 or len(kinds) != 1:
            return
        if next(iter(kinds)) not in ("detected", "fitted"):
            return
        entry = self.entry_combo.currentText()
        if not entry:
            return
        self._delete_peaks_scoped(list(sels), entry)

    def _delete_peaks_scoped(
        self, sels: list[SelectedPeak], entry: str,
    ) -> None:
        """Shared, undoable delete for 1..N same-kind detected/fitted peaks.

        Confirms once, snapshots the full rows (so undo can re-create
        them verbatim — including the 2D-fit params PeakTable doesn't
        surface), deletes them in one ``_detached_silx_tree`` scope,
        reindexes ``matched_*`` ``peak_list`` on the frame for fitted
        deletes (dropping the deleted peaks from any structure that
        referenced them, keeping every structure), and pushes one
        grouped ``_CallbackAction`` so a single Ctrl+Z restores every
        row. Undo re-adds the fitted rows with fresh ids appended at the
        end; it does **not** re-add them to the matched structures they
        were in (the structures already reference the surviving peaks
        correctly — re-run Matching if a peak's membership matters).
        """
        if self._is_busy():
            return
        if not sels or not entry:
            return
        kind = sels[0].kind
        if kind not in ("detected", "fitted"):
            return
        frame = int(self.viewer.current_frame)
        on_frame = [
            s for s in sels if s.kind == kind and int(s.frame) == frame
        ]
        if not on_frame:
            return
        ids = [int(s.peak_id) for s in on_frame]
        n = len(ids)
        kind_human = "detected" if kind == "detected" else "fitted"
        if kind == "detected":
            siblings_msg = (
                "Fitted and matched rows stay intact."
            )
        else:
            siblings_msg = (
                "Detected rows stay intact. Matched structures that "
                "include the deleted peak(s) lose just those peaks; "
                "structures that don't reference them are left unchanged."
            )
        target = (
            f"id={ids[0]} on frame {frame}" if n == 1
            else f"{n} {kind_human} peaks on frame {frame}"
        )
        reply = QMessageBox.question(
            self,
            f"Delete {kind_human} peak" + ("s" if n > 1 else ""),
            f"Delete {target}?\n\n{siblings_msg} This can be undone "
            "with Ctrl+Z.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        removed = 0
        snapshot: list[dict] = []
        matched_changed = 0
        with self._detached_silx_tree():
            try:
                # Snapshot full rows (incl. theta/A/B/C for fitted) so
                # undo can re-create them. Read inside the detached
                # scope so no handle contends with silx.
                snapshot = file_model.read_peak_rows(
                    self.session.temp_path, entry, frame, kind, ids,
                )
                # matched_* peak_list stores positions into fitted_peaks,
                # not ids, and deleting compacts the array — capture the
                # rows' positions BEFORE deleting so matched can be
                # reindexed afterwards.
                deleted_positions = (
                    file_model.fitted_positions_for_ids(
                        self.session.temp_path, entry, frame, ids,
                    )
                    if kind == "fitted" else []
                )
                for peak_id in ids:
                    removed += file_model.delete_peak_row(
                        self.session.temp_path, entry,
                        frame=frame, kind=kind, peak_id=peak_id,
                    )
                if kind == "fitted" and removed > 0:
                    # Drop the deleted peaks from any matched structure
                    # that referenced them and shift the surviving
                    # indices down; structures are kept (the old code
                    # wiped every matched_* on the frame instead).
                    matched_changed = file_model.remap_matched_peak_lists(
                        self.session.temp_path, entry, frame,
                        deleted_positions,
                    )
            except Exception as exc:
                QMessageBox.critical(
                    self, f"Delete {kind_human} peak", str(exc),
                )
                return
        if removed == 0:
            self.pipeline_panel.append_log(
                f"Delete {kind_human}: no matching rows on "
                f"{entry}/frame{frame:05d} (already gone?)"
            )
            return

        self.session.mark_dirty()
        self._update_title()
        # Drop only the deleted ids' pending geom edits (they would key
        # off rows that no longer exist); prior undoable ops stay on the
        # stack so a run of deletes remains fully undoable / redoable.
        self.viewer.prune_file_geom_history(frame, kind, ids)
        self.viewer.clear_selection()
        self._load_entry_into_viewer(entry, preserve_view=True)
        self._push_delete_undo(
            entry=entry, frame=frame, kind=kind, snapshot=snapshot,
            cascade_matched=(kind == "fitted"),
        )
        msg = (
            f"Deleted {removed} {kind_human} peak(s) on "
            f"{entry}/frame{frame:05d}"
            + (
                f" ({matched_changed} matched structure(s) updated)"
                if matched_changed else ""
            )
        )
        self.statusBar().showMessage(msg, 4000)
        self.pipeline_panel.append_log(msg)

    def _readd_peak_row(self, entry: str, frame: int, kind: str, d: dict) -> int:
        """Re-create one detected/fitted row from a ``read_peak_rows``
        snapshot dict. Returns the new (freshly-assigned) id."""
        if kind == "detected":
            return file_model.add_detected_peak_row(
                self.session.temp_path, entry, int(frame),
                radius=d["radius"], angle=d["angle"],
                radius_width=d["radius_width"], angle_width=d["angle_width"],
                score=d.get("score", 0.0), is_ring=d.get("is_ring", False),
            )
        return file_model.add_fitted_peak_row(
            self.session.temp_path, entry, int(frame),
            angle=d["angle"], angle_width=d["angle_width"],
            radius=d["radius"], radius_width=d["radius_width"],
            amplitude=d.get("amplitude", 0.0), is_ring=d.get("is_ring", False),
            theta=d.get("theta", 0.0), score=d.get("score", 0.0),
            A=d.get("A", 0.0), B=d.get("B", 0.0), C=d.get("C", 0.0),
            is_cut_qz=d.get("is_cut_qz", False),
            is_cut_qxy=d.get("is_cut_qxy", False),
            visibility=d.get("visibility", 0),
        )

    def _push_delete_undo(
        self, *, entry: str, frame: int, kind: str,
        snapshot: list[dict], cascade_matched: bool,
    ) -> None:
        """Undo / redo for a (single or bulk) detected/fitted delete.

        The delete already happened, so ``undo`` re-adds the rows (fresh
        ids) and ``redo`` deletes them again. ``state["ids"]`` rotates:
        populated by ``undo`` with the re-assigned ids so a following
        ``redo`` deletes the right rows. For fitted, ``redo`` re-applies
        the ``matched_*`` reindex (recomputing the deleted rows'
        positions from the current ids first, since the re-added rows
        sit at fresh positions); ``undo`` leaves ``matched_*`` as-is —
        see ``_delete_peaks_scoped``.
        """
        snap = [dict(d) for d in snapshot]
        state: dict[str, list[int]] = {"ids": []}

        def _undo() -> None:
            def write():
                new_ids: list[int] = []
                for d in snap:
                    new_ids.append(
                        int(self._readd_peak_row(entry, frame, kind, d))
                    )
                return new_ids

            def after(new_ids) -> None:
                state["ids"] = new_ids
                self._add_detected_to_selection(
                    entry, int(frame), new_ids, kind=kind,
                )
                self.pipeline_panel.append_log(
                    f"Undo delete: restored {len(new_ids)} {kind} "
                    f"peak(s) on {entry}/frame{int(frame):05d}"
                )

            self._undoable_write_step(
                entry=entry, frame=frame,
                title="Undo delete", write=write, after=after,
            )

        def _redo() -> None:
            def write():
                deleted_positions = (
                    file_model.fitted_positions_for_ids(
                        self.session.temp_path, entry, int(frame),
                        [int(p) for p in state["ids"]],
                    )
                    if cascade_matched else []
                )
                for peak_id in state["ids"]:
                    file_model.delete_peak_row(
                        self.session.temp_path, entry,
                        frame=int(frame), kind=kind,
                        peak_id=int(peak_id),
                    )
                if cascade_matched and state["ids"]:
                    file_model.remap_matched_peak_lists(
                        self.session.temp_path, entry, int(frame),
                        deleted_positions,
                    )

            def after(_result) -> None:
                self._drop_detected_from_selection(
                    int(frame), state["ids"], kind=kind,
                )
                self.pipeline_panel.append_log(
                    f"Redo delete: removed {len(state['ids'])} {kind} "
                    f"peak(s) on {entry}/frame{int(frame):05d}"
                )

            self._undoable_write_step(
                entry=entry, frame=frame,
                title="Redo delete", write=write, after=after,
            )

        self.viewer._push_undo(_CallbackAction(_undo, _redo))

    # -- UI state --

    def _refresh_peaks_table(self) -> None:
        """Repopulate the Peaks dock from the viewer's current state.

        Called on entry-load completion, post-pipeline reload (via
        ``_load_entry_into_viewer``), and on every frame change (via
        ``_refresh_peaks_table_on_frame``). Empties all three tabs
        when no session is active.
        """
        if self.session is None or self.session.kind != "nexus":
            self.peaks_table_panel.clear()
            return
        frame = self.viewer.current_frame
        peaks_for_frame = self.viewer._frame_peaks.get(frame) or {}
        matched = self.viewer.matched_structures(frame)
        self.peaks_table_panel.set_frame_peaks(
            frame,
            peaks_for_frame.get("detected"),
            peaks_for_frame.get("fitted"),
            matched,
        )

    def _refresh_peaks_table_on_frame(self, _frame: int) -> None:
        """Slot for ``viewer.frameChanged`` — drops the unused frame
        index since ``_refresh_peaks_table`` re-reads it from the
        viewer."""
        self._refresh_peaks_table()

    def _on_peak_selected_from_table(self, sel: SelectedPeak | None) -> None:
        """Route a table-row click back into the image viewer.

        The panel builds the SelectedPeak with ``frame=0`` since it
        has no viewer context; stamp the real current_frame here
        before handing it to ``_set_selected``. The viewer's
        equality guard short-circuits the round-trip if the
        selection didn't actually change.
        """
        if sel is None:
            return
        sel.frame = self.viewer.current_frame
        self.viewer._set_selected(sel)
