"""Wiring for the Structure tab: file-browser selection in, panel out.

Plain mixin over ``MainWindow`` — no ``__init__``, no Signals; all state
lives on the combined class, like every other ``main_window_*`` mixin.

Two rules shape this file.

**The dock stays the navigator.** The Structure tab has no tree of its
own; it renders whatever the File-browser dock has selected. That is why
the wiring hangs off ``_on_tree_selection_changed`` and why a click is
*deferred* while the tab is hidden, exactly as the Data tab's node is
(``_set_or_defer_data_node``): a single click on the Image tab must not
pay for work the user cannot see.

**Reads go through one accessor.** ``_structure_read`` yields the
already-open edit handle when there is one and a short-lived ``r`` open
otherwise, so no code path here has to know which of the two is in play.
A read-only open alongside silx's own read handles is safe; the
write-handle rules live in ``mlgidlab.h5_edit``.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import h5py
from PySide6.QtWidgets import QApplication, QMessageBox

from mlgidlab import h5_edit
from mlgidlab.browser_widgets import _ImageFileNode
from mlgidlab.h5_edit import MISSING, EditError, EditHandle
from mlgidlab.h5_edit_ops import EditHistory, RenameAttrOp, SetAttrOp, WriteCellsOp

import logging

logger = logging.getLogger(__name__)


class StructureMixin:
    # -- the write handle --------------------------------------------------

    def _structure_history_for(self, file_path) -> EditHistory:
        """The undo history for one file, created on first use.

        Per file rather than per window: two files open side by side have
        unrelated edits, and switching between them must not wipe either
        list. Dropped when the session closes.
        """
        key = str(file_path)
        history = self._structure_histories.get(key)
        if history is None:
            history = EditHistory()
            self._structure_histories[key] = history
        return history

    def _acquire_edit_handle(self, file_path) -> EditHandle:
        """Open — or reuse — the ``r+`` handle for ``file_path``.

        The acquire happens inside one detached window because HDF5
        refuses ``r+`` while any read handle is open, and silx, the
        viewer's FrameSource and the prefetch worker all hold one. That
        costs a single tree rebuild per editing session; every edit
        afterwards writes through the already-open handle with no detach
        at all, which is what makes editing feel immediate.

        The handle is released again by ``_detach_silx_tree`` — the choke
        point every pipeline run, save-as, reload and session close
        already passes through — and re-acquired here on the next edit.
        """
        path = Path(file_path)
        handle = self._h5_edit_handle
        if handle is not None and handle.path != path:
            handle.release()
            handle = None
        if handle is None:
            handle = EditHandle(path)
            self._h5_edit_handle = handle
        if not handle.is_open:
            with self._detached_silx_tree():
                handle.acquire()
        return handle

    def _release_edit_handle(self) -> None:
        """Drop the write handle. Called from ``_detach_silx_tree``."""
        handle = getattr(self, "_h5_edit_handle", None)
        if handle is not None:
            handle.release()

    # -- reading -----------------------------------------------------------

    @contextmanager
    def _structure_read(self, file_path: Path | str):
        """Yield an ``h5py.File`` for ``file_path``, however it is open.

        Reuses the editor's write handle when it already covers this file
        — opening a second handle would be pointless and, for ``r+``,
        impossible while another read handle is out. Otherwise opens
        ``r`` for the duration of the block.

        Yields ``None`` rather than raising when the file cannot be
        opened at all: every caller is a selection handler, and a bad
        click must not wedge the browser.
        """
        path = Path(file_path)
        handle = getattr(self, "_h5_edit_handle", None)
        if handle is not None and handle.is_open and handle.path == path:
            yield handle.file
            return
        try:
            with h5py.File(path, "r") as f:
                yield f
        except OSError:
            logger.debug("structure read failed for %s", path, exc_info=True)
            yield None

    # -- selection ---------------------------------------------------------

    def _set_or_defer_structure_node(self, node) -> None:
        """Render ``node`` if the Structure tab is showing, else remember it.

        The deferral is the same bargain the Data tab makes: browsing on
        the Image tab stays free, and the cost is paid once, on the
        switch to a tab that will actually show the result.
        """
        if self.tabs.currentWidget() is self.structure_panel:
            self._render_structure_node(node)
        else:
            self._pending_structure_node = node

    def _render_structure_node(self, node) -> None:
        """Fill the Structure panel from a file-browser node.

        Guarded end to end: a stale silx node, a file that vanished
        under us, or a broken link must leave the panel with an
        explanation rather than propagate out of a tree click.
        """
        self._pending_structure_node = None
        panel = self.structure_panel
        if isinstance(node, _ImageFileNode):
            panel.clear(
                "This is a standalone image file, not an HDF5 file — there "
                "is no structure to show. Convert it first."
            )
            panel.set_note("")
            panel.set_issues([])
            return

        file_path = self._node_filename(node)
        # ``_node_h5_path`` (FramesMixin) returns the in-file path with the
        # slashes stripped, because the raw-entry matcher it was written
        # for compares against pygid's slash-less dataset paths. Normalize
        # it back to an absolute HDF5 path — "" is the file root.
        raw_path = self._node_h5_path(node)
        h5_path = None if raw_path is None else h5_edit.normalize_path(raw_path)
        if file_path is None or h5_path is None:
            panel.clear()
            self._structure_target = None
            return

        # ``_session_for_node`` resolves the path before matching, which
        # is what a symlinked or relative temp path needs; the raw batch
        # match is O(1) through its cached string set.
        session = self._session_for_node(node)
        is_raw = session is not None and session.kind == "raw"

        with self._structure_read(file_path) as f:
            if f is None:
                panel.clear(f"{Path(file_path).name} could not be read.")
                return
            try:
                info = h5_edit.node_info(f, h5_path)
                # A node that is not there has no attributes to read and
                # no value to preview; asking would raise. The panel
                # still shows the path and says it is missing, which is
                # what a stale tree row needs to look like.
                attrs = (
                    {} if info.kind == "missing"
                    else h5_edit.read_attrs(f, h5_path)
                )
                preview = (
                    "" if info.kind == "missing"
                    else h5_edit.value_preview(f, h5_path)
                )
            except Exception as exc:
                logger.debug("structure render failed", exc_info=True)
                panel.clear(f"{h5_path} could not be read — {exc}")
                return
            panel.set_node(
                info, attrs, preview, file_label=Path(file_path).name
            )
            self._refresh_structure_issues(file_path, f)

        # Only a NeXus session has a working copy to edit; a raw input is
        # the user's original data and stays read-only, and a file with no
        # session behind it is not ours to write to at all.
        editable = (
            session is not None
            and session.kind == "nexus"
            and info.kind != "missing"
        )
        self._structure_target = (
            (Path(file_path), h5_path) if editable else None
        )
        panel.set_editable(editable)
        panel.set_note(
            "Raw detector files are read-only in mlgidLAB — convert one to "
            "NeXus to edit it." if is_raw else ""
        )
        panel.set_changes(self._structure_history_for(file_path).entries())

    def _refresh_structure_issues(self, file_path, f) -> None:
        """Re-run the layout check when the file changed, or on demand.

        Cached per file: the check is cheap (it never follows an external
        link) but it is pointless to repeat on every click within one
        file, and the Re-check button exists for when the user wants it
        run again.
        """
        key = str(file_path)
        if getattr(self, "_structure_checked_file", None) == key:
            return
        try:
            issues = h5_edit.validate(f)
        except Exception:
            logger.debug("structure validation failed", exc_info=True)
            issues = []
        self._structure_checked_file = key
        self.structure_panel.set_issues(issues)

    # -- actions -----------------------------------------------------------

    def _on_structure_recheck(self) -> None:
        """Re-check button: drop the cache and re-render the selection."""
        self._structure_checked_file = None
        node = self._pending_structure_node
        if node is None:
            nodes = self._safe_selected_h5_nodes()
            node = nodes[0] if nodes else None
        if node is not None:
            self._render_structure_node(node)

    def _clear_structure_panel(self) -> None:
        """Empty the panel — called when the session it described goes away.

        The write handle is already released at this point (the caller is
        inside a detached scope), so all that is left is dropping the
        edit history for a file that no longer exists.
        """
        self._pending_structure_node = None
        self._structure_checked_file = None
        self._structure_target = None
        handle = getattr(self, "_h5_edit_handle", None)
        if handle is not None:
            self._structure_histories.pop(str(handle.path), None)
            self._h5_edit_handle = None
        if hasattr(self, "structure_panel"):
            self.structure_panel.clear()
            self.structure_panel.set_note("")
            self.structure_panel.set_issues([])

    # -- editing -----------------------------------------------------------

    def _structure_edit_target(self):
        """``(handle, h5_path)`` for the node on show, or None.

        None means the panel is showing something that cannot be written
        to — a raw input, a file with no session, a node that vanished —
        and every edit handler returns quietly on it. The panel's own
        controls are already disabled in those cases; this is the guard
        for a shortcut or a stale signal arriving anyway.
        """
        target = getattr(self, "_structure_target", None)
        if target is None:
            return None
        file_path, h5_path = target
        try:
            return self._acquire_edit_handle(file_path), h5_path
        except EditError as exc:
            QMessageBox.critical(self, "Edit file", str(exc))
            return None

    def _confirm_protected(self, action: str, what: str, reason: str) -> bool:
        """Ask before a destructive edit to something the app depends on.

        The dialog names what breaks rather than warning in the abstract,
        because "are you sure?" teaches nobody anything. Nothing is
        forbidden — this is the whole of the guardrail.
        """
        answer = QMessageBox.warning(
            self,
            f"{action} {what}?",
            f"{what} is {reason}.\n\n"
            f"{action} it and this file may stop opening in the viewer or "
            f"running through the pipeline.\n\n"
            "The change lands in the working copy only — the file on disk "
            "is untouched until you save, and Ctrl+Z reverses it.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _commit_structure_edit(self, handle, op, path: str) -> None:
        """Record an applied edit and bring the rest of the window in step.

        Flush first: ``File > Save`` copies the working copy with a plain
        ``shutil.copy2`` and never detaches, so the bytes on disk have to
        be consistent after every single edit, not just at teardown.
        """
        handle.flush()
        self._structure_history_for(handle.path).push(op)
        if self.session is not None:
            self.session.mark_dirty()
            self._update_title()
        # A layout check is cheap and an edit may have fixed or broken
        # exactly what it reports, so it is re-run rather than reasoned
        # about.
        self._structure_checked_file = None
        self._refresh_viewer_after_structure_edit(path)
        self._rerender_structure()

    def _refresh_viewer_after_structure_edit(self, path: str) -> None:
        """Reload the viewer when an edit touched what it is displaying.

        Scoped to the image stack, the q axes and the peak tables of the
        *current* entry — everything the viewer actually reads. An edit
        to a sample name, or to a title on the entry group itself, leaves
        the image alone, as it should.
        """
        if not h5_edit.affects_viewer(path):
            return
        entry = h5_edit.entry_of(path)
        if entry is None or entry != self.entry_combo.currentText():
            return
        try:
            self._load_entry_into_viewer(entry, preserve_view=True)
        except Exception:
            logger.debug("viewer reload after a structure edit failed",
                         exc_info=True)

    def _rerender_structure(self) -> None:
        """Re-read the node on show, so the panel matches the file.

        Every handler ends here, success or failure: the panel holds no
        authoritative state, so reconciling it with the file is both the
        refresh after a write and the rollback after a rejected one.
        """
        node = self._pending_structure_node
        if node is None:
            nodes = self._safe_selected_h5_nodes()
            node = nodes[0] if nodes else None
        if node is not None:
            self._render_structure_node(node)

    def _structure_failed(self, title: str, exc: Exception) -> None:
        QMessageBox.warning(self, title, str(exc))
        self._rerender_structure()

    # -- attribute intents -------------------------------------------------

    def _on_structure_attribute_edited(self, name: str, text: str) -> None:
        """A value cell was typed into: keep the attribute's own type."""
        target = self._structure_edit_target()
        if target is None:
            return
        handle, path = target
        reason = h5_edit.attr_protection_reason(path, name)
        try:
            current = h5_edit.read_attrs(handle.file, path).get(name, MISSING)
            value = h5_edit.parse_attr_value(text, current)
        except EditError as exc:
            self._structure_failed("Attribute", exc)
            return
        if reason and not self._confirm_protected("Change", name, reason):
            self._rerender_structure()
            return
        try:
            old = h5_edit.set_attr(handle.file, path, name, value)
        except EditError as exc:
            self._structure_failed("Attribute", exc)
            return
        self._commit_structure_edit(
            handle, SetAttrOp(path, name, old, value), path)

    def _on_structure_attribute_renamed(self, old_name: str, new_name: str) -> None:
        target = self._structure_edit_target()
        if target is None:
            return
        handle, path = target
        reason = h5_edit.attr_protection_reason(path, old_name)
        if reason and not self._confirm_protected("Rename", old_name, reason):
            self._rerender_structure()
            return
        try:
            h5_edit.rename_attr(handle.file, path, old_name, new_name)
        except EditError as exc:
            self._structure_failed("Rename attribute", exc)
            return
        self._commit_structure_edit(
            handle, RenameAttrOp(path, old_name, new_name), path)

    def _on_structure_attribute_added(
        self, name: str, type_name: str, text: str
    ) -> None:
        target = self._structure_edit_target()
        if target is None:
            return
        handle, path = target
        try:
            value = h5_edit.parse_scalar(text, type_name)
            old = h5_edit.set_attr(handle.file, path, name, value)
        except EditError as exc:
            self._structure_failed("New attribute", exc)
            return
        self._commit_structure_edit(
            handle, SetAttrOp(path, name, old, value), path)

    def _on_structure_attribute_removed(self, name: str) -> None:
        target = self._structure_edit_target()
        if target is None:
            return
        handle, path = target
        reason = h5_edit.attr_protection_reason(path, name)
        if reason and not self._confirm_protected("Remove", name, reason):
            return
        try:
            old = h5_edit.delete_attr(handle.file, path, name)
        except EditError as exc:
            self._structure_failed("Remove attribute", exc)
            return
        self._commit_structure_edit(
            handle, SetAttrOp(path, name, old, MISSING), path)

    # -- value intents -----------------------------------------------------

    def _on_structure_value_edited(self, text: str) -> None:
        """Write a one-cell dataset from the Value field."""
        target = self._structure_edit_target()
        if target is None:
            return
        handle, path = target
        reason = h5_edit.protection_reason(path)
        if reason and not self._confirm_protected("Change", path, reason):
            self._rerender_structure()
            return
        try:
            info = h5_edit.node_info(handle.file, path)
            if info.kind != "dataset" or info.n_cells != 1 or info.is_compound:
                raise EditError(
                    "Only a single-cell dataset can be set from this field.")
            index = () if not info.shape else (0,) * len(info.shape)
            value = (
                text if info.dtype.startswith("object")
                else h5_edit.parse_scalar(text, info.dtype)
            )
            before = h5_edit.write_cells(handle.file, path, {index: value})
        except EditError as exc:
            self._structure_failed("Set value", exc)
            return
        self._commit_structure_edit(
            handle, WriteCellsOp(path, before, {index: value}), path)

    # -- undo / redo -------------------------------------------------------

    def _structure_owns_undo(self) -> bool:
        """Whether Ctrl+Z belongs to the editor right now.

        Only while the Structure tab is the one in front, and only when
        it has something to reverse. The image viewer's own history is
        untouched by this and keeps Ctrl+Z everywhere else.
        """
        if not hasattr(self, "structure_panel"):
            return False
        if self.tabs.currentWidget() is not self.structure_panel:
            return False
        handle = getattr(self, "_h5_edit_handle", None)
        if handle is None:
            return False
        history = self._structure_histories.get(str(handle.path))
        # Either direction counts: after undoing the only edit the undo
        # stack is empty but Ctrl+Y still belongs here, not to the viewer.
        return history is not None and (history.can_undo or history.can_redo)

    def _structure_undo(self) -> bool:
        """Reverse the last editor change. True when it handled the key."""
        return self._structure_step(redo=False)

    def _structure_redo(self) -> bool:
        return self._structure_step(redo=True)

    def _structure_step(self, *, redo: bool) -> bool:
        target = self._structure_edit_target()
        if target is None:
            return False
        handle, _ = target
        history = self._structure_history_for(handle.path)
        if redo and not history.can_redo:
            return False
        if not redo and not history.can_undo:
            return False
        try:
            op = history.redo(handle.file) if redo else history.undo(handle.file)
        except EditError as exc:
            QMessageBox.warning(self, "Undo", str(exc))
            return True
        if op is None:
            return False
        handle.flush()
        if self.session is not None:
            self.session.mark_dirty()
            self._update_title()
        self._structure_checked_file = None
        self._refresh_viewer_after_structure_edit(getattr(op, "path", ""))
        self._rerender_structure()
        return True

    # -- changes list ------------------------------------------------------

    def _on_structure_copy_changes(self) -> None:
        handle = getattr(self, "_h5_edit_handle", None)
        if handle is None:
            return
        text = self._structure_history_for(handle.path).as_text()
        if text:
            QApplication.clipboard().setText(text)
            self.statusBar().showMessage("Changes copied to the clipboard", 3000)
