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
import numpy as np
from PySide6.QtCore import QItemSelectionModel, QModelIndex, Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QInputDialog,
    QMessageBox,
)

from mlgidlab import h5_edit, nexus_schema
from silx.gui.hdf5 import Hdf5TreeModel

from mlgidlab.browser_widgets import _ImageFileNode
from mlgidlab.h5_edit import MISSING, EditError, EditHandle
from mlgidlab.array_edit_dialog import ArrayEditDialog
from mlgidlab.h5_edit_ops import (
    CreateNodeOp,
    DeleteLinkOp,
    DeleteNodeOp,
    EditHistory,
    MoveNodeOp,
    RenameAttrOp,
    ReplaceDataOp,
    RetargetLinkOp,
    SetAttrOp,
    WriteCellsOp,
)
from mlgidlab.structure_clipboard import StructureClip, free_name
from mlgidlab.structure_dialogs import (
    NewDatasetDialog,
    NewGroupDialog,
    NewLinkDialog,
    PickNodeDialog,
    parse_shape,
)

import logging

logger = logging.getLogger(__name__)

#: Past this many changed cells a grid edit is written as one array
#: replace rather than a per-cell diff. Hand editing never reaches it;
#: a find-and-replace over a whole frame would, and doing that cell by
#: cell would freeze the window.
MAX_SPARSE_CELLS = 4096


def _apply_template(f, path: str, template) -> None:
    """Create a template's fields inside a freshly made group.

    Scalar datasets with the template's default value, plus the units
    attribute where the template names one. Deliberately shallow: a
    template pre-creates what a group nearly always has, and anything
    optional is better added deliberately than deleted from a bloated
    skeleton.
    """
    for spec in template.fields:
        if spec.kind == "str":
            h5_edit.create_dataset(
                f, path, spec.name, dtype="str", data=str(spec.value),
                attrs={"units": spec.units} if spec.units else None,
            )
        else:
            dtype = "float64" if spec.kind == "float" else "int64"
            h5_edit.create_dataset(
                f, path, spec.name, dtype=dtype, shape=(), fill=spec.value,
                attrs={"units": spec.units} if spec.units else None,
            )


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
            # The detach rebuilds the browser, which would otherwise
            # collapse the tree the user is working in — on the very
            # first edit of a session, before anything has gone wrong.
            state = self._capture_tree_state()
            self._detach_silx_tree()
            try:
                handle.acquire()
            finally:
                self._reattach_tree_safely()
            self._restore_tree_state(state)
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

    def _render_structure_node(self, node, *, follow: bool = False) -> None:
        """Fill the Structure panel from a file-browser node.

        ``follow`` resolves an external link instead of describing it —
        set only by the Follow button, never by a selection change, so
        browsing a master of external links stays free.

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
                info = h5_edit.node_info(f, h5_path, follow=follow)
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

        Two ways in, because there are two ways to edit. The Structure
        tab being in front is the obvious one. The File browser having
        focus is the other: every structural edit starts from that dock's
        context menu, and after one the focus is back on the tree — so
        without this, creating a group and pressing Ctrl+Z would hand the
        key to the image viewer, which had nothing to do with it.

        The viewer's own history is untouched and keeps Ctrl+Z
        everywhere else.
        """
        if not hasattr(self, "structure_panel"):
            return False
        in_editor = self.tabs.currentWidget() is self.structure_panel
        in_browser = hasattr(self, "tree") and self.tree.hasFocus()
        if not (in_editor or in_browser):
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
        """Walk the editor's history one step.

        Keyed on the file being edited, NOT on the node the panel happens
        to be showing. Those came apart the moment structural edits
        landed: renaming or deleting the selected node leaves the panel
        pointing at a path that no longer exists, which made the panel
        non-editable, which silently took Ctrl+Z away — exactly when the
        user most wants it.
        """
        handle = getattr(self, "_h5_edit_handle", None)
        if handle is None:
            return False
        if not handle.is_open:
            try:
                handle = self._acquire_edit_handle(handle.path)
            except EditError as exc:
                QMessageBox.warning(self, "Undo", str(exc))
                return False
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
        # Reversing a structural edit changes the tree's shape as much as
        # making it did, so the browser has to be rebuilt here too. An
        # attribute or value step leaves the shape alone and skips it.
        if isinstance(op, (CreateNodeOp, DeleteNodeOp, MoveNodeOp)):
            self._rebuild_tree_preserving(self._capture_tree_state(), handle)
            self._repopulate_entries_after_structure_edit()
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

    # -- the file-browser context menu -------------------------------------

    def _install_structure_context_menu(self) -> None:
        """Hang the edit actions off the File browser's context menu.

        silx offers ``addContextMenuCallback`` for exactly this, so the
        dock's own context-menu policy, model and click handling are
        untouched — the tree gains a menu and nothing else. The callback
        is stored on ``self`` because silx keeps callbacks as safe refs.
        """
        self._structure_menu_callback = self._structure_context_actions
        self.tree.addContextMenuCallback(self._structure_menu_callback)

    def _structure_context_actions(self, event) -> None:
        """Fill the context menu for the hovered node.

        Everything is built against the hovered node rather than the
        selection, which is what a right-click means: silx has already
        resolved the node under the cursor and hands it over.
        """
        try:
            node = event.hoveredObject()
            menu = event.menu()
        except Exception:
            logger.debug("context menu event unusable", exc_info=True)
            return
        file_path = self._node_filename(node)
        raw_path = self._node_h5_path(node)
        if file_path is None or raw_path is None:
            return
        h5_path = h5_edit.normalize_path(raw_path)
        session = self._session_for_node(node)
        if session is None or session.kind != "nexus":
            # Raw inputs and files with no session behind them stay
            # read-only; adding greyed-out actions would only imply
            # otherwise.
            return

        menu.addSeparator()
        is_group = self._structure_node_is_group(file_path, h5_path)
        target = (Path(file_path), h5_path)

        if is_group:
            menu.addAction(
                "New group…",
                lambda: self._on_structure_new_group(target),
            )
            menu.addAction(
                "New field…",
                lambda: self._on_structure_new_dataset(target),
            )
            menu.addAction(
                "New link…",
                lambda: self._on_structure_new_link(target),
            )
        elif self._structure_link_kind(file_path, h5_path) in ("soft", "external"):
            menu.addAction(
                "Retarget link…",
                lambda: self._on_structure_retarget_link(target),
            )
        menu.addSeparator()
        menu.addAction(
            "Copy", lambda: self._on_structure_copy(target)
        ).setEnabled(h5_path != "/")
        menu.addAction(
            "Cut", lambda: self._on_structure_copy(target, cut=True)
        ).setEnabled(h5_path != "/")
        paste = menu.addAction(
            "Paste", lambda: self._on_structure_paste(target))
        paste.setEnabled(getattr(self, "_structure_clip", None) is not None)
        menu.addAction(
            "Paste from file…",
            lambda: self._on_structure_paste_from_file(target),
        )
        menu.addSeparator()
        menu.addAction(
            "Rename…", lambda: self._on_structure_rename(target)
        ).setEnabled(h5_path != "/")
        menu.addAction(
            "Delete", lambda: self._on_structure_delete(target)
        ).setEnabled(h5_path != "/")

    def _structure_link_kind(self, file_path, h5_path: str) -> str:
        """The link kind at a path — ``hard`` when it is a plain object.

        Reads the link, never the target, so asking the question about a
        row in a master file opens nothing.
        """
        with self._structure_read(file_path) as f:
            if f is None:
                return "hard"
            try:
                link = h5_edit.link_info(f, h5_path)
            except Exception:
                logger.debug("link kind probe failed", exc_info=True)
                return "hard"
            return link.kind if link is not None else "hard"

    def _structure_node_is_group(self, file_path, h5_path: str) -> bool:
        """Whether a node can hold children, without resolving a link."""
        with self._structure_read(file_path) as f:
            if f is None:
                return False
            try:
                return h5_edit.node_info(f, h5_path).kind == "group"
            except Exception:
                logger.debug("node kind probe failed", exc_info=True)
                return False

    # -- structure intents -------------------------------------------------

    def _structure_handle_for(self, target):
        """``(handle, h5_path)`` for an explicit target, or None."""
        file_path, h5_path = target
        try:
            return self._acquire_edit_handle(file_path), h5_path
        except EditError as exc:
            QMessageBox.critical(self, "Edit file", str(exc))
            return None

    def _on_structure_new_group(self, target) -> None:
        opened = self._structure_handle_for(target)
        if opened is None:
            return
        handle, parent_path = opened
        dialog = NewGroupDialog(self, parent_path=parent_path)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name, nx_class, template_key = dialog.result_values()
        if not name:
            return

        def create(f):
            path = h5_edit.create_group(
                f, parent_path, name, nx_class=nx_class or None)
            if template_key is not None:
                _apply_template(f, path, nexus_schema.TEMPLATES[template_key])
            return path

        try:
            new_path = create(handle.file)
        except EditError as exc:
            self._structure_failed("New group", exc)
            return
        self._commit_structure_change(
            handle,
            CreateNodeOp(new_path, nx_class or "group", recreate=create),
            new_path,
        )

    def _on_structure_new_dataset(self, target) -> None:
        opened = self._structure_handle_for(target)
        if opened is None:
            return
        handle, parent_path = opened
        dialog = NewDatasetDialog(self, parent_path=parent_path)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name, dtype, shape_text, value_text, units, resizable = dialog.result_values()
        if not name:
            return

        def create(f):
            shape = parse_shape(shape_text)
            attrs = {"units": units} if units else None
            if dtype == "str":
                return h5_edit.create_dataset(
                    f, parent_path, name, dtype="str",
                    data=value_text if not shape else None,
                    shape=None if not shape else shape,
                    attrs=attrs, resizable=resizable,
                )
            fill = h5_edit.parse_scalar(value_text or "0", dtype)
            return h5_edit.create_dataset(
                f, parent_path, name, dtype=dtype, shape=shape, fill=fill,
                attrs=attrs, resizable=resizable,
            )

        try:
            new_path = create(handle.file)
        except (EditError, ValueError) as exc:
            self._structure_failed("New field", EditError(str(exc)))
            return
        self._commit_structure_change(
            handle, CreateNodeOp(new_path, dtype, recreate=create), new_path)

    def _on_structure_rename(self, target) -> None:
        opened = self._structure_handle_for(target)
        if opened is None:
            return
        handle, path = opened
        parent_path, old_name = h5_edit.split_path(path)
        reason = h5_edit.protection_reason(path)
        if reason and not self._confirm_protected("Rename", path, reason):
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename", f"New name for {old_name}:", text=old_name)
        if not ok or not new_name.strip() or new_name.strip() == old_name:
            return
        try:
            new_path = h5_edit.rename_node(
                handle.file, path, new_name.strip())
        except EditError as exc:
            self._structure_failed("Rename", exc)
            return
        self._commit_structure_change(
            handle, MoveNodeOp(path, new_path), new_path)

    def _on_structure_delete(self, target) -> None:
        """Delete a node, snapshotting it first when it is small enough.

        The size probe walks hard links only and stops at the cap, so
        asking "can this be undone?" never costs a walk of a 226-entry
        master.
        """
        opened = self._structure_handle_for(target)
        if opened is None:
            return
        handle, path = opened
        reason = h5_edit.protection_reason(path)
        if reason and not self._confirm_protected("Delete", path, reason):
            return
        # A soft or external link is three strings, not a payload.
        # Snapshotting one would mean copying its TARGET — for an
        # external link, opening the very file the link exists to keep
        # closed, and for a dangling one, impossible. Remember the link
        # instead and write it back on undo.
        link = h5_edit.link_info(handle.file, path)
        if link is not None and link.kind in ("soft", "external"):
            try:
                h5_edit.delete_node(handle.file, path)
            except EditError as exc:
                self._structure_failed("Delete", exc)
                return
            self._commit_structure_change(
                handle,
                DeleteLinkOp(path, link.kind, link.target, link.filename),
                path,
            )
            return

        try:
            size = h5_edit.node_nbytes(
                handle.file, path, cap=h5_edit.UNDO_SNAPSHOT_LIMIT)
            info = h5_edit.node_info(handle.file, path)
        except EditError as exc:
            self._structure_failed("Delete", exc)
            return

        snapshot = None
        if size <= h5_edit.UNDO_SNAPSHOT_LIMIT:
            try:
                snapshot = h5_edit.snapshot_node(handle.file, path)
            except EditError:
                logger.debug("snapshot before delete failed", exc_info=True)
        if snapshot is None and not self._confirm_final_delete(path, size):
            return

        try:
            h5_edit.delete_node(handle.file, path)
        except EditError as exc:
            self._structure_failed("Delete", exc)
            return
        self._commit_structure_change(
            handle,
            DeleteNodeOp(path, info.kind, snapshot=snapshot),
            path,
        )

    def _confirm_final_delete(self, path: str, size: int) -> bool:
        """Ask before a delete that Ctrl+Z will not be able to reverse."""
        answer = QMessageBox.warning(
            self,
            "Delete permanently?",
            f"{path} holds about {size / 1e6:.0f} MB, too much to keep in "
            "memory for an undo.\n\n"
            "Deleting it cannot be reversed from inside this session. The "
            "file on disk is still untouched until you save, so closing "
            "without saving remains a way back.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _commit_structure_change(self, handle, op, path: str) -> None:
        """Commit an edit that changed the tree's shape.

        Same as an attribute commit, plus the two things a structural
        change makes stale: the browser tree, which is rebuilt with its
        expansion and selection restored, and the entry dropdown, when a
        top-level group came or went.
        """
        state = self._capture_tree_state()
        self._commit_structure_edit(handle, op, path)
        self._rebuild_tree_preserving(state, handle)
        if h5_edit.entry_of(path) is not None or path.count("/") == 1:
            self._repopulate_entries_after_structure_edit()

    def _repopulate_entries_after_structure_edit(self) -> None:
        """Refresh the entry dropdown, keeping the current entry if it lives."""
        current = self.entry_combo.currentText()
        try:
            self._populate_entries()
        except Exception:
            logger.debug("entry repopulate after a structure edit failed",
                         exc_info=True)
            return
        if current and self.entry_combo.findText(current) >= 0:
            self.entry_combo.setCurrentText(current)

    # -- tree state --------------------------------------------------------

    def _capture_tree_state(self):
        """Which browser rows are open, and which one is selected.

        Rows are identified structurally: the file of the root row plus
        the chain of display names below it. That deliberately avoids
        asking each node for its h5py object — the row under the cursor
        may be an external link, and resolving one to answer "where am
        I?" is the cost this whole feature is built to avoid. Only
        expanded rows are walked, so the work is bounded by what the
        user already opened.
        """
        view = self.tree
        model = view.model()
        expanded: list[tuple[str, tuple[str, ...]]] = []
        selected: tuple[str, tuple[str, ...]] | None = None
        chosen = set(view.selectionModel().selectedIndexes()) if (
            view.selectionModel() is not None) else set()

        def walk(parent, file_key: str | None, parts: tuple[str, ...]) -> None:
            nonlocal selected
            for row in range(model.rowCount(parent)):
                index = model.index(row, 0, parent)
                if file_key is None:
                    key = self._tree_root_file(index)
                    if key is None:
                        continue
                    child_parts: tuple[str, ...] = ()
                else:
                    key = file_key
                    name = model.data(index, Qt.ItemDataRole.DisplayRole)
                    if not name:
                        continue
                    child_parts = parts + (str(name),)
                if index in chosen:
                    selected = (key, child_parts)
                if not view.isExpanded(index):
                    continue
                expanded.append((key, child_parts))
                walk(index, key, child_parts)

        try:
            walk(QModelIndex(), None, ())
        except Exception:
            logger.debug("capturing tree state failed", exc_info=True)
            return [], None
        return expanded, selected

    def _tree_root_file(self, index) -> str | None:
        """The filesystem path behind a root row of the browser.

        A root row *is* the open file, so asking it for its h5py object
        resolves nothing that was not already open.
        """
        try:
            obj = index.model().data(index, Hdf5TreeModel.H5PY_OBJECT_ROLE)
            return str(obj.file.filename)
        except Exception:
            logger.debug("tree root lookup failed", exc_info=True)
            return None

    def _find_tree_index(self, key):
        """The proxy index for ``(file, name parts)``, or None.

        Walks down one path component at a time, matching display names,
        so only the groups on that single path are ever populated. A path
        that no longer exists — the node was just deleted — returns None.
        """
        file_key, parts = key
        view = self.tree
        model = view.model()
        current = None
        for row in range(model.rowCount(QModelIndex())):
            index = model.index(row, 0, QModelIndex())
            if self._tree_root_file(index) == file_key:
                current = index
                break
        if current is None:
            return None
        for part in parts:
            nxt = None
            for row in range(model.rowCount(current)):
                index = model.index(row, 0, current)
                if model.data(index, Qt.ItemDataRole.DisplayRole) == part:
                    nxt = index
                    break
            if nxt is None:
                return None
            current = nxt
        return current

    def _reattach_tree_safely(self) -> None:
        """Reattach the browser, surviving a file the viewer can no longer read.

        ``_reattach_silx_tree`` ends by reopening the viewer's
        FrameSource, which raises if the entry it was showing has just
        lost its image stack or a q axis. That is a state the user can
        now reach deliberately (delete q_z, confirm the warning), so it
        has to degrade rather than raise out of a commit that already
        succeeded: the tree is rebuilt, the viewer is emptied, and the
        log says why.
        """
        try:
            self._reattach_silx_tree()
        except Exception as exc:
            logger.debug("reattach after a structure edit failed", exc_info=True)
            try:
                self.viewer.clear()
                self.pipeline_panel.append_log(
                    f"The current entry can no longer be displayed — {exc}"
                )
            except Exception:
                logger.debug("viewer cleanup after a failed reattach failed",
                             exc_info=True)

    def _restore_tree_state(self, state) -> None:
        """Put the browser's open rows and selection back after a rebuild."""
        expanded, selected = state
        for key in expanded:
            index = self._find_tree_index(key)
            if index is not None and index.isValid():
                self.tree.expand(index)
        if selected is None:
            return
        index = self._find_tree_index(selected)
        if index is not None and index.isValid():
            self.tree.selectionModel().select(
                index,
                QItemSelectionModel.SelectionFlag.ClearAndSelect
                | QItemSelectionModel.SelectionFlag.Rows,
            )
            self.tree.selectionModel().setCurrentIndex(
                index, QItemSelectionModel.SelectionFlag.Current)

    def _rebuild_tree_preserving(self, state, handle) -> None:
        """Rebuild the browser and put its open rows and selection back.

        A structural edit changes what the tree should show, and silx's
        model has no public "refresh this subtree" — so the tree is torn
        down and rebuilt, exactly as every other file operation in this
        app already does it. What is new is restoring the state
        afterwards, because here the tree is the surface the user is
        working on and losing their place mid-edit is not acceptable.

        The write handle is re-taken *inside* the detached window. The
        detach releases it along with every other handle, and re-opening
        it before silx reattaches is what keeps the ordering legal: r+
        first, silx's r behind it. Without this the next edit would pay
        for a second detach/reattach of its own.

        Scoped to this call on purpose: the existing detach/reattach
        paths are left exactly as they were.
        """
        self._detach_silx_tree()
        try:
            handle.acquire()
        except EditError:
            # Losing the handle here is recoverable — the next edit
            # acquires again. Better a rebuilt tree than a raise out of a
            # commit that already succeeded.
            logger.debug("re-acquiring the edit handle failed", exc_info=True)
        self._reattach_tree_safely()
        self._restore_tree_state(state)

    # -- dock visibility ---------------------------------------------------

    #: The docks the Structure tab folds away. The File browser is not
    #: among them on purpose: it is this tab's navigator, so hiding it
    #: would leave the panel with nothing to describe.
    _STRUCTURE_FOLDED_DOCKS = (
        "_display_dock",
        "_pipeline_dock",
        "_sim_dock",
        "_conversion_dock",
        "_logs_dock",
        "_profile_dock",
        "_peaks_dock",
        "_scan_tracking_dock",
    )

    def _sync_structure_docks(self) -> None:
        """Fold the side and bottom docks away while the Structure tab is up.

        Nothing in Display, Pipeline, Profiles, Peaks or Scan tracking
        applies to editing a file's structure, and on a laptop they leave
        the panel a narrow column. They come back exactly as they were:
        only docks that were actually open when the tab was entered are
        remembered, so one the user had already closed — or one the
        session mode hides, like Conversion on a converted file — stays
        closed on the way out.

        The set is added to rather than replaced, because a session-mode
        change can re-show a dock while the tab is still in front.
        """
        if not hasattr(self, "structure_panel") or not hasattr(self, "tabs"):
            return
        on_structure = self.tabs.currentWidget() is self.structure_panel
        if on_structure:
            changed = False
            for name in self._STRUCTURE_FOLDED_DOCKS:
                dock = getattr(self, name, None)
                if dock is not None and dock.isVisible():
                    self._structure_folded_docks.add(name)
                    dock.hide()
                    changed = True
            if changed:
                self._settle_dock_chrome()
            self._prepare_structure_editing()
            return
        if not self._structure_folded_docks:
            return
        for name in self._structure_folded_docks:
            dock = getattr(self, name, None)
            if dock is not None:
                dock.show()
        self._structure_folded_docks.clear()
        self._settle_dock_chrome()

    def _settle_dock_chrome(self) -> None:
        """Clean up after showing or hiding a tabified dock.

        Hiding and re-showing docks that share a tab group is a
        re-tabify as far as Qt is concerned, and this window already
        carries two fixes for what that leaves behind: a stale QTabBar
        painted into a corner of the window, and fresh tabs that come up
        without their glyphs. Both have to run here for the same reason
        they run after every other re-tabify.
        """
        for cleanup in ("_hide_stale_dock_tab_bars", "_apply_dock_tab_icons"):
            handler = getattr(self, cleanup, None)
            if handler is None:
                continue
            try:
                handler()
            except Exception:
                logger.debug("dock chrome cleanup %s failed", cleanup,
                             exc_info=True)

    def _prepare_structure_editing(self) -> None:
        """Take the write handle when the tab opens, not on the first edit.

        Acquiring it detaches and rebuilds the file browser, which on a
        large file or a slow mount is a visible pause. Paying that on the
        tab switch — where a moment's work is expected, and where the
        docks are folding anyway — keeps the first actual edit as
        immediate as every edit after it.

        Failure is not reported: the user has not asked for anything yet.
        The next edit acquires again and surfaces the error then.
        """
        session = getattr(self, "_active_session", None)
        if session is None or session.kind != "nexus":
            return
        handle = getattr(self, "_h5_edit_handle", None)
        if handle is not None and handle.is_open and handle.path == Path(
            session.temp_path
        ):
            return
        try:
            self._acquire_edit_handle(session.temp_path)
        except EditError:
            logger.debug("could not take the write handle on tab entry",
                         exc_info=True)

    # -- link intents ------------------------------------------------------

    def _on_structure_new_link(self, target) -> None:
        opened = self._structure_handle_for(target)
        if opened is None:
            return
        handle, parent_path = opened
        dialog = NewLinkDialog(self, parent_path=parent_path)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        kind, name, link_target, filename = dialog.result_values()
        if not name or not link_target:
            return

        def create(f):
            return h5_edit.create_link(
                f, parent_path, name, kind, link_target,
                filename=filename or None,
            )

        try:
            new_path = create(handle.file)
        except EditError as exc:
            self._structure_failed("New link", exc)
            return
        self._commit_structure_change(
            handle, CreateNodeOp(new_path, f"{kind} link", recreate=create),
            new_path,
        )

    def _on_structure_retarget_link(self, target=None) -> None:
        """Point an existing soft or external link somewhere else.

        Called with an explicit target from the browser's context menu,
        and with none from the panel's own button, where the link on show
        is the one meant.
        """
        if target is None:
            target = getattr(self, "_structure_target", None)
            if target is None:
                return
        opened = self._structure_handle_for(target)
        if opened is None:
            return
        handle, path = opened
        before = h5_edit.link_info(handle.file, path)
        if before is None or before.kind == "hard":
            self._structure_failed(
                "Retarget link",
                EditError(
                    "A hard link is the object itself, not a pointer — "
                    "there is nothing to retarget."
                ),
            )
            return
        parent_path, name = h5_edit.split_path(path)
        dialog = NewLinkDialog(
            self, parent_path=path, kind=before.kind, name=name,
            target=before.target, filename=before.filename, retarget=True,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        kind, _, link_target, filename = dialog.result_values()
        if not link_target:
            return
        try:
            h5_edit.retarget_link(
                handle.file, path, kind, link_target,
                filename=filename or None,
            )
        except EditError as exc:
            self._structure_failed("Retarget link", exc)
            return
        self._commit_structure_change(
            handle,
            RetargetLinkOp(
                path,
                before.kind, before.target, before.filename,
                kind, link_target, filename,
            ),
            path,
        )

    def _on_structure_follow_link(self) -> None:
        """Resolve the link on show and describe what it points at.

        The only place the editor opens an external link, and only
        because the user asked. Everything else — the tree walk, the
        search, the layout check, the panel's own render — describes the
        link and leaves the target closed.
        """
        node = self._pending_structure_node
        if node is None:
            nodes = self._safe_selected_h5_nodes()
            node = nodes[0] if nodes else None
        if node is not None:
            self._render_structure_node(node, follow=True)

    # -- copy and paste ----------------------------------------------------

    def _structure_owns_clipboard(self) -> bool:
        """Whether Ctrl+C / Ctrl+V belong to the editor right now.

        The same two ways in as Ctrl+Z: the Structure tab in front, or
        the File browser holding focus. Everywhere else these stay the
        peak clipboard's, unchanged.
        """
        if not hasattr(self, "structure_panel"):
            return False
        in_editor = self.tabs.currentWidget() is self.structure_panel
        in_browser = hasattr(self, "tree") and self.tree.hasFocus()
        return in_editor or in_browser

    def _structure_selected_target(self):
        """``(file, path)`` for the current selection, however it was made."""
        target = getattr(self, "_structure_target", None)
        if target is not None:
            return target
        nodes = self._safe_selected_h5_nodes()
        if not nodes:
            return None
        file_path = self._node_filename(nodes[0])
        raw_path = self._node_h5_path(nodes[0])
        if file_path is None or raw_path is None:
            return None
        return (Path(file_path), h5_edit.normalize_path(raw_path))

    def _on_structure_copy(self, target=None, *, cut: bool = False) -> None:
        """Put a node — or the selected attribute — on the clipboard.

        A reference, not a payload: copying a 4 GB stack costs nothing
        until it is pasted.
        """
        attr = self.structure_panel.selected_attribute()
        if target is None:
            target = self._structure_selected_target()
        if target is None:
            return
        file_path, path = target
        if attr and not cut:
            self._structure_clip = StructureClip(
                "attribute", Path(file_path), path, attr=attr)
        else:
            self._structure_clip = StructureClip(
                "node", Path(file_path), path,
                mode="cut" if cut else "copy")
        self.statusBar().showMessage(self._structure_clip.describe(), 4000)

    def _on_structure_paste(self, target=None) -> None:
        """Paste the clipboard into the selected group."""
        clip = getattr(self, "_structure_clip", None)
        if clip is None:
            return
        if target is None:
            target = self._structure_selected_target()
        if target is None:
            return
        self._paste_clip(clip, target)

    def _on_structure_paste_from_file(self, target=None) -> None:
        """Pick a node in a file that is not open, and paste it here."""
        if target is None:
            target = self._structure_selected_target()
        if target is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Copy from file", "",
            "HDF5 / NeXus files (*.h5 *.hdf5 *.nxs);;All files (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not path:
            return
        dialog = PickNodeDialog(path, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        source = dialog.selected_path()
        if not source:
            return
        self._paste_clip(
            StructureClip("node", Path(path), source), target)

    def _paste_destination(self, handle, path: str) -> str | None:
        """The group a paste lands in: the node itself, or its parent.

        Dropping onto a dataset means "beside this", which is what a
        user expects from every file manager they have ever used.
        """
        try:
            info = h5_edit.node_info(handle.file, path)
        except EditError as exc:
            self._structure_failed("Paste", exc)
            return None
        if info.kind == "group":
            return path
        if path == "/":
            return "/"
        return h5_edit.split_path(path)[0]

    def _paste_clip(self, clip, target) -> None:
        opened = self._structure_handle_for(target)
        if opened is None:
            return
        handle, path = opened
        destination = self._paste_destination(handle, path)
        if destination is None:
            return

        if clip.kind == "attribute":
            self._paste_attribute(handle, clip, destination)
            return

        same_file = Path(clip.file_path) == handle.path
        if clip.mode == "cut" and not same_file:
            self._structure_failed(
                "Paste",
                EditError(
                    "Cut moves a node inside one file. To bring it into "
                    "another file, copy it and delete the original."
                ),
            )
            return
        if clip.mode == "cut" and h5_edit.normalize_path(destination).startswith(
            clip.path.rstrip("/") + "/"
        ):
            self._structure_failed(
                "Paste", EditError("A group cannot be moved inside itself."))
            return

        try:
            existing = list(handle.file[destination].keys())
        except (KeyError, OSError) as exc:
            self._structure_failed("Paste", EditError(str(exc)))
            return
        name = self._ask_paste_name(clip.name, existing, destination)
        if name is None:
            return

        if clip.mode == "cut":
            try:
                new_path = h5_edit.move_node(
                    handle.file, clip.path, destination, name)
            except EditError as exc:
                self._structure_failed("Paste", exc)
                return
            self._structure_clip = None
            self._commit_structure_change(
                handle, MoveNodeOp(clip.path, new_path), new_path)
            return

        source_file = None if same_file else Path(clip.file_path)

        def paste(f):
            if source_file is None:
                return h5_edit.copy_node(f, destination, name, f, clip.path)
            with h5py.File(source_file, "r") as src:
                return h5_edit.copy_node(f, destination, name, src, clip.path)

        try:
            new_path = paste(handle.file)
        except (EditError, OSError) as exc:
            self._structure_failed("Paste", EditError(str(exc)))
            return
        self._commit_structure_change(
            handle, CreateNodeOp(new_path, "copy", recreate=paste), new_path)

    def _ask_paste_name(self, wanted: str, existing, destination: str):
        """The name to paste under, or None if the user backed out.

        A free name goes in without a prompt. A collision asks, with a
        suggestion pre-filled — silently renaming someone's data is how
        a file ends up with ``q_xy_copy3`` and nobody knowing why.
        """
        if wanted not in set(existing):
            return wanted
        suggestion = free_name(existing, wanted)
        name, ok = QInputDialog.getText(
            self, "Paste",
            f"{destination} already has {wanted!r}.\nPaste as:",
            text=suggestion,
        )
        if not ok or not name.strip():
            return None
        return name.strip()

    def _paste_attribute(self, handle, clip, destination: str) -> None:
        """Copy one attribute onto the node the paste landed on."""
        try:
            if Path(clip.file_path) == handle.path:
                value = h5_edit.read_attrs(handle.file, clip.path)[clip.attr]
            else:
                with h5py.File(clip.file_path, "r") as src:
                    value = h5_edit.read_attrs(src, clip.path)[clip.attr]
        except (EditError, KeyError, OSError) as exc:
            self._structure_failed("Paste attribute", EditError(str(exc)))
            return
        # An attribute belongs to whatever is selected, group or dataset
        # alike, so it pastes onto the node itself rather than its parent.
        landing = self._structure_selected_target()
        path = landing[1] if landing is not None else destination
        try:
            old = h5_edit.set_attr(handle.file, path, clip.attr, value)
        except EditError as exc:
            self._structure_failed("Paste attribute", exc)
            return
        self._commit_structure_edit(
            handle, SetAttrOp(path, clip.attr, old, value), path)

    # -- the value grid ----------------------------------------------------

    def _on_structure_edit_values(self) -> None:
        """Open the dataset on show in the value grid.

        Three bands, by size. Small enough to edit by hand: an editable
        grid. Large but still loadable: read-only, with an explicit "Edit
        anyway" — a grid over a detector image is a way to make a mistake,
        not a way to fix one. Beyond that it is not loaded at all, because
        the grid holds the whole array in memory and the Data tab already
        streams what it cannot.
        """
        target = getattr(self, "_structure_target", None)
        if target is None:
            return
        opened = self._structure_handle_for(target)
        if opened is None:
            return
        handle, path = opened
        try:
            info = h5_edit.node_info(handle.file, path)
            if info.kind != "dataset":
                raise EditError(f"{path} is not a dataset.")
            if info.n_cells > h5_edit.VIEW_CELL_LIMIT:
                raise EditError(
                    f"{path} holds {info.n_cells:,} values — too many to "
                    "load into a grid. Use the Data tab to look at it."
                )
            array = h5_edit.read_block(handle.file, path)
        except EditError as exc:
            self._structure_failed("Edit values", exc)
            return

        editable = info.n_cells <= h5_edit.EDITABLE_CELL_LIMIT
        reason = "" if editable else (
            f"{info.n_cells:,} values — opened read-only so a stray "
            "keystroke cannot rewrite a detector image."
        )
        dialog = ArrayEditDialog(
            self, path=path, array=array, editable=editable,
            read_only_reason=reason,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._apply_grid_result(handle, path, array, dialog.result_array())

    def _apply_grid_result(self, handle, path, before, after) -> None:
        """Turn what the grid returned into one undoable edit.

        A changed row count moves every index after it, so a sparse cell
        diff cannot describe it — that becomes a whole-array replace. An
        ordinary edit session touches a handful of cells out of however
        many, so it stays sparse and its undo costs the same handful.
        """
        replace = before.shape != after.shape or before.dtype != after.dtype
        changed = {} if replace else self._diff_cells(before, after)
        if not replace and len(changed) > MAX_SPARSE_CELLS:
            # A sparse diff is the right shape for a few hand-typed
            # cells. Past this many it is neither: building the dict,
            # writing cell by cell and holding it for undo all become
            # per-cell Python work on the GUI thread. One array write
            # says the same thing in one operation.
            replace, changed = True, {}
        if not replace and not changed:
            # OK on a grid nobody typed into. Nothing happened, so
            # nothing is asked and nothing is recorded.
            return
        protection = h5_edit.protection_reason(path)
        if protection and not self._confirm_protected("Change", path, protection):
            return
        try:
            if replace:
                h5_edit.rewrite_dataset(handle.file, path, after)
                op = ReplaceDataOp(path, before, after)
            else:
                previous = h5_edit.write_cells(handle.file, path, changed)
                op = WriteCellsOp(path, previous, changed)
        except EditError as exc:
            self._structure_failed("Edit values", exc)
            return
        self._commit_structure_edit(handle, op, path)

    @staticmethod
    def _diff_cells(before, after) -> dict:
        """``{index: new value}`` for every cell that differs.

        Compound arrays are compared field by field so the index is
        ``(row, field)`` — the same shape ``write_cells`` takes, and the
        same one an undo writes back.
        """
        changed: dict = {}
        if before.dtype.names:
            for field in before.dtype.names:
                rows = np.nonzero(before[field] != after[field])[0]
                for row in rows:
                    changed[(int(row), field)] = after[field][row]
            return changed
        if before.ndim == 0:
            return {(): after[()]} if before[()] != after[()] else {}
        for index in zip(*np.nonzero(before != after)):
            changed[tuple(int(i) for i in index)] = after[index]
        return changed
