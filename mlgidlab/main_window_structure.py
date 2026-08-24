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

from mlgidlab import h5_edit
from mlgidlab.browser_widgets import _ImageFileNode

import logging

logger = logging.getLogger(__name__)


class StructureMixin:
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

    @staticmethod
    def _node_h5_path(node) -> str | None:
        """The absolute in-file path of a silx tree node.

        silx exposes it as ``local_name``; the h5py object's ``name`` is
        the fallback, matching how ``_node_entry_name`` resolves the same
        thing.
        """
        for getter in (
            lambda n: getattr(n, "local_name", None),
            lambda n: n.h5py_object.name,
        ):
            try:
                value = getter(node)
            except Exception:
                logger.debug(
                    "suppressed exception in _node_h5_path", exc_info=True)
                continue
            if value:
                return str(value)
        return None

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
        h5_path = self._node_h5_path(node)
        if file_path is None or h5_path is None:
            panel.clear()
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

        panel.set_note(
            "Raw detector files are read-only in mlgidLAB — convert one to "
            "NeXus to edit it." if is_raw else ""
        )

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
        """Empty the panel — called when the session it described goes away."""
        self._pending_structure_node = None
        self._structure_checked_file = None
        if hasattr(self, "structure_panel"):
            self.structure_panel.clear()
            self.structure_panel.set_note("")
            self.structure_panel.set_issues([])
