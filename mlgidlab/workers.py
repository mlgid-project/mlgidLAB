from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal, Slot

from mlgidlab.conversion import execute as conversion_execute
from mlgidlab.conversion_panel import ConversionConfig, RawScan
from mlgidlab.pipeline import (
    PipelineCommand,
    execute,
    parse_cif_input,
)
from mlgidlab.session import Session

logger = logging.getLogger(__name__)


def _trigger_pipeline_imports() -> None:
    """Force ``mlgidbase`` (and its transitive ``pygid`` import) to load.

    Several ``pygid`` submodules (``coordmaps``, ``datasaver``,
    ``conversion``, ``dataloader``) call ``logging.basicConfig`` at
    module top — which first **removes every handler already attached
    to the root logger** before installing their own ``StreamHandler``.
    If we attached our ``_SignalLogHandler`` and *then* triggered the
    lazy import, our handler would be ripped out before the first log
    line ever reaches it. Calling this helper before installing our
    handler means pygid's destructive `removeHandler` sequence has
    already run; module bodies don't re-execute on subsequent imports
    so our handler stays attached for the rest of the worker's run.

    Failures are swallowed: a missing ``mlgidbase`` install just means
    the worker will surface the actual ``ImportError`` from
    ``execute`` later, with the proper exception channel.
    """
    try:
        import mlgidbase  # noqa: F401
    except Exception:
        logger.debug("suppressed exception in _trigger_pipeline_imports", exc_info=True)
        pass


def _trigger_conversion_imports() -> None:
    """Same idea as ``_trigger_pipeline_imports`` for raw conversion.

    ``conversion.execute`` lazily imports ``pygid``; make that happen
    before the worker attaches its log handler so pygid's basicConfig
    side effect doesn't strip our sink.
    """
    try:
        import pygid  # noqa: F401
    except Exception:
        logger.debug("suppressed exception in _trigger_conversion_imports", exc_info=True)
        pass


class CifParseWorker(QObject):
    """Parses CIF input (raw .cif files / folder / pickle) into a CifPattern.

    Runs on a worker thread because raw CIF parsing simulates a 2D
    diffraction pattern per CIF and can take several seconds for a
    typical batch — we don't want the GUI thread blocked while it works.
    Emits ``finished(CifPattern | str | None, Exception | None)`` —
    the result is the cached object (CifPattern for raw, str path for a
    pickle, None for empty input).
    """

    finished = Signal(object, object)

    def __init__(
        self,
        cif_input: str,
        nexus_file: Path,
        entry: str | None = None,
    ) -> None:
        super().__init__()
        self._cif_input = cif_input
        self._nexus_file = nexus_file
        self._entry = entry

    def run(self) -> None:
        try:
            result = parse_cif_input(
                self._cif_input, self._nexus_file, self._entry
            )
            self.finished.emit(result, None)
        except Exception as exc:
            logger.debug("suppressed exception in CifParseWorker.run", exc_info=True)
            self.finished.emit(None, exc)


class CopyWorker(QObject):
    """Classify + (for NeXus) copy + pre-warm the first entry — all off the
    GUI thread, so the Open click is instant and the window never freezes.

    The open path NEVER resolves the master's external links (doing so
    opens every linked scan, which is slow on network storage and holds
    the GIL, freezing the GUI even from a worker thread). Entry names come
    from a shallow ``list_entry_names`` read of the master, and only the
    first entry's scan is opened — for the initial frame. Only NeXus files
    are copied + warmed; raw files are classified and reported back for the
    GUI to bundle. ``finished`` carries one result dict::

        {"path", "kind", "session", "prewarm", "entries", "error"}

    where ``kind`` is ``"nexus" | "raw" | None``, ``prewarm`` is
    ``(first_entry, FrameSource)`` or ``None``, ``entries`` is the
    (shallow, unresolved) entry-name list the GUI fills the combo from, or
    ``None``, and ``raw_entries`` is the ``RawEntry`` list found while
    classifying a raw file (cached so the GUI never re-walks the file's
    full metadata on its own thread — on a big beamtime file that walk
    took seconds and was the raw-open freeze).

    ``progress(percent, label)`` drives the determinate bottom-left bar:
    coarse stages for the cheap steps, granular ticks for the two costs
    that actually take time (the temp copy, in bytes; the raw metadata
    walk, in top-level groups).
    """

    finished = Signal(object)  # the result dict above
    progress = Signal(int, str)  # (percent 0..100, status label)

    def __init__(self, original_path: Path):
        super().__init__()
        self._original_path = original_path

    def run(self) -> None:
        result: dict = {
            "path": self._original_path,
            "kind": None,
            "session": None,
            "prewarm": None,
            "prewarm_overlays": None,
            "entries": None,
            "raw_entries": None,
            "error": None,
        }
        try:
            from mlgidlab import file_model  # lazy: avoid import cycle at load
            from mlgidlab.session import NexusSession

            # Standalone image files (TIFF/CBF/EDF) are read via fabio, not
            # h5py — short-circuit before the HDF5 shallow-list + raw walk
            # (which would just throw on a non-HDF5 file). They open as a
            # raw-mode session; the stack entry is built lazily at
            # activation (main_window._populate_raw_entries) from the bundled
            # paths, so no per-file entry list is attached here (raw_entries
            # stays None, exactly like a raw file whose walk found nothing
            # to pre-cache).
            if file_model.is_fabio_image(self._original_path):
                result["kind"] = "raw"
                self.progress.emit(100, "Done")
                self.finished.emit(result)
                return

            self.progress.emit(10, "Reading file")
            # SHALLOW listing: read only the master's entry-group link names
            # (``keys()`` does not dereference external links). The previous
            # version opened every linked scan here to read its ``signal``
            # attr — seconds-to-minutes on a network share, and h5py holds
            # the GIL across those opens, so the GUI froze even though this
            # runs off the GUI thread. Reading just the master is ~1 ms.
            try:
                names = file_model.list_entry_names(self._original_path)
            except Exception:
                logger.debug("suppressed shallow list in CopyWorker.run", exc_info=True)
                names = []

            if names and self._entries_are_mlgid(file_model, names):
                # Entry_* group names alone aren't enough — LIMA/Eiger
                # detector files also use ``entry_0000``-style roots but
                # carry no mlgid signal, and feeding them to the NeXus
                # loader fails with "component not found"; they must fall
                # through to the raw classifier below. The signal probe
                # resolves at most a couple of entries (one external link
                # each on a master, same cost as the prewarm), never all.
                result["kind"] = "nexus"
                result["entries"] = names
                self.progress.emit(40, "Copying file")
                # Copy to temp with byte progress mapped into 40..90 —
                # a converted file can be several GB, so the copy is
                # where the open actually spends its time.
                session = NexusSession.open(
                    self._original_path,
                    progress=self._make_copy_tick(),
                )
                result["session"] = session
                # Only the FIRST entry's scan is opened here (frame 0, for
                # the initial render) — one external link, not all of them.
                self.progress.emit(90, "Loading first entry")
                result["prewarm"] = self._prewarm_first_entry(
                    session, file_model, names[0]
                )
                # Read the first frame's peaks off-thread too (via the
                # prewarm source's open handle), so the GUI never does an
                # SFTP read to populate overlays on open. Over high-latency
                # SFTP even that small read on the GUI thread froze.
                pw = result["prewarm"]
                if pw is not None:
                    try:
                        peaks, matched = pw[1].read_frame_overlays(0)
                        result["prewarm_overlays"] = (0, peaks, matched)
                    except Exception:
                        logger.debug("suppressed overlay prewarm in CopyWorker", exc_info=True)
                self.progress.emit(100, "Done")
            else:
                # No entry_* groups → not a NeXus master. Fall back to the
                # raw-detector walk. On a big beamtime file this visits
                # every group's metadata (seconds), so it ticks the bar per
                # top-level group AND its result is kept — the GUI used to
                # re-run the same walk on its own thread to fill the combo
                # + Conversion panel, which was the raw-open freeze.
                self.progress.emit(20, "Scanning for raw detector data")
                try:
                    raw_entries = file_model.list_raw_entries(
                        self._original_path, progress=self._make_scan_tick()
                    )
                except Exception:
                    logger.debug("suppressed raw check in CopyWorker.run", exc_info=True)
                    raw_entries = None
                if raw_entries:
                    result["kind"] = "raw"
                    result["raw_entries"] = raw_entries
                    self.progress.emit(100, "Done")
        except Exception as exc:
            logger.debug("suppressed exception in CopyWorker.run", exc_info=True)
            result["error"] = exc
        self.finished.emit(result)

    def _make_copy_tick(self):
        """Byte-progress callback for the temp copy, mapped into 40..90.

        Emits only when the mapped percent changes so a many-chunk copy
        doesn't flood the queued-signal connection.
        """
        last = [-1]

        def tick(done: int, total: int) -> None:
            pct = 40 + int(50 * done / max(total, 1))
            if pct != last[0]:
                last[0] = pct
                mb_done, mb_total = done // 2**20, total // 2**20
                self.progress.emit(
                    pct, f"Copying file ({mb_done} / {mb_total} MB)"
                )

        return tick

    def _make_scan_tick(self):
        """Group-progress callback for the raw-detector walk, 20..95.

        Same percent-dedup as the copy tick (a file can have thousands
        of top-level groups).
        """
        last = [-1]

        def tick(done: int, total: int) -> None:
            pct = 20 + int(75 * done / max(total, 1))
            if pct != last[0]:
                last[0] = pct
                self.progress.emit(
                    pct, f"Scanning raw datasets ({done}/{total})"
                )

        return tick

    def _entries_are_mlgid(self, file_model, names: list) -> bool:
        """Whether the entry_* groups carry mlgid signals (NeXus output).

        Probes up to the first three entries; the first one that
        RESOLVES decides ("mlgid" → nexus, "foreign" layout → not).
        Extra probes only run for unreadable entries (broken external
        link), so a healthy file costs exactly one entry resolve. When
        nothing is readable (master with its scans offline) the entry_*
        names are trusted — the file still opens as nexus with its
        entry list, and individual loads fail gracefully later.
        """
        for name in names[:3]:
            verdict = file_model.classify_entry_data(self._original_path, name)
            if verdict == "mlgid":
                return True
            if verdict == "foreign":
                return False
        return True

    @staticmethod
    def _prewarm_first_entry(session, file_model, first):
        """Open ``first`` entry's FrameSource and pull frame 0 (+ its polar
        resample, the default view) into its LRU — the slow read, done here
        so the GUI installs the file instantly. Best-effort: returns None on
        any failure (the GUI then falls back to a synchronous load)."""
        try:
            source = file_model.FrameSource(file_path=session.temp_path, entry=first)
            source.acquire()
            source.get_cartesian(0)
            try:
                source.get_polar(0)
            except Exception:
                logger.debug("suppressed polar prewarm in CopyWorker", exc_info=True)
            return (first, source)
        except Exception:
            logger.debug("suppressed prewarm in CopyWorker", exc_info=True)
            return None


class EntryLoadWorker(QObject):
    """Open + warm an entry's first frame off the GUI thread.

    Switching entries on a master that links external scans means opening
    that entry's (possibly remote) scan and reading a full detector frame
    — seconds on a slow share, which froze the GUI when it ran inline on
    every combo / file-browser click. This does that read on a worker
    thread and hands the GUI a ready ``FrameSource`` (frame 0 cartesian +
    polar already in its LRU), exactly like ``CopyWorker``'s prewarm; the
    GUI installs it with ``file_model.stack_from_source`` without any
    further disk I/O.

    Persistent (one long-lived thread, driven by a queued ``load`` slot)
    so rapid switching doesn't churn threads. Each request carries a
    monotonic ``request_id``; the GUI ignores (and releases) results whose
    id is stale, so only the latest switch renders.
    """

    # request_id, entry, FrameSource | None, overlays | None
    # overlays = (frame, peaks_dict, matched_list) for the landed frame.
    loaded = Signal(int, str, object, object)
    # request_id, combo label, LazyRawStack | None — raw-mode counterpart
    # of ``loaded`` (raw stacks have no overlays / polar view).
    raw_loaded = Signal(int, str, object)

    @Slot(str, str, int)
    def load(self, file_path: str, entry: str, request_id: int) -> None:
        from mlgidlab import file_model  # lazy: avoid import cycle at load

        source = None
        overlays = None
        try:
            source = file_model.FrameSource(file_path=Path(file_path), entry=entry)
            source.acquire()
            source.get_cartesian(0)
            try:
                source.get_polar(0)
            except Exception:
                logger.debug("suppressed polar warm in EntryLoadWorker", exc_info=True)
            # Read frame 0's peaks here too (same open handle), so the GUI
            # does ZERO SFTP I/O when it installs the entry — the peak read
            # on the GUI thread was the residual per-switch freeze on a
            # high-latency SFTP mount.
            try:
                peaks, matched = source.read_frame_overlays(0)
                overlays = (0, peaks, matched)
            except Exception:
                logger.debug("suppressed overlay read in EntryLoadWorker", exc_info=True)
        except Exception:
            logger.debug("suppressed read in EntryLoadWorker.load", exc_info=True)
            if source is not None:
                try:
                    source.release()
                except Exception:
                    logger.debug("suppressed release in EntryLoadWorker", exc_info=True)
            source = None
            overlays = None
        self.loaded.emit(request_id, entry, source, overlays)

    @Slot(object, int)
    def load_raw(self, raw_entry, request_id: int) -> None:
        """Open one raw detector dataset and warm its first frame.

        Raw counterpart of ``load``: hands the GUI a ready
        ``LazyRawStack`` (frame 0 already in its LRU) so a raw entry
        click renders without any GUI-thread disk I/O — the eager
        full-stack ``load_raw_dataset`` this replaces froze the window
        for the duration of a whole-dataset read.
        """
        from mlgidlab import file_model  # lazy: avoid import cycle at load

        stack = None
        try:
            stack = file_model.build_raw_stack(raw_entry)
            stack.acquire()
            stack.get_frame(0)
        except Exception:
            logger.debug("suppressed read in EntryLoadWorker.load_raw", exc_info=True)
            if stack is not None:
                try:
                    stack.release()
                except Exception:
                    logger.debug("suppressed release in EntryLoadWorker.load_raw", exc_info=True)
            stack = None
        self.raw_loaded.emit(request_id, raw_entry.label, stack)


class _SignalLogHandler(logging.Handler):
    """Forwards log records to a Qt signal for cross-thread display."""

    def __init__(self, sink: Signal) -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._sink.emit(self.format(record))
        except Exception:
            logger.debug("suppressed exception in _SignalLogHandler.emit", exc_info=True)
            pass


# mlgidbase emits one of these lines per frame at the end of each
# frame's processing for detection / fitting / matching. We use them
# as the only available "frame N done" signal, since none of
# mlgidbase, pygidfit, pygid or mlgidmatch accepts a progress
# callback. Keep this in sync with mlgidbase's log strings:
#   mlgidbase/mlgiddetect_functions.py:~182
#   mlgidbase/pygidfit_functions.py    (fitted)
#   mlgidbase/mlgidmatch_functions.py:~244
#
# Anchored at start with re.match (not re.search) so noise records
# don't pay for a full scan. The ``_FRAME_DONE_PREFIX`` fast-path
# check skips records whose raw msg doesn't begin with the prefix,
# avoiding a getMessage() call on the 99% of records that aren't
# frame-completion lines.
_FRAME_DONE_PREFIX = "Saved "
_FRAME_DONE_RE = re.compile(
    r"Saved (?P<kind>\w+) peaks to file: .*, entry: (?P<entry>[^,]+), "
    r"frame: (?P<frame>\d+)"
)


class _FrameProgressHandler(logging.Handler):
    """Counts mlgidbase per-frame completion log lines and emits a Qt
    signal so the GUI can size a progress bar.

    Composed alongside ``_SignalLogHandler`` rather than folded into
    it: the string handler still forwards every record to the log
    panel verbatim; this one classifies the subset that match
    ``_FRAME_DONE_RE`` and turns them into structured progress events.
    Decoupling the two means a future tweak to the visible log format
    can't accidentally break progress tracking and vice versa.

    Performance: the raw-msg prefix check (``startswith``) is the only
    work paid by the overwhelming majority of records (clustering /
    fitting INFO lines, etc.). Only the per-frame "Saved … peaks"
    records pay ``getMessage`` + regex match + Qt signal emit, and
    those are emitted at most once per frame by mlgidbase.
    """

    def __init__(self, sink: Signal, total: int, op_name: str) -> None:
        super().__init__()
        self._sink = sink
        self._total = int(total)
        self._op = str(op_name)
        self._done = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Fast-path: bail before computing the formatted message
            # on records whose raw format string can't be a frame-
            # completion line. mlgidbase uses f-strings for these
            # lines so ``record.msg`` is the literal final text
            # including the ``Saved `` prefix.
            raw = record.msg
            if not isinstance(raw, str) or not raw.startswith(_FRAME_DONE_PREFIX):
                return
            # Anchored match — the line is fully described by the
            # regex from char 0 so re.match avoids a full-string scan.
            m = _FRAME_DONE_RE.match(record.getMessage())
            if not m:
                return
            self._done += 1
            # Don't clamp _done > _total: if mlgidbase emits more
            # frame-completion lines than we expected (multi-entry
            # scope where our pre-count missed an entry, e.g.), the
            # bar will look saturated which beats stalling at <100%.
            # Cap the emitted ``done`` at ``total`` so the panel UI
            # never sees a value past max.
            done_capped = min(self._done, self._total) if self._total > 0 else self._done
            self._sink.emit(
                done_capped,
                self._total,
                self._op,
                m.group("entry"),
            )
        except Exception:
            logger.debug("suppressed exception in _FrameProgressHandler.emit", exc_info=True)
            pass


class ConversionWorker(QObject):
    """Runs ``pygid`` raw → NeXus conversion off the GUI thread.

    Mirrors ``PipelineWorker``: streams every ``pygid`` log record through
    ``log`` and emits ``finished(list[Path] | None, Exception | None)``
    with the produced output files (or the failure exception). One
    progress-style ``progress`` signal fires per scan boundary so the
    host's QProgressDialog can show batch progress.
    """

    finished = Signal(object, object)  # (list[Path] | None, Exception | None)
    log = Signal(str)
    progress = Signal(int, int)        # (done, total)

    def __init__(
        self,
        scans: list[RawScan],
        cfg: ConversionConfig,
    ) -> None:
        super().__init__()
        self._scans = list(scans)
        self._cfg = cfg

    def run(self) -> None:
        # Force pygid's import-time basicConfig side effect to run
        # before we install our handler — see the helper docstring
        # for the gory details.
        _trigger_conversion_imports()

        handler = _SignalLogHandler(self.log)
        handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
        # Attach to the root logger so propagated records from pygid
        # (which logs through class-named loggers like ``CoordMaps``,
        # ``DataLoader``, ``Datasaver``, plus the unnamed root logger)
        # all reach our sink. Every pygid logger has the default
        # ``propagate=True``.
        root = logging.getLogger()
        root.addHandler(handler)
        prev_level = root.level
        if root.level == logging.NOTSET or root.level > logging.INFO:
            root.setLevel(logging.INFO)

        total = len(self._scans)
        try:
            self.progress.emit(0, total)
            # ``conversion.execute`` runs all scans internally; per-scan
            # progress would require slicing it open, which we don't do
            # in v1. Fire the start + end progress events so the dialog
            # shows the work range; v2 can break this into per-scan
            # callbacks.
            outputs = conversion_execute(self._scans, self._cfg)
            self.progress.emit(total, total)
            self.finished.emit(outputs, None)
        except Exception as exc:
            logger.debug("suppressed exception in ConversionWorker.run", exc_info=True)
            import traceback
            self.log.emit(traceback.format_exc())
            self.finished.emit(None, exc)
        finally:
            root.removeHandler(handler)
            root.setLevel(prev_level)


class PipelineWorker(QObject):
    """Runs one mlgidbase pipeline command and streams its log records.

    Emits ``frameProgress(done, total, op, entry)`` for the host's
    multi-frame progress bar. ``total`` is pre-computed from the file
    + the scope encoded in ``command.kwargs`` so the bar's range is
    fixed for the lifetime of the run; ``done`` increments whenever
    mlgidbase logs a per-frame completion line (see
    ``_FrameProgressHandler``). The host's panel UI hides the bar
    when ``total <= 1`` so single-frame runs do not show a
    meaningless 1/1 widget.
    """

    finished = Signal(object, object)  # (result, Exception | None)
    log = Signal(str)
    frameProgress = Signal(int, int, str, str)  # (done, total, op, entry)

    def __init__(self, file_path: Path, command: PipelineCommand) -> None:
        super().__init__()
        self._file_path = file_path
        self._command = command

    def _resolve_total_frames(self) -> tuple[int, str]:
        """Resolve ``(total_frames, entry_label)`` for the run's scope.

        The mlgidBASE methods we drive iterate frames internally —
        per-entry (when ``entry`` is pinned and ``frame_num`` is None),
        a single frame (when ``frame_num`` is set), or every entry +
        every frame (when neither is set). Pre-computing the total
        lets the panel size a determinate progress bar; for the
        single-frame case we return ``(1, entry)`` so the host can
        decide to hide the bar.

        Best-effort: any failure resolving counts (file just closed,
        bad metadata) falls back to ``(0, "")`` which the host treats
        as "indeterminate, don't show a bar".
        """
        from mlgidlab.file_model import count_frames, list_entries

        kw = self._command.kwargs
        entry = kw.get("entry")
        frame_num = kw.get("frame_num")
        try:
            # interpolate_tracks carries its work list in kwargs: one
            # progress tick per planned gap frame (the op logs a
            # "Saved fitted peaks" line per frame).
            if self._command.op_name == "interpolate_tracks":
                plan = kw.get("plan") or {}
                return len(plan), str(entry or "")
            # A frame list (e.g. the matching pass chained after
            # interpolate_tracks) ticks once per listed frame.
            if isinstance(frame_num, (list, tuple)):
                return len(frame_num), str(entry or "")
            # Single-frame scope: skip a list_entries open, hide the
            # bar at the panel layer.
            if isinstance(frame_num, int):
                return 1, str(entry or "")
            if isinstance(entry, str) and entry:
                return count_frames(self._file_path, entry), entry
            # All entries: sum across every q-image entry in the file.
            entries = list_entries(self._file_path)
            total = sum(count_frames(self._file_path, e) for e in entries)
            label = "all entries" if entries else ""
            return total, label
        except Exception:
            logger.debug("suppressed exception in PipelineWorker._resolve_total_frames", exc_info=True)
            return 0, ""

    def run(self) -> None:
        # Force pygid / mlgidbase import-time basicConfig side effects
        # to run before we install our handler — see the helper
        # docstring for why this ordering matters.
        _trigger_pipeline_imports()

        handler = _SignalLogHandler(self.log)
        handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
        # Attach to the root logger. Pipeline modules use a mix of
        # class-named loggers (``mlgidBASE``, ``DataLoader``,
        # ``CoordMaps``) and the unnamed root logger
        # (``logging.getLogger()`` in mlgidbase.peak_operations,
        # mlgidbase.nexus_operations, mlgidbase.mlgiddetect_functions,
        # pygidfit.process_scans, …). Hooking every one of those by
        # name is fragile; attaching to root catches them all via
        # default propagation.
        root = logging.getLogger()
        root.addHandler(handler)
        prev_level = root.level
        if root.level == logging.NOTSET or root.level > logging.INFO:
            root.setLevel(logging.INFO)

        total, entry_label = self._resolve_total_frames()
        op_name = self._command.op_name
        progress_handler = _FrameProgressHandler(self.frameProgress, total, op_name)
        root.addHandler(progress_handler)
        # Initial emit so the panel can paint a 0% bar before
        # mlgidbase starts logging its first frame. The host keeps the
        # bar hidden when ``total <= 1`` so single-frame runs stay quiet.
        self.frameProgress.emit(0, total, op_name, entry_label)

        try:
            result = execute(self._file_path, self._command)
            # Cap at total on the successful path. mlgidbase may have
            # emitted slightly fewer frame-completion lines than our
            # pre-count expected (e.g. it skipped a frame that had no
            # work). Pin to total so the bar always reads "done" on
            # success.
            if total > 0:
                self.frameProgress.emit(total, total, op_name, entry_label)
            self.finished.emit(result, None)
        except Exception as exc:
            # Stream the traceback through the log channel so the user
            # can see *where* mlgidBASE failed — the modal dialog only
            # shows the bare exception message which is often opaque
            # ("invalid index to scalar variable" doesn't tell anyone
            # which dataset / function tripped). The traceback lands in
            # the panel log alongside the mlgidbase log lines.
            logger.debug("suppressed exception in PipelineWorker.run", exc_info=True)
            import traceback
            self.log.emit(traceback.format_exc())
            self.finished.emit(None, exc)
        finally:
            root.removeHandler(handler)
            root.removeHandler(progress_handler)
            root.setLevel(prev_level)


class ScanProfileWorker(QObject):
    """Computes per-scan display aids for the phase-tracking views.

    Two modes, both read-only through the worker's OWN h5py handle
    (never the viewer's FrameSource), frame by frame so memory stays
    O(one frame) even on 100+ GB scans:

    - ``"waterfall"``: per frame, resample to polar
      (``mlgidlab.polar.cartesian_to_polar``) and take the angular
      nanmean — one radial profile per frame. Result:
      ``{"kind": "waterfall", "data": (n_frames, n_radius) float32,
      "radius": (n_radius,), "entry": str}``.
    - ``"mean_image"``: nan-aware mean of the Cartesian frames in
      ``[frame_lo, frame_hi]`` (inclusive, clamped). Result:
      ``{"kind": "mean_image", "data": (n_qz, n_qxy) float32,
      "q_xy", "q_z", "lo", "hi", "entry"}``.

    Emits ``progress(done, total)`` per frame and
    ``finished(result_dict | None, Exception | None)``.
    """

    finished = Signal(object, object)
    progress = Signal(int, int)

    def __init__(
        self,
        file_path: Path,
        entry: str,
        mode: str,
        frame_lo: int | None = None,
        frame_hi: int | None = None,
        n_radius: int = 512,
        n_angle: int = 256,
    ) -> None:
        super().__init__()
        if mode not in ("waterfall", "mean_image"):
            raise ValueError(f"unknown ScanProfileWorker mode {mode!r}")
        self._file_path = file_path
        self._entry = entry
        self._mode = mode
        self._frame_lo = frame_lo
        self._frame_hi = frame_hi
        self._n_radius = int(n_radius)
        self._n_angle = int(n_angle)

    def run(self) -> None:
        try:
            import warnings

            import h5py

            from mlgidlab.polar import cartesian_to_polar

            with h5py.File(self._file_path, "r") as f:
                data = f[self._entry]["data"]
                signal = data.attrs.get("signal")
                if isinstance(signal, bytes):
                    signal = signal.decode("utf-8", errors="replace")
                ds = data[signal]
                q_xy = np.asarray(data["q_xy"], dtype=float)
                q_z = np.asarray(data["q_z"], dtype=float)
                n = int(ds.shape[0])

                if self._mode == "waterfall":
                    rows = []
                    radius = None
                    for i in range(n):
                        pol = cartesian_to_polar(
                            np.asarray(ds[i], dtype=np.float32),
                            q_xy, q_z,
                            n_radius=self._n_radius,
                            n_angle=self._n_angle,
                        )
                        with warnings.catch_warnings():
                            # All-NaN rows (uncovered polar regions)
                            # are expected — nanmean warns per row.
                            warnings.simplefilter("ignore", RuntimeWarning)
                            rows.append(np.nanmean(pol.image, axis=1))
                        radius = pol.radius
                        self.progress.emit(i + 1, n)
                    result = {
                        "kind": "waterfall",
                        "data": np.vstack(rows).astype(np.float32),
                        "radius": radius,
                        "entry": self._entry,
                    }
                else:
                    lo = 0 if self._frame_lo is None else int(self._frame_lo)
                    hi = n - 1 if self._frame_hi is None else int(self._frame_hi)
                    lo = max(0, min(lo, n - 1))
                    hi = max(lo, min(hi, n - 1))
                    acc = None
                    count = None
                    total = hi - lo + 1
                    for j, i in enumerate(range(lo, hi + 1)):
                        frame = np.asarray(ds[i], dtype=np.float64)
                        finite = np.isfinite(frame)
                        if acc is None:
                            acc = np.zeros(frame.shape, dtype=np.float64)
                            count = np.zeros(frame.shape, dtype=np.int64)
                        acc[finite] += frame[finite]
                        count += finite
                        self.progress.emit(j + 1, total)
                    mean = acc / np.maximum(count, 1)
                    mean[count == 0] = np.nan
                    result = {
                        "kind": "mean_image",
                        "data": mean.astype(np.float32),
                        "q_xy": q_xy,
                        "q_z": q_z,
                        "lo": lo,
                        "hi": hi,
                        "entry": self._entry,
                    }
            self.finished.emit(result, None)
        except Exception as exc:
            logger.debug(
                "suppressed exception in ScanProfileWorker.run",
                exc_info=True,
            )
            self.finished.emit(None, exc)


class PrefetchWorker(QObject):
    """Background filler for the FrameSource's per-frame LRU caches.

    Lives on its own ``QThread``. Owns an **independent**
    ``h5py.File`` handle so cross-thread h5py access doesn't share
    state with the main thread's ``FrameSource``. Reads one
    Cartesian frame, computes its polar resample, and emits both to
    the main thread; the main thread then deposits the result into
    the FrameSource's LRU via ``warm_cartesian`` / ``warm_polar``.

    The worker walks frames ahead of the current play-head using a
    **sliding window**: prefetches up to ``window`` frames forward
    of ``head``, pauses when caught up, restarts when ``head``
    advances. ``window`` is set by the host to ``LRU_size - 1`` so
    the prefetcher never overflows the LRU (which would evict
    frames the play-head still needs to reach).

    The internal QTimer ticks at 15 ms and prefetches **one frame
    per tick** — yields between frames so other Qt work on the
    worker's thread (signal delivery, shutdown) doesn't starve.

    Lifecycle slots (all run on the worker's thread via queued
    connections from the host):

    - ``configure(file_path, entry, n_frames, window)`` — (re)open
      the h5py handle for a new entry. Clears the ``_done`` set.
    - ``update_state(head, active, step)`` — push the current play-
      head, the active flag, and the playback frame-stride. Starts /
      stops the internal timer. ``step > 1`` means the host is
      skipping frames during playback (sub-50 ms requested interval);
      the worker walks the same stride so it doesn't prefetch frames
      the player will skip.
    - ``release()`` — close the h5py handle. Called when MainWindow
      detaches silx for a pipeline run.
    """

    # Emitted on every successful frame prefetch. Args:
    # (frame_idx, cart_frame, polar_frame, polar_radius, polar_angle)
    # The two axis arrays are sent so the host can prime
    # FrameSource.polar_axes() on the first emit without a duplicate
    # resample.
    prefetched = Signal(int, object, object, object, object)
    # Emitted when the worker has nothing left to do for the current
    # window — diagnostic only.
    idle = Signal()

    # How aggressively the worker schedules itself. 15 ms is fast
    # enough that one warm frame queues up nearly every play-tick,
    # but slow enough that the worker can't starve the main thread
    # on slow disks.
    _TICK_INTERVAL_MS = 15

    def __init__(self) -> None:
        super().__init__()
        self._file = None
        self._dataset = None
        self._q_xy: np.ndarray | None = None
        self._q_z: np.ndarray | None = None
        self._n_frames: int = 0
        self._head: int = 0
        self._prev_head: int = 0
        self._window: int = 0
        self._active: bool = False
        # Stride to walk when prefetching. Mirrors the player's
        # frame-step so we don't burn disk + CPU on frames the player
        # will skip. Set via ``update_state``; defaults to 1.
        self._step: int = 1
        # Indices we've already sent to the main thread. Cleared on
        # configure() and whenever the head jumps backwards (user
        # scrubbed back or wrapped to 0).
        self._done: set[int] = set()
        self._timer: QTimer | None = None

    @Slot(str, str, int, int)
    def configure(
        self, file_path: str, entry: str, n_frames: int, window: int,
    ) -> None:
        """(Re)open the h5py handle for a new entry."""
        self._release_file()
        try:
            import h5py
            self._file = h5py.File(file_path, "r")
            self._dataset = self._file[f"{entry}/data/img_gid_q"]
            self._q_xy = np.asarray(
                self._file[f"{entry}/data/q_xy"][()], dtype=float
            )
            self._q_z = np.asarray(
                self._file[f"{entry}/data/q_z"][()], dtype=float
            )
            self._n_frames = int(n_frames)
            self._window = max(0, int(window))
            self._done.clear()
            self._prev_head = self._head
        except Exception:
            # Anything fails → leave the worker in a clean closed
            # state. The host will reconfigure on the next entry
            # load; quietly dropping a bad configure beats raising
            # across a queued connection.
            logger.debug("suppressed exception in PrefetchWorker.configure", exc_info=True)
            self._release_file()
            return
        if self._timer is None:
            self._timer = QTimer()
            self._timer.setInterval(self._TICK_INTERVAL_MS)
            self._timer.timeout.connect(self._tick)
        # update_state controls whether the timer is actually running.

    @Slot(int, bool, int)
    def update_state(self, head: int, active: bool, step: int = 1) -> None:
        """Set the play-head, active flag, and prefetch stride.

        If ``head`` moves backwards by more than one frame (user
        scrubbed back or wrapped), clear ``_done`` so the worker
        re-prefetches the new window. Forward advance by 1 (normal
        playback) doesn't reset — the previous frame's spot in
        ``_done`` is harmless. A ``step`` change also clears
        ``_done`` because the set of "useful" frames changes.

        ``step`` mirrors the host's playback frame-step (1 in normal
        playback, >1 when the host is skipping frames to honour a
        sub-50 ms requested interval).
        """
        head = int(head)
        step = max(1, int(step))
        if head < self._prev_head - 1 or step != self._step:
            self._done.clear()
        self._prev_head = head
        self._head = head
        self._active = bool(active)
        self._step = step
        if self._timer is None:
            return
        if self._active and not self._timer.isActive():
            self._timer.start()
        elif not self._active and self._timer.isActive():
            self._timer.stop()

    @Slot()
    def release(self) -> None:
        """Close the h5py handle. Idempotent."""
        if self._timer is not None and self._timer.isActive():
            self._timer.stop()
        self._release_file()

    def _release_file(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                logger.debug("suppressed exception in PrefetchWorker._release_file", exc_info=True)
                pass
        self._file = None
        self._dataset = None
        self._q_xy = None
        self._q_z = None
        self._n_frames = 0
        self._done.clear()

    def _tick(self) -> None:
        if (
            not self._active
            or self._dataset is None
            or self._n_frames <= 1
            or self._window <= 0
            or self._q_xy is None
            or self._q_z is None
        ):
            return
        head = self._head
        step = max(1, self._step)
        # Window covers ``self._window`` *future stops* of the player,
        # not raw frames — so a step>1 schedule still spans the LRU
        # without overshooting into unreachable territory.
        window_end = min(head + step * self._window + 1, self._n_frames)
        # Find the next frame in the window that hasn't been sent yet.
        # One per tick so other queued slots (update_state, release)
        # get a chance to run.
        for i in range(head + step, window_end, step):
            if i in self._done:
                continue
            try:
                cart = np.asarray(self._dataset[i])
                from mlgidlab.polar import cartesian_to_polar
                pol = cartesian_to_polar(cart, self._q_xy, self._q_z)
            except Exception:
                # File closed under us (silx-dance race), dataset
                # moved, polar resample failed. Bail quietly — the
                # host will reconfigure when the dance settles.
                logger.debug("suppressed exception in PrefetchWorker._tick", exc_info=True)
                return
            self.prefetched.emit(i, cart, pol.image, pol.radius, pol.angle)
            self._done.add(i)
            return
        # No work left in the window — idle until the head advances.
        self.idle.emit()
