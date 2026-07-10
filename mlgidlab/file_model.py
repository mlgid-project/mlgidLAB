"""Pure-Python readers for the mlgidBASE NeXus file format.

Schema reference: project_repos/mlgidBASE/docs/tutorials/output_file_format.md
No Qt imports — keep this module independently testable.
"""
from __future__ import annotations

import csv
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

import logging
logger = logging.getLogger(__name__)

ENTRY_PREFIX = "entry_"
FRAME_KEY_FMT = "frame{:05d}"

# Target RAM ceiling for the per-frame caches owned by FrameSource. Sized
# at ~1 GB combined across Cartesian + polar so a typical 8 GB-RAM laptop
# stays headroom-positive while still keeping scrub instant for a small
# window around the current frame. MIN keeps single-frame scrub from
# falling off the cache; MAX prevents pathological cases where tiny
# frames (<1 MB) would otherwise spawn thousands of LRU entries.
FRAME_LRU_BYTES_TARGET = 1_024 * 1_024 * 1_024
FRAME_LRU_MIN = 4
FRAME_LRU_MAX = 64


def is_entry_group_name(name: str) -> bool:
    """Whether ``name`` looks like a NeXus entry group.

    pygid's default writer uses ``entry_0000``, ``entry_0001``, …, but
    other writers expose a single ``entry`` group (no numeric suffix)
    or use mnemonic suffixes like ``entry_horiz``. Both forms are valid
    NeXus and we shouldn't filter out files just because their entries
    don't follow the numeric-suffix convention.
    """
    return name == "entry" or name.startswith(ENTRY_PREFIX)


def entry_group_names(f: h5py.File) -> list[str]:
    """Top-level NeXus entry-group names, in the file's native order.

    The fast path is the name convention (``entry`` / ``entry_*``), which
    also avoids resolving external links on a master (its linked scans are
    external links that always follow that convention). A top-level group
    whose name does NOT follow the convention is still recognised as an
    entry when it is a LOCAL group (never an external / soft link) carrying
    ``NX_class == 'NXentry'`` -- e.g. a converted file whose single entry is
    named after the sample (mlgidFIT writes ``<sample>`` rather than
    ``entry_0000``). External links are skipped without being resolved, so
    the open-path cost on big masters is unchanged.
    """
    names: list[str] = []
    for name in f.keys():
        if is_entry_group_name(name):
            names.append(name)
            continue
        # Off-convention name: inspect ONLY local groups. getlink does not
        # dereference, so an external/soft link (master scan) is rejected
        # cheaply without opening it.
        if not isinstance(f.get(name, getlink=True), h5py.HardLink):
            continue
        try:
            grp = f[name]
        except Exception:
            continue
        if not isinstance(grp, h5py.Group):
            continue
        nx_class = grp.attrs.get("NX_class")
        if isinstance(nx_class, bytes):
            nx_class = nx_class.decode("utf-8", errors="replace")
        if nx_class == "NXentry":
            names.append(name)
    return names


ANALYSIS_REL = "data/analysis"
IMG_REL = "data/img_gid_q"
QXY_REL = "data/q_xy"
QZ_REL = "data/q_z"

PeakKind = str  # "detected" | "fitted"


@dataclass
class PeakTable:
    """Centers, widths, and ids for a set of peaks at one frame.

    Both Cartesian (q_xy, q_z) and polar (angle, radius) coordinates are kept
    because the peak operations API exposes both.
    """

    q_xy: np.ndarray
    q_z: np.ndarray
    angle: np.ndarray
    radius: np.ndarray
    angle_width: np.ndarray
    radius_width: np.ndarray
    is_ring: np.ndarray
    ids: np.ndarray
    # mlgidDETECT writes the model confidence into ``score``; mlgidFIT
    # copies it onto every fitted row. Manual peaks carry zeros since
    # they have no model provenance. Kept as a parallel array so that
    # rendering helpers (which iterate by index) can stay agnostic.
    score: np.ndarray
    # Peak amplitude (2D-Gaussian peak height). Present on both
    # detected and fitted rows; manual peaks carry zeros.
    amplitude: np.ndarray

    @classmethod
    def from_dataset(cls, ds: h5py.Dataset) -> PeakTable:
        arr = ds[()]
        n = int(arr.shape[0])
        # Tolerate older files written before optional fields
        # existed — return zeros so downstream code can still index.
        score = (
            np.asarray(arr["score"], dtype=float)
            if "score" in arr.dtype.names
            else np.zeros(n, dtype=float)
        )
        amplitude = (
            np.asarray(arr["amplitude"], dtype=float)
            if "amplitude" in arr.dtype.names
            else np.zeros(n, dtype=float)
        )
        return cls(
            q_xy=np.asarray(arr["q_xy"], dtype=float),
            q_z=np.asarray(arr["q_z"], dtype=float),
            angle=np.asarray(arr["angle"], dtype=float),
            radius=np.asarray(arr["radius"], dtype=float),
            angle_width=np.asarray(arr["angle_width"], dtype=float),
            radius_width=np.asarray(arr["radius_width"], dtype=float),
            is_ring=np.asarray(arr["is_ring"], dtype=bool),
            ids=np.asarray(arr["id"], dtype=int),
            score=score,
            amplitude=amplitude,
        )

    def __len__(self) -> int:
        return int(self.ids.size)


@dataclass
class EntryStack:
    """Image stack + reciprocal-space axes for one entry."""

    image_stack: np.ndarray  # shape (n_frames, n_qz, n_qxy)
    q_xy: np.ndarray         # shape (n_qxy,)
    q_z: np.ndarray          # shape (n_qz,)

    @property
    def n_frames(self) -> int:
        return int(self.image_stack.shape[0])


def list_entries(file_path: Path) -> list[str]:
    """Return q-image entry names in sorted order.

    Only ``img_gid_q`` entries are exposed: those are the ones the viewer
    knows how to render and the only ones mlgidDETECT / mlgidFIT actually
    process. Polar entries (``img_gid_pol``) and any non-q sibling groups
    are filtered out so the entry dropdown can't lead the user to an
    entry that would silently no-op a pipeline run or crash the loader.

    The filter mirrors pygid's reading convention: each entry's ``data``
    group has a ``signal`` attribute naming the active image dataset.

    WARNING: this resolves every entry's ``data`` group, which on a master
    that links external scans means opening *every* linked scan file. On
    slow / network storage that is seconds-to-minutes and holds the Python
    GIL while it runs, so it must NOT be called on the open path — use
    ``list_entry_names`` there. Kept for callers that genuinely need the
    q-filter on a local / already-resolved file.
    """
    return [name for name, signal in list_entry_signals(file_path).items()
            if signal == "img_gid_q"]


def list_entry_names(file_path: Path) -> list[str]:
    """Return entry-group link names from the master WITHOUT resolving them.

    h5py's ``keys()`` iterates a group's *links* and does not dereference
    external links, so this reads only the (small) master file: ~1 ms even
    for a 226-scan / 123 GB master, versus the seconds-to-minutes (and
    GIL-held, GUI-freezing) cost of ``list_entries`` / ``list_entry_signals``,
    which open every linked scan to read its ``signal`` attribute. This is
    what lets the open path populate the entry combo + classify the file
    without ever touching the linked scans on either the GUI or the worker
    thread.

    Unlike ``list_entries`` it does NOT filter to ``img_gid_q`` (that check
    needs the link resolved); a non-renderable entry is handled gracefully
    when it is actually selected (``_load_entry_into_viewer`` warns). For a
    pygid master every entry is a q-image scan, so the two agree there.
    """
    with h5py.File(file_path, "r") as f:
        return entry_group_names(f)


def classify_h5_path(file_path: Path) -> str | None:
    """Classify an HDF5 file by content: ``"nexus"``, ``"raw"`` or ``None``.

    NeXus = at least one entry whose ``data`` group has
    ``signal == "img_gid_q"``; raw = any 3-D detector-shaped dataset.
    Both readers open the file, so this is called **off the GUI thread**
    (in ``CopyWorker``) — opening a master that links many external scans
    over slow storage is exactly what froze the open click. Exceptions are
    swallowed so a non-HDF5 / unreadable file returns ``None``.
    """
    try:
        if list_entries(file_path):
            return "nexus"
    except Exception:
        logger.debug("suppressed exception in classify_h5_path (nexus)", exc_info=True)
    try:
        if list_raw_entries(file_path):
            return "raw"
    except Exception:
        logger.debug("suppressed exception in classify_h5_path (raw)", exc_info=True)
    return None


def read_geometry_for_entry(
    file_path: Path, entry: str, frame: int = 0,
) -> dict | None:
    """Read the geometry fields a single-peak 2D fit needs.

    Returns ``{'wavelength_angstrom', 'q_xy_max', 'q_z_max', 'ai_deg',
    'q_z_axis'}`` on success, or ``None`` when any required field is
    missing. The GUI's "Add to fitted" handler uses this to feed
    ``mlgidlab.manual_fit.fit_one_peak`` without dragging in
    ``pygidsim`` (so the headless CI suite can still exercise the
    Add-to-fitted path's fallback branch by patching pygidfit out).

    Wavelength is stored in metres in NeXus; the fitter expects
    Ångströms, so we convert here at the boundary. The angle of
    incidence falls back to ``0.0`` (transmission geometry) when the
    dataset is missing, since pygidfit's default for the same
    parameter is ``0``. ``frame`` selects which entry of a
    per-frame ``angle_of_incidence`` array to read (0-D / length-1
    arrays use index 0 regardless). The q_xy / q_z extents are
    required; ``q_z_axis`` is the 1-D q_z array (returned because
    ``pygidfit.process_scans.img_preprocessing`` masks rows below
    the sample horizon via a q_z comparison).
    """
    try:
        with h5py.File(file_path, "r") as f:
            if entry not in f:
                return None
            grp = f[entry]
            q_xy_arr = np.asarray(grp["data/q_xy"][()])
            q_z_arr = np.asarray(grp["data/q_z"][()])
            q_xy_max = float(np.max(np.abs(q_xy_arr)))
            q_z_max = float(np.max(np.abs(q_z_arr)))
            wl_m = float(
                np.asarray(grp["instrument/monochromator/wavelength"]).ravel()[0]
            )
            if wl_m <= 0:
                return None
            try:
                ai_raw = np.asarray(grp["instrument/angle_of_incidence"]).ravel()
                ai_idx = int(frame) if 0 <= int(frame) < ai_raw.size else 0
                ai_deg = float(ai_raw[ai_idx])
            except Exception:
                ai_deg = 0.0
    except Exception:
        logger.debug("suppressed exception in read_geometry_for_entry", exc_info=True)
        return None
    return {
        "wavelength_angstrom": wl_m * 1e10,
        "q_xy_max": q_xy_max,
        "q_z_max": q_z_max,
        "ai_deg": ai_deg,
        "q_z_axis": q_z_arr,
    }


def count_frames(file_path: Path, entry: str) -> int:
    """Return the number of frames in ``entry`` without loading any data.

    Reads only the shape header of ``<entry>/data/img_gid_q`` (or whatever
    the entry's signal points at) via h5py — sub-millisecond on local
    storage. Returns ``0`` on any error: missing entry, missing /data,
    missing signal, dataset isn't 3-D, or any other shape mismatch. Used
    by ``PipelineWorker`` to size the multi-frame progress bar before
    invoking mlgidBASE.
    """
    try:
        with h5py.File(file_path, "r") as f:
            if entry not in f:
                return 0
            data = f[entry].get("data")
            if not isinstance(data, h5py.Group):
                return 0
            signal = data.attrs.get("signal")
            if isinstance(signal, bytes):
                signal = signal.decode("utf-8", errors="replace")
            if not isinstance(signal, str) or signal not in data:
                return 0
            ds = data[signal]
            if ds.ndim < 1:
                return 0
            return int(ds.shape[0])
    except Exception:
        logger.debug("suppressed exception in count_frames", exc_info=True)
        return 0


# Signal names pygid's writer emits (datasaver key list). A file whose
# entries carry one of these is mlgid NeXus output; an entry_* group with
# a foreign/absent signal is some other instrument's layout (e.g. a LIMA
# detector file's ``entry_0000/measurement/data``) and must classify as
# RAW so the Conversion workflow gets it — feeding it to the NeXus
# loader fails with "component not found".
MLGID_SIGNALS = frozenset({
    "img_gid_q", "img_q", "img_gid_pol", "img_pol",
    "img_gid_pseudopol", "img_pseudopol",
    "rad_cut_gid", "azim_cut_gid", "horiz_cut_gid", "vert_cut_gid",
    "rad_cut", "azim_cut",
})


def classify_entry_data(file_path: Path, entry: str) -> str:
    """Probe ONE entry's layout: ``"mlgid"``, ``"foreign"`` or
    ``"unreadable"``.

    Single-entry counterpart of ``list_entry_signals`` for the open-path
    classifier — it resolves at most this one entry (one external link
    on a master, the same link the prewarm opens right after), never the
    whole file. ``"mlgid"``: the entry has a ``data`` group whose
    ``signal`` is one pygid writes. ``"foreign"``: the entry RESOLVES
    but doesn't have that layout (e.g. a LIMA detector file's
    ``entry_0000/measurement/data``) — proof the file is not mlgid
    NeXus. ``"unreadable"``: the entry itself can't be opened (broken
    external link, offline share) — proof of nothing.
    """
    try:
        with h5py.File(file_path, "r") as f:
            try:
                obj = f[entry]
            except (KeyError, OSError):
                return "unreadable"
            data = obj.get("data") if isinstance(obj, h5py.Group) else None
            if not isinstance(data, h5py.Group):
                return "foreign"
            signal = data.attrs.get("signal")
    except OSError:
        return "unreadable"
    if isinstance(signal, bytes):
        signal = signal.decode("utf-8", errors="replace")
    return "mlgid" if signal in MLGID_SIGNALS else "foreign"


def list_entry_signals(file_path: Path) -> dict[str, str | None]:
    """Return ``{entry_name: signal_or_None}`` for every entry_* group.

    Used by the host to diagnose files that load with no usable entries
    (the entry combo would otherwise look mysteriously empty when the
    file contains only 1D-cut entries or polar-only data).

    Iteration uses the file's native ``f.keys()`` order — for files
    created with ``track_order=True`` (recent pygid conversions, the
    GUI's own appends) this is the original insertion order, so the
    Entry combo lists scans in the order they were collected. Files
    written without ``track_order`` fall back to HDF5's default
    alphanumeric order, which matches the previous behaviour.

    Like ``list_entries`` this reads each entry's ``data`` group and so
    resolves external links — NOT for the open path (use
    ``list_entry_names`` there); this stays for empty-file diagnostics.
    """
    out: dict[str, str | None] = {}
    with h5py.File(file_path, "r") as f:
        for name in entry_group_names(f):
            data = f.get(f"{name}/data")
            if data is None:
                out[name] = None
                continue
            signal = data.attrs.get("signal")
            if isinstance(signal, bytes):
                signal = signal.decode("utf-8", errors="replace")
            out[name] = signal if isinstance(signal, str) else None
    return out


def list_pygid_incompatible_top_level(file_path: Path) -> list[str]:
    """Return top-level group names that would crash ``pygid.NexusFile``.

    pygid's ``NexusFile.read_structure`` iterates *every* top-level
    group with ``for entry in root`` and unconditionally indexes
    ``root[f"/{entry}/data"]`` (see
    ``pygid/nexus_reader.py::get_entry_type``) so HDF5 refuses
    to open with ``KeyError: "object 'data' doesn't exist"`` the
    moment any top-level group lacks a ``data`` child. This trips on:

      - raw detector files whose layout is ``/entry/data0/image``
        rather than ``/entry/data/...`` (the user "opened raw as
        NeXus" mistake), and
      - mixed files where an extra metadata / log / calibration
        group was added next to the entries.

    A group counts as incompatible when either ``/<name>/data`` is
    missing or its ``signal`` attribute names a dataset that isn't
    actually present under ``/<name>/data``. The check mirrors the
    two lookups pygid performs in ``get_entry_type``: the ``/data``
    open and the ``/data/<signal>`` shape probe.
    """
    bad: list[str] = []
    with h5py.File(file_path, "r") as f:
        for name in f.keys():
            obj = f.get(name)
            if not isinstance(obj, h5py.Group):
                bad.append(name)
                continue
            data = obj.get("data")
            if not isinstance(data, h5py.Group):
                bad.append(name)
                continue
            signal = data.attrs.get("signal")
            if isinstance(signal, bytes):
                signal = signal.decode("utf-8", errors="replace")
            if not isinstance(signal, str) or signal not in data:
                bad.append(name)
    return bad


def load_entry(file_path: Path, entry: str) -> EntryStack:
    """Open one entry as a lazily-readable stack.

    Returns an ``EntryStack`` whose ``image_stack`` is a tiny
    ``_LazyImageStack`` wrapper rather than a fully materialised numpy
    array. The wrapper holds a reference to a ``FrameSource`` that owns
    a long-lived ``h5py.File`` handle; per-frame reads happen on
    demand via ``FrameSource.get_cartesian(i)``. Axes ``q_xy`` / ``q_z``
    are still loaded eagerly (small, constant per entry).

    This switches the GUI from "load 100 GB to RAM" to "stream one
    frame at a time" without changing the public ``EntryStack`` shape.
    Callers that previously did ``stack.image_stack[i]`` keep working;
    callers that try to treat ``image_stack`` as a numpy ndarray
    (``.view()``, ``.copy()``, ``np.asarray(stack.image_stack)``) will
    fail loudly — by design, so accidental whole-stack reads surface
    in review rather than silently materialising 100 GB.
    """
    source = FrameSource(file_path=Path(file_path), entry=entry)
    source.acquire()
    return stack_from_source(source)


def stack_from_source(source: "FrameSource") -> EntryStack:
    """Build an ``EntryStack`` around an already-``acquire``d FrameSource.

    Used when the source was opened + warmed off the GUI thread (by
    ``CopyWorker``) so the GUI reuses that open handle and warm caches
    instead of re-opening the file. ``source`` must already be acquired.
    """
    return EntryStack(
        image_stack=_LazyImageStack(source),  # type: ignore[arg-type]  # duck-types as ndarray
        q_xy=source.q_xy,
        q_z=source.q_z,
    )


class FrameSource:
    """Owns one long-lived ``h5py.File`` handle and per-frame LRU caches.

    Backs an entry's image stack without ever pulling the full 3D array
    into memory. Cartesian frames come straight from the h5py dataset
    (chunked reads are cheap because pygid writes with ``chunks=True``).
    Polar frames are computed on demand via
    ``polar.cartesian_to_polar`` and cached separately. Axes
    (``q_xy``, ``q_z``, polar radius / angle) are read once on
    ``acquire()`` and cached as small numpy arrays.

    Lifecycle is coordinated with the silx-detach/reattach dance that
    wraps every file write: ``release()`` closes the handle and clears
    the caches; ``acquire()`` reopens. The viewer calls these via
    ``GIWAXSImageViewer.release_frame_source`` /
    ``acquire_frame_source`` so the same FrameSource instance can
    survive across pipeline runs without rebuilding from scratch.
    """

    def __init__(self, file_path: Path, entry: str) -> None:
        self._file_path = Path(file_path)
        self._entry = entry

        # State populated on acquire(); cleared on release().
        self._file: h5py.File | None = None
        self._dataset: h5py.Dataset | None = None
        self._q_xy: np.ndarray | None = None
        self._q_z: np.ndarray | None = None
        self._shape: tuple[int, int, int] | None = None
        self._dtype: np.dtype | None = None

        # LRU caches. Sized on first acquire once the frame shape is known.
        # OrderedDict gives us move-to-end on hit + popitem(last=False) for
        # eviction, no external dependency.
        self._cart_lru: OrderedDict[int, np.ndarray] = OrderedDict()
        self._polar_lru: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cart_lru_max: int = FRAME_LRU_MIN
        self._polar_lru_max: int = max(FRAME_LRU_MIN // 2, 2)

        # Polar grid axes (computed once on first polar request and reused;
        # they're functions of q_xy / q_z only, constant across frames).
        self._polar_radius: np.ndarray | None = None
        self._polar_angle: np.ndarray | None = None

    # -- Lifecycle ----------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._file is not None

    @property
    def entry(self) -> str:
        return self._entry

    def read_frame_overlays(
        self, frame: int
    ) -> tuple[dict[PeakKind, "PeakTable | None"], list["MatchedStructure"]]:
        """Read one frame's detected/fitted/matched peaks through THIS
        source's already-open handle.

        The handle's master file already has ``entry``'s external link
        resolved (acquire opened it for the image), so reading the tiny
        peak tables here is an in-file navigation — no fresh ``h5py.File``
        open, hence no multi-second network round-trip. The GUI thread uses
        this so entry switches / frame scrubs don't block. Requires the
        source to be open.
        """
        if self._file is None:
            raise RuntimeError("FrameSource not acquired")
        peaks = read_peaks(self._file, self._entry, frame)
        matched = read_matched_peaks(
            self._file, self._entry, frame, peaks.get("fitted")
        )
        return peaks, matched

    def acquire(self) -> None:
        """Open the h5py file and load axes + shape.

        Idempotent: a second acquire on an already-open FrameSource is
        a no-op. Polar axes survive across release/acquire cycles
        because they're derived purely from q_xy / q_z, not the file.
        """
        if self._file is not None:
            return
        self._file = h5py.File(self._file_path, "r")
        try:
            self._dataset = self._file[f"{self._entry}/{IMG_REL}"]
            self._q_xy = np.asarray(
                self._file[f"{self._entry}/{QXY_REL}"][()], dtype=float
            )
            self._q_z = np.asarray(
                self._file[f"{self._entry}/{QZ_REL}"][()], dtype=float
            )
            self._shape = tuple(self._dataset.shape)  # type: ignore[assignment]
            self._dtype = self._dataset.dtype
            # Now that we know the frame size, size the Cartesian LRU so
            # `cart_lru_max * frame_bytes` lands near half the target.
            frame_bytes = int(self._shape[1]) * int(self._shape[2]) * int(
                np.dtype(self._dtype).itemsize
            )
            if frame_bytes > 0:
                half = FRAME_LRU_BYTES_TARGET // 2
                self._cart_lru_max = max(
                    FRAME_LRU_MIN, min(FRAME_LRU_MAX, half // frame_bytes)
                )
                # Polar grid is denser than Cartesian for the typical
                # (n_radius=1024, n_angle=512) defaults (= ``524288``
                # samples per frame, matching pygidfit's pipeline
                # polar grid) — size the polar LRU at half the
                # Cartesian count so total stays under the target.
                self._polar_lru_max = max(
                    FRAME_LRU_MIN // 2, self._cart_lru_max // 2
                )
        except Exception:
            # If anything failed mid-acquire, leave the FrameSource in a
            # clean closed state so a retry doesn't trip on partial state.
            self._file.close()
            self._file = None
            self._dataset = None
            raise

    def release(self) -> None:
        """Close the h5py file and drop all per-frame caches.

        Called by ``GIWAXSImageViewer.release_frame_source`` before any
        write path that needs r+ on the same file (pipeline runs, ROI
        commit, Add-to-fitted, clear-peaks). After release, calls to
        ``get_cartesian`` / ``get_polar`` will lazily re-acquire (via
        the host's reattach pairing); a manual ``acquire()`` is the
        normal way to bring the source back online.
        """
        self._cart_lru.clear()
        self._polar_lru.clear()
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                logger.debug("suppressed exception in FrameSource.release", exc_info=True)
                pass
        self._file = None
        self._dataset = None

    def relocate(self, new_path: Path) -> None:
        """Point the FrameSource at a new filesystem path.

        Used by Save As: the temp file is renamed to match the new
        basename, so the FrameSource's stored ``_file_path`` would
        otherwise become stale and the next ``acquire()`` would fail.
        Caller is responsible for ordering the rename and the
        release/acquire calls correctly — typically
        ``release()`` → rename → ``relocate(new)`` → ``acquire()``.
        """
        self._file_path = Path(new_path)

    # -- Shape + axes -------------------------------------------------------

    @property
    def n_frames(self) -> int:
        if self._shape is None:
            return 0
        return int(self._shape[0])

    @property
    def q_xy(self) -> np.ndarray:
        if self._q_xy is None:
            raise RuntimeError("FrameSource not acquired")
        return self._q_xy

    @property
    def q_z(self) -> np.ndarray:
        if self._q_z is None:
            raise RuntimeError("FrameSource not acquired")
        return self._q_z

    @property
    def frame_shape(self) -> tuple[int, int]:
        """(n_qz, n_qxy) — the shape of one Cartesian frame on disk."""
        if self._shape is None:
            raise RuntimeError("FrameSource not acquired")
        return int(self._shape[1]), int(self._shape[2])

    @property
    def dtype(self) -> np.dtype:
        if self._dtype is None:
            raise RuntimeError("FrameSource not acquired")
        return self._dtype

    @property
    def cart_lru_size(self) -> int:
        return self._cart_lru_max

    @property
    def polar_lru_size(self) -> int:
        return self._polar_lru_max

    # -- Hot path -----------------------------------------------------------

    def get_cartesian(self, i: int) -> np.ndarray:
        """Read frame ``i`` from disk, returning a numpy 2D array.

        LRU-cached: a recently-viewed frame is returned without a disk
        round-trip. Caller receives the cached array reference — do not
        mutate the returned array, the LRU stores it as-is.
        """
        if self._dataset is None:
            raise RuntimeError("FrameSource not acquired")
        if i in self._cart_lru:
            self._cart_lru.move_to_end(i)
            return self._cart_lru[i]
        frame = np.asarray(self._dataset[i])
        self._cart_lru[i] = frame
        while len(self._cart_lru) > self._cart_lru_max:
            self._cart_lru.popitem(last=False)
        return frame

    def get_polar(self, i: int) -> np.ndarray:
        """Polar-resample frame ``i`` on demand.

        The polar grid (radius / angle) is computed once on first call
        and reused for every subsequent frame, since it's derived from
        ``q_xy`` / ``q_z`` only. Per-frame polar arrays are LRU-cached
        independently from the Cartesian cache.
        """
        if self._dataset is None:
            raise RuntimeError("FrameSource not acquired")
        if i in self._polar_lru:
            self._polar_lru.move_to_end(i)
            return self._polar_lru[i]
        # Local import to avoid a circular dep — polar.py imports
        # numpy/scipy only, but importing it at module top would make
        # any test of file_model pull the polar transform too.
        from mlgidlab.polar import cartesian_to_polar

        cart = self.get_cartesian(i)
        polar = cartesian_to_polar(cart, self.q_xy, self.q_z)
        if self._polar_radius is None or self._polar_angle is None:
            self._polar_radius = polar.radius
            self._polar_angle = polar.angle
        self._polar_lru[i] = polar.image
        while len(self._polar_lru) > self._polar_lru_max:
            self._polar_lru.popitem(last=False)
        return polar.image

    def polar_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(radius, angle)`` for the polar grid.

        Triggers a single polar resampling of frame 0 if the axes
        haven't been computed yet — needed by the viewer's polar
        axis labels + cursor lookup.
        """
        if self._polar_radius is None or self._polar_angle is None:
            self.get_polar(0)
        assert self._polar_radius is not None and self._polar_angle is not None
        return self._polar_radius, self._polar_angle

    # -- LRU deposit (used by the background prefetch worker) -------------

    def warm_cartesian(self, i: int, frame: np.ndarray) -> None:
        """Deposit a pre-read Cartesian frame into the LRU.

        Called only from the main thread, via MainWindow's
        prefetched-signal slot. The background prefetch worker has
        its own independent ``h5py.File`` handle and does the disk
        read on a worker thread; this method just deposits the
        result so subsequent ``get_cartesian(i)`` calls hit cache.

        No-op when the FrameSource is currently released (the
        worker's signal might still be in flight after MainWindow
        detached silx for a pipeline run) or when the index is
        out of range (entry shape changed between emit and slot).
        """
        if self._dataset is None:
            return
        if not (0 <= i < self.n_frames):
            return
        self._cart_lru[i] = frame
        self._cart_lru.move_to_end(i)
        while len(self._cart_lru) > self._cart_lru_max:
            self._cart_lru.popitem(last=False)

    def warm_polar(
        self,
        i: int,
        polar_frame: np.ndarray,
        radius: np.ndarray | None = None,
        angle: np.ndarray | None = None,
    ) -> None:
        """Deposit a pre-computed polar frame into the LRU.

        ``radius`` / ``angle`` are stashed on the FrameSource only
        when our own axes haven't been computed yet — that way the
        worker's first emit primes ``polar_axes()`` and saves the
        viewer from a duplicate frame-0 resample when entering
        polar mode for the first time. Subsequent emits ignore the
        axis args (they're identical anyway).
        """
        if self._dataset is None:
            return
        if not (0 <= i < self.n_frames):
            return
        self._polar_lru[i] = polar_frame
        self._polar_lru.move_to_end(i)
        while len(self._polar_lru) > self._polar_lru_max:
            self._polar_lru.popitem(last=False)
        if self._polar_radius is None and radius is not None:
            self._polar_radius = np.asarray(radius)
        if self._polar_angle is None and angle is not None:
            self._polar_angle = np.asarray(angle)


class _LazyImageStack:
    """Thin shim presenting a ``(n_frames, n_qz, n_qxy)`` indexable view.

    Exposes only the surface mlgidLAB actually uses: ``__getitem__``,
    ``shape``, ``ndim``, ``dtype``. Deliberately NOT a numpy ndarray
    subclass and deliberately NOT supporting ``view`` / ``copy`` /
    ``__array__`` — any caller that tries to treat it as a real array
    fails loudly, which catches accidental whole-stack reads in review.
    """

    __slots__ = ("_source",)

    def __init__(self, source: FrameSource) -> None:
        self._source = source

    @property
    def source(self) -> FrameSource:
        return self._source

    @property
    def shape(self) -> tuple[int, int, int]:
        n_qz, n_qxy = self._source.frame_shape
        return (self._source.n_frames, n_qz, n_qxy)

    @property
    def ndim(self) -> int:
        return 3

    @property
    def dtype(self) -> np.dtype:
        return self._source.dtype

    def __len__(self) -> int:
        return self._source.n_frames

    def __getitem__(self, key):
        """Frame index or (frame, row, col) tuple index.

        - ``stack[i]`` returns the 2D Cartesian frame (n_qz, n_qxy).
        - ``stack[frame, r, c]`` returns one pixel — used by the
          cursor-readout's intensity lookup.

        Slices and ellipsis intentionally raise — we want to know
        about callers that try to grab subsets of the stack so they
        can be moved onto a FrameSource access pattern.
        """
        if isinstance(key, (int, np.integer)):
            return self._source.get_cartesian(int(key))
        if isinstance(key, tuple) and len(key) == 3:
            frame, r, c = key
            if (
                isinstance(frame, (int, np.integer))
                and isinstance(r, (int, np.integer))
                and isinstance(c, (int, np.integer))
            ):
                return self._source.get_cartesian(int(frame))[int(r), int(c)]
        raise TypeError(
            f"_LazyImageStack supports stack[i] or stack[frame, r, c] "
            f"only; got {key!r}."
        )


class _LazyPolarStack:
    """``(n_frames, n_radius, n_angle)`` indexable view on polar frames.

    Mirrors ``_LazyImageStack`` but routes to ``FrameSource.get_polar``.
    Supports two indexing forms used by the GUI:

    - ``stack[i]`` — return a polar frame (``profile_viewer.py:341``)
    - ``stack[frame, r, a]`` — single-pixel lookup used by the cursor
      readout (``image_viewer._compute_cursor_info``)
    """

    __slots__ = ("_source",)

    def __init__(self, source: FrameSource) -> None:
        self._source = source

    @property
    def shape(self) -> tuple[int, int, int]:
        radius, angle = self._source.polar_axes()
        return (self._source.n_frames, int(radius.size), int(angle.size))

    @property
    def ndim(self) -> int:
        return 3

    def __len__(self) -> int:
        return self._source.n_frames

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self._source.get_polar(int(key))
        if isinstance(key, tuple) and len(key) == 3:
            frame, r, a = key
            if (
                isinstance(frame, (int, np.integer))
                and isinstance(r, (int, np.integer))
                and isinstance(a, (int, np.integer))
            ):
                return self._source.get_polar(int(frame))[int(r), int(a)]
        raise TypeError(
            f"_LazyPolarStack supports stack[i] or stack[frame, r, a] "
            f"only; got {key!r}."
        )


# Raw-detector dataset enumeration & loading. These are used only by the
# Conversion (raw-mode) workflow — no overlap with the converted-NeXus
# readers above.

# Minimum H/W in pixels for a dataset to count as a detector image.
# Filters out coordinate axes (1D), small lookup tables, and per-frame
# scalar arrays that h5py would otherwise misclassify as 3D.
RAW_MIN_DETECTOR_HW = 32

# Standalone detector-image formats read via ``fabio`` (the same set
# pygid's DataLoader documents). These are NOT HDF5 — they open as raw
# image files, one frame per file, and go through the raw-mode workflow.
FABIO_EXTENSIONS = frozenset({".tif", ".tiff", ".cbf", ".edf"})


def is_fabio_image(path: "Path | str") -> bool:
    """True when ``path`` is a standalone image file we open via fabio."""
    return Path(path).suffix.lower() in FABIO_EXTENSIONS


_fabio_quieted = False


def _quiet_fabio() -> None:
    """Silence fabio's per-read TIFF chatter once (mirrors pygid's DataLoader).

    Many beamline TIFFs trip fabio's "Non standard TIFF" UserWarning on every
    open; without this, scrolling a stack would spam one line per frame.
    Idempotent and lazy so it only runs when images are actually used.
    """
    global _fabio_quieted
    if _fabio_quieted:
        return
    import warnings

    warnings.filterwarnings(
        "ignore", category=UserWarning, module="fabio.TiffIO"
    )
    logging.getLogger("fabio.TiffIO").setLevel(logging.ERROR)
    _fabio_quieted = True


@dataclass
class RawEntry:
    """One raw-detector source: an HDF5 dataset or a fabio image stack.

    ``dataset_path`` is the absolute path inside the HDF5 file (without
    the leading slash) — pygid's ``DataLoader`` accepts this form
    directly as its ``dataset`` kwarg.

    ``frame_map`` distinguishes the two backings and is the single flag
    the rest of the code branches on:

    * ``None`` → an HDF5 detector dataset (read via ``LazyRawStack``,
      the existing path — completely unchanged).
    * set → a fabio image stack (read via ``LazyFabioStack``). Frame
      ``i`` maps to ``frame_map[i] == (file, in_file_index)`` (the index
      is ``0`` for the one-frame-per-file case). ``dataset_path`` is
      ``""`` because pygid ignores it for non-HDF5 files.
    """

    file_path: Path
    dataset_path: str
    shape: tuple[int, int, int]
    dtype: str
    frame_map: list[tuple[Path, int]] | None = None

    @property
    def label(self) -> str:
        """Human-readable identifier for the entry combo / log lines."""
        if self.frame_map is not None:
            n = len(self.frame_map)
            return (
                self.file_path.name
                if n == 1
                else f"{self.file_path.name} (+{n - 1} more)"
            )
        return f"{self.file_path.name}::{self.dataset_path}"


def list_raw_entries(
    file_path: Path,
    progress: "callable | None" = None,
) -> list[RawEntry]:
    """Walk a raw HDF5 file and return every 3D detector-image candidate.

    A dataset qualifies when it is 3D ``(N, H, W)`` with both spatial
    dimensions ≥ ``RAW_MIN_DETECTOR_HW`` and a numeric dtype. Candidates
    keep the FILE's own order (``keys()`` iterates in link-creation
    order for ``track_order`` files — acquisition order on Bliss/pygid
    masters — and in name order otherwise). Sorting alphabetically here
    scrambled beamline scan numbering in every entry list built from
    this walk (``10.1`` sorts before ``2.1``).

    The walk iterates the TOP-LEVEL keys explicitly (resolving each —
    this follows soft/external links one level deep, so beamline masters
    whose scans are external links, e.g. ESRF Bliss ``1.1``/``2.1``
    groups, are discovered) and ``visititems`` within each resolved
    group (``visititems`` itself never follows links). Paths are
    composed master-side (``key/subpath``) so ``load_raw_dataset`` /
    ``LazyRawStack`` can re-resolve them through the master file.
    Broken links are skipped.

    ``progress(done, total)`` is called after each top-level key —
    drives the determinate open bar (a beamtime file can hold hundreds
    of scans, so this walk is the dominant open cost and the one worth
    showing). Reads only metadata (shape + dtype); pixel data is not
    loaded until a frame is actually requested.
    """
    out: list[RawEntry] = []

    def check(ds, path: str) -> None:
        if not isinstance(ds, h5py.Dataset) or ds.ndim != 3:
            return
        n, h, w = ds.shape
        if h < RAW_MIN_DETECTOR_HW or w < RAW_MIN_DETECTOR_HW:
            return
        if ds.dtype.kind not in ("i", "u", "f"):
            return
        out.append(
            RawEntry(
                file_path=file_path,
                dataset_path=path,
                shape=(int(n), int(h), int(w)),
                dtype=str(ds.dtype),
            )
        )

    with h5py.File(file_path, "r") as f:
        keys = list(f.keys())
        total = len(keys)
        for done, key in enumerate(keys, start=1):
            try:
                obj = f[key]
            except (KeyError, OSError):
                logger.debug(
                    "skipping unresolvable top-level link %r in %s",
                    key, file_path, exc_info=True,
                )
                obj = None
            if isinstance(obj, h5py.Dataset):
                check(obj, key)
            elif isinstance(obj, h5py.Group):
                obj.visititems(
                    lambda name, o, _key=key: check(o, f"{_key}/{name}")
                )
            if progress is not None:
                progress(done, total)
    return out


def list_fabio_entries(paths: "list[Path]") -> "list[RawEntry]":
    """Build one ``RawEntry`` PER image file — separate 1-frame entries.

    Multiple images open as independent entries (not a combined stack), so
    each file is selected, viewed and converted on its own. Only the FIRST
    file is read (for ``(H, W)`` + dtype, assumed uniform across the batch)
    so activation stays off-disk on the GUI thread even for a big folder;
    the frame pixels themselves are read lazily by ``LazyFabioStack``.
    ``dataset_path`` is empty — pygid ignores it for non-HDF5 files.
    """
    import fabio  # lazy: fabio is only needed when images are actually used

    _quiet_fabio()
    resolved = [Path(p) for p in paths]
    if not resolved:
        raise ValueError("list_fabio_entries requires at least one path")
    data = np.asarray(fabio.open(str(resolved[0])).data)
    h, w = int(data.shape[-2]), int(data.shape[-1])
    dtype = str(data.dtype)
    return [
        RawEntry(
            file_path=p,
            dataset_path="",
            shape=(1, h, w),
            dtype=dtype,
            frame_map=[(p, 0)],
        )
        for p in resolved
    ]


def load_raw_dataset(entry: RawEntry) -> np.ndarray:
    """Read the full 3D pixel data for one ``RawEntry``.

    Returns a contiguous float32 array; integer detector data is
    upcast so pyqtgraph's intensity scaling and the same percentile-
    based level helpers used for converted data work uniformly.

    Materializes the WHOLE stack — only safe for small datasets (tests,
    conversion previews). The viewer renders raw stacks through
    ``LazyRawStack`` instead: a beamtime raw file can hold a stack of
    tens of GB, and reading it all just to show frame 0 froze the GUI
    for minutes and exhausted RAM.
    """
    with h5py.File(entry.file_path, "r") as f:
        ds = f[entry.dataset_path]
        arr = np.asarray(ds[()])
    if arr.dtype.kind in ("i", "u"):
        arr = arr.astype(np.float32, copy=False)
    return np.ascontiguousarray(arr)


class LazyRawStack:
    """``(N, H, W)`` indexable view over one raw detector dataset.

    Frames are read from disk on demand (small LRU) instead of
    materializing the full 3D stack — the eager ``load_raw_dataset``
    froze the GUI for the duration of a full-stack read on big raw
    files. Mirrors ``_LazyImageStack``'s surface (``shape`` / ``ndim``
    / ``dtype`` / ``len`` / ``stack[i]`` / ``stack[frame, r, c]``) so
    the viewer's raw mode renders it like an ndarray; deliberately not
    an ndarray subclass and no ``__array__`` so accidental whole-stack
    reads fail loudly.

    Owns a persistent read-only h5py handle (``acquire`` / ``release``);
    raw inputs are never written by the GUI, so the handle can stay
    open for the session without the silx detach/reattach dance.
    Frames are upcast to float32 like ``load_raw_dataset`` so contrast
    helpers behave identically.
    """

    _LRU_MAX = 8

    def __init__(self, entry: RawEntry) -> None:
        self._entry = entry
        self._file: h5py.File | None = None
        self._dataset: h5py.Dataset | None = None
        self._lru: OrderedDict[int, np.ndarray] = OrderedDict()

    @property
    def entry(self) -> RawEntry:
        return self._entry

    @property
    def is_open(self) -> bool:
        return self._dataset is not None

    def acquire(self) -> None:
        """Open the file and resolve the dataset path. Idempotent."""
        if self._dataset is not None:
            return
        self._file = h5py.File(self._entry.file_path, "r")
        try:
            self._dataset = self._file[self._entry.dataset_path]
        except Exception:
            self._file.close()
            self._file = None
            raise

    def release(self) -> None:
        """Close the handle and drop cached frames. Idempotent."""
        self._dataset = None
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                logger.debug("suppressed close in LazyRawStack.release", exc_info=True)
            self._file = None
        self._lru.clear()

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._entry.shape

    @property
    def ndim(self) -> int:
        return 3

    @property
    def dtype(self) -> np.dtype:
        # Frames are served upcast — report the served dtype, not the
        # on-disk one, so level/contrast helpers see what they'll get.
        kind = np.dtype(self._entry.dtype).kind
        return np.dtype(np.float32) if kind in ("i", "u") else np.dtype(self._entry.dtype)

    def __len__(self) -> int:
        return self._entry.shape[0]

    def get_frame(self, i: int) -> np.ndarray:
        """One ``(H, W)`` frame, float32-upcast, LRU-cached."""
        if self._dataset is None:
            raise RuntimeError("LazyRawStack not acquired")
        if i in self._lru:
            self._lru.move_to_end(i)
            return self._lru[i]
        arr = np.asarray(self._dataset[i])
        if arr.dtype.kind in ("i", "u"):
            arr = arr.astype(np.float32, copy=False)
        arr = np.ascontiguousarray(arr)
        self._lru[i] = arr
        while len(self._lru) > self._LRU_MAX:
            self._lru.popitem(last=False)
        return arr

    def __array__(self, *args, **kwargs):
        """Refuse whole-stack materialization, loudly.

        Without this, ``np.asarray(stack)`` would silently iterate
        ``stack[0..N-1]`` and read the entire dataset — the exact
        full-stack read this class exists to prevent.
        """
        raise TypeError(
            "LazyRawStack cannot be materialized as a full array; "
            "read frames individually via stack[i]."
        )

    def __getitem__(self, key):
        """``stack[i]`` → 2D frame; ``stack[frame, r, c]`` → one pixel
        (cursor readout). Slices intentionally raise — same contract as
        ``_LazyImageStack``."""
        if isinstance(key, (int, np.integer)):
            return self.get_frame(int(key))
        if isinstance(key, tuple) and len(key) == 3:
            frame, r, c = key
            if (
                isinstance(frame, (int, np.integer))
                and isinstance(r, (int, np.integer))
                and isinstance(c, (int, np.integer))
            ):
                return self.get_frame(int(frame))[int(r), int(c)]
        raise TypeError(
            f"LazyRawStack supports stack[i] or stack[frame, r, c] "
            f"only; got {key!r}."
        )


class LazyFabioStack:
    """``(N, H, W)`` indexable view over a fabio image stack.

    Fabio counterpart of ``LazyRawStack`` with the identical surface
    (``shape`` / ``ndim`` / ``dtype`` / ``len`` / ``stack[i]`` /
    ``stack[frame, r, c]``, ``acquire`` / ``release``, ``__array__``
    raises) so the image viewer and ``EntryLoadWorker`` treat both the
    same. Each frame is a different file, so there is no persistent
    handle — ``acquire`` just flips an open flag to keep the "not
    acquired" guard; frames are read on demand via ``fabio.open`` and
    LRU-cached. Integer frames are upcast to float32 like ``LazyRawStack``.
    """

    _LRU_MAX = 8

    def __init__(self, entry: RawEntry) -> None:
        if entry.frame_map is None:
            raise ValueError("LazyFabioStack requires a fabio RawEntry")
        self._entry = entry
        self._frame_map = entry.frame_map
        self._lru: OrderedDict[int, np.ndarray] = OrderedDict()
        self._open = False

    @property
    def entry(self) -> RawEntry:
        return self._entry

    @property
    def is_open(self) -> bool:
        return self._open

    def acquire(self) -> None:
        """Mark the stack ready to read. Idempotent."""
        self._open = True

    def release(self) -> None:
        """Drop cached frames and mark closed. Idempotent."""
        self._open = False
        self._lru.clear()

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._entry.shape

    @property
    def ndim(self) -> int:
        return 3

    @property
    def dtype(self) -> np.dtype:
        kind = np.dtype(self._entry.dtype).kind
        return np.dtype(np.float32) if kind in ("i", "u") else np.dtype(self._entry.dtype)

    def __len__(self) -> int:
        return self._entry.shape[0]

    def get_frame(self, i: int) -> np.ndarray:
        """One ``(H, W)`` frame, float32-upcast, LRU-cached."""
        import fabio

        if not self._open:
            raise RuntimeError("LazyFabioStack not acquired")
        if i in self._lru:
            self._lru.move_to_end(i)
            return self._lru[i]
        _quiet_fabio()
        path, idx = self._frame_map[i]
        img = fabio.open(str(path))
        arr = np.asarray(img.data if idx == 0 else img.getframe(idx).data)
        if arr.dtype.kind in ("i", "u"):
            arr = arr.astype(np.float32, copy=False)
        arr = np.ascontiguousarray(arr)
        self._lru[i] = arr
        while len(self._lru) > self._LRU_MAX:
            self._lru.popitem(last=False)
        return arr

    def __array__(self, *args, **kwargs):
        raise TypeError(
            "LazyFabioStack cannot be materialized as a full array; "
            "read frames individually via stack[i]."
        )

    def __getitem__(self, key):
        """``stack[i]`` → 2D frame; ``stack[frame, r, c]`` → one pixel."""
        if isinstance(key, (int, np.integer)):
            return self.get_frame(int(key))
        if isinstance(key, tuple) and len(key) == 3:
            frame, r, c = key
            if (
                isinstance(frame, (int, np.integer))
                and isinstance(r, (int, np.integer))
                and isinstance(c, (int, np.integer))
            ):
                return self.get_frame(int(frame))[int(r), int(c)]
        raise TypeError(
            f"LazyFabioStack supports stack[i] or stack[frame, r, c] "
            f"only; got {key!r}."
        )


def build_raw_stack(entry: RawEntry):
    """Return the right lazy stack for ``entry`` (fabio vs HDF5).

    The single switch used by ``EntryLoadWorker.load_raw`` so raw-mode
    stays backing-agnostic. HDF5 entries (``frame_map is None``) keep
    using ``LazyRawStack`` unchanged.
    """
    if entry.frame_map is not None:
        return LazyFabioStack(entry)
    return LazyRawStack(entry)


def _frame_analysis_group(
    f: "h5py.File | h5py.Group", entry: str, frame: int
) -> "h5py.Group | None":
    """Return the per-frame analysis group, tolerating the frame-key naming of
    the current pygid output (``frameNNNNN``) and of older mlgid output
    (``NNNNN``)."""
    base = f"{entry}/{ANALYSIS_REL}"
    if base not in f:
        return None
    analysis = f[base]
    for key in (FRAME_KEY_FMT.format(frame), f"{frame:05d}", str(frame)):
        if key in analysis:
            return analysis[key]
    return None


def _detected_table(group: "h5py.Group") -> "PeakTable | None":
    """Read a frame's detected peaks from either layout: ``detected_peaks`` as
    a structured dataset (current pygid output) or ``detected_peaks/results``
    (older mlgid output)."""
    obj = group.get("detected_peaks")
    if isinstance(obj, h5py.Dataset):
        return PeakTable.from_dataset(obj)
    if isinstance(obj, h5py.Group) and isinstance(obj.get("results"), h5py.Dataset):
        return PeakTable.from_dataset(obj["results"])
    return None


def _fitted_table(group: "h5py.Group") -> "PeakTable | None":
    """Read a frame's fitted peaks from either layout: ``fitted_peaks`` as a
    structured dataset (current pygid output) or a ``fitted_peaks`` group of
    separate 1-D column datasets (older mlgid output)."""
    obj = group.get("fitted_peaks")
    if isinstance(obj, h5py.Dataset):
        return PeakTable.from_dataset(obj)
    if isinstance(obj, h5py.Group):
        return _fitted_table_from_columns(obj)
    return None


def _fitted_table_from_columns(g: "h5py.Group") -> "PeakTable | None":
    """Build a PeakTable from the older mlgid ``fitted_peaks`` group, whose
    fields live in separate 1-D datasets (``peaks_qxy``, ``peaks_qz``,
    ``angle``, ``radius``, …) instead of one structured array. Positions come
    from ``peaks_qxy`` / ``peaks_qz``; ``confidence_level`` maps to ``score``;
    ids are assigned by row order (the columnar layout has no id column)."""
    if not (isinstance(g.get("peaks_qxy"), h5py.Dataset)
            and isinstance(g.get("peaks_qz"), h5py.Dataset)):
        return None
    q_xy = np.asarray(g["peaks_qxy"][()], dtype=float)
    n = q_xy.shape[0]

    def col(name: str, dtype=float) -> np.ndarray:
        ds = g.get(name)
        if isinstance(ds, h5py.Dataset):
            return np.asarray(ds[()], dtype=dtype)
        return np.zeros(n, dtype=dtype)

    return PeakTable(
        q_xy=q_xy,
        q_z=np.asarray(g["peaks_qz"][()], dtype=float),
        angle=col("angle"),
        radius=col("radius"),
        angle_width=col("angle_width"),
        radius_width=col("radius_width"),
        is_ring=col("is_ring", bool),
        ids=np.arange(n, dtype=int),
        score=col("confidence_level"),
        amplitude=col("amplitude"),
    )


def read_peaks(
    f: "h5py.File | h5py.Group", entry: str, frame: int
) -> dict[PeakKind, PeakTable | None]:
    """Read detected/fitted peak tables for one frame from an ALREADY-OPEN
    handle; missing tables → None.

    Takes an open handle so callers that already hold one (the viewer's
    ``FrameSource``, whose master handle has the entry's external link
    resolved) can read the tiny peak tables without a fresh ``h5py.File``
    open. Opening a master that links external scans costs a multi-second
    network round-trip; navigating the already-open handle does not. The
    GUI thread reads peaks through this path so entry switches / frame
    scrubs never block on that open. ``load_peaks`` is the open-by-path
    wrapper for callers without a live handle.

    Tolerates both the current pygid analysis layout and the older mlgid
    layout (see ``_frame_analysis_group`` / ``_detected_table`` /
    ``_fitted_table``).
    """
    out: dict[PeakKind, PeakTable | None] = {"detected": None, "fitted": None}
    group = _frame_analysis_group(f, entry, frame)
    if group is None:
        return out
    out["detected"] = _detected_table(group)
    out["fitted"] = _fitted_table(group)
    return out


def load_peaks(
    file_path: Path, entry: str, frame: int
) -> dict[PeakKind, PeakTable | None]:
    """Open ``file_path`` and read one frame's peak tables (see ``read_peaks``)."""
    with h5py.File(file_path, "r") as f:
        return read_peaks(f, entry, frame)


@dataclass
class MatchedStructure:
    """One matched crystal structure within a frame.

    Each ``matched_*`` NeXus dataset can contain several rows — multi-phase
    solutions — and each row corresponds to one ``MatchedStructure``. The
    ``peaks`` table is a subset of the frame's ``fitted_peaks`` taken at the
    indices listed in ``peak_list``, so the existing peak-rendering helpers
    can draw matched peaks without any special-casing.
    """

    solution_field: str  # e.g. "matched_segments_0000"
    local_idx: int       # row within ``solution_field``
    cif: str             # CIF basename (no extension)
    h: int
    k: int
    l: int
    probability: float
    peaks: PeakTable
    # Indices into the frame's ``fitted_peaks`` table that produced ``peaks``.
    # Kept so matched overlays can be re-derived after an in-memory edit of a
    # fitted peak without a second file read.
    peak_list: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    @property
    def unique_id(self) -> str:
        """Stable handle within one frame."""
        return f"{self.solution_field}/{self.local_idx}"

    @property
    def color_key(self) -> tuple:
        """Cross-frame identity of the structure (CIF + Miller indices).

        Unlike ``unique_id`` (a per-frame dataset position that shifts
        between frames), this is stable across the whole scan, so the
        overlay colour and the show/hide state can be keyed to it — a
        given (CIF, hkl) reads the same colour and honours the same
        checkbox on every frame."""
        return (self.cif, int(self.h), int(self.k), int(self.l))

    @property
    def label(self) -> str:
        ori = "rand" if (self.h, self.k, self.l) == (0, 0, 0) else (
            f"({self.h}{self.k}{self.l})"
        )
        return f"{self.cif} {ori}  p={self.probability:.2f}"


def read_matched_peaks(
    f: "h5py.File | h5py.Group",
    entry: str,
    frame: int,
    fitted_peaks: PeakTable | None,
) -> list[MatchedStructure]:
    """Read all ``matched_*`` solutions for a frame from an ALREADY-OPEN
    handle (the handle-reusing counterpart of ``load_matched_peaks`` — see
    ``read_peaks`` for why the open handle matters)."""
    if fitted_peaks is None or len(fitted_peaks) == 0:
        return []
    n_fit = len(fitted_peaks)
    out: list[MatchedStructure] = []
    group = _frame_analysis_group(f, entry, frame)
    if group is None:
        return out
    for name in sorted(group.keys()):
        if not name.startswith("matched_"):
            continue
        ds = group[name]
        arr = ds[()]
        for i in range(len(arr)):
            cif_raw = arr["CIF"][i]
            cif_str = (
                cif_raw.decode("utf-8", errors="replace")
                if isinstance(cif_raw, bytes)
                else str(cif_raw)
            )
            # Strip the .cif extension for display.
            if cif_str.lower().endswith(".cif"):
                cif_str = cif_str[:-4]
            idx = np.asarray(arr["peak_list"][i], dtype=int)
            # Tolerate stale matches: drop indices that no longer exist
            # in fitted_peaks (e.g. fitting was re-run after matching).
            # Physics-audit F-04 is closed at the *write* side — the
            # ``pipeline.execute`` pre-flight clears matched_* on every
            # entry/frame before run_fitting rewrites fitted_peaks, so
            # this clamp should never fire on files produced by the
            # current GUI. Kept as defence-in-depth for files written
            # by older GUI builds (or external tooling) where matches
            # may still reference stale fitted positions.
            idx = idx[(idx >= 0) & (idx < n_fit)]
            if idx.size == 0:
                continue
            subset = PeakTable(
                q_xy=fitted_peaks.q_xy[idx],
                q_z=fitted_peaks.q_z[idx],
                angle=fitted_peaks.angle[idx],
                radius=fitted_peaks.radius[idx],
                angle_width=fitted_peaks.angle_width[idx],
                radius_width=fitted_peaks.radius_width[idx],
                is_ring=fitted_peaks.is_ring[idx],
                ids=fitted_peaks.ids[idx],
                score=fitted_peaks.score[idx],
                amplitude=fitted_peaks.amplitude[idx],
            )
            out.append(
                MatchedStructure(
                    solution_field=name,
                    local_idx=i,
                    cif=cif_str,
                    h=int(arr["h"][i]),
                    k=int(arr["k"][i]),
                    l=int(arr["l"][i]),
                    probability=float(arr["probability"][i]),
                    peaks=subset,
                    peak_list=idx,
                )
            )
    return out


def load_matched_peaks(
    file_path: Path,
    entry: str,
    frame: int,
    fitted_peaks: PeakTable | None,
) -> list[MatchedStructure]:
    """Open ``file_path`` and read a frame's matched solutions.

    Each row of each ``matched_*`` dataset becomes one ``MatchedStructure``.
    Returns an empty list when the file has no matches for this frame, or
    when ``fitted_peaks`` is None — peak_list indices reference the frame's
    fitted_peaks, so without them the matched peaks have no geometry. See
    ``read_matched_peaks`` for the open-handle variant the viewer uses.
    """
    if fitted_peaks is None or len(fitted_peaks) == 0:
        return []
    with h5py.File(file_path, "r") as f:
        return read_matched_peaks(f, entry, frame, fitted_peaks)


def _add_peak_row(
    file_path: Path,
    entry: str,
    frame: int,
    *,
    dataset_name: str,
    angle: float,
    angle_width: float,
    radius: float,
    radius_width: float,
    amplitude: float = 0.0,
    is_ring: bool = False,
    theta: float = 0.0,
    score: float = 0.0,
    A: float = 0.0,
    B: float = 0.0,
    C: float = 0.0,
    is_cut_qz: bool = False,
    is_cut_qxy: bool = False,
    visibility: int = 0,
) -> int:
    """Append a new row to ``{entry}/analysis/frameNNNNN/{dataset_name}``.

    Shared body for ``add_fitted_peak_row`` and ``add_detected_peak_row`` —
    detected_peaks and fitted_peaks use the same structured dtype, so the
    write path is identical apart from the dataset name. q_xy/q_z are
    recomputed from polar; unset fields default to zero / False; the new
    peak's id is ``max(id)+1`` (or 0 for the first row). Returns the new id.
    """
    frame_key = FRAME_KEY_FMT.format(frame)
    with h5py.File(file_path, "r+") as f:
        ds_path = f"{entry}/{ANALYSIS_REL}/{frame_key}/{dataset_name}"
        if ds_path not in f:
            raise KeyError(
                f"{dataset_name} dataset missing at {ds_path} — run the "
                "appropriate pipeline stage at least once on this frame "
                "before adding rows."
            )
        ds = f[ds_path]
        arr = ds[()]
        new_id = int(arr["id"].max() + 1) if len(arr) > 0 else 0
        # Build the new row by name to stay schema-resilient if the dtype
        # ever gains/loses a field.
        new_row = np.zeros(1, dtype=arr.dtype)
        from mlgidlab.polar import polar_to_qxyz
        q_xy_val, q_z_val = polar_to_qxyz(radius, angle)
        fields = {
            "amplitude": amplitude,
            "angle": angle,
            "angle_width": angle_width,
            "radius": radius,
            "radius_width": radius_width,
            "q_z": q_z_val,
            "q_xy": q_xy_val,
            "theta": theta,
            "score": score,
            "A": A,
            "B": B,
            "C": C,
            "is_ring": is_ring,
            "is_cut_qz": is_cut_qz,
            "is_cut_qxy": is_cut_qxy,
            "visibility": visibility,
            "id": new_id,
        }
        for name, value in fields.items():
            if name in arr.dtype.names:
                new_row[name] = value
        new_arr = np.concatenate([arr, new_row])
        # pygid creates fixed-shape datasets (maxshape == shape), so resize
        # is unavailable — delete and recreate, preserving any attrs.
        attrs = dict(ds.attrs)
        del f[ds_path]
        new_ds = f.create_dataset(ds_path, data=new_arr)
        for k, v in attrs.items():
            new_ds.attrs[k] = v
        return new_id


def add_fitted_peak_row(
    file_path: Path,
    entry: str,
    frame: int,
    *,
    angle: float,
    angle_width: float,
    radius: float,
    radius_width: float,
    amplitude: float,
    is_ring: bool = False,
    theta: float = 0.0,
    score: float = 0.0,
    A: float = 0.0,
    B: float = 0.0,
    C: float = 0.0,
    is_cut_qz: bool = False,
    is_cut_qxy: bool = False,
    visibility: int = 0,
) -> int:
    """Append a new row to the frame's ``fitted_peaks`` dataset.

    Mirrors mlgidbase's row layout (see pygid.datasaver pygid_results_dtype):
    polar geometry + amplitude + 2D-Gaussian shape params (A, B, C, theta)
    + flags + an auto-assigned id. q_xy/q_z are recomputed from polar.

    Caller-supplied fields drive only the meaningful values; the 2D-shape
    params (A/B/C/theta) and score default to zero because a 1D-fit-derived
    row carries no 2D context. Returns the new peak's ``id``.
    """
    return _add_peak_row(
        file_path, entry, frame,
        dataset_name="fitted_peaks",
        angle=angle, angle_width=angle_width,
        radius=radius, radius_width=radius_width,
        amplitude=amplitude, is_ring=is_ring,
        theta=theta, score=score,
        A=A, B=B, C=C,
        is_cut_qz=is_cut_qz, is_cut_qxy=is_cut_qxy,
        visibility=visibility,
    )


def add_detected_peak_row(
    file_path: Path,
    entry: str,
    frame: int,
    *,
    angle: float,
    angle_width: float,
    radius: float,
    radius_width: float,
    score: float = 0.0,
    is_ring: bool = False,
) -> int:
    """Append a new row to the frame's ``detected_peaks`` dataset.

    Used by the copy/paste flow to replay a detected peak on another
    frame without going through mlgidbase's ``add_peak`` (which would
    need the ``pipeline`` extra installed and run async via the
    PipelineWorker queue). The structured dtype is the same as
    fitted_peaks, so we share the write body via ``_add_peak_row``; the
    2D-shape fields (A/B/C/theta) and ``amplitude`` default to zero
    since a detected row carries no fit context.

    Callers pass ``score`` explicitly: paste forwards the source peak's
    confidence verbatim (copy must not change it); the undo re-add
    forwards the snapshot's stored value. The ``0.0`` default is only a
    fallback for callers that have no score to carry.

    Returns the new peak's ``id``.
    """
    return _add_peak_row(
        file_path, entry, frame,
        dataset_name="detected_peaks",
        angle=angle, angle_width=angle_width,
        radius=radius, radius_width=radius_width,
        score=score, is_ring=is_ring,
    )


def normalize_for_pygid(
    file_path: Path, entry: str | None = None
) -> dict[str, list[str]]:
    """Patch entries so pygid's per-frame writers / readers behave.

    With ``entry`` given, only that one entry is normalized; with
    ``entry=None`` every entry is. Scoping to one entry is what keeps a
    huge master file fast to open: reading ``data[signal].shape[0]``
    opens that entry's external-linked scan, so doing it for all 226
    entries up front is what froze the GUI. The GUI now normalizes
    lazily — one entry, just before its first pipeline run (see
    ``MainWindow._ensure_entry_normalized``).

    Two normalizations are applied to the temp copy (the original file
    is never touched):

    1. **0-D ``angle_of_incidence``** → 1-D length-``n_frames`` array.
       ``pygid.nexus_reader.get_ai`` indexes by ``frame_num``; a 0-D
       scalar dataset (some writers do this for single-frame data)
       raises ``IndexError: invalid index to scalar variable`` and
       blocks every pipeline op before it starts.

    2. **Missing per-frame analysis groups** → empty groups created.
       ``pygid.datasaver._save_img_container_detect`` (called from
       ``mlgidbase.save_detect`` / ``save_fit`` / matched-write paths)
       assumes ``data/analysis/frame{N:05d}`` already exists. Files
       written for a single frame only ship with ``frame00000``; the
       per-frame detection / fitting code then errors with
       ``KeyError: 'frameXXXXX' doesn't exist`` when run on any
       higher-numbered frame. We pre-create the missing groups with
       the same NX attrs pygid uses in the bulk save path.

    Returns a dict ``{"angle": [...], "frames": [...]}`` describing
    which entries were patched in each pass (for logging). Idempotent
    on already-normalized files.
    """
    patched_angle: list[str] = []
    patched_frames: list[str] = []
    with h5py.File(file_path, "r+") as f:
        if entry is not None:
            names = [entry] if entry in f else []
        else:
            names = entry_group_names(f)
        for entry_name in names:
            entry = f[entry_name]
            # Look up n_frames from the entry's primary signal dataset.
            data = entry.get("data")
            if data is None:
                continue
            signal = data.attrs.get("signal")
            if isinstance(signal, bytes):
                signal = signal.decode("utf-8", errors="replace")
            if not signal or signal not in data:
                continue
            n_frames = int(data[signal].shape[0])

            # --- 1. angle_of_incidence: scalar → 1-D
            ai_path = "instrument/angle_of_incidence"
            if ai_path in entry:
                ai_ds = entry[ai_path]
                # A length-mismatched 1-D field is intentionally not
                # "fixed" because the per-frame values may legitimately
                # differ (true per-frame metadata) and broadcasting
                # would silently lie.
                if ai_ds.ndim == 0:
                    scalar_value = float(ai_ds[()])
                    new_arr = np.full((n_frames,), scalar_value, dtype=np.float64)
                    attrs = dict(ai_ds.attrs)
                    del entry[ai_path]
                    new_ds = entry.create_dataset(ai_path, data=new_arr)
                    for k, v in attrs.items():
                        new_ds.attrs[k] = v
                    patched_angle.append(entry_name)

            # --- 2. analysis/frameXXXXX groups: pre-create missing ones
            analysis = entry.require_group("data/analysis")
            created_any = False
            for i in range(n_frames):
                group_name = FRAME_KEY_FMT.format(i)
                if group_name not in analysis:
                    g = analysis.create_group(group_name)
                    # Match the NX attrs pygid sets in its bulk save path
                    # so downstream readers don't choke on stricter
                    # NeXus validators.
                    g.attrs["NX_class"] = "NXparameters"
                    g.attrs["EX_required"] = "true"
                    created_any = True
            if created_any:
                patched_frames.append(entry_name)
    return {"angle": patched_angle, "frames": patched_frames}


def read_peak_rows(
    file_path: Path,
    entry: str,
    frame: int,
    kind: str,
    peak_ids: list[int],
) -> list[dict]:
    """Snapshot full field dicts for the given peak ids on one frame.

    Returns one dict per id that exists (in the order of ``peak_ids``),
    carrying every writable column — including the 2D-fit params
    (``theta``/``A``/``B``/``C``/``amplitude``) that ``PeakTable``
    doesn't surface. This lets a delete be undone by re-creating the
    rows verbatim via ``add_detected_peak_row`` / ``add_fitted_peak_row``
    (``id`` and ``q_xy``/``q_z`` are intentionally omitted: the add
    helpers assign a fresh id and recompute the q-fields from polar).
    Ids not present in the dataset are skipped. ``kind`` is
    ``"detected"`` or ``"fitted"``; read-only (mode ``"r"``).
    """
    if kind not in ("detected", "fitted"):
        raise ValueError(
            f"read_peak_rows: kind must be detected or fitted, got {kind!r}"
        )
    ds_name = f"{kind}_peaks"
    frame_key = FRAME_KEY_FMT.format(int(frame))
    wanted = (
        "amplitude", "angle", "angle_width", "radius", "radius_width",
        "theta", "score", "A", "B", "C",
        "is_ring", "is_cut_qz", "is_cut_qxy", "visibility",
    )
    out: list[dict] = []
    with h5py.File(file_path, "r") as f:
        ds_path = f"{entry}/{ANALYSIS_REL}/{frame_key}/{ds_name}"
        if ds_path not in f:
            return out
        arr = f[ds_path][()]
        names = set(arr.dtype.names or ())
        by_id = {int(row["id"]): row for row in arr}
        for pid in peak_ids:
            row = by_id.get(int(pid))
            if row is None:
                continue
            d: dict = {}
            for nm in wanted:
                if nm not in names:
                    continue
                val = row[nm]
                if nm in ("is_ring", "is_cut_qz", "is_cut_qxy"):
                    d[nm] = bool(val)
                elif nm == "visibility":
                    d[nm] = int(val)
                else:
                    d[nm] = float(val)
            out.append(d)
    return out


def delete_peak_row(
    file_path: Path,
    entry: str,
    frame: int,
    kind: str,
    peak_id: int,
) -> int:
    """Delete a single row from one kind's ``<kind>_peaks`` dataset.

    ``kind`` is ``"detected"`` or ``"fitted"``. Removes the row whose
    ``id`` field matches ``peak_id`` and writes the surviving rows
    back; the dataset is recreated at the new length, preserving
    dtype and attrs. Returns the number of rows removed (0 if no
    row had the requested id, in which case the file is untouched).

    **Does not cascade across kinds.** mlgidbase's
    ``_delete_peak_single_frame`` removes the row from detected,
    fitted, and matched_* in one shot (and reindexes ids on each).
    This helper is the opposite: deletes only the kind you ask for,
    leaves the other kinds alone, and **does not reindex**
    ``id`` — the remaining ids stay stable so a delete operation
    can't silently shift external references. Callers that need a
    cascade (e.g. fitted-delete invalidates matched ``peak_list``
    integer indices into ``fitted_peaks``) must invoke
    ``clear_peaks(..., "matched", frame=frame)`` separately.

    ``raise ValueError`` for unsupported ``kind`` (matched can't be
    row-deleted through here because ``matched_*`` lives across
    multiple per-solution datasets; use ``clear_peaks(...,
    "matched", ...)`` to wipe them or write a dedicated helper).
    """
    if kind not in ("detected", "fitted"):
        raise ValueError(
            f"delete_peak_row: kind must be detected or fitted, got {kind!r}"
        )
    ds_name = f"{kind}_peaks"
    frame_key = FRAME_KEY_FMT.format(int(frame))
    with h5py.File(file_path, "r+") as f:
        ds_path = f"{entry}/{ANALYSIS_REL}/{frame_key}/{ds_name}"
        if ds_path not in f:
            return 0
        ds = f[ds_path]
        arr = ds[()]
        keep = arr["id"] != int(peak_id)
        if keep.all():
            return 0
        new_arr = arr[keep]
        # Preserve dtype + attrs by recreating the dataset, mirroring
        # the in-place delete + recreate pattern used by clear_peaks
        # and add_fitted_peak_row. pygid creates these as fixed-shape
        # datasets so .resize is unavailable.
        attrs = dict(ds.attrs)
        parent = ds.parent
        del parent[ds_name]
        new_ds = parent.create_dataset(
            ds_name, data=new_arr,
        )
        for k, v in attrs.items():
            new_ds.attrs[k] = v
        # ``fitted_peaks`` row ordering changed → matched_* peak_list
        # integer indices into this frame's fitted_peaks may now be
        # stale. The caller is expected to invoke
        # ``remap_matched_peak_lists(...)`` (preferred — keeps the
        # surviving structures) or ``clear_peaks(..., "matched",
        # frame=frame)`` after this helper for the fitted case; we don't
        # do it here so the contract stays "this helper touches exactly
        # one kind".
        return int(arr.shape[0] - new_arr.shape[0])


def fitted_positions_for_ids(
    file_path: Path,
    entry: str,
    frame: int,
    peak_ids: list[int],
) -> list[int]:
    """Positional indices of fitted rows whose ``id`` is in ``peak_ids``.

    ``matched_*`` ``peak_list`` entries are **positional** indices into
    the frame's ``fitted_peaks`` (file order — the same order
    ``PeakTable.from_dataset`` / ``load_matched_peaks`` read), NOT peak
    ids. mlgidLAB keeps fitted ids stable across deletes (it does not
    reindex), so id and position diverge. A caller about to delete fitted
    rows must capture their positions **before** deleting so it can hand
    them to ``remap_matched_peak_lists``; this helper returns those
    positions, sorted ascending. Ids not present are skipped.
    """
    wanted = {int(pid) for pid in peak_ids}
    if not wanted:
        return []
    frame_key = FRAME_KEY_FMT.format(int(frame))
    ds_path = f"{entry}/{ANALYSIS_REL}/{frame_key}/fitted_peaks"
    with h5py.File(file_path, "r") as f:
        if ds_path not in f:
            return []
        ids = np.asarray(f[ds_path][()]["id"], dtype=int)
    return [pos for pos, pid in enumerate(ids) if int(pid) in wanted]


def remap_matched_peak_lists(
    file_path: Path,
    entry: str,
    frame: int,
    deleted_positions: list[int],
) -> int:
    """Reindex every ``matched_*`` ``peak_list`` after fitted rows at
    ``deleted_positions`` were removed, keeping the structures.

    ``deleted_positions`` are positional indices (file order) into the
    *pre-delete* ``fitted_peaks``, as returned by
    ``fitted_positions_for_ids``. For each matched solution row's
    ``peak_list`` (a vlen int array of fitted positions):

    - a position equal to a deleted one is dropped (that peak left the
      structure), and
    - a position above a deleted one is shifted down by the count of
      deleted positions below it (the fitted array compacted).

    A solution row is **never removed**, even if its ``peak_list`` becomes
    empty: deleting a fitted peak removes its membership in a structure,
    not the structure itself (an empty one simply stops rendering, since
    ``load_matched_peaks`` skips zero-length lists). Solutions that
    reference none of the deleted peaks and sit entirely below them are
    left byte-identical. Returns the number of solution rows changed.

    Row count per ``matched_*`` dataset is unchanged, so each dataset is
    rewritten in place (``ds[...] = arr``), preserving the vlen dtype +
    attrs without a delete/recreate.
    """
    deleted = sorted({int(p) for p in deleted_positions})
    if not deleted:
        return 0
    deleted_set = set(deleted)

    def _shift(x: int) -> int:
        # New position = old minus the number of deleted slots below it.
        return x - sum(1 for d in deleted if d < x)

    frame_key = FRAME_KEY_FMT.format(int(frame))
    group_path = f"{entry}/{ANALYSIS_REL}/{frame_key}"
    changed = 0
    with h5py.File(file_path, "r+") as f:
        if group_path not in f:
            return 0
        group = f[group_path]
        for name in list(group.keys()):
            if not name.startswith("matched_"):
                continue
            ds = group[name]
            arr = ds[()]
            row_changed = False
            for i in range(len(arr)):
                old = np.asarray(arr["peak_list"][i], dtype=int)
                new = np.asarray(
                    [_shift(int(x)) for x in old if int(x) not in deleted_set],
                    dtype=np.int32,
                )
                if new.size != old.size or (
                    new.size and not np.array_equal(new, old.astype(np.int32))
                ):
                    arr["peak_list"][i] = new
                    row_changed = True
                    changed += 1
            if row_changed:
                ds[...] = arr
    return changed


def clear_peaks(
    file_path: Path,
    entry: str,
    kind: str,
    frame: int | None = None,
) -> int:
    """Empty every ``<kind>_peaks`` dataset under ``entry/data/analysis/``.

    ``kind`` is one of:

    - ``"detected"``   — empties ``detected_peaks`` per frame.
    - ``"fitted"``     — empties ``fitted_peaks`` and ``fitted_peaks_errors``
                         per frame (the two are paired by id).
    - ``"matched"``    — empties every ``matched_*`` dataset per frame.

    ``frame`` restricts the wipe to a single ``frameNNNNN`` group when
    given; the default ``None`` clears every frame in the entry. The
    frame-scoped form is used by the Tools → Reset → Active-frame
    action; pipeline-level cascades pass ``None``.

    Datasets are recreated empty (shape ``(0,)``) preserving dtype + attrs,
    because pygid creates them as fixed-shape datasets so ``.resize`` is
    unavailable. Returns the number of rows removed across all frames.

    Manual peaks live in memory only — clear them via the viewer.
    """
    if kind not in ("detected", "fitted", "matched"):
        raise ValueError(
            f"clear_peaks: kind must be detected/fitted/matched, got {kind!r}"
        )
    removed = 0
    with h5py.File(file_path, "r+") as f:
        ana_path = f"{entry}/{ANALYSIS_REL}"
        if ana_path not in f:
            return 0
        ana_group = f[ana_path]
        if frame is None:
            frame_names = list(ana_group.keys())
        else:
            target = FRAME_KEY_FMT.format(frame)
            frame_names = [target] if target in ana_group else []
        for frame_name in frame_names:
            frame_group = ana_group[frame_name]
            if not isinstance(frame_group, h5py.Group):
                continue
            if kind == "matched":
                for ds_name in list(frame_group.keys()):
                    if ds_name.startswith("matched_"):
                        removed += _empty_dataset_in_place(frame_group, ds_name)
            elif kind == "fitted":
                for ds_name in ("fitted_peaks", "fitted_peaks_errors"):
                    if ds_name in frame_group:
                        removed += _empty_dataset_in_place(frame_group, ds_name)
            else:
                ds_name = "detected_peaks"
                if ds_name in frame_group:
                    removed += _empty_dataset_in_place(frame_group, ds_name)
    return removed


def _iter_frame_keys(
    ana_group: h5py.Group, frame: int | None
) -> Iterator[tuple[str, int]]:
    """Yield ``(frame_group_name, frame_index)`` pairs in the entry.

    When ``frame`` is None every ``frameNNNNN`` group under the entry's
    analysis group is yielded; otherwise just the matching one (if it
    exists). Frame index is parsed from the trailing digits of the
    group name so it matches the value the viewer reports for
    ``current_frame`` and the ``frame_num`` column the rest of the
    pipeline writes into peak rows.
    """
    if frame is None:
        for name in ana_group.keys():
            if not name.startswith("frame"):
                continue
            try:
                idx = int(name[len("frame"):])
            except ValueError:
                continue
            yield name, idx
    else:
        target = FRAME_KEY_FMT.format(frame)
        if target in ana_group:
            yield target, int(frame)


def _csv_value(v):
    """Coerce a structured-array element into something the csv module
    can serialise without surprises (bytes → utf-8 str, numpy scalars
    pass through as their repr)."""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return v


def export_peaks_csv(
    file_path: Path,
    targets: list[tuple[str, int | None]],
    kind: str,
    out_path: Path,
) -> int:
    """Write detected or fitted peaks to ``out_path`` as CSV.

    ``targets`` is a list of ``(entry, frame_or_None)`` tuples; pass
    ``frame=None`` to dump every frame in the entry. ``kind`` is
    ``"detected"`` or ``"fitted"``; for fitted, the ``fitted_peaks_errors``
    sibling dataset is joined per row and its fields appear as
    ``<name>_err`` columns. Returns the row count written.

    The header is the union of every dtype field seen in the scope
    (in first-encounter order), prefixed with ``entry`` and
    ``frame_num``. Missing fields on a row become empty strings —
    the writer uses ``DictWriter`` so heterogeneity across entries
    is tolerated.
    """
    if kind not in ("detected", "fitted"):
        raise ValueError(f"export_peaks_csv: kind must be detected/fitted, got {kind!r}")
    ds_name = "detected_peaks" if kind == "detected" else "fitted_peaks"
    err_ds_name = "fitted_peaks_errors" if kind == "fitted" else None

    rows: list[dict] = []
    fieldnames: list[str] = ["entry", "frame_num"]
    seen: set[str] = set(fieldnames)

    with h5py.File(file_path, "r") as f:
        for entry, frame in targets:
            ana_path = f"{entry}/{ANALYSIS_REL}"
            if ana_path not in f:
                continue
            ana_group = f[ana_path]
            for frame_name, frame_idx in _iter_frame_keys(ana_group, frame):
                fg = ana_group[frame_name]
                if ds_name not in fg:
                    continue
                arr = fg[ds_name][()]
                if len(arr) == 0:
                    continue
                err_arr = None
                if err_ds_name is not None and err_ds_name in fg:
                    err_arr = fg[err_ds_name][()]
                for i in range(len(arr)):
                    row = {"entry": entry, "frame_num": frame_idx}
                    for name in arr.dtype.names:
                        row[name] = _csv_value(arr[name][i])
                        if name not in seen:
                            fieldnames.append(name)
                            seen.add(name)
                    if err_arr is not None and i < len(err_arr):
                        for name in err_arr.dtype.names:
                            col = f"{name}_err"
                            row[col] = _csv_value(err_arr[name][i])
                            if col not in seen:
                                fieldnames.append(col)
                                seen.add(col)
                    rows.append(row)

    with open(out_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def export_matched_csv(
    file_path: Path,
    targets: list[tuple[str, int | None]],
    out_path: Path,
) -> int:
    """Write matched solutions to ``out_path`` as CSV, one row per solution.

    Mirrors what silx shows in its Data tab for the ``matched_*``
    datasets: every solution gets a single row, with its referenced
    fitted peaks compacted into a ``peak_list`` cell formatted as
    ``[id_a, id_b, …]``. CIF is decoded and stripped of its ``.cif``
    suffix to match the Display dock; any extra dtype fields on the
    matched dataset are passed through verbatim so schema additions
    don't silently disappear from the export.

    Returns the row count written.
    """
    rows: list[dict] = []
    fieldnames: list[str] = [
        "entry", "frame_num", "solution_field", "local_idx",
        "cif", "h", "k", "l", "probability", "peak_list",
    ]
    seen: set[str] = set(fieldnames)
    extra_solution_fields: list[str] = []  # any dtype field beyond the well-known ones

    with h5py.File(file_path, "r") as f:
        for entry, frame in targets:
            ana_path = f"{entry}/{ANALYSIS_REL}"
            if ana_path not in f:
                continue
            ana_group = f[ana_path]
            for frame_name, frame_idx in _iter_frame_keys(ana_group, frame):
                fg = ana_group[frame_name]
                for sol_name in sorted(fg.keys()):
                    if not sol_name.startswith("matched_"):
                        continue
                    sol_arr = fg[sol_name][()]
                    if len(sol_arr) == 0:
                        continue
                    sol_dtype_names = sol_arr.dtype.names or ()
                    well_known = {"CIF", "h", "k", "l", "probability", "peak_list"}
                    for name in sol_dtype_names:
                        if name in well_known or name in seen:
                            continue
                        fieldnames.append(name)
                        seen.add(name)
                        extra_solution_fields.append(name)
                    for i in range(len(sol_arr)):
                        cif_raw = sol_arr["CIF"][i]
                        cif_str = (
                            cif_raw.decode("utf-8", errors="replace")
                            if isinstance(cif_raw, bytes) else str(cif_raw)
                        )
                        if cif_str.lower().endswith(".cif"):
                            cif_str = cif_str[:-4]
                        peak_list = np.atleast_1d(
                            np.asarray(sol_arr["peak_list"][i], dtype=int)
                        )
                        # Verbatim list of indices as silx shows them in
                        # the Data tab. Negative padding sentinels are
                        # filtered so the cell shows just the real
                        # peaks; users who need raw values can read the
                        # HDF5 dataset directly.
                        peak_ids = [int(x) for x in peak_list if int(x) >= 0]
                        row = {
                            "entry": entry,
                            "frame_num": frame_idx,
                            "solution_field": sol_name,
                            "local_idx": int(i),
                            "cif": cif_str,
                            "h": int(sol_arr["h"][i]),
                            "k": int(sol_arr["k"][i]),
                            "l": int(sol_arr["l"][i]),
                            "probability": float(sol_arr["probability"][i]),
                            "peak_list": "[" + ", ".join(str(x) for x in peak_ids) + "]",
                        }
                        for name in extra_solution_fields:
                            if name in sol_dtype_names:
                                row[name] = _csv_value(sol_arr[name][i])
                        rows.append(row)

    with open(out_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def _empty_dataset_in_place(parent: h5py.Group, name: str) -> int:
    """Replace ``parent[name]`` with an empty array of the same dtype, keep attrs.

    Returns the row count that was removed (for logging). Used only by
    ``clear_peaks`` — internal helper, not part of the public surface.
    """
    ds = parent[name]
    n = int(ds.shape[0]) if len(ds.shape) > 0 else 0
    attrs = dict(ds.attrs)
    empty = np.zeros(0, dtype=ds.dtype)
    del parent[name]
    new_ds = parent.create_dataset(name, data=empty)
    for k, v in attrs.items():
        new_ds.attrs[k] = v
    return n


def update_peak_row(
    file_path: Path,
    entry: str,
    frame: int,
    kind: str,
    peak_id: int,
    *,
    angle: float,
    angle_width: float,
    radius: float,
    radius_width: float,
) -> None:
    """Mutate one row of detected_peaks/fitted_peaks in place, keyed by `id`.

    The id field is unique within a frame (mlgidbase reindexes on delete), so
    finding a row by id is unambiguous. Polar fields are written verbatim;
    q_xy/q_z are recomputed from the new (radius, angle).

    Raises KeyError when no row with ``peak_id`` exists — the caller should
    treat this as "the file moved on under us" and clear the undo stack.
    """
    if kind not in ("detected", "fitted"):
        raise ValueError(f"update_peak_row only handles detected/fitted, not {kind!r}")
    ds_name = f"{kind}_peaks"
    frame_key = FRAME_KEY_FMT.format(frame)
    with h5py.File(file_path, "r+") as f:
        ds = f[f"{entry}/{ANALYSIS_REL}/{frame_key}/{ds_name}"]
        arr = ds[()]
        matches = np.where(arr["id"] == peak_id)[0]
        if matches.size == 0:
            raise KeyError(
                f"peak_id={peak_id} not found in {entry}/{frame_key}/{ds_name}"
            )
        idx = int(matches[0])
        arr["radius"][idx] = radius
        arr["radius_width"][idx] = radius_width
        arr["angle"][idx] = angle
        arr["angle_width"][idx] = angle_width
        from mlgidlab.polar import polar_to_qxyz
        arr["q_xy"][idx], arr["q_z"][idx] = polar_to_qxyz(radius, angle)
        ds[...] = arr
