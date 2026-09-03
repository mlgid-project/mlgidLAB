"""Lazy wrappers around mlgidbase pipeline stages.

Imports of mlgidbase are deferred so the GUI can run without it installed.
No Qt imports — keep this module independently testable.
"""
from __future__ import annotations

import contextlib

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlgidlab.detection_model import ensure_detection_model, safe_console

import logging
logger = logging.getLogger(__name__)

@dataclass
class PipelineCommand:
    """A single mlgidbase method call."""

    op_name: str  # "run_detection" | "run_fitting" | "run_matching"
    kwargs: dict[str, Any] = field(default_factory=dict)
    #: Optional table-swap plan (``peak_lists.swap_plan``) making the
    #: run read and write a registered "primary" list instead of the
    #: standard tables. Plain data, stamped by the GUI at
    #: ``MainWindow._enqueue_pipeline`` -- the worker has no business
    #: reading QSettings, which is also why the plan travels here
    #: rather than being recomputed on this side.
    table_swap: dict | None = None


# --- Primary-table swapping ------------------------------------------------
#
# mlgidbase writes fixed dataset names (``detected_peaks``,
# ``fitted_peaks`` + ``fitted_peaks_errors``, resolved deep inside
# pygid's ``_save_img_container_*``) and takes no output-name argument,
# so redirecting a run has to happen around the call: park the standard
# tables, move the user's primary tables into their place, run, then
# put everything back. Every step is an HDF5 link rename, so the cost
# is independent of how many peaks or frames are involved.
#
# Own copy of the prefix rather than importing ``peak_lists``: that
# module needs QSettings, and this one stays Qt-free so it can be
# tested (and CI-run) without a GUI. ``test_pipeline_primary_table``
# pins the two equal.

_STASH_PREFIX = "__mlgidlab_stash__"
#: Where the swapped-in data goes when a stash is found with no marker
#: to say what it belongs to. Never overwritten, never deleted: the
#: sweep's job is to make the standard tables right again without
#: destroying whatever the interrupted run had put in their place.
_ORPHAN_PREFIX = "__mlgidlab_orphan__"
#: In-entry, NOT top-level: ``pygid.NexusFile`` opens ``/<name>/data``
#: for every root key, so a stray top-level group is exactly the
#: failure ``execute``'s first pre-flight exists to catch.
_SWAP_MARKER_REL = "process/mlgidlab"
_SWAP_MARKER_NAME = "pipeline_swap"


def _frame_group_names(f, entry: str, frame: Any) -> list[str]:
    """The analysis frame groups a run covers.

    ``frame is None`` means the whole entry, matching mlgidbase's own
    ``frame_num=None``. Listed live rather than cached because
    detection CREATES frame groups: the post-run pass must see groups
    that did not exist when the pre-run pass looked.
    """
    import h5py

    from mlgidlab.file_model import ANALYSIS_REL, FRAME_KEY_FMT

    ana_path = f"{entry}/{ANALYSIS_REL}"
    if ana_path not in f:
        return []
    ana = f[ana_path]
    if frame is None:
        return [
            name for name in ana.keys()
            if isinstance(ana.get(name), h5py.Group)
        ]
    key = FRAME_KEY_FMT.format(int(frame))
    return [key] if key in ana else []


def _move(group, src: str, dst: str) -> bool:
    """Rename ``src`` to ``dst`` inside ``group``; True when it moved.

    A missing source is normal, not an error: a primary table need not
    exist before its first run, and an entry need not have been
    detected or fitted yet either.
    """
    if src not in group:
        return False
    if dst in group:
        del group[dst]
    group.move(src, dst)
    return True


def _swap_in(file_path: Path, entry: str, frame: Any, plan: dict) -> None:
    """Park the standard tables and move the primary ones into place.

    The marker is written FIRST, in the same open: if the process dies
    between here and the restore (a detection run can take the GPU down
    with it), the marker is the only thing that says which name the
    swapped-in data belongs back under.
    """
    import json

    import h5py

    from mlgidlab.file_model import ANALYSIS_REL

    with h5py.File(file_path, "r+") as f:
        marker_group = f.require_group(f"{entry}/{_SWAP_MARKER_REL}")
        if _SWAP_MARKER_NAME in marker_group:
            del marker_group[_SWAP_MARKER_NAME]
        marker_group.create_dataset(
            _SWAP_MARKER_NAME,
            data=json.dumps({"entry": entry, "frame": frame, "plan": plan}),
        )
        for name in _frame_group_names(f, entry, frame):
            group = f[f"{entry}/{ANALYSIS_REL}/{name}"]
            for standard in plan.get("stash", ()):
                _move(group, standard, f"{_STASH_PREFIX}{standard}")
            for own, standard in plan.get("swap_in", ()):
                _move(group, own, standard)


def _swap_out(file_path: Path, entry: str, frame: Any, plan: dict) -> None:
    """Capture the run's output under the primary names, then restore.

    Order matters: capture first, unstash second. The standard name
    holds the run's result at this point, and the stash holds what was
    there before; doing it the other way round would overwrite the
    result with the old table.
    """
    import h5py

    from mlgidlab.file_model import ANALYSIS_REL

    with h5py.File(file_path, "r+") as f:
        for name in _frame_group_names(f, entry, frame):
            group = f[f"{entry}/{ANALYSIS_REL}/{name}"]
            for standard, own in plan.get("capture", ()):
                _move(group, standard, own)
            for standard in plan.get("stash", ()):
                _move(group, f"{_STASH_PREFIX}{standard}", standard)
        marker_path = f"{entry}/{_SWAP_MARKER_REL}/{_SWAP_MARKER_NAME}"
        if marker_path in f:
            del f[marker_path]


@contextlib.contextmanager
def _swapped_tables(file_path: Path, entry: Any, frame: Any, plan: dict | None):
    """Run the body with the primary tables standing in for the standard ones.

    A no-op without a plan, so the ordinary pipeline path does not even
    open the file. ``finally`` covers an exception; a hard kill is what
    the marker and ``recover_swaps`` are for.
    """
    if not plan or not isinstance(entry, str) or not entry:
        yield
        return
    _swap_in(Path(file_path), entry, frame, plan)
    try:
        yield
    finally:
        _swap_out(Path(file_path), entry, frame, plan)


def recover_swaps(file_path: Path) -> int:
    """Undo any table swap a previous run left behind. Returns entries fixed.

    Two levels, because the two failure shapes differ:

    * A **marker** says exactly what was swapped, so the restore is the
      same ``_swap_out`` the successful path runs.
    * A **stash with no marker** (hand-edited file, marker lost) still
      identifies the standard table unambiguously by name. The
      swapped-in data sitting under that name is moved aside to
      ``__mlgidlab_orphan__`` rather than deleted -- the sweep's job is
      to make the standard tables right, never to destroy whatever the
      interrupted run had in their place.
    """
    import json

    import h5py

    from mlgidlab.file_model import ANALYSIS_REL

    fixed = 0
    try:
        with h5py.File(file_path, "r+") as f:
            entries = list(f.keys())
    except Exception:
        logger.debug("suppressed open in recover_swaps", exc_info=True)
        return 0

    for entry in entries:
        plan = None
        frame = None
        try:
            with h5py.File(file_path, "r") as f:
                marker_path = f"{entry}/{_SWAP_MARKER_REL}/{_SWAP_MARKER_NAME}"
                if marker_path in f:
                    raw = f[marker_path][()]
                    payload = json.loads(
                        raw.decode() if isinstance(raw, bytes) else str(raw)
                    )
                    plan = payload.get("plan")
                    frame = payload.get("frame")
        except Exception:
            logger.debug("suppressed marker read in recover_swaps", exc_info=True)
        if plan:
            try:
                _swap_out(Path(file_path), entry, frame, plan)
                logger.warning(
                    "recovered an interrupted pipeline table swap in %s/%s "
                    "— the standard peak tables have been restored.",
                    Path(file_path).name, entry,
                )
                fixed += 1
                continue
            except Exception:
                logger.debug("suppressed marker restore", exc_info=True)

        # No usable marker: fall back to the stash names themselves.
        try:
            with h5py.File(file_path, "r+") as f:
                ana_path = f"{entry}/{ANALYSIS_REL}"
                if ana_path not in f:
                    continue
                ana = f[ana_path]
                touched = False
                for name in list(ana.keys()):
                    group = ana.get(name)
                    if not isinstance(group, h5py.Group):
                        continue
                    for key in list(group.keys()):
                        if not key.startswith(_STASH_PREFIX):
                            continue
                        standard = key[len(_STASH_PREFIX):]
                        _move(group, standard, f"{_ORPHAN_PREFIX}{standard}")
                        _move(group, key, standard)
                        touched = True
                if touched:
                    logger.warning(
                        "found parked peak tables with no swap marker in "
                        "%s/%s — restored the standard tables; the "
                        "interrupted run's output is under %s.",
                        Path(file_path).name, entry, _ORPHAN_PREFIX,
                    )
                    fixed += 1
        except Exception:
            logger.debug("suppressed stash sweep in recover_swaps", exc_info=True)
    return fixed


def is_mlgidbase_available() -> bool:
    """True when the pipeline backend imports.

    The real error is logged so "backend unavailable" is diagnosable
    from the log. (A torch-first import retry lived here while the
    stack floated on onnxruntime-gpu >= 1.27, which hard-links
    libcudart.so.13 at import time; mlgiddetect >= 0.2.7 pins the
    lazily-importing 1.26.0 line, so the retry is gone.)
    """
    try:
        import mlgidbase  # noqa: F401
        return True
    except ImportError as exc:
        logger.warning(
            "pipeline backend unavailable — import mlgidbase "
            "failed: %s", exc,
        )
        return False


_ONNX_GPU_PRELOADED = False


@contextlib.contextmanager
def detection_on_gpu():
    """Preload onnxruntime's CUDA libraries before a detection run.

    mlgiddetect >= 0.2.7 selects CPU vs GPU natively: it gates on the
    ORT CUDA provider + a driver probe (never on torch) and falls back
    to CPU if the CUDA EP cannot initialize; ``MODEL_FORCE_CPU`` in the
    detection config forces CPU. (A ``torch.cuda.is_available``
    monkeypatch lived here for mlgiddetect <= 0.2.3, which wrongly
    gated GPU detection on torch's CUDA build.)

    What remains: ``onnxruntime.preload_dlls()`` puts the pip-installed
    ``nvidia-*`` CUDA/cuDNN wheels on the loader path so the CUDA EP
    can initialize without an ``LD_LIBRARY_PATH`` shim. run_matching
    keeps its own (torch-based) device path, which still needs a real
    CUDA-torch build to use the GPU.
    """
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            yield
            return
    except Exception:
        yield
        return

    global _ONNX_GPU_PRELOADED
    if not _ONNX_GPU_PRELOADED:
        try:
            ort.preload_dlls()  # onnxruntime >= 1.21; loads bundled nvidia libs
        except Exception:
            logger.debug("onnxruntime.preload_dlls() unavailable/failed", exc_info=True)
        _ONNX_GPU_PRELOADED = True
    yield


def _finite_or(value, fallback: float) -> float:
    """``float(value)`` when finite, else ``fallback`` — ring fits leave
    angular shape coefficients NaN/inf, which must not land in stored
    rows that downstream readers treat as numbers."""
    import numpy as np

    v = float(value)
    return v if np.isfinite(v) else float(fallback)


# Plausibility gate for injected-box fits: a fit is only kept when its
# fitted width stays within this factor of the injected box width (a
# blow-up means pygidfit modelled the local background, not a peak).
INJECT_MAX_WIDTH_RATIO = 2.0


def _implausible_fit(fit, spec: dict, is_ring: bool) -> str | None:
    """Why a single-box fit cannot be the peak the box predicted.

    Injected boxes assert "a peak belongs HERE" (a track gap or a
    forward-simulated reflection); pygidfit happily returns a result
    even when the box only contains background, and those fits are
    what pollute the file with unrealistic peaks. Reject the fit when:

    * the amplitude is non-positive/non-finite (nothing was there);
    * the fitted centre left the injected box (the fit locked onto a
      different feature — and a peak more than half a box-width off
      would also fall below the IoM coverage threshold, so the same
      reflection would be injected AGAIN on the next run);
    * a fitted width is non-positive or more than
      ``INJECT_MAX_WIDTH_RATIO`` times the box width (background
      gradient, not a peak).

    Rings are judged radially only. Returns a human-readable reason,
    or None when the fit is plausible.
    """
    import numpy as np

    amp = float(fit.amplitude)
    if not np.isfinite(amp) or amp <= 0:
        return "non-positive amplitude"
    rw = abs(float(spec["radius_width"]))
    if (
        not np.isfinite(float(fit.radius))
        or abs(float(fit.radius) - float(spec["radius"])) > rw / 2
    ):
        return "fitted centre left the box radially"
    frw = float(fit.radius_width)
    if not np.isfinite(frw) or frw <= 0 or frw > INJECT_MAX_WIDTH_RATIO * rw:
        return f"implausible radial width ({frw:g} vs box {rw:g})"
    if not is_ring:
        aw = abs(float(spec["angle_width"]))
        if (
            not np.isfinite(float(fit.angle))
            or abs(float(fit.angle) - float(spec["angle"])) > aw / 2
        ):
            return "fitted centre left the box in angle"
        faw = float(fit.angle_width)
        # Non-finite angular width is tolerated (substituted with the
        # box width on persist — long-standing behaviour for spots
        # whose angular profile degenerates).
        if np.isfinite(faw) and (
            faw <= 0 or faw > INJECT_MAX_WIDTH_RATIO * aw
        ):
            return f"implausible angular width ({faw:g} vs box {aw:g})"
    return None


def _execute_interpolate_tracks(file_path: Path, kwargs: dict) -> list:
    """Fill tracked-peak GAP frames with real detected + fitted rows.

    Thin wrapper around ``_execute_peak_injection`` kept so log lines
    and the ``execute`` dispatch keep reading "interpolate_tracks".
    Specs come from ``phase_tracking.plan_gap_fills`` and carry a
    ``track`` field, which is echoed into each result record — the
    host appends the fills as new track members.
    """
    return _execute_peak_injection(
        file_path, kwargs, op_label="interpolate_tracks"
    )


def _execute_peak_injection(
    file_path: Path, kwargs: dict, op_label: str
) -> list:
    """Inject detected boxes, 2D-fit each, persist the fitted rows.

    Shared executor behind ``interpolate_tracks`` (gap-filling planned
    by ``phase_tracking.plan_gap_fills``; specs carry a ``track``) and
    ``inject_fitted_peaks`` (predicted reflections selected on the
    "Expected pattern" overlay; no ``track``). ``kwargs["plan"]`` =
    ``{frame: [{"radius", "angle", "radius_width", "angle_width",
    "is_ring"[, "track"]}, ...]}``.

    Per spec: append a detected box at the given position/size, run the
    single-box 2D fit (``manual_fit.fit_one_peak`` — byte-identical to
    what the pipeline ``run_fitting`` writes for the same box), and
    persist the fitted row. Ring specs carry a finite quadrant-spanning
    box (pygidfit classifies rings geometrically and runs its radial
    ring fit on them); the persisted ring row uses mlgidLAB's ring
    convention (``angle 45 / angle_width inf / is_ring``). The host
    chains ``run_matching`` command(s) over the affected frames
    afterwards — segments and rings separately — so the new peaks join
    each frame's matched solutions.

    The injected detected row carries ``score=1.0`` — the box exists on
    the strength of its provenance (a track spanning the gap, or a
    forward-simulated reflection the user vetted), not an mlgidDETECT
    confidence, and must not vanish under the Display dock's min-score
    slider.

    Returns ``[{"frame", "detected_id", "fitted_id", "is_ring"
    [, "track"]}, ...]`` for the boxes whose fit succeeded (``track``
    present iff the spec carried it). A box whose fit fails is skipped
    and its injected detected row deleted again (no orphan geometry in
    the file); the skip is logged. Frames with missing geometry /
    datasets are skipped whole. ``kwargs["fit_params"]`` carries the
    Pipeline panel's fit config so the fills use the SAME settings the
    next ``run_fitting`` would.

    Emits one ``Saved fitted peaks … frame: N`` log line per completed
    frame — the exact shape ``workers._FrameProgressHandler`` counts,
    so the host's progress bar ticks per frame.
    """
    import logging as _logging

    import h5py
    import numpy as np

    from mlgidlab import file_model
    from mlgidlab.manual_fit import ManualFitError, fit_one_peak

    log = _logging.getLogger("mlgidBASE")
    entry = str(kwargs.get("entry") or "")
    plan = kwargs.get("plan") or {}
    fit_params = dict(kwargs.get("fit_params") or {})
    # The GUI reads the fitted/detected link setting and passes the
    # answer in; a worker has no business touching QSettings. Each spec
    # here writes a detected box and its fit as a pair, so when the link
    # is on the fit simply takes the detected row's id.
    link_ids = bool(kwargs.get("link_ids", False))
    if not entry or not plan:
        return []

    def _src(spec: dict) -> str:
        return (
            f"track {spec['track']}" if "track" in spec
            else "predicted box"
        )

    results: list = []
    for frame in sorted(plan):
        specs = plan[frame]
        frame = int(frame)
        if not specs:
            continue
        geom = file_model.read_geometry_for_entry(
            file_path, entry, frame=frame
        )
        if geom is None:
            log.warning(
                "%s: missing instrument geometry for "
                "entry %s — skipping frame %d", op_label, entry, frame,
            )
            continue
        geom = dict(geom)
        geom.pop("q_z_axis", None)
        try:
            with h5py.File(file_path, "r") as f:
                grp = f[entry]
                cart = np.asarray(grp[file_model.IMG_REL][frame])
                q_xy = np.asarray(
                    grp[file_model.QXY_REL][()], dtype=float
                )
                q_z = np.asarray(grp[file_model.QZ_REL][()], dtype=float)
        except Exception as exc:
            log.warning(
                "%s: could not read frame %d of %s: %s",
                op_label, frame, entry, exc,
            )
            continue
        for spec in specs:
            is_ring = bool(spec.get("is_ring", False))
            try:
                # Ring boxes are stored with their FINITE quadrant-
                # spanning geometry (pygidfit classifies rings
                # geometrically and an inf width breaks its grid math
                # on a later whole-frame refit); the is_ring flag makes
                # the viewer draw the box as a ring, not a segment.
                det_id = int(file_model.add_detected_peak_row(
                    file_path, entry, frame,
                    angle=float(spec["angle"]),
                    angle_width=float(spec["angle_width"]),
                    radius=float(spec["radius"]),
                    radius_width=float(spec["radius_width"]),
                    score=1.0, is_ring=is_ring,
                ))
            except Exception as exc:
                log.warning(
                    "%s: could not inject a detected box "
                    "on frame %d (%s): %s",
                    op_label, frame, _src(spec), exc,
                )
                continue
            try:
                fit = fit_one_peak(
                    cart, q_xy, q_z,
                    radius=float(spec["radius"]),
                    radius_width=float(spec["radius_width"]),
                    angle=float(spec["angle"]),
                    angle_width=float(spec["angle_width"]),
                    crit_angle=float(fit_params.get("crit_angle", 0.0)),
                    theta_fixed=bool(fit_params.get("theta_fixed", True)),
                    clustering_distance_peaks=float(
                        fit_params.get("clustering_distance_peaks", 10.0)
                    ),
                    clustering_distance_rings=float(
                        fit_params.get("clustering_distance_rings", 10.0)
                    ),
                    clustering_extend=int(
                        fit_params.get("clustering_extend", 2)
                    ),
                    **geom,
                )
            except ManualFitError as exc:
                # No orphan geometry: the injected detected box exists
                # only to be fitted, so remove it again on failure.
                try:
                    file_model.delete_peak_row(
                        file_path, entry, frame, "detected", det_id
                    )
                except Exception:
                    logger.debug(
                        "suppressed exception cleaning up an injected "
                        "detected box", exc_info=True,
                    )
                log.warning(
                    "%s: 2D fit failed on frame %d "
                    "(%s) — box skipped: %s",
                    op_label, frame, _src(spec), exc,
                )
                continue
            problem = _implausible_fit(fit, spec, is_ring)
            if problem is not None:
                # The fit converged onto something that cannot be the
                # predicted peak — treat it like a failed fit so no
                # unrealistic rows reach the file or the matcher.
                try:
                    file_model.delete_peak_row(
                        file_path, entry, frame, "detected", det_id
                    )
                except Exception:
                    logger.debug(
                        "suppressed exception cleaning up an injected "
                        "detected box", exc_info=True,
                    )
                log.warning(
                    "%s: implausible fit on frame %d "
                    "(%s) — box skipped: %s",
                    op_label, frame, _src(spec), problem,
                )
                continue
            # Persist. A ring fit returns angle = NaN / angle_width =
            # inf (pygidfit fits rings radially); store mlgidLAB's ring
            # row convention (angle 45, angle_width inf, is_ring) so
            # the row reads like every other ring in the file. Non-
            # finite shape coefficients collapse to 0.0.
            fit_angle = float(fit.angle)
            fit_aw = float(fit.angle_width)
            if is_ring or not np.isfinite(fit_angle):
                fit_angle = 45.0
            if is_ring:
                fit_aw = float("inf")
            elif not np.isfinite(fit_aw):
                fit_aw = float(spec["angle_width"])
            new_fid = int(file_model.add_fitted_peak_row(
                file_path, entry, frame,
                peak_id=det_id if link_ids else None,
                radius=fit.radius, radius_width=fit.radius_width,
                angle=fit_angle, angle_width=fit_aw,
                amplitude=fit.amplitude, is_ring=is_ring,
                theta=_finite_or(fit.theta, 0.0),
                A=_finite_or(fit.A, 0.0),
                B=_finite_or(fit.B, 0.0),
                C=_finite_or(fit.C, 0.0),
            ))
            record = {
                "frame": frame,
                "detected_id": det_id,
                "fitted_id": new_fid,
                "is_ring": is_ring,
            }
            # Gap-fill specs carry their origin track; predicted-box
            # specs don't (there is no track to append the fill to).
            if "track" in spec:
                record["track"] = int(spec["track"])
            results.append(record)
        # One per-frame completion line, in the exact shape
        # _FrameProgressHandler counts for the progress bar.
        log.info(
            f"Saved fitted peaks to file: {Path(file_path).name}, "
            f"entry: {entry}, frame: {frame}"
        )
    return results


def execute(file_path: Path, command: PipelineCommand) -> Any:
    """Run one pipeline command on a NeXus file. Lazily imports mlgidbase.

    Import order matters: the ``from mlgidbase import mlgidBASE`` line
    is intentionally deferred to **after** the file-shape pre-flights
    below so the headless CI suite (which does not ship the private
    ``mlgidbase`` / ``pygidsim`` backends) can still exercise those
    pre-flights against synthetic NeXus files. The pre-flights raise
    actionable ``RuntimeError``s the tests assert on; pulling
    ``mlgidbase`` earlier would short-circuit with
    ``ModuleNotFoundError`` before any of them runs.
    """
    import logging

    # Pre-flight: refuse to invoke mlgidBASE when the file has a
    # top-level group pygid can't handle. ``pygid.NexusFile`` iterates
    # every root key and unconditionally opens ``/<name>/data``, so a
    # single raw-style or stray-metadata group at the top level brings
    # down the whole open with an opaque ``KeyError: "object 'data'
    # doesn't exist"`` deep inside h5py. Surface a clear, actionable
    # error instead — naming the offending groups — so the user can
    # remove / rename them rather than chasing the h5py stack.
    from mlgidlab.file_model import list_entries, list_pygid_incompatible_top_level

    # An earlier run that died mid-swap (a hard kill, or the GPU taking
    # the process with it) leaves the primary tables sitting under the
    # standard names. Put them back before doing anything else: running
    # on top of a swapped file would compound the damage, and every
    # later read would show the wrong table as "detected_peaks".
    recover_swaps(file_path)

    bad = list_pygid_incompatible_top_level(file_path)
    if bad:
        raise RuntimeError(
            f"Cannot run {command.op_name!r}: {Path(file_path).name} "
            f"contains top-level group(s) that pygid cannot read — "
            f"each entry must expose a /data subgroup with a valid "
            f"'signal' attribute. Offending: "
            f"{', '.join(repr(n) for n in bad)}. Remove or rename "
            f"them, or open the source raw file via the Conversion "
            f"workflow to produce a proper NeXus output."
        )

    # Pre-flight: if the caller pinned an entry, make sure it's a
    # 2D ``img_gid_q`` entry the file actually carries. Otherwise
    # pygidfit/mlgidbase raises an opaque
    # ``ValueError("entry not found in the NeXus file")`` deep inside
    # ``ProcessDataFromFile.process_data_from_file`` and the user has
    # no idea which entries are even available. Common triggers:
    # the entry combo carried a stale name across a file-modification
    # event, or an entry was deleted externally between selection and
    # run. The list is the same one ``MainWindow._populate_entries``
    # uses to fill the combo, so the contract is symmetric.
    requested_entry = command.kwargs.get("entry")
    if isinstance(requested_entry, str) and requested_entry:
        valid_entries = list_entries(file_path)
        if requested_entry not in valid_entries:
            available = ", ".join(repr(e) for e in valid_entries) if valid_entries else "<none>"
            raise RuntimeError(
                f"Cannot run {command.op_name!r}: entry {requested_entry!r} "
                f"is not present in {Path(file_path).name} (or its 'data' "
                f"group does not declare signal='img_gid_q'). Available "
                f"2D q-image entries: {available}. The GUI's entry selector "
                f"should be in sync; this usually means the file was "
                f"modified externally between selection and run, or the "
                f"selection survived a file swap."
            )

    # The injection ops run entirely on mlgidlab primitives (h5py +
    # pygidfit via manual_fit) — deliberately BEFORE the mlgidbase
    # import so they neither need the private backend nor compete with
    # a pygid r+ handle on the same file.
    if command.op_name == "interpolate_tracks":
        return _execute_interpolate_tracks(file_path, dict(command.kwargs))
    if command.op_name == "inject_fitted_peaks":
        return _execute_peak_injection(
            file_path, dict(command.kwargs), op_label="inject_fitted_peaks"
        )

    # Some labeled training files (e.g. ``organic_labeled.h5``) carry
    # fitted_peaks rows that only have polar coordinates — Cartesian
    # ``q_xy`` / ``q_z`` and ``amplitude`` are stored as zeros. mlgidmatch
    # reads ``q_xy`` / ``q_z`` directly and filters by ``amplitude >
    # intensity_threshold``, so without back-filling every peak collapses
    # to the origin with zero intensity and matching silently returns
    # "no solutions". Normalise *before* opening the file via mlgidBASE
    # so the rebuilt analysis sees the patched dataset. track_peaks
    # needs the same treatment: its trajectory payload reads q_xy/q_z
    # straight off the fitted rows, so polar-only rows would collapse
    # every track to the origin in the views.
    if command.op_name in ("run_matching", "track_peaks"):
        try:
            _backfill_fitted_peaks_polar_to_cartesian(
                file_path, command.kwargs.get("entry"),
                command.kwargs.get("frame_num"),
                logging.getLogger("mlgidBASE"),
            )
        except Exception as exc:
            logging.getLogger("mlgidBASE").warning(
                "Could not normalise fitted_peaks in %s: %s",
                file_path, exc,
            )

    # Physics-audit F-04 closure: invalidate every ``matched_*`` row
    # on the scope we're about to refit. pygidFIT replaces an entry's
    # ``fitted_peaks`` wholesale and gives no index-stability
    # guarantee — old ``peak_list`` integer indices that survive
    # ``load_matched_peaks``'s read-side clamp (``file_model.py``)
    # could silently mis-render against re-ordered fitted positions.
    # Clearing matches before the run gives the user a clean slate
    # and forces a deliberate re-match. The clear lands in the same
    # write window as the silx detach the caller has already done.
    # ...unless the run is redirected: a primary fitted table means
    # ``fitted_peaks`` is parked untouched for the whole run, so the
    # matched rows still index exactly the peaks they were matched
    # against and clearing them would destroy good solutions. The skip
    # is keyed on ``fitted_peaks`` specifically -- a primary DETECTED
    # table alone still leaves fitting rewriting the standard fitted
    # table, and the F-04 reasoning applies unchanged.
    _fitted_redirected = bool(
        command.table_swap
        and "fitted_peaks" in command.table_swap.get("stash", ())
    )
    if command.op_name == "run_fitting" and not _fitted_redirected:
        try:
            from mlgidlab.file_model import clear_peaks
            entry_arg = command.kwargs.get("entry")
            frame_arg = command.kwargs.get("frame_num")
            frame_to_clear = int(frame_arg) if isinstance(frame_arg, int) else None
            if isinstance(entry_arg, str) and entry_arg:
                entries_to_clear = [entry_arg]
            else:
                entries_to_clear = list_entries(file_path)
            total_removed = 0
            for ent in entries_to_clear:
                total_removed += clear_peaks(
                    file_path, ent, "matched", frame=frame_to_clear
                )
            if total_removed > 0:
                logging.getLogger("mlgidBASE").info(
                    "Invalidated %d stale matched-peak row(s) before "
                    "run_fitting on %s (physics-audit F-04: "
                    "fitted_peaks rewrite breaks peak_list index "
                    "stability).",
                    total_removed, Path(file_path).name,
                )
        except Exception as exc:
            logging.getLogger("mlgidBASE").warning(
                "Could not invalidate matched solutions before "
                "run_fitting on %s: %s",
                Path(file_path).name, exc,
            )

    # Big scans: upstream track_peaks builds a dense all-against-all
    # IoU matrix over the scan's total fitted-peak count N and needs
    # ~64 * N**2 bytes — the kernel OOM-kills the whole app when that
    # exceeds RAM (observed at 602 frames / 43645 peaks = ~122 GB on
    # a 30 GB machine). When the estimate says the dense run would not
    # fit comfortably, run the GUI's blocked equivalent instead:
    # identical boxes, IoU rule and length cut, bounded memory (see
    # phase_tracking.track_peaks_blocked). Placed BEFORE the mlgidbase
    # import so it reads via h5py only and never competes with a pygid
    # r+ handle on the same file.
    if command.op_name == "track_peaks":
        from mlgidlab.phase_tracking import (
            TRACKING_DENSE_MEM_FRACTION,
            available_memory_bytes,
            estimate_tracking_memory,
            track_peaks_blocked,
        )

        track_entry = str(command.kwargs.get("entry") or "")
        n_peaks, needed = estimate_tracking_memory(file_path, track_entry)
        mem_available = available_memory_bytes()
        if (
            needed and mem_available
            and needed > mem_available * TRACKING_DENSE_MEM_FRACTION
        ):
            logging.getLogger("mlgidBASE").info(
                "track_peaks: %s fitted peaks would need ~%.0f GB with "
                "the upstream dense IoU (%.0f GB available) — running "
                "the memory-safe blocked computation instead "
                "(identical result).",
                f"{n_peaks:,}", needed / 1e9, mem_available / 1e9,
            )
            try:
                return track_peaks_blocked(
                    file_path,
                    track_entry,
                    threshold=float(command.kwargs.get("threshold", 0.5)),
                    length=int(command.kwargs.get("length", 10)),
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"track_peaks found no fitted peaks in entry "
                    f"{command.kwargs.get('entry')!r} — run Fitting "
                    f"(all frames) first. ({exc})"
                ) from exc

    # Pre-flight: make sure the detection weights are cached and intact
    # *before* mlgidbase opens the file. mlgidDETECT ships no ``.onnx``
    # and downloads on first use from inside ``Inference.__init__``,
    # where every failure — no network, a GUI process without a usable
    # ``sys.stdout``, a half-written file from a previous attempt —
    # collapses into the same opaque "Detection failed. Couldn't load
    # the model." Doing it here means progress and errors reach the log
    # panel, and a failed download can never leave a corpse behind that
    # poisons every later run. See ``mlgidlab.detection_model``.
    if command.op_name == "run_detection":
        with safe_console():
            ensure_detection_model(
                model_type=command.kwargs.get("model_type"),
                config_detect=command.kwargs.get("config_detect"),
            )

    # Import lazily — see the function docstring. Every pre-flight
    # above must be able to run on a CI box that lacks the private
    # ``mlgidbase`` backend.
    from mlgidbase import mlgidBASE  # noqa: N814
    analysis = mlgidBASE(filename=str(file_path))
    kwargs = dict(command.kwargs)
    # Run-matching takes a ``cif_prepr`` value that mlgidBASE accepts as
    # either a path-to-pickle string or a CifPattern instance. The GUI lets
    # the user point at raw .cif files / a folder of CIFs too — translate
    # those into a CifPattern here, on the worker thread, since pattern
    # construction simulates each CIF and can take seconds.
    if command.op_name == "run_matching":
        cif_in = kwargs.get("cif_prepr")
        active_entry = kwargs.get("entry")
        if isinstance(cif_in, str) and cif_in:
            # Build the CifPattern against the *active entry's*
            # ExpParameters so multi-energy datasets get correct
            # per-entry simulations. Otherwise we'd silently match
            # against entry_0's wavelength for every entry and yield
            # zero solutions whenever entry_0 differs from the one
            # being processed.
            wrapped = _maybe_build_cif_pattern_from_raw(
                cif_in, file_path, active_entry
            )
            if wrapped is not None:
                kwargs["cif_prepr"] = wrapped
        elif _looks_like_cif_pattern(cif_in) and active_entry is not None:
            # Cached / pre-parsed CifPattern carries fixed ExpParameters
            # from whichever entry the panel preloaded against. Surface
            # a warning when those params don't match the entry being
            # matched right now — common on per-material datasets like
            # ``organic_labeled.h5`` where each entry has its own beam
            # energy and incidence angle. The match still runs, but the
            # log line tells the user why no solutions appeared.
            try:
                _warn_if_cif_params_mismatch(
                    cif_in, file_path, active_entry,
                    logging.getLogger("mlgidBASE"),
                )
            except Exception:
                logger.debug("suppressed exception in execute", exc_info=True)
                pass
    if command.op_name == "track_peaks" and not hasattr(analysis, "track_peaks"):
        # Feature-gate by capability so a stale env (older than the
        # pinned mlgidbase) fails with a named remedy, not an
        # AttributeError.
        raise RuntimeError(
            "The installed mlgidbase has no track_peaks — peak tracking "
            "needs mlgidbase >= 0.1.5 (the [pipeline] pin). See "
            "mlgidLAB/docs/backend_compatibility.md."
        )
    method = getattr(analysis, command.op_name)
    # Only detection and fitting ever carry a plan (``swap_plan``
    # returns None for every other op, and the GUI only stamps those
    # two), so this is a no-op everywhere else.
    swap_entry = kwargs.get("entry")
    swap_frame = kwargs.get("frame_num")
    if command.op_name == "run_detection":
        # ``safe_console`` stays on for the call itself: the pre-flight
        # above covers the model mlgidbase resolves, but any other
        # ``sys.stdout.write`` deeper in mlgidDETECT (e.g. a download we
        # did not anticipate) would otherwise still crash a GUI process
        # that has no console.
        with _swapped_tables(
            file_path, swap_entry, swap_frame, command.table_swap
        ), detection_on_gpu(), safe_console():
            result = method(**kwargs)
    elif command.op_name == "track_peaks":
        # The member-level tracking data is recovered through the
        # capture hook (see phase_tracking.capture_tracking); the
        # amplitudes come from the RETURN value, whose axis table has
        # no 'amplitude' entry. Force the capture contract regardless
        # of what the caller put in kwargs: plot_result must be truthy
        # for the hook to fire (no matplotlib runs — the recorder
        # replaces the plot call) and axis='radius' is always valid.
        from mlgidlab.phase_tracking import (
            amplitudes_from_track_result,
            build_payload,
            capture_tracking,
        )

        kwargs["axis"] = "radius"
        kwargs["plot_params"] = {"plot_result": True, "save_fig": False}
        try:
            with capture_tracking() as rec:
                ret = method(**kwargs)
        except ValueError as exc:
            # np.vstack on an empty box list — the entry has no fitted
            # peaks at all. Name the actual remedy instead of leaking
            # "need at least one array to concatenate".
            raise RuntimeError(
                f"track_peaks found no fitted peaks in entry "
                f"{kwargs.get('entry')!r} — run Fitting (all frames) "
                f"first. ({exc})"
            ) from exc
        result = build_payload(
            rec,
            entry=str(kwargs.get("entry") or ""),
            threshold=float(kwargs.get("threshold", 0.5)),
            length=int(kwargs.get("length", 10)),
            amplitudes=amplitudes_from_track_result(ret, rec),
        )
    else:
        with _swapped_tables(
            file_path, swap_entry, swap_frame, command.table_swap
        ):
            result = method(**kwargs)

    # Consolidate matched_<type>_NNNN groups produced by mlgidmatch.
    # mlgidmatch returns one "solution" per consistent combination of
    # structures, so the same (CIF, h, k, l) structure appears across
    # many groups — usually with a different peak subset each time —
    # when it is shared across multiple combinations. Visually that
    # reads as the exact same structure listed many times over.
    # Replace the per-solution groups with a single group holding one
    # row per unique structure (peak lists unioned, max probability).
    if command.op_name == "run_matching":
        try:
            _dedupe_matched_groups(
                file_path,
                command.kwargs.get("entry"),
                command.kwargs.get("frame_num"),
                command.kwargs.get("peaks_type", "segments"),
                logging.getLogger("mlgidBASE"),
            )
        except Exception as exc:
            logging.getLogger("mlgidBASE").warning(
                "Could not dedupe matched groups in %s: %s",
                file_path, exc,
            )

    return result


def _backfill_fitted_peaks_polar_to_cartesian(
    file_path: Path,
    entry: str | None,
    frame_num: Any,
    logger: Any,
) -> None:
    """Back-fill missing q_xy / q_z / amplitude in fitted_peaks rows.

    Some files store fitted_peaks with only the polar coordinates
    populated (``radius`` / ``angle``) and leave the Cartesian fields
    plus ``amplitude`` at zero. mlgidmatch reads q_xy / q_z directly
    and filters by ``amplitude > intensity_threshold``, so without
    this normalisation every peak ends up at (0, 0) with zero
    intensity and is dropped before any structure matching happens.

    Walks every ``data/analysis/frameNNNNN/fitted_peaks`` dataset under
    each affected entry and, *only* on rows where (q_xy == 0 AND
    q_z == 0 AND radius > 0), writes:

      - ``q_xy = radius * cos(angle_deg)``
      - ``q_z  = radius * sin(angle_deg)``

    On rows where ``amplitude == 0`` we substitute ``1.0`` so the
    default ``intensity_threshold=0`` filter (`amplitude > 0`) lets
    the row through. Rows whose Cartesian or amplitude fields are
    already non-zero are left untouched — this routine never
    overwrites real data.

    No-op when the file is already populated correctly.
    """
    import h5py

    from mlgidlab import polar
    from mlgidlab.file_model import FRAME_KEY_FMT, entry_group_names

    if entry is not None:
        entries = [entry]
    else:
        with h5py.File(file_path, "r") as f:
            entries = sorted(entry_group_names(f))

    patched_total = 0
    with h5py.File(file_path, "r+") as f:
        for ent in entries:
            analysis_grp = f.get(f"{ent}/data/analysis")
            if analysis_grp is None:
                continue
            if frame_num is None:
                frame_keys = list(analysis_grp.keys())
            elif isinstance(frame_num, (list, tuple)):
                frame_keys = [FRAME_KEY_FMT.format(int(n)) for n in frame_num]
            else:
                frame_keys = [FRAME_KEY_FMT.format(int(frame_num))]

            for fk in frame_keys:
                ds_path = f"{ent}/data/analysis/{fk}/fitted_peaks"
                if ds_path not in f:
                    continue
                ds = f[ds_path]
                fp = ds[()]
                # Identify rows that need back-fill. q_xy/q_z exactly 0
                # is suspicious only when radius is meaningfully > 0;
                # leave genuine origin peaks alone.
                need_xy = (
                    (fp["q_xy"] == 0.0)
                    & (fp["q_z"] == 0.0)
                    & (fp["radius"] > 0.0)
                )
                need_amp = fp["amplitude"] == 0.0
                if not need_xy.any() and not need_amp.any():
                    continue
                if need_xy.any():
                    fp["q_xy"][need_xy], fp["q_z"][need_xy] = polar.polar_to_qxyz(
                        fp["radius"][need_xy], fp["angle"][need_xy]
                    )
                if need_amp.any():
                    # Placeholder unit amplitude so intensity-filtered
                    # matching still sees the peak. Real amplitudes
                    # require re-running run_fitting against the image.
                    fp["amplitude"][need_amp] = 1.0
                ds[...] = fp
                patched_total += int(max(need_xy.sum(), need_amp.sum()))
                logger.warning(
                    "fitted_peaks @ %s/%s: back-filled %d row(s) where "
                    "Cartesian q_xy/q_z were zero (computed from polar) "
                    "and %d row(s) with zero amplitude (set to 1.0 as a "
                    "placeholder so the intensity filter does not drop "
                    "them). Re-run fitting for real amplitudes.",
                    ent, fk, int(need_xy.sum()), int(need_amp.sum()),
                )
    if patched_total:
        logger.info(
            "Normalised fitted_peaks in %s: %d row(s) patched across "
            "%d entry(ies). Matching can now proceed.",
            file_path, patched_total, len(entries),
        )


def _dedupe_matched_groups(
    file_path: Path,
    entry: str | None,
    frame_num: Any,
    peaks_type: str,
    logger: Any,
) -> None:
    """Collapse repeated structure identifications across matched groups.

    mlgidmatch emits one ``matched_<type>_NNNN`` group per "solution"
    (a self-consistent combination of structures). When a single
    structure participates in many combinations, a (CIF, h, k, l) row
    for it gets written into every one of those groups — usually with
    a DIFFERENT peak subset per combination, because the tree search
    reaches the structure with different peaks already consumed. The
    GUI then lists the exact same structure many times over.

    This helper, run after each ``run_matching`` call, walks the
    affected ``data/analysis/frameNNNNN`` groups, gathers every row
    across all ``matched_<peaks_type>_*`` datasets, merges rows by the
    (CIF, h, k, l) key — peak lists are UNIONED, the probability is
    the maximum over the merged rows — and rewrites the result as a
    single ``matched_<peaks_type>_0000`` dataset. The original groups
    are deleted in the same write pass.

    No-op when at most one group exists already (a single mlgidmatch
    solution never repeats a structure, and our own consolidated
    output is already unique).
    """
    import h5py
    import numpy as np

    from mlgidlab.file_model import FRAME_KEY_FMT, entry_group_names

    prefix = f"matched_{peaks_type}_"

    if entry is not None:
        entries = [entry]
    else:
        with h5py.File(file_path, "r") as f:
            entries = sorted(entry_group_names(f))

    consolidated_total = 0
    with h5py.File(file_path, "r+") as f:
        for ent in entries:
            analysis_grp = f.get(f"{ent}/data/analysis")
            if analysis_grp is None:
                continue
            if frame_num is None:
                frame_keys = list(analysis_grp.keys())
            elif isinstance(frame_num, (list, tuple)):
                frame_keys = [FRAME_KEY_FMT.format(int(n)) for n in frame_num]
            else:
                frame_keys = [FRAME_KEY_FMT.format(int(frame_num))]

            for fk in frame_keys:
                frame_grp = analysis_grp.get(fk)
                if frame_grp is None:
                    continue
                matched_keys = sorted(
                    k for k in frame_grp.keys() if k.startswith(prefix)
                )
                if len(matched_keys) <= 1:
                    continue

                # Pick a reference dtype so the consolidated dataset
                # round-trips through pygid's reader unchanged.
                ref_ds = frame_grp[matched_keys[0]]
                ref_dtype = ref_ds.dtype

                # Map (cif, h, k, l) -> best row + unioned peak set.
                best: dict[tuple, dict] = {}
                row_count_in = 0
                for mk in matched_keys:
                    rows = frame_grp[mk][()]
                    row_count_in += int(rows.shape[0])
                    for row in rows:
                        peak_arr = np.asarray(row["peak_list"]).astype(np.int32)
                        key = (
                            bytes(row["CIF"]),
                            int(row["h"]), int(row["k"]), int(row["l"]),
                        )
                        prev = best.get(key)
                        if prev is None:
                            # Copy the row so freeing the source array
                            # doesn't take the data with it.
                            best[key] = {
                                "row": np.asarray(row, dtype=ref_dtype).copy(),
                                "peaks": set(peak_arr.tolist()),
                            }
                        else:
                            prev["peaks"].update(peak_arr.tolist())
                            if (
                                float(row["probability"])
                                > float(prev["row"]["probability"])
                            ):
                                prev["row"] = np.asarray(
                                    row, dtype=ref_dtype
                                ).copy()

                if not best:
                    continue

                # Sort consolidated rows by probability descending so
                # the highest-confidence identifications come first.
                ordered = sorted(
                    best.values(),
                    key=lambda e: float(e["row"]["probability"]),
                    reverse=True,
                )
                consolidated = np.empty(len(ordered), dtype=ref_dtype)
                for i, entry_rec in enumerate(ordered):
                    consolidated[i] = entry_rec["row"]
                    consolidated["peak_list"][i] = np.asarray(
                        sorted(entry_rec["peaks"]), dtype=np.int32
                    )

                # Delete every existing matched_<type>_NNNN dataset
                # before writing the consolidated one — pygid's
                # _save_matched_data uses the same delete-then-write
                # pattern so this stays consistent with the rest of
                # the codebase.
                for mk in matched_keys:
                    del frame_grp[mk]
                frame_grp.create_dataset(f"{prefix}0000", data=consolidated)
                consolidated_total += row_count_in - len(ordered)
                logger.info(
                    "Consolidated %d matched %s rows from %d groups into "
                    "1 group with %d unique identifications at %s/%s.",
                    row_count_in, peaks_type, len(matched_keys),
                    len(ordered), ent, fk,
                )
    if consolidated_total:
        logger.info(
            "Removed %d duplicate matched-row(s) total in %s.",
            consolidated_total, file_path,
        )


def _looks_like_cif_pattern(obj: Any) -> bool:
    """True if ``obj`` quacks like a ``CifPattern`` — has a ``params``
    attribute exposing ``en`` / ``ai``. Avoids importing mlgidmatch
    just for an isinstance check."""
    if obj is None:
        return False
    params = getattr(obj, "params", None)
    return params is not None and hasattr(params, "en") and hasattr(params, "ai")


def _warn_if_cif_params_mismatch(
    cif_pattern: Any, nexus_file: Path, entry: str, logger: Any,
) -> None:
    """Emit a warning when the cached CifPattern's params disagree with
    the active entry's params. Threshold: >0.5% relative on either
    energy or angle of incidence (well above HDF5 round-trip noise)."""
    cur = _exp_params_from_nexus(nexus_file, entry)
    cached_en = float(getattr(cif_pattern.params, "en", 0.0))
    cached_ai = float(getattr(cif_pattern.params, "ai", 0.0))
    cur_en = float(getattr(cur, "en", 0.0))
    cur_ai = float(getattr(cur, "ai", 0.0))
    en_ok = cached_en > 0 and abs(cached_en - cur_en) / cached_en < 0.005
    ai_ok = (
        abs(cached_ai - cur_ai) < 1e-3
        if abs(cached_ai) < 1e-3
        else abs(cached_ai - cur_ai) / abs(cached_ai) < 0.05
    )
    if not (en_ok and ai_ok):
        logger.warning(
            "Pre-parsed CifPattern was built against ExpParameters "
            "(en=%.0f eV, ai=%.4f) that don't match entry %r "
            "(en=%.0f eV, ai=%.4f). Pattern simulation is using the "
            "wrong wavelength/incidence — re-parse CIFs with this "
            "entry active, or run matching from the raw CIF folder so "
            "each entry gets its own simulation.",
            cached_en, cached_ai, entry, cur_en, cur_ai,
        )


def parse_cif_input(
    cif_input: str, nexus_file: Path, entry: str | None = None
) -> Any:
    """Pre-load a CIF input into a reusable cache value.

    Returns:
      • ``CifPattern`` for raw .cif paths or a CIF folder (slow — simulates
        every CIF; the GUI runs this in a worker thread so the parsing
        cost is paid once, then reused for every Run-Matching).
      • ``CifPattern`` for a pickle (loaded via ``pickle.load``).

    Either result can be passed verbatim to ``mlgidBASE.run_matching`` as
    ``cif_prepr`` — ``load_cif_prepr`` accepts a ``CifPattern`` and a
    string path, so the cache is uniformly a ``CifPattern``.

    For raw .cif input, pass ``entry`` to bind the simulation to one
    specific entry's ExpParameters — the cached pattern is then only
    valid for that entry. The panel uses this so caches stay in sync
    with the active entry on multi-energy datasets.
    """
    import pickle

    paths = [p.strip() for p in (cif_input or "").split(";") if p.strip()]
    if not paths:
        raise ValueError("Empty CIF input")

    # Single pickle → load it once into memory. mlgidBASE would otherwise
    # re-open it on every run; not expensive but free to avoid.
    if len(paths) == 1 and paths[0].lower().endswith((".pickle", ".pkl")):
        with open(paths[0], "rb") as fh:
            return pickle.load(fh)

    # Raw .cif input goes through CifPattern construction.
    return _build_cif_pattern_from_raw(paths, nexus_file, entry)


def _maybe_build_cif_pattern_from_raw(
    cif_in: str, nexus_file: Path, entry: str | None = None
) -> Any:
    """Return a CifPattern when ``cif_in`` points at raw CIFs, else None.

    Used by ``execute`` to translate string-form ``cif_prepr`` kwargs to
    ``CifPattern`` instances when the user hasn't pre-parsed the input via
    the panel's "Parse CIFs" button. Returns ``None`` for pickle input so
    mlgidBASE's own loader sees the path. ``parse_cif_input`` is the
    panel-facing equivalent that always returns a usable cache value.

    ``entry`` is forwarded to ``_build_cif_pattern_from_raw`` so the
    simulation uses the per-entry ExpParameters; vital for files with
    mixed-energy entries.
    """
    paths = [p.strip() for p in (cif_in or "").split(";") if p.strip()]
    if not paths:
        return None
    # Single pickle → forward as-is. mlgidBASE.load_cif_prepr handles it.
    if len(paths) == 1 and paths[0].lower().endswith((".pickle", ".pkl")):
        return None
    return _build_cif_pattern_from_raw(paths, nexus_file, entry)


def _build_cif_pattern_from_raw(
    paths: list[str], nexus_file: Path, entry: str | None = None
) -> Any:
    """Construct a ``CifPattern`` from a list of raw .cif paths or a folder.

    Internal helper shared by ``parse_cif_input`` (panel preload path) and
    ``_maybe_build_cif_pattern_from_raw`` (worker fallback path).
    Validates that all .cif paths share a folder (CifPattern indexes into
    a single folder) and falls back to scanning a directory's .cifs when
    that's the only path given.

    ``entry`` selects which NeXus entry's instrument metadata is used
    when computing ExpParameters — required for multi-entry files
    where energies differ across entries.
    """
    import os

    # Directory → use every .cif inside.
    if len(paths) == 1 and os.path.isdir(paths[0]):
        folder_path = paths[0]
        cifs = sorted(
            f for f in os.listdir(folder_path) if f.lower().endswith(".cif")
        )
        if not cifs:
            raise FileNotFoundError(
                f"No .cif files found in folder: {folder_path}"
            )
    else:
        # One-or-more raw .cif paths. They must share a folder so CifPattern
        # can locate them through ``folder_path``.
        if not all(p.lower().endswith(".cif") for p in paths):
            raise ValueError(
                "Mixed input — pass either a single pickle, a folder, or "
                "one-or-more .cif files (semicolon-separated)."
            )
        folders = {os.path.dirname(os.path.abspath(p)) for p in paths}
        if len(folders) != 1:
            raise ValueError(
                "When passing multiple .cif files, they must live in the "
                "same folder. Got: " + ", ".join(sorted(folders))
            )
        folder_path = folders.pop()
        cifs = [os.path.basename(p) for p in paths]

    from mlgidmatch.preprocess.cif_preprocess import CifPattern  # noqa: E501

    params = _exp_params_from_nexus(nexus_file, entry)
    # ``create_all=True`` populates the 1D ring patterns (``all_patterns_q1d``
    # / ``all_patterns_int1d``) in addition to the default 2D segment data.
    # mlgidmatch's ``test_rings`` reads the 1D fields directly and crashes
    # with ``'NoneType' object is not subscriptable`` when they're None,
    # so we always populate both — the cache is built once per Parse click
    # anyway and the 1D pass is cheap relative to the 3D pattern simulation
    # that ``create_elementary`` already does.
    return CifPattern(
        params=params,
        folder_path=folder_path,
        cifs=cifs,
        create_all=True,
    )


class _EnergyOutOfRangeError(ValueError):
    """Raised when photon energy derived in ``_exp_params_from_nexus``
    falls outside the plausible X-ray range.

    Subclass of ``ValueError`` so it surfaces as a normal validation
    error to callers; named distinctly so the broad ``except Exception``
    in ``_exp_params_from_nexus`` can re-raise it instead of swallowing
    it into the silent defaults fallback. Closes physics-audit
    finding F-02 (energy derivation unit contract): a future pygidsim
    flip from eV to keV would shrink the computed value 1000x and
    trip the lower bound here before any CIF pattern is simulated.
    """


def _exp_params_from_nexus(
    nexus_file: Path, entry: str | None = None
) -> Any:
    """Derive ExpParameters from a NeXus file's instrument metadata.

    Energy is recovered from the wavelength stored in
    ``instrument/monochromator/wavelength`` (meters); the angle of
    incidence from ``instrument/angle_of_incidence``. q-axis extents
    come from ``data/q_xy`` / ``data/q_z``.

    Pass ``entry`` to read from a specific entry — multi-entry files
    where each entry was collected at a different beam energy / angle
    of incidence (e.g. ``organic_labeled.h5`` with per-material
    entries) need per-entry params, otherwise CIF pattern simulation
    runs against the wrong wavelength and matching silently returns
    "no solutions". When ``entry`` is None, the alphabetically-first
    entry is used as a fallback (preserves previous behaviour for
    callers that don't yet plumb through an entry name).

    Per-field fallbacks (physics-audit F-05 closure). Each datum is
    read independently; if any one is missing or unreadable, only
    that field falls back to its plausible default and a structured
    ``WARNING`` log line is emitted naming the substituted fields,
    the entry, and the file. The pipeline-panel log handler renders
    these warnings prominently, so a matching run on a file with
    incomplete instrument metadata is no longer silently physically
    wrong — the user is told exactly which geometry was guessed.
    """
    from pygidsim.experiment import ExpParameters

    defaults = dict(q_xy_max=2.7, q_z_max=2.7, ai=0.3, en=18000.0)

    import h5py
    import numpy as np

    from mlgidlab.file_model import entry_group_names

    # File / entry resolution: any failure at this layer is whole-
    # file-level (path bad, no q-entries inside) and falls back to
    # the full defaults dict. This is qualitatively different from
    # per-field fallback — a single warning naming the file is the
    # right granularity.
    try:
        h5_handle = h5py.File(nexus_file, "r")
    except OSError as exc:
        logger.warning(
            "Geometry fallback: could not open %s (%s); "
            "using ExpParameters defaults %r for the matching run.",
            nexus_file, exc, defaults,
        )
        return ExpParameters(**defaults)
    try:
        with h5_handle as f:
            if entry is not None and entry in f:
                entry_name = entry
            elif entry is not None:
                logger.warning(
                    "Geometry fallback: entry %r not present in %s; "
                    "using ExpParameters defaults %r for the matching run.",
                    entry, nexus_file.name, defaults,
                )
                return ExpParameters(**defaults)
            else:
                entry_names = sorted(entry_group_names(f))
                if not entry_names:
                    logger.warning(
                        "Geometry fallback: no entry groups in %s; "
                        "using ExpParameters defaults %r for the matching run.",
                        nexus_file.name, defaults,
                    )
                    return ExpParameters(**defaults)
                entry_name = entry_names[0]
            grp = f[entry_name]

            # Per-field reads. ``fallbacks`` tracks (field, reason)
            # tuples so the warning log line can name exactly which
            # parameters were guessed and why.
            fallbacks: list[tuple[str, str]] = []

            def _read_scalar(rel_path: str, field: str):
                try:
                    return float(np.asarray(grp[rel_path]).ravel()[0])
                except Exception as inner:
                    fallbacks.append((field, f"could not read {rel_path}: {inner}"))
                    return None

            def _read_axis_max(rel_path: str, field: str):
                try:
                    return float(np.max(np.abs(grp[rel_path][()])))
                except Exception as inner:
                    fallbacks.append((field, f"could not read {rel_path}: {inner}"))
                    return None

            q_xy_max = _read_axis_max("data/q_xy", "q_xy_max")
            q_z_max = _read_axis_max("data/q_z", "q_z_max")
            wl_m = _read_scalar("instrument/monochromator/wavelength", "wavelength")
            ai = _read_scalar("instrument/angle_of_incidence", "ai")
    except _EnergyOutOfRangeError:
        raise  # never set here — defensive against future restructures
    except Exception as exc:
        # File was opened but a structural read past the entry pick
        # itself failed — e.g. the entry's ``data`` group is missing
        # entirely. Surface as one whole-file warning.
        logger.warning(
            "Geometry fallback: structural read failed on %s (%s); "
            "using ExpParameters defaults %r for the matching run.",
            nexus_file.name, exc, defaults,
        )
        return ExpParameters(**defaults)

    # h*c / λ in eV — physical constants are exact in SI 2019.
    h = 6.62607015e-34
    c = 299792458.0
    eV = 1.602176634e-19
    if wl_m is None or wl_m <= 0:
        en = defaults["en"]
        if wl_m is None:
            # already in fallbacks via _read_scalar
            pass
        else:
            fallbacks.append(("en", f"wavelength {wl_m} not positive"))
    else:
        en = (h * c / wl_m) / eV
        # F-02 guard. Plausible X-ray energies sit between ~1 keV
        # (soft, e.g. tender X-ray near 1.5 nm) and ~200 keV (hard
        # X-ray, beyond which we are into γ territory). Anything
        # outside that bracket means either:
        #   * the wavelength datum is malformed (wrong units, e.g.
        #     stored in Å rather than m → en under 1 keV by 1e10),
        #   * pygidsim has silently changed its ``en`` contract from
        #     eV to keV (would shrink en 1000x and trip the lower
        #     bound) — this is the divergence risk the physics audit
        #     calls out.
        # In either case, fail loud here rather than feed a wrong
        # wavelength into the CIF pattern simulation and produce
        # plausible-looking but physically invalid matches.
        if not (1e3 <= en <= 2e5):
            raise _EnergyOutOfRangeError(
                f"Photon energy derived from "
                f"{nexus_file.name}/{entry_name}/instrument/monochromator/"
                f"wavelength={wl_m:.6e} m is en={en:.4g} eV, outside the "
                f"plausible 1-200 keV X-ray range (1e3-2e5 eV). Inspect "
                f"the wavelength datum, or check that pygidsim still "
                f"expects ``en`` in eV (physics-audit finding F-02)."
            )

    # Apply per-field defaults for anything that did not read.
    if q_xy_max is None:
        q_xy_max = defaults["q_xy_max"]
    if q_z_max is None:
        q_z_max = defaults["q_z_max"]
    if ai is None:
        ai = defaults["ai"]

    if fallbacks:
        # One structured WARNING per call — the pipeline panel log
        # surfaces these prominently. Naming each substituted field
        # is the F-05 closure: the user knows which geometry was
        # guessed and which datum to fix in the NeXus file.
        substituted = ", ".join(f"{name}={defaults.get(name, '?')!r}" for name, _ in fallbacks)
        reasons = "; ".join(f"{name}: {reason}" for name, reason in fallbacks)
        logger.warning(
            "Geometry fallback in %s/%s: substituted %s. Reasons: %s. "
            "Matching will run against these default values; results "
            "may be physically invalid if the real geometry differs "
            "(physics-audit finding F-05).",
            nexus_file.name, entry_name, substituted, reasons,
        )

    return ExpParameters(
        q_xy_max=q_xy_max,
        q_z_max=q_z_max,
        ai=ai,
        en=en,
    )
