"""Lazy wrappers around pygid raw → NeXus conversion.

Builds a single global ``ExpParams`` + ``CoordMaps`` per run (matching the
roadmap's "Global Objects" concept), then iterates the user's selected
scans through the matching pygid ``Conversion`` method. Output paths are
returned so MainWindow can auto-open the freshly-converted file in NeXus
mode.

Group naming follows the NeXus convention the rest of mlgidLAB's
NeXus reader expects: ``entry_NNNN`` (zero-padded, four digits). The reader
also accepts off-convention names (e.g. mlgidFIT writes ``<sample>``) as
long as the group is ``NX_class == 'NXentry'`` -- see
``file_model.entry_group_names``.

Imports of ``pygid`` are deferred so the GUI can run without it installed
(``pipeline`` extra). No Qt imports — keep this module independently
testable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from mlgidlab.conversion_panel import (
    CONV_DET2POL,
    CONV_DET2POL_GID,
    CONV_DET2Q,
    CONV_DET2Q_GID,
    OUTPUT_SEPARATE_FILES,
    OUTPUT_SINGLE_ENTRY,
    ConversionConfig,
    RawScan,
)

import logging
logger = logging.getLogger(__name__)


def execute(
    scans: list[RawScan], cfg: ConversionConfig
) -> list[Path]:
    """Run pygid conversion over every scan in ``scans``.

    Builds one shared ``pygid.ExpParams`` and one shared
    ``pygid.CoordMaps`` (the roadmap's "global objects"). For each scan,
    instantiates a ``pygid.Conversion`` and dispatches on ``cfg.conv_type``
    to call the appropriate ``det2q*`` / ``det2pol*`` method with
    ``save_result=True`` so the data lands on disk in NeXus form.

    Returns a list of output file paths actually written. In
    ``OUTPUT_SEPARATE_FILES`` mode this is one path per scan
    (deduplicated); in ``OUTPUT_SEPARATE_DATASETS`` mode it's a single
    path written incrementally with one group per scan; in
    ``OUTPUT_SINGLE_ENTRY`` mode it's a single path holding one fresh
    ``entry_NNNN`` whose image stack grows by one frame per scan.
    """
    import pygid  # lazy

    if not scans:
        raise ValueError("No scans selected for conversion")
    if cfg.poni_path is None:
        raise ValueError("PONI file is required")
    if cfg.output_dir is None:
        raise ValueError("Output directory is required")
    if cfg.geometry == "GID" and cfg.ai is None:
        # pygid silently defaults a missing GID ai to 0, but a 0° incidence
        # angle is almost never what the user wants. Refuse early with a
        # clear message rather than producing nonsense converted data.
        raise ValueError(
            "Angle of incidence (ai) is required for GID geometry"
        )

    output_dir = Path(cfg.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Build the shared ExpParams + CoordMaps -------------------------------
    expparam_kwargs = dict(cfg.expmeta_overrides)
    expparam_kwargs.update(
        poni_path=str(cfg.poni_path),
        ai=cfg.ai,
    )
    if cfg.mask_path is not None:
        expparam_kwargs["mask_path"] = str(cfg.mask_path)
    params = pygid.ExpParams(**expparam_kwargs)

    coordmap_kwargs: dict[str, Any] = dict(
        hor_positive=cfg.hor_positive,
        vert_positive=cfg.vert_positive,
    )
    if cfg.dq is not None:
        coordmap_kwargs["dq"] = cfg.dq
    if cfg.dang is not None:
        coordmap_kwargs["dang"] = cfg.dang
    if cfg.q_xy_range is not None:
        coordmap_kwargs["q_xy_range"] = cfg.q_xy_range
    if cfg.q_z_range is not None:
        coordmap_kwargs["q_z_range"] = cfg.q_z_range
    if cfg.q_x_range is not None:
        coordmap_kwargs["q_x_range"] = cfg.q_x_range
    if cfg.q_y_range is not None:
        coordmap_kwargs["q_y_range"] = cfg.q_y_range
    if cfg.radial_range is not None:
        coordmap_kwargs["radial_range"] = cfg.radial_range
    if cfg.angular_range is not None:
        coordmap_kwargs["angular_range"] = cfg.angular_range
    matrix = pygid.CoordMaps(params, **coordmap_kwargs)

    # --- Construct shared metadata objects ------------------------------------
    smpl_metadata = _build_sample_metadata(pygid, cfg.smplmeta_yaml)
    exp_metadata = _build_exp_metadata(pygid, cfg.expmeta_kv)

    # --- Plan output paths + groups -------------------------------------------
    written: list[Path] = []
    seen_paths: set[Path] = set()
    # Pre-resolve the per-raw-file output path. Keys are ``Path``, values
    # are the absolute output path for that raw file. In separate-datasets
    # mode every raw file maps to the same shared output path.
    raw_file_outputs = _plan_output_paths(scans, cfg, output_dir)
    # ``entry_NNNN`` counters are scoped per output file so the names
    # match the NeXus convention the rest of the GUI's reader expects.
    # When ``cfg.overwrite_file`` is False and the target file already
    # exists, we start the counter ABOVE the highest existing index so
    # successive conversion runs append new entries instead of clobbering
    # ``entry_0000``. With ``overwrite_file=True`` the file is truncated
    # on the first scan and the counter starts fresh at zero.
    entry_counters: dict[Path, int] = {}
    for raw_path, out_path in raw_file_outputs.items():
        if out_path in entry_counters:
            continue
        if cfg.overwrite_file or not out_path.exists():
            entry_counters[out_path] = 0
        else:
            entry_counters[out_path] = _next_entry_index(out_path)

    # Append-frames mode: every scan's frames extend ONE existing entry
    # of ONE existing output file instead of landing in fresh entry_NNNN
    # groups. pygid handles the mechanics (datasets are resizable along
    # the frame axis; per-frame analysis groups are added for the new
    # frames; on a frame-shape mismatch it diverts to a new sibling
    # group with a warning rather than corrupting the stack).
    if cfg.append_frames:
        if cfg.output_mode == OUTPUT_SINGLE_ENTRY:
            # The panel locks this combination out; refuse defensively —
            # one mode creates a fresh entry, the other requires an
            # existing one.
            raise ValueError(
                "Append frames and single-entry output are mutually "
                "exclusive."
            )
        _validate_append_target(cfg, raw_file_outputs)

    # Single-entry mode: every scan's frames land in ONE fresh
    # ``entry_NNNN`` of the shared output file. The first scan creates
    # the group; each following scan writes into it with all overwrite
    # flags off, which pygid's DataSaver treats as an append (image
    # stack + frame_ind + angle_of_incidence are resizable, per-frame
    # analysis groups extend). The result is a real N-frame scan —
    # frame slider + peak tracking work, unlike N sibling entries.
    single_entry_group: str | None = None
    if cfg.output_mode == OUTPUT_SINGLE_ENTRY:
        shared_out = next(iter(set(raw_file_outputs.values())))
        idx = entry_counters[shared_out]
        entry_counters[shared_out] = idx + 1
        single_entry_group = _entry_group_name(idx)

    for scan in scans:
        out_path = raw_file_outputs[scan.file_path]
        if cfg.append_frames:
            h5_group = cfg.append_entry
        elif single_entry_group is not None:
            h5_group = single_entry_group
        else:
            idx = entry_counters[out_path]
            entry_counters[out_path] = idx + 1
            h5_group = _entry_group_name(idx)

        # ``overwrite_file`` may only fire once per output file: pygid
        # truncates the file on the first call, then the next call into
        # the same path must append into a fresh group. Track first-touch
        # per output path. Append mode never overwrites anything; in
        # single-entry mode only the group-creating first scan may
        # overwrite (a later overwrite would truncate the growing stack).
        first_touch = out_path not in seen_paths
        if cfg.append_frames:
            scan_overwrite_file = False
            scan_overwrite_group = False
        elif single_entry_group is not None:
            scan_overwrite_file = cfg.overwrite_file if first_touch else False
            scan_overwrite_group = (
                cfg.overwrite_dataset if first_touch else False
            )
        else:
            scan_overwrite_file = cfg.overwrite_file if first_touch else False
            scan_overwrite_group = cfg.overwrite_dataset

        analysis = pygid.Conversion(
            matrix=matrix,
            path=str(scan.file_path),
            dataset=scan.entry,
            frame_num=scan.frame_num,
        )
        method_name = cfg.conv_type
        method = getattr(analysis, method_name, None)
        if method is None:
            raise AttributeError(
                f"pygid.Conversion has no method named {method_name!r}"
            )
        method_kwargs = _method_kwargs_for(cfg, scan)
        method_kwargs.update(
            save_result=True,
            path_to_save=str(out_path),
            h5_group=h5_group,
            overwrite_file=scan_overwrite_file,
            overwrite_group=scan_overwrite_group,
        )
        if exp_metadata is not None:
            method_kwargs["exp_metadata"] = exp_metadata
        if smpl_metadata is not None:
            method_kwargs["smpl_metadata"] = smpl_metadata
        method(**method_kwargs)

        if first_touch:
            written.append(out_path)
            seen_paths.add(out_path)

    if single_entry_group is not None and written:
        _warn_if_frames_diverted(written[0], single_entry_group, len(scans))

    return written


def import_converted_stack(
    paths: list[Path],
    out_path: Path,
    entry_name: str = "entry_0000",
    qxy_range: tuple[float, float] | None = None,
    qz_range: tuple[float, float] | None = None,
    ai: float = 0.0,
    flip_vertical: bool = False,
    wavelength_A: float | None = None,
    progress=None,
) -> Path:
    """Write already-converted images as ONE N-frame mlgid NeXus entry.

    For q-space maps produced OUTSIDE mlgidLAB (other software /
    beamline pipelines): no conversion runs — each image's pixels are
    copied verbatim into ``data/img_gid_q``, streamed frame-by-frame
    via fabio so memory stays at one frame regardless of N. Frames are
    written in the order given (callers sort; the GUI uses natural
    filename order).

    Axes: with ``qxy_range``/``qz_range`` the q vectors are linear
    ramps over the image width/height (1/Angstrom); without them the
    axes fall back to pixel indices so the entry still classifies as a
    converted mlgid scan and renders. ``ai`` fills
    ``instrument/angle_of_incidence`` (one global value, repeated per
    frame). ``flip_vertical`` flips each frame's rows for sources whose
    q_z runs opposite to pygid's convention.

    The written schema mirrors pygid's DataSaver layout for the pieces
    the GUI reads (resizable ``img_gid_q`` + ``frame_ind``, per-frame
    ``analysis/frameNNNNN`` groups, NXdata signal/axes attrs).

    Pipeline capability is decided by ``wavelength_A``: with a real
    wavelength AND both q ranges, a full ``instrument`` block is
    written — real wavelength + incidence angle, plus explicit ZERO
    placeholders for the detector fields (SDD, beam center, pixel
    size, rotations). pygid's ``load_params`` hard-reads every one of
    those datasets, but detection consumes only image + q axes,
    fitting additionally only wavelength + ai (pygidfit's missing-
    wedge / critical-angle math), and matching only fitted peaks + q
    maxima — verified against mlgidbase 0.1.5 / pygidfit 0.1.3, and
    recorded in docs/backend_compatibility.md so future backend bumps
    re-check it. The placeholders carry a ``placeholder`` attr so the
    file stays honest. Without a wavelength (or with pixel axes) no
    geometry is written and the GUI refuses pipeline ops with a clear
    message; a ``process/mlgidlab`` note records the provenance either
    way.

    All frames must match the first frame's (H, W) — mismatches raise
    ``ValueError`` naming the offending file. Overwrites ``out_path``.
    """
    import datetime

    import fabio
    import h5py
    import numpy as np

    from mlgidlab import __version__, file_model

    if not paths:
        raise ValueError("No image files to import")
    file_model._quiet_fabio()

    first = np.asarray(fabio.open(str(paths[0])).data)
    if first.ndim != 2:
        raise ValueError(
            f"{Path(paths[0]).name}: expected a single 2-D image, got "
            f"shape {first.shape}"
        )
    h, w = first.shape
    n = len(paths)

    if qxy_range is not None:
        q_xy = np.linspace(qxy_range[0], qxy_range[1], w)
    else:
        q_xy = np.arange(w, dtype=np.float64)
    if qz_range is not None:
        q_z = np.linspace(qz_range[0], qz_range[1], h)
    else:
        q_z = np.arange(h, dtype=np.float64)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["NX_class"] = "NXroot"
        f.attrs["default"] = entry_name
        entry = f.create_group(entry_name)
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["default"] = "data"
        entry["title"] = entry_name

        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data.attrs["signal"] = "img_gid_q"
        data.attrs["axes"] = ["frame_ind", "q_z", "q_xy"]

        img = data.create_dataset(
            "img_gid_q",
            shape=(n, h, w),
            maxshape=(None, h, w),
            chunks=(1, h, w),
            dtype=np.float32,
        )
        for ax_name, vec in (("q_xy", q_xy), ("q_z", q_z)):
            ds = data.create_dataset(ax_name, data=vec.astype(np.float64))
            ds.attrs["interpretation"] = "axis"
            if qxy_range is not None or qz_range is not None:
                ds.attrs["units"] = "1/Angstrom"
        data.create_dataset(
            "frame_ind",
            data=np.arange(n, dtype=np.float32),
            maxshape=(None,),
        )
        data["filename"] = [str(p) for p in paths]
        data["frame_num"] = np.arange(n, dtype=np.int64)
        analysis = data.create_group("analysis")
        analysis.attrs["NX_class"] = "NXparameters"
        for i in range(n):
            g = analysis.create_group(f"frame{i:05d}")
            g.attrs["NX_class"] = "NXparameters"

        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = "NXinstrument"
        instrument["angle_of_incidence"] = np.full(n, float(ai))

        with_geometry = (
            wavelength_A is not None
            and wavelength_A > 0
            and qxy_range is not None
            and qz_range is not None
        )
        if with_geometry:
            mono = instrument.create_group("monochromator")
            mono.attrs["NX_class"] = "NXmonochromator"
            mono["wavelength"] = float(wavelength_A) * 1e-10  # Å → m
            det = instrument.create_group("detector")
            det.attrs["NX_class"] = "NXdetector"
            for field in (
                "distance", "x_pixel_size", "y_pixel_size",
                "beam_center_x", "beam_center_y", "polar_angle",
                "aequatorial_angle", "rotation_angle",
            ):
                ds = det.create_dataset(field, data=0.0)
                ds.attrs["placeholder"] = (
                    "imported data — true value unknown; unused by "
                    "detection/fitting/matching"
                )

        process = entry.create_group("process")
        process.attrs["NX_class"] = "NXprocess"
        prov = process.create_group("mlgidlab")
        prov.attrs["NX_class"] = "NXprocess"
        prov["program"] = "mlgidlab"
        prov["version"] = __version__
        prov["date"] = datetime.datetime.now().isoformat(timespec="seconds")
        if with_geometry:
            prov["NOTE"] = (
                "Imported from pre-converted image files. Wavelength "
                "and incidence angle are user-supplied; the detector "
                "fields are zero placeholders (unused by "
                "detection/fitting/matching, which consume only the "
                "image, q axes, wavelength and incidence angle)."
            )
        else:
            prov["NOTE"] = (
                "Imported from pre-converted image files; no detector "
                "geometry available. Pipeline operations "
                "(detection/fit/match) require re-importing with q "
                "ranges and a wavelength, or a real conversion."
            )

        for i, p in enumerate(paths):
            frame = np.asarray(fabio.open(str(p)).data)
            if frame.shape != (h, w):
                raise ValueError(
                    f"{Path(p).name}: shape {frame.shape} does not match "
                    f"the first frame's {(h, w)} — all imported images "
                    "must share one detector size"
                )
            if flip_vertical:
                frame = np.flipud(frame)
            img[i] = frame.astype(np.float32, copy=False)
            if progress is not None:
                progress(i + 1, n)

    return out_path


# -------------- helpers --------------


def _warn_if_frames_diverted(
    out_path: Path, group: str, expected: int
) -> None:
    """Log a warning when a single-entry run did not land every frame.

    pygid diverts a frame whose converted shape mismatches the growing
    stack to a fresh sibling group (with only a ``UserWarning``), so a
    mixed-size batch would silently produce extra entries. The frame
    count lives in ``<group>/data/frame_ind``. Purely advisory — never
    raises; the data that DID land is intact.
    """
    import h5py

    try:
        with h5py.File(out_path, "r") as f:
            n = int(f[f"{group}/data/frame_ind"].shape[0])
    except (OSError, KeyError):
        logger.debug("suppressed frame-count check", exc_info=True)
        return
    if n != expected:
        logger.warning(
            "Single-entry conversion: only %d of %d frames landed in "
            "%s — pygid diverted mismatched-shape frames to sibling "
            "entries (check that every selected image has the same "
            "detector size).",
            n, expected, group,
        )


def _validate_append_target(
    cfg: ConversionConfig, raw_file_outputs: dict[Path, Path]
) -> None:
    """Check that append-frames mode has a usable target.

    Requirements: every scan resolves to ONE output file (separate-files
    mode with multiple raw inputs is ambiguous — which file would the
    frames extend?), that file exists, and ``cfg.append_entry`` names an
    existing group in it. Raises ``ValueError`` with a user-readable
    message otherwise. Pure h5py — callable without pygid (unit tests).
    """
    import h5py

    targets = set(raw_file_outputs.values())
    if len(targets) != 1:
        raise ValueError(
            "Append frames needs a single output file, but the current "
            "output settings map the selected scans to "
            f"{len(targets)} different files. Use 'Separate datasets in "
            "single file' mode or select scans from one raw file."
        )
    target = next(iter(targets))
    if not target.is_file():
        raise ValueError(
            f"Append frames: output file does not exist yet: {target}"
        )
    if not cfg.append_entry:
        raise ValueError("Append frames: no target entry selected.")
    try:
        with h5py.File(target, "r") as f:
            present = cfg.append_entry in f
    except OSError as exc:
        raise ValueError(
            f"Append frames: could not open {target}: {exc}"
        ) from exc
    if not present:
        raise ValueError(
            f"Append frames: entry {cfg.append_entry!r} not found in "
            f"{target.name}."
        )


def _entry_group_name(index: int) -> str:
    """Format an HDF5 entry-group name in mlgidLAB's NeXus shape.

    Our own writer always uses the ``entry_NNNN`` convention. (The reader,
    ``file_model.entry_group_names``, is more lenient and also accepts an
    off-convention name carrying ``NX_class == 'NXentry'`` -- e.g. mlgidFIT
    output named after the sample.)
    """
    return f"entry_{index:04d}"


def _next_entry_index(path: Path) -> int:
    """Return the smallest unused ``entry_NNNN`` index in ``path``.

    Used when re-converting into an existing file with
    ``overwrite_file=False``: we want the new entry to land alongside
    the old ones, not on top of them. Returns 0 if the file has no
    pre-existing ``entry_*`` groups (or can't be opened).
    """
    import h5py

    try:
        with h5py.File(path, "r") as f:
            indices: list[int] = []
            for name in f.keys():
                if not name.startswith("entry_"):
                    continue
                suffix = name[len("entry_"):]
                if suffix.isdigit():
                    indices.append(int(suffix))
            if not indices:
                return 0
            return max(indices) + 1
    except (OSError, KeyError):
        return 0


def _plan_output_paths(
    scans: list[RawScan], cfg: ConversionConfig, output_dir: Path
) -> dict[Path, Path]:
    """Map each raw file to the path its converted data will land at.

    Honours ``cfg.output_filename`` per the rules below:

    Separate-datasets mode:
        Every raw file in the batch maps to a single shared output file.
        Default name ``converted.h5``; the user-supplied
        ``output_filename`` overrides it (a missing ``.h5`` extension
        is added).

    Separate-files mode:
        Each raw file maps to its own output file.

        - ``output_filename`` empty (default): ``{raw_stem}_converted.h5``.
        - ``output_filename`` set + single raw file: use the supplied
          name verbatim.
        - ``output_filename`` set + multiple raw files: treat the
          supplied name as a prefix; the raw stem is appended so the
          batch produces unique paths
          (``{prefix}_{raw_stem}.h5``).

    Returns a dict keyed on raw file paths; values are absolute output
    paths.
    """
    is_separate = cfg.output_mode == OUTPUT_SEPARATE_FILES
    raw_files = [scan.file_path for scan in scans]
    # Preserve insertion order while deduplicating — multiple scans from
    # the same raw file share an output path.
    unique_raw: list[Path] = list(dict.fromkeys(raw_files))
    custom = (cfg.output_filename or "").strip()

    if not is_separate:
        if custom:
            name = custom if custom.lower().endswith((".h5", ".hdf5", ".nxs")) else f"{custom}.h5"
        else:
            name = "converted.h5"
        shared = output_dir / name
        return {raw: shared for raw in unique_raw}

    # Separate-files mode.
    out: dict[Path, Path] = {}
    if custom and len(unique_raw) == 1:
        # Single-file batch with custom name → use verbatim.
        name = custom if custom.lower().endswith((".h5", ".hdf5", ".nxs")) else f"{custom}.h5"
        out[unique_raw[0]] = output_dir / name
        return out
    if custom:
        # Multi-file batch with custom name → use as prefix; append raw
        # stem so the outputs stay unique. Strip the ``.h5`` extension
        # off the prefix if the user typed it.
        prefix = custom
        for ext in (".h5", ".hdf5", ".nxs"):
            if prefix.lower().endswith(ext):
                prefix = prefix[: -len(ext)]
                break
        for raw in unique_raw:
            out[raw] = output_dir / f"{prefix}_{raw.stem}.h5"
        return out
    # Default per-file naming.
    for raw in unique_raw:
        out[raw] = output_dir / f"{raw.stem}_converted.h5"
    return out


def _method_kwargs_for(
    cfg: ConversionConfig, scan: RawScan
) -> dict[str, Any]:
    """Build the per-call kwargs for the pygid conversion method.

    Range / step kwargs live on CoordMaps already (built once globally),
    but pygid lets the user override per-call too — we forward the
    config values so the method picks them up regardless of which path
    the underlying pygid version honours.
    """
    kwargs: dict[str, Any] = {
        "frame_num": scan.frame_num,
        "return_result": False,
    }
    conv = cfg.conv_type
    if conv == CONV_DET2Q_GID:
        if cfg.q_xy_range is not None:
            kwargs["q_xy_range"] = cfg.q_xy_range
        if cfg.q_z_range is not None:
            kwargs["q_z_range"] = cfg.q_z_range
        if cfg.dq is not None:
            kwargs["dq"] = cfg.dq
    elif conv == CONV_DET2Q:
        if cfg.q_x_range is not None:
            kwargs["q_x_range"] = cfg.q_x_range
        if cfg.q_y_range is not None:
            kwargs["q_y_range"] = cfg.q_y_range
        if cfg.dq is not None:
            kwargs["dq"] = cfg.dq
    elif conv in (CONV_DET2POL_GID, CONV_DET2POL):
        if cfg.radial_range is not None:
            kwargs["radial_range"] = cfg.radial_range
        if cfg.angular_range is not None:
            kwargs["angular_range"] = cfg.angular_range
        if cfg.dq is not None:
            kwargs["dq"] = cfg.dq
        if cfg.dang is not None:
            kwargs["dang"] = cfg.dang
    return kwargs


def _build_sample_metadata(pygid_mod: Any, yaml_text: str) -> Any:
    """Parse user-supplied YAML into a ``pygid.SampleMetadata`` instance.

    Returns None on empty input so the per-scan kwargs stay clean.
    Empty / pure-whitespace YAML is treated as no metadata; YAML that
    parses to non-dict (e.g. a list at the top level) raises so the
    user finds the typo before the conversion runs.
    """
    text = (yaml_text or "").strip()
    if not text:
        return None
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to parse sample metadata. Install with "
            "`pip install PyYAML`."
        ) from exc
    parsed = yaml.safe_load(text)
    if parsed is None:
        return None
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Sample metadata YAML must parse to a dict at the top level, "
            f"got {type(parsed).__name__}"
        )
    # pygid expects ``data`` at the root. If the user wrapped their
    # metadata in ``data:`` already, pass through; otherwise wrap.
    if "data" not in parsed:
        parsed = {"data": parsed}
    return pygid_mod.SampleMetadata(data=parsed["data"])


def _build_exp_metadata(pygid_mod: Any, kv: dict[str, str]) -> Any:
    """Build a ``pygid.ExpMetadata`` from key/value pairs.

    Returns None on empty input. Values are stored as-is (strings); the
    user can use the panel's HDF5 picker to copy in numeric values when
    a metadata entry needs to round-trip as a number rather than a
    label.
    """
    cleaned = {k: v for k, v in (kv or {}).items() if k}
    if not cleaned:
        return None
    obj = pygid_mod.ExpMetadata(**cleaned)
    # pygid's ExpMetadata uses ``extend_fields`` to flag which fields are
    # appended on multi-frame writes. Mark every user-provided key so a
    # batch run that touches multiple frames stays self-consistent.
    try:
        obj.extend_fields = list(cleaned.keys())
    except Exception:
        # Older pygid versions might not expose extend_fields as a
        # writable attribute; that's fine, the field still gets written
        # once per output.
        logger.debug("suppressed exception in _build_exp_metadata", exc_info=True)
        pass
    return obj
