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
    path written incrementally with one group per scan.
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
        _validate_append_target(cfg, raw_file_outputs)

    for scan in scans:
        out_path = raw_file_outputs[scan.file_path]
        if cfg.append_frames:
            h5_group = cfg.append_entry
        else:
            idx = entry_counters[out_path]
            entry_counters[out_path] = idx + 1
            h5_group = _entry_group_name(idx)

        # ``overwrite_file`` may only fire once per output file: pygid
        # truncates the file on the first call, then the next call into
        # the same path must append into a fresh group. Track first-touch
        # per output path. Append mode never overwrites anything.
        first_touch = out_path not in seen_paths
        if cfg.append_frames:
            scan_overwrite_file = False
            scan_overwrite_group = False
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

    return written


# -------------- helpers --------------


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
