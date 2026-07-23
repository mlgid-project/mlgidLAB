# Backend compatibility — verifying mlgidbase & the pipeline stack

mlgidLAB runs standalone (view + edit existing NeXus results) on its
public-PyPI runtime dependencies alone. Detection, fitting, and
matching require the in-house pipeline stack — installed via the
`[pipeline]` extra. That stack moves faster than the GUI, so this
document is the procedure for confirming a given pipeline release still
drives mlgidLAB the same way, plus the exact API surface the GUI
depends on (the things that break if an upstream signature drifts).

Run this whenever you bump `mlgidbase` (or any backend), before
shipping a release, and as part of the alpha pre-flight.

## Verified-good baseline

The full suite passes with this set installed (the backend-dependent
tests un-skip and run, not just the pure-h5py ones):

| Package | Baseline version | Reached how |
|-|-|-|
| `mlgidbase` | 0.1.5 | declared in `[pipeline]` |
| `pygid` | 0.2.13 | declared in `[pipeline]` |
| `pygidfit` | 0.1.3 | declared in `[pipeline]` (GUI imports it directly) |
| `mlgidmatch` | 0.1.3 | declared in `[pipeline]` (GUI imports it directly) |
| `pygidsim` | 0.1.4 | declared in `[pipeline]` (GUI imports it directly) |
| `mlgiddetect` | 0.2.8 | transitive via `mlgidbase` (GUI never imports it) |
| `networkx` | any | declared in `[pipeline]` (mlgidbase needs it, doesn't declare it) |

**Pinning policy — exact `==`, not floors.** Each in-house backend
release can shift the detection/fitting/matching numerics or the
on-disk schema, so a bump must be a deliberate, test-rechecked change
rather than something pip picks up silently. The `[pipeline]` extra
therefore pins every directly-imported backend to an exact version
(the baseline above). `mlgidbase 0.1.5` itself pins its own backends
with `==` too (`pygid==0.2.13`, `mlgiddetect==0.2.8`, `pygidfit==0.1.3`,
`mlgidmatch==0.1.3`); `pygidsim==0.1.4` arrives via `mlgidmatch` and
`pygid`. The GUI still declares `pygidfit`/`mlgidmatch`/`pygidsim`
explicitly — rather than leaning on `mlgidbase`'s transitive closure —
because it imports those three *directly*, so a future `mlgidbase`
dropping one must surface as a clear resolver error, not a runtime
`ImportError`. `mlgiddetect` is left transitive (the GUI never imports
it); its per-Python `onnxruntime-gpu` pins (CUDA 12 builds; 1.26.0 on
Python 3.11+) carry the GPU runtime, so the GUI no longer pins
onnxruntime itself. **To move the baseline up:** bump the pins here
and in `pyproject.toml`, re-run the full suite + the end-to-end demo
loop, then commit.

**Runtime (non-pipeline) deps stay on `>=` floors on purpose.** The GUI
stack — PySide6, pyqtgraph, silx, numpy, etc. — is *not* pinned exact,
so the CI matrix can keep testing forward-compat across Python
3.11-3.14 against whatever those projects release. A cyclic-GC-during-
construction segfault that the newest PySide6 (6.11.1) + Python 3.13/14
exposed is handled at the source in `tests/conftest.py` (`gc.disable()`
for the session) plus an idempotent `closeEvent`, not by pinning the
GUI stack down — verified green with PySide6 6.11.1 + pyqtgraph 0.14.0
on both Python 3.12 and 3.14.

## Step 1 — Is mlgidbase installable from your index?

```bash
pip index versions mlgidbase        # lists versions visible on the active index
pip download mlgidbase --no-deps -d /tmp/mlgidbase_check   # confirms it resolves + fetches
```

If those fail against public PyPI, the package is coming from a
private index — inspect the configured indexes:

```bash
pip config list                     # look for index-url / extra-index-url
cat ~/.config/pip/pip.conf 2>/dev/null
```

Web cross-check: open `https://pypi.org/project/mlgidbase/`.

## Step 2 — Does the latest stack still drive the GUI?

In a throwaway environment, so the verified-good env stays intact:

```bash
conda create -n mlgidlab_compat python=3.12 -y
conda activate mlgidlab_compat
git clone https://github.com/mlgid-project/mlgidLAB && cd mlgidLAB
pip install mlgidbase pygid pygidfit mlgidmatch     # pull the latest from PyPI
pip install -e ".[dev]"
python -c "import importlib.metadata as m; \
  [print(p, m.version(p)) for p in \
  ('mlgidbase','pygid','pygidfit','mlgidmatch','pygidsim','mlgiddetect')]"
pytest -q          # backend tests un-skip when the stack imports
```

Watch for: the pipeline / manual-fit / energy-guard / matched-
invalidation tests must **run** (not `importorskip`-skip) and pass. A
skip there means a backend failed to import — read the collection log.

Then a behavioral end-to-end pass in the GUI (the demo loop):

```bash
python -m mlgidlab
```

1. Open `example/eiger4m_0000.h5` (raw) and convert it.
2. On the converted file: Pipeline dock → Run Detection → Run Fitting.
3. Parse CIFs (or load `example/prepr_cifs.pickle`) → Run Matching.
4. Overlays + Peaks-dock tables populate for all three kinds.

If any stage errors, the failing call is in the API-surface table
below — that tells you which upstream signature changed.

## Step 3 — API surface the GUI depends on

If a stage breaks after a bump, the cause is almost always one of
these. Verbatim call sites are in the code; this is the contract.

**`mlgidbase.mlgidBASE`** (`pipeline.py`, `figure_export_window.py`)
- `mlgidBASE(filename=str(path))` constructor.
- `analysis.run_detection(**kwargs)`, `run_fitting(**kwargs)`,
  `run_matching(**kwargs)` — resolved dynamically by name.
- `analysis.set_plot_defaults(**style)` and
  `analysis.plot_analysis_results(save_fig=True, path_to_save_fig=...,
  plot_result=False, return_result=False, **kwargs)` (figure export).

**`pygid`** (`conversion.py`)
- `pygid.ExpParams(poni_path=..., ai=..., mask_path=...)`
- `pygid.CoordMaps(params, hor_positive=..., vert_positive=..., dq=...,
  dang=..., q_xy_range=..., q_z_range=..., ...)`
- `pygid.Conversion(matrix=..., path=..., dataset=..., frame_num=...)`
  then `det2q_gid` / `det2q` / `det2pol_gid` / `det2pol`
  `(frame_num=..., return_result=False, save_result=True,
  path_to_save=..., h5_group=..., overwrite_file=..., overwrite_group=...)`
- `pygid.SampleMetadata(data=...)`, `pygid.ExpMetadata(**fields)`, and
  the writable `ExpMetadata.extend_fields` attribute (set inside a
  `try/except` — older pygid lacking it is tolerated).

**`pygidsim.experiment.ExpParameters`** (`pipeline.py`)
- `ExpParameters(q_xy_max=..., q_z_max=..., ai=..., en=...)` — `en` in
  eV; the energy guard expects `1e3 <= en <= 2e5`.

**`mlgidmatch.preprocess.cif_preprocess.CifPattern`** (`pipeline.py`)
- `CifPattern(params=..., folder_path=..., cifs=..., create_all=True)`;
  the GUI reads `cif_pattern.params.en` and `.params.ai` (via
  `getattr(..., default)`) to detect a parameter mismatch.

**`pygidfit.process_scans`** (`manual_fit.py`)
- `img_preprocessing(cartesian, ai_deg, crit_angle, wavelength_A, q_z)`
- `_get_polar_grid(img_shape, (512, 1024), [0, 0])` — the polar grid is
  hardcoded `(512, 1024)`; a change here breaks byte-identity between
  manual 2D fits and pipeline fits.
- `polar_conversion(img_pre, yy, zz, cv2.INTER_LINEAR)`
- `fit_data(polar_img, radius=..., radius_width=..., angle=...,
  angle_width=..., wavelength=..., q_xy_max=..., q_z_max=..., ...)`

**Data contracts** (file_model + pipeline post-processing)
- NeXus paths: `{entry}/data/img_gid_q`, `{entry}/data/q_xy`,
  `{entry}/data/q_z`, `{entry}/data/analysis/frameNNNNN/<kind>_peaks`.
- `fitted_peaks` dtype fields: `q_xy, q_z, radius, angle,
  radius_width, angle_width, amplitude, id, score, is_ring`.
- `matched_peaks` per-solution fields: `CIF, h, k, l, peak_list,
  probability` (`peak_list` recast to int32; rows deduped by max
  `probability`).

## Step 4 — Record the result

If the latest stack passes, note the confirmed versions here (update
the baseline table) and, if a version moved, bump the `[pipeline]` pins
in `pyproject.toml`. If it fails, file the failing call + the version
that introduced the break as a known issue and keep the last-good
pin in place.

## Recorded bumps

### 2026-07-05 — mlgidbase main @ `561edfa` (unreleased, self-labelled 0.1.3)

Installed for the phase-tracking feature (`track_peaks`, "First
implementation of the peak tracking function"):

```bash
# reference clone fast-forwarded to origin/main first
pip install --force-reinstall --no-deps project_repos/mlgidBASE
```

- **Full suite green** against it (352 passed locally, backends
  installed) — the standard Step-2 verification.
- The `[pipeline]` pin in `pyproject.toml` is **unchanged** (still
  `mlgidbase` 0.1.3): the commit is unreleased and still calls itself
  0.1.3, so there is nothing meaningful to pin. Re-pin when upstream
  tags the release containing `561edfa`.
- **Caveats the GUI codes around** (checked at this sha; see
  `mlgidlab/phase_tracking.py`):
  - `track_peaks` exists only past the 0.1.3 release — the GUI
    feature-gates on availability and raises a named RuntimeError
    ("needs mlgidbase newer than 0.1.3") for older installs.
  - `networkx` is imported by `mlgidbase.peak_operations` but **not
    declared** in mlgidbase's dependencies (upstream packaging gap).
    It is present in the GUI_project env; a fresh env needs
    `pip install networkx` until upstream declares it.
  - Upstream `_track_peaks` **returns None** (docstring claims
    otherwise); the GUI recovers the data via a capture hook on
    `mlgidbase.peak_operations._plot_tracked_peaks` (positional
    8-arg contract, validated at call time —
    `phase_tracking.UpstreamContractError` fires loudly if a future
    mlgidbase changes the call shape).
  - **Rings are never tracked by upstream**: a ring's
    `angle_width = inf` makes every IoU against its box NaN,
    `_track_peaks` zeroes NaNs, so each ring is a 1-member component
    discarded by the `length` cut. The GUI compensates by tracking
    rings itself — `phase_tracking.track_rings` runs the same
    all-against-all / connected-components / `length`-cut algorithm on
    the 1-D IoU of the radius intervals (rings have no azimuthal
    position) and appends the ring tracks to the captured payload. A
    proper upstream fix (ring↔ring radial IoU inside `_track_peaks`,
    e.g. treating `angle_width = inf` as full-arc overlap) would let
    the GUI drop this pass — report alongside the return-None and
    networkx items.
- Rollback: `pip install mlgidbase==0.1.3` (phase tracking then
  reports the feature as unavailable; everything else unchanged).

### 2026-07-21 — mlgidbase main @ `1b6514f` (unreleased, self-labelled 0.1.4)

Moves the env past the 0.1.4 release (2026-07-07) onto current main,
still `--no-deps` so the rest of the verified stack (pygid 0.2.10,
mlgiddetect 0.2.3, onnxruntime-gpu 1.26.0) stays untouched — main
pins `pygid==0.2.12` / `mlgiddetect==0.2.5`, neither vetted here:

```bash
pip install --no-deps --force-reinstall \
    "git+https://github.com/mlgid-project/mlgidBASE@1b6514f"
```

- **Full suite green** against it after the GUI adaptations below.
- What changed upstream since `561edfa` (0.1.4 tracking rework + the
  fig/ax return commits):
  - `track_peaks` **removed the `'amplitude'` axis** (ValueError:
    invalid axis). The GUI now requests `axis='radius'` and recovers
    amplitudes from the **return value**, which is no longer None:
    pre-0.1.4 flat `(axis_arr, amplitude_all, G_comps_list)` and
    0.1.4+ per-component frame-sorted `(axis_list, amplitude_list,
    frame_num_list)` are both understood
    (`phase_tracking.amplitudes_from_track_result`). Members outside
    every component get NaN amplitude; `member_ids` therefore uses
    amplitude only as a duplicate-breaker when finite.
  - The `_plot_tracked_peaks` capture contract (positional 8-arg
    call) is **unchanged** — verified at this sha.
  - `plot_analysis_results` renamed `return_result` → `return_fig`;
    the figure-export window now omits the kwarg entirely (default is
    off under both names), so it works against every build.
  - Upstream now clamps ring `angle_width = inf` to 45° inside
    `_track_peaks` (in-memory only). Rings still form 1-member
    components in practice; the GUI's own `track_rings` pass stays.
  - The networkx packaging gap from the `561edfa` entry still holds.
- Rollback: `pip install mlgidbase==0.1.3` or reinstall `561edfa`
  (the GUI adaptations are backward-compatible with both).

### 2026-07-21 — `onnxruntime-gpu==1.26.0` pinned in `[pipeline]`

`mlgiddetect<=0.2.3` requires `onnxruntime-gpu` without a version.
onnxruntime-gpu **1.27.0 (2026-06-18) hard-links `libcudart.so.13`
at import time**; the CUDA 13 runtime arrives as pip `nvidia-*`
wheels (via torch >= 2.12), but nothing puts them on the loader path,
so in a fresh `[pipeline]` env `import mlgidbase` (→ mlgiddetect →
onnxruntime) fails with *"libcudart.so.13: cannot open shared object
file"* and the GUI misread it as "backend not installed". Two-part
fix:

- pin `onnxruntime-gpu==1.26.0` (the verified GPU-capable build that
  imports lazily), and
- `pipeline.is_mlgidbase_available` retries the import with `torch`
  imported first (torch dlopens its bundled CUDA libraries, which
  also satisfies onnxruntime 1.27+) and logs the real ImportError
  instead of silently reporting "not installed".

Trap for the next bump: **mlgiddetect >= 0.2.4 (2026-07-17) switched
its dependency to CPU-only `onnxruntime`**, which ships the same
Python module as `onnxruntime-gpu` — co-installing both clobbers
files nondeterministically, and mlgidbase 0.1.4 additionally declares
plain `onnxruntime` itself. When the extra moves to a stack with
mlgiddetect >= 0.2.4, drop this pin and decide the GPU story
explicitly (GPU detection needs `onnxruntime-gpu` present INSTEAD of
the CPU package, and `pipeline.detection_on_gpu` handles provider
selection + `preload_dlls`).

### 2026-07-23 — mlgidbase 0.1.5 from PyPI, full released stack

The first PyPI release containing the tracking rework: the env moves
off git installs entirely — `pip install mlgidbase==0.1.5` WITH
dependencies (pygid 0.2.13, mlgiddetect 0.2.8, pygidfit 0.1.3,
mlgidmatch 0.1.3). Code-wise 0.1.5 is the `1b6514f` build the env
already ran plus ensemble METADATA keys; the meaningful changes are
dependency-level. Full suite green before and after the GUI
adaptations below.

- **Workarounds dropped** (fixed upstream):
  - `is_mlgidbase_available`'s torch-first import retry — mlgiddetect
    0.2.8 pins per-Python `onnxruntime-gpu` (CUDA 12, lazily
    importing; 1.26.0 on py3.11+), so the libcudart.so.13 import trap
    is gone; the `[pipeline]` `onnxruntime-gpu==1.26.0` pin is dropped
    for the same reason.
  - `detection_on_gpu`'s `torch.cuda.is_available` monkeypatch —
    mlgiddetect >= 0.2.7 gates GPU on the ORT providers + a CUDA
    driver probe (never torch) and falls back to CPU;
    `MODEL_FORCE_CPU` forces CPU. The context manager now only
    preloads onnxruntime's CUDA DLLs.
  - `amplitudes_from_track_result`'s pre-0.1.4 flat-shape branch —
    the pin guarantees the per-component return; the flat shape now
    raises `UpstreamContractError`.
  - `figure_export_window`'s omitted figure-return kwarg — passes
    `return_fig=False` explicitly (the 0.1.4 rename is settled).
- **Ring tracking, the fine print.** 0.1.5 clamps ring
  `angle_width` inf -> 45 inside `_track_peaks`, but pygidfit
  persists rings with `angle = NaN` (`pygidfit/box_utils.py`,
  `make_box_attributes`), and NaN still poisons the (angle, radius)
  IoU box — backend-fitted rings STILL end as discarded 1-member
  components, so the GUI's `track_rings` pass stays. Rings stored
  with a FINITE angle (mlgidLAB's injected rings persist
  `angle = 45`) CAN now form native components; the result handler
  (`_on_phase_track_result`) drops ring-member components from the
  native payload before appending `track_rings` output, so every
  physical ring yields exactly one track (regression:
  `test_native_ring_components_not_double_counted`). Report upstream:
  ring tracking wants radial IoU for `is_ring` rows, not an angle
  clamp.
- **Still true in 0.1.5 / still GUI-side** (checked at this bump):
  `_plot_tracked_peaks` capture hook (the return value has per-track
  arrays but no member coordinates or peak ids), `member_ids`
  reconstruction, `_backfill_fitted_peaks_polar_to_cartesian`,
  `_dedupe_matched_groups`, matched-row invalidation before refit,
  `create_all=True` on CifPattern, the pygid pre-flight guards,
  the energy-range guard, `normalize_for_pygid` (pygid 0.2.13's
  `get_ai` still assumes 1-D `angle_of_incidence`), the manual
  injection stack (`add_peak` upstream appends an unfitted detected
  row only — no 2D fit, no plausibility gate, no rollback), GUI-side
  peak delete (different cascade semantics), and the networkx
  packaging gap (still undeclared by mlgidbase; now declared in the
  GUI's `[pipeline]` extra).
- **pygid 0.2.10 -> 0.2.13** (skimmed `v0.2.10..origin/main`):
  fig/ax returns on visualization, `Conversion.get_result()`,
  `NexusFile.set_beamtime_info()`, simulation range follows the image
  range, and the module-level root-handler stripping in
  `coordmaps`/`datasaver` is REMOVED (only a harmless plain
  `basicConfig` remains in `conversion`). `det2*`, the datasaver
  results dtype, and `get_ai` are unchanged — the GUI's row writers
  and conversion path are unaffected.
- **Detection results change on purpose:** mlgiddetect 0.2.6+
  defaults to the 2-class `DINO_classAwareBaseline` (vs the
  single-class dino that 0.2.3 shipped). Decision 2026-07-23: adopt
  the new default; re-vet detection quality manually (legacy model
  reachable via `ONNX_BASE: dino_old` + `CLASSAWARE_NMS: False`).
- Rollback: reinstall the previous set
  (`pip install --no-deps --force-reinstall "git+https://github.com/mlgid-project/mlgidBASE@1b6514f" pygid==0.2.10 mlgiddetect==0.2.3 onnxruntime-gpu==1.26.0`)
  and revert the GUI commits of this bump (the dropped torch-gate
  workaround matters on mlgiddetect <= 0.2.3).
