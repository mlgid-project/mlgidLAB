# Changelog

All notable changes to mlgidLAB are recorded here. Versions follow
[PEP 440](https://peps.python.org/pep-0440/); `aN` suffixes are alpha
pre-releases.

## 0.1.0a11 — eleventh alpha (2026-07-10)

Feature alpha on `0.1.0a10`: cross-frame peak tracking with interactive
views, gap filling and export. No on-disk schema changes; the
`[pipeline]` pins are unchanged. The tracking features need an
mlgidbase build newer than the pinned `0.1.3` release (`track_peaks`);
they detect a too-old build and say so — everything else runs on the
pins.

### Added

- **Cross-frame peak tracking.** A new **Scan tracking** dock runs
  mlgidBASE's `track_peaks` over the active entry and lists the surviving
  tracks (frame span, member count, matched structures). Clicking a row
  jumps the viewer to the track's first frame and selects its fitted peak;
  selecting a tracked peak mid-scan highlights its row without bouncing the
  frame. A **Show only tracked peaks** filter hides all non-tracked
  fitted/matched peaks during playback. Rings are tracked GUI-side (radial
  overlap across frames), since upstream tracking covers spots only.
  Results invalidate automatically when fitting rewrites the entry.
- **Phase tracking views.** **Show views…** opens a window with four
  interactive tabs: per-track trajectories of any axis vs frame (click a
  point to jump the viewer there); a (q_xy, q_z) map of the tracks,
  optionally over a nan-mean image of a chosen frame window; amplitude
  evolution with smoothing/normalization, grouping by matched structure, a
  median-per-structure mode with a transparent interquartile band, and thin
  dashed max/min reference lines per structure; and a frame × |q| radial
  waterfall of the whole scan. Tracks colour by matched phase (CIF) with
  per-structure show/hide toggles that govern every view at once, and
  **Save official mlgidBASE figures…** writes upstream's own matplotlib
  exports.
- **Plot data + image export.** Every views-window plot exports its numbers
  to CSV — complete tables with track/structure identity per row; the
  amplitude export follows the displayed mode (median rows carry
  median/quartile columns) and matrix exports carry their axes — and its
  rendering to PNG/SVG. The image export is styled interactively: choose
  the structures to include, line width/style, marker size, output width
  and an optional white background. Both work per plot or for several
  plots at once.
- **Interpolate track (real gap filling).** For a tracked spot peak fitted
  on some frames of its span but missing in between, the Scan-tracking
  dock's **Interpolate track** button injects a detected box at the
  estimated position (size from the bracketing frames), runs the same 2D
  fit the pipeline uses, re-matches it ONLY against the structures the
  track was found in (a gap fill can never introduce a new structure), and
  adds the new peaks to the track. Ring tracks are filled along |q| with
  the ring conventions preserved; a progress dialog mirrors the pipeline
  panel's per-frame progress across the fit + re-match chain.
- **Matched display styles.** A style combo next to the Matched-peaks
  toggle: **Boxes** (as before) or **Markers** — screen-fixed circles for
  peaks and dashed arcs for rings, one colour per structure with the shape
  distinguishing ring from peak. The palette is a new set of ten
  well-separated hues, so ten structures stay tellable apart.
- **"Unmatched fitted peaks" display group.** A grey row under Matched
  peaks showing every fitted peak no loaded structure claims (off by
  default) — the whole frame can read in marker style, and tracking
  coverage is easy to judge.
- **CIF-parse progress.** A small busy indicator next to "Parse CIFs"
  while the CIF files are being parsed.
- **Open detector images (TIFF/CBF/EDF).** Standalone image files now open in
  raw mode alongside HDF5 detector files, using the same view + Conversion
  workflow (pygid reads them via `fabio`). Selecting several images — or
  dropping a folder of them — opens each as its own entry (one per file, in
  natural filename order), switchable from the entry combo and the file
  browser; converting writes each checked image to NeXus. HDF5/NeXus handling
  is unchanged.

### Fixed

- **Matched-structure colours and visibility are stable across frames.**
  Both were keyed by the structure's position in the per-frame list, so a
  structure could change colour from frame to frame and hiding it did not
  persist to other frames; both now key to the structure's identity
  (CIF + hkl).

## 0.1.0a10 — tenth alpha (2026-07-01)

Feature alpha on `0.1.0a9`. No on-disk schema or backend changes; the
`[pipeline]` pins are unchanged. Adds in-app update notifications and
self-update, plus a live raw-preview flip.

### Added

- **In-app update notifications.** On launch mlgidLAB checks GitHub for a
  newer release and, if one exists, shows a dismissible banner above the
  tabs (`Help → Check for updates…` runs the same check on demand). A first
  launch after updating also shows an offline "what's new" dialog built from
  the bundled changelog. Everything is best-effort and silent when offline.
- **One-click self-update.** For a normal pip install the banner offers an
  **Update now** button that pip-installs the new release into the current
  environment (`pip install --upgrade "mlgidlab[pipeline]? @ git+…@<tag>"`,
  run off the GUI thread), then offers to restart. `Help → Update now…` runs
  the same check + install on demand, and `Help → Automatically install
  updates` (off by default) does it on launch after one confirmation.
  Self-update is disabled for editable / development installs (the banner
  keeps only "View release"); the `[pipeline]` extra is preserved when it is
  installed.
- **Live raw-preview flip.** Ticking the Conversion panel's Flip L-R /
  Flip U-D checkboxes now flips the raw preview to match the orientation the
  conversion will write (mirrors pygid's flipud-then-fliplr on the file-order
  frame); the converted display is unchanged.

## 0.1.0a9 — ninth alpha (2026-06-30)

Feature + bugfix alpha on `0.1.0a8`. No on-disk schema or backend
changes; the `[pipeline]` pins are unchanged. Converted files written by
older mlgid versions now load and show their peaks, and ML detection
uses the GPU by default.

### Added

- **Loads converted files from older mlgid versions.** Entries are now
  recognised by `NX_class == "NXentry"`, not just an `entry`/`entry_*`
  name, so files whose entry is named after the sample (mlgidFIT writes
  `<sample>`) open and list their entry. Detected and fitted peaks are
  read from the older analysis layout (`analysis/NNNNN`,
  `detected_peaks/results`, columnar `fitted_peaks`) in addition to the
  current pygid structured format. The current format is unchanged and
  external-link masters stay on the fast open path.
- **ML detection runs on the GPU by default.** When onnxruntime's CUDA
  provider is available the detector now uses it (~0.13 s/frame vs
  ~3.9 s on CPU) instead of being pinned to the CPU by a torch-only
  availability check. The matching step's device handling is unchanged.

### Fixed

- **Per-session temporary working copies no longer pile up in the system
  temp dir.** They are PID-tagged, removed on exit (including abnormal
  exit, via an `atexit` handler), and any left by a killed prior run are
  swept on startup. The test suite likewise cleans its per-run config
  root, which previously leaked one directory per run.

## 0.1.0a8 — eighth alpha (2026-06-29)

Bugfix alpha on `0.1.0a7`. No on-disk schema or backend changes; the
`[pipeline]` pins are unchanged.

### Fixed

- **Polar view no longer mixes solid-black and transparent masked
  regions.** Grid points outside the converted image's data box are now
  filled with NaN (rendered transparent) instead of 0 (which painted a
  solid colormap-bottom block), matching the NaN-masked detector pixels
  already produced upstream. "No data" is now a single consistent value
  end to end. Affects the polar display only; the Cartesian view is
  unchanged.

## 0.1.0a7 — seventh alpha (2026-06-12)

Feature alpha on `0.1.0a6`. Large-file and raw-file performance,
conversion-workflow upgrades, and file-management additions. No on-disk
schema or backend changes; the `[pipeline]` pins are unchanged.

### Added

- **Raw files browse from the file browser.** Clicking a detector
  dataset (or its scan group) displays it in the image viewer, exactly
  like NeXus `entry_*` nodes.
- **PONI autofill.** Loading a PONI file pre-fills the Manual-override
  fields (centerX/centerY/SDD/wavelength) with the values pygid derives
  from it — a readout you can tweak instead of blank fields.
- **Append-frames conversion mode.** Converted images can be added as
  new frames of an existing entry in the output file (entry dropdown in
  the Output section), instead of always creating a new entry.
- **File-browser Refresh** (button + `F5`). Re-syncs open files with
  disk: deleted originals close (kept open with a warning when they
  have unsaved changes), files changed on disk reload when clean, and
  conflicts are reported without touching your edits.

### Changed

- **Conversion panel layout.** Flip horizontally/vertically moved out
  of Manual overrides into the Experimental-parameters form (always
  honoured when checked); Manual overrides is now a collapsible
  subsection whose fields are value-driven — "(unset)" reads from the
  PONI, a set value overrides it.
- **Entry lists keep the file's own order** (acquisition order on
  beamline masters) in the Display dropdown and Conversion selection
  tree, instead of alphabetical sorting (`10.1` no longer sorts before
  `2.1`).
- **Switching between open files is instant.** The previous file is
  restored from memory, on the entry you were viewing, with no
  re-reading from disk.
- **Re-opening an already-open file replaces the old instance** (e.g.
  the conversion auto-open after appending to an open output file) —
  one file, one entry in the browser. Unsaved changes still prompt.

### Fixed

- **Big raw files open without freezing the GUI.** Frames are read
  lazily on demand instead of materializing the whole 3D stack; the
  metadata walk runs off-thread with a real progress bar (scan progress
  for raw files, MB-by-MB copy progress for converted ones), and all
  opens — including from the Recent menu — go through the background
  worker.
- **LIMA/Eiger detector files are recognized as raw.** Files with
  `entry_*`-style roots but no mlgid data layout previously
  misclassified as NeXus and failed with "component not found".

### Install

```bash
# GUI only (view + edit existing NeXus results, in-GUI pyFAI calibration)
pip install "git+https://github.com/mlgid-project/mlgidLAB@v0.1.0a7"

# Full pipeline (adds detection / fitting / matching + raw conversion)
pip install "mlgidlab[pipeline] @ git+https://github.com/mlgid-project/mlgidLAB@v0.1.0a7"

mlgidlab        # launch
```

The `[pipeline]` extra pins the same verified-good backend set as
`0.1.0a1`: `mlgidbase==0.1.3`, `pygid==0.2.10`, `pygidfit==0.1.3`,
`mlgidmatch==0.1.3`, `pygidsim==0.1.4`.

## 0.1.0a6 — sixth alpha (2026-06-09)

Feature alpha on `0.1.0a5`. New image-viewer and export controls plus a
theme fix. No on-disk schema or backend changes; the `[pipeline]` pins
are unchanged.

### Added

- **Image aspect-ratio control** in the viewer toolbar (`Fit` /
  `Default` / `Custom`). `Default` (the startup choice) uses a per-mode
  shape — 1:1 for Cartesian, 2:1 for polar — and follows mode switches;
  `Custom` locks an on-screen width:height ratio; `Fit` stretches to
  fill. Scrolling over a single axis switches to `Custom` and adjusts
  the ratio live (x wider, y taller); double-clicking the image snaps
  back to `Default`.
- **Remove a file with `Delete`.** Pressing `Delete` with a row selected
  in the file browser closes that file, mirroring `File → Close`
  (`Ctrl+W`).
- **SVG figure export.** Tools → Export figure now saves vector **SVG**
  as well as raster PNG — the format follows the file extension you pick.

### Changed

- **Clear / Reset / delete-peak confirmations default to Yes**, so a
  single Enter confirms.

### Fixed

- **Light theme is actually light.** Both themes now apply a real
  qdarkstyle palette (dark / light) rather than falling back to the OS
  palette, so light mode reads as light on every desktop. Switching
  themes restyles the whole UI immediately — window chrome, plots, and
  the contrast slider — instead of only after a restart.

### Install

```bash
# GUI only (view + edit existing NeXus results, in-GUI pyFAI calibration)
pip install "git+https://github.com/mlgid-project/mlgidLAB@v0.1.0a6"

# Full pipeline (adds detection / fitting / matching + raw conversion)
pip install "mlgidlab[pipeline] @ git+https://github.com/mlgid-project/mlgidLAB@v0.1.0a6"

mlgidlab        # launch
```

The `[pipeline]` extra pins the same verified-good backend set as
`0.1.0a1`: `mlgidbase==0.1.3`, `pygid==0.2.10`, `pygidfit==0.1.3`,
`mlgidmatch==0.1.3`, `pygidsim==0.1.4`.

## 0.1.0a5 — fifth alpha (2026-06-05)

Documentation-only alpha on `0.1.0a4`. No code, on-disk schema, or
backend changes; the `[pipeline]` pins are unchanged. Cut to publish the
alpha manual test plan and ship the example dataset as a release asset.

### Added

- **Public manual test plan** (`docs/manual_test_plan.html`) — a
  self-contained, click-through checklist (13 areas, ~30-45 min) that
  records pass/fail per step and copies an email-ready summary. Open it
  in any browser; progress autosaves locally. Linked from the README.
- **Example dataset as a release asset.** The reference files the test
  plan uses (NeXus stacks, a raw Eiger frame + PONI/mask, the matching
  CIF pickle) ship as `example.zip` attached to this release instead of
  living in the repo, so the clone stays small. Download it and unzip
  next to the test plan.

### Install

```bash
# GUI only (view + edit existing NeXus results, in-GUI pyFAI calibration)
pip install "git+https://github.com/mlgid-project/mlgidLAB@v0.1.0a5"

# Full pipeline (adds detection / fitting / matching + raw conversion)
pip install "mlgidlab[pipeline] @ git+https://github.com/mlgid-project/mlgidLAB@v0.1.0a5"

mlgidlab        # launch
```

The `[pipeline]` extra pins the same verified-good backend set as
`0.1.0a1`: `mlgidbase==0.1.3`, `pygid==0.2.10`, `pygidfit==0.1.3`,
`mlgidmatch==0.1.3`, `pygidsim==0.1.4`.

## 0.1.0a4 — fourth alpha (2026-06-03)

Bug-fix alpha on `0.1.0a3`. No on-disk schema or backend changes;
the `[pipeline]` pins are unchanged.

### Fixed

- **Contrast no longer resets when you edit or run the pipeline.** The
  contrast set with the histogram slider is now remembered and reused
  across re-renders (adding a peak, running the pipeline, scrubbing
  frames), instead of snapping back to the auto-computed default. It
  still re-auto-contrasts when the data actually changes: opening a file,
  switching entries, or toggling log/linear. Switching Cartesian/Polar
  keeps it (same data, just resampled).

### Install

```bash
# GUI only (view + edit existing NeXus results, in-GUI pyFAI calibration)
pip install "git+https://github.com/mlgid-project/mlgidLAB@v0.1.0a4"

# Full pipeline (adds detection / fitting / matching + raw conversion)
pip install "mlgidlab[pipeline] @ git+https://github.com/mlgid-project/mlgidLAB@v0.1.0a4"

mlgidlab        # launch
```

The `[pipeline]` extra pins the same verified-good backend set as
`0.1.0a1`: `mlgidbase==0.1.3`, `pygid==0.2.10`, `pygidfit==0.1.3`,
`mlgidmatch==0.1.3`, `pygidsim==0.1.4`.

## 0.1.0a3 — third alpha (2026-06-02)

Bug-fix alpha on `0.1.0a2`. No on-disk schema or backend changes;
the `[pipeline]` pins are unchanged.

### Fixed

- **Deleting a fitted peak no longer wipes all matched structures.**
  Matched `peak_list` entries are positions into `fitted_peaks`, which
  shift when a row is removed, so the previous code cleared every
  `matched_*` solution on the frame as a blunt invalidation. It now
  reindexes instead: the deleted peak is dropped from any structure
  that referenced it, the surviving indices shift to keep pointing at
  the same peaks, structures that didn't reference it are left intact,
  and no structure is removed (one that loses its last peak is kept,
  just no longer drawn).

### Install

```bash
# GUI only (view + edit existing NeXus results, in-GUI pyFAI calibration)
pip install "git+https://github.com/mlgid-project/mlgidLAB@v0.1.0a3"

# Full pipeline (adds detection / fitting / matching + raw conversion)
pip install "mlgidlab[pipeline] @ git+https://github.com/mlgid-project/mlgidLAB@v0.1.0a3"

mlgidlab        # launch
```

The `[pipeline]` extra pins the same verified-good backend set as
`0.1.0a1`: `mlgidbase==0.1.3`, `pygid==0.2.10`, `pygidfit==0.1.3`,
`mlgidmatch==0.1.3`, `pygidsim==0.1.4`.

## 0.1.0a2 — second alpha (2026-06-02)

Second evaluation alpha. Incremental on `0.1.0a1`: a crash fix on the
hot interaction path, more robust undo/redo, a small peak-editing
addition, and repository slimming. No on-disk schema or backend
changes: `0.1.0a1` files load unchanged and the `[pipeline]` pins are
the same verified-good set.

### Fixed

- **Viewer no longer crashes during a write.** Moving the cursor over
  the polar plot, or toggling the Cartesian/Polar view, while a write
  was in flight (pipeline run, ROI commit, Add-to-fitted, clear-peaks,
  Save As) raised `RuntimeError: FrameSource not acquired` on every
  event, because the frame reader's file handle is closed for the
  duration of that write. The cursor readout now degrades to a blank
  intensity and the view toggle defers its render until the handle
  reopens, instead of throwing.
- **Undo/redo survives shortcut conflicts.** `Ctrl+Z` / `Ctrl+Shift+Z`
  / `Ctrl+Y` are now intercepted before the ambiguous-shortcut
  resolver, so they keep working even when silx mask-tools or pyFAI
  peak-picking (pulled in by the calibration dialog) register the same
  chords. Multi-level undo/redo across consecutive delete and paste
  operations was also fixed; earlier ops are no longer dropped from
  history.

### Added

- **Confidence level for Add-to-detected.** Committing a manual box to
  `detected_peaks` now writes the score chosen in the Parameter panel
  (High = 1.0 / Medium = 0.5 / Low = 0.1) instead of a fixed value; the
  add stays undoable.

### Changed

- Removed the bundled `example/` dataset (~150 MB of HDF5 / mask /
  prepared-CIF binaries) from the repository to keep clones small; the
  getting-started guide walks through opening your own data.
- README slimmed and reorganised; added
  [`docs/getting_started.md`](docs/getting_started.md) (per-OS install +
  first-file walkthrough) and
  [`docs/backend_compatibility.md`](docs/backend_compatibility.md)
  (backend version policy).

### Install

```bash
# GUI only (view + edit existing NeXus results, in-GUI pyFAI calibration)
pip install "git+https://github.com/mlgid-project/mlgidLAB@v0.1.0a2"

# Full pipeline (adds detection / fitting / matching + raw conversion)
pip install "mlgidlab[pipeline] @ git+https://github.com/mlgid-project/mlgidLAB@v0.1.0a2"

mlgidlab        # launch
```

The `[pipeline]` extra pins the same verified-good backend set as
`0.1.0a1`: `mlgidbase==0.1.3`, `pygid==0.2.10`, `pygidfit==0.1.3`,
`mlgidmatch==0.1.3`, `pygidsim==0.1.4`.

## 0.1.0a1 — first alpha (2026-05-29)

First public alpha of **mlgidLAB**, a desktop GUI for the
[mlgidBASE](https://github.com/mlgid-project/mlgidBASE) GIWAXS analysis
pipeline. It wraps the full `pygid → mlgidDETECT → pygidFIT → mlgidMATCH`
workflow — raw detector frames can be converted to NeXus, then detected,
fitted, matched, reviewed, and edited, all in one PySide6 window. The
upstream algorithms are unchanged; mlgidLAB adds visual control.

This is an **alpha** for evaluation inside the research group: the core
detect → fit → match → edit loop works end-to-end, but expect rough
edges. Please report issues (see *Known limitations* below).

### Highlights

**View & navigate**
- Open NeXus or raw HDF5 by content auto-detection (File → Open,
  drag-and-drop, or Open recent). Multiple files open at once.
- Cartesian (q_xy / q_z) ↔ polar image toggle; raw files show the
  detector image in pixels. Frame slider + Play, colormap, histogram
  levels, log/linear contrast, live cursor readout.
- Lazy frame loading with bounded per-frame LRU caches and a
  background prefetcher, so multi-GB stacks open on a laptop.

**Convert (raw → NeXus)**
- Multi-file batch conversion; GID / Transmission geometry and all
  four pygid conversion types.
- PONI + mask + angle-of-incidence inputs, with an embedded pyFAI
  calibration dialog (Create… buttons) seeded from the active scan.
- Separate-files or separate-datasets output, append-vs-overwrite.

**Detect / fit / match**
- Run Detection, Fitting, Matching (or the full pipeline) from the
  Pipeline dock; every mlgidBASE parameter is exposed.
- Matching from a preprocessed CIF pickle, raw `.cif` files, or a CIF
  folder; per-entry experimental parameters from instrument metadata.
- Run scopes: active entry / all entries / active frame / all frames.

**Edit peaks**
- Manual peaks via `Ctrl+Alt`-drag; commit with Add-to-detected or
  Add-to-fitted (2D pygidfit, or 1D scipy fallback) with a live cyan
  preview box and on-image ROI resize.
- **Multi-select** detected *or* fitted peaks with `Ctrl+click`
  (kind-aware) and `Ctrl+A` (all of the current kind on the frame).
- **Copy/paste** detected peaks with `Ctrl+C` / `Ctrl+V`, and
  **paste to a frame range** (`Ctrl+Shift+V`, e.g. `0-34,37`).
- **Batch 2D fit** ("Fit selected (2D)") over every selected detected
  peak, with a cancellable progress dialog.
- **Bulk delete** a multi-selection (`Delete`); single and bulk delete
  are both **undoable** (`Ctrl+Z` / `Ctrl+Shift+Z`), including the
  fitted 2D-shape parameters.

**Inspect & export**
- Peaks dock: sortable Detected / Fitted / Matched tables with
  bidirectional click-sync to the image; Display dock overlay toggles
  + score / probability / CIF-name filters.
- Profiles dock: live radial + angular Gaussian fits of the selected
  peak.
- Tools → Export figure… (matplotlib via mlgidbase), Export peaks as
  CSV…, Clear / Reset peaks at frame / entry / all scopes.
- Help → Controls & shortcuts (F1), About, Copy diagnostics.

### Install

Requires **Python ≥ 3.11** (Linux, macOS, Windows).

```bash
# GUI only (view + edit existing NeXus results, in-GUI pyFAI calibration)
pip install "git+https://github.com/mlgid-project/mlgidLAB@v0.1.0a1"

# Full pipeline (adds detection / fitting / matching + raw conversion)
pip install "mlgidlab[pipeline] @ git+https://github.com/mlgid-project/mlgidLAB@v0.1.0a1"

mlgidlab        # launch
```

The `[pipeline]` extra pins the verified-good backend set:
`mlgidbase==0.1.3`, `pygid==0.2.10`, `pygidfit==0.1.3`,
`mlgidmatch==0.1.3`, `pygidsim==0.1.4`. The GUI runs without them in
view-only mode (Run buttons disabled). See
[`docs/getting_started.md`](docs/getting_started.md) for per-OS install
detail and a first-file walkthrough, and
[`docs/backend_compatibility.md`](docs/backend_compatibility.md) for the
backend version policy.

### Known limitations

- **Alpha**: APIs, file layout, and UI may change before a stable
  release. Back up data before editing.
- The `[pipeline]` extra installs the heavy ML stack (torch + the
  in-house mlgid packages); first install is large.
- `mlgidMATCH` is GPL-3.0 while mlgidlab is MIT — it is an optional,
  separately-installed dependency the GUI only calls (aggregation, not
  a derived work).
- Some error paths in the calibration dialog and background workers
  log rather than surface to the UI; if an operation seems to do
  nothing, check the Logs dock.
- mlgidlab's fitted-peak add/delete touches `fitted_peaks` only, not
  the paired `fitted_peaks_errors`; re-run Matching after editing
  fitted peaks (matched indices reference the fitted-row ordering).

### License & contact

MIT — see [`LICENSE`](LICENSE). Maintainer: Nico Lerch
(<nico.lerch@uni-tuebingen.de>); issues via
https://github.com/mlgid-project/mlgidLAB/issues.
