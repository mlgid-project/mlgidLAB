# Changelog

All notable changes to mlgidLAB are recorded here. Versions follow
[PEP 440](https://peps.python.org/pep-0440/); `aN` suffixes are alpha
pre-releases.

## Unreleased

### Added

- **Interactive NeXus/HDF5 editing.** A new **Structure** tab, beside
  Image and Data, makes every field, attribute, group and link in the
  open file editable in place. Edits land in the session's working copy
  the moment they are committed; the file on disk is untouched until you
  save, so closing without saving is a complete revert.

  - Attributes: add, rename, retype, edit, remove. An edited attribute
    keeps its own type, and `NX_class` and `units` come with real
    suggestions instead of a blank box.
  - Structure: create groups and fields from the tree's action row or
    its context menu, with NeXus templates that pre-create what a group
    nearly always has. Rename, move, delete.
  - Links: create hard, soft and external links, see what one points at,
    retarget it, or unlink it — all without following it. **Follow** is
    the only thing that opens an external link, and only on request.
  - Copy and paste, within a file and *between* two open files, plus
    **Paste from file…** for a file that is not open at all. The
    clipboard holds a reference, so copying a large stack costs nothing
    until it is pasted, and copying a link keeps it a link.
  - A value grid for datasets: axis selectors for N-d arrays, one column
    per field for compound tables, and row insert and delete.
  - Undo and redo for all of it, with a running list of what has changed
    that can be copied out as text.
  - **Find** by name, attribute name or attribute value.
  - A **Check** section that reports what would stop the viewer or the
    pipeline from reading the file, and names the fix.
  - Raw detector files are read-only throughout: they have no working
    copy, and the editor never opens one for writing.

  - **Its own tree**, of every file you have open — so a group can be
    copied out of one and pasted into another without leaving the tab.
    It lists lazily and never follows a link, so a master of external
    scans opens as fast as any other file, and it holds no file handle
    of its own.

- The **workflow strip** above the image can be folded away with the
  chevron at its left, and remembers the choice.

- The Structure tab is a workspace rather than a page: find and the tree
  on the left, the selected node on the right, the file's health and
  this session's changes below. **Nothing scrolls at the tab level** —
  each section scrolls inside itself — and every division is a splitter
  you can drag, remembered between launches. Each region is drawn as a
  bordered pane, so a section has an edge rather than a rule that stops
  in mid-air.

- The Structure tab folds every dock away while it is in front — the
  File browser included, since the tab navigates from its own tree — and
  puts back exactly the ones that were open.

  Three things it used to do per click are now done once, when you
  leave the tab: rebuilding the File browser after a structural edit,
  switching the image to the entry you last clicked, and loading the
  frame behind the entry combo. All three update something the tab
  hides, and doing them per edit made a large file feel slow for no
  visible benefit. Switching entry from the toolbar while editing is
  now instant; the image catches up when you go back to it.

- **Shift-select in the Structure tree, and copy a whole range at once.**
  The tab's tree takes extended selections, and Copy, Cut and Delete act
  on all of it — the context menu says "Copy 6 nodes" so it is clear
  which it means. Everything else stays per-node, since there is no
  sensible reading of "rename these six".

  The whole batch is **one entry in the changes list and one Ctrl+Z**,
  not one per node. Selecting a group and one of its children copies the
  group once rather than pasting the child twice. A node whose name
  clashes still asks, and backing out of that one leaves the rest of the
  batch to land.

- **Find takes a path, and can be told to respect case.** A query with
  a slash is read as a place and then a term: `sample/material` finds a
  match for *material* under a group matching *sample*, and the parts do
  not have to be adjacent. Searching for a slash still works —
  `1/Angstrom` is a unit — because both readings run over the same walk.
  The new **Aa** button matches upper and lower case exactly; it is off
  by default, and flipping it re-runs what is in the box. Between them,
  a two-letter term like `Si` is usable in a file that also contains
  *signal* and *silicon*.

  Find now also searches **small dataset values**, which is where NeXus
  keeps most of its metadata — a sample name, a chemical formula, a
  start time. Large datasets are skipped by their declared size, so a
  detector stack is never read to answer a search.

- **Rename a node where its name is printed.** The Structure tab's
  header splits at the last `/`: where the node lives stays muted and
  read-only, and its own name can be typed over. Double-click it (or
  press F2), Enter commits, Escape puts the old name back, and so does
  clicking away — the same rename as the context menu's, with the same
  undo entry and the same warning on a protected node, minus the
  dialog. The panel now follows a renamed node instead of going on
  naming a path the rename emptied, which also fixes the header after
  the older dialog routes, and after undo and redo.

- **A crash log.** A hard crash — a segfault out of Qt, h5py or a
  driver — used to take the window with it and leave nothing behind.
  Every run now appends to `mlgidlab_crash.log` in the system temp
  directory, and a crash writes the Python stack of every thread into
  it. Nothing is written while the app is healthy beyond one line per
  start.

### Fixed

- **Opening a second file while editing the first could kill the
  window.** Not an error dialog — the process died outright. After any
  rebuild of the File browser the app puts the user's selection back,
  and that selection was being delivered to the click handler as though
  the user had made it. The handler promotes the selected file to the
  active session, which — with the Structure tab in front — tears the
  browser down again mid-restore, leaving the next line holding an index
  into a model that no longer had one. Qt's sort proxy then walked it.
  Rare, because it needed the restored selection to sit in a file that
  was not already active, which is exactly what opening a second file
  arranges. A restore is now marked as a restore, and the handler
  ignores it.

- **The view controls above the image could be dead on a single-frame
  file.** With one frame the whole frame transport is hidden, and the
  layout skips a cluster with nothing visible in it — but a skipped
  widget keeps its default size, so the emptied cluster sat on top of
  the Cartesian / Polar buttons and the colormap picker and swallowed
  every click. Loading a second, multi-frame file appeared to fix it,
  because that gave the cluster real content.

- **Flipping the detector image now moves the beam center with it.**
  Ticking *Flip LR* or *Flip UD* on its own flipped the image but left
  the beam center where it was, so the converted frame came out with its
  missing wedge mirrored relative to the data. The cause was the PONI
  autofill: it pre-fills the Manual-override boxes as a readout, and
  those values were being sent back as overrides, which put pygid in the
  one state where it applies the flips to the image but not to the
  center. An untouched override field is no longer treated as an
  override, and a center you *do* type in now genuinely wins over the
  PONI (it never reached the q maps before).
- **Peaks can be added before the pipeline has ever run.** Drawing a box
  and pressing *Add to detected* / *Add to fitted* — or committing one
  with quick select — failed on any frame that had never been through a
  detection or fitting run, with a warning telling you to run the
  pipeline first. That is every frame of a freshly converted file, so
  labelling from scratch, which is the whole point of quick select, was
  the one case that could not work. The frame's peaks table (and any
  missing analysis group above it) is now created on demand, exactly as
  pygid would have created it.

## 0.1.0a17 — seventeenth alpha (2026-08-19)

### Added

- **Quick select: label without leaving the image.** A new checkbox in
  the Display dock, with a dropdown beside it. While it is on, drawing
  the next box commits the previous one — as a detected peak, a fitted
  one, or both — so labelling a frame is draw, draw, draw instead of
  draw, cross to the panel, press a button, come back. A box drawn on
  top of the pending one replaces it instead, which is how you correct
  an attempt. The last box is committed when you click away, press
  Enter, change frame or turn the mode off; Esc still discards it. One
  Ctrl+Z turns a committed peak back into the box it was drawn as. A fit
  that will not converge never interrupts the run with a dialog — the
  box is kept as a detected peak and the reason goes to the log.
- **Select and delete peaks from the Peaks tables.** The Detected and
  Fitted tabs now take Ctrl+click, Shift+click and Ctrl+A for several
  rows at once, and the image follows along (and the other way round —
  a multi-selection made on the image now highlights every one of its
  rows in the table). Pressing Delete with rows selected removes them,
  through exactly the same confirmation and undo as the Delete key on
  the image. The Matched tab is unchanged: a row there is a whole
  structure, not a peak.
- **One fitted peak per detected peak.** A fit is now stored under the id
  of the detected peak it came from, so fitting a peak again replaces its
  fit instead of adding a second one, and the fitted peaks can no longer
  outnumber the detected ones. Deleting a detected peak deletes its fit
  too; deleting a fit leaves the detection alone unless you ask for that
  as well. A hand-drawn box is added to the detected peaks first, so
  every fit has a detection behind it. Both parts are switches in
  Settings — the link is on by default, the reverse delete is off — and
  with the link off everything behaves exactly as it did before.
- **A hover preview for peak selection.** Moving over the image outlines
  the box a click would take, and when boxes are stacked the status bar
  says how many are under the cursor.
- **A workflow rail.** A strip above the image showing the five stages
  in order — Convert, Detect, Fit, Match, Track — with what each has
  produced for the current frame. Clicking a stage brings its dock
  forward; the run glyph presses that dock's own Run button, so nothing
  about how a run is configured or queued changes.
- **A welcome page.** With no file open the window used to be empty
  axes and an empty tree. It now shows the mark, Open and Import, the
  recent files, a reminder that files can be dropped anywhere on the
  window, and — when the analysis backend is not installed — one line
  saying so, which previously only surfaced when a run failed. The
  empty Peaks and Scan-tracking tables say what would fill them.
- **An application icon.** mlgidLAB now has a window / taskbar / Alt-Tab
  icon, and a family-consistent `mlgidLAB` wordmark in the README and
  the About dialog. Both are drawn from one description of the mlgid
  family mark, whose geometry was measured off the family logo, so the
  app icon is literally the same node graph the rest of mlgid-project
  uses — recoloured to the detector colormap.
- **Icons in the menus and on the controls.** 43 monochrome SVG glyphs,
  recoloured to the live theme at load time, on the File / Edit / Tools
  / View / Settings / Help entries, the dock toggles, the dock tabs,
  the playback transport and the file-browser Refresh button.

### Changed

- **The right-hand docks open wide enough to read.** The column was
  pinned to 350 px, which compressed the Pipeline form (its
  "Config (yaml)" field showed as a stub) — the panels sit in scroll
  areas with the horizontal scrollbar locked off, so a narrow column
  elides rather than scrolls. The width is now measured from the
  panels themselves, including the sections that start collapsed, and
  the window opens at 1600x950 where the screen allows it so there is
  room for it.

- **The strips above the image wrap instead of setting a width floor.**
  The viewer's control bar and the workflow rail were single rows whose
  minimum width is the sum of everything in them, which stopped the
  central column from being dragged narrower than about 740 px. Both now
  reflow onto a second row, so the floor is the widest single cluster
  (about 280 px) and the side docks can be pulled out much further.
  Controls wrap as clusters, never between a label and its own control.

- **Primary and destructive actions now look like what they are.** Run
  detection / fitting / matching, Run full pipeline, Convert, Track
  peaks, Save figure and the calibration "Add to conversion" buttons
  carry an accent border and label; Delete peak and the metadata Remove
  read as destructive. The six "Clear" buttons deliberately stay
  neutral - they clear a path field, not analysis results.
- **Progress bars fill with the accent.** The pipeline's two run bars,
  the CIF parse bar, both status-bar bars and every progress dialog now
  fill in the theme's accent instead of the default blue, so a run in
  flight is the one moving, coloured thing on screen.
- **The status bar reports activity, not just state.** Unsaved changes
  show as a dot in front of the file name instead of a `*` glued to it;
  a pipeline run in flight colours its cell and gets a live bar, and
  clicking that cell opens the Logs dock; the cursor readout is
  monospace and now carries `d` and, where the entry has a wavelength,
  the scattering angle `2θ`. The polar map's azimuth is written `χ` so
  it cannot be misread as `2θ` beside it.
- **The viewer's control strip reads as controls.** Cartesian / Polar is
  a segmented toggle instead of two radios, Log scale is an on/off
  button, and the colormap dropdown shows each ramp as a colour chip
  rather than only its name.
- **Every section now has the same shape.** Sections were built three
  ways (a hand-bolded label, a collapsible header, a group box), so the
  Display, Pipeline and Conversion docks each read like a different
  program. They share one card now: title, hairline, body. "Selected
  peak" loses its box and joins the column it lives in.
- **Section headers, tables and the status bar were restyled.**
  Collapsible sections get a chevron and a hairline rule instead of
  Qt's stock triangle; the peaks, scan-tracking and conversion tables
  share one look; status-bar fields separate cleanly with the file name
  carrying the emphasis.
- Colours moved into a semantic token table (`mlgidlab/theme_tokens.py`)
  behind a scoped stylesheet layer (`mlgidlab/skin.py`). Third-party
  widgets - silx's tree and data viewer, pyqtgraph's plots, pyFAI's
  calibration pages - are deliberately untouched.

### Fixed

- **The welcome page draws at cold start.** The page the window opens
  on was empty: an unset label where the wordmark goes, and a "Recent"
  heading with nothing under it. Everything visible on it is filled by
  `_refresh_welcome_view`, which was only reachable from
  `_apply_session_mode` — and that first runs when a file is opened, so
  the state the page exists for was the one state it never ran in.

- **The workflow rail no longer reports one run behind.** Run detection
  and the Detect chip still said "not run"; run matching and Match said
  "not run" next to three matched phases in the Display dock. The rail
  counts what the viewer holds, and it was refreshed before the finished
  run's results were loaded into the viewer, so it always described the
  previous run. It now refreshes again after the reload.

- **The 1D profile fit stops refusing narrow peaks and inventing
  backgrounds.** It used to fit five parameters — peak plus a straight
  line — inside a window that was exactly the box, where a straight
  line cannot be identified at all. Narrow boxes got no fit (under
  about 0.025 Å⁻¹ radially there were too few samples, and the curve
  was simply cleared), and the ones that did converge could return a
  background nothing like the data, taking the amplitude with it. Now
  the window reaches beyond the box, the background is measured from
  the parts outside it and held fixed, and the peak is fitted on the
  box with three parameters. A neighbouring peak in that outside
  region is recognised and left out of the background. The drawn curve
  no longer extends past the data the fit was based on.
- **A single-peak 2D fit no longer wanders onto its neighbour.** "Add to
  fitted" and "Fit selected (2D)" handed pygidfit one box at a time.
  pygidfit masks and joint-fits only the boxes it gets in one call, so
  a neighbouring peak's intensity sat unmasked inside the target's fit
  region and could pull the result out of the box that was drawn. Both
  paths now pass the frame's nearby peak boxes along with the target,
  which is the same input the pipeline fit gets for a whole frame. Far
  boxes, rings and duplicates are filtered out, so a click still costs
  one fit rather than a frame's worth.
- **The hover outline no longer lingers.** It could survive the cursor
  leaving: a re-render while a run was in flight left it painted, and a
  re-render after the pointer had left the window redrew it. Every exit
  now clears it — leaving the plot or the viewer, a run starting, a
  view-mode switch, and closing the file.
- **Highlights cover the box they mark, and look the same on every
  kind.** The hover preview was translucent, so a detected box's dashed
  red showed through it as pink while a fitted box's cyan did not. Both
  the preview and the selection outline are now solid, opaque and wider
  than any overlay pen; the preview is the accent colour and the
  selection stays white, so "under the cursor" and "selected" remain
  distinguishable.
- **A small peak box inside a larger one is selectable.** Boxes had to
  be hit exactly, so missing an 8 px detection by three pixels selected
  the box around it — and with only that box in the stack there was
  nothing to step to. Clicks now carry a few pixels of tolerance.
- **A selected matched peak is highlighted like every other peak.** The
  structure's own box was painted over the white selection outline (the
  matched overlay is rebuilt on every render, so it ended up on top),
  leaving a sliver of white instead of a highlight. Fitted and detected
  were never affected.
- **Clicking overlapping peak boxes now has a rule you can see.** Which
  box a click took used to come down to table order, which is invisible
  (each overlay kind is drawn as one path with one pen). The familiar
  kind priority still comes first (manual, fitted, detected, matched),
  and inside a kind the **smallest box wins**, so a box drawn inside
  another is reachable instead of being shadowed by it. Clicking the
  same spot again steps to the next box underneath, so the outer one is
  still one click away: a click never hands back the box that is already
  selected.
- **Rings no longer swallow the peaks in their band.** A ring's box
  spans every angle at its radius, so it used to compete with every spot
  peak there. A ring is now picked by clicking within a few pixels of
  its inner or outer radius.
- **The light theme is correct again.** Roughly 30 colours were
  hardcoded for the dark theme and never followed a theme switch: the
  figure-export preview painted a dark rectangle inside a light window,
  status-bar separators stayed dark grey, the CIF-cache line's four
  states were washed out, table striping was near-invisible, and the
  peak-selection highlight and profile curve were white on a white
  plot ground.

## 0.1.0a16 — sixteenth alpha (2026-08-13)

### Changed

- **Internal: source layout split into focused modules.** No
  user-visible changes. ``main_window.py`` (12,500 lines) and
  ``image_viewer.py`` (4,700 lines) are now assembled from plain mixin
  modules (``main_window_*.py``, ``image_viewer_*.py``,
  ``viewer_*.py``); the conversion vocabulary moved to a Qt-free
  ``conversion_config.py`` (removing the package's only import cycle);
  small shared widgets live in ``widgets.py``; duplicated h5py, undo,
  teardown and progress-dialog idioms were collapsed into single
  helpers. Every previously importable name still resolves from its
  old module via re-export shims.

## 0.1.0a15 — fifteenth alpha (2026-08-11)

### Added

- **Integrated ROI intensity as a second tracking metric.** The
  amplitude-evolution tab gains a Metric selector: "amplitude" (the
  fitted peak amplitude, as before — the default) or "integrated
  intensity (ROI)" — the background-subtracted integrated intensity
  of a q-space box around each tracked peak, summed from the image
  data itself. The ROI trace has a value on EVERY frame: frames where
  the fit failed (or before the peak first appeared) center the box
  on the nearest fitted position, so fit dropouts no longer punch
  gaps or spurious zeros into the evolution curves, and the value is
  proportional to the diffracting volume instead of swinging with the
  fit's amplitude-width trade-off on clipped peaks. The local
  background is the pooled MEDIAN per-pixel intensity of two strips
  flanking the box radially (along q_z for near-axis peaks, along
  q_xy otherwise) times the box area — two flanks so a neighboring
  peak under one of them cannot poison the estimate, the median so a
  bright feature under part of the strips is ignored outright. Box
  half-widths (default 0.03 Å⁻¹) and strip gap/width (2 px / 4 px)
  are adjustable next to the selector. Traces are computed in the
  background (one full pass over the scan, progress in the status
  bar) and cached until tracking, the entry or the ROI settings
  change; grouping, per-structure medians, normalization, smoothing,
  the frame interval, track selection and the CSV export all work
  identically for both metrics.
- **Expected pattern: orientation modes instead of the big hkl
  list.** The overlay's orientation is now picked by mode: "matched"
  (the current frame's matched orientations for the structure,
  probability-labelled, with a clear hint when the frame has none),
  "random (powder rings)" (the orientation-free ring pattern), or
  "user-specified hkl" — type the three Miller indices directly
  instead of scrolling every symmetry-distinct orientation. Typed
  indices are validated against the structure's precomputed
  orientations: equivalent spellings (2 2 0, 0 0 -1, 0 0 1 for a
  stored 0 0 2) resolve automatically with a note, and invalid or
  unsimulated indices show a red hint plus a status-bar message
  instead of failing silently. Until the mode is touched it follows
  the data: matched when matches exist, random otherwise.
- **Amplitude CSV export: one row per frame, metric recorded.** The
  export now writes a row for EVERY frame of the exported range per
  track (or structure, in median mode) — frames without a value carry
  an explicit `nan` instead of being silently skipped, so plotting
  the CSV can no longer interpolate across fit gaps unnoticed. The
  provenance line records the selected metric (plus the ROI geometry
  when active), and the value column is named after the metric.
- **Amplitude evolution: zero baseline and brightest-tracks limit.**
  Two new controls on the phase-views amplitude tab. "Zeros at
  start/end" (median mode) pads each structure's MEDIAN curve with
  zero-intensity points from the visible start to its first observed
  frame and from its last one to the visible end, so the evolution
  rises from zero at frame 0 and falls back to zero after vanishing.
  The zeros are added to the finished median only — individual tracks
  never contribute zeros to the statistics, so a track absent on some
  frames cannot drag the median down there; gaps inside the series
  stay open. "Select tracks…" opens a per-structure table of every
  track — sortable by track id, frame count, first/last frame, mean
  |q| and mean amplitude — with a checkbox per track: untick a track
  and it drops out of that structure's grouped band and median
  immediately (the plots update live next to the modeless dialog, and
  the button shows how many tracks are off). Next to the tables, a
  non-interactive scan preview with a frame slider shows the SELECTED
  row's track on the image: clicking a row jumps the slider to the
  track's first frame, rings its peaks on the current frame in the
  structure colour and draws the full trajectory faintly — judging a
  faint or spurious track is a matter of looking, not of reading
  numbers. The selection resets on
  a new tracking run or a track deletion. The amplitude CSV export
  mirrors both controls and records them in its provenance line, and
  the phase-views frame interval now spans the whole scan (not just
  the tracked frames) so frame 0 is always reachable.
- **Frame interval for the phase views.** A window-wide "Frames:
  from..to" control in the tracking views narrows the trajectories and
  amplitude-evolution plots to that interval (members outside it are
  not drawn); the q-map and waterfall stay frame-complete. Bounds
  follow the tracked scan and reset to the full range on a new
  tracking run; the amplitude CSV export mirrors the visible interval
  and records it in its provenance line.
- **Frame range for pipeline operations.** The Frames dropdown of
  Detection, Fitting and Matching gains a "Frame range…" option with
  an expression field (comma-separated indices and inclusive A-B
  ranges, e.g. `0-34,40`): only those frames are processed. The range
  is validated against the scan when you press Run — malformed input
  or a range entirely outside the scan blocks the run with a message;
  frames partly outside are dropped with a log note. Progress totals
  follow the restricted count.
- **Pick your own colour per matched structure.** The colour swatch in
  the matched-peaks legend (Display dock and its Expected-pattern
  twin) is a button now: clicking it opens a compact grid of 40
  preset colours plus "Automatic" (back to the assigned palette
  colour) and "More…" (full colour dialog). The chosen colour follows
  the structure everywhere — image overlays in both box and marker
  style, both legends, the selected-peak swatch, and the phase-views
  window (q-map, amplitude bands, structure toggles and legend) for
  that CIF. Colours are remembered across sessions per structure
  (CIF + hkl), in any file.
- **"Single entry in single file" conversion output mode.** All
  selected images convert into ONE fresh entry of one new file — each
  image becomes a frame of a real N-frame scan instead of its own
  `entry_NNNN` group. The result browses with the frame slider and
  can be peak-tracked across the scan (impossible with N sibling
  entries), and the auto-open at the end is a single file. Mutually
  exclusive with append-to-existing-entry; a post-run check warns if
  any frame was diverted because its detector size didn't match.
- **Import pre-converted images as one scan.** For q-space maps
  produced outside mlgidLAB: File → Import images as converted scan…
  stacks N image files as one N-frame scan saved to a new .h5 (no
  conversion runs — the pixels are copied verbatim, streamed with
  constant memory). Optional q_xy/q_z ranges give the entry real
  reciprocal-space axes; without them it shows pixel axes. Opening a
  batch of float-pixel images offers this flow automatically (raw
  detector frames are integer counts; interpolated maps are float) —
  one click opens them as raw instead if the guess is wrong.
- **Imported scans can run the full pipeline when you supply the
  wavelength.** Typing the beam wavelength (with the q ranges) in the
  import dialog makes detection, fitting and matching work on the
  imported scan: the entry gets a real wavelength + incidence angle
  and documented zero placeholders for the detector fields none of
  the three operations consume (verified against mlgidbase 0.1.5 with
  a real fitting run). Without the wavelength the import is view-only
  and pipeline runs are refused with a message explaining how to
  re-import.
- **Track list: structure column and Delete-to-remove.** The
  Scan-tracking table now shows which matched crystal structures each
  track belongs to (dominant phase first, filled once Matching has
  run), and pressing Delete on a selected row removes that track from
  the results — from the table, the phase views, the only-tracked
  filter and the phase colouring. Display state only: no peaks are
  deleted from the file.
- **Progress indicator for big image-batch loads.** While the file
  browser streams a batch's rows in, the status bar shows a small
  determinate "Browser: n/N files" bar (separate from the open/load
  indicator, which the first image's async load drives). Only appears
  for batches of 10 or more files; clears when the fill completes.
- **"Select all" in the conversion Selection tree.** One checkbox
  above the file/entry tree checks or unchecks the whole batch (0.02 s
  even at 6000 image files); its state mirrors manual edits as
  checked, unchecked, or partial. A click never lands on "partial" —
  anything mixed becomes fully checked.
- **File-browser rows for image files are clickable now.** Clicking a
  TIFF/CBF/EDF row in the file browser selects that image in the entry
  dropdown and shows it in the viewer — the same behavior clicking an
  `entry_*` group has for NeXus files. Double-click additionally
  renders the image in the Data tab. Previously image rows were inert,
  which made the browser feel dead for image batches.

### Changed

- **The Controls & shortcuts reference (F1) is a real dialog now.**
  Modeless and filterable (type to narrow, e.g. "paste" or "track"),
  with grouped sections and key badges, styled for both the dark and
  light theme and following live theme switches. The inventory is
  complete: it now lists the copy/paste shortcuts, Ctrl+A,
  Ctrl+click multi-select, F11, F5, the tracking-table Delete key,
  trajectory-point and colour-swatch clicks, drag & drop, and more
  that the old static box omitted; two wrong entries were corrected
  (Esc deletes the selected manual peak; ROI dragging resizes manual
  and detected peaks only).

### Fixed

- **Phase-views colours match the Display legend.** The tracking
  views (q-map, amplitude bands, structure toggles, legend) used
  their own hue wheel and only followed the matched-peaks legend for
  hand-picked custom colours — an automatically coloured structure
  showed one colour in the Display dock and a different one in the
  views. The host now pushes the viewer's EFFECTIVE per-CIF colours
  (automatic palette plus custom picks, a pick on any hkl row of a
  CIF winning for that CIF) at window open, on every pick and on
  every tracking run, so both sides always agree.
- **Amplitude plot x-axis no longer runs away.** The amplitude
  evolution's auto-range could inflate the frame axis far beyond the
  data (thousands of frames on a 350-frame scan): the grouped view's
  pixel-sized structure labels fed back into the range computation.
  The labels are excluded from ranging now and the x-axis is pinned to
  the tracked frame interval — the same range the CSV export covers.
- **Phase colouring follows the frames matching actually claimed.** A
  track fitted on frames x..y but matched only from frame z was
  painted with its phase colour over the whole span. The views now
  attribute each tracked peak per frame: in the trajectories
  ("matched" colouring), the grouped/median amplitude bands, the
  q-map phase overlay and its legend, the claimed members render in
  the phase colour and the never-claimed ones as unmatched grey. The
  "unmatched" toggle then shows/hides exactly the unclaimed portions;
  the amplitude CSV export in median mode groups the same way.
- **Amplitude band labels no longer sit on the curves.** The structure
  labels of the grouped amplitude view were vertically centred inside
  their band; they now sit in the gap above the band's top reference
  line.
- **Opening large image batches no longer freezes the window.** With
  1000 detector TIFFs the open froze the GUI for ~8 s on warm NVMe
  (longer on cold or network storage) before anything rendered; with
  6000 the browser was unusable. Causes, all fixed: every image spun
  up its own classifier thread (classification is a pure extension
  check — it now runs inline); every image became a full silx browser
  row, each insert decoding the whole image on the GUI thread AND
  pinning its pixels for the session (~10 MB per row — the browser
  alone held ~10 GB for a 1000-file batch); and the recent-files menu
  was rebuilt once per file (now one batched update). Image files are
  now listed as lightweight name-only rows (no decode, no pixel
  memory, streamed in chunks off a timer), so ALL files of a batch
  appear in the browser regardless of size. For 6000 detector images:
  first image renders in ~0.5 s, the browser fills within ~7 s in the
  background, clicks answer in ~1 ms, and memory stays flat. Rows
  queued behind a tree detach are re-queued by the reattach, so
  pipeline runs and session closes stay consistent mid-fill (the
  reattach itself previously re-froze the window on every pipeline
  run while a big batch was open).
- **The Open dialog stays responsive in huge image directories.**
  File → Open now uses Qt's own dialog with a constant-time icon
  provider instead of the platform-native picker: native dialogs
  preview/thumbnail image files, which made a directory of thousands
  of detector TIFFs take ages to even list. The last-visited
  directory now also persists across app restarts.
- **Closing no longer lags with a big batch open.** Emptying the file
  browser (session close, app exit, and the detach before every
  pipeline run or save) removed rows one at a time — each removal a
  model event the sort proxy and view reacted to, quadratic overall.
  The browser now empties in a single model reset; h5 file handles
  are still released per row.
- **The pipeline progress bar no longer lags the log and jumps at the
  end.** The per-frame bar counted only mlgidbase's "Saved … peaks"
  lines, but frames that produce nothing end with a different line
  instead (matching: "No solutions …"; detection: "No peaks detected
  …" or "Detection failed …"). On runs with such frames the bar fell
  behind the log output and then leapt to 100% when the run finished.
  Those skip lines now count as completed frames too, so the bar
  tracks the log one-to-one. Most visible on matching, where
  no-solution frames are routine.
- **Peak tracking no longer gets the app killed on big scans — it
  runs in bounded memory instead.** mlgidBASE's tracking compares
  every fitted peak of the scan with every other one in a dense
  matrix, so its memory grows with the square of the total peak count
  (64 bytes per peak pair, measured) — a 602-frame scan with ~44k
  fitted peaks needs over 120 GB and the kernel OOM-killed the whole
  window mid-run. The pipeline now pre-estimates that need against
  the machine's available memory and, when the dense run would not
  fit, computes the identical result with a memory-safe blocked
  equivalent: the IoU streams in row blocks keeping only the
  above-threshold pairs, and tracks come from sparse connected
  components — same boxes, same IoU formula, same threshold and
  length semantics, verified edge-for-edge and payload-for-payload
  against the official upstream run. The 602-frame scan that used to
  kill the app now tracks in ~80 s within ~1.3 GB (122 tracks).
  Only the official mlgidBASE figure export still needs the dense
  upstream run; on oversized scans it refuses with the numbers and
  points at the phase-views image export instead.
- **The tracking dialog shows a busy marquee instead of freezing at
  95%.** The old bar faked progress toward 95% within seconds and
  then sat there for the rest of the run (tracking is one opaque
  mlgidBASE call with no per-frame progress). It is now an honest
  indeterminate busy bar that animates for the whole run; the
  Interpolate-track dialog keeps its real per-frame percentages.

## 0.1.0a14 — fourteenth alpha (2026-08-05)

Hotfix on `0.1.0a13`: the in-GUI updater could not update a running
install on Windows. No other changes, no on-disk schema changes.

### Fixed

- **In-GUI updates work on Windows.** "Update now" failed with
  `[WinError 32]` when the app was started via the Start-menu /
  `Scripts\mlgidlab.exe` launcher: pip's uninstall of the old version
  has to remove that exe, but Windows locks the image file of a running
  process. The updater now renames the launcher aside before running pip
  (renaming a running exe is allowed) and restores it if the install
  fails; stale backups are cleaned up on the next update. The pip child
  also no longer flashes a transient console window under the windowed
  launcher.

## 0.1.0a13 — thirteenth alpha (2026-08-04)

Hotfix on `0.1.0a12`: detection was unusable on a fresh install that had
never run mlgidDETECT from a terminal. No other changes, no on-disk
schema changes.

### Fixed

- **Detection models now download on a GUI-only install.** mlgidDETECT
  ships no `.onnx` weights; it fetches them on first use into a per-user
  cache. That download crashed in any GUI process without a usable
  console — notably on Windows, where the `gui-scripts` entry point runs
  under `pythonw.exe` and `sys.stdout` is `None`, so mlgidDETECT's
  progress bar died on its first `sys.stdout.write`. The failure was
  then invisible twice over: the aborted transfer left a **zero-byte
  `.onnx`** in the cache, which mlgidDETECT's existence-only check
  accepts forever after (so no later run retried the download), and
  every symptom surfaced as the same opaque *"Detection failed. Couldn't
  load the model."* Anyone who had installed mlgidDETECT separately, and
  so had a populated cache, was unaffected — which is why this survived
  release testing.

  The GUI now pre-flights the model before handing off to mlgidbase:
  it guarantees writable `stdout`/`stderr` for the backend call,
  downloads through a temp file that is only moved into place once the
  byte count matches the server's `Content-Length`, reports progress and
  errors through the Pipeline log panel, and names the cache directory
  and source URL when a download fails so it can be completed by hand on
  a proxied or air-gapped machine.

  Existing broken installs self-heal: truncated and zero-byte cache
  entries are removed and re-fetched on the next detection run. Complete
  models are reused as before, verified once per session against the
  server's size, and an unreachable server never invalidates a good
  cached copy.

### Changed

- `mlgiddetect==0.2.8` is now declared explicitly in the `[pipeline]`
  extra. It still resolves identically through mlgidbase; the GUI simply
  imports it directly now (for the model cache directory and URL table),
  and this repo declares what it imports.

## 0.1.0a12 — twelfth alpha (2026-07-23)

Feature alpha on `0.1.0a11`: the Expected-pattern workflow and the move
to the released mlgid backend stack. No on-disk schema changes.

### Added

- **Expected pattern workflow.** A new right-side "Expected pattern"
  dock overlays the forward-simulated reflections of a parsed CIF on
  the image (diamonds for spots, dashed arcs for rings, marker size
  encodes simulated intensity; CIF + orientation combos are fed from
  the Pipeline panel's parsed-CIF cache). Reflections are colour-coded:
  green = explained by a peak matched to the selected phase, orange =
  missed. Click markers (or **Select missed**) to select reflections,
  then **Add selected peaks (fit + match)**: each selected position is
  injected as a detected box, 2D-fitted and re-matched; injected peaks
  the matcher does not claim are rolled back. **Find all matched
  structures (all frames)** applies every matched (CIF, orientation)
  combo across the whole scan with per-structure reflection caps.
  Safety rails: matched-solution snapshots are union-merged back after
  every re-match, and duplicate matched structures are consolidated at
  write and read, so existing identifications never disappear. The dock
  also carries a matched-structures legend mirroring the Display legend
  both ways.
- **Editable frame index.** A spinbox next to the frame slider seeks to
  a typed frame number; slider, spinbox and the "/ max" label stay in
  sync with the active stack.

### Changed

- **Backend stack: released PyPI pins.** The `[pipeline]` extra moves
  to `mlgidbase==0.1.5` / `pygid==0.2.13`; `mlgiddetect 0.2.8` arrives
  transitively and carries per-Python CUDA 12 `onnxruntime-gpu` pins,
  so the GUI's own onnxruntime pin is dropped (`networkx` is now
  declared here — mlgidbase needs it but does not declare it).
  Detection defaults to the new 2-class DINO model of mlgiddetect
  0.2.6+ — detections differ from the previous single-class model; the
  legacy model stays reachable via a detection config with
  `ONNX_BASE: dino_old`. GPU selection is native to mlgiddetect now
  (ORT provider + driver probe, CPU fallback); the GUI's torch-based
  workarounds are removed.
- **Peak tracking targets the mlgidbase 0.1.5 contract.** Per-member
  amplitudes come from `track_peaks`' per-component return value; a
  too-old mlgidbase fails with a named "needs mlgidbase >= 0.1.5"
  error instead of a raw AttributeError.
- **Min-intensity cutoff is a single spinbox** with adaptive-decimal
  stepping (arrow/wheel steps scale with the current value's
  magnitude), replacing the linear slider that was useless for the
  decades-spanning simulated intensities.

### Fixed

- **Ring tracks no longer appear twice.** mlgidbase 0.1.5 can natively
  track rings stored with a finite angle (for example rings injected by
  the GUI), which would have duplicated the GUI's own radial ring
  tracking; native ring components are dropped so every physical ring
  yields exactly one track.

### Removed

- Dead code retired in a sweep: the legacy 2D-preview override path in
  the profile viewer, the unused 1D-Gaussian helpers in `fit.py`, and
  stale helpers in the polar/clipboard/conversion modules. No behavior
  change; the corresponding test module went with it.

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
