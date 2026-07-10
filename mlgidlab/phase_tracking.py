"""Adapter around mlgidBASE's official peak tracking (``track_peaks``).

Upstream (mlgidBASE commit 561edfa, "First implementation of the peak
tracking function") tracks FITTED peaks across a scan's frames: boxes
``(angle ± angle_width/2, radius ± radius_width/2)`` for every fitted
peak of every frame are IoU-compared all-against-all
(``peak_operations.calculate_iou_matrix``), thresholded into a graph,
and NetworkX connected components become the tracks; components with
more than ``length`` members survive.

The current upstream implementation RETURNS NOTHING — its only output
is two matplotlib figures drawn by
``mlgidbase.peak_operations._plot_tracked_peaks`` (module-level import
in ``peak_operations``). ``capture_tracking`` therefore swaps that
symbol for a recorder for the duration of one ``track_peaks`` call and
rebuilds the data (``TrackingPayload``) from the recorder's arguments;
no matplotlib executes. Run upstream with
``plot_params={'plot_result': True, 'save_fig': False}`` (the hook only
fires when one of the two is truthy) and ``axis='amplitude'`` (q_xy /
q_z / frame numbers are always passed; radius, |q| and angle are
derivable, so requesting the amplitude axis makes ONE call sufficient
for every view).

Everything else here is pure numpy/scipy helpers for the views: derived
axes (convention pinned to ``mlgidlab.polar.polar_to_qxyz``: angle in
degrees from +q_xy toward +q_z), per-track spans/means, amplitude
smoothing (median filter + nanmax normalization, mirroring the
prototype notebook), and best-effort reconstruction of each member's
``(frame, fitted_peak_id)`` for table→image selection sync.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import median_filter

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .file_model import PeakTable

# Axis names offered to the user. "radius" follows upstream's meaning:
# the fitted ``radius`` column IS the q magnitude, so radius == |q|.
AXIS_NAMES = ("radius", "angle", "amplitude", "q_xy", "q_z")

AXIS_LABELS = {
    "radius": "|q| (Å⁻¹)",
    "angle": "angle (°)",
    "amplitude": "amplitude (arb. u.)",
    "q_xy": "q_xy (Å⁻¹)",
    "q_z": "q_z (Å⁻¹)",
}


@dataclass
class TrackingPayload:
    """Everything one upstream ``track_peaks`` run computed.

    The four member arrays are parallel over ALL fitted-peak instances
    of the scan (every fitted row of every frame, concatenated in
    upstream's frame order). ``components`` holds the surviving tracks
    as lists of indices into those arrays. Note a component may contain
    several members of the SAME frame (upstream clusters spatially
    overlapping same-frame boxes too), so "frames present" and "member
    count" are distinct quantities.
    """

    entry: str
    threshold: float
    length: int
    q_xy: np.ndarray
    q_z: np.ndarray
    frame_num: np.ndarray
    amplitude: np.ndarray
    components: list = field(default_factory=list)  # list[list[int]]

    def __post_init__(self) -> None:
        self.q_xy = np.asarray(self.q_xy, dtype=float)
        self.q_z = np.asarray(self.q_z, dtype=float)
        self.frame_num = np.asarray(self.frame_num, dtype=int)
        self.amplitude = np.asarray(self.amplitude, dtype=float)
        n = self.q_xy.size
        if not (self.q_z.size == self.frame_num.size == self.amplitude.size == n):
            raise ValueError(
                "TrackingPayload member arrays disagree in length: "
                f"q_xy={n}, q_z={self.q_z.size}, "
                f"frame_num={self.frame_num.size}, "
                f"amplitude={self.amplitude.size}"
            )
        self.components = [
            sorted(int(i) for i in comp) for comp in self.components
        ]

    @property
    def n_members(self) -> int:
        return int(self.q_xy.size)

    @property
    def n_tracks(self) -> int:
        return len(self.components)

    @property
    def radius(self) -> np.ndarray:
        """|q| of every member. The fitted ``radius`` column upstream
        is exactly this magnitude, recomputed here from (q_xy, q_z)."""
        return np.hypot(self.q_xy, self.q_z)

    @property
    def angle(self) -> np.ndarray:
        """Polar angle in degrees, +q_xy toward +q_z — the same
        convention as ``mlgidlab.polar.polar_to_qxyz``."""
        return np.rad2deg(np.arctan2(self.q_z, self.q_xy))

    def axis_values(self, name: str) -> np.ndarray:
        if name not in AXIS_NAMES:
            raise ValueError(
                f"unknown axis {name!r}; valid: {AXIS_NAMES}"
            )
        return getattr(self, name)

    # --- Per-track derivations (k = index into components) ---

    def track_members(self, k: int) -> np.ndarray:
        """Member indices of track ``k``, sorted by frame number."""
        idx = np.asarray(self.components[k], dtype=int)
        return idx[np.argsort(self.frame_num[idx], kind="stable")]

    def track_span(self, k: int) -> tuple:
        """``(first_frame, last_frame, frames_present, n_members)``."""
        frames = self.frame_num[np.asarray(self.components[k], dtype=int)]
        return (
            int(frames.min()),
            int(frames.max()),
            int(np.unique(frames).size),
            int(frames.size),
        )

    def track_mean(self, name: str, k: int) -> float:
        vals = self.axis_values(name)[
            np.asarray(self.components[k], dtype=int)
        ]
        return float(np.nanmean(vals))


class UpstreamContractError(RuntimeError):
    """Upstream ``track_peaks`` changed shape under us.

    The GUI reads tracking data through a capture hook on
    ``mlgidbase.peak_operations._plot_tracked_peaks`` because upstream's
    first implementation returns None. If this error fires, the
    installed mlgidbase no longer matches the contract verified at
    commit 561edfa — see mlgidLAB/docs/backend_compatibility.md.
    """


@contextmanager
def capture_tracking():
    """Swap upstream's plot call for a recorder for one run.

    Usage::

        with capture_tracking() as rec:
            analysis.track_peaks(entry=..., threshold=..., length=...,
                                 axis="amplitude",
                                 plot_params={"plot_result": True,
                                              "save_fig": False})
        payload = build_payload(rec, entry, threshold, length)

    The recorder never draws anything; the original symbol is restored
    on exit even if the run raises.
    """
    import mlgidbase.peak_operations as po

    if not hasattr(po, "_plot_tracked_peaks"):
        raise UpstreamContractError(
            "mlgidbase.peak_operations has no _plot_tracked_peaks — "
            "the installed mlgidbase predates or diverged from the "
            "peak-tracking implementation (need main @ 561edfa+)."
        )

    rec: dict = {}
    orig = po._plot_tracked_peaks

    def _record(*args, **kwargs):
        # Upstream calls positionally: (plot_defaults, q_xy_all,
        # q_z_all, frame_num_all, G_comps_list, axis_arr, label,
        # plot_params). Validate the arity so a silent upstream
        # signature change becomes a loud, named failure.
        if len(args) != 8 or kwargs:
            raise UpstreamContractError(
                "_plot_tracked_peaks was called with an unexpected "
                f"signature ({len(args)} positional args, "
                f"{sorted(kwargs)} kwargs); the capture contract is "
                "pinned to mlgidBASE 561edfa."
            )
        rec["q_xy"] = np.asarray(args[1], dtype=float)
        rec["q_z"] = np.asarray(args[2], dtype=float)
        rec["frame_num"] = np.asarray(args[3], dtype=int)
        rec["components"] = [list(c) for c in args[4]]
        rec["axis_arr"] = np.asarray(args[5], dtype=float)
        rec["label"] = str(args[6])

    po._plot_tracked_peaks = _record
    try:
        yield rec
    finally:
        po._plot_tracked_peaks = orig


def build_payload(
    rec: dict, entry: str, threshold: float, length: int
) -> TrackingPayload:
    """Build a ``TrackingPayload`` from a ``capture_tracking`` record.

    The capture must have run with ``axis='amplitude'`` so the recorded
    ``axis_arr`` is the member amplitudes (every other view axis is
    derivable from q_xy/q_z).
    """
    if "q_xy" not in rec:
        raise UpstreamContractError(
            "track_peaks completed without invoking its plot hook — "
            "either plot_params disabled plotting (the capture needs "
            "plot_result=True) or upstream changed the call flow."
        )
    return TrackingPayload(
        entry=str(entry),
        threshold=float(threshold),
        length=int(length),
        q_xy=rec["q_xy"],
        q_z=rec["q_z"],
        frame_num=rec["frame_num"],
        amplitude=rec["axis_arr"],
        components=rec["components"],
    )


def smooth_normalize_amplitude(
    frames: np.ndarray,
    amplitudes: np.ndarray,
    window: int = 7,
    normalize: bool = True,
) -> tuple:
    """Frame-sorted, median-smoothed, optionally nanmax-normalized
    amplitude trace for one track — the processing the prototype
    notebook applies before plotting amplitude evolution.

    ``window`` is the median-filter size (1 = no smoothing;
    mode='nearest' like the notebook). Returns
    ``(sorted_frames, processed_amplitudes)``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    frames = np.asarray(frames)
    amps = np.asarray(amplitudes, dtype=float)
    order = np.argsort(frames, kind="stable")
    frames = frames[order]
    amps = amps[order].copy()
    if window > 1 and amps.size:
        amps = median_filter(amps, size=min(window, amps.size), mode="nearest")
    if normalize and amps.size:
        peak = np.nanmax(amps)
        if np.isfinite(peak) and peak > 0:
            amps = amps / peak
    return frames, amps


# Fit-box geometry for an interpolated RING: pygidfit classifies a box
# as a ring geometrically (its angular extent covers the reachable
# span — ``find_box_type``), so the injected box must span the whole
# 0-90 deg quadrant with FINITE numbers (an ``inf`` width breaks
# ``boxes_preprocessing``'s integer grid math). The stored fitted ring
# row still uses mlgidLAB's ring convention (``angle_width = inf``).
RING_BOX_ANGLE = 45.0
RING_BOX_ANGLE_WIDTH = 90.0


def plan_gap_fills(
    payload: TrackingPayload, ids: list, fitted_tables: dict,
    ring_tracks=frozenset(),
) -> dict:
    """Plan the detected-box injections that fill each track's GAP frames.

    A gap frame is an integer frame strictly between two of a track's
    real (anchor) frames where the track has no fitted member. For each
    such frame this returns an injection spec:

    * position (``radius`` — and ``angle`` for spots): linearly
      interpolated between the bracketing anchors (frame-distance
      weighted);
    * box size (``radius_width`` — and ``angle_width`` for spots): the
      mean of the two bracketing anchors' sizes — a seed only; the
      subsequent 2D fit refines both position and size against the
      actual image;
    * ``is_ring``: ``True`` for tracks in ``ring_tracks`` (component
      indices — the GUI-side ring tracks). A ring has no azimuthal
      position, so its spec carries the quadrant-spanning fit box
      (``RING_BOX_ANGLE`` / ``RING_BOX_ANGLE_WIDTH``) instead of an
      interpolated angle, and only radius / radius_width interpolate.

    Anchor geometry comes from the fitted tables via the reconstructed
    ``ids`` (same-frame members average into one anchor, mirroring the
    old display interpolation). Tracks with fewer than two usable
    anchors have no bracketed gaps and contribute nothing. Anchor
    frames themselves are never emitted — the real fitted peak is
    already there.

    Returns ``{frame: [{"track", "radius", "angle", "radius_width",
    "angle_width", "is_ring"}, ...]}`` — pure planning data for the
    ``interpolate_tracks`` pipeline op; nothing is read from or written
    to any file here.
    """
    fills: dict = {}
    for comp_idx, comp in enumerate(payload.components):
        is_ring_track = comp_idx in ring_tracks
        anchors: dict = {}
        for i in comp:
            tagged = ids[int(i)]
            if tagged is None:
                continue
            frame, pid = int(tagged[0]), int(tagged[1])
            table = fitted_tables.get(frame)
            if table is None or len(table) == 0:
                continue
            hits = np.flatnonzero(np.asarray(table.ids, dtype=int) == pid)
            if hits.size != 1:
                continue
            j = int(hits[0])
            if bool(table.is_ring[j]) != is_ring_track:
                continue  # defensive: never mix ring and spot anchors
            if is_ring_track:
                # Rings: radial geometry only. Stored ring rows carry
                # angle 45 / angle_width inf (or NaN from the pipeline
                # fit), neither of which may leak into interpolation.
                anchors.setdefault(frame, []).append(np.array([
                    float(table.radius[j]),
                    RING_BOX_ANGLE,
                    float(table.radius_width[j]),
                    RING_BOX_ANGLE_WIDTH,
                ]))
            else:
                anchors.setdefault(frame, []).append(np.array([
                    float(table.radius[j]),
                    float(table.angle[j]),
                    float(table.radius_width[j]),
                    float(table.angle_width[j]),
                ]))
        if len(anchors) < 2:
            continue
        seq = sorted(
            (frame, np.mean(np.vstack(rows), axis=0))
            for frame, rows in anchors.items()
        )
        for (f0, v0), (f1, v1) in zip(seq, seq[1:]):
            for frame in range(f0 + 1, f1):   # strictly between = gaps
                t = (frame - f0) / (f1 - f0)
                fills.setdefault(int(frame), []).append({
                    "track": int(comp_idx),
                    "radius": float(v0[0] + t * (v1[0] - v0[0])),
                    "angle": (
                        RING_BOX_ANGLE if is_ring_track
                        else float(v0[1] + t * (v1[1] - v0[1]))
                    ),
                    "radius_width": float((v0[2] + v1[2]) / 2.0),
                    "angle_width": (
                        RING_BOX_ANGLE_WIDTH if is_ring_track
                        else float((v0[3] + v1[3]) / 2.0)
                    ),
                    "is_ring": is_ring_track,
                })
    return fills


def _connected_components(n: int, pairs) -> list:
    """Union-find connected components over ``n`` nodes and an iterable
    of ``(a, b)`` edges. Returns a list of node-index lists.

    Kept local (no networkx) so ring tracking runs on the CI box that
    lacks the private backend — unlike upstream's spot tracking, which
    needs mlgidbase + networkx.
    """
    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def track_rings(
    payload: TrackingPayload,
    ids: list,
    fitted_tables: dict,
    threshold: float,
    length: int,
) -> list:
    """Track ring peaks across frames by 1-D radial IoU.

    mlgidBASE's ``track_peaks`` cannot track rings at all: a ring's
    ``angle_width`` is infinite, so the (angle, radius) box it builds is
    infinitely wide, every IoU against it evaluates to NaN, and
    ``_track_peaks`` zeroes NaNs — each ring ends as an isolated
    single-member component the ``length`` cut discards. Rings have no
    azimuthal position (they span the whole arc), so the correct
    overlap measure is the 1-D IoU of their radius intervals
    ``[radius - radius_width/2, radius + radius_width/2]``.

    This mirrors upstream's spot algorithm on that 1-D measure:
    all-ring-against-all-ring radial IoU, thresholded into a graph,
    connected components, components with MORE than ``length`` members
    kept. Ring members are located in the payload via the reconstructed
    ``ids`` (their ``is_ring`` flag and radius come from
    ``fitted_tables``).

    Returns a list of components (each a sorted list of payload member
    indices) to append to ``payload.components`` — so tracked rings
    flow through the table, the views, and the only-tracked filter
    exactly like tracked spots. Ambiguous / unidentifiable ring members
    are skipped (best-effort, like ``member_ids``).
    """
    members = []  # (member_index, r_lo, r_hi)
    for i in range(payload.n_members):
        tagged = ids[int(i)]
        if tagged is None:
            continue
        frame, pid = int(tagged[0]), int(tagged[1])
        table = fitted_tables.get(frame)
        if table is None or len(table) == 0:
            continue
        hits = np.flatnonzero(np.asarray(table.ids, dtype=int) == pid)
        if hits.size != 1:
            continue
        j = int(hits[0])
        if not bool(table.is_ring[j]):
            continue
        r = float(table.radius[j])
        rw = float(table.radius_width[j])
        members.append((int(i), r - rw / 2.0, r + rw / 2.0))
    if len(members) < 2:
        return []

    member_idx = np.array([m[0] for m in members])
    lo = np.array([m[1] for m in members])
    hi = np.array([m[2] for m in members])
    inter = np.clip(
        np.minimum(hi[:, None], hi[None, :])
        - np.maximum(lo[:, None], lo[None, :]),
        0.0, None,
    )
    len_i = hi - lo
    union = len_i[:, None] + len_i[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0.0, inter / union, 0.0)
    pairs = np.argwhere(np.triu(iou >= float(threshold), k=1))
    out = []
    for comp in _connected_components(len(members), pairs):
        if len(comp) > int(length):
            out.append(sorted(int(member_idx[k]) for k in comp))
    return out


def member_ids(
    payload: TrackingPayload, fitted_tables: dict
) -> list:
    """Best-effort ``(frame, fitted_peak_id)`` per payload member.

    Upstream's plot hook carries no peak ids, so they are reconstructed
    by matching each member's ``(q_xy, q_z, amplitude)`` against the
    frame's fitted ``PeakTable`` (``fitted_tables`` maps frame ->
    PeakTable or None). Both sides read the same HDF5 columns, so exact
    float equality is expected; ``np.isclose`` is the fallback. A
    member with no match or an ambiguous match yields ``None`` (the
    caller degrades to jump-to-frame-only).
    """
    out: list = []
    for i in range(payload.n_members):
        frame = int(payload.frame_num[i])
        table = fitted_tables.get(frame)
        if table is None or len(table) == 0:
            out.append(None)
            continue
        exact = (
            (np.asarray(table.q_xy) == payload.q_xy[i])
            & (np.asarray(table.q_z) == payload.q_z[i])
            & (np.asarray(table.amplitude) == payload.amplitude[i])
        )
        hits = np.flatnonzero(exact)
        if hits.size != 1:
            close = (
                np.isclose(table.q_xy, payload.q_xy[i], rtol=1e-9, atol=1e-12)
                & np.isclose(table.q_z, payload.q_z[i], rtol=1e-9, atol=1e-12)
                & np.isclose(
                    table.amplitude, payload.amplitude[i],
                    rtol=1e-9, atol=1e-12,
                )
            )
            hits = np.flatnonzero(close)
        if hits.size == 1:
            out.append((frame, int(table.ids[hits[0]])))
        else:
            out.append(None)
    return out


def match_tracks_to_structures(
    payload: TrackingPayload, ids: list, matched_tables: dict
) -> dict:
    """Map each track to the matched crystal phases (CIFs) it belongs to.

    For every track (component index ``k``), each member's reconstructed
    ``(frame, fitted_id)`` (from ``ids``) is checked against the frame's
    matched structures (``matched_tables`` maps frame ->
    ``list[MatchedStructure]``): a structure claims the member when its
    ``peaks.ids`` contain ``fitted_id``. The CIFs of all claiming
    structures are tallied across the track's members.

    Returns ``{k: [cif, ...]}`` with the CIFs sorted by descending
    member-match count then name (dominant phase first). Tracks with no
    matched member are omitted, so the map doubles as the set of
    phase-identified tracks. Pure; feeds the phase-views q-map colouring.
    """
    from collections import Counter

    # Per-frame index: fitted_id -> set of CIF names claiming it.
    per_frame: dict = {}
    for frame, structures in matched_tables.items():
        if not structures:
            continue
        claim: dict = {}
        for s in structures:
            cif = str(s.cif)
            for pid in np.asarray(s.peaks.ids, dtype=int):
                claim.setdefault(int(pid), set()).add(cif)
        per_frame[int(frame)] = claim

    out: dict = {}
    for k, comp in enumerate(payload.components):
        counts: Counter = Counter()
        for i in comp:
            tagged = ids[int(i)]
            if tagged is None:
                continue
            frame, pid = int(tagged[0]), int(tagged[1])
            for cif in per_frame.get(frame, {}).get(pid, ()):
                counts[cif] += 1
        if counts:
            # Dominant first: descending count, then name for stability.
            out[k] = [
                cif for cif, _n in sorted(
                    counts.items(), key=lambda kv: (-kv[1], kv[0])
                )
            ]
    return out


