"""``mlgidlab.phase_tracking`` — payload math, capture contract, helpers.

Pure numpy tests run everywhere; the capture / integration tests need
the private ``mlgidbase`` backend (plus its undeclared ``networkx``
dependency) and skip cleanly without it.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlgidlab.file_model import MatchedStructure, PeakTable
from mlgidlab.phase_tracking import (
    AXIS_NAMES,
    TrackingPayload,
    UpstreamContractError,
    build_payload,
    match_tracks_to_structures,
    member_ids,
    smooth_normalize_amplitude,
)
from mlgidlab.polar import polar_to_qxyz


def _payload(**overrides):
    """Two tracks over 4 frames: persistent (members 0-3) + a frame-2
    pair (members 4-5 — same frame twice, exercising the frames ≠
    members distinction)."""
    kw = dict(
        entry="entry_0000",
        threshold=0.5,
        length=1,
        q_xy=np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        q_z=np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0]),
        frame_num=np.array([0, 1, 2, 3, 2, 2]),
        amplitude=np.array([10.0, 12.0, 14.0, 16.0, 5.0, 6.0]),
        components=[[0, 1, 2, 3], [4, 5]],
    )
    kw.update(overrides)
    return TrackingPayload(**kw)


# --- TrackingPayload ---

def test_payload_derived_axes_match_polar_convention():
    p = _payload()
    # radius/angle must invert polar_to_qxyz (the pinned convention).
    q_xy, q_z = polar_to_qxyz(p.radius, p.angle)
    np.testing.assert_allclose(q_xy, p.q_xy, atol=1e-12)
    np.testing.assert_allclose(q_z, p.q_z, atol=1e-12)


def test_payload_track_span_counts_frames_and_members():
    p = _payload()
    assert p.track_span(0) == (0, 3, 4, 4)
    # Two members on the same frame: 1 distinct frame, 2 members.
    assert p.track_span(1) == (2, 2, 1, 2)


def test_payload_track_members_sorted_by_frame():
    p = _payload(components=[[3, 0, 2, 1], [4, 5]])
    assert list(p.track_members(0)) == [0, 1, 2, 3]


def test_payload_track_mean_and_axis_values():
    p = _payload()
    assert p.track_mean("amplitude", 0) == pytest.approx(13.0)
    assert p.track_mean("q_z", 1) == pytest.approx(2.0)
    for name in AXIS_NAMES:
        assert p.axis_values(name).shape == (6,)
    with pytest.raises(ValueError):
        p.axis_values("nope")


def test_payload_rejects_mismatched_arrays():
    with pytest.raises(ValueError):
        _payload(q_z=np.array([1.0, 2.0]))


# --- smooth_normalize_amplitude ---

def test_amplitude_window_one_is_sorted_passthrough():
    frames = np.array([3, 1, 2])
    amps = np.array([30.0, 10.0, 20.0])
    f, a = smooth_normalize_amplitude(frames, amps, window=1, normalize=False)
    assert list(f) == [1, 2, 3]
    np.testing.assert_allclose(a, [10.0, 20.0, 30.0])


def test_amplitude_median_smooths_spike():
    frames = np.arange(5)
    amps = np.array([1.0, 1.0, 100.0, 1.0, 1.0])
    _f, a = smooth_normalize_amplitude(frames, amps, window=3, normalize=False)
    assert a[2] == pytest.approx(1.0)  # the spike is median-filtered away


def test_amplitude_normalize_by_nanmax():
    frames = np.arange(3)
    amps = np.array([2.0, 4.0, 8.0])
    _f, a = smooth_normalize_amplitude(frames, amps, window=1, normalize=True)
    assert np.nanmax(a) == pytest.approx(1.0)
    np.testing.assert_allclose(a, [0.25, 0.5, 1.0])


def test_amplitude_rejects_bad_window():
    with pytest.raises(ValueError):
        smooth_normalize_amplitude(np.arange(2), np.ones(2), window=0)


# --- member_ids ---

def _table(rows):
    """PeakTable from [(id, q_xy, q_z, amplitude), ...]."""
    n = len(rows)
    ids = np.array([r[0] for r in rows], dtype=int)
    q_xy = np.array([r[1] for r in rows], dtype=float)
    q_z = np.array([r[2] for r in rows], dtype=float)
    amp = np.array([r[3] for r in rows], dtype=float)
    zeros = np.zeros(n, dtype=float)
    return PeakTable(
        q_xy=q_xy, q_z=q_z, angle=zeros.copy(), radius=zeros.copy(),
        angle_width=zeros.copy(), radius_width=zeros.copy(),
        is_ring=np.zeros(n, dtype=bool), ids=ids,
        score=zeros.copy(), amplitude=amp,
    )


def test_member_ids_exact_and_missing_and_ambiguous():
    p = _payload()
    tables = {
        0: _table([(7, 1.0, 1.0, 10.0)]),
        1: None,                                # no fitted table
        2: _table([                             # frame 2 holds four rows:
            (0, 1.0, 1.0, 14.0),                #  the persistent member,
            (1, 0.5, 2.0, 5.0),                 #  blip amp-5.0,
            (2, 0.5, 2.0, 5.0),                 #  its exact duplicate,
            (3, 0.5, 2.0, 6.0),                 #  blip amp-6.0 (unique)
        ]),
        3: _table([(9, 1.0, 1.0, 16.0)]),
    }
    ids = member_ids(p, tables)
    assert ids[0] == (0, 7)
    assert ids[1] is None            # frame without a table
    assert ids[2] == (2, 0)
    assert ids[3] == (3, 9)
    assert ids[4] is None            # ambiguous duplicate coordinates
    assert ids[5] == (2, 3)          # amplitude 6.0 is unique -> id 3


def test_member_ids_isclose_fallback():
    p = _payload(components=[[0]])
    eps = 1e-15
    tables = {0: _table([(3, 1.0 + eps, 1.0 - eps, 10.0 + eps)])}
    ids = member_ids(p, tables)
    assert ids[0] == (0, 3)


# --- plan_gap_fills ---

def _fitted_table_for(frame_specs):
    """{frame: PeakTable} from {frame: [(id, radius, angle, rw, aw[, is_ring]), ...]}."""
    out = {}
    for frame, rows in frame_specs.items():
        n = len(rows)
        zeros = np.zeros(n, dtype=float)
        radius = np.array([r[1] for r in rows], dtype=float)
        angle = np.array([r[2] for r in rows], dtype=float)
        out[frame] = PeakTable(
            q_xy=radius * np.cos(np.deg2rad(angle)),
            q_z=radius * np.sin(np.deg2rad(angle)),
            angle=angle, radius=radius,
            angle_width=np.array([r[4] for r in rows], dtype=float),
            radius_width=np.array([r[3] for r in rows], dtype=float),
            is_ring=np.array(
                [bool(r[5]) if len(r) > 5 else False for r in rows]
            ),
            ids=np.array([r[0] for r in rows], dtype=int),
            score=zeros.copy(), amplitude=zeros.copy() + 10.0,
        )
    return out


def test_plan_gap_fills_interpolates_gaps_only():
    from mlgidlab.phase_tracking import plan_gap_fills

    # One track, members on frames 0, 1, 3, 6 -> fills ONLY the gap
    # frames (2, 4, 5): position interpolated between the bracketing
    # anchors, size = mean of their sizes. Anchor frames are never
    # emitted — the real fitted peak is already there.
    frames = np.array([0, 1, 3, 6])
    radius = np.array([1.00, 1.10, 1.30, 1.60])
    rw = {0: 0.20, 1: 0.20, 3: 0.30, 6: 0.50}
    payload = TrackingPayload(
        entry="e", threshold=0.5, length=1,
        q_xy=radius, q_z=np.zeros(4), frame_num=frames,
        amplitude=np.full(4, 10.0), components=[[0, 1, 2, 3]],
    )
    ids = [(int(f), 0) for f in frames]
    tables = _fitted_table_for({
        int(f): [(0, float(r), 45.0, rw[int(f)], 5.0)]
        for f, r in zip(frames, radius)
    })
    plan = plan_gap_fills(payload, ids, tables)
    assert sorted(plan) == [2, 4, 5]                        # gaps only
    assert all(len(v) == 1 for v in plan.values())
    # Position interpolates between the bracketing anchors.
    assert plan[2][0]["radius"] == pytest.approx(1.20)      # midpoint 1->3
    assert plan[4][0]["radius"] == pytest.approx(1.40)      # 1/3 of 3->6
    assert plan[5][0]["radius"] == pytest.approx(1.50)      # 2/3 of 3->6
    assert plan[2][0]["angle"] == pytest.approx(45.0)
    # Size = mean of the bracketing anchors' sizes (NOT interpolated).
    assert plan[2][0]["radius_width"] == pytest.approx(0.25)  # (0.2+0.3)/2
    assert plan[4][0]["radius_width"] == pytest.approx(0.40)  # (0.3+0.5)/2
    assert plan[5][0]["radius_width"] == pytest.approx(0.40)  # same brackets
    assert plan[2][0]["angle_width"] == pytest.approx(5.0)
    assert plan[2][0]["track"] == 0


def test_plan_gap_fills_needs_two_anchors_and_skips_gapless():
    from mlgidlab.phase_tracking import plan_gap_fills

    payload = TrackingPayload(
        entry="e", threshold=0.5, length=1,
        q_xy=np.array([1.0]), q_z=np.zeros(1),
        frame_num=np.array([2]), amplitude=np.array([10.0]),
        components=[[0]],
    )
    # Single anchor (and a member with no reconstructed id) -> nothing.
    tables = _fitted_table_for({2: [(0, 1.0, 45.0, 0.2, 5.0)]})
    assert plan_gap_fills(payload, [(2, 0)], tables) == {}
    assert plan_gap_fills(payload, [None], tables) == {}

    # A gapless track (members on every frame of its span) -> nothing.
    payload2 = TrackingPayload(
        entry="e", threshold=0.5, length=1,
        q_xy=np.zeros(3), q_z=np.zeros(3),
        frame_num=np.array([0, 1, 2]), amplitude=np.full(3, 10.0),
        components=[[0, 1, 2]],
    )
    ids2 = [(0, 0), (1, 0), (2, 0)]
    tables2 = _fitted_table_for({
        f: [(0, 1.0, 45.0, 0.2, 5.0)] for f in (0, 1, 2)
    })
    assert plan_gap_fills(payload2, ids2, tables2) == {}


def test_plan_gap_fills_rings_flagged_with_quadrant_box():
    """Ring tracks interpolate too: radial geometry only, flagged
    ``is_ring``, and carrying the finite quadrant-spanning fit box
    (angle 45 / width 90) instead of an interpolated angle."""
    from mlgidlab.phase_tracking import (
        RING_BOX_ANGLE, RING_BOX_ANGLE_WIDTH, plan_gap_fills,
    )

    # Two tracks with a gap on frame 1: track 0 is a spot, track 1 a
    # ring (ring anchor rows: is_ring=True, angle_width inf).
    payload = TrackingPayload(
        entry="e", threshold=0.5, length=1,
        q_xy=np.zeros(4), q_z=np.zeros(4),
        frame_num=np.array([0, 2, 0, 2]), amplitude=np.full(4, 10.0),
        components=[[0, 1], [2, 3]],
    )
    ids = [(0, 0), (2, 0), (0, 1), (2, 1)]
    tables = _fitted_table_for({
        0: [(0, 1.0, 45.0, 0.2, 5.0),
            (1, 3.0, 45.0, 0.2, np.inf, True)],
        2: [(0, 1.0, 45.0, 0.2, 5.0),
            (1, 3.2, 45.0, 0.4, np.inf, True)],
    })
    plan = plan_gap_fills(payload, ids, tables, ring_tracks={1})
    assert sorted(plan) == [1] and len(plan[1]) == 2
    spot = next(s for s in plan[1] if not s["is_ring"])
    ring = next(s for s in plan[1] if s["is_ring"])
    assert spot["track"] == 0 and spot["angle_width"] == pytest.approx(5.0)
    assert ring["track"] == 1
    assert ring["radius"] == pytest.approx(3.1)          # midpoint 3.0->3.2
    assert ring["radius_width"] == pytest.approx(0.3)    # mean of sizes
    # The ring's fit box is the finite quadrant span, never the stored
    # inf convention (inf breaks pygidfit's grid math).
    assert ring["angle"] == pytest.approx(RING_BOX_ANGLE)
    assert ring["angle_width"] == pytest.approx(RING_BOX_ANGLE_WIDTH)
    # An unflagged ring track finds no matching (spot) anchors -> its
    # ring rows never seed a spot fit.
    unflagged = plan_gap_fills(payload, ids, tables)
    assert [s["track"] for s in unflagged.get(1, [])] == [0]


# --- track_rings ---

def _ring_tables(frame_specs):
    """{frame: PeakTable of rings} from {frame: [(id, radius), ...]}.

    Rings carry ``is_ring=True`` and ``angle_width=inf`` — the shape
    that defeats upstream's tracker and forces the GUI-side 1-D radial
    pass."""
    out = {}
    for frame, rows in frame_specs.items():
        n = len(rows)
        z = np.zeros(n, dtype=float)
        out[frame] = PeakTable(
            q_xy=z.copy(), q_z=z.copy(), angle=z.copy(),
            radius=np.array([r[1] for r in rows], dtype=float),
            angle_width=np.full(n, np.inf),
            radius_width=np.full(n, 0.2),
            is_ring=np.ones(n, dtype=bool),
            ids=np.array([r[0] for r in rows], dtype=int),
            score=z.copy(), amplitude=z.copy() + 10.0,
        )
    return out


def test_track_rings_links_persistent_ring_by_radius():
    from mlgidlab.phase_tracking import track_rings

    # A ring at r=1.8 on every frame 0-5 (the persistent phase ring)
    # plus a one-off ring at r=2.5 on frame 2 only. Member arrays are in
    # upstream's frame order; upstream tracked nothing (components=[]).
    member_frames = [0, 1, 2, 2, 3, 4, 5]
    member_pids = [0, 0, 0, 1, 0, 0, 0]
    n = len(member_frames)
    payload = TrackingPayload(
        entry="e", threshold=0.5, length=3,
        q_xy=np.zeros(n), q_z=np.zeros(n),
        frame_num=np.array(member_frames), amplitude=np.full(n, 10.0),
        components=[],
    )
    ids = list(zip(member_frames, member_pids))
    tables = {f: [(0, 1.8)] for f in range(6)}
    tables[2] = [(0, 1.8), (1, 2.5)]
    comps = track_rings(payload, ids, _ring_tables(tables), 0.5, 3)
    # The persistent ring (6 members, indices 0,1,2,4,5,6) survives; the
    # one-off (index 3, isolated) is a single-member component cut by
    # length.
    assert comps == [[0, 1, 2, 4, 5, 6]]


def test_track_rings_ignores_spots_and_short_tracks():
    from mlgidlab.phase_tracking import track_rings

    # Two spots (not rings) + a ring on just one frame -> nothing:
    # spots are excluded by is_ring, the lone ring is < length+1.
    payload = TrackingPayload(
        entry="e", threshold=0.5, length=1,
        q_xy=np.zeros(3), q_z=np.zeros(3),
        frame_num=np.array([0, 1, 2]), amplitude=np.full(3, 10.0),
        components=[],
    )
    ids = [(0, 0), (1, 0), (2, 0)]
    spot = _fitted_table_for({
        0: [(0, 1.0, 45.0, 0.2, 5.0)],
        1: [(0, 1.0, 45.0, 0.2, 5.0)],
    })
    spot.update(_ring_tables({2: [(0, 1.8)]}))
    assert track_rings(payload, ids, spot, 0.5, 1) == []


def test_track_rings_needs_overlapping_radii():
    from mlgidlab.phase_tracking import track_rings

    # Two rings at very different radii never link, however many frames.
    member_frames = [0, 1, 2, 3]
    payload = TrackingPayload(
        entry="e", threshold=0.5, length=1,
        q_xy=np.zeros(4), q_z=np.zeros(4),
        frame_num=np.array(member_frames), amplitude=np.full(4, 10.0),
        components=[],
    )
    ids = [(f, 0) for f in member_frames]
    tables = _ring_tables({0: [(0, 1.0)], 1: [(0, 1.0)],
                           2: [(0, 3.5)], 3: [(0, 3.5)]})
    comps = track_rings(payload, ids, tables, 0.5, 1)
    # Two separate 2-member ring tracks (r=1.0 and r=3.5), each > 1.
    assert sorted(comps) == [[0, 1], [2, 3]]


# --- match_tracks_to_structures ---

def _matched(cif, ids):
    """Minimal MatchedStructure with the given cif + fitted ids in peaks."""
    n = len(ids)
    z = np.zeros(n, dtype=float)
    peaks = PeakTable(
        q_xy=z.copy(), q_z=z.copy(), angle=z.copy(), radius=z.copy(),
        angle_width=z.copy(), radius_width=z.copy(),
        is_ring=np.zeros(n, dtype=bool), ids=np.array(ids, dtype=int),
        score=z.copy(), amplitude=z.copy(),
    )
    return MatchedStructure(
        solution_field="matched_segments_0000", local_idx=0, cif=cif,
        h=1, k=1, l=0, probability=0.9, peaks=peaks,
        peak_list=np.array(ids, dtype=int),
    )


def _two_track_payload():
    # track 0: members (f0,id0),(f1,id0),(f2,id0); track 1: (f0,id5),(f1,id5)
    payload = TrackingPayload(
        entry="e", threshold=0.5, length=1,
        q_xy=np.zeros(5), q_z=np.zeros(5),
        frame_num=np.array([0, 1, 2, 0, 1]), amplitude=np.full(5, 10.0),
        components=[[0, 1, 2], [3, 4]],
    )
    ids = [(0, 0), (1, 0), (2, 0), (0, 5), (1, 5)]
    return payload, ids


def test_match_tracks_to_structures_maps_tracks_to_cifs():
    payload, ids = _two_track_payload()
    matched = {
        0: [_matched("A", [0]), _matched("B", [5])],
        2: [_matched("A", [0])],
    }
    # Track 0's fitted id 0 is matched by CIF A (frames 0 and 2);
    # track 1's fitted id 5 by CIF B (frame 0).
    assert match_tracks_to_structures(payload, ids, matched) == {
        0: ["A"], 1: ["B"],
    }


def test_match_tracks_to_structures_orders_by_dominance():
    payload, ids = _two_track_payload()
    # Track 0's id 0 is claimed by A on frames 0 & 2 (count 2) and B on
    # frame 0 (count 1) -> dominant A first.
    matched = {
        0: [_matched("A", [0]), _matched("B", [0])],
        2: [_matched("A", [0])],
    }
    out = match_tracks_to_structures(payload, ids, matched)
    assert out[0] == ["A", "B"]


def test_match_tracks_to_structures_omits_unmatched_and_empty():
    payload, ids = _two_track_payload()
    # Only track 1 (id 5) is matched; track 0 is omitted entirely.
    matched = {0: [_matched("B", [5])]}
    assert match_tracks_to_structures(payload, ids, matched) == {1: ["B"]}
    # No matched data -> empty map.
    assert match_tracks_to_structures(payload, ids, {}) == {}
    # id-less members contribute nothing.
    assert match_tracks_to_structures(
        payload, [None] * 5, matched
    ) == {}


# --- capture contract (needs mlgidbase) ---
# importorskip is called INSIDE each test (not at module level) so the
# pure tests above still run on CI boxes without the private backend.


def test_capture_records_hook_args():
    pytest.importorskip("mlgidbase")
    from mlgidlab.phase_tracking import capture_tracking
    import mlgidbase.peak_operations as po

    with capture_tracking() as rec:
        # Simulate upstream's positional call.
        po._plot_tracked_peaks(
            {}, np.array([1.0]), np.array([2.0]), np.array([0]),
            [[0]], np.array([5.0]), "label", {},
        )
    payload = build_payload(rec, entry="e", threshold=0.4, length=2)
    assert payload.n_members == 1 and payload.n_tracks == 1
    assert payload.amplitude[0] == pytest.approx(5.0)
    assert payload.threshold == 0.4 and payload.length == 2
    # The original symbol is restored on exit.
    assert po._plot_tracked_peaks.__name__ != "_record"


def test_capture_rejects_signature_drift():
    pytest.importorskip("mlgidbase")
    from mlgidlab.phase_tracking import capture_tracking
    import mlgidbase.peak_operations as po

    with capture_tracking():
        with pytest.raises(UpstreamContractError):
            po._plot_tracked_peaks(1, 2, 3)  # wrong arity


def test_build_payload_requires_fired_hook():
    with pytest.raises(UpstreamContractError):
        build_payload({}, entry="e", threshold=0.5, length=1)


def test_track_peaks_integration(synthetic_fitted_scan):
    """Real upstream run through pipeline.execute: the persistent peak
    forms one surviving track; the frame-2 blip's single-member
    component is cut by ``length``."""
    pytest.importorskip("mlgidbase")
    pytest.importorskip("networkx")
    from mlgidlab.pipeline import PipelineCommand, execute

    payload = execute(
        synthetic_fitted_scan,
        PipelineCommand("track_peaks", {
            "entry": "entry_0000", "threshold": 0.5, "length": 3,
        }),
    )
    assert isinstance(payload, TrackingPayload)
    assert payload.n_members == 7          # 6 persistent + 1 blip
    assert payload.n_tracks == 1
    assert payload.track_span(0) == (0, 5, 6, 6)
    assert payload.track_mean("radius", 0) == pytest.approx(1.005, abs=1e-3)
    # Components partition into valid member indices.
    members = payload.track_members(0)
    assert set(members) <= set(range(payload.n_members))
    assert (payload.frame_num[members] == np.arange(6)).all()


def test_track_peaks_old_backend_named_error(
    synthetic_fitted_scan, monkeypatch,
):
    """An mlgidbase without track_peaks (the 0.1.3 release) fails with
    the named capability message, not a raw AttributeError."""
    pytest.importorskip("mlgidbase")
    from mlgidbase import mlgidBASE

    from mlgidlab.pipeline import PipelineCommand, execute

    monkeypatch.delattr(mlgidBASE, "track_peaks")
    with pytest.raises(RuntimeError, match="newer than the 0.1.3"):
        execute(
            synthetic_fitted_scan,
            PipelineCommand("track_peaks", {
                "entry": "entry_0000", "threshold": 0.5, "length": 3,
            }),
        )


def test_track_peaks_no_fitted_peaks_message(synthetic_fitted_scan):
    """A file with no fitted rows fails with the actionable message,
    not numpy's vstack error. (The file must still pass pygid's entry
    validation, so strip the analysis groups from the valid fixture.)"""
    pytest.importorskip("mlgidbase")
    pytest.importorskip("networkx")
    import h5py

    from mlgidlab.pipeline import PipelineCommand, execute

    with h5py.File(synthetic_fitted_scan, "r+") as f:
        del f["entry_0000/data/analysis"]
    with pytest.raises(RuntimeError, match="run Fitting"):
        execute(
            synthetic_fitted_scan,
            PipelineCommand("track_peaks", {
                "entry": "entry_0000", "threshold": 0.5, "length": 3,
            }),
        )
