"""Memory-safe blocked tracking (``phase_tracking.track_peaks_blocked``).

Upstream mlgidBASE ``track_peaks`` needs 64 * N**2 bytes for its dense
all-against-all IoU (the kernel OOM-killed the app at 602 frames /
43645 peaks = ~122 GB). The blocked path computes the identical
thresholded IoU graph in ~1 GB by streaming row blocks and keeping
only surviving edges; these tests pin the equivalence at the edge
level (against upstream's own ``calculate_iou_matrix`` + NetworkX)
and at the payload level (against the official ``track_peaks`` run),
plus the pipeline routing that switches to the blocked path when the
dense run would not fit in memory.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlgidlab import phase_tracking
from mlgidlab.phase_tracking import _blocked_iou_edges, track_peaks_blocked
from mlgidlab.pipeline import PipelineCommand, execute

ENTRY = "entry_0000"


def _random_boxes(n: int, seed: int) -> np.ndarray:
    """Random (x0, y0, x1, y1) boxes with real-data pathologies: a
    NaN box (NaN widths in fitted rows) and exact duplicates (IoU 1)."""
    rng = np.random.default_rng(seed)
    lo = rng.uniform(0, 10, size=(n, 2))
    wh = rng.uniform(0.1, 3, size=(n, 2))
    boxes = np.column_stack((lo, lo + wh)).astype(np.float64)
    boxes[3] = np.nan
    boxes[7] = boxes[5]
    return boxes


def _dense_edges(boxes: np.ndarray, threshold: float) -> set:
    """Upstream's exact edge set: ``calculate_iou_matrix`` with its
    thresholding sequence, then the nonzero off-diagonal pairs that
    ``nx.from_numpy_array`` would turn into graph edges."""
    from mlgidbase.peak_operations import calculate_iou_matrix

    iou = calculate_iou_matrix(boxes, boxes)
    iou[iou < threshold] = 0
    iou[np.isnan(iou)] = 0
    iou[iou >= threshold] = 1
    ii, jj = np.nonzero(iou)
    return {(int(i), int(j)) for i, j in zip(ii, jj) if i < j}


@pytest.mark.parametrize("threshold", [0.5, 0.3])
def test_blocked_edges_match_upstream_dense(threshold, monkeypatch):
    """Edge-level parity: the blocked stream yields exactly the pairs
    upstream's dense matrix marks, including NaN boxes (no edges) and
    duplicates (IoU 1). The tiny block target forces many blocks so
    the row partitioning itself is exercised."""
    pytest.importorskip("mlgidbase")
    monkeypatch.setattr(phase_tracking, "_TRACKING_BLOCK_TARGET_BYTES", 1)
    boxes = _random_boxes(300, seed=42)
    i, j = _blocked_iou_edges(boxes, threshold)
    assert set(zip(i.tolist(), j.tolist())) == _dense_edges(boxes, threshold)


def test_blocked_edge_cap_raises(monkeypatch):
    """A near-zero threshold connects everything; the documented cap
    turns the would-be graph explosion into a named error."""
    monkeypatch.setattr(phase_tracking, "_TRACKING_EDGE_CAP", 10)
    boxes = _random_boxes(50, seed=1)
    with pytest.raises(RuntimeError, match="threshold is too low"):
        _blocked_iou_edges(boxes, 1e-9)


def test_blocked_payload_matches_official_run(synthetic_fitted_scan):
    """Payload-level parity on a real file: member arrays, component
    structure and (per surviving member) amplitudes all agree with the
    official upstream run driven exactly as the pipeline drives it.
    Blocked runs first — upstream's pygid handle opens r+ and must not
    coexist with the blocked path's h5py read handle."""
    pytest.importorskip("mlgidbase")
    pytest.importorskip("networkx")
    from mlgidbase import mlgidBASE  # noqa: N814

    from mlgidlab.phase_tracking import (
        amplitudes_from_track_result,
        build_payload,
        capture_tracking,
    )

    blocked = track_peaks_blocked(
        synthetic_fitted_scan, ENTRY, threshold=0.5, length=3
    )

    analysis = mlgidBASE(filename=str(synthetic_fitted_scan))
    with capture_tracking() as rec:
        ret = analysis.track_peaks(
            entry=ENTRY, threshold=0.5, length=3, axis="radius",
            plot_params={"plot_result": True, "save_fig": False},
        )
    official = build_payload(
        rec, entry=ENTRY, threshold=0.5, length=3,
        amplitudes=amplitudes_from_track_result(ret, rec),
    )

    np.testing.assert_array_equal(blocked.frame_num, official.frame_num)
    np.testing.assert_array_equal(blocked.q_xy, official.q_xy)
    np.testing.assert_array_equal(blocked.q_z, official.q_z)
    assert blocked.components == official.components
    for comp in official.components:
        for m in comp:
            assert blocked.amplitude[m] == official.amplitude[m]


def test_pipeline_routes_big_scan_to_blocked(
    synthetic_fitted_scan, monkeypatch,
):
    """With (mocked) tiny available memory the pipeline runs the
    blocked path — before the mlgidbase import, so no pygid handle —
    and the payload matches the known result for the synthetic scan
    (one surviving track spanning frames 0-5)."""
    calls: list = []
    real = phase_tracking.track_peaks_blocked

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(phase_tracking, "available_memory_bytes", lambda: 1000)
    monkeypatch.setattr(phase_tracking, "track_peaks_blocked", _spy)

    payload = execute(
        synthetic_fitted_scan,
        PipelineCommand(
            "track_peaks", {"entry": ENTRY, "threshold": 0.5, "length": 3}
        ),
    )
    assert len(calls) == 1
    assert payload.n_tracks == 1
    assert payload.track_span(0) == (0, 5, 6, 6)
    assert payload.n_members == 7


def test_blocked_raises_on_entry_without_fitted_rows(tmp_path):
    """No fitted_peaks datasets -> ValueError (the same exception type
    upstream's np.vstack produces, so the pipeline's friendly
    "run Fitting first" wrapper covers both paths)."""
    import h5py

    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as f:
        d = f.create_group(f"{ENTRY}/data")
        d.attrs["signal"] = "img_gid_q"
        d.create_dataset("img_gid_q", data=np.zeros((2, 4, 4), "f4"))
    with pytest.raises(ValueError, match="no fitted_peaks"):
        track_peaks_blocked(path, ENTRY)
