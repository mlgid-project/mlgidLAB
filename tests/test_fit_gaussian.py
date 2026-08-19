"""Coverage of ``fit.fit_gaussian_on_axis`` — pure scipy/numpy, no Qt.

Recovery is asserted on a synthetic Gaussian + linear background with
fixed-seed noise; only the *deterministic* None guards are asserted
(scipy non-convergence is matrix-flaky and never asserted). Source:
fit.py:110-244.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlgidlab.fit import fit_gaussian_on_axis

_FWHM_K = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _synthetic(center, sigma, amplitude, slope, intercept, *, n=200, noise=0.0):
    axis = np.linspace(-10.0, 10.0, n)
    clean = (
        amplitude * np.exp(-((axis - center) ** 2) / (2.0 * sigma**2))
        + slope * axis
        + intercept
    )
    if noise:
        clean = clean + np.random.default_rng(42).normal(0.0, noise, size=n)
    return axis, clean


@pytest.mark.parametrize("center", [-3.0, 0.0, 2.5])
def test_recovers_parameters(center):
    sigma, amplitude, slope, intercept = 1.2, 50.0, 0.7, 3.0
    axis, data = _synthetic(
        center, sigma, amplitude, slope, intercept, noise=0.5
    )
    fit = fit_gaussian_on_axis(axis, data, center_init=center, width_init=2.0)
    assert fit is not None
    assert fit.center == pytest.approx(center, abs=0.2)
    assert fit.sigma == pytest.approx(sigma, rel=0.15)
    assert fit.amplitude == pytest.approx(amplitude, rel=0.15)
    assert fit.slope == pytest.approx(slope, abs=0.3)
    assert fit.intercept == pytest.approx(intercept, abs=1.5)


def test_fwhm_property():
    axis, data = _synthetic(0.0, 1.5, 40.0, 0.0, 2.0)
    fit = fit_gaussian_on_axis(axis, data, center_init=0.0, width_init=2.5)
    assert fit is not None
    assert fit.fwhm == pytest.approx(_FWHM_K * fit.sigma)


def test_none_short_axis():
    axis = np.linspace(0.0, 1.0, 5)  # len < 8
    assert fit_gaussian_on_axis(axis, axis, 0.5, 1.0) is None


def test_none_nonpositive_width():
    axis, data = _synthetic(0.0, 1.0, 10.0, 0.0, 1.0)
    assert fit_gaussian_on_axis(axis, data, 0.0, 0.0) is None
    assert fit_gaussian_on_axis(axis, data, 0.0, -1.0) is None


def test_none_nonfinite_init():
    axis, data = _synthetic(0.0, 1.0, 10.0, 0.0, 1.0)
    assert fit_gaussian_on_axis(axis, data, np.nan, 1.0) is None
    assert fit_gaussian_on_axis(axis, data, 0.0, np.inf) is None


def test_none_inverted_fit_range():
    axis, data = _synthetic(0.0, 1.0, 10.0, 0.0, 1.0)
    assert (
        fit_gaussian_on_axis(axis, data, 0.0, 1.0, fit_range=(5.0, -5.0))
        is None
    )


def test_none_too_few_samples_in_window():
    # A 10-point axis spanning a huge range with a tiny fit window:
    # fewer than 6 samples fall inside → None.
    axis = np.linspace(0.0, 1000.0, 10)
    data = np.zeros(10)
    assert (
        fit_gaussian_on_axis(
            axis, data, center_init=500.0, width_init=0.01,
            fit_range=(499.9, 500.1),
        )
        is None
    )


# -- Background pinned by the flanks, peak region only bounds the centre --
#
# Both behaviours below were user-reported failures of the old
# five-free-parameter fit inside the box: narrow boxes got no fit at
# all, and the ones that did converge invented a background. The
# synthetic here uses the real radial grid density (1024 samples over
# a 3 x 3 A^-1 q range, so ~0.0041 A^-1 per sample) because the sample
# count inside the box is exactly what used to decide the outcome.

_R_AXIS = np.linspace(0.0, float(np.sqrt(18.0)), 1024)
_R_CENTRE, _R_SIGMA, _R_AMP = 1.5, 0.012, 100.0
_R_SLOPE, _R_INTERCEPT = -5.0, 50.0


def _radial_profile(noise=2.0, extra=None, seed=0):
    """Gaussian on a sloped background, on the real radial grid.
    ``extra`` adds further ``(centre, sigma, amplitude)`` peaks."""
    y = (
        _R_AMP * np.exp(-((_R_AXIS - _R_CENTRE) ** 2) / (2.0 * _R_SIGMA ** 2))
        + _R_SLOPE * _R_AXIS + _R_INTERCEPT
    )
    for centre, sigma, amp in (extra or ()):
        y = y + amp * np.exp(-((_R_AXIS - centre) ** 2) / (2.0 * sigma ** 2))
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise, size=y.shape)
    return y


def _box(width):
    return (_R_CENTRE - width / 2.0, _R_CENTRE + width / 2.0)


def _true_background_at_centre():
    return _R_SLOPE * _R_CENTRE + _R_INTERCEPT


@pytest.mark.parametrize("width", [0.020, 0.025, 0.030, 0.050, 0.080])
def test_narrow_boxes_still_get_a_fit(width):
    """A 0.020 A^-1 box holds five samples of the radial axis. The old
    fit needed six *inside the box* and returned None; the window is
    now four times the box, so the sample count stops deciding."""
    data = _radial_profile()
    fit = fit_gaussian_on_axis(
        _R_AXIS, data, _R_CENTRE, width, fit_range=_box(width),
    )
    assert fit is not None
    assert fit.amplitude == pytest.approx(_R_AMP, rel=0.1)
    assert fit.sigma == pytest.approx(_R_SIGMA, rel=0.15)


@pytest.mark.parametrize("width", [0.025, 0.050, 0.080])
def test_the_background_lands_on_the_real_baseline(width):
    """The background is estimated from the padding around the box, so
    it has to agree with the data's own baseline under the peak. The
    old fit, with the line free inside a window that was all peak,
    returned slope +393 / intercept -489 here against a true -5 / +50."""
    data = _radial_profile()
    fit = fit_gaussian_on_axis(
        _R_AXIS, data, _R_CENTRE, width, fit_range=_box(width),
    )
    assert fit is not None
    bg_at_centre = fit.slope * _R_CENTRE + fit.intercept
    assert bg_at_centre == pytest.approx(_true_background_at_centre(), abs=8.0)


def test_a_neighbour_in_the_padding_does_not_lift_the_background():
    """The padding can contain another peak. A least-squares line
    through it would ride up on that peak and steal amplitude from the
    target; the lower-envelope baseline drops those points."""
    width = 0.05
    lo, hi = _box(width)
    neighbour_centre = hi + 0.04  # inside the padding (1.5 x box each side)
    data = _radial_profile(extra=[(neighbour_centre, 0.012, 600.0)])
    fit = fit_gaussian_on_axis(
        _R_AXIS, data, _R_CENTRE, width, fit_range=(lo, hi),
    )
    assert fit is not None
    bg_at_centre = fit.slope * _R_CENTRE + fit.intercept
    assert bg_at_centre == pytest.approx(_true_background_at_centre(), abs=15.0)
    assert fit.amplitude == pytest.approx(_R_AMP, rel=0.2)
    # The centre is bounded to the box, so the fit cannot walk onto the
    # brighter neighbour however close it sits.
    assert lo <= fit.center <= hi


def test_the_rendered_curve_never_leaves_the_fit_window():
    """A background line extrapolated past the data that constrained it
    is what used to send the overlay off the plot. ``render_range`` is
    clipped to the window the fit actually saw."""
    width = 0.05
    lo, hi = _box(width)
    data = _radial_profile()
    fit = fit_gaussian_on_axis(
        _R_AXIS, data, _R_CENTRE, width, fit_range=(lo, hi),
        render_range=(lo - 3.0 * width, hi + 3.0 * width),
    )
    assert fit is not None
    pad = 1.5 * width  # BG_WINDOW_FACTOR
    assert fit.x[0] >= lo - pad - 1e-9
    assert fit.x[-1] <= hi + pad + 1e-9
    # And the curve stays in the same order of magnitude as the data.
    assert fit.y.min() > data.min() - abs(_R_AMP)
    assert fit.y.max() < data.max() + abs(_R_AMP)


def test_the_baseline_takes_a_real_tilt_but_not_a_contaminated_flank():
    """Unit check on the flank rule. A tilted background raises one
    flank's level while both keep a similar spread, so the tilt is
    used. A peak sitting in one flank raises its level *and* its
    spread, so that flank is not trusted and the background goes flat
    at the lower level."""
    from mlgidlab.fit import _flank_baseline

    x = np.linspace(0.0, 10.0, 120)
    lo, hi = 4.0, 6.0

    tilted = 2.0 * x + 5.0
    slope, intercept = _flank_baseline(x, tilted, lo, hi)
    assert slope == pytest.approx(2.0, rel=0.1)
    assert slope * 5.0 + intercept == pytest.approx(15.0, abs=1.0)

    contaminated = np.full_like(x, 5.0)
    contaminated[x > 8.0] += 400.0
    slope, intercept = _flank_baseline(x, contaminated, lo, hi)
    assert slope == 0.0
    assert intercept == pytest.approx(5.0, abs=0.5)


def test_a_window_with_no_flanks_falls_back_to_a_flat_baseline():
    """When the peak region covers the whole window there is nothing to
    estimate a tilt from, and a tilt nothing constrains is exactly the
    bug this replaced. The baseline goes flat under the data."""
    from mlgidlab.fit import _flank_baseline

    x = np.linspace(0.0, 1.0, 64)
    y = 10.0 * np.exp(-((x - 0.5) ** 2) / (2.0 * 0.05 ** 2)) + 3.0
    slope, intercept = _flank_baseline(x, y, -1.0, 2.0)
    assert slope == 0.0
    assert intercept == pytest.approx(float(y.min()))
