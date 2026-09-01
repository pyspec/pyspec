"""Tests for pyspec._core -- pure numpy, no xarray dependency required."""

from __future__ import annotations

import numpy as np
import pytest

from pyspec import _core


@pytest.mark.parametrize("n", [10, 11, 21, 500, 871, 1000, 1001])
@pytest.mark.parametrize("dt", [0.25, 1.0, 1.67, 6.0, 20.0])
def test_periodogram_1d_parseval(n, dt, rng=np.random.default_rng(0)):
    """Total variance recovered from the PSD must match the variance of
    the *analyzed* (detrended + windowed) signal (Parseval's theorem).

    Note this compares against the detrended/windowed signal, not the raw
    input: detrending and windowing deliberately change the signal being
    analyzed, so Parseval only has to hold for what actually went into the
    FFT. (The original implementation's test compared against the raw
    input's variance, which only worked because of a since-fixed bug where
    the constructor mutated the caller's array in place -- by the time the
    test read `phi.var()`, `phi` itself had already been overwritten with
    the windowed signal.)
    """
    phi = rng.standard_normal(n)
    analyzed = _core.detrend(phi, "constant") * _core.hann_window(n)

    freq, psd, df = _core.periodogram_1d(phi, dt, detrend_kind="constant")

    var_analyzed = analyzed.var()
    var_spec = df * psd[1:].sum()

    assert var_spec == pytest.approx(var_analyzed, rel=1e-10)


def test_periodogram_1d_does_not_mutate_input():
    rng = np.random.default_rng(1)
    phi = rng.standard_normal(64)
    original = phi.copy()
    _core.periodogram_1d(phi, dt=1.0)
    np.testing.assert_array_equal(phi, original)


def test_periodogram_1d_frequency_grid():
    freq, psd, df = _core.periodogram_1d(np.zeros(10), dt=2.0, window=None, detrend_kind=False)
    assert freq.size == 6
    assert df == pytest.approx(0.05)
    np.testing.assert_allclose(freq, df * np.arange(6))


def test_periodogram_1d_white_noise_is_flat_on_average():
    rng = np.random.default_rng(2)
    n = 20000
    dt = 1.0
    phi = rng.standard_normal(n)
    freq, psd, df = _core.periodogram_1d(phi, dt=dt, window=None, detrend_kind=False)
    f_nyquist = 0.5 / dt
    expected_level = phi.var() / f_nyquist
    mean_level = psd[1:].mean()
    assert mean_level == pytest.approx(expected_level, rel=0.1)


@pytest.mark.parametrize("n", [10, 11, 21, 33, 128])
@pytest.mark.parametrize("d", [0.25, 1.0, 6.0])
def test_periodogram_2d_parseval(n, d, rng=np.random.default_rng(3)):
    """Checked against the actually-analyzed (windowed) field, with the
    zero-wavenumber (DC) bin excluded -- consistent with the 1-D
    convention of dropping the zero-frequency term. (These two were
    inconsistent in the original: 2-D `var` included the DC bin, 1-D
    didn't. Fixed here via `pyspec._core.variance_2d`.)
    """
    phi = rng.standard_normal((n, n))
    phi = phi - phi.mean()

    win = _core.hann_window(n)[np.newaxis, ...] * _core.hann_window(n)[..., np.newaxis]
    analyzed = phi * win

    k1, k2, psd, dk1, dk2 = _core.periodogram_2d(phi, d, d, detrend_kind=False, window="hann")
    var_spec = _core.variance_2d(k1, k2, psd, dk1, dk2, n1=n)

    assert var_spec == pytest.approx(analyzed.var(), rel=1e-9)


def test_periodogram_2d_does_not_mutate_input():
    rng = np.random.default_rng(4)
    phi = rng.standard_normal((32, 32))
    original = phi.copy()
    _core.periodogram_2d(phi, 1.0, 1.0)
    np.testing.assert_array_equal(phi, original)


def test_periodogram_2d_wavenumber_grid_matches_psd_shape():
    """Regression test for an off-by-one in the original: for odd n1, the
    original's hand-rolled arange(0., n1/2 + 1) produced one more
    one-sided wavenumber bin than rfft actually computes."""
    rng = np.random.default_rng(8)
    for n in (17, 33, 65, 128, 129):
        phi = rng.standard_normal((n, n))
        k1, k2, psd, *_ = _core.periodogram_2d(phi, 1.0, 1.0)
        assert k1.size == psd.shape[1]
        assert k2.size == psd.shape[0]


def _isotropic_average_reference(k1, k2, psd):
    """Direct port of the original pyspec loop, used only as a test oracle.

    This uses closed intervals on both sides of each bin (>= and <=).
    Every grid point sitting exactly on the diagonal k1 == |k2| lands
    exactly on a radial bin boundary -- not a floating-point coincidence,
    but pure algebra: hypot(i*dk1, i*dk2) == i*hypot(dk1, dk2) exactly,
    for any spacing. The original loop therefore double-counts every such
    diagonal point. This happens on every real 2-D spectrum along the
    whole diagonal, not just synthetic unit-spaced test grids.
    """
    dk = np.abs(k1[2] - k1[1])
    dl = np.abs(k2[2] - k2[1])
    kk1, kk2 = np.meshgrid(k1, k2)
    wv = np.sqrt(kk1**2 + kk2**2)
    kmax = min(k1.max(), k2.max())
    dkr = np.sqrt(dk**2 + dl**2)
    kr = np.arange(dkr / 2.0, kmax + dkr / 2.0, dkr)
    Er = np.zeros(kr.size)
    for i in range(kr.size):
        fkr = (wv >= kr[i] - dkr / 2) & (wv <= kr[i] + dkr / 2)
        dth = np.pi / (fkr.sum() - 1)
        Er[i] = (psd[fkr] * (wv[fkr] * dth)).sum()
    return kr, Er


@pytest.mark.parametrize("n", [17, 33, 64, 65, 127, 192, 256])
@pytest.mark.parametrize("spacing", [(1.0, 1.0), (1.37, 1.373), (2.0, 2.0)])
def test_isotropic_average_matches_reference_off_diagonal(n, spacing):
    d1, d2 = spacing
    rng = np.random.default_rng(5)
    phi = rng.standard_normal((n, n))
    k1, k2, psd, dk1, dk2 = _core.periodogram_2d(phi, d1, d2)

    kr_fast, Er_fast = _core.isotropic_average(k1, k2, psd)
    kr_ref, Er_ref = _isotropic_average_reference(k1, k2, psd)
    assert kr_fast.size == kr_ref.size

    diag_bins = set()
    for i in range(1, kr_fast.size):
        diag_bins.add(i - 1)
        diag_bins.add(i)
    off_diag = np.array([i for i in range(kr_fast.size) if i not in diag_bins])

    if off_diag.size:
        np.testing.assert_allclose(kr_fast[off_diag], kr_ref[off_diag], rtol=1e-10)
        np.testing.assert_allclose(Er_fast[off_diag], Er_ref[off_diag], rtol=1e-6)


def test_isotropic_average_no_double_counting_or_dropped_points():
    n = 64
    k1, k2, psd, dk1, dk2 = _core.periodogram_2d(np.ones((n, n)), 1.0, 1.0)
    psd_flat = np.ones_like(psd)

    kr, Er = _core.isotropic_average(k1, k2, psd_flat)
    kr_ref, Er_ref = _isotropic_average_reference(k1, k2, psd_flat)

    assert Er.sum() <= Er_ref.sum() + 1e-8

    kk1, kk2 = np.meshgrid(k1, k2)
    wv = np.sqrt(kk1**2 + kk2**2)
    dkr = np.hypot(dk1, dk2)
    bin_idx = np.maximum(np.ceil(wv / dkr).astype(int) - 1, 0)
    retained_fraction = (bin_idx < kr.size).mean()
    assert retained_fraction > 0.5


def test_isotropic_average_is_much_faster(n=256):
    import time

    rng = np.random.default_rng(6)
    phi = rng.standard_normal((n, n))
    k1, k2, psd, *_ = _core.periodogram_2d(phi, 1.0, 1.0)

    t0 = time.perf_counter()
    _core.isotropic_average(k1, k2, psd)
    t_fast = time.perf_counter() - t0

    t0 = time.perf_counter()
    _isotropic_average_reference(k1, k2, psd)
    t_ref = time.perf_counter() - t0

    assert t_fast < t_ref


def test_confidence_interval_brackets_the_estimate():
    psd = np.array([1.0, 2.0, 5.0])
    lo, hi = _core.confidence_interval(psd, dof=10, ci=0.95)
    assert np.all(lo < psd)
    assert np.all(hi > psd)


def test_confidence_interval_narrows_with_more_dof():
    psd = np.array([3.0])
    lo_few, hi_few = _core.confidence_interval(psd, dof=2, ci=0.95)
    lo_many, hi_many = _core.confidence_interval(psd, dof=200, ci=0.95)
    assert (hi_few - lo_few) > (hi_many - lo_many)


def test_confidence_interval_accepts_per_element_dof():
    psd = np.array([1.0, 1.0, 1.0])
    dof = np.array([5, 20, 100])
    lo, hi = _core.confidence_interval(psd, dof=dof, ci=0.95)
    widths = hi - lo
    assert widths[0] > widths[1] > widths[2]


def test_avg_per_decade_basic():
    k = np.logspace(-2, 2, 400)
    E = np.ones_like(k)
    ki, Ei = _core.avg_per_decade(k, E, n_bins=10)
    assert ki.size > 0
    np.testing.assert_allclose(Ei, 1.0)


def test_spectral_slope_recovers_known_power_law():
    k = np.logspace(-2, 1, 200)
    E = k**-2.0
    slope, slope_err = _core.spectral_slope(k, E, kmin=k.min(), kmax=k.max(), std_E=1.0)
    assert slope == pytest.approx(-2.0, abs=1e-8)
    assert slope_err >= 0
