"""Tests for pyspec.gm -- pure numpy, no xarray dependency required."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyspec.gm import GMParams, compute_gm_reference, coriolis_frequency

DATA_DIR = Path(__file__).parent / "data"


def test_coriolis_frequency_sign_and_magnitude():
    # Southern hemisphere -> negative f; magnitude matches the standard
    # formula f = 2*Omega*sin(lat) to within the Omega constant used
    f = coriolis_frequency(-58.0)
    assert f < 0
    assert abs(f) == pytest.approx(1.2365e-4, rel=1e-3)


def test_coriolis_frequency_zero_at_equator():
    assert coriolis_frequency(0.0) == pytest.approx(0.0, abs=1e-12)


def test_coriolis_frequency_omega_is_overridable():
    """Earth's rotation rate must be a real, overridable parameter -- not
    a hardcoded constant baked into the function."""
    f_default = coriolis_frequency(45.0)
    f_custom = coriolis_frequency(45.0, omega=1.0e-4)
    assert f_default != pytest.approx(f_custom)
    assert f_custom == pytest.approx(2.0 * 1.0e-4 * np.sin(np.radians(45.0)))


def test_compute_gm_reference_reproduces_original_drake_passage_data():
    """Regression test: this is the whole point of the rewrite. The
    original pyspec shipped a single hardcoded reference file, computed
    once for Drake Passage and silently reused as the default for any
    location. compute_gm_reference replaces that hardcoded file with an
    on-the-fly, location-parameterized calculation -- this test confirms
    it reproduces the *exact* algorithm (recovered from the original
    author's own research repository for this project) that produced the
    original file, for the original file's own parameters.

    A ~0.07% residual difference is expected and accepted: it matches
    what's seen when directly re-running the original script verbatim
    with a modern scipy (`simps` was renamed `simpson`; the two are not
    bit-identical across scipy versions for non-uniform grids), not an
    algorithmic difference.
    """
    ref = np.load(DATA_DIR / "gm_omega_star_drake_passage_reference.npz")
    k_stored, rgm_stored = ref["k"] * 1.0e3, ref["rgm"]  # cycles/m -> cycles/km

    params = GMParams(f=0.0001236454124196069, N0=3.0e-3, b=1.0e3, js=3.0)
    k, rgm = compute_gm_reference(params, kmin=k_stored.min(), kmax=k_stored.max(), n_k=k_stored.size)

    np.testing.assert_allclose(k, k_stored, rtol=1e-6)
    finite = np.isfinite(rgm) & np.isfinite(rgm_stored)
    assert finite.sum() > 0.9 * finite.size  # most points should be finite
    np.testing.assert_allclose(rgm[finite], rgm_stored[finite], rtol=2e-3)


def test_compute_gm_reference_monotonically_decreasing():
    """f^2/omega^2 should decrease with k: higher horizontal wavenumber
    at fixed vertical mode content implies higher frequency (further from
    the inertial frequency), by the internal-wave dispersion relation."""
    params = GMParams(f=coriolis_frequency(-45.0), N0=5.0e-3, b=1300.0)
    k, rgm = compute_gm_reference(params, n_k=100)
    finite = np.isfinite(rgm)
    rgm_f = rgm[finite]
    # allow for a small amount of numerical non-monotonicity but the
    # overall trend must be strongly decreasing
    assert rgm_f[0] > rgm_f[len(rgm_f) // 2] > rgm_f[-1]


def test_compute_gm_reference_bounded_between_zero_and_one():
    """f^2/omega^2 is a ratio of frequencies with omega >= f always
    (internal waves can't be slower than inertial), so this must stay
    in [0, 1]."""
    params = GMParams(f=coriolis_frequency(30.0), N0=4.0e-3, b=1300.0)
    k, rgm = compute_gm_reference(params, n_k=100)
    finite = np.isfinite(rgm)
    assert np.all(rgm[finite] >= 0.0)
    assert np.all(rgm[finite] <= 1.0 + 1e-6)


def test_compute_gm_reference_different_locations_differ():
    """Sanity check that this is now actually location-dependent (the
    entire point of this rewrite) -- different N0/f must give different
    results, not silently reuse one fixed answer."""
    k1, rgm1 = compute_gm_reference(GMParams(f=coriolis_frequency(-58), N0=3e-3, b=1000.0), n_k=50)
    k2, rgm2 = compute_gm_reference(GMParams(f=coriolis_frequency(20), N0=6e-3, b=1300.0), n_k=50)
    finite = np.isfinite(rgm1) & np.isfinite(rgm2)
    assert not np.allclose(rgm1[finite], rgm2[finite], rtol=0.05)
