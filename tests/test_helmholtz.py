"""Tests for pyspec.helmholtz.spec_helm_decomp. Requires xarray."""

from __future__ import annotations

import numpy as np
import xarray as xr

from pyspec import HelmholtzDecomposition, spec_helm_decomp


def test_returns_dataclass_with_psi_and_phi():
    k = np.logspace(-2, 1, 40)
    Cu = k**-2.0
    Cv = k**-2.0
    result = spec_helm_decomp(k, Cu, Cv)

    assert isinstance(result, HelmholtzDecomposition)
    assert isinstance(result.psi, xr.DataArray)
    assert isinstance(result.phi, xr.DataArray)
    assert result.u_wave is None


def test_finite_output_for_power_law_input():
    k = np.logspace(-2, 1, 60)
    Cu = Cv = k**-2.0
    result = spec_helm_decomp(k, Cu, Cv)
    assert np.all(np.isfinite(result.psi.values))
    assert np.all(np.isfinite(result.phi.values))


def test_psi_plus_phi_reconstructs_velocity_variance_not_ke():
    """Locks in the convention documented in the module docstring: psi
    and phi are velocity variance spectra, in the same units as Cu/Cv --
    not kinetic energy spectra. psi + phi must reconstruct Cu + Cv
    exactly, not 0.5*(Cu + Cv) (which is what it would be if the KE
    factor of 1/2 were already baked in). Getting this backwards is an
    easy, real mistake (see MIGRATION.md / the example notebooks) --
    this test exists so it can't silently regress."""
    k = np.logspace(-2, 1, 60)
    Cu = k**-2.0
    Cv = 0.8 * k**-2.0  # deliberately asymmetric, so this isn't trivially true by symmetry
    result = spec_helm_decomp(k, Cu, Cv)

    np.testing.assert_allclose(
        (result.psi + result.phi).values, Cu + Cv, rtol=1e-10
    )
    # and explicitly NOT half of that
    assert not np.allclose((result.psi + result.phi).values, 0.5 * (Cu + Cv), rtol=0.1)


def test_accepts_dataarray_input_and_preserves_dim_name():
    k = xr.DataArray(np.logspace(-2, 1, 30), dims=["wavenumber"])
    Cu = xr.DataArray(k.values**-2.0, dims=["wavenumber"])
    Cv = xr.DataArray(k.values**-2.0, dims=["wavenumber"])
    result = spec_helm_decomp(k, Cu, Cv)
    assert result.psi.dims == ("wavenumber",)


def test_gm_true_populates_wave_vortex_fields():
    k = np.logspace(-2, 1, 40)
    Cu = Cv = k**-2.0
    result = spec_helm_decomp(k, Cu, Cv, gm=True)

    for field in (
        result.u_wave,
        result.v_wave,
        result.u_vortex,
        result.v_vortex,
        result.ke_wave,
        result.buoyancy_wave,
    ):
        assert isinstance(field, xr.DataArray)
        assert np.all(np.isfinite(field.values))


def test_gm_loads_packaged_data_not_hardcoded_path():
    """Regression test for the original hardcoded-absolute-path bug."""
    from pyspec.helmholtz import _load_gm_spectrum

    f2omg2, ks = _load_gm_spectrum()
    assert f2omg2.size > 0
    assert ks.size == f2omg2.size
