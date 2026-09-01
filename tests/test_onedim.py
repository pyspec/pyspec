"""Tests for pyspec.onedim.Spectrum1D. Requires xarray."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pyspec import Spectrum1D


def test_plain_array_input_requires_dt():
    with pytest.raises(ValueError, match="dt must be given"):
        Spectrum1D(np.random.randn(10))


def test_plain_array_output_shape_and_dim_name():
    rng = np.random.default_rng(0)
    phi = rng.standard_normal(100)
    s = Spectrum1D(phi, dt=1.0)

    assert isinstance(s.psd, xr.DataArray)
    assert isinstance(s.freq, xr.DataArray)
    assert s.psd.dims == ("freq",)
    assert s.n == 100
    assert s.dt == 1.0


def test_dataarray_input_infers_dt_and_dim_name():
    rng = np.random.default_rng(1)
    time = np.arange(0, 200, 2.0)
    phi = xr.DataArray(rng.standard_normal(time.size), dims=["time"], coords={"time": time})

    s = Spectrum1D(phi)

    assert s.dt == pytest.approx(2.0)
    assert s.psd.dims == ("time",)


def test_dataarray_input_rejects_uneven_coordinate():
    coord = np.array([0.0, 1.0, 1.5, 3.0])
    phi = xr.DataArray(np.random.randn(4), dims=["t"], coords={"t": coord})
    with pytest.raises(ValueError, match="not evenly spaced"):
        Spectrum1D(phi)


def test_explicit_dt_overrides_inference():
    time = np.arange(0, 20, 1.0)
    phi = xr.DataArray(np.random.randn(time.size), dims=["time"], coords={"time": time})
    s = Spectrum1D(phi, dt=5.0)
    assert s.dt == 5.0


def test_does_not_mutate_input_dataarray():
    rng = np.random.default_rng(2)
    time = np.arange(64.0)
    phi = xr.DataArray(rng.standard_normal(time.size), dims=["time"], coords={"time": time})
    original = phi.values.copy()
    Spectrum1D(phi)
    np.testing.assert_array_equal(phi.values, original)


def test_custom_dim_name():
    rng = np.random.default_rng(3)
    phi = rng.standard_normal(50)
    s = Spectrum1D(phi, dt=1.0, dim="wavenumber")
    assert s.psd.dims == ("wavenumber",)
    assert "wavenumber" in s.psd.coords


def test_var_matches_manual_parseval_check():
    rng = np.random.default_rng(4)
    phi = rng.standard_normal(500)
    s = Spectrum1D(phi, dt=1.0, detrend="constant", window="hann")
    manual = s.df * s.psd.values[1:].sum()
    assert s.var == pytest.approx(manual)
