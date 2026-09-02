"""Confidence intervals and spectral-slope fitting."""

from __future__ import annotations

import numpy as np
import xarray as xr

from pyspec import _core


def confidence_interval(
    psd, dof, ci: float = 0.95
) -> tuple[xr.DataArray, xr.DataArray]:
    """Chi-squared confidence interval for a spectral estimate."""
    if isinstance(psd, xr.DataArray):
        lo, hi = _core.confidence_interval(psd.values, dof, ci=ci)
        lower = psd.copy(data=lo)
        lower.name = "lower"
        upper = psd.copy(data=hi)
        upper.name = "upper"
        return lower, upper

    arr = np.asarray(psd, dtype=float)
    lo, hi = _core.confidence_interval(arr, dof, ci=ci)
    dim = "x"
    coords = {dim: np.arange(arr.size)}
    lower = xr.DataArray(lo, dims=[dim], coords=coords, name="lower")
    upper = xr.DataArray(hi, dims=[dim], coords=coords, name="upper")
    return lower, upper


def spectral_slope(k, E, kmin: float, kmax: float, std_E) -> tuple[float, float]:
    """Least-squares slope of log10(E) vs log10(k) over ``[kmin, kmax]``."""
    k_arr = k.values if isinstance(k, xr.DataArray) else np.asarray(k)
    E_arr = E.values if isinstance(E, xr.DataArray) else np.asarray(E)
    std_arr = std_E.values if isinstance(std_E, xr.DataArray) else std_E
    return _core.spectral_slope(k_arr, E_arr, kmin, kmax, std_arr)
