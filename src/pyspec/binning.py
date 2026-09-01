"""Log-space binning of spectra."""

from __future__ import annotations

import numpy as np
import xarray as xr

from pyspec import _core


def avg_per_decade(k, E, n_bins: int = 10, dim: str | None = None) -> xr.DataArray:
    """Average a spectrum in log-wavenumber bins, ``n_bins`` bins per decade."""
    k_arr = k.values if isinstance(k, xr.DataArray) else np.asarray(k)

    if isinstance(E, xr.DataArray):
        E_arr = E.values
        dim = dim or (E.dims[0] if E.ndim == 1 else "k")
    else:
        E_arr = np.asarray(E)
        dim = dim or "k"

    ki, Ei = _core.avg_per_decade(k_arr, E_arr, n_bins=n_bins)
    return xr.DataArray(
        Ei,
        dims=[dim],
        coords={dim: ki},
        name="psd_binned",
        attrs={
            "long_name": "log-decade-averaged spectral density",
            "bins_per_decade": n_bins,
        },
    )
