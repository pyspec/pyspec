"""One-dimensional power spectral density estimation."""

from __future__ import annotations

import xarray as xr

from pyspec import _core
from pyspec._xr_utils import as_1d_array_and_dt


class Spectrum1D:
    """Power spectral density of a single 1-D real-valued signal.

    Parameters
    ----------
    data : array_like or xarray.DataArray
        The signal to analyze. Must be 1-D. If a ``DataArray``, its
        coordinate is used to infer the sampling interval ``dt`` (which
        must be evenly spaced) unless ``dt`` is given explicitly, and its
        dimension name is reused for the output frequency dimension.
    dt : float, optional
        Sampling interval. Required if ``data`` is a plain array.
    detrend : {'constant', 'linear', False}, default 'constant'
        Detrending applied before windowing.
    window : {'hann', None}, default 'hann'
        Taper applied before the FFT.
    dim : str, optional
        Name to use for the frequency dimension/coordinate in the output.
        Defaults to the input ``DataArray``'s dimension name, or
        ``"freq"`` for plain-array input.

    Attributes
    ----------
    psd : xarray.DataArray
        One-sided power spectral density.
    freq : xarray.DataArray
        Frequency (or wavenumber) coordinate.
    var : float
        Total variance recovered by integrating ``psd`` over frequency.
    n, dt, df : int, float, float

    Notes
    -----
    Unlike the original implementation, this never mutates the array you
    pass in.
    """

    def __init__(
        self,
        data,
        dt: float | None = None,
        detrend: str | bool = "constant",
        window: str | None = "hann",
        dim: str | None = None,
    ):
        arr, dt_, dim_name = as_1d_array_and_dt(data, dt=dt, dim=dim)
        freq, psd, df = _core.periodogram_1d(
            arr, dt_, detrend_kind=detrend, window=window
        )

        self.n = arr.size
        self.dt = dt_
        self.df = df
        self.var = float(df * psd[1:].sum())

        coords = {dim_name: freq}
        self.freq = xr.DataArray(
            freq,
            dims=[dim_name],
            coords=coords,
            name=dim_name,
            attrs={"long_name": "frequency"},
        )
        self.psd = xr.DataArray(
            psd,
            dims=[dim_name],
            coords=coords,
            name="psd",
            attrs={
                "long_name": "power spectral density",
                "n": self.n,
                "dt": self.dt,
                "df": self.df,
                "detrend": str(detrend),
                "window": str(window),
            },
        )

    def __repr__(self) -> str:
        return (
            f"Spectrum1D(n={self.n}, dt={self.dt!r}, "
            f"df={self.df:.4g}, var={self.var:.4g})"
        )
