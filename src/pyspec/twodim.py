"""Two-dimensional power spectral density estimation."""

from __future__ import annotations

import numpy as np
import xarray as xr

from pyspec import _core
from pyspec._xr_utils import as_2d_array_and_spacing


class IsotropicSpectrum:
    """Azimuthally-averaged (isotropic) 1-D spectrum derived from a 2-D one."""

    def __init__(self, kr: np.ndarray, Er: np.ndarray, dim_name: str = "kr"):
        self.psd = xr.DataArray(
            Er,
            dims=[dim_name],
            coords={dim_name: kr},
            name="isotropic_psd",
            attrs={"long_name": "azimuthally-averaged power spectral density"},
        )
        self.freq = self.psd.coords[dim_name]

    def __repr__(self) -> str:
        return f"IsotropicSpectrum(n={self.psd.size})"


class Spectrum2D:
    """Power spectral density of a single 2-D real-valued field.

    Parameters
    ----------
    data : array_like or xarray.DataArray, shape (n2, n1)
        The field to analyze, with axis -2 spaced by ``d2`` and axis -1
        spaced by ``d1`` (matches the original pyspec axis convention).
    d1, d2 : float, optional
        Sample spacing. Inferred from coordinates if ``data`` is a
        ``DataArray``, unless given explicitly.
    detrend : {'constant', 'linear', False}, default 'linear'
    window : {'hann', None}, default 'hann'
    dims : tuple[str, str], optional
        Names for the ``(k2, k1)`` output dimensions.

    Attributes
    ----------
    psd : xarray.DataArray, dims (k2, k1)
    kappa : xarray.DataArray, dims (k2, k1)
    var : float
    """

    def __init__(
        self,
        data,
        d1: float | None = None,
        d2: float | None = None,
        detrend: str | bool = "linear",
        window: str | None = "hann",
        dims: tuple[str, str] | None = None,
    ):
        arr, d1_, d2_, dim_names = as_2d_array_and_spacing(
            data, d1=d1, d2=d2, dims=dims
        )
        dim2, dim1 = dim_names

        k1, k2, psd, dk1, dk2 = _core.periodogram_2d(
            arr, d1_, d2_, detrend_kind=detrend, window=window
        )

        self.n2, self.n1 = arr.shape
        self.d1, self.d2 = d1_, d2_
        self.dk1, self.dk2 = dk1, dk2
        self._k1_name, self._k2_name = dim1, dim2

        kk1, kk2 = np.meshgrid(k1, k2)
        kk1 = np.fft.fftshift(kk1, axes=0)
        kk2 = np.fft.fftshift(kk2, axes=0)
        kappa = np.sqrt(kk1**2 + kk2**2)

        coords = {dim1: k1, dim2: k2}
        self.psd = xr.DataArray(
            psd,
            dims=[dim2, dim1],
            coords=coords,
            name="psd",
            attrs={
                "long_name": "2-D power spectral density",
                "d1": d1_,
                "d2": d2_,
                "detrend": str(detrend),
                "window": str(window),
            },
        )
        self.kappa = xr.DataArray(
            kappa,
            dims=[dim2, dim1],
            coords=coords,
            name="kappa",
            attrs={"long_name": "isotropic wavenumber magnitude"},
        )

        self.var = _core.variance_2d(k1, k2, psd, dk1, dk2, n1=self.n1)

    def isotropic(self, dim: str = "kr") -> IsotropicSpectrum:
        """Azimuthally-average the 2-D spectrum onto a 1-D radial wavenumber grid."""
        k1 = self.psd.coords[self._k1_name].values
        k2 = self.psd.coords[self._k2_name].values
        kr, Er = _core.isotropic_average(k1, k2, self.psd.values)
        return IsotropicSpectrum(kr, Er, dim_name=dim)

    def __repr__(self) -> str:
        return (
            f"Spectrum2D(shape=({self.n2}, {self.n1}), "
            f"d1={self.d1!r}, d2={self.d2!r}, var={self.var:.4g})"
        )
