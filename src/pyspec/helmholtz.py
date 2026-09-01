"""Helmholtz (rotational/divergent) decomposition of horizontal KE spectra.

Implements the Buhler, Callies & Ferrari (2014, JFM) decomposition of
across-track/along-track velocity spectra into rotational and divergent
components, with an optional Garrett-Munk-based wave/vortex split.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import numpy as np
import xarray as xr
from scipy.integrate import simpson


def _central_diff(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0, x1, x2 = x[:-2], x[1:-1], x[2:]
    y0, y1, y2 = y[:-2], y[1:-1], y[2:]
    f = (x2 - x1) / (x2 - x0)
    return (1 - f) * (y2 - y1) / (x2 - x1) + f * (y1 - y0) / (x1 - x0)


def _load_gm_spectrum() -> tuple[np.ndarray, np.ndarray]:
    """Load the packaged Garrett-Munk reference spectrum.

    The original implementation loaded this from a hardcoded absolute
    path (/Users/crocha/Projects/dp_spectra/GM/gm_omega_star.npz) that
    only ever existed on the original author's own machine. The data file
    now ships inside the package itself.
    """
    ref = resources.files("pyspec.data").joinpath("gm_omega_star.npz")
    with resources.as_file(ref) as path, np.load(path) as gm:
        return gm["rgm"], gm["k"] * 1.0e3


@dataclass
class HelmholtzDecomposition:
    """Result of a Helmholtz decomposition of a horizontal KE spectrum.

    ``psi`` (rotational) and ``phi`` (divergent) are always populated.
    The wave/vortex fields are only populated when the decomposition was
    run with ``gm=True``; otherwise they are ``None``.
    """

    psi: xr.DataArray
    phi: xr.DataArray
    u_wave: xr.DataArray | None = None
    v_wave: xr.DataArray | None = None
    u_vortex: xr.DataArray | None = None
    v_vortex: xr.DataArray | None = None
    ke_wave: xr.DataArray | None = None
    buoyancy_wave: xr.DataArray | None = None


def spec_helm_decomp(k, Cu, Cv, gm: bool = False) -> HelmholtzDecomposition:
    """Decompose across-/along-track KE spectra into rotational + divergent parts.

    Parameters
    ----------
    k : array_like or xarray.DataArray
        Wavenumber.
    Cu, Cv : array_like or xarray.DataArray
        Spectra of the across-track and along-track velocity components.
    gm : bool, default False
        If True, further split into wave and vortex components using a
        reference Garrett-Munk spectrum shipped with the package.
    """
    k_arr = k.values if isinstance(k, xr.DataArray) else np.asarray(k, dtype=float)
    Cu_arr = Cu.values if isinstance(Cu, xr.DataArray) else np.asarray(Cu, dtype=float)
    Cv_arr = Cv.values if isinstance(Cv, xr.DataArray) else np.asarray(Cv, dtype=float)
    dim = k.dims[0] if isinstance(k, xr.DataArray) else "k"

    s = np.log(k_arr)
    Fphi = np.zeros_like(Cu_arr)
    Fpsi = np.zeros_like(Cu_arr)

    for i in range(s.size - 1):
        sh = np.sinh(s[i] - s[i:])
        ch = np.cosh(s[i] - s[i:])
        Fp = Cu_arr[i:] * sh + Cv_arr[i:] * ch
        Fs = Cv_arr[i:] * sh + Cu_arr[i:] * ch
        Fpsi[i] = simpson(Fs, x=s[i:])
        Fphi[i] = simpson(Fp, x=s[i:])

    Fpsi = np.clip(Fpsi, 0.0, None)
    Fphi = np.clip(Fphi, 0.0, None)

    Cpsi = Fpsi - Fphi + Cu_arr
    Cphi = Fphi - Fpsi + Cv_arr

    coords = {dim: k_arr}
    psi_da = xr.DataArray(
        Cpsi, dims=[dim], coords=coords, name="psi",
        attrs={"long_name": "rotational KE spectrum"},
    )
    phi_da = xr.DataArray(
        Cphi, dims=[dim], coords=coords, name="phi",
        attrs={"long_name": "divergent KE spectrum"},
    )

    if not gm:
        return HelmholtzDecomposition(psi=psi_da, phi=phi_da)

    f2omg2, ks = _load_gm_spectrum()
    f2omg2i = np.interp(k_arr, ks, f2omg2)

    Cv_w = f2omg2i * Fphi - Fpsi + Cv_arr
    Cv_v = Cv_arr - Cv_w

    kdkromg = _central_diff(ks, f2omg2)
    kdkromg = np.interp(k_arr, ks[1:-1], kdkromg)

    dFphi = _central_diff(k_arr, Fphi)
    dFphi = np.interp(k_arr, k_arr[1:-1], dFphi.real)
    E_w = Fphi - k_arr * dFphi

    Cu_w = -k_arr * kdkromg * Fphi + f2omg2i * (-Fpsi + Cv_arr) + Fphi
    Cu_v = Cu_arr - Cu_w

    Cb_w = E_w - (Cu_w + Cv_w) / 2.0

    def _wrap(data, name, long_name):
        return xr.DataArray(
            data, dims=[dim], coords=coords, name=name,
            attrs={"long_name": long_name},
        )

    return HelmholtzDecomposition(
        psi=psi_da,
        phi=phi_da,
        u_wave=_wrap(Cu_w, "u_wave", "wave component of across-track KE"),
        v_wave=_wrap(Cv_w, "v_wave", "wave component of along-track KE"),
        u_vortex=_wrap(Cu_v, "u_vortex", "vortex component of across-track KE"),
        v_vortex=_wrap(Cv_v, "v_vortex", "vortex component of along-track KE"),
        ke_wave=_wrap(E_w, "ke_wave", "total wave KE"),
        buoyancy_wave=_wrap(Cb_w, "buoyancy_wave", "wave buoyancy variance"),
    )
