"""Helmholtz (rotational/divergent) decomposition of horizontal velocity spectra.

Implements the Buhler, Callies & Ferrari (2014, JFM) decomposition of
across-track/along-track velocity spectra into rotational and divergent
components, with an optional Garrett-Munk-based wave/vortex split.

A note on units, since this is an easy place to get a factor of 2 wrong
(see the worked example in the package's example notebooks): every
spectrum in this module -- inputs and outputs alike -- follows the same
convention as ``Cu``/``Cv``: a per-component *velocity variance* spectrum
(units of velocity^2 per unit wavenumber), not a kinetic energy spectrum.
Specific kinetic energy is KE = (u^2 + v^2)/2, so if you want an actual
KE spectrum from any output here, you multiply by 0.5 yourself -- e.g.
``0.5 * result.psi`` for the rotational KE spectrum, or
``0.5 * (result.psi + result.phi)`` for the total. This is verified
directly by the algorithm's construction: ``result.psi + result.phi``
reconstructs ``Cu + Cv`` exactly (to machine precision), not half of it.

A note on the wave/vortex (``gm``) split: the original implementation
used a single Garrett-Munk reference spectrum hardcoded for one specific
location (Drake Passage) as a silent default -- meaning it was, without
any warning, using Drake Passage's internal-wave field as the reference
for whatever data you gave it, anywhere in the world. That's fixed here:
``gm`` now takes a :class:`pyspec.gm.GMParams` with your own location's
Coriolis parameter and stratification, and the reference spectrum is
computed on the fly via :func:`pyspec.gm.compute_gm_reference`. See that
module for details.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.integrate import simpson

from pyspec.gm import GMParams, compute_gm_reference


def _central_diff(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0, x1, x2 = x[:-2], x[1:-1], x[2:]
    y0, y1, y2 = y[:-2], y[1:-1], y[2:]
    f = (x2 - x1) / (x2 - x0)
    return (1 - f) * (y2 - y1) / (x2 - x1) + f * (y1 - y0) / (x1 - x0)


@dataclass
class HelmholtzDecomposition:
    """Result of a Helmholtz decomposition of horizontal velocity spectra.

    ``psi`` (rotational) and ``phi`` (divergent) are always populated.
    The wave/vortex fields are only populated when the decomposition was
    run with a ``gm`` reference; otherwise they are ``None``.

    All fields are velocity variance spectra, in the same convention as
    the ``Cu``/``Cv`` inputs to :func:`spec_helm_decomp` -- not kinetic
    energy spectra. See the module docstring for what that means and how
    to get an actual KE spectrum from these.
    """

    psi: xr.DataArray
    phi: xr.DataArray
    u_wave: xr.DataArray | None = None
    v_wave: xr.DataArray | None = None
    u_vortex: xr.DataArray | None = None
    v_vortex: xr.DataArray | None = None
    ke_wave: xr.DataArray | None = None
    buoyancy_wave: xr.DataArray | None = None


def spec_helm_decomp(k, Cu, Cv, gm: GMParams | None = None) -> HelmholtzDecomposition:
    """Decompose across-/along-track velocity spectra into rotational + divergent parts.

    Parameters
    ----------
    k : array_like or xarray.DataArray
        Wavenumber, in cycles/km (this matters if ``gm`` is given: the GM
        reference spectrum is computed directly over your ``k``'s min/max
        range, in the same units).
    Cu, Cv : array_like or xarray.DataArray
        Velocity variance spectra of the across-track and along-track
        components -- i.e. the direct spectra of ``u`` and ``v``
        themselves, *not* already multiplied by 1/2 to make them kinetic
        energy spectra. See the module docstring for why this matters:
        conflating the two conventions is an easy way to end up with
        results that are off by a factor of 2.
    gm : pyspec.gm.GMParams, optional
        If given, further split into wave and vortex components using a
        Garrett-Munk reference spectrum computed for the location
        described by ``gm`` (its Coriolis parameter and stratification --
        see :class:`pyspec.gm.GMParams`). If omitted (the default), only
        ``psi``/``phi`` are computed and the wave/vortex fields are
        ``None``. These additional fields (``u_wave``, ``v_wave``,
        ``ke_wave``, ``buoyancy_wave``, etc.) are a direct port of the
        original implementation's formulas and follow the same Cu/Cv
        variance convention as everything else here; if you need
        publication-grade certainty about their exact normalization,
        cross-check against Buhler, Callies & Ferrari (2014) directly --
        that part of the algorithm hasn't been independently re-derived
        from scratch here, only ported and regression-tested against the
        original code's numerical output.
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
        attrs={"long_name": "rotational velocity variance spectrum"},
    )
    phi_da = xr.DataArray(
        Cphi, dims=[dim], coords=coords, name="phi",
        attrs={"long_name": "divergent velocity variance spectrum"},
    )

    if gm is None:
        return HelmholtzDecomposition(psi=psi_da, phi=phi_da)

    # Pad the requested range beyond k_arr's own bounds, and drop any
    # non-finite points before interpolating: compute_gm_reference's
    # rgm is NaN at the last point or two of its own grid (the Fpsi/Fphi
    # ratio has no integration width left there). If kmax were matched
    # exactly to k_arr.max(), that NaN would land exactly on your last
    # data point and propagate into every wave/vortex output -- this
    # padding keeps it safely outside the range you actually care about.
    ks, f2omg2 = compute_gm_reference(
        gm,
        kmin=min(1e-3, float(k_arr.min()) * 0.5),
        kmax=max(100.0, float(k_arr.max()) * 2.0),
    )
    valid = np.isfinite(f2omg2)
    ks, f2omg2 = ks[valid], f2omg2[valid]
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
        u_wave=_wrap(Cu_w, "u_wave", "wave component of across-track velocity variance"),
        v_wave=_wrap(Cv_w, "v_wave", "wave component of along-track velocity variance"),
        u_vortex=_wrap(Cu_v, "u_vortex", "vortex component of across-track velocity variance"),
        v_vortex=_wrap(Cv_v, "v_vortex", "vortex component of along-track velocity variance"),
        ke_wave=_wrap(E_w, "ke_wave", "wave-associated energy-like quantity E_w (Buhler et al. 2014); variance convention, see module docstring"),
        buoyancy_wave=_wrap(Cb_w, "buoyancy_wave", "wave buoyancy variance"),
    )
