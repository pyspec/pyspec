"""Garrett-Munk (GM76) reference spectrum for the wave/vortex decomposition.

This replaces what used to be a single hardcoded reference file
(``gm_omega_star.npz``), precomputed for one specific location (Drake
Passage) with one specific set of stratification/latitude parameters.
That meant ``spec_helm_decomp(..., gm=True)`` silently used Drake
Passage's internal-wave field as the reference for *any* dataset, which
is wrong anywhere else. This module computes the reference spectrum
on the fly from parameters you supply for your own location.

The algorithm (mode sum over the GM76 vertical-mode/frequency spectrum,
converted to an isotropic 2-D wavenumber spectrum via the internal-wave
dispersion relation, then projected onto a 1-D along-track spectrum) is
a direct port of the script that generated the original Drake Passage
reference file, recovered from the original author's own research
repository for this exact purpose. See ``tests/test_gm.py`` for a
regression test confirming this reproduces that original file's values
to within numerical-precision differences (~0.07%, consistent with a
`scipy.integrate.simps`-vs-`simpson` version difference across scipy
releases, not an algorithmic difference).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson

EARTH_ANGULAR_VELOCITY = 7.2921e-5  # rad/s, sidereal rotation rate (default)


def coriolis_frequency(latitude_deg: float, omega: float = EARTH_ANGULAR_VELOCITY) -> float:
    """Coriolis parameter f = 2*omega*sin(latitude), in rad/s.

    A dependency-free replacement for the (deprecated, Python-2-era)
    ``seawater.f()`` function used originally -- computed directly from
    Earth's rotation rate rather than pulled from an external package.

    Parameters
    ----------
    latitude_deg : float
        Latitude, in degrees (negative for the southern hemisphere).
    omega : float, default EARTH_ANGULAR_VELOCITY (7.2921e-5 rad/s)
        Earth's angular rotation rate, in rad/s. The default (the
        sidereal rotation rate) matches ``seawater.f()`` to within
        ~0.03% -- negligible for GM-spectrum purposes -- but is exposed
        here so you can override it if you need a different convention
        (e.g. the mean solar day rate, 2*pi/86400 rad/s) or a value
        matching some other reference you're comparing against.
    """
    return 2.0 * omega * np.sin(np.radians(latitude_deg))


@dataclass(frozen=True)
class GMParams:
    """Parameters for a GM76 reference spectrum at a given location.

    Parameters
    ----------
    f : float
        Coriolis parameter [rad/s]. Use :func:`coriolis_frequency` to
        compute this from a latitude if you don't have it already.
    N0 : float
        Buoyancy (Brunt-Vaisala) frequency [rad/s], representative of
        the water column you're analyzing -- e.g. a vertically-averaged
        or pycnocline value. This is the main "supply your own
        stratification" parameter.
    b : float, default 1300.0
        Vertical (thermocline) e-folding/depth scale [m]. The standard
        GM76 literature default is ~1300 m; the original Drake Passage
        analysis this was ported from used 1000 m for that specific,
        weakly-stratified location. There's no universally-correct
        default -- pick a value representative of your own water column,
        or leave the 1300 m default if you don't have a better estimate.
    js : float, default 3.0
        GM76 mode-number bandwidth parameter (standard value; rarely
        changed in practice).
    """

    f: float
    N0: float
    b: float = 1300.0
    js: float = 3.0


def compute_gm_reference(
    params: GMParams,
    kmin: float = 1e-3,
    kmax: float = 100.0,
    n_k: int = 500,
    n_modes: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the GM76 rotational/divergent reference ratio Fpsi/Fphi.

    This is the quantity :func:`pyspec.spec_helm_decomp` needs when
    ``gm`` is given: the ratio of theoretical GM-model rotational to
    divergent 1-D wavenumber spectra, used as the reference against
    which an observed field's wave component is identified.

    Parameters
    ----------
    params : GMParams
        Location-specific GM76 parameters.
    kmin, kmax : float
        Wavenumber range to compute over, in cycles/km. Defaults span the
        same physical range as the original Drake Passage script this was
        ported from (1e-3 to 100 cycles/km); narrow this to your actual
        data's wavenumber range for faster computation.
    n_k : int
        Number of (log-spaced) wavenumber points.
    n_modes : int
        Number of vertical modes to sum. 10000 (the original's choice)
        is generously converged; you can typically reduce this for speed
        with little effect on the result, since the GM76 mode-number
        weight H(j) decays like 1/j^2.

    Returns
    -------
    k : ndarray
        Wavenumber, cycles/km.
    rgm : ndarray
        Fpsi_GM(k) / Fphi_GM(k). NaN at k values where Fphi is not yet
        defined (the last point or two of the grid), matching the
        original's behavior.
    """
    f, N0, b, js = params.f, params.N0, params.b, params.js

    j = np.arange(1, n_modes + 1, dtype=float)
    jsum = ((j**2 + js**2) ** -1).sum()

    kh = 2 * np.pi * np.logspace(np.log10(kmin / 1e3), np.log10(kmax / 1e3), n_k)

    Cphi = np.zeros(kh.size)
    Cpsi = np.zeros(kh.size)
    for j_idx in j:
        jj = np.pi * j_idx / b
        omega = np.sqrt((f**2 * jj**2 + N0**2 * kh**2) / (jj**2 + kh**2))
        B = (2 / np.pi) * f / omega / np.sqrt(omega**2 - f**2)
        B = B / B.sum()
        H = (j_idx**2 + js**2) ** -1 / jsum
        domega_dk = (N0**2 - f**2) * kh * jj**2 / (omega * (kh**2 + jj**2) ** 2)
        Cphi += B * H * domega_dk / kh**2
        Cpsi += (f**2 / omega**2) * B * H * domega_dk / kh**2

    Fphi = np.zeros(kh.size)
    Fpsi = np.zeros(kh.size)
    for i in range(kh.size):
        # clip to zero before sqrt: kh[i:][0] is kh[i] itself, so this
        # difference is mathematically >= 0 everywhere, but vectorized
        # vs scalar squaring can round a few ULP differently and produce
        # a tiny (~1e-17) spurious negative at that first element
        l = np.sqrt(np.maximum(kh[i:] ** 2 - kh[i] ** 2, 0.0))
        Fphi[i] = simpson(Cphi[i:] * l, x=kh[i:])
        Fpsi[i] = simpson(Cpsi[i:] * l, x=kh[i:])

    k_out = kh / (2 * np.pi) * 1.0e3  # rad/m -> cycles/km
    with np.errstate(invalid="ignore", divide="ignore"):
        rgm = Fpsi / Fphi
    return k_out, rgm
