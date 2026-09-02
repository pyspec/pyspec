"""Pure-numpy computational core for pyspec.

Every function here takes and returns plain :class:`numpy.ndarray` objects
(no xarray). Keeping the math isolated from the labeled-array plumbing makes
it fast, trivially unit-testable, and reusable from anywhere.

Nothing in this module mutates its input arrays.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import fft as sp_fft
from scipy import signal
from scipy.special import gammainc

Detrend = Literal["constant", "linear", False]
Window = Literal["hann", None]

__all__ = [
    "hann_window",
    "detrend",
    "periodogram_1d",
    "periodogram_2d",
    "variance_2d",
    "isotropic_average",
    "avg_per_decade",
    "spectral_slope",
    "confidence_interval",
]


# --------------------------------------------------------------------------
# windowing / detrending
# --------------------------------------------------------------------------

def hann_window(n: int) -> np.ndarray:
    """Hann window normalized so that it preserves total variance.

    Scaled such that ``sum(window**2) == n``, matching the convention used
    throughout this package so that Parseval's theorem holds for the
    windowed periodogram.
    """
    win = np.hanning(n)
    scale = np.sqrt(n / (win**2).sum())
    return scale * win


def detrend(x: np.ndarray, kind: Detrend = "constant", axis: int = -1) -> np.ndarray:
    """Detrend ``x`` along ``axis``. ``kind=False`` is a no-op (returns a copy)."""
    if kind is False or kind is None:
        return np.array(x, copy=True)
    return signal.detrend(x, axis=axis, type=kind)


def _apply_window_nd(x: np.ndarray, window: Window, axes: tuple[int, ...]) -> np.ndarray:
    """Apply a separable Hann window along the given axes of an n-d array."""
    if window is None:
        return x
    if window != "hann":
        raise ValueError(f"unsupported window {window!r}; only 'hann' or None")
    out = x
    for ax in axes:
        n = x.shape[ax]
        w = hann_window(n)
        shape = [1] * x.ndim
        shape[ax] = n
        out = out * w.reshape(shape)
    return out


# --------------------------------------------------------------------------
# 1-D periodogram
# --------------------------------------------------------------------------

def periodogram_1d(
    x: np.ndarray,
    dt: float,
    detrend_kind: Detrend = "constant",
    window: Window = "hann",
) -> tuple[np.ndarray, np.ndarray, float]:
    """One-sided power spectral density of a real 1-D signal.

    Parameters
    ----------
    x : (n,) ndarray, real
    dt : sampling interval
    detrend_kind : {'constant', 'linear', False}
    window : {'hann', None}

    Returns
    -------
    freq : (n//2 + 1,) or ((n-1)//2 + 1,) ndarray
    psd : ndarray, same shape as freq
    df : float, frequency resolution
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("need at least 2 samples to estimate a spectrum")

    xw = detrend(x, detrend_kind)
    win = hann_window(n) if window == "hann" else np.ones(n)
    xw = xw * win

    neven = n % 2 == 0
    df = 1.0 / (n * dt)
    n_freq = n // 2 + 1 if neven else (n - 1) // 2 + 1
    freq = df * np.arange(n_freq)

    xh = sp_fft.rfft(xw)
    xh = xh[:n_freq]
    psd = 2.0 * (xh * xh.conj()).real / df / n**2
    psd[0] /= 2.0
    if neven:
        psd[-1] /= 2.0

    return freq, psd, df


# --------------------------------------------------------------------------
# 2-D periodogram
# --------------------------------------------------------------------------

def periodogram_2d(
    phi: np.ndarray,
    d1: float,
    d2: float,
    detrend_kind: Detrend = "linear",
    window: Window = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Two-dimensional power spectral density of a real field.

    Parameters
    ----------
    phi : (n2, n1) ndarray, real. Axis -2 has spacing ``d2``, axis -1 has
        spacing ``d1`` (matches the original pyspec convention).
    d1, d2 : sample spacing along each axis
    detrend_kind : applied independently along both axes when truthy
    window : separable 2-D Hann window

    Returns
    -------
    k1, k2 : wavenumber arrays for axis -1 (one-sided) and axis -2 (full,
        fftshift-ordered)
    psd : (n2, n1//2+1) ndarray, fftshift-ordered along axis 0
    dk1, dk2 : wavenumber resolution along each axis
    """
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 2:
        raise ValueError(f"periodogram_2d expects a 2-D array, got ndim={phi.ndim}")
    n2, n1 = phi.shape
    l1, l2 = d1 * n1, d2 * n2

    x = phi
    if detrend_kind:
        x = detrend(x, detrend_kind, axis=-1)
        x = detrend(x, detrend_kind, axis=-2)
    else:
        x = np.array(x, copy=True)
    x = _apply_window_nd(x, window, axes=(-1, -2))

    dk1, dk2 = 1.0 / l1, 1.0 / l2
    # np.fft.rfftfreq/fftfreq handle the even/odd-n edge cases correctly
    # (a hand-rolled `arange(0., n/2 + 1)` formula silently produces an
    # extra, spurious wavenumber whenever n is odd)
    k1 = np.fft.rfftfreq(n1, d=d1)
    k2 = np.fft.fftshift(np.fft.fftfreq(n2, d=d2))

    xh = sp_fft.rfft2(x)
    psd = 2.0 * (xh * xh.conj()).real / (dk1 * dk2) / (n1 * n2) ** 2
    psd = np.fft.fftshift(psd, axes=0)

    return k1, k2, psd, dk1, dk2


def variance_2d(
    k1: np.ndarray,
    k2: np.ndarray,
    psd: np.ndarray,
    dk1: float,
    dk2: float,
    n1: int,
) -> float:
    """Total variance recovered from a 2-D power spectral density.

    Folds the one-sided ``k1`` axis back to a full-plane sum (halving the
    ``k1=0`` column, and the ``k1`` Nyquist column when ``n1`` is even, so
    neither is double-counted), and excludes the ``(k1=0, k2=0)`` DC bin --
    matching the convention used by :func:`periodogram_1d`, whose variance
    is ``df * psd[1:].sum()`` (the zero-frequency term is dropped, so
    "variance" means variability around the mean rather than total mean
    square). ``psd`` must be arranged as returned by :func:`periodogram_2d`,
    and ``n1`` is the original signal length along the ``k1`` axis.
    """
    var_dens = psd.copy()
    var_dens[:, 0] /= 2.0
    if n1 % 2 == 0:
        var_dens[:, -1] /= 2.0

    dc_row = int(np.argmin(np.abs(k2)))
    var_dens[dc_row, 0] = 0.0  # drop the true DC bin (k1=0 and k2=0)

    return float(var_dens.sum() * dk1 * dk2)


# --------------------------------------------------------------------------
# isotropic (azimuthal) averaging
# --------------------------------------------------------------------------

def isotropic_average(
    k1: np.ndarray, k2: np.ndarray, psd: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthally-average a 2-D spectrum onto a 1-D radial wavenumber grid.

    Vectorized replacement for the original bin-by-bin Python loop.

    ``psd`` must be arranged as returned by :func:`periodogram_2d`: shape
    ``(n2, n1)`` with ``k1`` one-sided (axis -1) and ``k2`` full and
    fftshift-ordered (axis -2).

    Returns
    -------
    kr : radial wavenumber bin centers
    Er : azimuthally-averaged spectral density at each ``kr``
    """
    dk = np.abs(k1[2] - k1[1])
    dl = np.abs(k2[2] - k2[1])
    dkr = np.sqrt(dk**2 + dl**2)

    kk1, kk2 = np.meshgrid(k1, k2)
    wv = np.sqrt(kk1**2 + kk2**2)

    kmax = min(k1.max(), k2.max())
    kr = np.arange(dkr / 2.0, kmax + dkr / 2.0, dkr)
    n_bins = kr.size

    # Bin assignment: bin i covers (i*dkr, (i+1)*dkr], half-open from below
    # and closed above. This matters because grid points land *exactly* on
    # bin edges surprisingly often (every point on the diagonal
    # k1 == |k2| satisfies hypot(i*dk1, i*dk2) == i*dkr exactly, by
    # algebra). ceil(x) - 1 matches floor(x) everywhere except at exact
    # integer multiples of dkr, where it assigns to the bin below the
    # shared edge -- giving every point exactly one bin.
    bin_idx = np.maximum(np.ceil(wv / dkr).astype(np.int64) - 1, 0)
    valid = bin_idx < n_bins
    bin_idx_valid = bin_idx[valid]

    counts = np.bincount(bin_idx_valid, minlength=n_bins)
    dtheta = np.where(counts > 1, np.pi / np.maximum(counts - 1, 1), 0.0)
    weight_valid = wv[valid] * dtheta[bin_idx_valid]

    Er = np.bincount(
        bin_idx_valid, weights=(psd[valid] * weight_valid), minlength=n_bins
    )
    Er = Er[: kr.size]

    return kr, Er


# --------------------------------------------------------------------------
# confidence intervals
# --------------------------------------------------------------------------

def _yn_bounds(sn: np.ndarray, y_grid: np.ndarray, ci: float) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized lower/upper yN bounds for one or many degrees-of-freedom."""
    sn = np.atleast_1d(np.asarray(sn, dtype=float))
    cdf = gammainc(sn[:, None], sn[:, None] * y_grid[None, :])
    lo_idx = np.abs(cdf - ci).argmin(axis=1)
    hi_idx = np.abs(cdf - (1.0 - ci)).argmin(axis=1)
    return y_grid[lo_idx], y_grid[hi_idx]


def confidence_interval(
    psd: np.ndarray, dof: np.ndarray | float, ci: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Chi-squared confidence interval for a spectral estimate.

    Parameters
    ----------
    psd : spectral estimate
    dof : degrees of freedom (number of independent realizations averaged
        into each spectral estimate). Scalar or array broadcastable to
        ``psd``.
    ci : confidence level, e.g. 0.95 for a 95% interval

    Returns
    -------
    lower, upper : bounds on the true spectral density, same shape as ``psd``
    """
    psd = np.asarray(psd, dtype=float)
    dbin = 0.005
    y_grid = np.arange(0.0, 2.0 + dbin, dbin)

    dof_arr = np.broadcast_to(np.atleast_1d(np.asarray(dof, dtype=float)), psd.shape)
    y_lo, y_hi = _yn_bounds(dof_arr.ravel(), y_grid, ci)
    y_lo = y_lo.reshape(psd.shape)
    y_hi = y_hi.reshape(psd.shape)

    return psd / y_lo, psd / y_hi


# --------------------------------------------------------------------------
# misc utilities
# --------------------------------------------------------------------------

def avg_per_decade(k: np.ndarray, E: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Log-bin a spectrum with ``n_bins`` bins per decade of wavenumber."""
    k = np.asarray(k, dtype=float)
    E = np.asarray(E, dtype=float)
    dk = 1.0 / n_bins
    logk = np.log10(k)

    edges = np.arange(np.floor(logk.min()) - dk / 2, np.ceil(logk.max()) + dk, dk)
    centers = edges[:-1] + dk / 2

    idx = np.digitize(logk, edges) - 1
    idx = np.clip(idx, 0, centers.size - 1)

    counts = np.bincount(idx, minlength=centers.size)
    sums = np.bincount(idx, weights=E, minlength=centers.size)

    nonempty = counts > 0
    Ei = sums[nonempty] / counts[nonempty]
    ki = 10 ** centers[nonempty]

    return ki, Ei


def spectral_slope(
    k: np.ndarray, E: np.ndarray, kmin: float, kmax: float, std_E: np.ndarray | float
) -> tuple[float, float]:
    """Least-squares slope (and its uncertainty) of log10(E) vs log10(k).

    Modernized replacement for the original ``numpy.matrix``-based
    implementation (deprecated); uses :func:`numpy.linalg.lstsq` and
    standard error propagation instead.
    """
    k = np.asarray(k, dtype=float)
    E = np.asarray(E, dtype=float)
    mask = (k >= kmin) & (k <= kmax)

    x = np.log10(k[mask])
    y = np.log10(np.real(E[mask]))
    G = np.column_stack([np.ones_like(x), x])

    coeffs, *_ = np.linalg.lstsq(G, y, rcond=None)

    std_log = np.abs(np.log10(std_E))
    d = np.full(x.size, std_log) if np.isscalar(std_E) else np.abs(np.log10(np.asarray(std_E)[mask]))
    GtG_inv = np.linalg.inv(G.T @ G)
    Gg = GtG_inv @ G.T
    cov = Gg @ np.diag(d**2) @ Gg.T

    slope = coeffs[1]
    slope_err = np.sqrt(cov[1, 1])
    return float(slope), float(slope_err)
