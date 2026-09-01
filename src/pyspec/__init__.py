"""pyspec: spectral estimates from in-situ oceanographic data and models."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("pyspec")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

from pyspec import _core  # noqa: F401  (pure numpy; always importable)
from pyspec.gm import (  # noqa: F401 (pure numpy; always importable)
    EARTH_ANGULAR_VELOCITY,
    GMParams,
    compute_gm_reference,
    coriolis_frequency,
)

try:
    from pyspec.binning import avg_per_decade
    from pyspec.errors import confidence_interval, spectral_slope
    from pyspec.helmholtz import HelmholtzDecomposition, spec_helm_decomp
    from pyspec.onedim import Spectrum1D
    from pyspec.twodim import IsotropicSpectrum, Spectrum2D
except ImportError as exc:  # pragma: no cover
    import warnings

    warnings.warn(
        "pyspec's high-level API requires xarray, which could not be "
        f"imported ({exc}). Only pyspec._core/pyspec.gm (plain-numpy "
        "functions) are available in this environment.",
        stacklevel=2,
    )
    Spectrum1D = Spectrum2D = IsotropicSpectrum = None  # type: ignore[assignment]
    spec_helm_decomp = HelmholtzDecomposition = None  # type: ignore[assignment]
    confidence_interval = spectral_slope = avg_per_decade = None  # type: ignore[assignment]

__all__ = [
    "__version__",
    "Spectrum1D",
    "Spectrum2D",
    "IsotropicSpectrum",
    "spec_helm_decomp",
    "HelmholtzDecomposition",
    "confidence_interval",
    "spectral_slope",
    "avg_per_decade",
    "GMParams",
    "coriolis_frequency",
    "compute_gm_reference",
    "EARTH_ANGULAR_VELOCITY",
]
