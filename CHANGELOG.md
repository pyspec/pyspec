# Changelog

All notable changes to `pyspec` are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/).

## [1.0.0]

A complete, from-scratch rewrite of `pyspec`, modernizing it from its
Python 2 / pre-xarray origins into a fully xarray-native Python 3.10+
package. Not backward compatible with pre-1.0 versions -- see
`MIGRATION.md` for the full API mapping and every behavioral change,
each backed by a specific test. This entry summarizes the highlights.

### Added

- **xarray-native API.** `Spectrum1D`, `Spectrum2D`, and
  `spec_helm_decomp` accept plain numpy arrays or `xarray.DataArray`s and
  always return `DataArray`s, with coordinate/units/long_name metadata
  attached. Sample spacing (`dt`/`d1`/`d2`) is inferred automatically
  from a `DataArray`'s coordinate when not given explicitly.
- **`pyspec.gm`**: parameterized Garrett-Munk reference spectrum
  computation (`GMParams`, `coriolis_frequency`, `compute_gm_reference`),
  replacing a single hardcoded, location-specific reference file. See
  "Fixed" below.
- **`pyspec._core`**: the computational core (periodograms, isotropic
  averaging, confidence intervals, binning, slope fitting) isolated as
  pure-numpy functions, with no xarray dependency -- fast, and directly
  unit-testable independent of the labeled-array plumbing.
- **117 tests**, including:
  - Parseval's-theorem checks to 1e-9 to 1e-10 relative precision, across
    dozens of size/spacing combinations, for both 1-D and 2-D spectra.
  - Two independent, closed-form analytic ground-truth checks (white
    noise PSD level, pure-sinusoid variance/peak-frequency recovery) that
    don't depend on any reference implementation.
  - A regression test locking in that `spec_helm_decomp`'s `psi + phi`
    reconstructs `Cu + Cv` exactly -- the velocity-variance-vs-kinetic-
    energy convention (see "Fixed" below).
  - A regression test confirming `compute_gm_reference` reproduces this
    package's original Drake Passage reference data to ~0.07% (a
    scipy-version floating-point difference, not an algorithmic one).
- **`MIGRATION.md`**: full API mapping and rationale for every behavioral
  change relative to the pre-1.0 implementation.

### Fixed

Real, if often subtle, bugs found while building the test suite:

- **2-D isotropic (azimuthal) averaging double-counted energy** at every
  grid point exactly on the diagonal `k1 == |k2|`. This isn't a rare
  edge case: `hypot(i*dk1, i*dk2) == i*hypot(dk1, dk2)` exactly, by
  algebra, for any spacing, so every such point landed exactly on a
  radial bin boundary, and the original's closed-interval (`>=`/`<=`)
  membership test counted it in two bins at once. Fixed with a proper
  half-open binning convention, and vectorized in the process (roughly
  an order of magnitude faster at a 256x256 grid).
- **2-D wavenumber grid was wrong for odd-sized inputs.** A hand-rolled
  `arange(0., n1/2 + 1)` produced one extra, spurious one-sided
  wavenumber bin whenever `n1` was odd. Replaced with
  `np.fft.rfftfreq`/`fftfreq`.
- **2-D variance included the DC term; 1-D variance didn't.** The two
  were inconsistent with each other. Reconciled: both now exclude the
  zero-wavenumber/frequency bin, meaning "variance" consistently means
  variability around the mean, not total mean square.
- **Both `Spectrum1D` and `Spectrum2D` mutated their input array in
  place** during windowing. Fixed; nothing here mutates what you pass in.
- **A factor-of-2 conflation between velocity variance spectra and
  kinetic energy spectra.** `spec_helm_decomp`'s `Cu`/`Cv` (and its
  `psi`/`phi` outputs) are per-component velocity variance spectra, not
  KE spectra -- KE requires an explicit `0.5 * (...)`. This distinction
  caused a real bug in this package's own example notebook (comparing
  `pyspec` against `scipy.signal.welch` with a spurious extra factor of
  2), now fixed and locked in with a regression test plus explicit
  documentation in the `helmholtz` module docstring.
- **The Garrett-Munk wave/vortex reference was hardcoded to one
  location.** The original loaded a GM76 reference spectrum precomputed
  once for Drake Passage from a hardcoded absolute path, and used it
  silently as the default for any dataset, anywhere. `spec_helm_decomp`'s
  `gm` parameter is now a `GMParams` (your own Coriolis parameter and
  stratification), and the reference is computed on the fly by
  `pyspec.gm.compute_gm_reference` -- transcribed from the original
  derivation script (recovered from the original author's own research
  repository) and regression-tested against its original output.
  - A follow-up fix: the first version of this parameterization requested
    the GM reference over a range matched exactly to the caller's own `k`
    array, which put a known NaN boundary point (`compute_gm_reference`'s
    output is undefined at the very edge of whatever range it's asked
    to compute) exactly on the caller's last data point, propagating
    into every wave/vortex output. Fixed by padding the requested range
    and stripping non-finite points before interpolating.
- **`scipy.integrate.simps`** (removed in modern scipy) **replaced with
  `simpson`**; **`numpy.matrix`** (deprecated) **replaced with
  `numpy.linalg.lstsq`** in `spectral_slope`.

### Changed

- Package layout: `src/` layout with `pyproject.toml`/hatchling,
  replacing the old flat layout and `setup.py`.
- `Spectrum1D`'s default `detrend` is now `"constant"` (the original
  never detrended before windowing, which leaves DC-adjacent frequency
  leakage). Pass `detrend=False` for the original's exact behavior.
- FFTs now go through `scipy.fft` rather than `numpy.fft` (drop-in,
  generally faster).

### Removed

- `THREEDimensional_spec`: already incomplete in the original (no
  isotropic-spectrum method) and unused/untested. Not carried over rather
  than shipped silently unfinished. The 2-D core's pattern generalizes
  directly if this is needed later.
- `spec_est`: an LLC-output-specific helper explicitly marked "temporary"
  in the original. Not general-purpose; not carried over.
- The packaged, hardcoded Drake Passage GM reference file -- see "Fixed"
  above. Kept only as a test fixture (`tests/data/`) for the regression
  test confirming `compute_gm_reference` reproduces it.
