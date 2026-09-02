# Migrating from the original pyspec

This is a from-scratch rewrite. There's no compatibility shim -- the API,
package layout, and (in a few documented cases) the numerical conventions
have changed. This document explains what changed and why.

## API changes

| Original | New |
|---|---|
| `pyspec.spectrum.Spectrum(phi, dt)` | `pyspec.Spectrum1D(phi, dt=dt)` |
| `pyspec.spectrum.TWODimensional_spec(phi, d1, d2)` | `pyspec.Spectrum2D(phi, d1=d1, d2=d2)` |
| `.spec` attribute | `.psd` attribute (always an `xarray.DataArray` now) |
| module-level `calc_ispec(k, l, E)` | `Spectrum2D.isotropic()` method, or `pyspec._core.isotropic_average` |
| `spec_error(E, sn, ci)` | `pyspec.confidence_interval(psd, dof, ci)` |
| `avg_per_decade(k, E, nbins)` | `pyspec.avg_per_decade(k, E, n_bins=...)` |
| `pyspec.helmholtz.spec_helm_decomp(...)` returning a plain tuple | `pyspec.spec_helm_decomp(...)` returning a `HelmholtzDecomposition` dataclass |
| `THREEDimensional_spec` | not carried over (see below) |

Inputs can be plain numpy arrays or `xarray.DataArray`s; outputs are
always `xarray.DataArray`s (or a `HelmholtzDecomposition` of them). If you
pass a `DataArray` with a coordinate, sample spacing (`dt`/`d1`/`d2`) is
inferred from it automatically.

`Spectrum1D`/`Spectrum2D` never mutate the array you pass in. The
original mutated its input in place while windowing it -- a real, if
subtle, bug.

## Not carried over

**`THREEDimensional_spec`** was already incomplete in the original (no
isotropic-spectrum method, dead commented-out code) and wasn't exercised
by any test or example. Dropped rather than silently shipping unfinished,
untested code. If you need it, the 2-D `_core.periodogram_2d` pattern
generalizes directly -- happy to build it as a follow-up.

**`spec_est`** (the LLC-output helper) was explicitly marked "temporary"
in the original and is llc-specific, not general-purpose. Not carried
over; ask if you still need it and I'll port it properly.

## Numerical/behavioral changes worth knowing about

These were found while writing the test suite, not invented for the sake
of change -- each one is backed by a test in `tests/test_core.py`.

1. **2-D isotropic averaging no longer double-counts diagonal points.**
   The original loop tested bin membership with `>=` and `<=` on both
   sides of every bin. Any grid point exactly on the diagonal
   `k1 == |k2|` lands *exactly* on a bin boundary -- provably, by algebra
   (`hypot(i*dk1, i*dk2) == i*hypot(dk1, dk2)` for any spacing), not by
   floating-point coincidence -- so those points were counted in two bins
   at once. This is fixed via a half-open binning convention. See
   `test_isotropic_average_no_double_counting_or_dropped_points`.

2. **2-D wavenumber grid fixed for odd-sized inputs.** The original's
   hand-rolled `np.arange(0., n1/2 + 1)` produced one extra, spurious
   one-sided wavenumber bin whenever `n1` was odd (e.g. 18 bins instead of
   the 17 that `rfft` actually computes). Now uses `np.fft.rfftfreq`/
   `fftfreq`, which handle this correctly.

3. **2-D variance now excludes the DC (zero-wavenumber) term, matching
   the 1-D convention.** The original's 1-D `var` explicitly skipped the
   zero-frequency bin (`spec[1:].sum()`) but the 2-D `var` didn't skip
   the equivalent zero-wavenumber bin. The two are now consistent via
   `pyspec._core.variance_2d`; "variance" means variability around the
   mean in both cases, not total mean square.

4. **`Spectrum1D` detrending default.** The original never detrended the
   1-D signal at all (only windowed it), which leaves DC-adjacent
   frequency leakage. The new default is `detrend="constant"` (remove the
   mean). Pass `detrend=False` to match the original's behavior exactly
   (aside from item 6 below).

5. **`helmholtz.py`'s Garrett-Munk reference is now computed from
   parameters you supply, not a hardcoded file.** This went through two
   stages. Initially, the original's hardcoded absolute path
   (`/Users/crocha/Projects/dp_spectra/GM/gm_omega_star.npz`, which only
   ever existed on the original author's machine) was fixed by shipping
   that same file inside the package. But that only fixed the *access*
   problem -- it left a deeper one: that file was a GM76 reference
   spectrum computed once for one specific location (Drake Passage, at
   58°S with locally-appropriate stratification), silently reused as the
   default for `gm=True` regardless of what location your own data came
   from. `spec_helm_decomp`'s `gm` parameter is now a
   `pyspec.gm.GMParams` (Coriolis parameter, buoyancy frequency, and
   optionally the thermocline depth scale and mode-bandwidth parameter),
   and the reference spectrum is computed on the fly by
   `pyspec.gm.compute_gm_reference` for your own location. There's no
   default location anymore -- you always supply your own `f`/`N0`.
   The underlying algorithm (GM76 mode sum -> internal-wave dispersion
   relation -> isotropic-to-1D projection) is unchanged from the
   original and is regression-tested (`tests/test_gm.py`) to reproduce
   the original Drake Passage file's values to within ~0.07% (a
   `scipy.integrate.simps`-vs-`simpson` version difference, not an
   algorithmic one) when given that file's original parameters
   (`GMParams(f=coriolis_frequency(-58), N0=3e-3, b=1000.)`).

6. **Neither `Spectrum1D` nor `Spectrum2D` mutate their input** (see
   above). If you were relying on the original's in-place windowing as a
   side effect, you'll need to window explicitly instead
   (`pyspec._core.hann_window`).

7. **`scipy.integrate.simps` (removed in modern scipy) replaced with
   `scipy.integrate.simpson`** in the Helmholtz decomposition -- same
   algorithm, current API.

8. **`spectral_slope` no longer uses `numpy.matrix`** (deprecated);
   reimplemented with `numpy.linalg.lstsq` and explicit covariance
   propagation. Same math, same result, modern API.

## Performance

`pyspec._core.isotropic_average` is a vectorized (`numpy.bincount`-based)
replacement for the original's per-bin Python loop, which recomputed a
boolean mask over the full 2-D grid once per radial bin
(`O(n_bins * n1 * n2)`). The vectorized version is `O(n1 * n2)` and is
roughly 5-10x faster at a 256x256 grid in this environment; the gap grows
with grid size. See `test_isotropic_average_is_much_faster`.

FFTs now go through `scipy.fft` rather than `numpy.fft` (drop-in,
generally faster, and supports multi-threading via a `workers` argument
if you need it for very large transforms).
