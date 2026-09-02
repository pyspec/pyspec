"""Helpers for accepting either numpy arrays or xarray.DataArrays as input,
while the public API always emits xarray.DataArrays as output.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _infer_spacing(coord: np.ndarray, name: str) -> float:
    if coord.size < 2:
        raise ValueError(
            f"coordinate {name!r} has fewer than 2 points; can't infer spacing"
        )
    diffs = np.diff(coord.astype(float))
    spacing = diffs[0]
    if spacing == 0 or not np.allclose(diffs, spacing, rtol=1e-6):
        raise ValueError(
            f"coordinate {name!r} is not evenly spaced; pass the sample "
            "spacing explicitly instead of relying on inference"
        )
    return float(spacing)


def as_1d_array_and_dt(
    data, dt: float | None = None, dim: str | None = None
) -> tuple[np.ndarray, float, str]:
    """Return ``(values, dt, dim_name)`` for a 1-D input.

    ``data`` may be a plain 1-D array (``dt`` then required) or a 1-D
    ``xarray.DataArray`` (``dt`` inferred from its coordinate unless given).
    """
    if isinstance(data, xr.DataArray):
        if data.ndim != 1:
            raise ValueError(f"expected a 1-D DataArray, got dims {data.dims}")
        dim_name = dim or data.dims[0]
        arr = np.asarray(data.values, dtype=float)
        if dt is None:
            if dim_name not in data.coords:
                raise ValueError(
                    f"DataArray has no coordinate for dim {dim_name!r}; "
                    "pass dt explicitly"
                )
            dt = _infer_spacing(data.coords[dim_name].values, dim_name)
        return arr, dt, dim_name

    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D array, got shape {arr.shape}")
    if dt is None:
        raise ValueError("dt must be given explicitly for plain numpy input")
    return arr, dt, (dim or "freq")


def as_2d_array_and_spacing(
    data,
    d1: float | None = None,
    d2: float | None = None,
    dims: tuple[str, str] | None = None,
) -> tuple[np.ndarray, float, float, tuple[str, str]]:
    """Return ``(values, d1, d2, (dim2_name, dim1_name))`` for a 2-D input.

    Axis -1 is treated as the "1" axis (spacing ``d1``), axis -2 as the
    "2" axis (spacing ``d2``), matching pyspec's original convention.
    """
    if isinstance(data, xr.DataArray):
        if data.ndim != 2:
            raise ValueError(f"expected a 2-D DataArray, got dims {data.dims}")
        dim2_name, dim1_name = dims or data.dims
        arr = np.asarray(data.values, dtype=float)
        if d1 is None:
            if dim1_name not in data.coords:
                raise ValueError(
                    f"DataArray has no coordinate for dim {dim1_name!r}; "
                    "pass d1 explicitly"
                )
            d1 = _infer_spacing(data.coords[dim1_name].values, dim1_name)
        if d2 is None:
            if dim2_name not in data.coords:
                raise ValueError(
                    f"DataArray has no coordinate for dim {dim2_name!r}; "
                    "pass d2 explicitly"
                )
            d2 = _infer_spacing(data.coords[dim2_name].values, dim2_name)
        return arr, d1, d2, (dim2_name, dim1_name)

    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {arr.shape}")
    if d1 is None or d2 is None:
        raise ValueError("d1 and d2 must both be given explicitly for plain numpy input")
    dim2_name, dim1_name = dims or ("k2", "k1")
    return arr, d1, d2, (dim2_name, dim1_name)
