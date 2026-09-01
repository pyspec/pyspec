import pytest

pytest.importorskip("xarray", reason="pyspec's public API requires xarray")
