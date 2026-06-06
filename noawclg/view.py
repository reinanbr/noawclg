"""Lightweight read-only wrapper over a selected xr.Dataset."""

from __future__ import annotations

import xarray as xr


class _DatasetView:
    """Subscript + attribute access over an already-selected xr.Dataset.

    Examples
    --------
    >>> view = noaa.get_data_from_place("Fortaleza")
    >>> view["t2m"]           # variable by key
    >>> view.to_dataframe()   # pandas DataFrame
    >>> view.to_dict()        # plain dict
    """

    def __init__(self, dataset: xr.Dataset) -> None:
        self._ds = dataset

    def __getitem__(self, key: str) -> xr.Variable:
        if key not in self._ds.variables:
            available = list(self._ds.variables)
            raise KeyError(f"Variable '{key}' not found. Available: {available}")
        return self._ds.variables[key]

    def __repr__(self) -> str:
        return f"<_DatasetView>\n{self._ds}"

    def to_dataframe(self):
        """Convert selection to a pandas DataFrame."""
        return self._ds.to_dataframe()

    def to_dict(self) -> dict:
        """Convert selection to a plain Python dict."""
        return self._ds.to_dict()

    @property
    def dataset(self) -> xr.Dataset:
        """Underlying xr.Dataset for advanced use."""
        return self._ds
