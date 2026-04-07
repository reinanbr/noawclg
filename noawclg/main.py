"""
Reinan Br <slimchatuba@gmail.com>
31/12/2021 15:46
Lib: noawclg (light version from noaawc) v0.0.1b0
=================================================================
Why it work use the GPLv3 LICENSE?
-----------------------------------------------------------------
    this project use the license GPLv3  because i have a hate
    for the other projects that 're privates in the social network's
    that use the 'open data' from noaa in the your closed source,
    and i see it, i getted it init that make the ant way
    from this method, and making the condiction that who use it
    project on your personal project, open your project, or win
    a process.

=================================================================
what's for it project?
-----------------------------------------------------------------
    This project is for a best development in works with
    climate prediction and getting it data from the
    opendata in noaa site on type netcdf.

=================================================================
it's a base from the noaawc lib
-----------------------------------------------------------------
    This will are the base from a lib very more big from it's,
    your name will call as 'nooawc', and because from need
    mpl_toolkits.basemap, your work will are possible only
    the anaconda.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from typing import Optional, Union

import numpy as np
import xarray as xr
from geopy.geocoders import Nominatim

from noawclg.gfs_dataset import GFSDatasetManager, VARIABLES

__version__ = "0.0.3"
__author__ = "Reinan Br"

log = logging.getLogger(__name__)

_GEOLOCATOR = Nominatim(user_agent="noawclg")

# Type alias
Coordinate = tuple[float, float]  # (lat, lon)

# ---------------------------------------------------------------------------
# Auto-detection of coordinate dimension names
# ---------------------------------------------------------------------------

# Candidate names, ordered by preference (most common first)
_LAT_CANDIDATES = ("lat", "latitude", "y", "nav_lat", "rlat", "XLAT")
_LON_CANDIDATES = ("lon", "longitude", "x", "nav_lon", "rlon", "XLONG")
_TIME_CANDIDATES = ("time", "Time", "t", "forecast_hour", "step")


def _find_dim(coords: list[str], candidates: tuple[str, ...], label: str) -> str:
    """
    Return the first candidate name that exists in *coords*.

    Raises ``KeyError`` with a helpful message if none match.
    """
    for name in candidates:
        if name in coords:
            log.debug("Auto-detected %s dimension: '%s'", label, name)
            return name
    raise KeyError(
        f"Cannot find a {label} coordinate in the dataset. "
        f"Tried: {candidates}. "
        f"Available coordinates: {coords}. "
        f"Pass {label}_dim=<name> explicitly to override."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_date(date: str) -> str:
    """Convert 'DD/MM/YYYY' → 'YYYYMMDD'."""
    return datetime.strptime(date, "%d/%m/%Y").strftime("%Y%m%d")


def _normalize_lon(lon: float, lon_min: float) -> float:
    """
    Normalize *lon* to match the dataset's longitude convention.

    Detects automatically whether the dataset uses [0, 360] or [-180, 180].
    """
    if lon_min < 0:
        # Dataset uses [-180, 180]
        return (lon + 180) % 360 - 180
    # Dataset uses [0, 360]
    return lon % 360


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box of a dataset's spatial domain."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def contains(self, lat: float, lon: float) -> bool:
        return (
            self.lat_min <= lat <= self.lat_max
            and self.lon_min <= lon <= self.lon_max
        )

    def __str__(self) -> str:
        return (
            f"lat=[{self.lat_min:.2f}, {self.lat_max:.2f}], "
            f"lon=[{self.lon_min:.2f}, {self.lon_max:.2f}]"
        )


# ---------------------------------------------------------------------------
# Lightweight wrapper over xr.Dataset for key-based access
# ---------------------------------------------------------------------------


class _DatasetView:
    """
    Subscript + attribute access over an already-selected xr.Dataset.

    Examples
    --------
    >>> view = noaa.get_data_from_place("Fortaleza")
    >>> view["t2m"]           # variable by key
    >>> view.to_dataframe()   # pandas DataFrame
    >>> view.to_dict()        # plain dict
    """

    def __init__(self, dataset: xr.Dataset) -> None:
        self._ds = dataset

    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> xr.Variable:
        if key not in self._ds.variables:
            available = list(self._ds.variables)
            raise KeyError(
                f"Variable '{key}' not found. Available: {available}"
            )
        return self._ds.variables[key]

    def __repr__(self) -> str:
        return f"<_DatasetView>\n{self._ds}"

    # ------------------------------------------------------------------
    # Convenience converters

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


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class get_noaa_data:
    """
    Opens a GFS NOAA dataset via OPeNDAP and exposes spatial/temporal queries.

    Parameters
    ----------
    date  : date in 'YYYYMMDD' format (e.g. '20240101').
            Defaults to today if omitted.
    cycle : initialization cycle ('00', '06', '12', '18').
    key   : variable key from VARIABLES (e.g. 't2m', 'u10', 'v10').
    hours : list of forecast hours to load. Defaults to every 3 h up to 384 h.
    """

    __version__ = __version__
    __author__ = __author__

    def __init__(
        self,
        date: Optional[str] = None,
        cycle: str = "00",
        keys: list[str] = ["t2m"],
        hours: Optional[list[int]] = None,
        *,
        lat_dim: Optional[str] = None,
        lon_dim: Optional[str] = None,
        time_dim: Optional[str] = None,
    ) -> None:
        if date is not None:
            try:
                date = _parse_date(date)
            except ValueError:
                raise ValueError(
                    f"Invalid date format: '{date}'. "
                    "Expected 'DD/MM/YYYY'."
                ) from None

        resolved_date = date or datetime.now().strftime("%Y%m%d")
        self.date = resolved_date
        self.cycle = cycle
        self.keys = keys 
        self.hours = hours if hours is not None else list(range(0, 385, 3))

        if not all(key in VARIABLES for key in self.keys):
            raise ValueError(
                f"Invalid variable keys: {self.keys    }. "
                f"Valid keys: {sorted(VARIABLES)}"
            )

        self.dataset: xr.Dataset = GFSDatasetManager(
            self.date, self.cycle
        )
        if len(self.keys) > 1:
            self._ds:xr.Dataset = self.dataset.build_multi_dataset(self.keys, self.hours)
        else:
            self._ds:xr.Dataset = self.dataset.build_dataset(self.keys[0], self.hours)
        # Auto-detect dimension names; user overrides take priority.
        coords = list(self._ds.coords)
        self._LAT_DIM  = lat_dim  or _find_dim(coords, _LAT_CANDIDATES,  "lat")
        self._LON_DIM  = lon_dim  or _find_dim(coords, _LON_CANDIDATES,  "lon")
        self._TIME_DIM = time_dim or _find_dim(coords, _TIME_CANDIDATES, "time")

        log.info(
            "Dimensions resolved → lat='%s', lon='%s', time='%s'",
            self._LAT_DIM, self._LON_DIM, self._TIME_DIM,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @cached_property
    def _lon_min(self) -> float:
        """Minimum longitude in the dataset (cached)."""
        return float(self._ds[self._LON_DIM].min())

    @cached_property
    def bounds(self) -> BoundingBox:
        """Spatial domain of the loaded dataset (cached)."""
        return BoundingBox(
            lat_min=float(self._ds[self._LAT_DIM].min()),
            lat_max=float(self._ds[self._LAT_DIM].max()),
            lon_min=float(self._ds[self._LON_DIM].min()),
            lon_max=float(self._ds[self._LON_DIM].max()),
        )

    def _normalize_point(self, lat: float, lon: float) -> tuple[float, float]:
        """Normalize lat/lon to the dataset's coordinate convention."""
        lon = _normalize_lon(lon, lon_min=self._lon_min)
        return lat, lon

    def _warn_if_out_of_bounds(self, lat: float, lon: float) -> None:
        if not self.bounds.contains(lat, lon):
            log.warning(
                "Point (lat=%.4f, lon=%.4f) is outside dataset bounds [%s]. "
                "xarray will return the nearest edge point.",
                lat,
                lon,
                self.bounds,
            )

    # ------------------------------------------------------------------
    # Direct key access  (keeps original behaviour)
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> xr.Variable:
        if key not in self._ds.variables:
            available = list(self._ds.variables)
            raise KeyError(
                f"Variable '{key}' not found. Available: {available}"
            )
        return self._ds.variables[key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_variable_names(self) -> dict[str, str]:
        """Return ``{variable: long_name}`` for every variable in the dataset."""
        return {
            k: self._ds.variables[k].attrs.get("long_name", "")
            for k in self._ds.variables
        }

    def get_data_from_point(
        self,
        point: Coordinate,
        *,
        time: Optional[Union[str, slice, list]] = None,
        tolerance: Optional[float] = None,
    ) -> _DatasetView:
        """
        Select the nearest grid point to *(lat, lon)*.

        Parameters
        ----------
        point     : ``(lat, lon)`` in degrees.
                    Longitude is normalised to the dataset's convention
                    automatically — pass values in either [-180, 180] or
                    [0, 360].
        time      : Optional time selector.  Accepts anything
                    ``xr.Dataset.sel`` understands: a single timestamp
                    (``"2024-01-01T00:00"``) , a ``slice``, or a list.
        tolerance : Maximum allowed distance in degrees.  Raises
                    ``KeyError`` if the nearest point is farther away.

        Returns
        -------
        _DatasetView
        """
        lat, lon = point
        lat, lon = self._normalize_point(lat, lon)
        self._warn_if_out_of_bounds(lat, lon)

        sel_kwargs: dict = {
            self._LAT_DIM: lat,
            self._LON_DIM: lon,
        }
        if time is not None:
            sel_kwargs[self._TIME_DIM] = time

        method_kwargs: dict = {"method": "nearest"}
        if tolerance is not None:
            method_kwargs["tolerance"] = tolerance

        log.debug(
            "Selecting nearest point → lat=%.4f, lon=%.4f%s",
            lat,
            lon,
            f", time={time}" if time is not None else "",
        )

        result = self._ds.sel(**sel_kwargs, **method_kwargs)
        return _DatasetView(result)

    def get_data_from_place(
        self,
        place: str,
        *,
        time: Optional[Union[str, slice, list]] = None,
        tolerance: Optional[float] = None,
    ) -> _DatasetView:
        """
        Geocode *place* and return data from the nearest grid point.

        Parameters
        ----------
        place     : Human-readable place name (e.g. ``"Fortaleza, Brazil"``).
        time      : Forwarded to :meth:`get_data_from_point`.
        tolerance : Forwarded to :meth:`get_data_from_point`.
        """
        location = _GEOLOCATOR.geocode(place)
        if location is None:
            raise ValueError(
                f"Could not geocode '{place}'. "
                "Try a more specific name or verify the spelling."
            )

        log.info(
            "Geocoded '%s' → lat=%.4f, lon=%.4f",
            place,
            location.latitude,
            location.longitude,
        )

        return self.get_data_from_point(
            (location.latitude, location.longitude),
            time=time,
            tolerance=tolerance,
        )

    def get_time_series(
        self,
        point: Coordinate,
        variable: Optional[str] = None,
    ) -> Union[xr.Dataset, xr.DataArray]:
        """
        Return the complete time series at the nearest grid point.

        Parameters
        ----------
        point    : ``(lat, lon)`` in degrees.
        variable : If given, return only that variable as a ``DataArray``.
                   If omitted, return the full ``Dataset``.
        """
        view = self.get_data_from_point(point)
        if variable is not None:
            if variable not in view.dataset:
                raise KeyError(
                    f"Variable '{variable}' not found. "
                    f"Available: {list(view.dataset.data_vars)}"
                )
            return view.dataset[variable]
        return view.dataset

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"get_noaa_data("
            f"date={self.date!r}, cycle={self.cycle!r}, "
            f"key={self.keys!r}, bounds=[{self.bounds}])"
        )