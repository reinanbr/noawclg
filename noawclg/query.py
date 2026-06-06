"""High-level query API: geocoding, point selection, and time-series access."""

from __future__ import annotations

import logging
from datetime import datetime
from functools import cached_property
from typing import Optional, Union

import xarray as xr

try:
    from geopy.geocoders import Nominatim
except ModuleNotFoundError:  # pragma: no cover - depends on environment
    Nominatim = None  # type: ignore[assignment]

from noawclg.catalog import VARIABLES
from noawclg.coords import (
    BoundingBox,
    _LAT_CANDIDATES,
    _LON_CANDIDATES,
    _TIME_CANDIDATES,
    _find_dim,
    _normalize_lon,
    _parse_date,
)
from noawclg.gfs_dataset import GFSDatasetManager
from noawclg.view import _DatasetView

__version__ = "2.2.7"
__author__ = "Reinan Br"

log = logging.getLogger(__name__)

Coordinate = tuple[float, float]  # (lat, lon)


class _MissingGeolocator:
    def geocode(self, place: str):
        raise ModuleNotFoundError(
            "geopy is required for geocoding places. Install it with: pip install geopy"
        )


_GEOLOCATOR = (
    Nominatim(user_agent="noawclg") if Nominatim is not None else _MissingGeolocator()
)


class get_noaa_data:
    """Opens a GFS NOAA dataset via NOMADS and exposes spatial/temporal queries.

    Parameters
    ----------
    date  : date in 'DD/MM/YYYY' format (e.g. '01/01/2024').
            Defaults to today if omitted.
    cycle : initialization cycle ('00', '06', '12', '18').
    keys  : variable keys from VARIABLES (e.g. ['t2m', 'u10']).
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
        region: Optional[dict] = None,
        time_dim: Optional[str] = None,
        output_dir: str = "./gfs_output",
        request_timeout: int = 30,
        pause: float = 1.5,
    ) -> None:
        if date is not None:
            try:
                date = _parse_date(date)
            except ValueError:
                raise ValueError(
                    f"Invalid date format: '{date}'. Expected 'DD/MM/YYYY'."
                ) from None

        resolved_date = date or datetime.now().strftime("%Y%m%d")
        self.date = resolved_date
        self.cycle = cycle
        self.keys = keys
        self.hours = hours if hours is not None else list(range(0, 385, 3))

        if not all(key in VARIABLES for key in self.keys):
            raise ValueError(
                f"Invalid variable keys: {self.keys}. Valid keys: {sorted(VARIABLES)}"
            )

        self.dataset: GFSDatasetManager = GFSDatasetManager(
            date=self.date,
            cycle=self.cycle,
            region=region,
            output_dir=output_dir,
            pause=pause,
        )
        self._ds: xr.Dataset
        if len(self.keys) > 1:
            self._ds = self.dataset.build_multi_dataset(self.keys, self.hours)
        else:
            self._ds = self.dataset.build_dataset(self.keys[0], self.hours)

        coords = list(self._ds.coords)
        self._LAT_DIM = lat_dim or _find_dim(coords, _LAT_CANDIDATES, "lat")
        self._LON_DIM = lon_dim or _find_dim(coords, _LON_CANDIDATES, "lon")
        self._TIME_DIM = time_dim or _find_dim(coords, _TIME_CANDIDATES, "time")

        log.info(
            "Dimensions resolved → lat='%s', lon='%s', time='%s'",
            self._LAT_DIM,
            self._LON_DIM,
            self._TIME_DIM,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @cached_property
    def _lon_min(self) -> float:
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
    # Direct key access
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> xr.Variable:
        if key not in self._ds.variables:
            available = list(self._ds.variables)
            raise KeyError(f"Variable '{key}' not found. Available: {available}")
        return self._ds.variables[key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_keys(self) -> dict[str, str]:
        """Return ``{variable: long_name}`` for every variable in the dataset."""
        return {
            str(k): str(v.attrs.get("long_name", ""))
            for k, v in self._ds.variables.items()
        }

    def get_data_from_point(
        self,
        point: Coordinate,
        *,
        time: Optional[Union[str, slice, list]] = None,
        tolerance: Optional[float] = None,
    ) -> _DatasetView:
        """Select the nearest grid point to *(lat, lon)*.

        Parameters
        ----------
        point     : ``(lat, lon)`` in degrees. Longitude is normalised to the
                    dataset's convention automatically.
        time      : Optional time selector accepted by ``xr.Dataset.sel``.
        tolerance : Maximum allowed distance in degrees.
        """
        lat, lon = point
        lat, lon = self._normalize_point(lat, lon)
        self._warn_if_out_of_bounds(lat, lon)

        sel_kwargs: dict = {self._LAT_DIM: lat, self._LON_DIM: lon}
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
        """Geocode *place* and return data from the nearest grid point."""
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
        """Return the complete time series at the nearest grid point.

        Parameters
        ----------
        point    : ``(lat, lon)`` in degrees.
        variable : If given, return only that variable as a ``DataArray``.
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

    def __repr__(self) -> str:
        return (
            f"get_noaa_data("
            f"date={self.date!r}, cycle={self.cycle!r}, "
            f"key={self.keys!r}, bounds=[{self.bounds}])"
        )
