"""Convenience wrapper: load GFS data directly as an xr.Dataset."""

from __future__ import annotations

from typing import Optional

import xarray as xr

from noawclg.query import get_noaa_data


def load(
    date: Optional[str] = None,
    cycle: str = "00",
    keys: list[str] = ["t2m"],
    hours: Optional[list[int]] = None,
    *,
    lat_dim: Optional[str] = None,
    lon_dim: Optional[str] = None,
    time_dim: Optional[str] = None,
    region: Optional[dict[str, float]] = None,
) -> xr.Dataset:
    """Load NOAA GFS data and return the underlying xr.Dataset directly.

    Parameters
    ----------
    date    : Date in 'DD/MM/YYYY' format. Defaults to today.
    cycle   : Model cycle ('00', '06', '12', '18').
    keys    : Variable keys to load (e.g. ['t2m', 'prate']).
    hours   : Forecast hours to include. Defaults to every 3 h up to 384 h.
    lat_dim : Override auto-detected latitude dimension name.
    lon_dim : Override auto-detected longitude dimension name.
    time_dim: Override auto-detected time dimension name.
    region  : Bounding box dict with toplat/bottomlat/leftlon/rightlon.

    Returns
    -------
    xr.Dataset
    """
    return get_noaa_data(
        date=date,
        cycle=cycle,
        keys=keys,
        hours=hours,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        time_dim=time_dim,
        region=region,
    )._ds
