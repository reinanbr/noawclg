"""Coordinate utilities: dimension detection, bbox, date/lon helpers."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Hashable

log = logging.getLogger(__name__)

# Candidate dimension names, ordered by preference (most common first)
_LAT_CANDIDATES = ("lat", "latitude", "y", "nav_lat", "rlat", "XLAT")
_LON_CANDIDATES = ("lon", "longitude", "x", "nav_lon", "rlon", "XLONG")
_TIME_CANDIDATES = ("time", "Time", "t", "forecast_hour", "step")


def _find_dim(
    coords: Sequence[Hashable], candidates: tuple[str, ...], label: str
) -> str:
    """Return the first candidate name present in *coords*.

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


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box of a dataset's spatial domain."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def contains(self, lat: float, lon: float) -> bool:
        return (
            self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max
        )

    def __str__(self) -> str:
        return (
            f"lat=[{self.lat_min:.2f}, {self.lat_max:.2f}], "
            f"lon=[{self.lon_min:.2f}, {self.lon_max:.2f}]"
        )


def _parse_date(date: str) -> str:
    """Convert 'DD/MM/YYYY' → 'YYYYMMDD'."""
    return datetime.strptime(date, "%d/%m/%Y").strftime("%Y%m%d")


def _normalize_lon(lon: float, lon_min: float) -> float:
    """Normalize *lon* to match the dataset's longitude convention.

    Detects automatically whether the dataset uses [0, 360] or [-180, 180].
    """
    if lon_min < 0:
        return (lon + 180) % 360 - 180
    return lon % 360


def auto_date(lag_days: int = 1) -> tuple[str, str]:
    """Return ``(date, cycle)`` ready for ``GFSDatasetManager``.

    ``date`` is in ``'DD/MM/YYYY'`` format; ``cycle`` is one of
    ``'00'``, ``'06'``, ``'12'``, ``'18'``. Using ``lag_days=1``
    targets yesterday's run, which is always available on NOMADS.
    """
    now = datetime.now(timezone.utc)
    run_date = now - timedelta(days=lag_days)
    available_hour = now.hour - 4  # GFS takes ~4 h to publish
    if available_hour >= 18:
        cycle = "18"
    elif available_hour >= 12:
        cycle = "12"
    elif available_hour >= 6:
        cycle = "06"
    else:
        cycle = "00"
    return run_date.strftime("%d/%m/%Y"), cycle
