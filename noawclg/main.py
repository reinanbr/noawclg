"""Backward-compatibility shim.

Everything that was once defined here has been moved to focused sub-modules.
Import from the canonical locations going forward:

    from noawclg.query  import get_noaa_data
    from noawclg.coords import BoundingBox, _find_dim, _parse_date, _normalize_lon
    from noawclg.view   import _DatasetView
"""

from noawclg.coords import (  # noqa: F401
    BoundingBox,
    _LAT_CANDIDATES,
    _LON_CANDIDATES,
    _TIME_CANDIDATES,
    _find_dim,
    _normalize_lon,
    _parse_date,
)
from noawclg.gfs_dataset import GFSDatasetManager  # noqa: F401  (kept for patch compat)
from noawclg.query import _GEOLOCATOR, get_noaa_data  # noqa: F401
from noawclg.view import _DatasetView  # noqa: F401

__version__ = "2.2.7"
__author__ = "Reinan Br"
