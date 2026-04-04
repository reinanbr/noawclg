"""
GFS Dataset Manager  (v2 — single-download)
============================================
Downloads **one complete GRIB2 file per forecast hour** via the multi-variable
grib-filter endpoint, then extracts every requested variable from that single
file — eliminating the redundant downloads of the previous version (which
fetched 1 file per variable per hour).

New download flow:
    For each requested hour:

    1. Build ONE grib-filter URL containing ALL requested variables/levels.
    2. Download the file once → cache to disk.
    3. Open the GRIB2 and extract each variable with cfgrib.

Old flow (removed):
    One download per variable × per hour  (5 vars × 24 h = 120 requests,
    now reduced to 24).

Example:
    Basic multi-variable download::

        from noawclg import GFSDatasetManager

        mgr = GFSDatasetManager(date="20260403", cycle="06")
        ds  = mgr.build_multi_dataset(
            var_keys=["t2m", "prate", "prmsl", "u10", "v10"],
            hours=list(range(0, 49, 6)),
        )
        mgr.save_netcdf(ds, "/tmp/gfs_48h.nc")

Dependencies:
    pip install requests cfgrib xarray numpy netCDF4 zarr eccodes tqdm
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cfgrib
import numpy as np
import requests
import xarray as xr
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

LOG = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Variable catalogue
# ══════════════════════════════════════════════════════════════════════════════

VARIABLES: dict[str, dict[str, Any]] = {

    # ── 2-metre / surface single-level ───────────────────────────────────────
    "t2m": {
        "short":     "t2m",
        "long_name": "2 metre temperature",
        "units":     "C",
        "tlev":      "heightAboveGround",
        "levels":    [2],
        "grib_var":  "var_TMP",
        "grib_lev":  "lev_2_m_above_ground",
        "converter": lambda x: x - 273.15,
    },
    "d2m": {
        "short":     "d2m",
        "long_name": "2 metre dewpoint temperature",
        "units":     "C",
        "tlev":      "heightAboveGround",
        "levels":    [2],
        "grib_var":  "var_DPT",
        "grib_lev":  "lev_2_m_above_ground",
        "converter": lambda x: x - 273.15,
    },
    "r2": {
        "short":     "r2",
        "long_name": "2 metre relative humidity",
        "units":     "%",
        "tlev":      "heightAboveGround",
        "levels":    [2],
        "grib_var":  "var_RH",
        "grib_lev":  "lev_2_m_above_ground",
        "converter": None,
    },
    "sh2": {
        "short":     "sh2",
        "long_name": "2 metre specific humidity",
        "units":     "kg kg**-1",
        "tlev":      "heightAboveGround",
        "levels":    [2],
        "grib_var":  "var_SPFH",
        "grib_lev":  "lev_2_m_above_ground",
        "converter": None,
    },
    "aptmp": {
        "short":     "aptmp",
        "long_name": "Apparent temperature",
        "units":     "C",
        "tlev":      "heightAboveGround",
        "levels":    [2],
        "grib_var":  "var_APTMP",
        "grib_lev":  "lev_2_m_above_ground",
        "converter": lambda x: x - 273.15,
    },

    # ── 10-metre wind ─────────────────────────────────────────────────────────
    "u10": {
        "short":     "u10",
        "long_name": "10 metre U wind component",
        "units":     "m s**-1",
        "tlev":      "heightAboveGround",
        "levels":    [10],
        "grib_var":  "var_UGRD",
        "grib_lev":  "lev_10_m_above_ground",
        "converter": None,
    },
    "v10": {
        "short":     "v10",
        "long_name": "10 metre V wind component",
        "units":     "m s**-1",
        "tlev":      "heightAboveGround",
        "levels":    [10],
        "grib_var":  "var_VGRD",
        "grib_lev":  "lev_10_m_above_ground",
        "converter": None,
    },
    "gust": {
        "short":     "gust",
        "long_name": "Wind speed (gust)",
        "units":     "m s**-1",
        "tlev":      "surface",
        "levels":    None,
        "grib_var":  "var_GUST",
        "grib_lev":  "lev_surface",
        "converter": None,
    },

    # ── surface / single-level ────────────────────────────────────────────────
    "prmsl": {
        "short":     "prmsl",
        "long_name": "Pressure reduced to MSL",
        "units":     "hPa",
        "tlev":      "meanSea",
        "levels":    None,
        "grib_var":  "var_PRMSL",
        "grib_lev":  "lev_mean_sea_level",
        "converter": lambda x: x / 100,
    },
    "mslet": {
        "short":     "mslet",
        "long_name": "MSLP (Eta model reduction)",
        "units":     "hPa",
        "tlev":      "meanSea",
        "levels":    None,
        "grib_var":  "var_MSLET",
        "grib_lev":  "lev_mean_sea_level",
        "converter": lambda x: x / 100,
    },
    "sp": {
        "short":     "sp",
        "long_name": "Surface pressure",
        "units":     "hPa",
        "tlev":      "surface",
        "levels":    None,
        "grib_var":  "var_PRES",
        "grib_lev":  "lev_surface",
        "converter": lambda x: x / 100,
    },
    "orog": {
        "short":     "orog",
        "long_name": "Orography",
        "units":     "m",
        "tlev":      "surface",
        "levels":    None,
        "grib_var":  "var_HGT",
        "grib_lev":  "lev_surface",
        "converter": None,
    },
    "lsm": {
        "short":     "lsm",
        "long_name": "Land-sea mask",
        "units":     "0 - 1",
        "tlev":      "surface",
        "levels":    None,
        "grib_var":  "var_LAND",
        "grib_lev":  "lev_surface",
        "converter": None,
    },
    "vis": {
        "short":     "vis",
        "long_name": "Visibility",
        "units":     "m",
        "tlev":      "surface",
        "levels":    None,
        "grib_var":  "var_VIS",
        "grib_lev":  "lev_surface",
        "converter": None,
    },

    # ── precipitation / hydrology ─────────────────────────────────────────────
    "prate": {
        "short":     "prate",
        "long_name": "Precipitation rate",
        "units":     "kg m**-2 s**-1",
        "tlev":      "surface",
        "levels":    None,
        "grib_var":  "var_PRATE",
        "grib_lev":  "lev_surface",
        "converter": None,
    },
    "cpofp": {
        "short":     "cpofp",
        "long_name": "Percent frozen precipitation",
        "units":     "%",
        "tlev":      "surface",
        "levels":    None,
        "grib_var":  "var_CPOFP",
        "grib_lev":  "lev_surface",
        "converter": None,
    },
    "crain": {
        "short":     "crain",
        "long_name": "Categorical rain",
        "units":     "Code table 4.222",
        "tlev":      "surface",
        "levels":    None,
        "grib_var":  "var_CRAIN",
        "grib_lev":  "lev_surface",
        "converter": None,
    },
    "csnow": {
        "short":     "csnow",
        "long_name": "Categorical snow",
        "units":     "Code table 4.222",
        "tlev":      "surface",
        "levels":    None,
        "grib_var":  "var_CSNOW",
        "grib_lev":  "lev_surface",
        "converter": None,
    },
    "cfrzr": {
        "short":     "cfrzr",
        "long_name": "Categorical freezing rain",
        "units":     "Code table 4.222",
        "tlev":      "surface",
        "levels":    None,
        "grib_var":  "var_CFRZR",
        "grib_lev":  "lev_surface",
        "converter": None,
    },
    "cicep": {
        "short":     "cicep",
        "long_name": "Categorical ice pellets",
        "units":     "Code table 4.222",
        "tlev":      "surface",
        "levels":    None,
        "grib_var":  "var_CICEP",
        "grib_lev":  "lev_surface",
        "converter": None,
    },
    "sde": {
        "short":     "sde",
        "long_name": "Snow depth",
        "units":     "m",
        "tlev":      "surface",
        "levels":    None,
        "grib_var":  "var_SNOD",
        "grib_lev":  "lev_surface",
        "converter": None,
    },
    "sdwe": {
        "short":     "sdwe",
        "long_name": "Water equivalent of accumulated snow depth",
        "units":     "kg m**-2",
        "tlev":      "surface",
        "levels":    None,
        "grib_var":  "var_WEASD",
        "grib_lev":  "lev_surface",
        "converter": None,
    },
    "pwat": {
        "short":     "pwat",
        "long_name": "Precipitable water",
        "units":     "kg m**-2",
        "tlev":      "atmosphereSingleLayer",
        "levels":    None,
        "grib_var":  "var_PWAT",
        "grib_lev":  "lev_entire_atmosphere_(considered_as_a_single_layer)",
        "converter": None,
    },
    "cwat": {
        "short":     "cwat",
        "long_name": "Cloud water",
        "units":     "kg m**-2",
        "tlev":      "atmosphere",
        "levels":    None,
        "grib_var":  "var_CWAT",
        "grib_lev":  "lev_entire_atmosphere",
        "converter": None,
    },

    # ── cloud cover ───────────────────────────────────────────────────────────
    "tcc": {
        "short":     "tcc",
        "long_name": "Total cloud cover",
        "units":     "%",
        "tlev":      "atmosphere",
        "levels":    None,
        "grib_var":  "var_TCDC",
        "grib_lev":  "lev_entire_atmosphere",
        "converter": None,
    },
    "lcc": {
        "short":     "lcc",
        "long_name": "Low cloud cover",
        "units":     "%",
        "tlev":      "lowCloudLayer",
        "levels":    None,
        "grib_var":  "var_TCDC",
        "grib_lev":  "lev_low_cloud_layer",
        "converter": None,
    },
    "mcc": {
        "short":     "mcc",
        "long_name": "Medium cloud cover",
        "units":     "%",
        "tlev":      "middleCloudLayer",
        "levels":    None,
        "grib_var":  "var_TCDC",
        "grib_lev":  "lev_middle_cloud_layer",
        "converter": None,
    },
    "hcc": {
        "short":     "hcc",
        "long_name": "High cloud cover",
        "units":     "%",
        "tlev":      "highCloudLayer",
        "levels":    None,
        "grib_var":  "var_TCDC",
        "grib_lev":  "lev_high_cloud_layer",
        "converter": None,
    },

    # ── convection / instability ──────────────────────────────────────────────
    "cape": {
        "short":     "cape",
        "long_name": "Convective available potential energy",
        "units":     "J kg**-1",
        "tlev":      "surface",
        "levels":    None,
        "grib_var":  "var_CAPE",
        "grib_lev":  "lev_surface",
        "converter": None,
        "multilevel": True,
    },
    "cin": {
        "short":     "cin",
        "long_name": "Convective inhibition",
        "units":     "J kg**-1",
        "tlev":      "surface",
        "levels":    None,
        "grib_var":  "var_CIN",
        "grib_lev":  "lev_surface",
        "converter": None,
        "multilevel": True,
    },
    "lftx": {
        "short":     "lftx",
        "long_name": "Surface lifted index",
        "units":     "K",
        "tlev":      "surface",
        "levels":    None,
        "grib_var":  "var_LFTX",
        "grib_lev":  "lev_surface",
        "converter": None,
    },
    "lftx4": {
        "short":     "lftx4",
        "long_name": "Best (4-layer) lifted index",
        "units":     "K",
        "tlev":      "surface",
        "levels":    None,
        "grib_var":  "var_4LFTX",
        "grib_lev":  "lev_surface",
        "converter": None,
    },
    "hlcy": {
        "short":     "hlcy",
        "long_name": "Storm relative helicity",
        "units":     "m**2 s**-2",
        "tlev":      "heightAboveGroundLayer",
        "levels":    None,
        "grib_var":  "var_HLCY",
        "grib_lev":  "lev_height_above_ground_layer",
        "converter": None,
    },

    # ── upper-air multi-level (isobaric) ──────────────────────────────────────
    "t": {
        "short":     "t",
        "long_name": "Temperature",
        "units":     "C",
        "tlev":      "isobaricInhPa",
        "levels":    [80, 100, 150, 200, 250, 300, 400, 500,
                      600, 700, 850, 925, 1000],
        "grib_var":  "var_TMP",
        "grib_lev":  "lev_mb",
        "converter": lambda x: x - 273.15,
        "multilevel": True,
    },
    "r": {
        "short":     "r",
        "long_name": "Relative humidity",
        "units":     "%",
        "tlev":      "isobaricInhPa",
        "levels":    [80, 100, 150, 200, 250, 300, 400, 500,
                      600, 700, 850, 925, 1000],
        "grib_var":  "var_RH",
        "grib_lev":  "lev_mb",
        "converter": None,
        "multilevel": True,
    },
    "q": {
        "short":     "q",
        "long_name": "Specific humidity",
        "units":     "kg kg**-1",
        "tlev":      "isobaricInhPa",
        "levels":    [80, 1000],
        "grib_var":  "var_SPFH",
        "grib_lev":  "lev_mb",
        "converter": None,
        "multilevel": True,
    },
    "gh": {
        "short":     "gh",
        "long_name": "Geopotential height",
        "units":     "gpm",
        "tlev":      "isobaricInhPa",
        "levels":    [500, 700, 850, 925, 1000],
        "grib_var":  "var_HGT",
        "grib_lev":  "lev_mb",
        "converter": None,
        "multilevel": True,
    },
    "u": {
        "short":     "u",
        "long_name": "U component of wind",
        "units":     "m s**-1",
        "tlev":      "isobaricInhPa",
        "levels":    [200, 250, 300, 400, 500, 700, 850, 925, 1000],
        "grib_var":  "var_UGRD",
        "grib_lev":  "lev_mb",
        "converter": None,
        "multilevel": True,
    },
    "v": {
        "short":     "v",
        "long_name": "V component of wind",
        "units":     "m s**-1",
        "tlev":      "isobaricInhPa",
        "levels":    [200, 250, 300, 400, 500, 700, 850, 925, 1000],
        "grib_var":  "var_VGRD",
        "grib_lev":  "lev_mb",
        "converter": None,
        "multilevel": True,
    },
    "w": {
        "short":     "w",
        "long_name": "Vertical velocity",
        "units":     "Pa s**-1",
        "tlev":      "isobaricInhPa",
        "levels":    [100, 200, 300, 400, 500, 600, 700, 850],
        "grib_var":  "var_VVEL",
        "grib_lev":  "lev_mb",
        "converter": None,
        "multilevel": True,
    },
    "absv": {
        "short":     "absv",
        "long_name": "Absolute vorticity",
        "units":     "s**-1",
        "tlev":      "isobaricInhPa",
        "levels":    [100, 200, 300, 400, 500, 700, 850, 1000],
        "grib_var":  "var_ABSV",
        "grib_lev":  "lev_mb",
        "converter": None,
        "multilevel": True,
    },

    # ── soil (4-layer) ────────────────────────────────────────────────────────
    "st": {
        "short":     "st",
        "long_name": "Soil temperature",
        "units":     "C",
        "tlev":      "depthBelowLandLayer",
        "levels":    [0, 10, 40, 100],
        "grib_var":  "var_TSOIL",
        "grib_lev":  "lev_depth_below_land_layer",
        "converter": lambda x: x - 273.15,
        "multilevel": True,
    },
    "soilw": {
        "short":     "soilw",
        "long_name": "Volumetric soil moisture content",
        "units":     "Proportion",
        "tlev":      "depthBelowLandLayer",
        "levels":    [0, 10, 40, 100],
        "grib_var":  "var_SOILW",
        "grib_lev":  "lev_depth_below_land_layer",
        "converter": None,
        "multilevel": True,
    },

    # ── diagnostics ───────────────────────────────────────────────────────────
    "refc": {
        "short":     "refc",
        "long_name": "Maximum/Composite radar reflectivity",
        "units":     "dB",
        "tlev":      "atmosphere",
        "levels":    None,
        "grib_var":  "var_REFC",
        "grib_lev":  "lev_entire_atmosphere",
        "converter": None,
    },
    "siconc": {
        "short":     "siconc",
        "long_name": "Sea ice area fraction",
        "units":     "0 - 1",
        "tlev":      "surface",
        "levels":    None,
        "grib_var":  "var_ICEC",
        "grib_lev":  "lev_surface",
        "converter": None,
    },
    "veg": {
        "short":     "veg",
        "long_name": "Vegetation",
        "units":     "%",
        "tlev":      "surface",
        "levels":    None,
        "grib_var":  "var_VEG",
        "grib_lev":  "lev_surface",
        "converter": None,
    },
    "tozne": {
        "short":     "tozne",
        "long_name": "Total ozone",
        "units":     "DU",
        "tlev":      "atmosphere",
        "levels":    None,
        "grib_var":  "var_TOZNE",
        "grib_lev":  "lev_entire_atmosphere",
        "converter": None,
    },
}

# ── convenient subsets ────────────────────────────────────────────────────────
SURFACE_VARS: list[str] = [
    k for k, v in VARIABLES.items() if not v.get("multilevel")
]
MULTILEVEL_VARS: list[str] = [
    k for k, v in VARIABLES.items() if v.get("multilevel")
]

# ── pre-defined hour sequences ────────────────────────────────────────────────
HOURS_16DAYS    = list(range(0, 121, 6)) + list(range(123, 385, 3))
HOURS_5DAYS_1H  = list(range(0, 121))
HOURS_10DAYS_3H = list(range(0, 241, 3))
HOURS_16DAYS_3H = list(range(0, 121, 3)) + list(range(123, 385, 3))

# NOMADS grib-filter URL template — accepts multiple &var_XXX=on&lev_XXX=on
_FILTER_BASE = (
    "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl"
    "?dir=/gfs.{date}/{cycle}/atmos"
    "&file=gfs.t{cycle}z.pgrb2.0p25.f{hour:03d}"
    "{var_params}"
    "{region_params}"
)

# HTTP session defaults
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; GFS-downloader/2.0; "
        "+https://github.com/reinanbr/noawclg)"
    )
}
_RETRY = Retry(
    total=4,
    backoff_factor=2,
    status_forcelist={429, 500, 502, 503, 504},
    allowed_methods={"GET", "HEAD"},
    raise_on_status=False,
)


def _build_session() -> requests.Session:
    """Create a :class:`requests.Session` pre-configured for NOMADS.

    Applies a browser-like User-Agent header (required — NOMADS blocks the
    default ``python-requests`` agent with HTTP 403) and an exponential-backoff
    retry policy on transient server errors.

    Returns:
        requests.Session: A ready-to-use session with headers and retry adapter
            mounted for both ``https://`` and ``http://``.
    """
    s = requests.Session()
    s.headers.update(_HEADERS)
    adapter = HTTPAdapter(max_retries=_RETRY)
    s.mount("https://", adapter)
    s.mount("http://",  adapter)
    return s


# ══════════════════════════════════════════════════════════════════════════════
# Manager
# ══════════════════════════════════════════════════════════════════════════════

class GFSDatasetManager:
    """Download GFS GRIB2 files (once per hour) and assemble xarray Datasets.

    The manager targets the NOAA NOMADS grib-filter endpoint and bundles every
    requested variable into a single URL per forecast hour, reducing HTTP
    traffic from ``N_vars × N_hours`` to just ``N_hours``.

    Downloaded GRIB2 files are cached under ``output_dir`` with deterministic
    filenames; subsequent calls with the same parameters skip the network
    entirely.

    Attributes:
        date (str): Model run date in ``YYYYMMDD`` format.
        cycle (str): Model run cycle (``"00"``, ``"06"``, ``"12"`` or ``"18"``).
        output_dir (Path): Absolute path to the GRIB2 cache directory.
        region (dict | None): Active bounding-box, or ``None`` for global.
        pause (float): Seconds to sleep between consecutive HTTP requests.

    Example:
        Download 2-m temperature over Brazil for a 48-h forecast::

            mgr = GFSDatasetManager(
                date="20260403",
                cycle="06",
                region={"toplat": 5, "bottomlat": -35,
                        "leftlon": -75, "rightlon": -34},
            )
            ds = mgr.build_dataset("t2m", hours=list(range(0, 49, 6)))
            mgr.save_netcdf(ds, "t2m_brasil_48h.nc")
    """

    def __init__(
        self,
        date: str,
        cycle: str = "00",
        output_dir: str = "./gfs_output",
        region: dict[str, float] | None = None,
        pause: float = 1.5,
    ) -> None:
        """Initialise the manager and validate run parameters.

        Args:
            date (str): Model run date in ``YYYYMMDD`` format
                (e.g. ``"20260403"``).
            cycle (str): Model run cycle. Must be one of ``"00"``, ``"06"``,
                ``"12"`` or ``"18"``. Defaults to ``"00"``.
            output_dir (str): Directory where GRIB2 files are cached.
                Created automatically if it does not exist. Relative paths
                are resolved to an absolute path via :func:`Path.resolve`.
                Defaults to ``"./gfs_output"``.
            region (dict | None): Optional spatial bounding-box with keys
                ``toplat``, ``bottomlat``, ``leftlon``, ``rightlon``
                (all floats, degrees). Pass ``None`` to download the full
                global grid. Defaults to ``None``.
            pause (float): Seconds to sleep between consecutive HTTP requests.
                Increase this value if NOMADS rate-limits your downloads.
                Defaults to ``1.5``.

        Raises:
            ValueError: If ``date`` does not match the ``YYYYMMDD`` format.
            ValueError: If ``cycle`` is not one of the four valid values.
        """
        datetime.strptime(date, "%Y%m%d")
        if cycle not in {"00", "06", "12", "18"}:
            raise ValueError("cycle must be one of: 00, 06, 12, 18")

        self.date       = date
        self.cycle      = cycle
        self.output_dir = Path(output_dir).resolve()   # always absolute
        self.region     = region
        self.pause      = pause
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._run_dt    = datetime.strptime(f"{date}{cycle}", "%Y%m%d%H")
        self._session   = _build_session()

    # ── private helpers ───────────────────────────────────────────────────────

    def _region_params(self) -> str:
        """Build the grib-filter sub-region query string fragment.

        Returns:
            str: A query string such as
                ``"&subregion=&toplat=5&bottomlat=-35&leftlon=-75&rightlon=-34"``
                when a region is set, or an empty string for the global grid.
        """
        if not self.region:
            return ""
        r = self.region
        return (
            f"&subregion=&toplat={r['toplat']}&bottomlat={r['bottomlat']}"
            f"&leftlon={r['leftlon']}&rightlon={r['rightlon']}"
        )

    def _var_params(self, var_keys: list[str]) -> str:
        """Build the multi-variable query string for the grib-filter URL.

        Iterates over ``var_keys``, looks up each variable's ``grib_var`` and
        ``grib_lev`` fields from :data:`VARIABLES`, and concatenates them into
        a single query-string fragment. Duplicate pairs (e.g. two variables
        that share the same level token) are de-duplicated.

        Args:
            var_keys (list[str]): Keys from :data:`VARIABLES` to include.

        Returns:
            str: A concatenated query string such as
                ``"&var_TMP=on&lev_2_m_above_ground=on&var_PRATE=on&lev_surface=on"``.
        """
        seen: set[str] = set()
        parts: list[str] = []
        for vk in var_keys:
            cfg  = VARIABLES[vk]
            pair = f"&{cfg['grib_var']}=on&{cfg['grib_lev']}=on"
            if pair not in seen:
                seen.add(pair)
                parts.append(pair)
        return "".join(parts)

    def _filter_url(self, var_keys: list[str], hour: int) -> str:
        """Construct the complete grib-filter URL for a given hour.

        The URL embeds ALL requested variables and the optional region
        bounding-box, so only one HTTP request is needed per forecast hour.

        Args:
            var_keys (list[str]): Variable keys to include in the request.
            hour (int): Forecast hour offset from the model run time
                (e.g. ``24`` for the 24-hour forecast).

        Returns:
            str: A fully-formed NOMADS grib-filter URL ready for GET.
        """
        return _FILTER_BASE.format(
            date=self.date,
            cycle=self.cycle,
            hour=hour,
            var_params=self._var_params(var_keys),
            region_params=self._region_params(),
        )

    def _cache_path(self, var_keys: list[str], hour: int) -> Path:
        """Return the deterministic on-disk path for a cached GRIB2 file.

        The filename encodes the run date, cycle, variable set (sorted and
        truncated to 60 characters), region tag, and forecast hour, ensuring
        that different variable combinations or regions never collide.

        Args:
            var_keys (list[str]): Variable keys included in the download.
            hour (int): Forecast hour offset.

        Returns:
            Path: Absolute path under ``output_dir``, e.g.::

                /data/gfs_output/gfs_20260403_06z_prate_t2m_5N35S75W34E_f024.grib2
        """
        tag = "global"
        if self.region:
            r   = self.region
            tag = (
                f"{r['toplat']}N{abs(r['bottomlat'])}S"
                f"{abs(r['leftlon'])}W{r['rightlon']}E"
            )
        vkey = "_".join(sorted(var_keys))[:60]
        return self.output_dir / (
            f"gfs_{self.date}_{self.cycle}z_{vkey}_{tag}_f{hour:03d}.grib2"
        )

    # ── download ──────────────────────────────────────────────────────────────

    def download_hours(
        self,
        var_keys: list[str],
        hours: list[int],
        force: bool = False,
    ) -> dict[int, Path]:
        """Download one GRIB2 file per forecast hour containing all variables.

        For each hour in ``hours`` the method:

        1. Checks whether the corresponding cache file already exists.
        2. If cached (and ``force=False``), adds it to the result immediately
           without any network I/O.
        3. Otherwise, issues a single GET request to the NOMADS grib-filter
           endpoint, streams the response to disk, validates the file size,
           and logs progress.

        A ``pause`` second sleep is inserted between requests to avoid
        overwhelming the NOMADS server.

        Args:
            var_keys (list[str]): Variable keys to bundle into each download
                URL. Must all be present in :data:`VARIABLES`.
            hours (list[int]): Forecast hour offsets to download
                (e.g. ``[0, 6, 12, 24]``).
            force (bool): If ``True``, re-download files that already exist on
                disk. Defaults to ``False``.

        Returns:
            dict[int, Path]: Mapping of ``{hour: path}`` for every hour that
                was successfully downloaded or found in cache. Hours that
                received an HTTP error or an empty response are omitted.

        Raises:
            KeyError: If any key in ``var_keys`` is not present in
                :data:`VARIABLES`.

        Example:
            ::

                files = mgr.download_hours(["t2m", "prate"], hours=[0, 6, 12])
                # {0: PosixPath('.../gfs_..._f000.grib2'),
                #  6: PosixPath('.../gfs_..._f006.grib2'),
                #  12: PosixPath('.../gfs_..._f012.grib2')}
        """
        unknown = [vk for vk in var_keys if vk not in VARIABLES]
        if unknown:
            raise KeyError(f"Unknown variables: {unknown}")

        results: dict[int, Path] = {}
        hours_to_fetch: list[int] = []

        for hour in hours:
            path = self._cache_path(var_keys, hour)
            if path.exists() and not force:
                LOG.info("[cache] f%03d  %s", hour, path.name)
                results[hour] = path
            else:
                hours_to_fetch.append(hour)

        if not hours_to_fetch:
            LOG.info("All files already cached.")
            return results

        LOG.info(
            "Download: %d hour(s) × 1 file each = %d request(s)  (vars: %s)",
            len(hours_to_fetch), len(hours_to_fetch), var_keys,
        )

        iterator = (
            tqdm(hours_to_fetch, desc="GFS download", unit="h")
            if _HAS_TQDM else hours_to_fetch
        )
        total   = len(hours_to_fetch)
        counter = 0
        t_start = time.time()

        for hour in iterator:
            t_iter    = time.time()
            path      = self._cache_path(var_keys, hour)
            url       = self._filter_url(var_keys, hour)
            var_label = var_keys[0] if len(var_keys) == 1 else "multi"
            LOG.info("[%s] → f%03d  %s", var_label, hour, url[:120])

            try:
                resp = self._session.get(url, timeout=60, stream=True)

                if resp.status_code != 200:
                    LOG.warning(
                        "var=%s  f%03d: HTTP %d — skipping",
                        var_label, hour, resp.status_code,
                    )
                    time.sleep(self.pause)
                    continue

                bytes_written = 0
                with path.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
                            bytes_written += len(chunk)

                if bytes_written < 100:
                    LOG.warning(
                        "var=%s  f%03d: empty response (%d bytes) — discarding",
                        var_label, hour, bytes_written,
                    )
                    path.unlink(missing_ok=True)
                else:
                    results[hour] = path
                    counter  += 1
                    elapsed   = time.time() - t_start
                    ping      = time.time() - t_iter
                    remaining = ping * (total - counter)
                    LOG.info(
                        "  [ok] f%03d  %.0f KB  |  %.1f%%  (%d/%d)"
                        "  elapsed=%.1fs  remaining≈%.1fs",
                        hour, bytes_written / 1024,
                        (counter / total) * 100, counter, total,
                        elapsed, remaining,
                    )

            except requests.RequestException as exc:
                LOG.error("var=%s  f%03d: network error — %s", var_label, hour, exc)

            time.sleep(self.pause)

        return results

    # ── cfgrib extraction ─────────────────────────────────────────────────────

    def _open_var(self, path: Path, var_key: str) -> xr.Dataset | None:
        """Open a GRIB2 file and isolate the dataset for one variable.

        Uses a cascade of ``cfgrib`` filter strategies, from most restrictive
        to most permissive, to handle GRIB table inconsistencies between NCEP
        and ECMWF encodings that cause the ``shortName`` field to differ across
        GFS versions or sub-region files:

        1. ``shortName + typeOfLevel + level`` — single-level vars only.
        2. ``shortName + typeOfLevel`` — drops the level filter.
        3. ``typeOfLevel + level`` — drops shortName; single-level vars only.
        4. ``typeOfLevel`` only — most relaxed filter.
        5. Full scan via ``cfgrib.open_datasets`` — last resort; searches
           first by ``shortName`` match, then by ``typeOfLevel`` match across
           all sub-datasets in the file.

        Args:
            path (Path): Path to the GRIB2 file on disk.
            var_key (str): Key from :data:`VARIABLES` identifying the variable
                to extract.

        Returns:
            xr.Dataset | None: An ``xr.Dataset`` containing at least one
                data variable if extraction succeeded, or ``None`` if all
                strategies failed (a warning is logged in that case).
        """
        cfg   = VARIABLES[var_key]
        is_ml = bool(cfg.get("multilevel"))

        def _non_empty(ds: xr.Dataset) -> xr.Dataset | None:
            """Return *ds* if it contains at least one data variable, else ``None``."""
            return ds if len(ds.data_vars) > 0 else None

        base_tlev = {"typeOfLevel": cfg["tlev"]}
        f_short   = {**base_tlev, "shortName": cfg["short"]}

        candidates: list[dict] = []
        if not is_ml and cfg.get("levels"):
            candidates.append({**f_short, "level": cfg["levels"][0]})
        candidates.append(f_short)
        if not is_ml and cfg.get("levels"):
            candidates.append({**base_tlev, "level": cfg["levels"][0]})
        candidates.append(base_tlev)

        last_exc: Exception | None = None

        for filters in candidates:
            try:
                ds     = cfgrib.open_dataset(path, filter_by_keys=filters, indexpath=None)
                result = _non_empty(ds)
                if result is not None:
                    LOG.debug(
                        "'%s' found with filters %s → vars: %s",
                        var_key, filters, list(ds.data_vars),
                    )
                    return result
            except Exception as exc:
                last_exc = exc

        # Last resort: full scan
        try:
            all_ds = cfgrib.open_datasets(path, indexpath=None)
            LOG.debug("'%s': full scan — %d sub-datasets found", var_key, len(all_ds))

            for ds in all_ds:   # pass 1: shortName match
                if not ds.data_vars:
                    continue
                if cfg["short"] in ds.data_vars:
                    LOG.debug("'%s' found via full scan (shortName match)", var_key)
                    return ds

            for ds in all_ds:   # pass 2: typeOfLevel match
                if not ds.data_vars:
                    continue
                tlev_val: str | None = None
                if "typeOfLevel" in ds.coords:
                    tlev_val = str(ds.coords["typeOfLevel"].values)
                elif "GRIB_typeOfLevel" in getattr(ds, "attrs", {}):
                    tlev_val = ds.attrs["GRIB_typeOfLevel"]
                else:
                    first_da = ds[list(ds.data_vars)[0]]
                    tlev_val = first_da.attrs.get("GRIB_typeOfLevel")

                if tlev_val == cfg["tlev"]:
                    LOG.debug(
                        "'%s' found via full scan (typeOfLevel=%s)",
                        var_key, tlev_val,
                    )
                    return ds

        except Exception as exc:
            last_exc = exc

        LOG.warning("Could not read '%s' from %s: %s", var_key, path.name, last_exc)
        return None

    def _extract(
        self, ds: xr.Dataset, var_key: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Extract and normalise (lats, lons, data) from an open xr.Dataset.

        Applies several correctness fixes over a naive ``np.squeeze`` approach:

        * Detects and rejects empty datasets before any array access.
        * Handles curvilinear 2-D lat/lon grids by collapsing them to 1-D
          (first column for latitudes, first row for longitudes).
        * Normalises longitudes from the 0–360° convention to –180–180° and
          reorders the longitude axis accordingly.
        * Removes leading singleton dimensions iteratively (e.g. a ``step``
          dimension of size 1 added by cfgrib) while always preserving the
          target shape: ``(lat, lon)`` for surface variables and
          ``(level, lat, lon)`` for multi-level variables.
        * Applies the variable's unit converter (e.g. K → °C) after all
          array manipulations.

        Args:
            ds (xr.Dataset): An open dataset as returned by :meth:`_open_var`.
            var_key (str): Key from :data:`VARIABLES` identifying the variable.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray] | None: A 3-tuple
                ``(lats, lons, data)`` where:

                * ``lats`` — 1-D latitude array, degrees north.
                * ``lons`` — 1-D longitude array, degrees east (–180 to 180).
                * ``data`` — 2-D ``(lat, lon)`` or 3-D ``(level, lat, lon)``
                  float array in the variable's native units after conversion.

                Returns ``None`` if the dataset is empty or the array shape
                is irrecoverably malformed (a warning is logged).
        """
        if not ds.data_vars:
            LOG.warning("'%s': dataset has no variables — skipping timestep", var_key)
            return None

        cfg   = VARIABLES[var_key]
        is_ml = bool(cfg.get("multilevel"))
        name  = list(ds.data_vars)[0]
        da    = ds[name]

        # ── lat/lon coordinates ───────────────────────────────────────────────
        if "latitude" in da.coords:
            lats_raw = da.latitude.values
        elif "lat" in da.coords:
            lats_raw = da.lat.values
        else:
            lats_raw = np.arange(da.shape[-2])

        if "longitude" in da.coords:
            lons_raw = da.longitude.values
        elif "lon" in da.coords:
            lons_raw = da.lon.values
        else:
            lons_raw = np.arange(da.shape[-1])

        # Curvilinear grid: collapse to 1-D
        if lats_raw.ndim == 2:
            lats_raw = lats_raw[:, 0]
        if lons_raw.ndim == 2:
            lons_raw = lons_raw[0, :]

        da_vals = da.values  # raw shape: (..., lat, lon)

        # ── normalise longitudes 0–360 → –180–180 ─────────────────────────────
        if lons_raw.ndim == 1 and lons_raw.max() > 180:
            lons_norm = np.where(lons_raw > 180, lons_raw - 360, lons_raw)
            sort_idx  = np.argsort(lons_norm)
            lons_raw  = lons_norm[sort_idx]
            da_vals   = da_vals[..., sort_idx]

        # ── strip leading singleton dims while preserving spatial structure ────
        keep_ndim = 3 if is_ml else 2
        while da_vals.ndim > keep_ndim and da_vals.shape[0] == 1:
            da_vals = da_vals[0]

        if da_vals.ndim < 2:
            LOG.warning(
                "'%s': unexpected shape %s after dim reduction — skipping",
                var_key, da.shape,
            )
            return None

        data = da_vals.astype(float)
        if cfg.get("converter"):
            data = cfg["converter"](data)

        return lats_raw, lons_raw, data

    # ── Dataset assembly ──────────────────────────────────────────────────────

    def _build_single_var_ds(
        self,
        var_key: str,
        files: dict[int, Path],
    ) -> xr.Dataset:
        """Extract one variable from every cached file and stack along time.

        Iterates over the forecast hours in ``files`` in ascending order,
        opens each GRIB2 with :meth:`_open_var`, normalises the array with
        :meth:`_extract`, and accumulates time slices. After all hours are
        processed the slices are stacked into a single NumPy array and wrapped
        in an :class:`xr.Dataset` with proper dimension names, coordinates,
        and metadata attributes.

        Args:
            var_key (str): Key from :data:`VARIABLES` identifying the variable
                to extract.
            files (dict[int, Path]): Mapping of ``{hour: grib2_path}`` as
                returned by :meth:`download_hours`. Hours whose files cannot
                be opened or parsed are silently skipped.

        Returns:
            xr.Dataset: Dataset with a single data variable named ``var_key``
                and the following dimensions and coordinates:

                * Surface variables — dims ``(time, latitude, longitude)``.
                * Multi-level variables — dims
                  ``(time, level, latitude, longitude)``.
                * ``time`` — :class:`datetime` objects for each valid time.
                * ``forecast_hour`` — integer forecast hour, aligned to
                  ``time``.
                * ``latitude`` / ``longitude`` — 1-D coordinate arrays.
                * ``level`` — pressure / depth levels (multi-level only).

        Raises:
            RuntimeError: If no timestep produced valid data after attempting
                all files.
        """
        cfg        = VARIABLES[var_key]
        is_ml      = bool(cfg.get("multilevel"))
        slices     : list[np.ndarray] = []
        times      : list[datetime]   = []
        fhours     : list[int]        = []
        lats_ref   = None
        lons_ref   = None
        levels_ref = None

        for hour in sorted(files):
            path = files[hour]
            ds   = self._open_var(path, var_key)
            if ds is None:
                continue

            result = self._extract(ds, var_key)
            if result is None:
                continue

            lats, lons, data = result
            if lats_ref is None:
                lats_ref = lats
                lons_ref = lons

            if is_ml and data.ndim == 3 and levels_ref is None:
                # Explicit None check — bool(DataArray) raises ValueError.
                _lev_names = ["isobaricInhPa", "depthBelowLandLayer",
                               "heightAboveGround", "level"]
                lev_coord = None
                for _n in _lev_names:
                    _c = ds.coords.get(_n)
                    if _c is not None:
                        lev_coord = _c
                        break
                levels_ref = (
                    lev_coord.values.tolist()
                    if lev_coord is not None
                    else list(range(data.shape[0]))
                )

            slices.append(data)
            times.append(self._run_dt + timedelta(hours=hour))
            fhours.append(hour)

        if not slices:
            raise RuntimeError(f"No valid data found for '{var_key}'.")

        stacked = np.stack(slices, axis=0)  # (time, [level,] lat, lon)

        coords: dict[str, Any] = {
            "time":          ("time", times),
            "forecast_hour": ("time", fhours),
            "latitude":      lats_ref,
            "longitude":     lons_ref,
        }

        if is_ml and stacked.ndim == 4:
            dims = ["time", "level", "latitude", "longitude"]
            coords["level"] = (
                levels_ref if levels_ref is not None
                else list(range(stacked.shape[1]))
            )
        else:
            dims = ["time", "latitude", "longitude"]

        return xr.Dataset(
            {
                var_key: xr.DataArray(
                    stacked,
                    dims=dims,
                    attrs={
                        "long_name": cfg["long_name"],
                        "units":     cfg["units"],
                        "gfs_run":   f"{self.date} {self.cycle}Z",
                    },
                )
            },
            coords=coords,
            attrs={
                "title":       f"GFS 0.25° — {cfg['long_name']}",
                "institution": "NCEP/NOAA",
                "source":      "GFS model output (NOMADS)",
                "run_date":    self.date,
                "run_cycle":   self.cycle,
                "created":     datetime.utcnow().isoformat() + "Z",
            },
        )

    # ── public API ────────────────────────────────────────────────────────────

    def build_dataset(
        self,
        var_key: str,
        hours: list[int],
        force_download: bool = False,
    ) -> xr.Dataset:
        """Download and assemble a Dataset for a single variable.

        Convenience wrapper that calls :meth:`download_hours` with a
        one-element variable list, then :meth:`_build_single_var_ds`.
        The total number of HTTP requests equals ``len(hours)`` (minus any
        cache hits).

        Args:
            var_key (str): Variable key from :data:`VARIABLES`
                (e.g. ``"t2m"``, ``"gh"``).
            hours (list[int]): Forecast hour offsets to include
                (e.g. ``[0, 6, 12, 24, 48]``).
            force_download (bool): Re-download even if files are cached.
                Defaults to ``False``.

        Returns:
            xr.Dataset: Dataset with a single data variable.

                * Surface variables — dims ``(time, latitude, longitude)``.
                * Multi-level variables — dims
                  ``(time, level, latitude, longitude)``.

        Raises:
            RuntimeError: If no files could be downloaded or all files failed
                to parse.

        Example:
            ::

                ds = mgr.build_dataset("t2m", hours=[0, 6, 12, 24])
                print(ds["t2m"].dims)
                # ('time', 'latitude', 'longitude')
        """
        LOG.info("Building Dataset for '%s' — hours: %s", var_key, hours)
        files = self.download_hours([var_key], hours, force=force_download)
        if not files:
            raise RuntimeError(f"No files downloaded for '{var_key}'.")
        return self._build_single_var_ds(var_key, files)

    def build_multi_dataset(
        self,
        var_keys: list[str],
        hours: list[int],
        force_download: bool = False,
    ) -> xr.Dataset:
        """Download one file per hour for all variables and build a merged Dataset.

        This is the recommended method when you need more than one variable.
        It issues ``len(hours)`` HTTP requests (instead of
        ``len(var_keys) × len(hours)``), extracts every variable from each
        cached file, and merges the individual Datasets with
        ``xr.merge(..., join="inner")``.

        Variables that fail to extract are logged as errors and skipped;
        a :class:`RuntimeError` is only raised when *all* variables fail.

        Args:
            var_keys (list[str]): List of variable keys from :data:`VARIABLES`.
            hours (list[int]): Forecast hour offsets to include.
            force_download (bool): Re-download even if files are cached.
                Defaults to ``False``.

        Returns:
            xr.Dataset: Merged Dataset containing all successfully extracted
                variables. Variables with a ``level`` dimension are included
                as-is alongside surface variables.

        Raises:
            RuntimeError: If no files are available or no variable could be
                extracted.

        Example:
            ::

                ds = mgr.build_multi_dataset(
                    var_keys=["t2m", "prmsl", "prate", "u10", "v10"],
                    hours=list(range(0, 25, 6)),
                )
                print(list(ds.data_vars))
                # ['t2m', 'prmsl', 'prate', 'u10', 'v10']
        """
        files = self.download_hours(var_keys, hours, force=force_download)
        if not files:
            raise RuntimeError("No files available.")

        datasets: list[xr.Dataset] = []
        for vk in var_keys:
            LOG.info("Extracting '%s' …", vk)
            try:
                ds = self._build_single_var_ds(vk, files)
                datasets.append(ds)
            except Exception as exc:
                LOG.error("Skipping '%s': %s", vk, exc)

        if not datasets:
            raise RuntimeError("No variables could be extracted.")

        merged = xr.merge(datasets, join="inner")
        merged.attrs["title"] = (
            f"GFS 0.25° — {', '.join(var_keys)} — {self.date} {self.cycle}Z"
        )
        return merged

    # ── persistence ───────────────────────────────────────────────────────────

    def save_netcdf(
        self,
        ds: xr.Dataset,
        filename: str,
        complevel: int = 4,
    ) -> Path:
        """Save a Dataset to a zlib-compressed NetCDF4 file.

        All data variables are compressed with zlib. The ``complevel``
        parameter trades off file size against write speed (higher = smaller
        but slower).

        Args:
            ds (xr.Dataset): Dataset to persist.
            filename (str): Output file path. Absolute paths are used as-is;
                relative paths are resolved against :attr:`output_dir`, which
                is always absolute.
            complevel (int): zlib compression level, 1 (fastest) to 9
                (smallest). Defaults to ``4``.

        Returns:
            Path: Absolute path of the written file.

        Example:
            ::

                path = mgr.save_netcdf(ds, "gfs_t2m_48h.nc")
                # saves to <output_dir>/gfs_t2m_48h.nc
                path = mgr.save_netcdf(ds, "/data/gfs_t2m_48h.nc")
                # saves to the absolute path provided
        """
        path = Path(filename)
        if not path.is_absolute():
            path = self.output_dir / path

        encoding = {
            v: {"zlib": True, "complevel": complevel}
            for v in ds.data_vars
        }
        ds.to_netcdf(path, encoding=encoding)
        mb = path.stat().st_size / 1024 ** 2
        LOG.info("Saved NetCDF: %s  (%.1f MB)", path, mb)
        print(f"[save] NetCDF → {path}  ({mb:.1f} MB)")
        return path

    def save_zarr(self, ds: xr.Dataset, store: str) -> Path:
        """Save a Dataset as a chunked Zarr store (directory).

        Zarr is preferred over NetCDF for large time-series because it
        supports lazy chunked reads and appending new timesteps without
        rewriting existing data. The dataset is chunked along the ``time``
        dimension with a chunk size of 1 before writing.

        Args:
            ds (xr.Dataset): Dataset to persist.
            store (str): Output directory path. Relative paths are resolved
                against :attr:`output_dir`.

        Returns:
            Path: Absolute path of the Zarr store directory.

        Example:
            ::

                path = mgr.save_zarr(ds, "gfs_16days.zarr")
        """
        path = Path(store)
        if not path.is_absolute():
            path = self.output_dir / path

        ds.chunk({"time": 1}).to_zarr(path, mode="w")
        LOG.info("Saved Zarr: %s", path)
        print(f"[save] Zarr  → {path}")
        return path

    # ── reload ────────────────────────────────────────────────────────────────

    @staticmethod
    def load_netcdf(path: str | Path) -> xr.Dataset:
        """Lazily open a previously saved NetCDF file.

        Uses Dask-backed chunking (``chunks="auto"``) so that data is only
        read from disk when computations are triggered.

        Args:
            path (str | Path): Path to the ``.nc`` file.

        Returns:
            xr.Dataset: Lazily loaded dataset. Call ``.compute()`` or index
                into a variable to materialise values.

        Example:
            ::

                ds = GFSDatasetManager.load_netcdf("/data/gfs_t2m_48h.nc")
                print(dict(ds.dims))
                # {'time': 9, 'latitude': 721, 'longitude': 1440}
        """
        ds = xr.open_dataset(path, chunks="auto")
        print(f"[load] {path}  →  {dict(ds.dims)}")
        return ds

    @staticmethod
    def load_zarr(store: str | Path) -> xr.Dataset:
        """Lazily open a previously saved Zarr store.

        Args:
            store (str | Path): Path to the Zarr store directory.

        Returns:
            xr.Dataset: Lazily loaded dataset backed by the Zarr store.

        Example:
            ::

                ds = GFSDatasetManager.load_zarr("gfs_16days.zarr")
        """
        ds = xr.open_zarr(store)
        print(f"[load] {store}  →  {dict(ds.dims)}")
        return ds

    def __del__(self) -> None:
        """Close the underlying HTTP session when the manager is garbage-collected."""
        try:
            self._session.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# CLI demo
# ══════════════════════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s  %(levelname)-8s  %(message)s",
#         datefmt="%H:%M:%S",
#     )

#     DATE       = "20260403"
#     CYCLE      = "06"
#     OUTPUT_DIR = "./gfs_output"

#     REGION = {
#         "toplat":    5,
#         "bottomlat": -35,
#         "leftlon":   -75,
#         "rightlon":  -34,
#     }

#     mgr = GFSDatasetManager(
#         date=DATE, cycle=CYCLE, output_dir=OUTPUT_DIR, region=REGION
#     )

#     print(f"\nAvailable variables ({len(VARIABLES)}):")
#     print(f"  Surface / single-level  ({len(SURFACE_VARS)}): {SURFACE_VARS}")
#     print(f"  Multi-level             ({len(MULTILEVEL_VARS)}): {MULTILEVEL_VARS}")

#     # ── example: 5 variables, 4 hours → 4 downloads (not 20) ─────────────────
#     # print("\n[1] t2m + prate + prmsl + u10 + v10 — 0..24 h (6-hourly)")
#     # ds = mgr.build_multi_dataset(
#     #     var_keys=["t2m", "prate", "prmsl", "u10", "v10"],
#     #     hours=list(range(0, 25, 6)),
#     # )
#     # print(ds)
#     # mgr.save_netcdf(ds, "gfs_surface_24h.nc")

#     # ── example: 16-day t2m series ────────────────────────────────────────────
#     ds_16d = mgr.build_dataset("t2m", hours=HOURS_16DAYS)
#     mgr.save_netcdf(ds_16d, "gfs_t2m_16days.nc")