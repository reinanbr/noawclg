"""Shared HTTP infrastructure for NOMADS downloads."""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; GFS-downloader/2.0; "
        "+https://github.com/reinanbr/noawclg)"
    )
}

_RETRY = Retry(
    total=2,
    backoff_factor=0.5,
    status_forcelist={429, 500, 502, 503, 504},
    allowed_methods={"GET", "HEAD"},
    raise_on_status=False,
)

# NOMADS grib-filter URL template — accepts multiple &var_XXX=on&lev_XXX=on
FILTER_BASE = (
    "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl"
    "?dir=/gfs.{date}/{cycle}/atmos"
    "&file=gfs.t{cycle}z.pgrb2.0p25.f{hour:03d}"
    "{var_params}"
    "{region_params}"
)


def _build_session() -> requests.Session:
    """Create a Session pre-configured for NOMADS.

    Applies a browser-like User-Agent header (required — NOMADS blocks the
    default ``python-requests`` agent with HTTP 403) and an exponential-backoff
    retry policy on transient server errors.
    """
    s = requests.Session()
    s.headers.update(_HEADERS)
    adapter = HTTPAdapter(max_retries=_RETRY)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s
