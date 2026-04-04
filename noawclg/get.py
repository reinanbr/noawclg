"""Utilities to collect 16-day GFS data.

Based on the schedule logic from help2.py.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import logging
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

LOG = logging.getLogger(__name__)

# FIX 1: NOMADS blocks the default 'python-requests' User-Agent with 403.
# A descriptive browser-like agent is required — same pattern as gfs_plots.py.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; GFS-downloader/1.0; "
        "+https://github.com/reinanbr/noawclg)"
    )
}

# FIX 2: Retry strategy — NOMADS connections fail intermittently.
# Retries on 429 (rate-limit), 500, 502, 503, 504 with exponential back-off.
_RETRY = Retry(
    total=4,
    backoff_factor=2,  # waits 2s, 4s, 8s, 16s between attempts
    status_forcelist={429, 500, 502, 503, 504},
    allowed_methods={"GET", "HEAD"},
    raise_on_status=False,
)

# Pause between requests to avoid hammering the NOMADS server (seconds).
# Mirrors PAUSA_DOWNLOAD from gfs_plots.py.
_PAUSA = 1.5


def _build_session() -> requests.Session:
    """Return a Session with headers and retry logic pre-configured."""
    session = requests.Session()
    session.headers.update(_HEADERS)
    adapter = HTTPAdapter(max_retries=_RETRY)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _full_16_day_hours() -> list[int]:
    """Return all GFS forecast hours for 16 days.

    Operational GFS schedule:
    - f000 to f120: 1-hour step
    - f123 to f384: 3-hour step
    """
    return list(range(0, 121)) + list(range(123, 385, 3))


def get_all_data_16_days(
    base_url: str,
    date: str,
    cycle: str = "00",
    timeout: int = 30,
    save_to: str | None = None,
    pause: float = _PAUSA,
) -> dict[str, Any]:
    """Fetch (and optionally download) all available 16-day data slots.

    Args:
        base_url: URL template with placeholders ``{date}``, ``{cycle}``,
            and ``{hour}``.
            Example (direct NOMADS file access — recommended):
            https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/
            gfs.{date}/{cycle}/atmos/gfs.t{cycle}z.pgrb2.0p25.f{hour:03d}
        date: Model run date in YYYYMMDD format.
        cycle: Model run cycle ("00", "06", "12", "18").
        timeout: Request timeout in seconds.
        save_to: Optional output directory to store downloaded files.
        pause: Seconds to wait between requests (default 1.5 s).

    Returns:
        Dictionary containing available and missing slots.
    """
    datetime.strptime(date, "%Y%m%d")

    if cycle not in {"00", "06", "12", "18"}:
        raise ValueError("cycle must be one of: 00, 06, 12, 18")

    hours = _full_16_day_hours()
    out_dir = Path(save_to) if save_to else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    available: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    session = _build_session()

    try:
        total_hours = len(hours)
        it_hour = 0
        for hour in tqdm(hours, desc="Fetching GFS data", unit="hour"):
            url = base_url.format(date=date, cycle=cycle, hour=hour)
            LOG.info("Fetching hour %03d from %s", hour, url)

            try:
                if out_dir:
                    # --- DOWNLOAD MODE ---
                    resp = session.get(url, timeout=timeout, stream=True)

                    if resp.status_code != 200:
                        missing.append(
                            {"hour": hour, "url": url, "status_code": resp.status_code}
                        )
                        LOG.warning("Missing hour %d: HTTP %d", hour, resp.status_code)
                        continue

                    filename = f"gfs_{date}_{cycle}z_f{hour:03d}.grib2"
                    path = out_dir / filename
                    bytes_written = 0

                    with path.open("wb") as fh:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                fh.write(chunk)
                                bytes_written += len(chunk)

                    # FIX 3: validate the file is not empty / an HTML error page.
                    # gfs_plots.py uses len(r.content) < 100 for the same reason.
                    if bytes_written < 100:
                        LOG.warning(
                            "Hour %d: file too small (%d bytes) — likely an "
                            "error response; discarding.",
                            hour,
                            bytes_written,
                        )
                        path.unlink(missing_ok=True)
                        missing.append(
                            {
                                "hour": hour,
                                "url": url,
                                "error": f"empty response ({bytes_written} bytes)",
                            }
                        )
                        continue

                    item: dict[str, Any] = {
                        "hour": hour,
                        "url": url,
                        "status_code": resp.status_code,
                        # FIX 4: Content-Length is unreliable with stream=True;
                        # use the actual bytes written instead.
                        "content_length": bytes_written,
                        "file": str(path),
                    }
                    LOG.info("  [ok] f%03d  %.0f KB", hour, bytes_written / 1024)

                else:
                    # --- CHECK-ONLY MODE (no download) ---
                    # FIX 5: use HEAD to avoid downloading the full file into
                    # memory just to check availability (was GET with stream=False).
                    resp = session.head(url, timeout=timeout)

                    if resp.status_code not in {200, 302}:
                        missing.append(
                            {"hour": hour, "url": url, "status_code": resp.status_code}
                        )
                        LOG.warning("Missing hour %d: HTTP %d", hour, resp.status_code)
                        continue

                    item = {
                        "hour": hour,
                        "url": url,
                        "status_code": resp.status_code,
                        "content_length": int(resp.headers.get("Content-Length", 0)),
                    }

                available.append(item)
                it_hour += 1
                percent_complete = (it_hour / total_hours) * 100
                LOG.info(
                    "Progress: %d/%d hours (%.1f%%)",
                    it_hour,
                    total_hours,
                    percent_complete,
                )

            except requests.RequestException as exc:
                missing.append({"hour": hour, "url": url, "error": str(exc)})
                LOG.error("Error fetching hour %d: %s", hour, exc)

            time.sleep(pause)

    finally:
        session.close()

    return {
        "date": date,
        "cycle": cycle,
        "total_slots": len(hours),
        "available": available,
        "missing": missing,
        "available_count": len(available),
        "missing_count": len(missing),
    }


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)

#     base = (
#         "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"
#         "gfs.{date}/{cycle}/atmos/gfs.t{cycle}z.pgrb2.0p25.f{hour:03d}"
#     )

#     # Check-only (no download):
#     result = get_all_data_16_days(base_url=base, date="20260403", cycle="06")

#     print(
#         f"Available: {result['available_count']} | "
#         f"Missing:   {result['missing_count']}"
#     )

# To download files, add:  save_to="./gfs_output"
