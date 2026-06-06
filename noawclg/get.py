"""Utilities to check or download all 16-day GFS data slots."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from noawclg.http import _build_session

LOG = logging.getLogger(__name__)

_PAUSA = 1.5  # seconds between requests


def _full_16_day_hours() -> list[int]:
    """Return all GFS forecast hours for 16 days (1-h steps then 3-h steps)."""
    return list(range(0, 121)) + list(range(123, 385, 3))


def get_all_data_16_days(
    base_url: str,
    date: str,
    cycle: str = "00",
    timeout: int = 30,
    save_to: str | None = None,
    pause: float = _PAUSA,
) -> dict[str, Any]:
    """Fetch (and optionally download) all available 16-day GFS data slots.

    Args:
        base_url: URL template with ``{date}``, ``{cycle}``, and ``{hour}``
            placeholders pointing to a NOMADS file.
        date: Model run date in ``YYYYMMDD`` format.
        cycle: Model run cycle (``'00'``, ``'06'``, ``'12'``, ``'18'``).
        timeout: Request timeout in seconds.
        save_to: Optional directory to store downloaded GRIB2 files.
            When omitted, only availability is checked (HEAD requests).
        pause: Seconds to sleep between requests.

    Returns:
        Dict with keys ``available``, ``missing``, ``available_count``,
        ``missing_count``, ``total_slots``, ``date``, ``cycle``.
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

                    if bytes_written < 100:
                        LOG.warning(
                            "Hour %d: file too small (%d bytes) — discarding.",
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
                        "content_length": bytes_written,
                        "file": str(path),
                    }
                    LOG.info("  [ok] f%03d  %.0f KB", hour, bytes_written / 1024)

                else:
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
                LOG.info(
                    "Progress: %d/%d hours (%.1f%%)",
                    it_hour,
                    total_hours,
                    (it_hour / total_hours) * 100,
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
