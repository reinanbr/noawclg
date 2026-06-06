"""GFS Dataset Manager — download, parse, and assemble xarray Datasets.

Downloads one GRIB2 file per forecast hour via the NOMADS grib-filter endpoint
(all requested variables bundled in a single URL), then extracts each variable
with cfgrib and assembles a time-labelled xr.Dataset.

Sub-modules that contain extracted responsibilities:
    catalog     — VARIABLES catalogue and hour-sequence constants
    http        — HTTP session builder and URL template
    coords      — Coordinate helpers (BoundingBox, _find_dim, …)
    persistence — NetCDF / Zarr save and load helpers
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

# ── sub-module imports ────────────────────────────────────────────────────────
from noawclg.catalog import (  # re-exported for backward compat
    HOURS_10DAYS_3H,
    HOURS_16DAYS,
    HOURS_16DAYS_3H,
    HOURS_5DAYS_1H,
    MULTILEVEL_VARS,
    SURFACE_VARS,
    VARIABLES,
)
from noawclg.http import FILTER_BASE, _build_session  # re-exported for backward compat
from noawclg import persistence as _persistence

try:
    from tqdm import tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

LOG = logging.getLogger(__name__)
logging.getLogger("cfgrib.messages").setLevel(logging.ERROR)

# ── convenient subsets (re-exported) ─────────────────────────────────────────
__all__ = [
    "GFSDatasetManager",
    "VARIABLES",
    "SURFACE_VARS",
    "MULTILEVEL_VARS",
    "HOURS_16DAYS",
    "HOURS_5DAYS_1H",
    "HOURS_10DAYS_3H",
    "HOURS_16DAYS_3H",
    "_build_session",
]


# ══════════════════════════════════════════════════════════════════════════════
# Manager
# ══════════════════════════════════════════════════════════════════════════════


class GFSDatasetManager:
    """Download GFS GRIB2 files (once per hour) and assemble xarray Datasets.

    The manager targets the NOAA NOMADS grib-filter endpoint and bundles every
    requested variable into a single URL per forecast hour, reducing HTTP
    traffic from ``N_vars × N_hours`` to just ``N_hours``.

    Downloaded GRIB2 files are cached under ``output_dir`` with deterministic
    filenames; subsequent calls with the same parameters skip the network.
    """

    def __init__(
        self,
        date: str,
        cycle: str = "00",
        output_dir: str = "./gfs_output",
        region: dict[str, float] | None = None,
        pause: float = 1.5,
        request_timeout: int = 30,
    ) -> None:
        datetime.strptime(date, "%Y%m%d")
        if cycle not in {"00", "06", "12", "18"}:
            raise ValueError("cycle must be one of: 00, 06, 12, 18")

        self.date = date
        self.cycle = cycle
        self.output_dir = Path(output_dir).resolve()
        self.region = region
        self.pause = pause
        self.request_timeout = request_timeout
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._run_dt = datetime.strptime(f"{date}{cycle}", "%Y%m%d%H")
        self._session = _build_session()

    # ── private helpers ───────────────────────────────────────────────────────

    def _region_params(self) -> str:
        if not self.region:
            return ""
        r = self.region
        return (
            f"&subregion=&toplat={r['toplat']}&bottomlat={r['bottomlat']}"
            f"&leftlon={r['leftlon']}&rightlon={r['rightlon']}"
        )

    def _var_params(self, var_keys: list[str]) -> str:
        seen: set[str] = set()
        parts: list[str] = []
        for vk in var_keys:
            cfg = VARIABLES[vk]
            var_token = f"&{cfg['grib_var']}=on"
            if var_token not in seen:
                seen.add(var_token)
                parts.append(var_token)

            if cfg["tlev"] == "isobaricInhPa" and cfg.get("levels"):
                for lev in cfg["levels"]:
                    lev_token = f"&lev_{lev}_mb=on"
                    if lev_token not in seen:
                        seen.add(lev_token)
                        parts.append(lev_token)
            else:
                lev_token = f"&{cfg['grib_lev']}=on"
                if lev_token not in seen:
                    seen.add(lev_token)
                    parts.append(lev_token)

        return "".join(parts)

    def _filter_url(self, var_keys: list[str], hour: int) -> str:
        return FILTER_BASE.format(
            date=self.date,
            cycle=self.cycle,
            hour=hour,
            var_params=self._var_params(var_keys),
            region_params=self._region_params(),
        )

    def _cache_path(self, var_keys: list[str], hour: int) -> Path:
        tag = "global"
        if self.region:
            r = self.region
            tag = (
                f"{r['toplat']}N{abs(r['bottomlat'])}S"
                f"{abs(r['leftlon'])}W{r['rightlon']}E"
            )
        vkey = "_".join(sorted(var_keys))[:60]
        return self.output_dir / (
            f"gfs_{self.date}_{self.cycle}z_{vkey}_{tag}_f{hour:03d}.grib2"
        )

    def _is_valid_grib_file(self, path: Path, min_size: int = 100) -> bool:
        try:
            if not path.exists() or path.stat().st_size < min_size:
                return False
            with path.open("rb") as fh:
                head = fh.read(4)
                if head != b"GRIB":
                    return True
                fh.seek(-4, 2)
                tail = fh.read(4)
            return tail == b"7777"
        except Exception:
            return False

    # ── download ──────────────────────────────────────────────────────────────

    def download_hours(
        self,
        var_keys: list[str],
        hours: list[int],
        force: bool = False,
    ) -> dict[int, Path]:
        """Download one GRIB2 file per forecast hour containing all variables."""
        unknown = [vk for vk in var_keys if vk not in VARIABLES]
        if unknown:
            raise KeyError(f"Unknown variables: {unknown}")

        results: dict[int, Path] = {}
        hours_to_fetch: list[int] = []

        for hour in hours:
            path = self._cache_path(var_keys, hour)
            if path.exists() and not force:
                if self._is_valid_grib_file(path):
                    LOG.info("[cache] f%03d  %s", hour, path.name)
                    results[hour] = path
                else:
                    LOG.warning(
                        "[cache-bad] f%03d  %s — re-downloading", hour, path.name
                    )
                    path.unlink(missing_ok=True)
                    hours_to_fetch.append(hour)
            else:
                hours_to_fetch.append(hour)

        if not hours_to_fetch:
            LOG.info("All files already cached.")
            return results

        LOG.info(
            "Download: %d hour(s) — vars: %s",
            len(hours_to_fetch),
            var_keys,
        )

        iterator = (
            tqdm(hours_to_fetch, desc="GFS download", unit="h")
            if _HAS_TQDM
            else hours_to_fetch
        )
        total = len(hours_to_fetch)
        counter = 0
        t_start = time.time()

        for hour in iterator:
            t_iter = time.time()
            path = self._cache_path(var_keys, hour)
            if path.exists() and not force:
                if self._is_valid_grib_file(path):
                    LOG.info("[cache] f%03d  %s", hour, path.name)
                    results[hour] = path
                    continue
                LOG.warning("[cache-bad] f%03d  %s — re-downloading", hour, path.name)
                path.unlink(missing_ok=True)

            url = self._filter_url(var_keys, hour)
            var_label = var_keys[0] if len(var_keys) == 1 else "multi"
            LOG.info("[%s] → f%03d  %s", var_label, hour, url[:120])

            try:
                resp = self._session.get(url, timeout=self.request_timeout, stream=True)

                if resp.status_code != 200:
                    LOG.warning(
                        "var=%s  f%03d: HTTP %d — skipping",
                        var_label,
                        hour,
                        resp.status_code,
                    )
                    time.sleep(self.pause)
                    continue

                bytes_written = 0
                with path.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
                            bytes_written += len(chunk)

                if bytes_written < 100 or not self._is_valid_grib_file(path):
                    LOG.warning(
                        "var=%s  f%03d: invalid response (%d bytes) — discarding",
                        var_label,
                        hour,
                        bytes_written,
                    )
                    path.unlink(missing_ok=True)
                else:
                    results[hour] = path
                    counter += 1
                    elapsed = time.time() - t_start
                    remaining = (time.time() - t_iter) * (total - counter)
                    LOG.info(
                        "  [ok] f%03d  %.0f KB  |  %.1f%%  (%d/%d)"
                        "  elapsed=%.1fs  remaining≈%.1fs",
                        hour,
                        bytes_written / 1024,
                        (counter / total) * 100,
                        counter,
                        total,
                        elapsed,
                        remaining,
                    )

            except requests.RequestException as exc:
                LOG.error("var=%s  f%03d: network error — %s", var_label, hour, exc)
            except KeyboardInterrupt:
                LOG.warning(
                    "Download interrupted at f%03d; %d file(s) collected.",
                    hour,
                    len(results),
                )
                break

            time.sleep(self.pause)

        return results

    # ── cfgrib extraction ─────────────────────────────────────────────────────

    def _open_var(self, path: Path, var_key: str) -> xr.Dataset | None:
        """Open a GRIB2 file and isolate the dataset for one variable."""
        cfg = VARIABLES[var_key]
        is_ml = bool(cfg.get("multilevel"))

        def _non_empty(ds: xr.Dataset) -> xr.Dataset | None:
            return ds if len(ds.data_vars) > 0 else None

        base_tlev = {"typeOfLevel": cfg["tlev"]}
        f_short = {**base_tlev, "shortName": cfg["short"]}

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
                ds = cfgrib.open_dataset(
                    path,
                    filter_by_keys=filters,
                    indexpath=None,
                    errors="ignore",
                )
                result = _non_empty(ds)
                if result is not None:
                    LOG.debug(
                        "'%s' found with filters %s → vars: %s",
                        var_key,
                        filters,
                        list(ds.data_vars),
                    )
                    return result
            except Exception as exc:
                last_exc = exc

        # Last resort: full scan
        try:
            all_ds = cfgrib.open_datasets(path, indexpath=None, errors="ignore")
            LOG.debug("'%s': full scan — %d sub-datasets", var_key, len(all_ds))

            for ds in all_ds:  # pass 1: shortName match
                if not ds.data_vars:
                    continue
                if cfg["short"] in ds.data_vars:
                    return ds

            for ds in all_ds:  # pass 2: typeOfLevel match
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
                    return ds

        except Exception as exc:
            last_exc = exc

        LOG.warning("Could not read '%s' from %s: %s", var_key, path.name, last_exc)
        return None

    def _extract(
        self, ds: xr.Dataset, var_key: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Extract and normalise (lats, lons, data) from an open xr.Dataset."""
        if not ds.data_vars:
            LOG.warning("'%s': dataset has no variables — skipping", var_key)
            return None

        cfg = VARIABLES[var_key]
        is_ml = bool(cfg.get("multilevel"))
        name = list(ds.data_vars)[0]
        da = ds[name]

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

        if lats_raw.ndim == 2:
            lats_raw = lats_raw[:, 0]
        if lons_raw.ndim == 2:
            lons_raw = lons_raw[0, :]

        da_vals = da.values

        if lons_raw.ndim == 1 and lons_raw.max() > 180:
            lons_norm = np.where(lons_raw > 180, lons_raw - 360, lons_raw)
            sort_idx = np.argsort(lons_norm)
            lons_raw = lons_norm[sort_idx]
            da_vals = da_vals[..., sort_idx]

        keep_ndim = 3 if is_ml else 2
        while da_vals.ndim > keep_ndim and da_vals.shape[0] == 1:
            da_vals = da_vals[0]

        if da_vals.ndim < 2:
            LOG.warning(
                "'%s': unexpected shape %s after dim reduction — skipping",
                var_key,
                da.shape,
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
        """Extract one variable from every cached file and stack along time."""
        cfg = VARIABLES[var_key]
        is_ml = bool(cfg.get("multilevel"))
        slices: list[np.ndarray] = []
        times: list[datetime] = []
        fhours: list[int] = []
        lats_ref = None
        lons_ref = None
        levels_ref = None

        for hour in sorted(files):
            path = files[hour]
            ds = self._open_var(path, var_key)
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
                _lev_names = [
                    "isobaricInhPa",
                    "depthBelowLandLayer",
                    "heightAboveGround",
                    "level",
                ]
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

        stacked = np.stack(slices, axis=0)

        coords: dict[str, Any] = {
            "time": ("time", times),
            "forecast_hour": ("time", fhours),
            "latitude": lats_ref,
            "longitude": lons_ref,
        }

        if is_ml and stacked.ndim == 4:
            dims = ["time", "level", "latitude", "longitude"]
            coords["level"] = (
                levels_ref if levels_ref is not None else list(range(stacked.shape[1]))
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
                        "units": cfg["units"],
                        "gfs_run": f"{self.date} {self.cycle}Z",
                    },
                )
            },
            coords=coords,
            attrs={
                "title": f"GFS 0.25° — {cfg['long_name']}",
                "institution": "NCEP/NOAA",
                "source": "GFS model output (NOMADS)",
                "run_date": self.date,
                "run_cycle": self.cycle,
                "created": datetime.utcnow().isoformat() + "Z",
            },
        )

    # ── public API ────────────────────────────────────────────────────────────

    def build_dataset(
        self,
        var_key: str,
        hours: list[int],
        force_download: bool = False,
    ) -> xr.Dataset:
        """Download and assemble a Dataset for a single variable."""
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
        """Download one file per hour for all variables and build a merged Dataset."""
        files = self.download_hours(var_keys, hours, force=force_download)
        if not files:
            raise RuntimeError(
                f"No files available for hours={hours}. "
                "NOMADS may not have published this run yet. "
                f"(date={self.date}, cycle={self.cycle})"
            )
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
        """Save a Dataset to a zlib-compressed NetCDF4 file."""
        return _persistence.save_netcdf(ds, filename, self.output_dir, complevel)

    def save_zarr(self, ds: xr.Dataset, store: str) -> Path:
        """Save a Dataset as a chunked Zarr store (directory)."""
        return _persistence.save_zarr(ds, store, self.output_dir)

    @staticmethod
    def load_netcdf(path: str | Path) -> xr.Dataset:
        """Lazily open a previously saved NetCDF file."""
        return _persistence.load_netcdf(path)

    @staticmethod
    def load_zarr(store: str | Path) -> xr.Dataset:
        """Lazily open a previously saved Zarr store."""
        return _persistence.load_zarr(store)

    # ── cleanup ───────────────────────────────────────────────────────────────

    def __del__(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass
