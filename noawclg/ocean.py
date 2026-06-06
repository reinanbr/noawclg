"""Ocean data access — GODAS subsurface fields and ENSO diagnostics.

Data sources
------------
**NOAA NCEP GODAS** — Monthly mean ocean fields on a ~1/3°×1° tripolar grid,
40 vertical levels (5 m – 4 478 m), 1980 – present, via OPeNDAP.

Variables: ``pottmp`` (K → °C), ``salt`` (kg/kg → PSU),
``ucur``/``vcur`` (m/s), ``sshg`` (m).

**NOAA ERSST v5** — Monthly SST, 2° grid, 1854 – present, single global file.

References
----------
https://www.psl.noaa.gov/data/gridded/data.godas.html

https://www.psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

LOG = logging.getLogger(__name__)

# ── URL templates ──────────────────────────────────────────────────────────────
_GODAS_BASE = "https://psl.noaa.gov/thredds/dodsC/Datasets/godas/{var}.{year}.nc"
_ERSST_URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.ersst.v5/sst.mnmean.nc"

# ── GODAS variable catalogue ───────────────────────────────────────────────────
#: Maps each GODAS variable name to its metadata
#: (long_name, units_in, units_out, has_levels, valid_min).
GODAS_VARS: dict[str, dict] = {
    "pottmp": {
        "long_name": "Potential temperature",
        "units_in": "K",
        "units_out": "°C",
        "has_levels": True,
        "valid_min": 200.0,  # any K < 200 is a fill value
    },
    "salt": {
        "long_name": "Salinity",
        "units_in": "kg/kg",
        "units_out": "PSU",
        "has_levels": True,
        "valid_min": 0.001,  # < 0.001 kg/kg treated as fill
    },
    "ucur": {
        "long_name": "U-component of ocean current (eastward)",
        "units_in": "m/s",
        "units_out": "m/s",
        "has_levels": True,
        "valid_min": None,
    },
    "vcur": {
        "long_name": "V-component of ocean current (northward)",
        "units_in": "m/s",
        "units_out": "m/s",
        "has_levels": True,
        "valid_min": None,
    },
    "sshg": {
        "long_name": "Sea Surface Height Relative to Geoid",
        "units_in": "m",
        "units_out": "m",
        "has_levels": False,
        "valid_min": None,
    },
}

#: Standard ENSO monitoring regions (longitude in 0–360 convention).
#: Keys: ``"1+2"``, ``"3"``, ``"3.4"``, ``"4"``.
NINO_BOXES: dict[str, dict] = {
    "1+2": {"lat": (-10.0, 0.0), "lon": (270.0, 280.0)},
    "3": {"lat": (-5.0, 5.0), "lon": (210.0, 270.0)},
    "3.4": {"lat": (-5.0, 5.0), "lon": (190.0, 240.0)},
    "4": {"lat": (-5.0, 5.0), "lon": (160.0, 210.0)},
}

# Warm water volume box: 5°S–5°N, 120°E–80°W (depth-integrated T > 20°C)
_WWV_BOX = {"lat": (-5.0, 5.0), "lon": (120.0, 280.0)}

# GODAS depth levels (m)
GODAS_LEVELS = np.array(
    [
        5,
        15,
        25,
        35,
        45,
        55,
        65,
        75,
        85,
        95,
        105,
        115,
        125,
        135,
        145,
        155,
        165,
        175,
        185,
        195,
        205,
        215,
        225,
        238,
        262,
        303,
        366,
        459,
        584,
        747,
        949,
        1193,
        1479,
        1807,
        2174,
        2579,
        3016,
        3483,
        3972,
        4478,
    ],
    dtype=float,
)

_K_TO_C = 273.15
_KGK_PSU = 1000.0  # kg/kg → PSU (g/kg)


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════


def _apply_region(ds: xr.Dataset, region: dict | None) -> xr.Dataset:
    if region is None:
        return ds
    return ds.sel(
        lat=slice(region["lat_min"], region["lat_max"]),
        lon=slice(region["lon_min"], region["lon_max"]),
    )


def _mask_and_convert(da: xr.DataArray, var: str) -> xr.DataArray:
    """Apply fill-value masking and unit conversion for a GODAS variable."""
    info = GODAS_VARS[var]
    vmin = info["valid_min"]
    fv = da.attrs.get("missing_value", da.attrs.get("_FillValue", None))

    # Mask explicit fill values
    if fv is not None:
        try:
            da = da.where(da != float(fv))
        except (TypeError, ValueError):
            pass

    # Mask physically impossible values
    if vmin is not None:
        da = da.where(da > vmin)

    # Large-magnitude current fill values (e.g. 9.96921e+36)
    if var in ("ucur", "vcur"):
        da = da.where(np.abs(da) < 100.0)

    # Unit conversions
    if var == "pottmp":
        da = da - _K_TO_C
    elif var == "salt":
        da = da * _KGK_PSU

    da.attrs.update(
        {
            "long_name": info["long_name"],
            "units": info["units_out"],
            "source": "NOAA NCEP GODAS",
        }
    )
    return da


def _concat_years(
    darrays: list[xr.DataArray],
) -> xr.DataArray:
    if len(darrays) == 1:
        return darrays[0]
    return xr.concat(darrays, dim="time")


# ══════════════════════════════════════════════════════════════════════════════
# Low-level GODAS access
# ══════════════════════════════════════════════════════════════════════════════


def open_godas(
    year: int,
    variable: str = "pottmp",
    depth_m: float | None = None,
    region: dict | None = None,
) -> xr.Dataset:
    """Open one year of GODAS data via OPeNDAP (lazy, no download).

    Parameters
    ----------
    year     : Four-digit year (1980–present).
    variable : GODAS variable name — one of ``pottmp``, ``salt``,
               ``ucur``, ``vcur``, ``sshg``.
    depth_m  : Nearest depth level (m).  ``None`` returns all 40 levels.
               Ignored for ``sshg`` (surface-only variable).
    region   : Bounding box dict with keys ``lat_min``, ``lat_max``,
               ``lon_min``, ``lon_max`` (longitude 0–360).

    Returns
    -------
    xr.Dataset
        Units converted to SI/human-readable (°C, PSU, m/s, m).
    """
    if variable not in GODAS_VARS:
        raise ValueError(
            f"Unknown GODAS variable '{variable}'. Choose from: {list(GODAS_VARS)}"
        )

    url = _GODAS_BASE.format(var=variable, year=year)
    LOG.info("Opening GODAS %s %d via OPeNDAP", variable, year)
    ds = xr.open_dataset(url, engine="netcdf4")

    info = GODAS_VARS[variable]
    if info["has_levels"] and depth_m is not None:
        ds = ds.sel(level=depth_m, method="nearest")  # separate from region sel

    ds = _apply_region(ds, region)

    # Convert and mask the data variable (same name as the file stem)
    ds[variable] = _mask_and_convert(ds[variable], variable)
    return ds


def get_godas(
    year_start: int,
    year_end: int | None = None,
    variable: str = "pottmp",
    depth_m: float | None = None,
    region: dict | None = None,
) -> xr.DataArray:
    """Multi-year GODAS field as a single DataArray.

    Parameters
    ----------
    year_start, year_end : Year range (inclusive). *year_end* defaults to
                           *year_start*.
    variable             : ``pottmp`` | ``salt`` | ``ucur`` | ``vcur`` |
                           ``sshg``.
    depth_m              : Target depth in metres (nearest level).
    region               : Optional lat/lon bounding box (0–360 longitude).

    Returns
    -------
    xr.DataArray  (time, [level,] lat, lon)
    """
    if year_end is None:
        year_end = year_start

    pieces: list[xr.DataArray] = []
    for yr in range(year_start, year_end + 1):
        try:
            ds = open_godas(yr, variable=variable, depth_m=depth_m, region=region)
            pieces.append(ds[variable])
        except Exception as exc:
            LOG.warning("Could not load GODAS %s %d: %s", variable, yr, exc)

    if not pieces:
        raise RuntimeError(
            f"No GODAS '{variable}' data loaded for {year_start}–{year_end}."
        )
    return _concat_years(pieces)


# ══════════════════════════════════════════════════════════════════════════════
# Typed convenience wrappers
# ══════════════════════════════════════════════════════════════════════════════


def get_ocean_temp(
    year_start: int,
    year_end: int | None = None,
    depth_m: float = 200.0,
    region: dict | None = None,
) -> xr.DataArray:
    """Ocean potential temperature (°C) at a given depth.

    Parameters
    ----------
    depth_m : Target depth in metres.  Default 200 m — a key ENSO indicator.

    Returns
    -------
    xr.DataArray  ``pottmp`` (°C), dims ``(time, lat, lon)``.
    """
    return get_godas(
        year_start, year_end, variable="pottmp", depth_m=depth_m, region=region
    )


def get_salinity(
    year_start: int,
    year_end: int | None = None,
    depth_m: float = 5.0,
    region: dict | None = None,
) -> xr.DataArray:
    """Ocean salinity (PSU) at a given depth.

    Parameters
    ----------
    depth_m : Target depth in metres.  Default 5 m (near-surface).

    Returns
    -------
    xr.DataArray  ``salt`` (PSU), dims ``(time, lat, lon)``.
    """
    return get_godas(
        year_start, year_end, variable="salt", depth_m=depth_m, region=region
    )


def get_currents(
    year_start: int,
    year_end: int | None = None,
    depth_m: float = 5.0,
    region: dict | None = None,
) -> xr.Dataset:
    """Ocean currents (m/s) at a given depth.

    Returns
    -------
    xr.Dataset with variables:
        ``ucur``  : eastward current component (m/s)
        ``vcur``  : northward current component (m/s)
        ``speed`` : current speed sqrt(u²+v²) (m/s)
    """
    u = get_godas(year_start, year_end, variable="ucur", depth_m=depth_m, region=region)
    v = get_godas(year_start, year_end, variable="vcur", depth_m=depth_m, region=region)
    speed = np.sqrt(u**2 + v**2)
    speed.attrs = {"long_name": "Ocean current speed", "units": "m/s"}

    return xr.Dataset({"ucur": u, "vcur": v, "speed": speed})


def get_ssh(
    year_start: int,
    year_end: int | None = None,
    region: dict | None = None,
) -> xr.DataArray:
    """Sea Surface Height relative to geoid (m).

    SSH anomaly is a key dynamical ENSO indicator: positive anomaly in the
    central/eastern Pacific during El Niño, negative during La Niña.

    Returns
    -------
    xr.DataArray  ``sshg`` (m), dims ``(time, lat, lon)``.
    """
    return get_godas(year_start, year_end, variable="sshg", region=region)


# ══════════════════════════════════════════════════════════════════════════════
# ERSST v5 — long-term SST record (1854–present)
# ══════════════════════════════════════════════════════════════════════════════


def open_ersst(
    year_start: int | None = None,
    year_end: int | None = None,
    region: dict | None = None,
) -> xr.DataArray:
    """Open NOAA ERSST v5 via OPeNDAP (lazy, single global file).

    ERSST extends the SST record back to 1854 — ideal for multi-decadal
    ENSO climatology that GODAS (1980–present) cannot provide.

    Parameters
    ----------
    year_start, year_end : Optional year slice.  Both default to full record.
    region               : Optional lat/lon bounding box.
                           Longitude convention: **−180 to +180**.

    Returns
    -------
    xr.DataArray  ``sst`` (°C), dims ``(time, lat, lon)``.
    """
    LOG.info("Opening ERSST v5 via OPeNDAP")
    ds = xr.open_dataset(_ERSST_URL, engine="netcdf4")

    sst = ds["sst"].squeeze()  # drop singleton 'lev' dim if present

    # Slice time
    if year_start is not None or year_end is not None:
        t0 = f"{year_start}-01" if year_start else None
        t1 = f"{year_end}-12" if year_end else None
        sst = sst.sel(time=slice(t0, t1))

    if region is not None:
        # ERSST lat may be decreasing (88 → -88): auto-orient the slice
        lat_dec = float(sst["lat"].values[0]) > float(sst["lat"].values[-1])
        lat_sel = (
            slice(region["lat_max"], region["lat_min"])
            if lat_dec
            else slice(region["lat_min"], region["lat_max"])
        )
        # ERSST lon is 0-360: use Niño-box values directly (no sign flip)
        sst = sst.sel(
            lat=lat_sel,
            lon=slice(region["lon_min"], region["lon_max"]),
        )

    # Mask fill values (ERSST fill = 9.96921e+36 or similar)
    fv = sst.attrs.get("missing_value", sst.attrs.get("_FillValue", None))
    if fv is not None:
        try:
            sst = sst.where(sst != float(fv))
        except (TypeError, ValueError):
            pass
    sst = sst.where(np.abs(sst) < 100)  # physical range guard

    sst.attrs.update(
        {
            "long_name": "Sea Surface Temperature",
            "units": "°C",
            "source": "NOAA ERSST v5",
        }
    )
    return sst


# ══════════════════════════════════════════════════════════════════════════════
# ENSO indices
# ══════════════════════════════════════════════════════════════════════════════


def get_sst_series(
    year_start: int,
    year_end: int | None = None,
    box: Literal["1+2", "3", "3.4", "4"] = "3.4",
    source: Literal["godas", "ersst"] = "godas",
) -> pd.Series:
    """Monthly mean SST (°C) averaged over a standard Niño box.

    Parameters
    ----------
    box    : Niño region key — ``"3.4"`` (default), ``"3"``, ``"4"``,
             or ``"1+2"``.
    source : ``"godas"`` (1980–present, 5 m level) or ``"ersst"``
             (1854–present).

    Returns
    -------
    pd.Series  (monthly DatetimeIndex, values in °C)
    """
    if year_end is None:
        year_end = year_start

    b = NINO_BOXES[box]

    if source == "ersst":
        # ERSST uses 0-360 longitude, same convention as NINO_BOXES
        region = {
            "lat_min": b["lat"][0],
            "lat_max": b["lat"][1],
            "lon_min": b["lon"][0],
            "lon_max": b["lon"][1],
        }
        da = open_ersst(year_start, year_end, region=region)
    else:
        region = {
            "lat_min": b["lat"][0],
            "lat_max": b["lat"][1],
            "lon_min": b["lon"][0],
            "lon_max": b["lon"][1],
        }
        da = get_ocean_temp(year_start, year_end, depth_m=5.0, region=region)

    sst_mean = da.mean(["lat", "lon"])
    return pd.Series(
        sst_mean.values,
        index=pd.DatetimeIndex(sst_mean["time"].values),
        name=f"SST_Nino{box}",
    )


def get_nino_anomaly(
    year_start: int,
    year_end: int | None = None,
    box: Literal["1+2", "3", "3.4", "4"] = "3.4",
    clim_start: int = 1991,
    clim_end: int = 2020,
    source: Literal["godas", "ersst"] = "godas",
) -> pd.Series:
    """Monthly SST anomaly (°C) relative to a reference climatology.

    Parameters
    ----------
    clim_start, clim_end : Climatology base period (default 1991–2020,
                           current WMO standard).
    source               : ``"godas"`` or ``"ersst"``.

    Returns
    -------
    pd.Series  anomaly in °C
    """
    if year_end is None:
        year_end = year_start

    all_start = min(year_start, clim_start)
    all_end = max(year_end, clim_end)

    sst = get_sst_series(all_start, all_end, box=box, source=source)

    clim_mask = (sst.index.year >= clim_start) & (sst.index.year <= clim_end)
    monthly_clim = sst[clim_mask].groupby(sst[clim_mask].index.month).mean()

    anomaly = sst.copy()
    for month, mean_val in monthly_clim.items():
        anomaly[anomaly.index.month == month] -= mean_val

    target = (anomaly.index.year >= year_start) & (anomaly.index.year <= year_end)
    result = anomaly[target]
    result.name = f"Nino{box}_anom"
    return result


def get_oni(
    year_start: int,
    year_end: int | None = None,
    clim_start: int = 1991,
    clim_end: int = 2020,
    source: Literal["godas", "ersst"] = "godas",
) -> pd.Series:
    """Oceanic Niño Index (ONI) — 3-month running mean of Niño 3.4 anomaly.

    Returns
    -------
    pd.Series  ONI (°C), indexed by the centre month of each season.
    """
    anom = get_nino_anomaly(
        year_start,
        year_end,
        box="3.4",
        clim_start=clim_start,
        clim_end=clim_end,
        source=source,
    )
    oni = anom.rolling(window=3, center=True, min_periods=3).mean()
    oni.name = "ONI"
    return oni


def classify_enso(
    oni: pd.Series,
    threshold: float = 0.5,
    min_consecutive: int = 5,
) -> pd.Series:
    """Classify each month as ``'El Niño'``, ``'La Niña'``, or ``'Neutral'``.

    Follows the NOAA CPC ONI rule: the anomaly must exceed *threshold* for
    at least *min_consecutive* consecutive overlapping 3-month seasons.

    Returns
    -------
    pd.Series of str  (same index as *oni*)
    """
    raw = pd.Series("Neutral", index=oni.index, name="ENSO_phase")

    for phase, sign in [("El Niño", 1), ("La Niña", -1)]:
        condition = (sign * oni >= threshold).astype(int)
        group_id = (condition != condition.shift()).cumsum()
        run_len = condition.groupby(group_id).transform("sum") * condition
        raw[run_len >= min_consecutive] = phase

    return raw


# ══════════════════════════════════════════════════════════════════════════════
# Thermocline depth — depth of 20 °C isotherm (D20)
# ══════════════════════════════════════════════════════════════════════════════


def get_thermocline_depth(
    year_start: int,
    year_end: int | None = None,
    region: dict | None = None,
    isotherm_temp: float = 20.0,
) -> xr.DataArray:
    """Depth (m) of the *isotherm_temp* °C isotherm — the D20 index.

    The 20 °C isotherm depth is the primary dynamical ENSO indicator:
    - Deeper in the east Pacific → El Niño (warm water piled east)
    - Shallower in the east Pacific → La Niña (thermocline shoals east)

    Returns
    -------
    xr.DataArray  ``d20`` (m), dims ``(time, lat, lon)``.
    """
    if year_end is None:
        year_end = year_start

    chunks: list[xr.DataArray] = []
    for yr in range(year_start, year_end + 1):
        try:
            ds = open_godas(yr, variable="pottmp", region=region)
            t = ds["pottmp"]  # (time, level, lat, lon) in °C

            levels = t["level"].values.astype(float)
            above = (t > isotherm_temp).astype(float)  # 1 where T > threshold

            # Index of the deepest level that is still above the isotherm
            idx_vals = np.clip(
                (above.sum("level").values - 1).astype(int), 0, len(levels) - 1
            )
            depth_arr = levels[idx_vals]  # shape (time, lat, lon)

            d20 = xr.DataArray(
                depth_arr,
                dims=("time", "lat", "lon"),
                coords={
                    "time": t.coords["time"],
                    "lat": t.coords["lat"],
                    "lon": t.coords["lon"],
                },
            )
            chunks.append(d20)
        except Exception as exc:
            LOG.warning("Could not compute D20 for %d: %s", yr, exc)

    if not chunks:
        raise RuntimeError(f"No D20 computed for {year_start}–{year_end}.")

    combined = xr.concat(chunks, dim="time")
    combined.attrs = {
        "long_name": f"Depth of {isotherm_temp} °C isotherm (D20)",
        "units": "m",
        "source": "NOAA NCEP GODAS",
    }
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# Warm Water Volume (WWV)
# ══════════════════════════════════════════════════════════════════════════════


def get_warm_water_volume(
    year_start: int,
    year_end: int | None = None,
    temp_threshold: float = 20.0,
    max_depth: float = 300.0,
) -> pd.Series:
    """Monthly Warm Water Volume index (m³ × 10¹⁴) in the equatorial Pacific.

    WWV is defined as the volume of water warmer than *temp_threshold* °C
    above *max_depth* in the 5°S–5°N, 120°E–80°W box.  It is a leading
    indicator of ENSO: large WWV → El Niño likely; small WWV → La Niña.

    Returns
    -------
    pd.Series  WWV (×10¹⁴ m³), monthly
    """
    if year_end is None:
        year_end = year_start

    reg = {
        "lat_min": _WWV_BOX["lat"][0],
        "lat_max": _WWV_BOX["lat"][1],
        "lon_min": _WWV_BOX["lon"][0],
        "lon_max": _WWV_BOX["lon"][1],
    }
    records: list[tuple[pd.Timestamp, float]] = []
    for yr in range(year_start, year_end + 1):
        try:
            ds = open_godas(yr, variable="pottmp", region=reg)
            t = ds["pottmp"].sel(level=slice(None, max_depth))
            # Boolean mask: warm water
            warm = (t > temp_threshold).astype(float)
            # Approximate grid cell volume (depth thickness × lat-lon area)
            lev = t["level"].values.astype(float)
            dlev = np.gradient(lev)  # thickness of each level (m)
            # Lat spacing ~1/3° near equator ≈ 37 km; lon spacing 1° ≈ 111 km
            dlat = np.abs(np.gradient(t["lat"].values)) * 111_000.0  # m
            dlon = (
                np.abs(np.gradient(t["lon"].values))
                * 111_000.0
                * np.cos(np.deg2rad(t["lat"].values))
            )

            dlon_da = xr.DataArray(dlon, dims=["lat"])
            dlat_da = xr.DataArray(dlat, dims=["lat"])
            dlev_da = xr.DataArray(dlev, dims=["level"])

            # cell volume: dz × dy × dx  (m³)
            cell_vol = dlev_da * dlat_da * dlon_da
            wwv_ts = (warm * cell_vol).sum(["level", "lat", "lon"])

            for i, tval in enumerate(wwv_ts["time"].values):
                records.append((pd.Timestamp(tval), float(wwv_ts.values[i]) / 1e14))
        except Exception as exc:
            LOG.warning("Could not compute WWV for %d: %s", yr, exc)

    if not records:
        raise RuntimeError(f"No WWV computed for {year_start}–{year_end}.")

    idx, vals = zip(*sorted(records))
    return pd.Series(vals, index=pd.DatetimeIndex(idx), name="WWV_1e14m3")


# ══════════════════════════════════════════════════════════════════════════════
# Convenience: full ENSO monitoring summary
# ══════════════════════════════════════════════════════════════════════════════


def enso_summary(
    year_start: int,
    year_end: int | None = None,
    clim_start: int = 1991,
    clim_end: int = 2020,
    source: Literal["godas", "ersst"] = "godas",
) -> pd.DataFrame:
    """DataFrame of monthly ENSO diagnostics.

    Columns
    -------
    sst_nino34  : Niño 3.4 mean SST (°C)
    anom_nino34 : Niño 3.4 SST anomaly (°C)
    oni         : Oceanic Niño Index — 3-month running mean (°C)
    phase       : ``'El Niño'`` | ``'La Niña'`` | ``'Neutral'``

    Example
    -------
    >>> from noawclg.ocean import enso_summary
    >>> df = enso_summary(2020, 2024)
    >>> print(df[df.phase != "Neutral"])
    """
    if year_end is None:
        year_end = year_start

    sst = get_sst_series(year_start, year_end, box="3.4", source=source)
    anom = get_nino_anomaly(
        year_start,
        year_end,
        box="3.4",
        clim_start=clim_start,
        clim_end=clim_end,
        source=source,
    )
    oni = get_oni(
        year_start, year_end, clim_start=clim_start, clim_end=clim_end, source=source
    )
    phase = classify_enso(oni)

    df = pd.DataFrame(
        {
            "sst_nino34": sst,
            "anom_nino34": anom,
            "oni": oni,
            "phase": phase,
        }
    )
    df.index.name = "month"
    return df
