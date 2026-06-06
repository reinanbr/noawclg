# noawclg

> **Download, analyse and visualise NOAA atmospheric and ocean data in Python.**

![PyPI Downloads](https://img.shields.io/pypi/dm/noawclg)
[![PyPI](https://img.shields.io/pypi/v/noawclg)](https://pypi.org/project/noawclg/)
[![Python](https://img.shields.io/pypi/pyversions/noawclg)](https://pypi.org/project/noawclg/)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![CI](https://github.com/reinanbr/noawclg/actions/workflows/ci.yml/badge.svg)](https://github.com/reinanbr/noawclg/actions/workflows/ci.yml)

`noawclg` gives you a clean Python API over two major NOAA data streams —
**GFS weather forecasts** and **GODAS/ERSST ocean analyses** — returning
`xarray.Dataset` objects ready for analysis and plotting, with no API key required.

---

## Table of contents

- [Features](#features)
- [Installation](#installation)
- [GFS weather forecasts](#gfs-weather-forecasts)
- [Ocean data — GODAS & ERSST](#ocean-data--godas--ersst)
- [ENSO diagnostics](#enso-diagnostics)
- [Plotting](#plotting)
- [API reference](#api-reference)
- [Module overview](#module-overview)
- [Contributing](#contributing)

---

## Features

| | |
|--|--|
| **GFS forecasts** | 3-hourly, 0–384 h, surface + multi-level, GRIB2 via NOMADS |
| **GODAS ocean** | `pottmp` · `salt` · `ucur` · `vcur` · `sshg` · 40 depth levels · 1980–present |
| **ERSST v5** | SST back to 1854 · long climatologies via OPeNDAP |
| **ENSO diagnostics** | ONI · Niño indices · D20 thermocline · WWV · phase classification |
| **26 plot functions** | Synoptic maps · globe · ENSO time series · thermocline sections · wind rose |
| **No API key** | All data via public OPeNDAP / NOMADS endpoints, lazy loading |

---

## Installation

```bash
pip install noawclg                   # core (GFS + ocean data)
pip install "noawclg[plots]"          # + matplotlib, cartopy, metpy, windrose, seaborn
```

> **GFS GRIB2 support** requires the `eccodes` C library.  
> On Ubuntu/Debian: `sudo apt install libeccodes-dev`  
> On macOS: `brew install eccodes`  
> On conda: `conda install -c conda-forge eccodes`

**Development install:**

```bash
git clone https://github.com/reinanbr/noawclg
cd noawclg
pip install -e ".[dev]"
pytest
```

---

## GFS weather forecasts

### Quick download with `load`

```python
from noawclg import load, auto_date

date, cycle = auto_date(lag_days=1)   # latest complete GFS run

ds = load(
    date=date,
    cycle=cycle,
    lat=-3.7,           # point of interest (Fortaleza, Brazil)
    lon=-38.5,
    region={
        "toplat": 5, "bottomlat": -15,
        "leftlon": -50, "rightlon": -30,
    },
    hours=list(range(0, 121, 3)),   # forecast hours 0–120 h every 3 h
)
print(ds)
```

The returned `xr.Dataset` contains:

| Variable | Description | Units |
|----------|-------------|-------|
| `t2m` | 2-m temperature | K |
| `d2m` | 2-m dew point | K |
| `prmsl` | Mean sea-level pressure | Pa |
| `u10` / `v10` | 10-m wind components | m/s |
| `gust` | Surface wind gust | m/s |
| `prate` | Precipitation rate | kg/m²/s |
| `r2` | 2-m relative humidity | % |
| `tcc` | Total cloud cover | 0–1 |
| `cape` | CAPE | J/kg |

### `auto_date` — pick the latest available GFS cycle

```python
from noawclg import auto_date

date, cycle = auto_date(lag_days=1)
# date  → "05/06/2026"  (DD/MM/YYYY)
# cycle → "12"          (00 / 06 / 12 / 18)
```

`lag_days=1` returns yesterday (safe; today's runs may still be publishing).  
`lag_days=0` returns today's most recently available cycle.

### `GFSDatasetManager` — full control

```python
from noawclg import GFSDatasetManager

mgr = GFSDatasetManager(
    date="05/06/2026",
    cycle="12",
    variables=["t2m", "u10", "v10", "prmsl"],
    region={"toplat": 10, "bottomlat": -20, "leftlon": -55, "rightlon": -25},
    output_dir="gfs_cache/",
)

mgr.download_hours(list(range(0, 49, 3)))
ds = mgr.build_multi_dataset(list(range(0, 49, 3)))

# Save / load
mgr.save_netcdf(ds, "forecast.nc")
ds2 = mgr.load_netcdf("forecast.nc")
mgr.save_zarr(ds, "forecast.zarr")
ds3 = mgr.load_zarr("forecast.zarr")
```

### `get_noaa_data` — query by place name

```python
from noawclg import get_noaa_data

gfs = get_noaa_data(
    date="05/06/2026",
    cycle="12",
    place="Recife PE",          # geocoded automatically
    hours=list(range(0, 73, 3)),
    variables=["t2m", "prmsl", "prate"],
)

# Access extracted time series
ts = gfs.time_series("t2m")      # pd.Series indexed by forecast time
pt = gfs.point_query("t2m", hour=24)  # scalar value at hour 24
```

### Pre-defined hour sequences

```python
from noawclg import HOURS_5DAYS_1H, HOURS_10DAYS_3H, HOURS_16DAYS_3H

# Download 5-day hourly forecast
ds = load(date=date, cycle=cycle, lat=0, lon=-40, hours=HOURS_5DAYS_1H)
```

| Constant | Hours | Step |
|----------|-------|------|
| `HOURS_5DAYS_1H` | 0–120 | 1 h |
| `HOURS_10DAYS_3H` | 0–240 | 3 h |
| `HOURS_16DAYS_3H` | 0–384 | 3 h |
| `HOURS_16DAYS` | 0–384 | 6 h |

---

## Ocean data — GODAS & ERSST

All ocean data is served via **OPeNDAP** — no files are downloaded, access is lazy.

### `open_godas` — single year, single variable

```python
from noawclg import open_godas

ds = open_godas(
    year=2024,
    variable="pottmp",    # "pottmp" | "salt" | "ucur" | "vcur" | "sshg"
    depth_m=200.0,        # nearest depth level; None = all 40 levels
    region={
        "lat_min": -10, "lat_max": 10,
        "lon_min": 120,  "lon_max": 290,
    },
)
print(ds["pottmp"])   # °C, (time=12, lat, lon)
```

**Available GODAS variables (`GODAS_VARS`):**

| Key | Description | Input units | Output units |
|-----|-------------|-------------|--------------|
| `pottmp` | Potential temperature | K | °C |
| `salt` | Salinity | kg/kg | PSU |
| `ucur` | U-current (eastward) | m/s | m/s |
| `vcur` | V-current (northward) | m/s | m/s |
| `sshg` | Sea surface height / geoid | m | m |

**40 depth levels:** 5, 15, 25, … 205, 215, 225, 238, 262, 303, 366, 459, 584, 747, 949, 1193, 1479, 1807, 2174, 2579, 3016, 3483, 3972, 4478 m.

### `get_godas` — multi-year concatenation

```python
from noawclg import get_godas

# All 2020–2024 temperature at 200 m
da = get_godas(2020, 2024, variable="pottmp", depth_m=200.0)
print(da)   # DataArray (time=60, lat, lon)
```

### Typed convenience wrappers

```python
from noawclg import get_ocean_temp, get_salinity, get_currents, get_ssh

t200 = get_ocean_temp(2024, depth_m=200)               # °C
t5   = get_ocean_temp(2024, depth_m=5)                 # surface temperature

sal  = get_salinity(2024, depth_m=5)                   # PSU

curr = get_currents(2024, depth_m=5)                   # Dataset: ucur, vcur, speed
print(curr["speed"].mean().item(), "m/s")

ssh  = get_ssh(2024)                                   # m, (time=12, lat, lon)
ssh5 = get_ssh(2020, 2024)                             # 5 years concatenated
```

### `open_ersst` — NOAA ERSST v5 (SST since 1854)

```python
from noawclg import open_ersst

# Niño 3.4 box, 1950–2024
sst = open_ersst(
    year_start=1950,
    year_end=2024,
    region={
        "lat_min": -5,  "lat_max": 5,
        "lon_min": 190, "lon_max": 240,
    },
)
print(sst["sst"])   # °C, (time=900, lat, lon)
```

ERSST uses a **decreasing latitude axis** (88 → −88); `open_ersst` handles this automatically.

### `get_sst_series` — monthly Niño-box SST time series

```python
from noawclg import get_sst_series

# From GODAS (1980+)
sst_godas = get_sst_series(2000, 2024, box="3.4", source="godas")

# From ERSST (1854+, longer climatology)
sst_ersst = get_sst_series(1950, 2024, box="3.4", source="ersst")
```

**Niño boxes (`NINO_BOXES`, longitude 0–360):**

| Key | Lat | Lon | Used for |
|-----|-----|-----|---------|
| `"1+2"` | 10°S–0° | 270–280°E | Near-coastal SST |
| `"3"` | 5°S–5°N | 210–270°E | Central/eastern Pacific |
| `"3.4"` | 5°S–5°N | 190–240°E | **ONI / official ENSO index** |
| `"4"` | 5°S–5°N | 160–210°E | Western Pacific |

---

## ENSO diagnostics

### ONI and phase classification

```python
from noawclg import get_nino_anomaly, get_oni, classify_enso

# SST anomaly relative to 1991–2020 climatology
anom = get_nino_anomaly(2000, 2024, box="3.4", source="ersst",
                        clim_start=1991, clim_end=2020)

# Oceanic Niño Index (3-month running mean of anomaly)
oni = get_oni(2000, 2024, source="ersst")

# Phase classification (CPC ONI rule: ≥5 consecutive seasons)
phase = classify_enso(oni)
# pd.Series with values: "El Niño" | "La Niña" | "Neutral"

print(oni.tail(6))
print(phase.tail(6))
```

### Complete ENSO summary table

```python
from noawclg import enso_summary

df = enso_summary(2015, 2024)
print(df.tail(12))
```

```
            sst_nino34  anom_nino34   oni      phase
month
2023-12-01    27.45        0.85      1.41    El Niño
2024-01-01    27.21        0.84      1.23    El Niño
2024-02-01    27.13        0.72      0.98    El Niño
...
```

Columns: `sst_nino34` (°C), `anom_nino34` (°C), `oni` (°C), `phase`.

### Thermocline depth D20

The depth of the 20 °C isotherm is the primary **dynamical** ENSO precursor.
When the thermocline deepens in the eastern Pacific, warm water accumulates → El Niño.

```python
from noawclg import get_thermocline_depth

d20 = get_thermocline_depth(2024, region={
    "lat_min": -30, "lat_max": 30,
    "lon_min": 120,  "lon_max": 290,
})
print(d20)   # DataArray (time=12, lat, lon) in metres
```

### Warm Water Volume (WWV)

WWV measures water warmer than 20 °C in the equatorial Pacific (5°S–5°N, 120°E–80°W)
above 300 m.  A large positive WWV anomaly typically precedes El Niño by 6–9 months.

```python
from noawclg import get_warm_water_volume

wwv = get_warm_water_volume(2020, 2024)
wwv.plot(title="Equatorial Pacific Warm Water Volume 2020–2024")
```

---

## Plotting

Install extras: `pip install "noawclg[plots]"`

All plot functions accept an `xarray.DataArray` or `Dataset` and return a
`matplotlib.Figure`.  Pass `save_path="file.png"` to save automatically.

### GFS plots

```python
from plots import (
    plot_synoptic_map,
    plot_wind_speed_map,
    plot_cloud_map,
    plot_cape_map,
    plot_precip_map,
    plot_timeseries,
    plot_temp_dewpoint,
    plot_wind_timeseries,
    plot_humidity_cloud,
    plot_cumulative_precip,
    plot_wind_rose,
    plot_hodograph,
    plot_vertical_profiles,
    plot_500hpa_map,
    plot_hovmoller,
    plot_precip_heatmap,
    plot_spread_matrix,
    plot_diurnal_distribution,
    plot_dashboard,
)

# Synoptic map — T2m filled + MSLP isobars + 10 m wind barbs (requires cartopy)
fig = plot_synoptic_map(ds, hour=24, title="GFS 24 h forecast", save_path="synoptic.png")

# Time-series at a point
fig = plot_timeseries(ds, city_name="Fortaleza CE", save_path="ts.png")

# Wind rose
fig = plot_wind_rose(ds, city_name="Fortaleza CE", save_path="rose.png")

# Upper-air hodograph
fig = plot_hodograph(ds_upper, lat=-3.7, lon=-38.5, save_path="hodo.png")

# Vertical temperature + humidity profiles
fig = plot_vertical_profiles(ds_upper, lat=-3.7, lon=-38.5, save_path="profiles.png")

# 500 hPa geopotential + jet stream (requires cartopy)
fig = plot_500hpa_map(ds_upper, hour=24, save_path="500hpa.png")

# Hovmöller — longitude vs time at fixed latitude
fig = plot_hovmoller(ds, mode="lon", lat=-3.7, variable="prate", save_path="hovmoller.png")

# 4-panel dashboard
fig = plot_dashboard(ds, hour=24, city_name="Fortaleza CE", save_path="dashboard.png")
```

### Ocean / ENSO plots

```python
from plots import (
    plot_enso_index,
    plot_ocean_temp_map,
    plot_thermocline_section,
    plot_ssh_map,
    plot_ocean_currents,
    plot_globe,
)

# ONI time series with El Niño / La Niña shading
from noawclg import get_oni, classify_enso
oni   = get_oni(2000, 2024)
phase = classify_enso(oni)
fig   = plot_enso_index(oni, phase, title="ONI 2000–2024", save_path="oni.png")

# Ocean temperature flat map with Niño-box overlays
from noawclg import get_ocean_temp
t200 = get_ocean_temp(2024, depth_m=200)
fig  = plot_ocean_temp_map(
    t200.mean("time"),
    title="Mean Ocean Temperature at 200 m — 2024",
    nino_boxes=True,     # draws Niño 1+2 / 3 / 3.4 / 4 rectangles
    cmap="RdYlBu_r",
    save_path="t200_map.png",
)

# Thermocline cross-section (depth–longitude, equatorial band)
from noawclg import get_godas
pottmp = get_godas(2024, variable="pottmp", region={
    "lat_min": -5, "lat_max": 5, "lon_min": 120, "lon_max": 290,
})
fig = plot_thermocline_section(pottmp, lat=0.0, isotherm=20.0, save_path="thermo.png")

# SSH anomaly map
from noawclg import get_ssh
ssh = get_ssh(2024)
fig = plot_ssh_map(ssh.isel(time=0), title="SSH — January 2024", save_path="ssh.png")

# Ocean currents — speed fill + quiver arrows
from noawclg import get_currents
curr = get_currents(2024, depth_m=5)
fig  = plot_ocean_currents(
    curr["ucur"].isel(time=0),
    curr["vcur"].isel(time=0),
    title="Surface Ocean Currents — January 2024",
    save_path="currents.png",
)

# Globe — any 2-D field on an Orthographic projection (requires cartopy)
fig = plot_globe(
    t200.mean("time"),
    title="Ocean Temperature at 200 m — Pacific view",
    cmap="RdYlBu_r",
    central_longitude=-150,     # Pacific-centred
    central_latitude=0,
    save_path="globe.png",
)

# Globe — SSH with symmetric colour scale
fig = plot_globe(
    ssh.mean("time"),
    cmap="RdBu_r",
    central_longitude=-150,
    symmetric=True,             # force vmin = -vmax
    save_path="globe_ssh.png",
)
```

### Colourmap reference

| Field | Default colourmap |
|-------|-------------------|
| Temperature | `cmocean.cm.thermal` / `RdYlBu_r` |
| Salinity | `cmocean.cm.haline` |
| SSH / anomaly | `RdBu_r` |
| Current speed | `plasma` |
| Precipitation | `Blues` |
| CAPE | `YlOrRd` |
| Wind speed | `viridis` |

---

## API reference

### `noawclg.load`

```python
load(date, cycle, lat, lon, region=None, hours=None, variables=None, **kwargs)
    → xr.Dataset
```

One-liner wrapper around `GFSDatasetManager`.  Returns an `xr.Dataset` with all
requested variables merged.

### `noawclg.auto_date`

```python
auto_date(lag_days=1) → tuple[str, str]
# Returns (date_str, cycle_str) e.g. ("05/06/2026", "12")
```

### `noawclg.GFSDatasetManager`

| Method | Description |
|--------|-------------|
| `download_hours(hours)` | Download GRIB2 files for given forecast hours |
| `build_dataset(hour, variables)` | Build `xr.Dataset` for a single hour |
| `build_multi_dataset(hours, variables)` | Build merged dataset for all hours |
| `save_netcdf(ds, filename)` | Save to NetCDF4 |
| `load_netcdf(filename)` | Load from NetCDF4 |
| `save_zarr(ds, path)` | Save to Zarr |
| `load_zarr(path)` | Load from Zarr |

### `noawclg.get_noaa_data`

```python
get_noaa_data(date, cycle, place=None, lat=None, lon=None,
              hours=None, variables=None, region=None, **kwargs)
```

High-level interface.  `place` is geocoded with `geopy` (e.g. `"São Paulo SP"`).

### Ocean functions

| Function | Description | Returns |
|----------|-------------|---------|
| `open_godas(year, variable, depth_m, region)` | Single-year GODAS via OPeNDAP | `xr.Dataset` |
| `get_godas(y0, y1, variable, depth_m, region)` | Multi-year GODAS | `xr.DataArray` |
| `get_ocean_temp(y0, y1, depth_m, region)` | Potential temperature (°C) | `xr.DataArray` |
| `get_salinity(y0, y1, depth_m, region)` | Salinity (PSU) | `xr.DataArray` |
| `get_currents(y0, y1, depth_m, region)` | U/V/speed (m/s) | `xr.Dataset` |
| `get_ssh(y0, y1, region)` | Sea Surface Height (m) | `xr.DataArray` |
| `open_ersst(y0, y1, region)` | ERSST v5 SST via OPeNDAP | `xr.Dataset` |
| `get_sst_series(y0, y1, box, source)` | Monthly Niño-box SST | `pd.Series` |
| `get_nino_anomaly(y0, y1, box, source, clim_start, clim_end)` | SST anomaly | `pd.Series` |
| `get_oni(y0, y1, source)` | Oceanic Niño Index | `pd.Series` |
| `classify_enso(oni)` | El Niño / La Niña / Neutral | `pd.Series[str]` |
| `get_thermocline_depth(y0, y1, region)` | D20 isotherm depth (m) | `xr.DataArray` |
| `get_warm_water_volume(y0, y1, max_depth)` | WWV index | `pd.Series` |
| `enso_summary(y0, y1)` | SST + anomaly + ONI + phase | `pd.DataFrame` |

All `year_end` parameters default to `year_start` (single year).  All `region`
dicts use keys `lat_min`, `lat_max`, `lon_min`, `lon_max` (longitude 0–360).

### `BoundingBox`

```python
from noawclg import BoundingBox

bb = BoundingBox(toplat=15, bottomlat=-15, leftlon=-80, rightlon=-30)
```

---

## Module overview

```
noawclg/
├── catalog.py      — GFS variable catalogue and hour sequences (VARIABLES, HOURS_*)
├── coords.py       — BoundingBox, auto_date, coordinate helpers
├── gfs_dataset.py  — GFSDatasetManager (download, build, cache)
├── http.py         — low-level GRIB2 download via NOMADS grib-filter
├── load.py         — noawclg.load() one-liner wrapper
├── main.py         — noawclg.main legacy entry-point
├── ocean.py        — GODAS / ERSST: temperature, salinity, currents,
│                     SSH, ENSO indices, WWV, D20, ONI classification
├── persistence.py  — NetCDF4 / Zarr save and load
├── query.py        — get_noaa_data() high-level interface
└── view.py         — dataset inspection helpers

plots.py            — 26 plot functions (GFS + ocean/ENSO)
enso_forecast.py    — real-data ENSO analysis and probability model
```

---

## Contributing

```bash
git clone https://github.com/reinanbr/noawclg
cd noawclg
pip install -e ".[dev]"
pytest tests/ -m "not integration"
```

Issues and pull requests are welcome on [GitHub](https://github.com/reinanbr/noawclg/issues).

## License

GPLv3 — see [LICENSE](LICENSE).
