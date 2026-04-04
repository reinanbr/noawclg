# noawclg · GFS Dataset Manager

> **Download, cache and analyse NOAA GFS forecast data in one line of Python.**

[![PyPI](https://img.shields.io/pypi/v/noawclg)](https://pypi.org/project/noawclg/)
[![Python](https://img.shields.io/pypi/pyversions/noawclg)](https://pypi.org/project/noawclg/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

`noawclg` wraps the [NOAA NOMADS grib-filter](https://nomads.ncep.noaa.gov/) endpoint and exposes a clean Python API that lets you:

- **Download** GFS 0.25° GRIB2 files with a single method call — one HTTP request per forecast hour regardless of how many variables you need.
- **Cache** raw GRIB2 files to disk so repeated runs cost nothing.
- **Extract** any combination of surface and upper-air variables into analysis-ready `xarray.Dataset` objects.
- **Save** output as compressed NetCDF4 or chunked Zarr for downstream processing.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [How It Works](#how-it-works)
4. [API Reference](#api-reference)
   - [GFSDatasetManager](#gfsdatasetmanager)
   - [build\_dataset](#build_dataset)
   - [build\_multi\_dataset](#build_multi_dataset)
   - [download\_hours](#download_hours)
   - [save\_netcdf](#save_netcdf)
   - [save\_zarr](#save_zarr)
   - [load\_netcdf](#load_netcdf)
   - [load\_zarr](#load_zarr)
5. [Variable Catalogue](#variable-catalogue)
6. [Pre-defined Hour Sequences](#pre-defined-hour-sequences)
7. [Region Subsetting](#region-subsetting)
8. [Logging](#logging)
9. [Examples](#examples)
10. [Contributing](#contributing)
11. [License](#license)

---

## Installation

```bash
pip install noawclg
```

### System dependency — eccodes

`cfgrib` requires the **eccodes** C library to decode GRIB2 files.

| Platform | Command |
|----------|---------|
| Ubuntu / Debian | `sudo apt install libeccodes-dev` |
| macOS (Homebrew) | `brew install eccodes` |
| Conda (any OS) | `conda install -c conda-forge eccodes` |

---

## Quick Start

```python
from noawclg import GFSDatasetManager

# Create a manager for the 06 Z run of 2026-04-03
mgr = GFSDatasetManager(date="20260403", cycle="06")

# Download t2m + precipitation for the next 48 h (6-hourly)
# → only 9 HTTP requests (one per hour), not 18
ds = mgr.build_multi_dataset(
    var_keys=["t2m", "prate"],
    hours=list(range(0, 49, 6)),
)

print(ds)
# <xarray.Dataset>
# Dimensions:  (time: 9, latitude: 721, longitude: 1440)
# Data variables:
#     t2m      (time, latitude, longitude) float64 ...
#     prate    (time, latitude, longitude) float64 ...

mgr.save_netcdf(ds, "/tmp/gfs_48h.nc")
```

---

## How It Works

### Single-download architecture

Previous approaches sent **one HTTP request per variable per forecast hour**.  
For 5 variables × 48 hours that means **240 requests**.

`noawclg` exploits the NOMADS grib-filter's multi-variable syntax to bundle every requested variable into a **single URL per hour**:

```
https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl
  ?dir=/gfs.20260403/06/atmos
  &file=gfs.t06z.pgrb2.0p25.f024
  &var_TMP=on&lev_2_m_above_ground=on       ← t2m
  &var_PRATE=on&lev_surface=on              ← prate
  &var_PRMSL=on&lev_mean_sea_level=on       ← prmsl
  &subregion=&toplat=5&bottomlat=-35&...    ← optional region
```

**Result:** 5 variables × 48 hours = **9 requests** (one per hour).

### Disk cache

Every downloaded GRIB2 file is saved under `output_dir` with a deterministic filename that encodes the date, cycle, variable set, region tag and forecast hour:

```
gfs_20260403_06z_prate_t2m_5N35S75W34E_f024.grib2
```

On subsequent runs the file is reused without any network I/O.

### cfgrib extraction

After downloading, each variable is extracted from the cached GRIB2 using `cfgrib` with a cascade of filter strategies (shortName → typeOfLevel → full scan) to handle the GRIB table inconsistencies that appear across GFS versions and sub-region files.

---

## API Reference

### `GFSDatasetManager`

```python
GFSDatasetManager(
    date: str,
    cycle: str = "00",
    output_dir: str = "./gfs_output",
    region: dict | None = None,
    pause: float = 1.5,
)
```

Main entry point. All other methods are called on an instance of this class.

| Parameter | Type | Description |
|-----------|------|-------------|
| `date` | `str` | Model run date in `YYYYMMDD` format. **Required.** |
| `cycle` | `str` | Model run cycle: `"00"`, `"06"`, `"12"` or `"18"`. Default `"00"`. |
| `output_dir` | `str` | Directory where GRIB2 files are cached. Created automatically. Default `"./gfs_output"`. |
| `region` | `dict \| None` | Bounding box for spatial subsetting (see [Region Subsetting](#region-subsetting)). `None` downloads the global grid. |
| `pause` | `float` | Seconds to sleep between consecutive HTTP requests. Helps avoid rate-limiting on NOMADS. Default `1.5`. |

**Raises:** `ValueError` if `cycle` is not one of the four valid values.  
**Raises:** `ValueError` if `date` does not match `YYYYMMDD`.

```python
from noawclg import GFSDatasetManager

mgr = GFSDatasetManager(
    date="20260403",
    cycle="06",
    output_dir="./cache",
    region={"toplat": 5, "bottomlat": -35, "leftlon": -75, "rightlon": -34},
    pause=2.0,
)
```

---

### `build_dataset`

```python
mgr.build_dataset(
    var_key: str,
    hours: list[int],
    force_download: bool = False,
) -> xr.Dataset
```

Download and assemble a Dataset for a **single variable**.

| Parameter | Type | Description |
|-----------|------|-------------|
| `var_key` | `str` | Variable key from the [Variable Catalogue](#variable-catalogue). |
| `hours` | `list[int]` | Forecast hours to include (e.g. `[0, 6, 12, 24]`). |
| `force_download` | `bool` | If `True`, re-download even if cached files exist. Default `False`. |

**Returns:** `xr.Dataset` with dimensions:
- Surface/single-level variables → `(time, latitude, longitude)`
- Multi-level variables → `(time, level, latitude, longitude)`

Both datasets include a `forecast_hour` coordinate aligned to the `time` dimension.

**Raises:** `RuntimeError` if no files could be downloaded or read.

```python
ds = mgr.build_dataset("t2m", hours=[0, 6, 12, 24, 48])
print(ds["t2m"].dims)   # ('time', 'latitude', 'longitude')
print(ds["t2m"].attrs)  # {'long_name': '2 metre temperature', 'units': 'C', ...}
```

---

### `build_multi_dataset`

```python
mgr.build_multi_dataset(
    var_keys: list[str],
    hours: list[int],
    force_download: bool = False,
) -> xr.Dataset
```

Download **one file per hour** containing **all** requested variables, then extract and merge them into a single Dataset.

This is the recommended method when you need more than one variable — it uses `N_hours` requests instead of `N_vars × N_hours`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `var_keys` | `list[str]` | List of variable keys from the [Variable Catalogue](#variable-catalogue). |
| `hours` | `list[int]` | Forecast hours to include. |
| `force_download` | `bool` | Re-download even if cached. Default `False`. |

**Returns:** `xr.Dataset` with all requested variables merged via `xr.merge(..., join="inner")`.

Variables that fail to extract are logged and skipped; a `RuntimeError` is raised only if *all* variables fail.

```python
ds = mgr.build_multi_dataset(
    var_keys=["t2m", "prmsl", "prate", "u10", "v10"],
    hours=list(range(0, 25, 6)),
)
# ds contains t2m, prmsl, prate, u10, v10 all on the same time axis
```

---

### `download_hours`

```python
mgr.download_hours(
    var_keys: list[str],
    hours: list[int],
    force: bool = False,
) -> dict[int, Path]
```

Low-level method that performs the actual HTTP downloads.  
Called internally by `build_dataset` and `build_multi_dataset`, but exposed for advanced use cases (e.g. downloading files without immediately building a Dataset).

| Parameter | Type | Description |
|-----------|------|-------------|
| `var_keys` | `list[str]` | Variables to bundle into each download URL. |
| `hours` | `list[int]` | Forecast hours to download. |
| `force` | `bool` | Re-download cached files. Default `False`. |

**Returns:** `dict[int, Path]` — mapping of `{hour: path_to_grib2_file}` for every successfully downloaded hour.

Files already on disk are returned immediately without any network I/O (cache hit is logged at `INFO` level).

```python
files = mgr.download_hours(["t2m", "prate"], hours=[0, 6, 12])
# {0: PosixPath('.../gfs_..._f000.grib2'),
#  6: PosixPath('.../gfs_..._f006.grib2'),
#  12: PosixPath('.../gfs_..._f012.grib2')}
```

---

### `save_netcdf`

```python
mgr.save_netcdf(
    ds: xr.Dataset,
    filename: str,
    complevel: int = 4,
) -> Path
```

Save a Dataset to a **zlib-compressed NetCDF4** file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ds` | `xr.Dataset` | Dataset to save. |
| `filename` | `str` | Output file path. Absolute paths are used as-is; relative paths are resolved against `output_dir`. |
| `complevel` | `int` | zlib compression level 1–9 (higher = smaller file, slower write). Default `4`. |

**Returns:** `Path` — absolute path of the saved file.

```python
path = mgr.save_netcdf(ds, "/data/gfs_t2m_48h.nc")
# or relative (saved inside output_dir):
path = mgr.save_netcdf(ds, "gfs_t2m_48h.nc")
```

---

### `save_zarr`

```python
mgr.save_zarr(
    ds: xr.Dataset,
    store: str,
) -> Path
```

Save a Dataset as a **chunked Zarr store** (directory).

Zarr is preferred over NetCDF for large time-series because it supports:
- Lazy chunked reads without loading the whole file into memory.
- Appending new timesteps without rewriting existing data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ds` | `xr.Dataset` | Dataset to save. |
| `store` | `str` | Output directory path. Relative paths are resolved against `output_dir`. |

**Returns:** `Path` — absolute path of the Zarr store directory.

```python
path = mgr.save_zarr(ds, "gfs_surface_16days.zarr")
```

---

### `load_netcdf`

```python
GFSDatasetManager.load_netcdf(path: str | Path) -> xr.Dataset
```

Static method. Lazily open a previously saved NetCDF file using Dask-backed chunking.

```python
ds = GFSDatasetManager.load_netcdf("/data/gfs_t2m_48h.nc")
print(dict(ds.dims))  # {'time': 9, 'latitude': 721, 'longitude': 1440}
```

---

### `load_zarr`

```python
GFSDatasetManager.load_zarr(store: str | Path) -> xr.Dataset
```

Static method. Lazily open a previously saved Zarr store.

```python
ds = GFSDatasetManager.load_zarr("gfs_surface_16days.zarr")
```

---

## Variable Catalogue

Access the full catalogue at runtime:

```python
from noawclg import VARIABLES, SURFACE_VARS, MULTILEVEL_VARS

print(SURFACE_VARS)    # all 2-D (no level dimension) variable keys
print(MULTILEVEL_VARS) # all variables with a vertical level dimension
```

### Surface / single-level variables

| Key | Long name | Units |
|-----|-----------|-------|
| `t2m` | 2 metre temperature | °C |
| `d2m` | 2 metre dewpoint temperature | °C |
| `r2` | 2 metre relative humidity | % |
| `sh2` | 2 metre specific humidity | kg kg⁻¹ |
| `aptmp` | Apparent temperature | °C |
| `u10` | 10 metre U wind component | m s⁻¹ |
| `v10` | 10 metre V wind component | m s⁻¹ |
| `gust` | Wind speed (gust) | m s⁻¹ |
| `prmsl` | Pressure reduced to MSL | hPa |
| `mslet` | MSLP (Eta model reduction) | hPa |
| `sp` | Surface pressure | hPa |
| `orog` | Orography | m |
| `lsm` | Land-sea mask | 0–1 |
| `vis` | Visibility | m |
| `prate` | Precipitation rate | kg m⁻² s⁻¹ |
| `cpofp` | Percent frozen precipitation | % |
| `crain` | Categorical rain | — |
| `csnow` | Categorical snow | — |
| `cfrzr` | Categorical freezing rain | — |
| `cicep` | Categorical ice pellets | — |
| `sde` | Snow depth | m |
| `sdwe` | Water equivalent of snow depth | kg m⁻² |
| `pwat` | Precipitable water | kg m⁻² |
| `cwat` | Cloud water | kg m⁻² |
| `tcc` | Total cloud cover | % |
| `lcc` | Low cloud cover | % |
| `mcc` | Medium cloud cover | % |
| `hcc` | High cloud cover | % |
| `lftx` | Surface lifted index | K |
| `lftx4` | Best (4-layer) lifted index | K |
| `hlcy` | Storm relative helicity | m² s⁻² |
| `refc` | Composite radar reflectivity | dB |
| `siconc` | Sea ice area fraction | 0–1 |
| `veg` | Vegetation | % |
| `tozne` | Total ozone | DU |

### Multi-level variables

These variables include a `level` dimension in the output Dataset.

| Key | Long name | Units | Levels |
|-----|-----------|-------|--------|
| `t` | Temperature | °C | 80–1000 hPa (13 levels) |
| `r` | Relative humidity | % | 80–1000 hPa (13 levels) |
| `q` | Specific humidity | kg kg⁻¹ | 80, 1000 hPa |
| `gh` | Geopotential height | gpm | 500–1000 hPa (5 levels) |
| `u` | U component of wind | m s⁻¹ | 200–1000 hPa (9 levels) |
| `v` | V component of wind | m s⁻¹ | 200–1000 hPa (9 levels) |
| `w` | Vertical velocity | Pa s⁻¹ | 100–850 hPa (8 levels) |
| `absv` | Absolute vorticity | s⁻¹ | 100–1000 hPa (8 levels) |
| `cape` | CAPE | J kg⁻¹ | surface layers |
| `cin` | Convective inhibition | J kg⁻¹ | surface layers |
| `st` | Soil temperature | °C | 0–100 cm (4 layers) |
| `soilw` | Volumetric soil moisture | Proportion | 0–100 cm (4 layers) |

---

## Pre-defined Hour Sequences

```python
from noawclg import (
    HOURS_16DAYS,     # 0–120 h (6-hourly) + 123–384 h (3-hourly) — full 16-day run
    HOURS_5DAYS_1H,   # 0–120 h (1-hourly)
    HOURS_10DAYS_3H,  # 0–240 h (3-hourly)
    HOURS_16DAYS_3H,  # 0–120 h (3-hourly) + 123–384 h (3-hourly)
)
```

Use them directly with `build_dataset` or `build_multi_dataset`:

```python
ds = mgr.build_dataset("t2m", hours=HOURS_16DAYS)
```

---

## Region Subsetting

Pass a `region` dict to download only the data inside a bounding box.  
This dramatically reduces file size and download time for regional studies.

```python
# South America
REGION_SA = {
    "toplat":    12,
    "bottomlat": -56,
    "leftlon":   -82,
    "rightlon":  -34,
}

# Brazil
REGION_BR = {
    "toplat":    5,
    "bottomlat": -35,
    "leftlon":   -75,
    "rightlon":  -34,
}

mgr = GFSDatasetManager(
    date="20260403",
    cycle="06",
    region=REGION_BR,
)
```

Pass `region=None` (the default) for a global download.

> **Note:** The region tag is embedded in the cache filename, so global and regional downloads never collide even when sharing the same `output_dir`.

---

## Logging

`noawclg` uses Python's standard `logging` module under the logger name `gfs_dataset`.  
Enable it in your application to see download progress, cache hits and extraction warnings:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
```

Sample output:

```
10:02:15  INFO      Download: 9 hour(s) × 1 file each = 9 request(s)  (vars: ['t2m', 'prate'])
10:02:17  INFO      [multi] → f000  https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?...
10:02:19  INFO        [ok] f000  284 KB  |  11.1%  (1/9)  elapsed=2.1s  remaining≈15.2s
10:02:21  INFO      [cache] f006  gfs_20260403_06z_prate_t2m_global_f006.grib2
10:02:21  INFO      Extracting 't2m' …
10:02:21  INFO      Extracting 'prate' …
```

---

## Examples

### 1 — Surface forecast for Brazil, 48 h

```python
from noawclg import GFSDatasetManager

mgr = GFSDatasetManager(
    date="20260403",
    cycle="06",
    region={"toplat": 5, "bottomlat": -35, "leftlon": -75, "rightlon": -34},
)

ds = mgr.build_multi_dataset(
    var_keys=["t2m", "prate", "prmsl", "u10", "v10"],
    hours=list(range(0, 49, 6)),
)
mgr.save_netcdf(ds, "gfs_brazil_48h.nc")
```

### 2 — Upper-air wind profile, global, 24 h

```python
ds = mgr.build_multi_dataset(
    var_keys=["u", "v", "gh"],   # multi-level isobaric
    hours=list(range(0, 25, 6)),
)
# ds["u"] has dims (time, level, latitude, longitude)
u_500 = ds["u"].sel(level=500)   # wind at 500 hPa
```

### 3 — 16-day t2m time-series, saved as Zarr

```python
from noawclg import GFSDatasetManager, HOURS_16DAYS

mgr = GFSDatasetManager(date="20260403", cycle="00")
ds  = mgr.build_dataset("t2m", hours=HOURS_16DAYS)
mgr.save_zarr(ds, "gfs_t2m_16days.zarr")
```

### 4 — Reload and compute a daily mean

```python
import xarray as xr
from noawclg import GFSDatasetManager

ds   = GFSDatasetManager.load_netcdf("gfs_brazil_48h.nc")
t2m  = ds["t2m"]
daily_mean = t2m.resample(time="1D").mean()
print(daily_mean)
```

### 5 — Download only (no Dataset construction)

```python
files = mgr.download_hours(
    var_keys=["t2m", "prate"],
    hours=[0, 6, 12, 24],
)
# {0: PosixPath('./gfs_output/gfs_20260403_06z_prate_t2m_global_f000.grib2'), ...}
```

---

## Contributing

Pull requests are welcome. For major changes please open an issue first to discuss what you would like to change.

```bash
git clone https://github.com/reinanbr/noawclg
cd noawclg
pip install -e ".[dev]"
```

---

## License

[MIT](LICENSE) © Reinan BR