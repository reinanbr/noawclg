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

### `auto_date` — pick the latest available GFS cycle

```python
from noawclg import auto_date

date, cycle = auto_date(lag_days=1)
# date  → "05/06/2026"  (DD/MM/YYYY — always this format)
# cycle → "12"          (00 / 06 / 12 / 18)
```

`lag_days=1` targets yesterday's run (always published on NOMADS).  
`lag_days=0` returns today's most recently published cycle (~4 h after initialization).

### `load` — quick download as `xr.Dataset`

`load()` returns an `xr.Dataset` directly. Use `keys=` to select variables and
`region=` to crop spatially. There are **no `lat=`/`lon=` parameters** — to
query a single point, select it from the returned dataset with `.sel`.

```python
from noawclg import load, auto_date

date, cycle = auto_date(lag_days=1)

ds = load(
    date=date,
    cycle=cycle,
    keys=["t2m", "u10", "v10", "prmsl", "prate", "r2", "cape"],
    region={
        "toplat": 5, "bottomlat": -15,
        "leftlon": -50, "rightlon": -30,
    },
    hours=list(range(0, 121, 3)),   # 0–120 h every 3 h
)

print(ds)
# Select a single grid point (nearest-neighbour)
point = ds.sel(latitude=-3.7, longitude=-38.5, method="nearest")
```

Variables returned with correct units after internal conversion:

| Key | Description | Units |
|-----|-------------|-------|
| `t2m` | 2 m temperature | °C |
| `d2m` | 2 m dew point | °C |
| `prmsl` | Mean sea-level pressure | hPa |
| `u10` / `v10` | 10 m wind components | m/s |
| `gust` | Surface wind gust | m/s |
| `prate` | Precipitation rate | kg/m²/s |
| `r2` | 2 m relative humidity | % |
| `tcc` | Total cloud cover | % |
| `cape` | CAPE | J/kg |

### Pre-defined hour sequences

```python
from noawclg import HOURS_5DAYS_1H, HOURS_10DAYS_3H, HOURS_16DAYS_3H

ds = load(date=date, cycle=cycle, keys=["t2m"], hours=HOURS_5DAYS_1H)
```

| Constant | Range | Step |
|----------|-------|------|
| `HOURS_5DAYS_1H` | 0–120 h | 1 h |
| `HOURS_10DAYS_3H` | 0–240 h | 3 h |
| `HOURS_16DAYS_3H` | 0–384 h | 3 h |
| `HOURS_16DAYS` | 0–384 h | 6 h |

### `GFSDatasetManager` — full control over download and storage

`GFSDatasetManager` receives `date` in **`YYYYMMDD`** format.  Variables are
passed to the download/build methods, not to the constructor.

```python
from noawclg import GFSDatasetManager

mgr = GFSDatasetManager(
    date="20260605",               # YYYYMMDD
    cycle="12",
    region={"toplat": 10, "bottomlat": -20, "leftlon": -55, "rightlon": -25},
    output_dir="gfs_cache/",
)

hours = list(range(0, 49, 3))

# One variable → build_dataset
ds_t = mgr.build_dataset("t2m", hours)

# Multiple variables at once → build_multi_dataset
ds = mgr.build_multi_dataset(["t2m", "u10", "v10", "prmsl"], hours)

# Persist and reload
mgr.save_netcdf(ds, "forecast.nc")
ds2 = mgr.load_netcdf("forecast.nc")

mgr.save_zarr(ds, "forecast.zarr")
ds3 = mgr.load_zarr("forecast.zarr")
```

### `get_noaa_data` — query by coordinates or place name

`get_noaa_data` is a class. Instantiate it with `keys=`, then use spatial
query methods. Date format is `DD/MM/YYYY` (same as `auto_date` output).

```python
from noawclg import get_noaa_data

gfs = get_noaa_data(
    date="05/06/2026",            # DD/MM/YYYY
    cycle="12",
    keys=["t2m", "prmsl", "prate", "u10", "v10"],
    hours=list(range(0, 73, 3)),
    region={"toplat": 5, "bottomlat": -15, "leftlon": -50, "rightlon": -30},
)

# Query by coordinates → returns a _DatasetView
view = gfs.get_data_from_point(point=(-3.7, -38.5))
df   = view.to_dataframe()        # pd.DataFrame

# Query by place name (geocoded automatically via Nominatim)
view2 = gfs.get_data_from_place("Recife PE")
df2   = view2.to_dataframe()

# Complete time series for one variable at a grid point
t2m_series = gfs.get_time_series(point=(-3.7, -38.5), variable="t2m")
# → xr.DataArray indexed by time

# List all loaded variables
print(gfs.get_keys())   # {"t2m": "2 metre temperature", ...}

# Access the raw xr.Dataset
print(gfs._ds)
```

---

## Mathematical analysis examples

All examples below use a dataset loaded with:

```python
from noawclg import load, auto_date
import numpy as np
import pandas as pd
from scipy import stats, signal

date, cycle = auto_date(lag_days=1)
ds = load(
    date=date, cycle=cycle,
    keys=["t2m", "d2m", "u10", "v10", "prmsl", "prate",
          "r2", "cape", "tcc", "gust"],
    region={"toplat": 5, "bottomlat": -15, "leftlon": -50, "rightlon": -30},
    hours=list(range(0, 121, 3)),
)
# Pick a single point for time-series examples
pt = ds.sel(latitude=-3.7, longitude=-38.5, method="nearest")
```

### Temperature — heat index, anomaly, trend

```python
# --- heat index (Rothfusz equation, °C in → °C out)
T = pt["t2m"].values          # °C
RH = pt["r2"].values          # %

HI = (
    -8.78469475556
    + 1.61139411 * T
    + 2.33854883889 * RH
    - 0.14611605 * T * RH
    - 0.012308094 * T**2
    - 0.0164248277778 * RH**2
    + 0.002211732 * T**2 * RH
    + 0.00072546 * T * RH**2
    - 0.000003582 * T**2 * RH**2
)

# --- anomaly relative to 0-h (analysis) step
t2m_anom = pt["t2m"].values - pt["t2m"].values[0]

# --- linear trend across the forecast window
hours_arr = np.array(list(range(0, 121, 3)), dtype=float)
slope, intercept, r, p, se = stats.linregress(hours_arr, pt["t2m"].values)
print(f"Warming rate: {slope:.3f} °C/h  (R²={r**2:.3f})")

# --- rolling 24-h mean using pandas
t_series = pd.Series(pt["t2m"].values, index=pd.to_timedelta(hours_arr, unit="h"))
t_roll24 = t_series.rolling("24h").mean()
```

### Dew-point depression and relative humidity check

```python
# Dew-point depression (dry-bulb minus dew-point, °C)
Td = pt["d2m"].values   # °C
T  = pt["t2m"].values

depression = T - Td     # 0 → fully saturated; > 10 → dry air

# Magnus formula: recompute RH from T and Td to cross-check
a, b = 17.625, 243.04   # Magnus constants
RH_check = 100 * np.exp(a * Td / (b + Td)) / np.exp(a * T / (b + T))
```

### Wind — speed, direction, wind stress, gusts

```python
u = pt["u10"].values   # m/s (positive = westerly)
v = pt["v10"].values   # m/s (positive = southerly)

# Scalar wind speed and meteorological direction (from, 0°=N)
wspd = np.hypot(u, v)
wdir = (270 - np.degrees(np.arctan2(v, u))) % 360

# Wind stress (bulk formula, air density ρ ≈ 1.225 kg/m³, Cd ≈ 1.3e-3)
rho, Cd = 1.225, 1.3e-3
tau_x = rho * Cd * wspd * u
tau_y = rho * Cd * wspd * v

# Normalised gust factor  (gust / sustained)
gust = pt["gust"].values
gust_factor = np.where(wspd > 0, gust / wspd, np.nan)

# Beaufort scale
beaufort = np.digitize(wspd, [0.3, 1.6, 3.4, 5.5, 8.0, 10.8,
                               13.9, 17.2, 20.8, 24.5, 28.5, 32.7])

# FFT on wind speed — dominant frequency in the forecast signal
fft_coeffs = np.fft.rfft(wspd - wspd.mean())
freqs = np.fft.rfftfreq(len(wspd), d=3.0)   # sampling interval = 3 h
dominant_period_h = 1.0 / freqs[np.argmax(np.abs(fft_coeffs[1:])) + 1]
```

### Pressure — gradient, smoothing, tendency

```python
prmsl = ds["prmsl"].values   # shape (time, lat, lon), hPa

# Spatial gradient (hPa / grid-cell) at each time step
dp_dy, dp_dx = np.gradient(prmsl, axis=(1, 2))

# Gaussian spatial smoothing (σ = 2 grid points)
from scipy.ndimage import gaussian_filter
prmsl_smooth = gaussian_filter(prmsl, sigma=(0, 2, 2))

# Pressure tendency (hPa/3 h) — central differences in time
tendency = np.gradient(prmsl, 3.0, axis=0)   # axis 0 = time

# At the chosen point: anomaly from first step
p_pt = pt["prmsl"].values
p_anom = p_pt - p_pt[0]
```

### Precipitation — accumulation, exceedance probability

```python
# prate is in kg/m²/s; multiply by 3600 to get mm/h,
# then multiply by timestep in hours to accumulate
dt_hours = 3
prate = pt["prate"].values          # kg/m²/s = mm/s

precip_rate_mm_h = prate * 3600                        # mm/h
precip_accum_mm  = np.cumsum(precip_rate_mm_h * dt_hours)  # mm total

# 24-h accumulated totals (rolling sum over 8 × 3-h steps)
precip_series = pd.Series(precip_rate_mm_h * dt_hours)
precip_24h = precip_series.rolling(8).sum()

# Probability of exceeding 5 mm/h — empirical from spatial domain
prate_all = ds["prate"].values   # (time, lat, lon)
prate_mm_h = prate_all * 3600
prob_5mm = np.mean(prate_mm_h > 5, axis=(1, 2))   # fraction per timestep
```

### CAPE — instability classification and spatial statistics

```python
cape = ds["cape"].values   # J/kg, shape (time, lat, lon)

# Instability categories at each grid point (latest forecast step)
cape_now = cape[0]
categories = np.select(
    [cape_now < 300, cape_now < 1000, cape_now < 2500],
    [0, 1, 2],            # 0=marginal, 1=moderate, 2=large
    default=3,            # 3=extreme
)

# Area fraction with extreme instability (CAPE > 2500 J/kg) per timestep
frac_extreme = np.mean(cape > 2500, axis=(1, 2))

# Spatial percentiles at each forecast hour
cape_flat = cape.reshape(cape.shape[0], -1)   # (time, n_points)
p25, p50, p75 = np.percentile(cape_flat, [25, 50, 75], axis=1)
```

### Upper-air multi-level variables — vertical profiles

```python
from noawclg import load, auto_date
import numpy as np
import pandas as pd

date, cycle = auto_date(lag_days=1)
ds_ua = load(
    date=date, cycle=cycle,
    keys=["t", "r", "gh", "u", "v"],
    region={"toplat": 5, "bottomlat": -15, "leftlon": -50, "rightlon": -30},
    hours=[0, 24, 48],
)

# Nearest-point vertical profile at analysis time (h=0)
prof = ds_ua.sel(latitude=-3.7, longitude=-38.5, method="nearest").isel(time=0)

levels = prof["level"].values        # hPa array  e.g. [200, 250, ... 1000]
T_prof = prof["t"].values            # °C
RH_prof = prof["r"].values           # %
gh_prof = prof["gh"].values          # gpm

# Layer thickness (proportional to mean temperature)
R, g = 287.05, 9.81
for i in range(len(levels) - 1):
    T_mean_K = (T_prof[i] + T_prof[i + 1]) / 2 + 273.15
    dz = (R * T_mean_K / g) * np.log(levels[i] / levels[i + 1])
    print(f"{levels[i]:4.0f}→{levels[i+1]:4.0f} hPa  Δz ≈ {dz:.0f} m")

# Wind shear (m/s per hPa) between levels
u_prof = prof["u"].values
v_prof = prof["v"].values
shear_u = np.diff(u_prof) / np.diff(levels)
shear_v = np.diff(v_prof) / np.diff(levels)
shear_mag = np.hypot(shear_u, shear_v)

# Lifted index approximation: T_env(500) - T_parcel(500)
# (simple: parcel raised dry-adiabatically from surface)
T_sfc = T_prof[-1] + 273.15          # near-surface level (1000 hPa)
T_500_env = T_prof[np.argmin(np.abs(levels - 500))] + 273.15
lapse_dry = 9.8 / 1000               # °C/m
dz_500 = gh_prof[np.argmin(np.abs(levels - 500))] - gh_prof[-1]
T_500_parcel = T_sfc - lapse_dry * dz_500
LI_approx = T_500_env - T_500_parcel
print(f"Lifted index ≈ {LI_approx:.1f} K")
```

### Converting dataset to pandas for time-series analysis

```python
# Flatten the spatial domain into a DataFrame
pt = ds.sel(latitude=-3.7, longitude=-38.5, method="nearest")
df = pt.to_dataframe().reset_index()

# Correlation matrix between all variables
numeric_cols = ["t2m", "d2m", "u10", "v10", "prmsl", "r2", "cape"]
corr = df[numeric_cols].corr()
print(corr)

# Descriptive statistics
print(df[numeric_cols].describe())

# Resample to 6-hourly means (data is on 3-h steps)
df = df.set_index("valid_time") if "valid_time" in df.columns else df
df_6h = df[numeric_cols].resample("6h").mean()

# Peak detection on temperature using scipy
from scipy.signal import find_peaks
peaks, props = find_peaks(df["t2m"], prominence=1.5)
print("Temperature maxima at hours:", df["t2m"].iloc[peaks].index.tolist())
```

---

## GFS variable catalogue

All 43 keys available in `noawclg.VARIABLES`.  Pass any subset as the `keys=`
argument to `load()`, `get_noaa_data()`, or the build methods of
`GFSDatasetManager`.

```python
from noawclg import VARIABLES, SURFACE_VARS, MULTILEVEL_VARS

print(list(VARIABLES.keys()))   # all keys
print(SURFACE_VARS)             # single-level keys (no level dimension)
print(MULTILEVEL_VARS)          # pressure-level / multi-layer keys

# Print key → description → units for every variable
for key, meta in VARIABLES.items():
    print(f"{key:8s}  {meta['long_name']:50s}  {meta['units']}")
```

> **Notes legend**  
> `K→°C` — raw GRIB value is in Kelvin, converted to Celsius on load  
> `Pa→hPa` — raw GRIB value is in Pascals, divided by 100 on load  
> `pgrb2b` — variable lives in the secondary GRIB2 file; may require separate filter request  
> `f000` — only present at forecast hour 0 (analysis step)  
> `multilevel` — has a `level` coordinate; select with `.sel(level=…)`

---

### 2 m / surface — temperature and humidity

| Key | Long name | Units | GRIB var | Notes |
|-----|-----------|-------|----------|-------|
| `t2m` | 2 metre temperature | °C | `TMP` @ 2 m | K→°C |
| `d2m` | 2 metre dewpoint temperature | °C | `DPT` @ 2 m | K→°C |
| `r2` | 2 metre relative humidity | % | `RH` @ 2 m | — |
| `sh2` | 2 metre specific humidity | kg kg⁻¹ | `SPFH` @ 2 m | — |
| `aptmp` | Apparent temperature (feels-like) | °C | `APTMP` @ 2 m | K→°C · pgrb2b |

---

### 10 m wind

| Key | Long name | Units | GRIB var | Notes |
|-----|-----------|-------|----------|-------|
| `u10` | 10 metre U wind component | m s⁻¹ | `UGRD` @ 10 m | positive = eastward |
| `v10` | 10 metre V wind component | m s⁻¹ | `VGRD` @ 10 m | positive = northward |
| `gust` | Wind speed (gust) | m s⁻¹ | `GUST` @ surface | maximum gust |

---

### Pressure and terrain

| Key | Long name | Units | GRIB var | Notes |
|-----|-----------|-------|----------|-------|
| `prmsl` | Pressure reduced to MSL | hPa | `PRMSL` @ mean sea level | Pa→hPa |
| `mslet` | MSLP — Eta model reduction | hPa | `MSLET` @ mean sea level | Pa→hPa · pgrb2b |
| `sp` | Surface pressure | hPa | `PRES` @ surface | Pa→hPa; at terrain height |
| `orog` | Orography | m | `HGT` @ surface | terrain elevation |
| `lsm` | Land-sea mask | 0–1 | `LAND` @ surface | 1 = land, 0 = sea |
| `vis` | Visibility | m | `VIS` @ surface | — |

---

### Precipitation and hydrology

| Key | Long name | Units | GRIB var | Notes |
|-----|-----------|-------|----------|-------|
| `prate` | Precipitation rate | kg m⁻² s⁻¹ | `PRATE` @ surface | multiply × 3600 → mm h⁻¹ |
| `cpofp` | Percent frozen precipitation | % | `CPOFP` @ surface | fraction of precip that is frozen |
| `crain` | Categorical rain | 0/1 | `CRAIN` @ surface | 1 = rain occurring |
| `csnow` | Categorical snow | 0/1 | `CSNOW` @ surface | 1 = snow occurring |
| `cfrzr` | Categorical freezing rain | 0/1 | `CFRZR` @ surface | 1 = freezing rain occurring |
| `cicep` | Categorical ice pellets | 0/1 | `CICEP` @ surface | 1 = ice pellets occurring |
| `sde` | Snow depth | m | `SNOD` @ surface | — |
| `sdwe` | Water equiv. of accum. snow depth | kg m⁻² | `WEASD` @ surface | liquid-equivalent snow mass |
| `pwat` | Precipitable water | kg m⁻² | `PWAT` @ entire atmosphere | total column water vapour |
| `cwat` | Cloud water | kg m⁻² | `CWAT` @ entire atmosphere | total column liquid + ice |

---

### Cloud cover

| Key | Long name | Units | GRIB var | Notes |
|-----|-----------|-------|----------|-------|
| `tcc` | Total cloud cover | % | `TCDC` @ entire atmosphere | all layers combined |
| `lcc` | Low cloud cover | % | `TCDC` @ low cloud layer | below ~2 km · pgrb2b |
| `mcc` | Medium cloud cover | % | `TCDC` @ middle cloud layer | ~2–6 km · pgrb2b |
| `hcc` | High cloud cover | % | `TCDC` @ high cloud layer | above ~6 km · pgrb2b |

---

### Convection and instability

| Key | Long name | Units | GRIB var | Notes |
|-----|-----------|-------|----------|-------|
| `cape` | Convective available potential energy | J kg⁻¹ | `CAPE` @ surface | > 0 = potential for deep convection |
| `cin` | Convective inhibition | J kg⁻¹ | `CIN` @ surface | negative = inhibits convection |
| `lftx` | Surface lifted index | K | `LFTX` @ surface | negative = unstable column |
| `lftx4` | Best (4-layer) lifted index | K | `4LFTX` @ surface | most unstable of 4 layers |
| `hlcy` | Storm relative helicity (0–3 km) | m² s⁻² | `HLCY` @ 0–3000 m layer | > 150 = elevated tornado risk |

---

### Upper-air — isobaric levels (multilevel)

These variables carry a `level` coordinate (hPa). Select with `.sel(level=500)`.

| Key | Long name | Units | GRIB var | Available levels (hPa) |
|-----|-----------|-------|----------|------------------------|
| `t` | Temperature | °C | `TMP` @ isobaric | 80 100 150 200 250 300 400 500 600 700 850 925 1000 · K→°C |
| `r` | Relative humidity | % | `RH` @ isobaric | 80 100 150 200 250 300 400 500 600 700 850 925 1000 |
| `q` | Specific humidity | kg kg⁻¹ | `SPFH` @ isobaric | 80 1000 |
| `gh` | Geopotential height | gpm | `HGT` @ isobaric | 500 700 850 925 1000 |
| `u` | U component of wind | m s⁻¹ | `UGRD` @ isobaric | 200 250 300 400 500 700 850 925 1000 |
| `v` | V component of wind | m s⁻¹ | `VGRD` @ isobaric | 200 250 300 400 500 700 850 925 1000 |
| `w` | Vertical velocity | Pa s⁻¹ | `VVEL` @ isobaric | 100 200 300 400 500 600 700 850 |
| `absv` | Absolute vorticity | s⁻¹ | `ABSV` @ isobaric | 100 200 300 400 500 700 850 1000 |

```python
from noawclg import load, auto_date

date, cycle = auto_date(lag_days=1)

ds = load(
    date=date, cycle=cycle,
    keys=["t", "r", "gh", "u", "v"],
    region={"toplat": 5, "bottomlat": -15, "leftlon": -50, "rightlon": -30},
    hours=[0, 24, 48],
)

t500  = ds["t"].sel(level=500)          # temperature at 500 hPa
gh850 = ds["gh"].sel(level=850)         # geopotential at 850 hPa

# Full vertical profile at a point (all levels, hour 0)
prof = ds.sel(latitude=-3.7, longitude=-38.5, method="nearest").isel(time=0)
print(prof["t"].values)   # °C for each level
```

---

### Soil — 4 depth layers (multilevel)

Layer key (`level=`) corresponds to the **top** of each depth range.

| Key | Long name | Units | GRIB var | Depth layers | Notes |
|-----|-----------|-------|----------|--------------|-------|
| `st` | Soil temperature | °C | `TSOIL` | 0–10, 10–40, 40–100, 100–200 cm | K→°C |
| `soilw` | Volumetric soil moisture content | proportion | `SOILW` | 0–10, 10–40, 40–100, 100–200 cm | 0 = dry, 1 = saturated |

```python
ds_soil = load(date=date, cycle=cycle, keys=["st", "soilw"], hours=[0, 24])

st_surface  = ds_soil["st"].sel(level=0)    # 0–10 cm layer
st_deep     = ds_soil["st"].sel(level=100)  # 100–200 cm layer
soilw_top   = ds_soil["soilw"].sel(level=0) # surface moisture
```

---

### Diagnostics

| Key | Long name | Units | GRIB var | Notes |
|-----|-----------|-------|----------|-------|
| `refc` | Maximum/Composite radar reflectivity | dB | `REFC` @ entire atmosphere | simulated composite reflectivity |
| `siconc` | Sea ice area fraction | 0–1 | `ICEC` @ surface | 0 = open ocean, 1 = full ice cover |
| `veg` | Vegetation | % | `VEG` @ surface | green vegetation fraction |
| `tozne` | Total ozone | DU | `TOZNE` @ entire atmosphere | f000 only — analysis step |

---

### Hour-sequence constants

```python
from noawclg import HOURS_16DAYS, HOURS_5DAYS_1H, HOURS_10DAYS_3H, HOURS_16DAYS_3H
```

| Constant | Range | Step | Total steps | Use case |
|----------|-------|------|-------------|----------|
| `HOURS_5DAYS_1H` | 0–120 h | 1 h | 121 | Hourly detail, short range |
| `HOURS_10DAYS_3H` | 0–240 h | 3 h | 81 | Medium range |
| `HOURS_16DAYS_3H` | 0–384 h | 3 h | 129 | Full extended range |
| `HOURS_16DAYS` | 0–120 h @ 6 h + 123–384 h @ 3 h | mixed | 107 | Legacy full run |

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
