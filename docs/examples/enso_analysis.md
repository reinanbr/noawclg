# ENSO monitoring and ocean analysis

noawclg provides direct access to NOAA GODAS and ERSST for subsurface ocean
monitoring and El Niño / La Niña detection.

## Data sources

| Source | Variables | Period | Resolution |
|--------|-----------|--------|------------|
| GODAS | `pottmp`, `salt`, `ucur`, `vcur`, `sshg` | 1980–present | Monthly, ~0.33°×1°, 40 levels |
| ERSST v5 | SST | 1854–present | Monthly, 2°×2° |

All data is served via **OPeNDAP** — no download required, lazy loading.

## 1 · ENSO summary table

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

## 2 · Oceanic Niño Index time series

```python
from noawclg import get_oni, classify_enso
from plots import plot_enso_index

oni   = get_oni(2000, 2024)
phase = classify_enso(oni)

fig = plot_enso_index(
    oni, phase,
    title="ONI — Oceanic Niño Index 2000–2024",
    save_path="oni.png",
)
```

The ONI (3-month running mean of Niño 3.4 anomaly) is the official NOAA
index for El Niño / La Niña:
- **El Niño** : ONI ≥ +0.5 °C for ≥ 5 consecutive 3-month seasons
- **La Niña** : ONI ≤ −0.5 °C for ≥ 5 consecutive 3-month seasons

## 3 · Ocean temperature at 200 m

```python
from noawclg import get_ocean_temp
from plots import plot_ocean_temp_map

t200 = get_ocean_temp(2024, depth_m=200)

# Niño 3.4 average
nino34 = t200.sel(lat=slice(-5, 5), lon=slice(190, 240))
print("Niño 3.4 T200 mean:", float(nino34.mean()), "°C")

# Map with Niño boxes
fig = plot_ocean_temp_map(
    t200.mean("time"),
    title="Mean Ocean Temperature at 200 m — 2024",
    save_path="t200_map.png",
)
```

## 4 · Thermocline depth (D20)

The depth of the 20 °C isotherm is the primary **dynamical** ENSO indicator.
When the thermocline deepens in the eastern Pacific, warm water accumulates
at depth → El Niño precursor.

```python
from noawclg import get_thermocline_depth

d20 = get_thermocline_depth(2024, region={
    "lat_min": -30, "lat_max": 30,
    "lon_min": 120, "lon_max": 290,
})
print(d20)   # DataArray (time, lat, lon) in metres
```

### Cross-section plot

```python
from noawclg import get_godas
from plots import plot_thermocline_section

# All 40 levels — no depth_m
pottmp = get_godas(2024, variable="pottmp", region={
    "lat_min": -5, "lat_max": 5,
    "lon_min": 120, "lon_max": 290,
})

fig = plot_thermocline_section(pottmp, lat=0.0, isotherm=20.0)
```

## 5 · Salinity

```python
from noawclg import get_salinity

sal = get_salinity(2024, depth_m=5, region={
    "lat_min": -30, "lat_max": 30,
    "lon_min": 120, "lon_max": 290,
})
print(sal)   # PSU, (time=12, lat, lon)
```

## 6 · Ocean currents

```python
from noawclg import get_currents

curr = get_currents(2024, depth_m=5)
print(curr["speed"].mean().item(), "m/s mean surface speed")

from plots import plot_ocean_currents
fig = plot_ocean_currents(
    curr["ucur"].isel(time=0),
    curr["vcur"].isel(time=0),
    title="Surface Ocean Currents — January 2024",
)
```

## 7 · Sea Surface Height

```python
from noawclg import get_ssh
from plots import plot_ssh_map

ssh = get_ssh(2024)
fig = plot_ssh_map(
    ssh.isel(time=0),
    title="Sea Surface Height — January 2024",
)
```

## 8 · Warm Water Volume (WWV)

WWV measures the total volume of water warmer than 20 °C in the equatorial
Pacific (5°S–5°N, 120°E–80°W) above 300 m.  It is a **leading indicator**
of ENSO: a large positive WWV anomaly typically precedes El Niño by 6–9 months.

```python
from noawclg import get_warm_water_volume

wwv = get_warm_water_volume(2020, 2024)
wwv.plot(title="Equatorial Pacific Warm Water Volume")
```

## 9 · Long-term SST with ERSST

ERSST extends the record back to 1854, enabling multi-decadal ENSO analysis
and accurate climatology computation.

```python
from noawclg import get_nino_anomaly

# 1854–2024 Niño 3.4 anomaly using ERSST
anom = get_nino_anomaly(1950, 2024, source="ersst",
                        clim_start=1991, clim_end=2020)
anom.plot(title="Niño 3.4 SST Anomaly 1950–2024 (ERSST v5)")
```
