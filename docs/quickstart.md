# Quick start

## Download a GFS forecast

```python
from noawclg import load, auto_date

# Latest available GFS run (yesterday, 00z)
date, cycle = auto_date(lag_days=1)

# Download 0–48 h surface variables over NE Brazil
ds = load(
    date=date,
    cycle=cycle,
    lat=-3.7,
    lon=-38.5,
    region={"toplat": 5, "bottomlat": -15, "leftlon": -50, "rightlon": -30},
    hours=list(range(0, 49, 3)),
)
print(ds)
```

## Plot a GFS synoptic map

```python
from plots import plot_synoptic_map

fig = plot_synoptic_map(
    ds,
    hour=24,
    title="GFS T2m + MSLP · 24 h",
    save_path="synoptic_24h.png",
)
```

## Access ocean data (GODAS)

```python
from noawclg import get_ocean_temp, get_salinity, get_currents, get_ssh

# Temperature at 200 m in the equatorial Pacific
t200 = get_ocean_temp(2023, 2024, depth_m=200, region={
    "lat_min": -30, "lat_max": 30,
    "lon_min": 120, "lon_max": 290,
})
print(t200)   # xr.DataArray (time=24, lat, lon)

# Surface salinity
sal = get_salinity(2024, depth_m=5)

# Surface currents (u, v, speed)
curr = get_currents(2024, depth_m=5)
print(curr["speed"].mean())

# Sea surface height
ssh = get_ssh(2024)
```

## ENSO monitoring

```python
from noawclg import enso_summary, get_oni, classify_enso

# Full ENSO diagnostics 2015–2024
df = enso_summary(2015, 2024)
print(df.tail(12))
#             sst_nino34  anom_nino34   oni       phase
# month
# 2024-01-01    27.21        0.84      1.23    El Niño
# ...

# Plot the ONI time series
from plots import plot_enso_index
oni   = get_oni(2015, 2024)
phase = classify_enso(oni)
fig   = plot_enso_index(oni, phase, save_path="oni.png")
```

## Globe plot

```python
from noawclg import get_ocean_temp
from plots import plot_globe

t200 = get_ocean_temp(2024, depth_m=200)

fig = plot_globe(
    t200.mean("time"),
    title="Mean Ocean Temperature at 200 m — 2024",
    cmap="RdYlBu_r",
    central_longitude=-150,
    save_path="globe_t200.png",
)
```

## Thermocline cross-section

```python
from noawclg import get_godas
from plots import plot_thermocline_section

# Full water column for cross-section (no depth_m → all 40 levels)
pottmp = get_godas(2024, variable="pottmp",
                   region={"lat_min": -5, "lat_max": 5,
                           "lon_min": 120, "lon_max": 290})

fig = plot_thermocline_section(
    pottmp,
    lat=0.0,
    isotherm=20.0,
    save_path="thermocline.png",
)
```
