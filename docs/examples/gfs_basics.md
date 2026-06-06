# GFS weather forecasts

This example shows how to download GFS data and produce common meteorological
plots using noawclg.

## 1 · Download surface fields

```python
from noawclg import load, auto_date

date, cycle = auto_date(lag_days=1)   # latest complete GFS run

ds = load(
    date=date,
    cycle=cycle,
    lat=-3.7,           # Fortaleza, Brazil
    lon=-38.5,
    region={
        "toplat": 5, "bottomlat": -15,
        "leftlon": -50, "rightlon": -30,
    },
    hours=list(range(0, 121, 3)),   # 0–120 h every 3 h
)
```

The returned `xr.Dataset` contains:

| Variable | Description | Units |
|----------|-------------|-------|
| `t2m`    | 2-m temperature | K |
| `d2m`    | 2-m dew point  | K |
| `prmsl`  | Mean sea-level pressure | Pa |
| `u10` / `v10` | 10-m wind components | m/s |
| `gust`   | Surface wind gust | m/s |
| `prate`  | Precipitation rate | kg/m²/s |
| `r2`     | 2-m relative humidity | % |
| `tcc`    | Total cloud cover | 0–1 |
| `cape`   | CAPE | J/kg |

## 2 · Synoptic map

```python
from plots import plot_synoptic_map

fig = plot_synoptic_map(ds, hour=24, title="GFS 24 h — T2m + MSLP + Wind")
fig.savefig("synoptic.png", dpi=150)
```

![Synoptic map](../_static/plots/plot_01_synoptic_map.png)

## 3 · Forecast time series at a city

```python
from plots import plot_timeseries

fig = plot_timeseries(
    ds, city_name="Fortaleza CE",
    save_path="timeseries.png",
)
```

![Time series](../_static/plots/plot_06_timeseries.png)

## 4 · Wind rose

```python
from plots import plot_wind_rose

fig = plot_wind_rose(ds, city_name="Fortaleza CE", save_path="windrose.png")
```

![Wind rose](../_static/plots/plot_11_wind_rose.png)

## 5 · Upper-air vertical profile

```python
from noawclg import load

ds_upper = load(
    date=date, cycle=cycle,
    lat=-3.7, lon=-38.5,
    hours=[0, 24],
    level_type="pressure",
    levels=[200, 300, 500, 700, 850, 925, 1000],
)

from plots import plot_vertical_profiles

fig = plot_vertical_profiles(ds_upper, lat=-3.7, lon=-38.5)
```

![Vertical profiles](../_static/plots/plot_13_vertical_profiles.png)

## 6 · Hovmöller diagram

```python
from plots import plot_hovmoller

# Zonal Hovmöller: longitude vs time at a fixed latitude
fig = plot_hovmoller(ds, mode="lon", lat=-3.7, variable="prate")
```

![Hovmöller](../_static/plots/plot_15_hovmoller_lon_precip.png)

## 7 · Dashboard (4 panels)

```python
from plots import plot_dashboard

fig = plot_dashboard(ds, hour=24, city_name="Fortaleza CE")
```

![Dashboard](../_static/plots/plot_20_dashboard.png)
