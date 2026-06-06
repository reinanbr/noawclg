# Maps and globe plots

noawclg includes several plot functions that render data on geographic
projections — from flat Plate Carrée maps to interactive 3-D globe views.

## Available projections

| Function | Projection | Cartopy required |
|----------|-----------|-----------------|
| `plot_synoptic_map` | Plate Carrée (flat) | Yes |
| `plot_wind_speed_map` | Plate Carrée | No |
| `plot_cloud_map` | Plate Carrée | No |
| `plot_cape_map` | Plate Carrée | No |
| `plot_ocean_temp_map` | Plate Carrée | No |
| `plot_ssh_map` | Plate Carrée | No |
| `plot_ocean_currents` | Plate Carrée | No |
| `plot_globe` | **Orthographic** | **Yes** |

## 1 · Flat map — GFS wind speed

```python
from noawclg import load, auto_date
from plots import plot_wind_speed_map

date, cycle = auto_date(lag_days=1)
ds = load(date=date, cycle=cycle, lat=0, lon=-60,
          region={"toplat": 15, "bottomlat": -15,
                  "leftlon": -80, "rightlon": -30},
          hours=list(range(0, 49, 3)))

fig = plot_wind_speed_map(ds, hour=24, save_path="wind_map.png")
```

![Wind speed map](../_static/plots/plot_02_wind_speed_map.png)

## 2 · Flat map — GFS CAPE

```python
from plots import plot_cape_map

fig = plot_cape_map(ds, hour=24, save_path="cape_map.png")
```

![CAPE map](../_static/plots/plot_04_cape_map.png)

## 3 · Flat map — ocean temperature with Niño boxes

```python
from noawclg import get_ocean_temp
from plots import plot_ocean_temp_map

t200 = get_ocean_temp(2024, depth_m=200)

fig = plot_ocean_temp_map(
    t200.mean("time"),
    title="Mean Ocean Temperature at 200 m — 2024",
    nino_boxes=True,   # draws Niño 1+2 / 3 / 3.4 / 4 rectangles
    cmap="RdYlBu_r",
    save_path="t200_flat.png",
)
```

## 4 · Globe — orthographic projection

The `plot_globe` function uses cartopy's Orthographic projection to render
any 2-D DataArray on a globe.  Rotate the view by setting
`central_longitude` and `central_latitude`.

```python
from noawclg import get_ocean_temp
from plots import plot_globe

# Pacific-centred view (default)
t200 = get_ocean_temp(2024, depth_m=200)
fig  = plot_globe(
    t200.mean("time"),
    title="Ocean Temperature at 200 m — Pacific view",
    cmap="RdYlBu_r",
    central_longitude=-150,   # Pacific centre
    central_latitude=0,
    save_path="globe_pacific.png",
)
```

```python
# Atlantic-centred view
fig = plot_globe(
    t200.mean("time"),
    title="Ocean Temperature at 200 m — Atlantic view",
    cmap="RdYlBu_r",
    central_longitude=-30,
    central_latitude=20,
    save_path="globe_atlantic.png",
)
```

```python
# SSH anomaly — symmetric colour scale
from noawclg import get_ssh

ssh = get_ssh(2024)
fig = plot_globe(
    ssh.mean("time"),
    title="Mean SSH Anomaly — 2024",
    cmap="RdBu_r",
    central_longitude=-150,
    symmetric=True,   # force vmin = -vmax
    save_path="globe_ssh.png",
)
```

## 5 · Synoptic map (cartopy full-feature)

```python
from plots import plot_synoptic_map

fig = plot_synoptic_map(
    ds,
    hour=0,
    title="GFS T2m + MSLP + 10 m wind",
    save_path="synoptic.png",
)
```

The synoptic map draws:
- Filled T2m (2-m temperature)
- MSLP isobars every 4 hPa
- 10-m wind barbs

![Synoptic map](../_static/plots/plot_01_synoptic_map.png)

## 6 · 500 hPa geopotential + jet stream

```python
from plots import plot_500hpa_map

fig = plot_500hpa_map(ds_upper, hour=24, save_path="500hpa.png")
```

![500 hPa](../_static/plots/plot_14_500hpa_jet.png)

## Colourmaps reference

noawclg uses `cmocean` for ocean fields when available:

| Field | Colourmap |
|-------|-----------|
| Temperature | `cmocean.cm.thermal` / `RdYlBu_r` |
| Salinity | `cmocean.cm.haline` |
| SSH / anomaly | `RdBu_r` |
| Current speed | `plasma` |
| Precipitation | `Blues` |
| CAPE | `YlOrRd` |
| Wind speed | `viridis` |
