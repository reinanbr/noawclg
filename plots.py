"""
noawclg/plots.py
================
Scientific plotting helpers for GFS data retrieved with noawclg.

All public functions accept an xarray.Dataset (as returned by
GFSDatasetManager or get_noaa_data) and return a matplotlib Figure.

Dependencies (install with `pip install noawclg[plots]`):
    matplotlib >= 3.8
    cartopy >= 0.22
    metpy >= 1.6
    windrose >= 1.9
    scipy >= 1.11
    seaborn >= 0.13

Memory design
-------------
Each plot function is self-contained and closes the Figure before returning
the saved path (when `save_path` is given). Use `generate_all` with
`save_and_close=True` (default) to process one plot at a time and free
memory immediately after saving, instead of holding all 8 figures in RAM.
"""

from __future__ import annotations

import os
import warnings
from typing import Generator, Sequence

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts and servers
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable

# ---------------------------------------------------------------------------
# Optional heavy deps — imported lazily per function
# ---------------------------------------------------------------------------
_CARTOPY_AVAILABLE = False
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _CARTOPY_AVAILABLE = True
except ImportError:
    pass

_METPY_AVAILABLE = False
try:
    import metpy.calc as mpcalc
    from metpy.plots import SkewT
    from metpy.units import units
    _METPY_AVAILABLE = True
except ImportError:
    pass

_WINDROSE_AVAILABLE = False
try:
    from windrose import WindroseAxes
    _WINDROSE_AVAILABLE = True
except ImportError:
    pass

_SCIPY_AVAILABLE = False
try:
    from scipy.ndimage import gaussian_filter
    from scipy.stats import gaussian_kde
    _SCIPY_AVAILABLE = True
except ImportError:
    pass

_SEABORN_AVAILABLE = False
try:
    import seaborn as sns
    _SEABORN_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f6",
    "axes.grid": True,
    "axes.grid.which": "major",
    "grid.color": "white",
    "grid.linewidth": 0.8,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.framealpha": 0.85,
}


def _apply_style() -> None:
    plt.rcParams.update(STYLE)


def _pick_hour(ds: xr.Dataset, hour: int) -> xr.Dataset:
    """Select a single forecast hour from the dataset."""
    try:
        return ds.sel(forecast_hour=hour)
    except KeyError:
        idx = list(ds["forecast_hour"].values).index(hour)
        return ds.isel(time=idx)


def _save_fig(fig: plt.Figure, path: str | None, dpi: int = 150) -> str | None:
    """
    Save *fig* to *path* (if given), then close it to free memory.

    Returns the absolute path on success, or None when path is None.
    """
    if path is None:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(path)


# ===========================================================================
# Plot 1 — Synoptic surface map
# ===========================================================================

def plot_synoptic_map(
    ds: xr.Dataset,
    hour: int = 0,
    scalar: str = "t2m",
    cmap: str = "RdBu_r",
    barb_density: int = 10,
    isobar_interval: float = 4.0,
    title: str | None = None,
    *,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | str | None:
    """
    Draw a synoptic surface map for a single forecast hour.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with at least `scalar`, ``prmsl``, ``u10``, ``v10``.
    hour : int
        Forecast hour to plot.
    scalar : str
        Variable used for the filled colour field (default ``"t2m"``).
    cmap : str
        Matplotlib colormap name.
    barb_density : int
        Sub-sampling stride for wind barbs (higher = fewer arrows).
    isobar_interval : float
        MSLP contour interval in hPa.
    title : str, optional
        Override the auto-generated title.
    save_path : str, optional
        If given, save the figure to this path, close it and return the path.
        Pass ``None`` to return the live Figure object instead.
    dpi : int
        Resolution used when saving.

    Returns
    -------
    plt.Figure | str | None
        Live Figure when ``save_path`` is None, saved path string otherwise.
    """
    if not _CARTOPY_AVAILABLE:
        raise ImportError("cartopy is required for plot_synoptic_map. "
                          "Install with: pip install noawclg[plots]")

    _apply_style()

    ds_h = _pick_hour(ds, hour)
    lats = ds_h["latitude"].values
    lons = ds_h["longitude"].values

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # --- scalar fill ---------------------------------------------------------
    field = ds_h[scalar].values
    vmin, vmax = np.nanpercentile(field, [2, 98])
    im = ax.contourf(
        lons, lats, field,
        levels=20, cmap=cmap, vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(), extend="both",
    )
    cbar = plt.colorbar(im, ax=ax, pad=0.03, shrink=0.85, aspect=30)
    cbar.set_label(
        f"{ds_h[scalar].attrs.get('long_name', scalar)} "
        f"({ds_h[scalar].attrs.get('units', '')})",
        fontsize=9,
    )

    # --- MSLP isobars --------------------------------------------------------
    if "prmsl" in ds_h:
        prmsl = ds_h["prmsl"].values
        p_levels = np.arange(
            np.floor(np.nanmin(prmsl) / isobar_interval) * isobar_interval,
            np.ceil(np.nanmax(prmsl) / isobar_interval) * isobar_interval + 1,
            isobar_interval,
        )
        cs = ax.contour(
            lons, lats, prmsl,
            levels=p_levels, colors="black", linewidths=0.6, alpha=0.7,
            transform=ccrs.PlateCarree(),
        )
        ax.clabel(cs, fmt="%d", fontsize=7, inline=True)

    # --- wind barbs ----------------------------------------------------------
    if "u10" in ds_h and "v10" in ds_h:
        s = barb_density
        ax.barbs(
            lons[::s], lats[::s],
            ds_h["u10"].values[::s, ::s],
            ds_h["v10"].values[::s, ::s],
            length=4.5, linewidth=0.7, pivot="middle",
            transform=ccrs.PlateCarree(),
        )

    # --- map decorations -----------------------------------------------------
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
    ax.add_feature(cfeature.LAND, facecolor="#eeebe3", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#dceef5", zorder=0)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = mticker.FuncFormatter(
        lambda v, _: f"{abs(v):.0f}°{'W' if v < 0 else 'E'}"
    )
    gl.yformatter = mticker.FuncFormatter(
        lambda v, _: f"{abs(v):.0f}°{'S' if v < 0 else 'N'}"
    )

    ax.set_title(title or f"GFS — {scalar.upper()} + MSLP + Wind  |  +{hour:03d} h")
    fig.tight_layout()

    return _save_fig(fig, save_path, dpi) if save_path else fig


# ===========================================================================
# Plot 2 — Temperature and precipitation time-series
# ===========================================================================

def plot_timeseries(
    ds: xr.Dataset,
    temp_var: str = "t2m",
    precip_var: str = "prate",
    accum_hours: float = 3.0,
    smooth_window: int | None = None,
    city_name: str = "",
    *,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | str | None:
    """
    Dual-axis time-series: temperature (left) and accumulated precipitation (right).

    Parameters
    ----------
    ds : xr.Dataset
        Point or spatially-averaged dataset with a ``time`` dimension.
    temp_var : str
        Temperature variable key.
    precip_var : str
        Precipitation rate variable key (kg m⁻² s⁻¹).
    accum_hours : float
        Time step in hours for rate → depth conversion.
    smooth_window : int, optional
        Rolling-mean window in steps applied to temperature.
    city_name : str
        Location label in the title.
    save_path : str, optional
        Save and close the figure; return the path. None returns the Figure.
    dpi : int
        Resolution used when saving.

    Returns
    -------
    plt.Figure | str | None
    """
    _apply_style()

    times = ds["time"].values
    temp  = ds[temp_var].values.squeeze()
    prate = ds[precip_var].values.squeeze() if precip_var in ds else None
    precip_mm = prate * 3600 * accum_hours if prate is not None else None

    if smooth_window and smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        temp_smooth = np.convolve(temp, kernel, mode="same")
    else:
        temp_smooth = temp

    fig, ax1 = plt.subplots(figsize=(12, 5))

    color_t = "#c0392b"
    ax1.plot(times, temp,        color=color_t, alpha=0.25, linewidth=0.8)
    ax1.plot(times, temp_smooth, color=color_t, linewidth=2.0, label="T2m (smoothed)")
    ax1.set_ylabel(
        f"Temperature ({ds[temp_var].attrs.get('units', '°C')})", color=color_t
    )
    ax1.tick_params(axis="y", labelcolor=color_t)
    ax1.set_xlabel("Forecast time (UTC)")

    if precip_mm is not None:
        ax2 = ax1.twinx()
        color_p = "#2980b9"
        ax2.bar(times, precip_mm, width=accum_hours / 24 * 0.8,
                color=color_p, alpha=0.55, label=f"Precip ({accum_hours}h accum.)")
        ax2.set_ylabel("Precipitation (mm)", color=color_p)
        ax2.tick_params(axis="y", labelcolor=color_p)
        ax2.set_ylim(bottom=0)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    else:
        ax1.legend(loc="upper left")

    ax1.set_title(
        "GFS Temperature & Precipitation Forecast"
        + (f" — {city_name}" if city_name else "")
    )
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    return _save_fig(fig, save_path, dpi) if save_path else fig


# ===========================================================================
# Plot 3 — Wind rose
# ===========================================================================

def plot_wind_rose(
    ds: xr.Dataset,
    u_var: str = "u10",
    v_var: str = "v10",
    bins: int = 5,
    calm_threshold: float = 0.5,
    cmap: str = "YlOrRd",
    *,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | str | None:
    """
    Polar wind-rose showing frequency and speed distribution.

    Parameters
    ----------
    ds : xr.Dataset
        Point dataset with wind components along ``time``.
    u_var, v_var : str
        Wind component variable keys.
    bins : int
        Number of speed bins.
    calm_threshold : float
        Winds below this speed (m s⁻¹) counted as calm and excluded.
    cmap : str
        Colormap for speed bins.
    save_path : str, optional
        Save and close the figure; return the path. None returns the Figure.
    dpi : int
        Resolution used when saving.

    Returns
    -------
    plt.Figure | str | None
    """
    if not _WINDROSE_AVAILABLE:
        raise ImportError("windrose is required for plot_wind_rose. "
                          "Install with: pip install noawclg[plots]")

    u = ds[u_var].values.squeeze()
    v = ds[v_var].values.squeeze()

    speed     = np.sqrt(u**2 + v**2)
    direction = (270 - np.degrees(np.arctan2(v, u))) % 360

    mask      = speed >= calm_threshold
    speed_f   = speed[mask]
    direction_f = direction[mask]
    bin_edges = np.linspace(0, np.nanpercentile(speed_f, 98), bins + 1)

    fig = plt.figure(figsize=(7, 7))
    rect = [0.1, 0.1, 0.8, 0.8]
    ax = WindroseAxes(fig, rect)
    fig.add_axes(ax)

    ax.bar(
        direction_f, speed_f,
        normed=True, bins=bin_edges,
        cmap=plt.get_cmap(cmap),
        opening=0.8, edgecolor="white", linewidth=0.4,
    )
    ax.set_legend(title="Speed (m s⁻¹)", loc="lower right", bbox_to_anchor=(1.25, 0))

    calm_pct = 100 * (1 - mask.mean())
    ax.set_title(
        f"Wind Rose — {u_var.upper()}/{v_var.upper()}\n"
        f"calm ({calm_threshold} m s⁻¹): {calm_pct:.1f}%",
        pad=20,
    )

    return _save_fig(fig, save_path, dpi) if save_path else fig


# ===========================================================================
# Plot 4 — Upper-air Skew-T log-P diagram
# ===========================================================================

def plot_skewt(
    ds: xr.Dataset,
    lat: float,
    lon: float,
    hour: int = 0,
    show_parcel: bool = True,
    title: str | None = None,
    *,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | str | None:
    """
    MetPy Skew-T log-P diagram at a single grid point and forecast hour.

    Parameters
    ----------
    ds : xr.Dataset
        Multi-level dataset containing ``t``, ``r``, ``u``, ``v``, ``gh``.
    lat, lon : float
        Coordinates of the sounding point.
    hour : int
        Forecast hour.
    show_parcel : bool
        Overlay the surface parcel ascent curve and CAPE/CIN shading.
    title : str, optional
        Override the auto-generated title.
    save_path : str, optional
        Save and close the figure; return the path. None returns the Figure.
    dpi : int
        Resolution used when saving.

    Returns
    -------
    plt.Figure | str | None
    """
    if not _METPY_AVAILABLE:
        raise ImportError("metpy is required for plot_skewt. "
                          "Install with: pip install noawclg[plots]")

    ds_h  = _pick_hour(ds, hour)
    ds_pt = ds_h.sel(latitude=lat, longitude=lon, method="nearest")

    p  = ds_pt["level"].values * units.hPa
    T  = ds_pt["t2m"].values * units.degC
    RH = ds_pt["r2"].values / 100.0
    Td = mpcalc.dewpoint_from_relative_humidity(T, RH)

    u_wind = ds_pt["u10"].values * units("m/s") if "u10" in ds_pt else None
    v_wind = ds_pt["v10"].values * units("m/s") if "v10" in ds_pt else None

    order = np.argsort(p.magnitude)[::-1]
    p, T, Td = p[order], T[order], Td[order]
    if u_wind is not None:
        u_wind, v_wind = u_wind[order], v_wind[order]

    fig  = plt.figure(figsize=(9, 11))
    skew = SkewT(fig, rotation=45)

    skew.plot(p, T,  "r", linewidth=2.0, label="Temperature")
    skew.plot(p, Td, "b", linewidth=2.0, label="Dewpoint")
    if u_wind is not None:
        skew.plot_barbs(p, u_wind, v_wind)

    if show_parcel:
        parcel_prof = mpcalc.parcel_profile(p, T[0], Td[0])
        skew.plot(p, parcel_prof, "k--", linewidth=1.5, label="Parcel")
        skew.shade_cape(p, T, parcel_prof)
        skew.shade_cin(p, T, parcel_prof)

    skew.ax.set_xlim(-40, 50)
    skew.ax.set_ylim(1025, 100)
    skew.ax.set_xlabel("Temperature (°C)")
    skew.ax.set_ylabel("Pressure (hPa)")
    skew.plot_dry_adiabats(linewidth=0.5, alpha=0.5)
    skew.plot_moist_adiabats(linewidth=0.5, alpha=0.5)
    skew.plot_mixing_lines(linewidth=0.5, alpha=0.5)
    skew.ax.legend(loc="upper left", fontsize=8)
    skew.ax.set_title(
        title or f"Skew-T log-P  |  ({lat:.2f}°, {lon:.2f}°)  +{hour:03d} h"
    )
    fig.tight_layout()

    return _save_fig(fig, save_path, dpi) if save_path else fig


# ===========================================================================
# Plot 5 — Hovmöller diagram
# ===========================================================================

def plot_hovmoller(
    ds: xr.Dataset,
    var: str,
    axis: str = "lon",
    lat_slice: tuple[float, float] = (-5.0, 5.0),
    lon_slice: tuple[float, float] = (-60.0, -30.0),
    cmap: str = "Blues",
    smooth_sigma: float = 1.0,
    *,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | str | None:
    """
    Time–longitude or time–latitude Hovmöller diagram.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with dimensions ``(time, latitude, longitude)``.
    var : str
        Variable to plot.
    axis : str
        ``"lon"`` or ``"lat"``.
    lat_slice, lon_slice : tuple
        Lat/lon bounds for spatial averaging.
    cmap : str
        Colormap.
    smooth_sigma : float
        Gaussian σ for smoothing (0 = none).
    save_path : str, optional
        Save and close the figure; return the path. None returns the Figure.
    dpi : int
        Resolution used when saving.

    Returns
    -------
    plt.Figure | str | None
    """
    da = ds[var]

    if axis == "lon":
        da_mean = da.sel(latitude=slice(*sorted(lat_slice, reverse=True))).mean("latitude")
        x       = da_mean["longitude"].values
        xlabel  = "Longitude (°)"
    else:
        da_mean = da.sel(longitude=slice(*sorted(lon_slice))).mean("longitude")
        x       = da_mean["latitude"].values
        xlabel  = "Latitude (°)"

    times = da_mean["time"].values
    data  = da_mean.values

    if smooth_sigma and _SCIPY_AVAILABLE:
        data = gaussian_filter(data, sigma=smooth_sigma)

    _apply_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.contourf(x, np.arange(len(times)), data, levels=20, cmap=cmap, extend="both")
    cbar = plt.colorbar(im, ax=ax, pad=0.02, aspect=30)
    cbar.set_label(
        f"{da.attrs.get('long_name', var)} ({da.attrs.get('units', '')})", fontsize=9
    )

    step = max(1, len(times) // 12)
    ax.set_yticks(np.arange(len(times))[::step])
    ax.set_yticklabels([str(t)[:13] for t in times[::step]], fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Forecast time (UTC)")
    ax.set_title(
        f"Hovmöller diagram — {var.upper()}"
        + (f" (avg {lat_slice[0]}°–{lat_slice[1]}° lat)" if axis == "lon"
           else f" (avg {lon_slice[0]}°–{lon_slice[1]}° lon)")
    )
    fig.tight_layout()

    return _save_fig(fig, save_path, dpi) if save_path else fig


# ===========================================================================
# Plot 6 — Vertical wind cross-section
# ===========================================================================

def plot_cross_section(
    ds: xr.Dataset,
    hour: int = 0,
    lat: float | None = None,
    lon: float | None = None,
    cmap: str = "plasma",
    levels: Sequence[float] | None = None,
    *,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | str | None:
    """
    Longitude–pressure or latitude–pressure cross-section of wind speed.

    Either ``lat`` or ``lon`` must be provided (not both).

    Parameters
    ----------
    ds : xr.Dataset
        Multi-level dataset with ``u``, ``v``, and ``t``.
    hour : int
        Forecast hour.
    lat : float, optional
        Fix latitude, slice along longitude.
    lon : float, optional
        Fix longitude, slice along latitude.
    cmap : str
        Colormap for wind speed shading.
    levels : sequence of float, optional
        Pressure levels (hPa) to include. Defaults to all in the dataset.
    save_path : str, optional
        Save and close the figure; return the path. None returns the Figure.
    dpi : int
        Resolution used when saving.

    Returns
    -------
    plt.Figure | str | None
    """
    if lat is None and lon is None:
        raise ValueError("Provide either lat or lon for the cross-section.")

    _apply_style()
    ds_h = _pick_hour(ds, hour)

    if lat is not None:
        ds_slice  = ds_h.sel(latitude=lat, method="nearest")
        x_vals    = ds_slice["longitude"].values
        x_label   = "Longitude (°)"
        slice_label = f"Lat = {lat:.1f}°"
    else:
        ds_slice  = ds_h.sel(longitude=lon, method="nearest")
        x_vals    = ds_slice["latitude"].values
        x_label   = "Latitude (°)"
        slice_label = f"Lon = {lon:.1f}°"

    p_all = ds_slice["level"].values
    if levels is not None:
        lev_mask = np.isin(p_all, levels)
        ds_slice = ds_slice.isel(level=lev_mask)
        p_all    = ds_slice["level"].values

    u_data = ds_slice["u10"].values
    v_data = ds_slice["v10"].values
    wspd   = np.sqrt(u_data**2 + v_data**2)

    theta = None
    if "t2m" in ds_slice:
        T_K   = ds_slice["t2m"].values + 273.15
        p_2d  = p_all[:, np.newaxis] * 100
        theta = T_K * (100000 / p_2d) ** 0.286

    fig, ax = plt.subplots(figsize=(13, 6))

    im = ax.contourf(x_vals, p_all, wspd, levels=20, cmap=cmap, extend="max")
    cbar = plt.colorbar(im, ax=ax, pad=0.02, aspect=30)
    cbar.set_label("Wind speed (m s⁻¹)", fontsize=9)

    if theta is not None:
        cs = ax.contour(x_vals, p_all, theta,
                        levels=12, colors="white", linewidths=0.7, alpha=0.6)
        ax.clabel(cs, fmt="%d K", fontsize=7)

    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_yticks(p_all[::2])
    ax.set_yticklabels([f"{p:.0f}" for p in p_all[::2]])
    ax.set_ylabel("Pressure (hPa)")
    ax.set_xlabel(x_label)
    ax.set_title(
        f"Vertical Cross-Section — Wind Speed + θ  |  {slice_label}  +{hour:03d} h"
    )
    fig.tight_layout()

    return _save_fig(fig, save_path, dpi) if save_path else fig


# ===========================================================================
# Plot 7 — Precipitation anomaly heatmap
# ===========================================================================

def plot_precip_heatmap(
    ds: xr.Dataset,
    precip_var: str = "prate",
    accum_hours: float = 3.0,
    cmap: str = "BuPu",
    threshold: float = 1.0,
    *,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | str | None:
    """
    Calendar-style daily precipitation heatmap.

    Parameters
    ----------
    ds : xr.Dataset
        Point dataset with ``precip_var`` along ``time``.
    precip_var : str
        Precipitation rate variable key.
    accum_hours : float
        Time step in hours for rate → depth conversion.
    cmap : str
        Colormap.
    threshold : float
        Annotate cells with accumulation above this value (mm).
    save_path : str, optional
        Save and close the figure; return the path. None returns the Figure.
    dpi : int
        Resolution used when saving.

    Returns
    -------
    plt.Figure | str | None
    """
    import pandas as pd

    _apply_style()

    rate  = ds[precip_var].values.squeeze()
    times = ds["time"].values
    depth = rate * 3600 * accum_hours

    df    = pd.Series(depth, index=pd.DatetimeIndex(times))
    daily = df.resample("1D").sum()

    daily_df = daily.to_frame(name="precip").assign(
        week=lambda d: d.index.isocalendar().week.values,
        dow=lambda d: d.index.dayofweek,
        label=lambda d: d.index.strftime("%d\n%b"),
    )

    weeks   = sorted(daily_df["week"].unique())
    n_weeks = len(weeks)
    matrix  = np.full((n_weeks, 7), np.nan)
    labels  = [[""] * 7 for _ in range(n_weeks)]

    for wi, wk in enumerate(weeks):
        sub = daily_df[daily_df["week"] == wk]
        for _, row in sub.iterrows():
            matrix[wi, int(row["dow"])] = row["precip"]
            labels[wi][int(row["dow"])] = row["label"]

    fig, ax = plt.subplots(figsize=(max(8, n_weeks * 0.9), 5))
    vmax = np.nanmax(matrix) if np.nanmax(matrix) > 0 else 1
    im   = ax.imshow(matrix.T, aspect="auto", cmap=cmap, vmin=0, vmax=vmax, origin="upper")
    plt.colorbar(im, ax=ax, orientation="vertical",
                 label="Precipitation (mm day⁻¹)", shrink=0.8)

    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(n_weeks))
    ax.set_xticklabels([f"W{w}" for w in weeks], fontsize=8)
    ax.set_xlabel("ISO week")

    for wi in range(n_weeks):
        for di in range(7):
            val = matrix[wi, di]
            if not np.isnan(val) and val >= threshold:
                ax.text(wi, di, f"{val:.1f}", ha="center", va="center",
                        fontsize=7, color="white" if val > vmax * 0.6 else "black")

    ax.set_title(f"Daily Precipitation Heatmap — {precip_var.upper()}")
    fig.tight_layout()

    return _save_fig(fig, save_path, dpi) if save_path else fig


# ===========================================================================
# Plot 8 — Ensemble (multi-variable) spread matrix
# ===========================================================================

def plot_spread_matrix(
    ds: xr.Dataset,
    vars: list[str],
    color_by: str = "time",
    cmap: str = "viridis",
    kde_bw: str | float = "scott",
    *,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure | str | None:
    """
    Pair-plot scatter matrix comparing multiple surface variables.

    Parameters
    ----------
    ds : xr.Dataset
        Point dataset with variables along ``time``.
    vars : list of str
        2–6 variable keys to compare.
    color_by : str
        Colour points by ``"time"``, ``"hour"`` or a variable key.
    cmap : str
        Colormap for the colour encoding.
    kde_bw : str or float
        Bandwidth for the diagonal KDE.
    save_path : str, optional
        Save and close the figure; return the path. None returns the Figure.
    dpi : int
        Resolution used when saving.

    Returns
    -------
    plt.Figure | str | None
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy is required for plot_spread_matrix. "
                          "Install with: pip install noawclg[plots]")

    _apply_style()

    n = len(vars)
    if n < 2 or n > 6:
        raise ValueError("vars must contain 2–6 variable keys.")

    data  = {v: ds[v].values.squeeze() for v in vars}
    times = ds["time"].values

    if color_by == "time":
        c_vals  = np.arange(len(times), dtype=float)
        c_label = "Forecast step"
    elif color_by == "hour":
        import pandas as pd
        c_vals  = pd.DatetimeIndex(times).hour.to_numpy(dtype=float)
        c_label = "Hour of day (UTC)"
    elif color_by in data:
        c_vals  = data[color_by]
        c_label = color_by
    else:
        c_vals  = np.arange(len(times), dtype=float)
        c_label = "Forecast step"

    norm   = mcolors.Normalize(vmin=c_vals.min(), vmax=c_vals.max())
    sm     = ScalarMappable(norm=norm, cmap=cmap)
    colors = sm.to_rgba(c_vals)

    fig, axes = plt.subplots(n, n, figsize=(2.8 * n, 2.8 * n))

    for row, v_row in enumerate(vars):
        for col, v_col in enumerate(vars):
            ax = axes[row, col]
            ax.set_facecolor("#f8f8f6")

            if row == col:
                x_arr = data[v_row]
                valid = x_arr[np.isfinite(x_arr)]
                if len(valid) > 1:
                    kde    = gaussian_kde(valid, bw_method=kde_bw)
                    x_grid = np.linspace(valid.min(), valid.max(), 200)
                    ax.fill_between(x_grid, kde(x_grid), alpha=0.4, color="#2980b9")
                    ax.plot(x_grid, kde(x_grid), color="#2980b9", linewidth=1.5)
                ax.set_yticks([])
            else:
                ax.scatter(data[v_col], data[v_row], c=colors, s=12,
                           alpha=0.7, edgecolors="none")

            if col == 0:
                ax.set_ylabel(v_row, fontsize=9)
            if row == n - 1:
                ax.set_xlabel(v_col, fontsize=9)

            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
            ax.tick_params(labelsize=7)

    cbar = fig.colorbar(sm, ax=axes, shrink=0.4, aspect=25, pad=0.03)
    cbar.set_label(c_label, fontsize=9)

    fig.suptitle("Multi-variable Spread Matrix — GFS Forecast",
                 fontsize=13, fontweight="semibold", y=1.01)
    fig.tight_layout()

    return _save_fig(fig, save_path, dpi) if save_path else fig


# ===========================================================================
# generate_all — load → plot → save → free, one at a time
# ===========================================================================

def generate_all(
    date: str,
    cycle: str = "00",
    lat: float = -3.73,
    lon: float = -38.52,
    place: str = "Fortaleza, Brazil",
    region: dict | None = None,
    hours: list[int] | None = None,
    output_dir: str = ".",
    dpi: int = 150,
    surface_keys: list[str] | None = None,
    upper_keys:   list[str] | None = None,
    point_keys:   list[str] | None = None,
) -> dict[str, str]:
    """
    Generate all 8 plots with minimal RAM usage.

    Each plot follows a strict cycle:
      1. Load only the variables that plot needs.
      2. Render and save the figure to disk.
      3. Close the figure (``plt.close``).
      4. Delete the dataset and call ``gc.collect()``.

    No two datasets are alive in memory at the same time.

    Parameters
    ----------
    date : str
        Model run date in ``DD/MM/YYYY`` format.
    cycle : str
        Model cycle: ``"00"``, ``"06"``, ``"12"`` or ``"18"``.
    lat : float
        Latitude for Skew-T and cross-section.
    lon : float
        Longitude for cross-section.
    place : str
        Place name geocoded for point-based plots (2, 3, 7, 8).
    region : dict, optional
        Bounding box for surface/upper downloads. ``None`` = global.
    hours : list[int], optional
        Forecast hours to request. Defaults to 0–384 every 3 h.
    output_dir : str
        Directory where PNG files are saved.
    dpi : int
        Resolution for saved figures.
    surface_keys : list[str], optional
        Override the default surface variable list.
    upper_keys : list[str], optional
        Override the default upper-air variable list.
    point_keys : list[str], optional
        Override the default point variable list.

    Returns
    -------
    dict[str, str]
        ``{plot_name: absolute_saved_path}`` for every successful plot.
    """
    import gc
    from noawclg import load
    from noawclg.main import get_noaa_data

    os.makedirs(output_dir, exist_ok=True)

    if hours is None:
        hours = list(range(0, 121, 3)) + list(range(123, 385, 3))

    _surface_keys = surface_keys or ["t2m", "prate", "prmsl", "u10", "v10"]
    _upper_keys   = upper_keys   or ["t2m", "r2", "u10", "v10", "gh"]
    _point_keys   = point_keys   or ["t2m", "prate", "u10", "v10", "r2", "gust"]

    saved: dict[str, str] = {}

    def _ok(name: str, result: str | None) -> None:
        if result:
            saved[name] = result
            print(f"  ✓  {name}  →  {result}")
        else:
            print(f"  ✗  {name}  (skipped or failed)")

    def _load_surface() -> xr.Dataset:
        return load(date=date, cycle=cycle, keys=_surface_keys,
                    hours=hours, region=region)

    def _load_upper() -> xr.Dataset:
        return load(date=date, cycle=cycle, keys=_upper_keys, hours=hours)

    def _load_point() -> xr.Dataset:
        noaa = get_noaa_data(date=date, cycle=cycle,
                             keys=_point_keys, hours=hours)
        return noaa.get_data_from_place(place)._ds

    # ------------------------------------------------------------------
    # Plot 1 — Synoptic map  (needs: surface)
    # ------------------------------------------------------------------
    print("[ 1/8 ] Synoptic map …")
    try:
        ds = _load_surface()
        hours_avail = sorted(ds["forecast_hour"].values.tolist())
        mid = hours_avail[len(hours_avail) // 2]
        result = plot_synoptic_map(
            ds, hour=mid,
            save_path=os.path.join(output_dir, "01_synoptic_map.png"), dpi=dpi,
        )
        _ok("01_synoptic_map", result)
    except Exception as e:
        warnings.warn(f"Plot 1 failed: {e}")
        _ok("01_synoptic_map", None)
        hours_avail = hours  # fallback so plots 5+ have a reference
    finally:
        try:
            del ds
        except NameError:
            pass
        gc.collect()

    # ------------------------------------------------------------------
    # Plot 2 — Temperature & precipitation time-series  (needs: point)
    # ------------------------------------------------------------------
    print("[ 2/8 ] Time-series …")
    try:
        ds = _load_point()
        result = plot_timeseries(
            ds,
            save_path=os.path.join(output_dir, "02_timeseries.png"), dpi=dpi,
        )
        _ok("02_timeseries", result)
    except Exception as e:
        warnings.warn(f"Plot 2 failed: {e}")
        _ok("02_timeseries", None)
    finally:
        try:
            del ds
        except NameError:
            pass
        gc.collect()

    # ------------------------------------------------------------------
    # Plot 3 — Wind rose  (needs: point  with u10 + v10)
    # ------------------------------------------------------------------
    print("[ 3/8 ] Wind rose …")
    try:
        ds = _load_point()
        if "u10" in ds and "v10" in ds:
            result = plot_wind_rose(
                ds,
                save_path=os.path.join(output_dir, "03_wind_rose.png"), dpi=dpi,
            )
            _ok("03_wind_rose", result)
        else:
            warnings.warn("Plot 3 skipped: u10/v10 not available in point dataset.")
            _ok("03_wind_rose", None)
    except Exception as e:
        warnings.warn(f"Plot 3 failed: {e}")
        _ok("03_wind_rose", None)
    finally:
        try:
            del ds
        except NameError:
            pass
        gc.collect()

    # ------------------------------------------------------------------
    # Plot 4 — Skew-T  (needs: upper)
    # ------------------------------------------------------------------
    print("[ 4/8 ] Skew-T …")
    try:
        ds = _load_upper()
        first_hour = sorted(ds["forecast_hour"].values.tolist())[0]
        result = plot_skewt(
            ds, lat=lat, lon=lon, hour=first_hour,
            save_path=os.path.join(output_dir, "04_skewt.png"), dpi=dpi,
        )
        _ok("04_skewt", result)
    except Exception as e:
        warnings.warn(f"Plot 4 failed: {e}")
        _ok("04_skewt", None)
    finally:
        try:
            del ds
        except NameError:
            pass
        gc.collect()

    # ------------------------------------------------------------------
    # Plot 5 — Hovmöller  (needs: surface)
    # ------------------------------------------------------------------
    print("[ 5/8 ] Hovmöller …")
    try:
        ds = _load_surface()
        var_hov = "prate" if "prate" in ds else list(ds.data_vars)[0]
        result = plot_hovmoller(
            ds, var=var_hov,
            save_path=os.path.join(output_dir, "05_hovmoller.png"), dpi=dpi,
        )
        _ok("05_hovmoller", result)
    except Exception as e:
        warnings.warn(f"Plot 5 failed: {e}")
        _ok("05_hovmoller", None)
    finally:
        try:
            del ds
        except NameError:
            pass
        gc.collect()

    # ------------------------------------------------------------------
    # Plot 6 — Cross-section  (needs: upper)
    # ------------------------------------------------------------------
    print("[ 6/8 ] Cross-section …")
    try:
        ds = _load_upper()
        first_hour = sorted(ds["forecast_hour"].values.tolist())[0]
        result = plot_cross_section(
            ds, hour=first_hour, lat=lat,
            save_path=os.path.join(output_dir, "06_cross_section.png"), dpi=dpi,
        )
        _ok("06_cross_section", result)
    except Exception as e:
        warnings.warn(f"Plot 6 failed: {e}")
        _ok("06_cross_section", None)
    finally:
        try:
            del ds
        except NameError:
            pass
        gc.collect()

    # ------------------------------------------------------------------
    # Plot 7 — Precip heatmap  (needs: point with prate)
    # ------------------------------------------------------------------
    print("[ 7/8 ] Precipitation heatmap …")
    try:
        ds = _load_point()
        if "prate" in ds:
            result = plot_precip_heatmap(
                ds,
                save_path=os.path.join(output_dir, "07_precip_heatmap.png"), dpi=dpi,
            )
            _ok("07_precip_heatmap", result)
        else:
            warnings.warn("Plot 7 skipped: prate not available in point dataset.")
            _ok("07_precip_heatmap", None)
    except Exception as e:
        warnings.warn(f"Plot 7 failed: {e}")
        _ok("07_precip_heatmap", None)
    finally:
        try:
            del ds
        except NameError:
            pass
        gc.collect()

    # ------------------------------------------------------------------
    # Plot 8 — Spread matrix  (needs: point)
    # ------------------------------------------------------------------
    print("[ 8/8 ] Spread matrix …")
    try:
        ds = _load_point()
        available = [v for v in ["t2m", "r2", "prate", "gust", "tcc"] if v in ds]
        if len(available) >= 2:
            result = plot_spread_matrix(
                ds, vars=available[:5],
                save_path=os.path.join(output_dir, "08_spread_matrix.png"), dpi=dpi,
            )
            _ok("08_spread_matrix", result)
        else:
            warnings.warn("Plot 8 skipped: fewer than 2 compatible variables in point dataset.")
            _ok("08_spread_matrix", None)
    except Exception as e:
        warnings.warn(f"Plot 8 failed: {e}")
        _ok("08_spread_matrix", None)
    finally:
        try:
            del ds
        except NameError:
            pass
        gc.collect()

    print(f"\nTotal de plots salvos: {len(saved)}/{8}")
    return saved


# ===========================================================================
# Ocean / ENSO plotting
# ===========================================================================

def plot_enso_index(
    oni: xr.DataArray,   # accepts pd.Series or any array-like with DatetimeIndex
    phase=None,
    title: str = "Oceanic Niño Index (ONI)",
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Time series of ONI with El Niño / La Niña shading.

    Parameters
    ----------
    oni   : Monthly ONI values (°C) — pd.Series with DatetimeIndex.
    phase : Optional ENSO phase series from ``classify_enso``.  When
            provided, positive anomalies are shaded red (El Niño) and
            negative anomalies are shaded blue (La Niña).
    """
    import pandas as pd

    fig, ax = plt.subplots(figsize=(12, 4))

    t = oni.index
    v = oni.values

    # Background phase shading
    ax.fill_between(t, v, 0,
                    where=v >= 0.5,  color="#d73027", alpha=0.35, label="El Niño")
    ax.fill_between(t, v, 0,
                    where=v <= -0.5, color="#4575b4", alpha=0.35, label="La Niña")
    ax.fill_between(t, v, 0,
                    where=(v > -0.5) & (v < 0.5),
                    color="#e0e0e0", alpha=0.40, label="Neutral")

    ax.plot(t, v, color="#333333", linewidth=1.4, zorder=3)
    ax.axhline(0.5,  color="#d73027", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(-0.5, color="#4575b4", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(0,    color="#444444", linewidth=0.5)

    ax.set_ylabel("SST Anomaly (°C)", fontsize=11)
    ax.set_xlabel("")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.85, fontsize=9)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_ocean_temp_map(
    da: xr.DataArray,
    title: str = "",
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    nino_boxes: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Global ocean temperature (or any 2-D ocean field) on a flat map.

    Optionally overlays the four standard Niño monitoring boxes.

    Parameters
    ----------
    da         : 2-D DataArray ``(lat, lon)`` or ``(time, lat, lon)``
                 (first time step used if 3-D).
    nino_boxes : Draw Niño 1+2 / 3 / 3.4 / 4 bounding rectangles.
    """
    import matplotlib.patches as mpatches

    if "time" in da.dims:
        da = da.isel(time=0)

    lons = da["lon"].values
    lats = da["lat"].values
    data = da.values

    # Convert 0-360 → -180/180 if needed for display
    if lons.max() > 180:
        shift = lons > 180
        lons  = lons.copy()
        lons[shift] -= 360

    sort_idx = np.argsort(lons)
    lons = lons[sort_idx]
    data = data[:, sort_idx] if data.ndim == 2 else data

    fig, ax = plt.subplots(figsize=(14, 6))
    vmin = vmin or np.nanpercentile(data, 2)
    vmax = vmax or np.nanpercentile(data, 98)

    pcm = ax.pcolormesh(lons, lats, data,
                        cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    cb = fig.colorbar(pcm, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label(da.attrs.get("units", ""), fontsize=10)

    # Niño boxes — convert to -180/180
    if nino_boxes:
        _BOX_COLORS = {"1+2": "#e41a1c", "3": "#ff7f00",
                       "3.4": "#984ea3", "4": "#377eb8"}
        try:
            from noawclg.ocean import NINO_BOXES
            for name, b in NINO_BOXES.items():
                lon0 = b["lon"][0] - 360 if b["lon"][0] > 180 else b["lon"][0]
                lon1 = b["lon"][1] - 360 if b["lon"][1] > 180 else b["lon"][1]
                rect = mpatches.Rectangle(
                    (lon0, b["lat"][0]),
                    lon1 - lon0, b["lat"][1] - b["lat"][0],
                    linewidth=1.6, edgecolor=_BOX_COLORS[name],
                    facecolor="none", zorder=4,
                )
                ax.add_patch(rect)
                ax.text(lon0 + 1, b["lat"][1] + 0.8, f"Niño {name}",
                        color=_BOX_COLORS[name], fontsize=7.5, fontweight="bold")
        except ImportError:
            pass

    ax.set_xlim(-180, 180)
    ax.set_ylim(lats.min(), lats.max())
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title or da.attrs.get("long_name", "Ocean Temperature"), fontsize=12)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.grid(linestyle="--", alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_thermocline_section(
    pottmp: xr.DataArray,
    lat: float = 0.0,
    isotherm: float = 20.0,
    title: str = "",
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Longitude–depth cross-section of ocean temperature along a latitude.

    Draws the *isotherm* °C contour (default 20 °C — the thermocline proxy)
    as a bold line over the filled temperature field.

    Parameters
    ----------
    pottmp   : Temperature DataArray with dims ``(time, level, lat, lon)``
               or ``(level, lat, lon)``.  First time step used if 4-D.
    lat      : Latitude of the cross-section (degrees).
    isotherm : Temperature of the isotherm contour to highlight (°C).
    """
    if "time" in pottmp.dims:
        pottmp = pottmp.isel(time=0)

    # Select nearest latitude
    section = pottmp.sel(lat=lat, method="nearest")   # (level, lon)
    levels  = section["level"].values
    lons    = section["lon"].values
    data    = section.values

    # 0-360 → -180/180
    if lons.max() > 180:
        shift    = lons > 180
        lons     = lons.copy();  lons[shift] -= 360
        sort_idx = np.argsort(lons)
        lons     = lons[sort_idx]
        data     = data[:, sort_idx]

    fig, ax = plt.subplots(figsize=(13, 5))
    vmin = np.nanpercentile(data, 2)
    vmax = np.nanpercentile(data, 98)

    pcm = ax.contourf(lons, levels, data,
                      levels=20, cmap="RdYlBu_r",
                      vmin=vmin, vmax=vmax, extend="both")
    fig.colorbar(pcm, ax=ax, pad=0.02, label="Temperature (°C)")

    # Thermocline isotherm
    cs = ax.contour(lons, levels, data,
                    levels=[isotherm], colors="black", linewidths=2.0)
    ax.clabel(cs, fmt=f"{isotherm}°C", fontsize=9, inline=True)

    ax.set_ylim(levels.max(), levels.min())   # depth increases downward
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Depth (m)")
    lat_lbl = f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}"
    ax.set_title(
        title or f"Ocean Temperature Cross-Section at {lat_lbl}",
        fontsize=12,
    )
    ax.grid(linestyle="--", alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_ssh_map(
    ssh: xr.DataArray,
    title: str = "Sea Surface Height Anomaly",
    cmap: str = "RdBu_r",
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Global SSH (or SSH anomaly) map with symmetric colour scale.

    SSH is a key ENSO indicator: positive anomaly in the east-central Pacific
    signals warm water downwelling (El Niño); negative signals upwelling
    (La Niña).
    """
    if "time" in ssh.dims:
        ssh = ssh.isel(time=0)

    lons = ssh["lon"].values.copy()
    lats = ssh["lat"].values
    data = ssh.values

    if lons.max() > 180:
        lons[lons > 180] -= 360
        sort_idx = np.argsort(lons)
        lons = lons[sort_idx]
        data = data[:, sort_idx]

    absmax = np.nanpercentile(np.abs(data[np.isfinite(data)]), 98)

    fig, ax = plt.subplots(figsize=(14, 6))
    pcm = ax.pcolormesh(lons, lats, data,
                        cmap=cmap, vmin=-absmax, vmax=absmax, shading="auto")
    fig.colorbar(pcm, ax=ax, pad=0.02, shrink=0.85, label="SSH (m)")

    ax.set_xlim(-180, 180)
    ax.set_ylim(lats.min(), lats.max())
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=12)
    ax.grid(linestyle="--", alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_globe(
    da: xr.DataArray,
    title: str = "",
    cmap: str = "RdBu_r",
    central_longitude: float = -150.0,
    central_latitude: float = 0.0,
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Plot a 2-D DataArray on an orthographic globe projection.

    Requires cartopy.

    Parameters
    ----------
    da               : 2-D DataArray ``(lat, lon)`` or ``(time, lat, lon)``.
    central_longitude: Longitude at the centre of the globe view.
    central_latitude : Latitude at the centre of the globe view.
    symmetric        : Force symmetric colour scale (``vmin = -vmax``).
    """
    if not _CARTOPY_AVAILABLE:
        raise ImportError("cartopy is required for plot_globe — "
                          "install with: pip install cartopy")

    if "time" in da.dims:
        da = da.isel(time=0)

    data = da.values
    lons = da["lon"].values
    lats = da["lat"].values

    # Determine colour limits
    finite = data[np.isfinite(data)]
    _vmin = vmin if vmin is not None else np.percentile(finite, 2)
    _vmax = vmax if vmax is not None else np.percentile(finite, 98)
    if symmetric:
        absmax = max(abs(_vmin), abs(_vmax))
        _vmin, _vmax = -absmax, absmax

    fig = plt.figure(figsize=(9, 9))
    proj = ccrs.Orthographic(
        central_longitude=central_longitude,
        central_latitude=central_latitude,
    )
    ax = fig.add_subplot(111, projection=proj)
    ax.set_global()

    # Filled contour / pcolormesh on globe
    lon2d, lat2d = np.meshgrid(lons, lats)
    pcm = ax.pcolormesh(
        lon2d, lat2d, data,
        transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=_vmin, vmax=_vmax,
        shading="auto",
    )

    ax.add_feature(cfeature.LAND,      facecolor="#d0d0d0", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6,        zorder=3)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, alpha=0.5, zorder=3)
    ax.gridlines(color="gray", linestyle="--", alpha=0.4, linewidth=0.5)

    fig.colorbar(pcm, ax=ax, orientation="horizontal",
                 fraction=0.046, pad=0.04,
                 label=da.attrs.get("units", ""))
    ax.set_title(title or da.attrs.get("long_name", ""), fontsize=12,
                 fontweight="bold", pad=14)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_ocean_currents(
    ucur: xr.DataArray,
    vcur: xr.DataArray,
    title: str = "Ocean Surface Currents",
    quiver_step: int = 6,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Speed-filled map of ocean currents with quiver arrows.

    Parameters
    ----------
    ucur, vcur   : Eastward/northward current DataArrays ``(lat, lon)``.
    quiver_step  : Subsample step for quiver arrows (reduce arrow density).
    """
    if "time" in ucur.dims:
        ucur = ucur.isel(time=0)
    if "time" in vcur.dims:
        vcur = vcur.isel(time=0)

    lons  = ucur["lon"].values.copy()
    lats  = ucur["lat"].values
    u     = ucur.values
    v     = vcur.values
    speed = np.sqrt(u**2 + v**2)

    if lons.max() > 180:
        lons[lons > 180] -= 360
        sort_idx = np.argsort(lons)
        lons = lons[sort_idx]
        u    = u[:, sort_idx]
        v    = v[:, sort_idx]
        speed = speed[:, sort_idx]

    fig, ax = plt.subplots(figsize=(14, 6))
    pcm = ax.pcolormesh(lons, lats, speed,
                        cmap="plasma", vmin=0,
                        vmax=np.nanpercentile(speed[np.isfinite(speed)], 97),
                        shading="auto")
    fig.colorbar(pcm, ax=ax, label="Speed (m/s)", pad=0.02)

    # Quiver arrows (subsampled)
    qs  = quiver_step
    lon_q, lat_q = np.meshgrid(lons[::qs], lats[::qs])
    ax.quiver(lon_q, lat_q, u[::qs, ::qs], v[::qs, ::qs],
              scale=10, width=0.002, color="white", alpha=0.7)

    ax.set_xlim(-180, 180)
    ax.set_ylim(lats.min(), lats.max())
    ax.axhline(0, color="k", linewidth=0.4, linestyle="--", alpha=0.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=12)
    ax.grid(linestyle="--", alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


# ===========================================================================
# Usage example (run as script)
# ===========================================================================

if __name__ == "__main__":
    from noawclg.gfs_dataset import auto_date
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    DATE, CYCLE = auto_date(lag_days=1)
    print(f"Usando GFS {DATE} ciclo {CYCLE}z")

    saved = generate_all(
        date=DATE,
        cycle=CYCLE,
        lat=-3.73,
        lon=-38.52,
        place="Fortaleza, Brazil",
        region={"toplat": 5, "bottomlat": -35, "leftlon": -75, "rightlon": -34},
        hours=list(range(0, 121, 3)) + list(range(123, 385, 3)),
        output_dir="./gfs_plots",
        dpi=150,
    )