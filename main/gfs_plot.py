"""
GFS Plot Suite  (v2 — animado + estático)
==========================================
Gera plots estáticos e animações (GIF / MP4) para as projeções
Orthographic e Mercator a partir de dados GFS baixados via GFSDatasetManager.

Modos de saída
--------------
  --mode static   → PNGs individuais por hora  (padrão)
  --mode gif      → GIF animado por variável
  --mode mp4      → MP4 animado por variável
  --mode panel    → painel 2×3 estático (hora única)
  --mode all      → tudo acima

Variáveis plotadas
------------------
  t2m    – Temperatura a 2 m  (°C)
  prmsl  – Pressão ao nível do mar (hPa)
  prate  – Precipitação  (mm h⁻¹)
  wind   – Velocidade + barbs (u10 + v10)

Uso rápido
----------
    python gfs_plot.py                          # static, hora 24
    python gfs_plot.py --mode gif --hours 0 24 48
    python gfs_plot.py --mode mp4 --hours $(seq 0 6 120 | tr '\\n' ' ')
    python gfs_plot.py --mode all --hours 0 12 24 36 48

Dependências
------------
    pip install cartopy matplotlib cmocean scipy requests cfgrib xarray
               numpy netCDF4 zarr eccodes tqdm pillow imageio[ffmpeg]
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# ── importar do módulo de dados ───────────────────────────────────────────────
from noawclg.base import GFSDatasetManager, VARIABLES  # noqa: E402

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Configuração padrão
# ══════════════════════════════════════════════════════════════════════════════

DATE       = "20260403"
CYCLE      = "06"
# Paths absolutos relativos ao diretório do script — evita bugs de CWD
_HERE      = Path(__file__).resolve().parent
OUTPUT_DIR = _HERE / 'gfs_output'
FIG_DIR    = _HERE / 'gfs_figs'

REGION_BR = {
    "toplat":    5,
    "bottomlat": -35,
    "leftlon":   -75,
    "rightlon":  -34,
}

# ══════════════════════════════════════════════════════════════════════════════
# Paletas
# ══════════════════════════════════════════════════════════════════════════════

TEMP_CMAP   = cmocean.cm.thermal
TEMP_LEVELS = np.arange(-20, 50, 2)

PRES_CMAP   = plt.cm.RdBu_r
PRES_LEVELS = np.arange(960, 1040, 2)

# Precipitação: colormap de alto contraste para valores baixos
# Os níveis são log-espaçados para realçar chuviscos (0.05 mm/h) até
# eventos extremos (64 mm/h) sem deixar o mapa todo escuro.
_prate_colors = [
    "#0d1117",  # fundo — sem chuva
    "#0a2a4a",  # traço
    "#0e4d92",  # fraco
    "#1565c0",
    "#1976d2",
    "#29b6f6",  # moderado
    "#00e5ff",
    "#69f0ae",  # forte
    "#ffeb3b",
    "#ff9800",
    "#f44336",  # intenso
    "#b71c1c",  # extremo
]
PRATE_CMAP = LinearSegmentedColormap.from_list("prate", _prate_colors, N=512)
# Níveis log-espaçados: 0.01 → 64 mm/h
PRATE_LEVELS = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]

SPD_CMAP      = cmocean.cm.speed
SPD_LEVELS_G  = np.arange(0, 26, 1)
SPD_LEVELS_BR = np.arange(0, 20, 0.5)

# Estilo escuro global
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#0d1117",
    "text.color":       "#e6edf3",
    "axes.labelcolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "font.family":      "monospace",
    "axes.titlepad":    10,
})

# ══════════════════════════════════════════════════════════════════════════════
# Helpers de mapa
# ══════════════════════════════════════════════════════════════════════════════

def _add_features(ax: plt.Axes, lw: float = 0.4) -> None:
    ax.add_feature(cfeature.LAND,      facecolor="#1c2128", edgecolor="none",   zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#090e16", edgecolor="none",   zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor="#58a6ff", linewidth=lw,       zorder=3)
    ax.add_feature(cfeature.BORDERS,   edgecolor="#484f58", linewidth=lw * 0.7,
                   linestyle="--", zorder=3)
    ax.add_feature(cfeature.STATES,    edgecolor="#30363d", linewidth=lw * 0.5, zorder=2)


def _add_gridlines(ax: plt.Axes, proj) -> None:
    draw = isinstance(proj, (ccrs.Mercator, ccrs.PlateCarree))
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=draw,
        linewidth=0.3, color="#21262d", alpha=0.8, linestyle="--", zorder=2,
    )
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlocator     = mticker.MultipleLocator(10)
    gl.ylocator     = mticker.MultipleLocator(10)
    gl.xlabel_style = {"size": 6, "color": "#8b949e"}
    gl.ylabel_style = {"size": 6, "color": "#8b949e"}


def _colorbar(fig, cf, ax, label: str, orientation: str = "horizontal") -> None:
    cb = fig.colorbar(cf, ax=ax, orientation=orientation,
                      pad=0.03, fraction=0.03, shrink=0.85)
    cb.set_label(label, fontsize=8, color="#8b949e")
    cb.ax.tick_params(labelsize=7, colors="#8b949e")
    cb.outline.set_edgecolor("#30363d")


def _title(ax, main: str, sub: str = "") -> None:
    ax.set_title(main, fontsize=10, fontweight="bold",
                 color="#e6edf3", loc="left", pad=6)
    if sub:
        ax.set_title(sub, fontsize=7, color="#8b949e", loc="right", pad=6)


def _downsample(lat, lon, data, factor: int = 4):
    return lat[::factor], lon[::factor], data[::factor, ::factor]


def _get(ds, var: str, time_idx: int = 0):
    da  = ds[var].isel(time=time_idx)
    return da.latitude.values, da.longitude.values, da.values


def _run_label(hour: int) -> str:
    return f"GFS {DATE} {CYCLE}Z  +{hour:03d}h"


# ══════════════════════════════════════════════════════════════════════════════
# Download
# ══════════════════════════════════════════════════════════════════════════════

def download_data(hours: list[int]):
    """Baixa dados global e Brasil para as horas pedidas."""
    vars_needed = ["t2m", "prmsl", "prate", "u10", "v10"]

    # Diretórios de cache de GRIB2 (subpastas separadas para não misturar arquivos)
    cache_global = OUTPUT_DIR / "global"
    cache_brasil = OUTPUT_DIR / "brasil"
    cache_global.mkdir(parents=True, exist_ok=True)
    cache_brasil.mkdir(parents=True, exist_ok=True)

    # NetCDFs consolidados ficam diretamente em OUTPUT_DIR
    # Usar path absoluto evita o bug de duplicação que ocorre quando
    # save_netcdf recebe um path relativo e o prepende com output_dir.
    nc_global = (OUTPUT_DIR / "gfs_global.nc").resolve()
    nc_brasil  = (OUTPUT_DIR / "gfs_brasil.nc").resolve()

    LOG.info("=== Download GLOBAL ===")
    mgr_g = GFSDatasetManager(
        date=DATE, cycle=CYCLE,
        output_dir=str(cache_global),
        region=None,
    )
    ds_g = mgr_g.build_multi_dataset(var_keys=vars_needed, hours=hours)
    mgr_g.save_netcdf(ds_g, str(nc_global))

    LOG.info("=== Download BRASIL ===")
    mgr_br = GFSDatasetManager(
        date=DATE, cycle=CYCLE,
        output_dir=str(cache_brasil),
        region=REGION_BR,
    )
    ds_br = mgr_br.build_multi_dataset(var_keys=vars_needed, hours=hours)
    mgr_br.save_netcdf(ds_br, str(nc_brasil))

    return ds_g, ds_br


# ══════════════════════════════════════════════════════════════════════════════
# Funções de frame  (retornam fig, ax prontos)
# ══════════════════════════════════════════════════════════════════════════════

def _frame_t2m_ortho(ds_g, tidx: int, hour: int) -> plt.Figure:
    lat, lon, t2m = _get(ds_g, "t2m", tidx)
    proj = ccrs.Orthographic(central_longitude=-50, central_latitude=-10)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": proj},
                            facecolor="#0d1117")
    ax.set_global()
    _add_features(ax, 0.5)
    norm = BoundaryNorm(TEMP_LEVELS, ncolors=TEMP_CMAP.N, clip=True)
    cf = ax.contourf(lon, lat, t2m, levels=TEMP_LEVELS, cmap=TEMP_CMAP, norm=norm,
                     transform=ccrs.PlateCarree(), zorder=1, extend="both")
    ax.contour(lon, lat, t2m, levels=TEMP_LEVELS[::5],
               colors="white", linewidths=0.25, alpha=0.4,
               transform=ccrs.PlateCarree(), zorder=2)
    _colorbar(fig, cf, ax, "Temperatura (°C)")
    _title(ax, "T2m — Global Ortho", _run_label(hour))
    fig.tight_layout()
    return fig


def _frame_t2m_brasil(ds_br, tidx: int, hour: int) -> plt.Figure:
    lat, lon, t2m = _get(ds_br, "t2m", tidx)
    proj = ccrs.Mercator()
    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={"projection": proj},
                            facecolor="#0d1117")
    ax.set_extent([-75, -34, -35, 5], crs=ccrs.PlateCarree())
    _add_features(ax, 0.6)
    _add_gridlines(ax, proj)
    norm = BoundaryNorm(TEMP_LEVELS, ncolors=TEMP_CMAP.N, clip=True)
    cf = ax.contourf(lon, lat, t2m, levels=TEMP_LEVELS, cmap=TEMP_CMAP, norm=norm,
                     transform=ccrs.PlateCarree(), zorder=1, extend="both")
    ax.contour(lon, lat, t2m, levels=TEMP_LEVELS[::4],
               colors="white", linewidths=0.3, alpha=0.5,
               transform=ccrs.PlateCarree(), zorder=2)
    _colorbar(fig, cf, ax, "Temperatura (°C)")
    _title(ax, "T2m — Brasil Mercator", _run_label(hour))
    fig.tight_layout()
    return fig


def _frame_prmsl_ortho(ds_g, tidx: int, hour: int) -> plt.Figure:
    lat, lon, slp = _get(ds_g, "prmsl", tidx)
    proj = ccrs.Orthographic(central_longitude=-50, central_latitude=-10)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": proj},
                            facecolor="#0d1117")
    ax.set_global()
    _add_features(ax, 0.5)
    norm = BoundaryNorm(PRES_LEVELS, ncolors=PRES_CMAP.N, clip=True)
    cf = ax.contourf(lon, lat, slp, levels=PRES_LEVELS, cmap=PRES_CMAP, norm=norm,
                     transform=ccrs.PlateCarree(), zorder=1, extend="both")
    cs = ax.contour(lon, lat, slp, levels=PRES_LEVELS[::2],
                    colors="white", linewidths=0.4, alpha=0.6,
                    transform=ccrs.PlateCarree(), zorder=2)
    ax.clabel(cs, inline=True, fontsize=5, fmt="%d", colors="#c9d1d9")
    _colorbar(fig, cf, ax, "PRMSL (hPa)")
    _title(ax, "Pressão NMM — Global Ortho", _run_label(hour))
    fig.tight_layout()
    return fig


def _frame_prate_brasil(ds_br, tidx: int, hour: int) -> plt.Figure:
    lat, lon, prate = _get(ds_br, "prate", tidx)
    prate_mmh = np.where(prate > 0, prate * 3600.0, np.nan)  # kg/m²/s → mm/h

    proj = ccrs.Mercator()
    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={"projection": proj},
                            facecolor="#0d1117")
    ax.set_extent([-75, -34, -35, 5], crs=ccrs.PlateCarree())
    _add_features(ax, 0.6)
    _add_gridlines(ax, proj)

    # Estatísticas para título informativo
    valid = prate_mmh[np.isfinite(prate_mmh)]
    pmax  = float(valid.max()) if valid.size else 0.0

    # BoundaryNorm log-espaçado — muito mais contraste para chuviscos
    norm = BoundaryNorm(PRATE_LEVELS, ncolors=PRATE_CMAP.N, clip=True)
    cf = ax.contourf(lon, lat, prate_mmh, levels=PRATE_LEVELS, cmap=PRATE_CMAP,
                     norm=norm, transform=ccrs.PlateCarree(), zorder=1, extend="max")

    _colorbar(fig, cf, ax, "Precipitação (mm h⁻¹)")
    _title(ax, f"Precipitação — Brasil  [máx: {pmax:.2f} mm h⁻¹]", _run_label(hour))
    fig.tight_layout()
    return fig


def _frame_wind_ortho(ds_g, tidx: int, hour: int) -> plt.Figure:
    lat, lon, u = _get(ds_g, "u10", tidx)
    _,   _,   v = _get(ds_g, "v10", tidx)
    spd = np.hypot(u, v)
    proj = ccrs.Orthographic(central_longitude=-50, central_latitude=-10)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": proj},
                            facecolor="#0d1117")
    ax.set_global()
    _add_features(ax, 0.5)
    norm = BoundaryNorm(SPD_LEVELS_G, ncolors=SPD_CMAP.N, clip=True)
    cf = ax.contourf(lon, lat, spd, levels=SPD_LEVELS_G, cmap=SPD_CMAP, norm=norm,
                     transform=ccrs.PlateCarree(), zorder=1, extend="max")
    blat, blon, bu = _downsample(lat, lon, u, factor=8)
    _,    _,    bv = _downsample(lat, lon, v, factor=8)
    BLON, BLAT = np.meshgrid(blon, blat)
    ax.barbs(BLON, BLAT, bu, bv, transform=ccrs.PlateCarree(),
             length=4, linewidth=0.5, barbcolor="#e6edf3",
             flagcolor="#f78166", alpha=0.75, zorder=4)
    _colorbar(fig, cf, ax, "Vento (m s⁻¹)")
    _title(ax, "Vento 10m — Global Ortho", _run_label(hour))
    fig.tight_layout()
    return fig


def _frame_wind_brasil(ds_br, tidx: int, hour: int) -> plt.Figure:
    lat, lon, u = _get(ds_br, "u10", tidx)
    _,   _,   v = _get(ds_br, "v10", tidx)
    spd = np.hypot(u, v)
    proj = ccrs.Mercator()
    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={"projection": proj},
                            facecolor="#0d1117")
    ax.set_extent([-75, -34, -35, 5], crs=ccrs.PlateCarree())
    _add_features(ax, 0.6)
    _add_gridlines(ax, proj)
    norm = BoundaryNorm(SPD_LEVELS_BR, ncolors=SPD_CMAP.N, clip=True)
    cf = ax.contourf(lon, lat, spd, levels=SPD_LEVELS_BR, cmap=SPD_CMAP, norm=norm,
                     transform=ccrs.PlateCarree(), zorder=1, extend="max")
    blat, blon, bu = _downsample(lat, lon, u, factor=3)
    _,    _,    bv = _downsample(lat, lon, v, factor=3)
    BLON, BLAT = np.meshgrid(blon, blat)
    ax.barbs(BLON, BLAT, bu, bv, transform=ccrs.PlateCarree(),
             length=5, linewidth=0.6, barbcolor="#e6edf3",
             flagcolor="#f78166", alpha=0.85, zorder=4)
    _colorbar(fig, cf, ax, "Vento (m s⁻¹)")
    _title(ax, "Vento 10m — Brasil Mercator", _run_label(hour))
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Painel 2×3 (hora única)
# ══════════════════════════════════════════════════════════════════════════════

def _frame_painel(ds_g, ds_br, tidx: int, hour: int) -> plt.Figure:
    fig = plt.figure(figsize=(22, 12), facecolor="#0d1117")
    gs  = GridSpec(2, 3, figure=fig,
                   hspace=0.12, wspace=0.08,
                   left=0.03, right=0.97, top=0.93, bottom=0.06)
    rl = _run_label(hour)

    # ── [0,0] T2m Ortho Global ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.Orthographic(-50, -10))
    ax.set_global(); _add_features(ax, 0.4)
    lat, lon, t2m = _get(ds_g, "t2m", tidx)
    norm = BoundaryNorm(TEMP_LEVELS, ncolors=TEMP_CMAP.N, clip=True)
    cf = ax.contourf(lon, lat, t2m, levels=TEMP_LEVELS, cmap=TEMP_CMAP, norm=norm,
                     transform=ccrs.PlateCarree(), zorder=1, extend="both")
    _title(ax, "T2m — Ortho", rl); _colorbar(fig, cf, ax, "°C")

    # ── [0,1] T2m Brasil Mercator ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1], projection=ccrs.Mercator())
    ax.set_extent([-75, -34, -35, 5], crs=ccrs.PlateCarree())
    _add_features(ax, 0.5); _add_gridlines(ax, ccrs.Mercator())
    lat, lon, t2m_br = _get(ds_br, "t2m", tidx)
    cf = ax.contourf(lon, lat, t2m_br, levels=TEMP_LEVELS, cmap=TEMP_CMAP, norm=norm,
                     transform=ccrs.PlateCarree(), zorder=1, extend="both")
    _title(ax, "T2m — Brasil", rl); _colorbar(fig, cf, ax, "°C")

    # ── [0,2] PRMSL Ortho ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2], projection=ccrs.Orthographic(-50, -10))
    ax.set_global(); _add_features(ax, 0.4)
    lat, lon, slp = _get(ds_g, "prmsl", tidx)
    norm_p = BoundaryNorm(PRES_LEVELS, ncolors=PRES_CMAP.N, clip=True)
    cf = ax.contourf(lon, lat, slp, levels=PRES_LEVELS, cmap=PRES_CMAP, norm=norm_p,
                     transform=ccrs.PlateCarree(), zorder=1, extend="both")
    cs = ax.contour(lon, lat, slp, levels=PRES_LEVELS[::4],
                    colors="white", linewidths=0.3, alpha=0.5,
                    transform=ccrs.PlateCarree(), zorder=2)
    ax.clabel(cs, inline=True, fontsize=4, fmt="%d", colors="#c9d1d9")
    _title(ax, "PRMSL — Ortho", rl); _colorbar(fig, cf, ax, "hPa")

    # ── [1,0] Precipitação Brasil ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0], projection=ccrs.Mercator())
    ax.set_extent([-75, -34, -35, 5], crs=ccrs.PlateCarree())
    _add_features(ax, 0.5); _add_gridlines(ax, ccrs.Mercator())
    lat, lon, prate_br = _get(ds_br, "prate", tidx)
    prate_mmh = np.where(prate_br > 0, prate_br * 3600.0, np.nan)
    norm_pr = BoundaryNorm(PRATE_LEVELS, ncolors=PRATE_CMAP.N, clip=True)
    cf = ax.contourf(lon, lat, prate_mmh, levels=PRATE_LEVELS, cmap=PRATE_CMAP,
                     norm=norm_pr, transform=ccrs.PlateCarree(), zorder=1, extend="max")
    _title(ax, "Precipitação — Brasil", rl); _colorbar(fig, cf, ax, "mm h⁻¹")

    # ── [1,1] Vento Global PlateCarree ────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
    ax.set_global(); _add_features(ax, 0.35); _add_gridlines(ax, ccrs.PlateCarree())
    lat, lon, u_g = _get(ds_g, "u10", tidx)
    _,   _,   v_g = _get(ds_g, "v10", tidx)
    spd_g = np.hypot(u_g, v_g)
    norm_s = BoundaryNorm(SPD_LEVELS_G, ncolors=SPD_CMAP.N, clip=True)
    cf = ax.contourf(lon, lat, spd_g, levels=SPD_LEVELS_G, cmap=SPD_CMAP, norm=norm_s,
                     transform=ccrs.PlateCarree(), zorder=1, extend="max")
    blat, blon, bu = _downsample(lat, lon, u_g, factor=8)
    _,    _,    bv = _downsample(lat, lon, v_g, factor=8)
    BLON, BLAT = np.meshgrid(blon, blat)
    ax.barbs(BLON, BLAT, bu, bv, transform=ccrs.PlateCarree(),
             length=3.5, linewidth=0.4, barbcolor="#e6edf3",
             flagcolor="#f78166", alpha=0.7, zorder=4)
    _title(ax, "Vento 10m — Global", rl); _colorbar(fig, cf, ax, "m s⁻¹")

    # ── [1,2] Vento Brasil Mercator ───────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2], projection=ccrs.Mercator())
    ax.set_extent([-75, -34, -35, 5], crs=ccrs.PlateCarree())
    _add_features(ax, 0.5); _add_gridlines(ax, ccrs.Mercator())
    lat, lon, u_br = _get(ds_br, "u10", tidx)
    _,   _,   v_br = _get(ds_br, "v10", tidx)
    spd_br = np.hypot(u_br, v_br)
    norm_sbr = BoundaryNorm(SPD_LEVELS_BR, ncolors=SPD_CMAP.N, clip=True)
    cf = ax.contourf(lon, lat, spd_br, levels=SPD_LEVELS_BR, cmap=SPD_CMAP, norm=norm_sbr,
                     transform=ccrs.PlateCarree(), zorder=1, extend="max")
    blat, blon, bu_br = _downsample(lat, lon, u_br, factor=3)
    _,    _,    bv_br = _downsample(lat, lon, v_br, factor=3)
    BLON, BLAT = np.meshgrid(blon, blat)
    ax.barbs(BLON, BLAT, bu_br, bv_br, transform=ccrs.PlateCarree(),
             length=5, linewidth=0.6, barbcolor="#e6edf3",
             flagcolor="#f78166", alpha=0.85, zorder=4)
    _title(ax, "Vento 10m — Brasil", rl); _colorbar(fig, cf, ax, "m s⁻¹")

    fig.suptitle(
        f"GFS 0.25°  ·  {DATE} {CYCLE}Z  ·  +{hour:03d}h",
        fontsize=14, fontweight="bold", color="#58a6ff", y=0.97,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Catálogo de plots: nome → (frame_func, args_from_ds, tamanho_fig)
# ══════════════════════════════════════════════════════════════════════════════

# Cada entrada: (slug, label, frame_builder)
# frame_builder(ds_g, ds_br, tidx, hour) → plt.Figure
PLOTS: list[tuple[str, str, Callable]] = [
    ("t2m_ortho",    "T2m Ortho Global",         lambda g, b, i, h: _frame_t2m_ortho  (g, i, h)),
    ("t2m_brasil",   "T2m Brasil Mercator",       lambda g, b, i, h: _frame_t2m_brasil (b, i, h)),
    ("prmsl_ortho",  "PRMSL Ortho Global",        lambda g, b, i, h: _frame_prmsl_ortho(g, i, h)),
    ("prate_brasil", "Precipitação Brasil",       lambda g, b, i, h: _frame_prate_brasil(b, i, h)),
    ("wind_ortho",   "Vento Ortho Global",        lambda g, b, i, h: _frame_wind_ortho (g, i, h)),
    ("wind_brasil",  "Vento Brasil Mercator",     lambda g, b, i, h: _frame_wind_brasil(b, i, h)),
    ("painel",       "Painel 2×3 Combinado",      lambda g, b, i, h: _frame_painel     (g, b, i, h)),
]


# ══════════════════════════════════════════════════════════════════════════════
# Exportadores
# ══════════════════════════════════════════════════════════════════════════════

def _fig_to_rgba(fig: plt.Figure, dpi: int | None = None) -> np.ndarray:
    """Converte figura matplotlib para array RGB uint8.

    Usa o backend Agg diretamente via buffer em memória — mais rápido que
    fig.canvas.draw() + tostring_argb() e independente do backend interativo.
    Se *dpi* for None, usa o DPI já configurado na figura.
    """
    import matplotlib.backends.backend_agg as _agg
    if dpi is not None and fig.get_dpi() != dpi:
        fig.set_dpi(dpi)
    canvas = _agg.FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()                     # memoryview RGBA uint8
    w, h = canvas.get_width_height()
    arr  = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    return arr[..., :3].copy()                     # retorna RGB (sem alpha)


def save_static(
    ds_g, ds_br,
    hours: list[int],
    out_dir: Path,
    plots: list[str] | None = None,
    dpi: int = 150,
    gif_colors: int = 256,        # ignorado aqui, mantido por assinatura uniforme
) -> None:
    """Salva PNGs individuais (hora × plot)."""
    chosen = [p for p in PLOTS if plots is None or p[0] in plots]
    total  = len(hours) * len(chosen)
    done   = 0
    t0     = time.time()
    for tidx, hour in enumerate(hours):
        for slug, label, builder in chosen:
            path = out_dir / f"{slug}_f{hour:03d}.png"
            fig  = builder(ds_g, ds_br, tidx, hour)
            fig.savefig(path, dpi=dpi, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            done += 1
            elapsed = time.time() - t0
            eta     = elapsed / done * (total - done)
            LOG.info("[static %d/%d] %s  |  ETA %s",
                     done, total, path.name,
                     time.strftime("%H:%M:%S", time.gmtime(eta)))


def save_gif(
    ds_g, ds_br,
    hours: list[int],
    out_dir: Path,
    plots: list[str] | None = None,
    fps: int = 2,
    dpi: int = 100,
    gif_colors: int = 192,
) -> None:
    """Salva GIFs animados (um arquivo por plot).

    Parâmetros de qualidade
    -----------------------
    dpi        : resolução de cada frame (72–300).  Para 16 dias, 80–100 é
                 o ponto ideal velocidade × qualidade.  300 fica lento.
    gif_colors : paleta do GIF (2–256).  192 equilibra cores e tamanho de
                 arquivo.  256 = máxima qualidade, porém GIF maior.
    fps        : frames por segundo da animação.
    """
    try:
        import imageio.v3 as iio
        from PIL import Image as _PILImage
    except ImportError:
        LOG.error("imageio / Pillow não encontrado — pip install imageio[ffmpeg] pillow")
        return

    chosen = [p for p in PLOTS if plots is None or p[0] in plots]
    n_frames = len(hours)
    duration_ms = int(1000 / fps)

    for slug, label, builder in chosen:
        path   = out_dir / f"{slug}_anim.gif"
        pil_frames: list = []
        t0 = time.time()

        LOG.info("[gif] %s — %d frames  dpi=%d  colors=%d  fps=%d",
                 slug, n_frames, dpi, gif_colors, fps)

        for tidx, hour in enumerate(hours):
            fig = builder(ds_g, ds_br, tidx, hour)
            rgb = _fig_to_rgba(fig, dpi=dpi)          # array H×W×3
            plt.close(fig)

            # Converter para PIL e quantizar (reduz paleta → GIF menor/rápido)
            pil_img = _PILImage.fromarray(rgb, mode="RGB")
            pil_img = pil_img.quantize(
                colors=gif_colors,
                method=_PILImage.Quantize.FASTOCTREE,  # ~3× mais rápido que MEDIANCUT
                dither=_PILImage.Dither.FLOYDSTEINBERG,
            )
            pil_frames.append(pil_img)

            elapsed = time.time() - t0
            done    = tidx + 1
            eta     = elapsed / done * (n_frames - done)
            LOG.info("  [gif %s %d/%d] f%03d  |  ETA %s",
                     slug, done, n_frames, hour,
                     time.strftime("%H:%M:%S", time.gmtime(eta)))

        # Salvar como GIF multi-frame com Pillow diretamente (mais controle)
        pil_frames[0].save(
            path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        size_mb = path.stat().st_size / 1024 ** 2
        LOG.info("[gif] Salvo → %s  (%.1f MB)", path, size_mb)


def save_mp4(
    ds_g, ds_br,
    hours: list[int],
    out_dir: Path,
    plots: list[str] | None = None,
    fps: int = 4,
    dpi: int = 150,
    gif_colors: int = 256,        # ignorado, mantido por assinatura uniforme
    crf: int = 23,
    preset: str = "fast",
) -> None:
    """Salva MP4 animados (um arquivo por plot).

    Parâmetros de qualidade
    -----------------------
    dpi    : resolução de cada frame.  150 é bom equilíbrio para MP4.
             300 gera vídeo em alta definição mas demora mais.
    crf    : Constant Rate Factor do libx264 (0=lossless, 51=pior).
             18 = alta qualidade, 23 = padrão, 28 = arquivo menor.
    preset : velocidade × compressão do ffmpeg.
             ultrafast/superfast/veryfast/faster/fast/medium/slow/veryslow
             Para 16 dias use "fast" ou "medium".
    fps    : frames por segundo.
    """
    try:
        import imageio.v3 as iio
    except ImportError:
        LOG.error("imageio[ffmpeg] não encontrado — pip install imageio[ffmpeg]")
        return

    chosen   = [p for p in PLOTS if plots is None or p[0] in plots]
    n_frames = len(hours)

    for slug, label, builder in chosen:
        path = out_dir / f"{slug}_anim.mp4"
        output_params = ["-crf", str(crf), "-preset", preset,
                         "-pix_fmt", "yuv420p"]
        t0 = time.time()
        LOG.info("[mp4] %s — %d frames  dpi=%d  crf=%d  preset=%s  fps=%d",
                 slug, n_frames, dpi, crf, preset, fps)

        with iio.imopen(str(path), "w", plugin="ffmpeg") as writer:
            writer.init_video_stream("libx264", fps=fps,
                                     output_params=output_params)
            for tidx, hour in enumerate(hours):
                fig = builder(ds_g, ds_br, tidx, hour)
                rgb = _fig_to_rgba(fig, dpi=dpi)
                writer.write_frame(rgb)
                plt.close(fig)
                elapsed = time.time() - t0
                done    = tidx + 1
                eta     = elapsed / done * (n_frames - done)
                LOG.info("  [mp4 %s %d/%d] f%03d  |  ETA %s",
                         slug, done, n_frames, hour,
                         time.strftime("%H:%M:%S", time.gmtime(eta)))
                LOG.info("[mp4] %s  f%03d", slug, hour)
        LOG.info("[mp4] Salvo → %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

# Presets de qualidade prontos para uso
_QUALITY_PRESETS = {
    #          dpi   gif_colors  crf   mp4_preset
    "draft":  ( 72,    128,      28,   "ultrafast"),
    "fast":   (100,    192,      26,   "veryfast" ),
    "normal": (150,    192,      23,   "fast"     ),
    "high":   (200,    256,      20,   "medium"   ),
    "ultra":  (300,    256,      18,   "slow"     ),
}

def _parse_args():
    p = argparse.ArgumentParser(
        description="GFS Plot Suite — estático, GIF ou MP4",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--mode", default="static",
        choices=["static", "gif", "mp4", "panel", "all"],
        help=(
            "static : PNGs individuais por hora\n"
            "gif    : GIF animado por variável\n"
            "mp4    : MP4 animado por variável\n"
            "panel  : painel 2×3 (apenas primeira hora)\n"
            "all    : tudo\n"
        ),
    )
    p.add_argument(
        "--hours", nargs="+", type=int,
        default=[0, 6, 12, 18, 24],
        help="Lista de horas de prognóstico (padrão: 0 6 12 18 24)",
    )
    p.add_argument(
        "--plots", nargs="+",
        choices=[p[0] for p in PLOTS] + ["all"],
        default=["all"],
        help="Plots a gerar (padrão: all)",
    )
    p.add_argument(
        "--fps", type=int, default=3,
        help="FPS para gif/mp4 (padrão: 3)",
    )
    # ── qualidade ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--quality", default="normal",
        choices=list(_QUALITY_PRESETS),
        help=(
            "Preset de qualidade (afeta dpi, cores do GIF, CRF do MP4):\n"
            "  draft  : dpi=72  — rascunho rápido (16 dias em minutos)\n"
            "  fast   : dpi=100 — bom para preview\n"
            "  normal : dpi=150 — padrão (painel e apresentação)\n"
            "  high   : dpi=200 — alta qualidade\n"
            "  ultra  : dpi=300 — máxima qualidade (lento)\n"
        ),
    )
    p.add_argument(
        "--dpi", type=int, default=None,
        help="Sobrepõe o DPI do preset (ex: --dpi 300)",
    )
    p.add_argument(
        "--gif-colors", type=int, default=None,
        metavar="N",
        help="Paleta GIF: 2–256 cores (sobrepõe preset)",
    )
    p.add_argument(
        "--crf", type=int, default=None,
        help="CRF do libx264 para MP4: 0=lossless 51=pior (sobrepõe preset)",
    )
    p.add_argument(
        "--mp4-preset", default=None,
        dest="mp4_preset",
        choices=["ultrafast","superfast","veryfast","faster",
                 "fast","medium","slow","veryslow"],
        help="Preset ffmpeg para MP4 (sobrepõe preset de qualidade)",
    )
    p.add_argument(
        "--no-download", action="store_true",
        help="Reutiliza NetCDFs já salvos em OUTPUT_DIR",
    )
    p.add_argument(
        "--out", default=str(FIG_DIR),
        help=f"Diretório de saída para figuras (padrão: {FIG_DIR})",
    )
    return p.parse_args()


def main() -> None:
    args     = _parse_args()
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    hours    = sorted(set(args.hours))
    plot_sel = None if "all" in args.plots else args.plots

    # ── resolver parâmetros de qualidade ──────────────────────────────────────
    preset_dpi, preset_gc, preset_crf, preset_mp4 = _QUALITY_PRESETS[args.quality]
    dpi        = args.dpi        if args.dpi        is not None else preset_dpi
    gif_colors = args.gif_colors if args.gif_colors is not None else preset_gc
    crf        = args.crf        if args.crf        is not None else preset_crf
    mp4_preset = args.mp4_preset if args.mp4_preset is not None else preset_mp4

    print("\n" + "=" * 60)
    print(f"  GFS Plot Suite  —  modo: {args.mode}")
    print(f"  Data: {DATE}  Ciclo: {CYCLE}Z")
    print(f"  Horas: {len(hours)}  ({hours[0]}–{hours[-1]}h)")
    print(f"  Qualidade: {args.quality}  |  DPI={dpi}  GIF_COLORS={gif_colors}")
    print(f"  MP4: CRF={crf}  preset={mp4_preset}  FPS={args.fps}")
    print(f"  Saída: {out_dir.resolve()}")
    print("=" * 60 + "\n")

    # ── dados ─────────────────────────────────────────────────────────────────
    if args.no_download:
        nc_g  = OUTPUT_DIR / "gfs_global.nc"
        nc_br = OUTPUT_DIR / "gfs_brasil.nc"
        missing = [p for p in (nc_g, nc_br) if not p.exists()]
        if missing:
            LOG.error(
                "Arquivo(s) não encontrado(s) para --no-download:\n  %s\n"
                "Execute sem --no-download primeiro para gerar os NetCDFs.",
                "\n  ".join(str(p) for p in missing),
            )
            raise SystemExit(1)
        LOG.info("Carregando NetCDFs do disco …")
        LOG.info("  global : %s", nc_g)
        LOG.info("  brasil : %s", nc_br)
        ds_g  = GFSDatasetManager.load_netcdf(nc_g)
        ds_br = GFSDatasetManager.load_netcdf(nc_br)
        # Filtrar apenas as horas pedidas (se houver subset)
        fh_g  = list(map(int, ds_g["forecast_hour"].values))
        fh_br = list(map(int, ds_br["forecast_hour"].values))
        sel_g  = [i for i, h in enumerate(fh_g)  if h in hours]
        sel_br = [i for i, h in enumerate(fh_br) if h in hours]
        if not sel_g:
            LOG.error("Nenhuma hora pedida (%s) encontrada no NetCDF global (%s).", hours, fh_g)
            raise SystemExit(1)
        ds_g  = ds_g.isel(time=sel_g)
        ds_br = ds_br.isel(time=sel_br) if sel_br else ds_br
    else:
        ds_g, ds_br = download_data(hours)

    # Sincronizar lista de horas com o que foi realmente baixado
    hours_ok = sorted(ds_g["forecast_hour"].values.tolist())

    # ── renderização ──────────────────────────────────────────────────────────
    if args.mode in ("static", "all"):
        save_static(ds_g, ds_br, hours_ok, out_dir, plot_sel,
                    dpi=dpi)

    if args.mode in ("gif", "all"):
        save_gif(ds_g, ds_br, hours_ok, out_dir, plot_sel,
                 fps=args.fps, dpi=dpi, gif_colors=gif_colors)

    if args.mode in ("mp4", "all"):
        save_mp4(ds_g, ds_br, hours_ok, out_dir, plot_sel,
                 fps=args.fps, dpi=dpi, crf=crf, preset=mp4_preset)

    if args.mode in ("panel", "all"):
        first_hour = hours_ok[0]
        fig = _frame_painel(ds_g, ds_br, 0, first_hour)
        path = out_dir / f"painel_f{first_hour:03d}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        LOG.info("[panel] Salvo → %s", path)

    print(f"\n✓ Pronto!  Arquivos em: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()