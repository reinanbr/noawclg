import xarray as xr

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

ds = xr.open_dataset("./gfs_output/gfs_t2m_16days.nc", engine="netcdf4")

print(ds['time'])

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

def _add_features(ax: plt.Axes, lw: float = 0.4) -> None:
    ax.add_feature(cfeature.LAND,      facecolor="#1c2128", edgecolor="none",   zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#090e16", edgecolor="none",   zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor="#58a6ff", linewidth=lw,       zorder=3)
    ax.add_feature(cfeature.BORDERS,   edgecolor="#484f58", linewidth=lw * 0.7,
                   linestyle="--", zorder=3)
    ax.add_feature(cfeature.STATES,    edgecolor="#30363d", linewidth=lw * 0.5, zorder=2)



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

save_mp4(ds_g, hours_ok, out_dir, plot_sel,
                 fps=args.fps, dpi=dpi, crf=crf, preset=mp4_preset)