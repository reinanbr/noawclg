"""
GFS Climate Plotter — com Grib Filter (recorte regional)
=========================================================
Baixa dados do GFS via gribfilter do NOMADS (apenas a região e variável
desejadas), plota em projeção ortográfica ou mercator (estático ou animado).

Dependências:
    pip install requests cfgrib xarray matplotlib cartopy numpy pillow eccodes
    # Para MP4: instalar ffmpeg no sistema (sudo apt install ffmpeg)

Uso rápido:
    python gfs_plots.py
"""

import os
import time
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cfgrib
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta

# ══════════════════════════════════════════════════════════════════════════════
# 1  CONFIGURACOES — edite aqui
# ══════════════════════════════════════════════════════════════════════════════

DATE  = "20260403"   # data da rodada  (YYYYMMDD)
CYCLE = "06"         # ciclo UTC       (00 | 06 | 12 | 18)

# Recorte geografico (None = global)
REGIAO = {
    "nome":      "Brasil",
    "toplat":     5,
    "bottomlat": -35,
    "leftlon":   -75,
    "rightlon":  -34,
}
# Para global: REGIAO = None

OUTPUT_DIR = "./gfs_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pausa entre downloads para nao sobrecarregar o servidor (segundos)
PAUSA_DOWNLOAD = 1.5

# ══════════════════════════════════════════════════════════════════════════════
# 2  CATALOGO DE VARIAVEIS
# ══════════════════════════════════════════════════════════════════════════════

VARIAVEIS = {
    "temperatura_2m": {
        "titulo":    "Temperatura a 2m",
        "grib_var":  "var_TMP",
        "grib_lev":  "lev_2_m_above_ground",
        "short":     "2t",
        "tlev":      "heightAboveGround",
        "level":     2,
        "unidade":   "C",
        "cmap":      "RdBu_r",
        "converter": lambda x: x - 273.15,
        "vmin": -20, "vmax": 45,
    },
    "precipitacao": {
        "titulo":    "Precipitacao Acumulada",
        "grib_var":  "var_APCP",
        "grib_lev":  "lev_surface",
        "short":     "tp",
        "tlev":      "surface",
        "level":     0,
        "unidade":   "mm",
        "cmap":      "Blues",
        "converter": None,
        "vmin": 0, "vmax": 80,
    },
    "pressao_msl": {
        "titulo":    "Pressao ao Nivel do Mar",
        "grib_var":  "var_PRMSL",
        "grib_lev":  "lev_mean_sea_level",
        "short":     "msl",
        "tlev":      "meanSea",
        "level":     0,
        "unidade":   "hPa",
        "cmap":      "coolwarm",
        "converter": lambda x: x / 100,
        "vmin": 990, "vmax": 1025,
    },
    "vento_u_10m": {
        "titulo":    "Componente U do Vento 10m",
        "grib_var":  "var_UGRD",
        "grib_lev":  "lev_10_m_above_ground",
        "short":     "10u",
        "tlev":      "heightAboveGround",
        "level":     10,
        "unidade":   "m/s",
        "cmap":      "PuOr",
        "converter": None,
        "vmin": -20, "vmax": 20,
    },
    "vento_v_10m": {
        "titulo":    "Componente V do Vento 10m",
        "grib_var":  "var_VGRD",
        "grib_lev":  "lev_10_m_above_ground",
        "short":     "10v",
        "tlev":      "heightAboveGround",
        "level":     10,
        "unidade":   "m/s",
        "cmap":      "PuOr",
        "converter": None,
        "vmin": -20, "vmax": 20,
    },
    "umidade_850": {
        "titulo":    "Umidade Relativa 850 hPa",
        "grib_var":  "var_RH",
        "grib_lev":  "lev_850_mb",
        "short":     "r",
        "tlev":      "isobaricInhPa",
        "level":     850,
        "unidade":   "%",
        "cmap":      "BrBG",
        "converter": None,
        "vmin": 0, "vmax": 100,
    },
    "geopotencial_500": {
        "titulo":    "Altura Geopotencial 500 hPa",
        "grib_var":  "var_HGT",
        "grib_lev":  "lev_500_mb",
        "short":     "gh",
        "tlev":      "isobaricInhPa",
        "level":     500,
        "unidade":   "m",
        "cmap":      "plasma",
        "converter": None,
        "vmin": 5400, "vmax": 5900,
    },
    "cape": {
        "titulo":    "CAPE - Energia Convectiva",
        "grib_var":  "var_CAPE",
        "grib_lev":  "lev_surface",
        "short":     "cape",
        "tlev":      "surface",
        "level":     0,
        "unidade":   "J/kg",
        "cmap":      "hot_r",
        "converter": None,
        "vmin": 0, "vmax": 4000,
    },
    "temperatura_850": {
        "titulo":    "Temperatura 850 hPa",
        "grib_var":  "var_TMP",
        "grib_lev":  "lev_850_mb",
        "short":     "t",
        "tlev":      "isobaricInhPa",
        "level":     850,
        "unidade":   "C",
        "cmap":      "RdBu_r",
        "converter": lambda x: x - 273.15,
        "vmin": -20, "vmax": 35,
    },
    "orvalho_2m": {
        "titulo":    "Ponto de Orvalho 2m",
        "grib_var":  "var_DPT",
        "grib_lev":  "lev_2_m_above_ground",
        "short":     "2d",
        "tlev":      "heightAboveGround",
        "level":     2,
        "unidade":   "C",
        "cmap":      "YlGnBu",
        "converter": lambda x: x - 273.15,
        "vmin": -10, "vmax": 30,
    },
    "agua_precipitavel": {
        "titulo":    "Agua Precipitavel",
        "grib_var":  "var_PWAT",
        "grib_lev":  "lev_entire_atmosphere_(considered_as_a_single_layer)",
        "short":     "pwat",
        "tlev":      "atmosphereSingleLayer",
        "level":     0,
        "unidade":   "kg/m2",
        "cmap":      "GnBu",
        "converter": None,
        "vmin": 0, "vmax": 70,
    },
    "nuvens_total": {
        "titulo":    "Cobertura de Nuvens Total",
        "grib_var":  "var_TCDC",
        "grib_lev":  "lev_entire_atmosphere",
        "short":     "tcc",
        "tlev":      "atmosphere",
        "level":     0,
        "unidade":   "%",
        "cmap":      "gray_r",
        "converter": None,
        "vmin": 0, "vmax": 100,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# 3  PROJECOES
# ══════════════════════════════════════════════════════════════════════════════

PROJECOES = {
    "ortho_brasil":   ccrs.Orthographic(central_longitude=-52, central_latitude=-15),
    "ortho_americas": ccrs.Orthographic(central_longitude=-60, central_latitude=15),
    "ortho_global":   ccrs.Orthographic(central_longitude=0,   central_latitude=30),
    "mercator":       ccrs.Mercator(),
    "robinson":       ccrs.Robinson(),
    "mollweide":      ccrs.Mollweide(),
    "platecarree":    ccrs.PlateCarree(),
}

# ══════════════════════════════════════════════════════════════════════════════
# 4  DOWNLOAD VIA GRIBFILTER
# ══════════════════════════════════════════════════════════════════════════════

FILTER_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl"


def _nome_arquivo(var_key, hora):
    regiao = REGIAO["nome"].lower().replace(" ", "_") if REGIAO else "global"
    return os.path.join(
        OUTPUT_DIR,
        f"gfs_{DATE}_{CYCLE}z_{var_key}_{regiao}_f{hora:03d}.grib2",
    )


def baixar_variavel(var_key, hora, forcar=False):
    """
    Baixa uma variavel/horario via gribfilter.
    Retorna o caminho do arquivo ou None em caso de falha.
    """
    path = _nome_arquivo(var_key, hora)
    if os.path.exists(path) and not forcar:
        print(f"    [cache] f{hora:03d}")
        return path

    cfg = VARIAVEIS[var_key]
    params = {
        "dir":          f"/gfs.{DATE}/{CYCLE}/atmos",
        "file":         f"gfs.t{CYCLE}z.pgrb2.0p25.f{hora:03d}",
        cfg["grib_var"]: "on",
        cfg["grib_lev"]: "on",
    }
    if REGIAO:
        params.update({
            "subregion": "",
            "toplat":    REGIAO["toplat"],
            "bottomlat": REGIAO["bottomlat"],
            "leftlon":   REGIAO["leftlon"],
            "rightlon":  REGIAO["rightlon"],
        })

    try:
        r = requests.get(FILTER_URL, params=params, timeout=60)
        r.raise_for_status()
        if len(r.content) < 100:
            print(f"    [aviso] f{hora:03d} conteudo vazio (variavel indisponivel neste horario)")
            return None
        with open(path, "wb") as fh:
            fh.write(r.content)
        print(f"    [ok] f{hora:03d} ({len(r.content)/1024:.0f} KB)")
        time.sleep(PAUSA_DOWNLOAD)
        return path
    except Exception as e:
        print(f"    [erro] f{hora:03d}: {e}")
        return None


def baixar_serie(var_key, horas):
    """Baixa uma lista de horarios e retorna {hora: caminho}."""
    print(f"\nBaixando '{var_key}' — {len(horas)} horarios...")
    resultado = {}
    for h in horas:
        p = baixar_variavel(var_key, h)
        if p:
            resultado[h] = p
    print(f"  {len(resultado)}/{len(horas)} horarios disponíveis.")
    return resultado


# ══════════════════════════════════════════════════════════════════════════════
# 5  LEITURA DOS DADOS
# ══════════════════════════════════════════════════════════════════════════════

def _centralizar_lons(lons, data):
    if lons.max() > 180:
        lons = np.where(lons > 180, lons - 360, lons)
        idx  = np.argsort(lons)
        lons = lons[idx]
        data = data[:, idx]
    return lons, data


def ler_campo(path, var_key):
    """Le um campo GRIB e retorna (lats, lons, data 2D)."""
    cfg = VARIAVEIS[var_key]
    filtros = {"shortName": cfg["short"], "typeOfLevel": cfg["tlev"]}
    if cfg["level"] not in (0, None):
        filtros["level"] = cfg["level"]

    try:
        ds = cfgrib.open_dataset(path, filter_by_keys=filtros, indexpath=None)
    except Exception:
        filtros.pop("level", None)
        ds = cfgrib.open_dataset(path, filter_by_keys=filtros, indexpath=None)

    nome = list(ds.data_vars)[0]
    da   = ds[nome]
    lats = da.latitude.values
    lons = da.longitude.values
    data = da.values.astype(float)

    if cfg["converter"]:
        data = cfg["converter"](data)

    lons, data = _centralizar_lons(lons, data)
    return lats, lons, data


# ══════════════════════════════════════════════════════════════════════════════
# 6  PLOTAGEM ESTATICA
# ══════════════════════════════════════════════════════════════════════════════

def _features(ax, lw=0.5):
    ax.add_feature(cfeature.LAND,      facecolor="#f0ede8", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#d6eaf8", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=lw,        zorder=2)
    ax.add_feature(cfeature.BORDERS,   linewidth=lw * 0.7,
                   linestyle=":", zorder=2)
    ax.add_feature(cfeature.LAKES,     facecolor="#d6eaf8", alpha=0.7, zorder=1)
    ax.add_feature(cfeature.RIVERS,    linewidth=lw * 0.5,
                   edgecolor="#5dade2", zorder=1)


def _titulo(var_key, hora):
    cfg       = VARIAVEIS[var_key]
    dt_rodada = datetime.strptime(f"{DATE}{CYCLE}", "%Y%m%d%H")
    dt_valido = dt_rodada + timedelta(hours=hora)
    reg       = REGIAO["nome"] if REGIAO else "Global"
    return (
        f"GFS 0.25 | {cfg['titulo']} | {reg}\n"
        f"Rodada: {dt_rodada.strftime('%d/%m/%Y %HZ')}  |  "
        f"Valido: {dt_valido.strftime('%d/%m/%Y %H UTC')}  |  +{hora}h"
    )


def plot_estatico(lats, lons, data, var_key,
                  hora=0, proj_key="mercator",
                  salvar=True, mostrar=False, extent=None):
    cfg      = VARIAVEIS[var_key]
    proj     = PROJECOES[proj_key]
    data_plt = ccrs.PlateCarree()

    fig, ax = plt.subplots(figsize=(13, 8),
                           subplot_kw={"projection": proj})

    if extent:
        ax.set_extent(extent, crs=data_plt)
    elif REGIAO and proj_key in ("mercator", "platecarree"):
        pad = 3
        ax.set_extent([
            REGIAO["leftlon"]   - pad, REGIAO["rightlon"] + pad,
            REGIAO["bottomlat"] - pad, REGIAO["toplat"]   + pad,
        ], crs=data_plt)
    else:
        ax.set_global()

    _features(ax)

    lon2d, lat2d = np.meshgrid(lons, lats)
    cf = ax.pcolormesh(
        lon2d, lat2d, data,
        transform=data_plt,
        cmap=cfg["cmap"],
        vmin=cfg["vmin"], vmax=cfg["vmax"],
        shading="auto", zorder=1,
    )

    cb = fig.colorbar(cf, ax=ax, orientation="vertical",
                      pad=0.02, fraction=0.03, shrink=0.85)
    cb.set_label(cfg["unidade"], fontsize=10)
    ax.set_title(_titulo(var_key, hora), fontsize=10, pad=8)
    plt.tight_layout()

    if salvar:
        fname = os.path.join(OUTPUT_DIR,
                             f"gfs_{var_key}_{proj_key}_f{hora:03d}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Salvo: {fname}")

    if mostrar:
        plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 7  ANIMACAO
# ══════════════════════════════════════════════════════════════════════════════

def criar_animacao(frames, var_key, proj_key="mercator",
                   fps=4, formato="gif", extent=None):
    """
    Cria animacao a partir de uma lista de frames.
    frames: [(lats, lons, data, hora), ...]
    """
    if len(frames) < 2:
        print("  Frames insuficientes para animacao.")
        return

    cfg      = VARIAVEIS[var_key]
    proj     = PROJECOES[proj_key]
    data_plt = ccrs.PlateCarree()

    fig, ax = plt.subplots(figsize=(13, 8),
                           subplot_kw={"projection": proj})

    if extent:
        ax.set_extent(extent, crs=data_plt)
    elif REGIAO and proj_key in ("mercator", "platecarree"):
        pad = 3
        ax.set_extent([
            REGIAO["leftlon"]   - pad, REGIAO["rightlon"] + pad,
            REGIAO["bottomlat"] - pad, REGIAO["toplat"]   + pad,
        ], crs=data_plt)
    else:
        ax.set_global()

    _features(ax)

    lats0, lons0, data0, _ = frames[0]
    lon2d, lat2d = np.meshgrid(lons0, lats0)
    cf = ax.pcolormesh(
        lon2d, lat2d, data0,
        transform=data_plt,
        cmap=cfg["cmap"],
        vmin=cfg["vmin"], vmax=cfg["vmax"],
        shading="auto", zorder=1,
    )
    cb = fig.colorbar(cf, ax=ax, orientation="vertical",
                      pad=0.02, fraction=0.03, shrink=0.85)
    cb.set_label(cfg["unidade"], fontsize=10)
    titulo_obj = ax.set_title("", fontsize=10)

    def update(i):
        _, _, data_i, hora_i = frames[i]
        cf.set_array(data_i.ravel())
        titulo_obj.set_text(_titulo(var_key, hora_i))
        return cf, titulo_obj

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=1000 // fps, blit=False,
    )

    fname = os.path.join(OUTPUT_DIR,
                         f"gfs_{var_key}_{proj_key}_anim.{formato}")
    print(f"  Salvando animacao: {fname} ...")
    if formato == "gif":
        ani.save(fname, writer="pillow", fps=fps, dpi=110)
    else:
        ani.save(fname, writer="ffmpeg", fps=fps, dpi=110,
                 extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    print(f"  Animacao salva: {fname}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 8  FUNCOES DE ALTO NIVEL
# ══════════════════════════════════════════════════════════════════════════════

def gerar_plot(var_key, hora=0, proj_key="mercator", mostrar=False):
    """Baixa e plota um unico horario."""
    path = baixar_variavel(var_key, hora)
    if not path:
        return
    lats, lons, data = ler_campo(path, var_key)
    plot_estatico(lats, lons, data, var_key,
                  hora=hora, proj_key=proj_key,
                  salvar=True, mostrar=mostrar)


def gerar_animacao(var_key, horas, proj_key="mercator",
                   fps=4, formato="gif"):
    """Baixa serie temporal e gera animacao."""
    arquivos = baixar_serie(var_key, horas)
    frames = []
    for h, path in sorted(arquivos.items()):
        try:
            lats, lons, data = ler_campo(path, var_key)
            frames.append((lats, lons, data, h))
        except Exception as e:
            print(f"  Erro ao ler f{h:03d}: {e}")
    criar_animacao(frames, var_key,
                   proj_key=proj_key, fps=fps, formato=formato)


# ══════════════════════════════════════════════════════════════════════════════
# 9  HORARIOS PRE-DEFINIDOS
# ══════════════════════════════════════════════════════════════════════════════

# 16 dias completos (6 em 6 horas — balanco entre cobertura e velocidade)
HORAS_16DIAS = list(range(0, 121, 6)) + list(range(123, 385, 3))

# 5 dias hora a hora
HORAS_5DIAS_1H = list(range(0, 121))

# 10 dias (3 em 3h)
HORAS_10DIAS = list(range(0, 241, 3))

# 16 dias (3 em 3h — maximo de frames)
HORAS_16DIAS_3H = list(range(0, 121, 3)) + list(range(123, 385, 3))


# ══════════════════════════════════════════════════════════════════════════════
# 10  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print("  GFS Climate Plotter — Grib Filter Edition")
    print(f"  Rodada : {DATE} {CYCLE}Z")
    print(f"  Regiao : {REGIAO['nome'] if REGIAO else 'Global'}")
    print(f"  Output : {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)

    # 1) Plots estaticos: temperatura f000 em 2 projecoes
    print("\n[1] Plot estatico — Temperatura 2m — f000")
    gerar_plot("temperatura_2m", hora=0, proj_key="mercator")
    gerar_plot("temperatura_2m", hora=0, proj_key="ortho_brasil")

    # 2) Plot estatico: CAPE em f024
    print("\n[2] Plot estatico — CAPE — f024")
    gerar_plot("cape", hora=24, proj_key="mercator")

    # 3) Animacao: temperatura 2m — 16 dias
    print("\n[3] Animacao — Temperatura 2m — 16 dias (6h/6h)")
    gerar_animacao(
        var_key  = "temperatura_2m",
        horas    = HORAS_16DIAS,
        proj_key = "mercator",
        fps      = 6,
        formato  = "gif",
    )

    # 4) Animacao: precipitacao — 16 dias
    print("\n[4] Animacao — Precipitacao — 16 dias (6h/6h)")
    gerar_animacao(
        var_key  = "precipitacao",
        horas    = HORAS_16DIAS,
        proj_key = "mercator",
        fps      = 6,
        formato  = "gif",
    )

    # 5) Animacao: pressao MSL — 16 dias — ortho_brasil
    print("\n[5] Animacao — Pressao MSL — 16 dias — ortho_brasil")
    gerar_animacao(
        var_key  = "pressao_msl",
        horas    = HORAS_16DIAS,
        proj_key = "ortho_brasil",
        fps      = 6,
        formato  = "gif",
    )

    # 6) Animacao: umidade 850 hPa — 10 dias
    print("\n[6] Animacao — Umidade 850 hPa — 10 dias")
    gerar_animacao(
        var_key  = "umidade_850",
        horas    = HORAS_10DIAS,
        proj_key = "mercator",
        fps      = 5,
        formato  = "gif",
    )

    # 7) MP4 (descomente se tiver ffmpeg instalado):
    # print("\n[7] Animacao — CAPE — 5 dias 1h/1h — MP4")
    # gerar_animacao(
    #     var_key  = "cape",
    #     horas    = HORAS_5DIAS_1H,
    #     proj_key = "mercator",
    #     fps      = 12,
    #     formato  = "mp4",
    # )

    print("\nConcluido! Arquivos em:", os.path.abspath(OUTPUT_DIR))

    # ══════════════════════════════════════════════════════════════════════
    # REFERENCIA RAPIDA — usar como modulo em outro script:
    #
    #   from gfs_plots import gerar_plot, gerar_animacao, HORAS_16DIAS
    #
    #   gerar_plot("geopotencial_500", hora=48, proj_key="ortho_global")
    #
    #   gerar_animacao(
    #       var_key  = "agua_precipitavel",
    #       horas    = HORAS_16DIAS,
    #       proj_key = "robinson",
    #       fps      = 8,
    #       formato  = "gif",
    #   )
    #
    # Variaveis disponiveis:
    #   temperatura_2m, precipitacao, pressao_msl,
    #   vento_u_10m, vento_v_10m, umidade_850,
    #   geopotencial_500, cape, temperatura_850,
    #   orvalho_2m, agua_precipitavel, nuvens_total
    #
    # Projecoes disponiveis:
    #   ortho_brasil, ortho_americas, ortho_global,
    #   mercator, robinson, mollweide, platecarree
    # ══════════════════════════════════════════════════════════════════════