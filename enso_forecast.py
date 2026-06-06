#!/usr/bin/env python3
"""
enso_forecast.py
================
Análise observacional de precursores de ENOS e avaliação
probabilística do desenvolvimento de El Niño nos próximos meses.

Metodologia
-----------
Seis indicadores observacionais são combinados para estimar
a probabilidade de El Niño:

1. ONI atual (Oceanic Niño Index)         — estado corrente
2. Tendência do ONI (últimos 3 meses)     — direção do sinal
3. Anomalia T200 no Niño 3.4              — precursor subsuperficial
4. Anomalia de SSH no Pacífico central/E  — sinal dinâmico (ondas Kelvin)
5. Inclinação da termoclina L-O           — diagnóstico dinâmico
6. Análogos históricos (ERSST 1950-2025)  — base empírica

Fontes de dados
---------------
- NOAA ERSST v5   : TSM mensal, 1854–presente  (via OPeNDAP)
- NOAA GODAS      : pottmp, salt, sshg, 1980–presente (via OPeNDAP)

Saída
-----
- gfs_plots/enso/enso_analysis.png   (figura principal 4×2 painéis)
- gfs_plots/enso/enso_indicators.png (indicadores líderes)
- Resumo textual no terminal
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ── Library imports — all data loading delegated to noawclg.ocean ──────────────
from noawclg.ocean import (
    get_ocean_temp,
    get_ssh,
    get_sst_series,
    NINO_BOXES,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
LOG = logging.getLogger(__name__)

# ── Saída ──────────────────────────────────────────────────────────────────────
OUT_DIR = Path("gfs_plots/enso")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuração ───────────────────────────────────────────────────────────────
TODAY         = datetime.today()
CURRENT_YEAR  = TODAY.year
LATEST_GODAS  = 2026          # atualizar quando novo ano disponível
LATEST_GODAS_MONTHS = 4       # quantos meses disponíveis no ano corrente

CLIM_START, CLIM_END = 1991, 2020   # climatologia padrão WMO

# Caixas ENOS (lon 0–360)
NINO34 = dict(lat=(-5.0,  5.0), lon=(190.0, 240.0))
NINO3  = dict(lat=(-5.0,  5.0), lon=(210.0, 270.0))
WWV    = dict(lat=(-5.0,  5.0), lon=(120.0, 280.0))
WP     = dict(lat=(-5.0,  5.0), lon=(120.0, 160.0))   # Pacífico Oeste
EP     = dict(lat=(-5.0,  5.0), lon=(205.0, 270.0))   # Pacífico Leste

# El Niño forte histórico (para referência nos gráficos)
NINO_STRONG_YEARS   = [1982, 1997, 2015, 2023]
NINO_MODERATE_YEARS = [1987, 1994, 2002, 2009, 2014, 2018]

PHASE_COLORS = {
    "El Niño":  "#d73027",
    "La Niña":  "#4575b4",
    "Neutral":  "#d0d0d0",
}

# ══════════════════════════════════════════════════════════════════════════════
# Funções de dados  — delegam para noawclg.ocean
# ══════════════════════════════════════════════════════════════════════════════

def _ersst_nino34(year_start: int, year_end: int) -> pd.Series:
    """Série mensal de TSM na caixa Niño 3.4 via ERSST v5 (noawclg.ocean)."""
    LOG.info("Carregando ERSST Niño 3.4 %d–%d …", year_start, year_end)
    return get_sst_series(year_start, year_end, box="3.4", source="ersst")


def _compute_oni(sst: pd.Series, clim_s: int = CLIM_START,
                 clim_e: int = CLIM_END) -> pd.Series:
    """ONI = média móvel 3 meses da anomalia de TSM Niño 3.4."""
    clim         = sst[(sst.index.year >= clim_s) & (sst.index.year <= clim_e)]
    monthly_mean = clim.groupby(clim.index.month).mean()
    anom = sst.copy()
    for m, v in monthly_mean.items():
        anom[anom.index.month == m] -= v
    oni      = anom.rolling(3, center=True, min_periods=2).mean()
    oni.name = "ONI"
    return oni


def _classify(oni: float) -> str:
    if oni >= 2.0:  return "El Niño Extremo"
    if oni >= 1.5:  return "El Niño Forte"
    if oni >= 0.5:  return "El Niño"
    if oni <= -1.5: return "La Niña Forte"
    if oni <= -0.5: return "La Niña"
    return "Neutro"


def _godas_t200_nino34(year_start: int, year_end: int) -> pd.Series:
    """Temperatura média a ~200 m na caixa Niño 3.4 (noawclg.ocean)."""
    LOG.info("Carregando GODAS T200 %d–%d …", year_start, year_end)
    b   = NINO_BOXES["3.4"]
    reg = {"lat_min": b["lat"][0], "lat_max": b["lat"][1],
           "lon_min": b["lon"][0], "lon_max": b["lon"][1]}
    da  = get_ocean_temp(year_start, year_end, depth_m=200.0, region=reg)
    mean = da.mean(["lat", "lon"])
    s   = pd.Series(mean.values, index=pd.DatetimeIndex(mean["time"].values),
                    name="T200_Nino34")
    return s


def _t200_anom(t200: pd.Series, clim_s: int = 2000, clim_e: int = 2020) -> pd.Series:
    """Anomalia de T200 relativa à climatologia GODAS."""
    if not isinstance(t200.index, pd.DatetimeIndex) or len(t200) == 0:
        return pd.Series(dtype=float, name="T200_anom")
    clim = t200[(t200.index.year >= clim_s) & (t200.index.year <= clim_e)]
    mm   = clim.groupby(clim.index.month).mean()
    a    = t200.copy()
    for m, v in mm.items():
        a[a.index.month == m] -= v
    a.name = "T200_anom"
    return a


def _godas_ssh_map(year: int, month_idx: int = 0) -> xr.DataArray:
    """SSH global para um mês específico (noawclg.ocean)."""
    LOG.info("Carregando GODAS SSH %d mês %d …", year, month_idx + 1)
    return get_ssh(year, year).isel(time=month_idx)


def _godas_t200_map(year: int, month_idx: int = 0) -> xr.DataArray:
    """T200 global para um mês específico (noawclg.ocean)."""
    LOG.info("Carregando GODAS T200 mapa %d mês %d …", year, month_idx + 1)
    return get_ocean_temp(year, year, depth_m=200.0).isel(time=month_idx)


def _historical_analogs(
    oni_full: pd.Series,
    n_months: int = 5,
    top_k: int = 6,
) -> tuple[list[int], pd.DataFrame]:
    """
    Encontra os anos históricos mais análogos ao ano atual (Jan–mês corrente).

    Retorna os k anos mais similares e a trajetória completa de cada um.
    """
    current_yr  = oni_full.index[-1].year
    # Trajetória atual: ONI Jan–mês_mais_recente do ano corrente
    cur_trace = oni_full[oni_full.index.year == current_yr]
    if len(cur_trace) < 2:
        cur_trace = oni_full[oni_full.index.year == current_yr - 1]
        current_yr -= 1
    cur_mean = float(cur_trace.mean())
    cur_trend = float(np.polyfit(range(len(cur_trace)), cur_trace.values, 1)[0])

    records = []
    for yr in range(1950, current_yr):
        yr_trace = oni_full[oni_full.index.year == yr].iloc[:n_months]
        if len(yr_trace) < n_months:
            continue
        # Comparação: média + tendência dos primeiros n_months meses
        yr_mean  = float(yr_trace.mean())
        yr_trend = float(np.polyfit(range(len(yr_trace)), yr_trace.values, 1)[0])
        dist = np.sqrt((yr_mean - cur_mean)**2 + 3.0 * (yr_trend - cur_trend)**2)
        # Pegar o ONI de Jul–Dez daquele ano (resultado)
        full_yr = oni_full[oni_full.index.year == yr]
        end_oni = float(full_yr.iloc[-1]) if len(full_yr) == 12 else np.nan
        records.append(dict(year=yr, dist=dist, mean=yr_mean,
                            trend=yr_trend, end_oni=end_oni))

    df = pd.DataFrame(records).sort_values("dist").head(top_k).reset_index(drop=True)
    analog_years = df["year"].tolist()

    # Montar trajetória mensalizada para plot (mês 1–12)
    traces = {}
    for yr in analog_years:
        tr = oni_full[oni_full.index.year == yr]
        if len(tr) == 12:
            traces[yr] = tr.values
    traces_df = pd.DataFrame(traces, index=range(1, 13))
    return analog_years, traces_df, df


def _analog_el_nino_prob(analog_df: pd.DataFrame, threshold: float = 0.5) -> float:
    """Fração dos análogos que resultou em El Niño no 2º semestre."""
    valid = analog_df["end_oni"].dropna()
    if len(valid) == 0:
        return 0.5
    return float((valid >= threshold).mean())


# ══════════════════════════════════════════════════════════════════════════════
# Avaliação de probabilidade multi-indicador
# ══════════════════════════════════════════════════════════════════════════════

def assess_probability(
    oni_current: float,
    oni_trend_3m: float,
    t200_anom_current: float,
    ssh_ep_current: float,
    analog_prob: float,
) -> dict:
    """
    Combina 5 indicadores em uma probabilidade de El Niño para os próximos
    6 meses.  Pesos e pontuações baseados em literatura de previsão de ENOS.

    Retorna dict com pontuação por indicador, probabilidade final e categoria.
    """
    scores = {}

    # 1. ONI atual (peso 0.30)
    if   oni_current >= 2.0: scores["oni"] = 1.00
    elif oni_current >= 1.5: scores["oni"] = 0.92
    elif oni_current >= 0.5: scores["oni"] = 0.75
    elif oni_current >= 0.0: scores["oni"] = 0.45
    elif oni_current >= -0.5: scores["oni"] = 0.20
    else:                    scores["oni"] = 0.05

    # 2. Tendência do ONI — 3 últimos meses (peso 0.20)
    if   oni_trend_3m >= 0.4:  scores["trend"] = 0.92
    elif oni_trend_3m >= 0.15: scores["trend"] = 0.72
    elif oni_trend_3m >= 0.0:  scores["trend"] = 0.50
    elif oni_trend_3m >= -0.2: scores["trend"] = 0.30
    else:                      scores["trend"] = 0.10

    # 3. Anomalia T200 Niño 3.4 (peso 0.25) — precursor subsuperficial
    if   t200_anom_current >= 1.5: scores["t200"] = 0.95
    elif t200_anom_current >= 0.8: scores["t200"] = 0.82
    elif t200_anom_current >= 0.3: scores["t200"] = 0.65
    elif t200_anom_current >= 0.0: scores["t200"] = 0.48
    elif t200_anom_current >= -0.5: scores["t200"] = 0.25
    else:                           scores["t200"] = 0.10

    # 4. SSH no Pacífico Leste (proxy de onda de Kelvin) (peso 0.15)
    if   ssh_ep_current >= 0.10: scores["ssh"] = 0.88
    elif ssh_ep_current >= 0.04: scores["ssh"] = 0.68
    elif ssh_ep_current >= 0.0:  scores["ssh"] = 0.50
    elif ssh_ep_current >= -0.05: scores["ssh"] = 0.32
    else:                         scores["ssh"] = 0.12

    # 5. Análogos históricos (peso 0.10)
    scores["analog"] = float(np.clip(analog_prob, 0.0, 1.0))

    # Probabilidade ponderada
    weights = dict(oni=0.30, trend=0.20, t200=0.25, ssh=0.15, analog=0.10)
    prob = sum(scores[k] * weights[k] for k in scores)

    # Categoria qualitativa
    if   prob >= 0.75: category = "Alta"
    elif prob >= 0.55: category = "Moderada-Alta"
    elif prob >= 0.40: category = "Moderada"
    elif prob >= 0.25: category = "Baixa-Moderada"
    else:              category = "Baixa"

    return dict(prob=prob, scores=scores, weights=weights, category=category)


# ══════════════════════════════════════════════════════════════════════════════
# Figuras
# ══════════════════════════════════════════════════════════════════════════════

def _plot_oni_series(ax: plt.Axes, oni: pd.Series, title: str = "ONI 2010–2026") -> None:
    t, v = oni.index, oni.values
    ax.fill_between(t, v, 0, where=v >= 0.5, color="#d73027", alpha=0.35, label="El Niño (≥+0.5°C)")
    ax.fill_between(t, v, 0, where=v <= -0.5, color="#4575b4", alpha=0.35, label="La Niña (≤-0.5°C)")
    ax.fill_between(t, v, 0, where=(v > -0.5) & (v < 0.5), color="#d0d0d0", alpha=0.35, label="Neutro")
    ax.plot(t, v, color="#222222", lw=1.5, zorder=3)

    # Destaque El Niño fortes históricos
    for yr in NINO_STRONG_YEARS:
        yr_vals = oni[oni.index.year == yr]
        if len(yr_vals):
            ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="#d73027",
                       lw=0.8, ls=":", alpha=0.6)
            ax.text(pd.Timestamp(f"{yr}-07-01"), float(v.max()) * 0.92,
                    str(yr), color="#d73027", fontsize=7.5, ha="center")

    ax.axhline(0.5, color="#d73027", lw=0.7, ls="--", alpha=0.7)
    ax.axhline(-0.5, color="#4575b4", lw=0.7, ls="--", alpha=0.7)
    ax.axhline(0, color="#555555", lw=0.5)
    ax.set_ylabel("Anomalia TSM (°C)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", ls="--", alpha=0.3)


def _plot_t200_series(ax: plt.Axes, t200_anom: pd.Series) -> None:
    t, v = t200_anom.index, t200_anom.values
    ax.fill_between(t, v, 0, where=v > 0, color="#d73027", alpha=0.4)
    ax.fill_between(t, v, 0, where=v < 0, color="#4575b4", alpha=0.4)
    ax.plot(t, v, color="#222222", lw=1.4)
    ax.axhline(0, color="#555555", lw=0.5)
    ax.axhline(0.5, color="#d73027", lw=0.7, ls="--", alpha=0.7)
    ax.axhline(-0.5, color="#4575b4", lw=0.7, ls="--", alpha=0.7)
    ax.set_ylabel("Anomalia T200 (°C)", fontsize=9)
    ax.set_title("Temperatura 200 m — Anomalia Niño 3.4", fontsize=10, fontweight="bold")
    ax.grid(axis="y", ls="--", alpha=0.3)


def _plot_field_map(ax: plt.Axes, da: xr.DataArray,
                    title: str, cmap: str, vmin: float, vmax: float,
                    label: str = "", symmetric: bool = False) -> None:
    lons = da["lon"].values.copy()
    lats = da["lat"].values
    data = da.values

    if lons.max() > 180:
        lons[lons > 180] -= 360
        si = np.argsort(lons)
        lons, data = lons[si], data[:, si]

    if symmetric:
        absmax = max(abs(vmin), abs(vmax))
        vmin, vmax = -absmax, absmax

    pcm = ax.pcolormesh(lons, lats, data, cmap=cmap,
                        vmin=vmin, vmax=vmax, shading="auto")
    plt.colorbar(pcm, ax=ax, orientation="horizontal",
                 fraction=0.048, pad=0.08, label=label)

    # Caixas Niño
    _BOX_COLORS = {"3.4": "#ff7f00", "3": "#984ea3"}
    for name, b in [("3.4", NINO34), ("3", NINO3)]:
        lon0 = b["lon"][0] - 360 if b["lon"][0] > 180 else b["lon"][0]
        lon1 = b["lon"][1] - 360 if b["lon"][1] > 180 else b["lon"][1]
        rect = mpatches.Rectangle(
            (lon0, b["lat"][0]), lon1 - lon0, b["lat"][1] - b["lat"][0],
            lw=1.5, edgecolor=_BOX_COLORS[name], facecolor="none", zorder=4)
        ax.add_patch(rect)
        ax.text(lon0 + 2, b["lat"][1] + 1.0, f"Niño {name}",
                color=_BOX_COLORS[name], fontsize=7, fontweight="bold")

    ax.set_xlim(-180, 180);  ax.set_ylim(-30, 30)
    ax.axhline(0, color="k", lw=0.4, ls="--", alpha=0.5)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(ls="--", alpha=0.25)


def _plot_analogs(ax: plt.Axes, traces_df: pd.DataFrame,
                  oni_current_year: pd.Series, analog_df: pd.DataFrame) -> None:
    """Trajetória do ONI dos análogos históricos vs. ano corrente."""
    months = list(range(1, 13))
    month_labels = ["J","F","M","A","M","J","J","A","S","O","N","D"]

    # Análogos históricos
    for yr in traces_df.columns:
        end = float(analog_df.loc[analog_df.year == yr, "end_oni"].values[0])
        col = "#d73027" if end >= 0.5 else ("#4575b4" if end <= -0.5 else "#888888")
        ax.plot(months, traces_df[yr], color=col, alpha=0.45, lw=1.1,
                label=f"{yr} ({end:+.1f}°C)")

    # Ano atual
    cur_months = oni_current_year.index.month.tolist()
    ax.plot(cur_months, oni_current_year.values,
            color="black", lw=2.4, zorder=5, label=f"2026 (atual)")

    ax.axhline(0.5, color="#d73027", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(-0.5, color="#4575b4", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(0, color="#555555", lw=0.5)
    ax.set_xticks(months); ax.set_xticklabels(month_labels, fontsize=8)
    ax.set_ylabel("ONI (°C)", fontsize=9)
    ax.set_title("Análogos históricos vs. 2026", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.85, ncol=2)
    ax.grid(ls="--", alpha=0.3)


def _plot_assessment(ax: plt.Axes, result: dict,
                     oni_current: float, t200_anom: float,
                     ssh_ep: float, analog_prob: float,
                     current_phase: str) -> None:
    """Painel textual com o resumo da avaliação."""
    ax.axis("off")
    prob     = result["prob"]
    category = result["category"]
    scores   = result["scores"]

    # Título
    ax.text(0.5, 0.97, "Avaliação de El Niño — Próximos 6 Meses",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color="#1a1a1a")

    # Barra de probabilidade
    bar_ax = ax.inset_axes([0.05, 0.80, 0.90, 0.08])
    cmap_prob = plt.cm.RdYlGn_r
    for i, (label_p, lim) in enumerate([
        ("Baixa\n(<25%)", 0.25), ("Mod.\n(40%)", 0.40),
        ("Mod-A\n(55%)", 0.55), ("Alta\n(>75%)", 1.0),
    ]):
        pass
    bar_ax.barh([0], [prob], color=cmap_prob(prob), height=0.7)
    bar_ax.barh([0], [1.0],  color="none", height=0.7,
                edgecolor="#888", linewidth=0.8)
    bar_ax.set_xlim(0, 1); bar_ax.set_ylim(-0.5, 0.5)
    bar_ax.axvline(0.25, color="#777", lw=0.7, ls=":")
    bar_ax.axvline(0.50, color="#777", lw=0.7, ls=":")
    bar_ax.axvline(0.75, color="#777", lw=0.7, ls=":")
    bar_ax.text(prob + 0.01, 0, f"{prob*100:.0f}%",
                va="center", fontsize=10, fontweight="bold",
                color=cmap_prob(prob))
    bar_ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    bar_ax.set_xticklabels(["0%","25%","50%","75%","100%"], fontsize=7)
    bar_ax.tick_params(left=False, labelleft=False)
    bar_ax.set_title(f"Probabilidade El Niño: {category}", fontsize=9, pad=4)

    # Indicadores individuais
    indicator_names = {
        "oni":    f"ONI atual ({oni_current:+.2f}°C)",
        "trend":  "Tendência ONI (3 meses)",
        "t200":   f"Anom. T200 Niño 3.4 ({t200_anom:+.2f}°C)",
        "ssh":    f"SSH Pacífico Leste ({ssh_ep*100:+.1f} cm)",
        "analog": f"Análogos históricos ({analog_prob*100:.0f}%→ElNiño)",
    }
    y0 = 0.67
    ax.text(0.05, y0, "Indicadores:", transform=ax.transAxes,
            fontsize=9, fontweight="bold")
    for i, (k, name) in enumerate(indicator_names.items()):
        sc = scores[k]
        color = "#d73027" if sc > 0.6 else ("#4575b4" if sc < 0.4 else "#f0a500")
        marker = "▲" if sc > 0.6 else ("▼" if sc < 0.4 else "●")
        ax.text(0.08, y0 - 0.11 * (i + 1), f"{marker}  {name}",
                transform=ax.transAxes, fontsize=8.5,
                color=color, va="top")
        ax.text(0.88, y0 - 0.11 * (i + 1), f"{sc*100:.0f}pts",
                transform=ax.transAxes, fontsize=8, color=color,
                va="top", ha="right")

    # Fase atual
    phase_col = {"El Niño": "#d73027", "La Niña": "#4575b4"}.get(
        current_phase, "#555555")
    ax.text(0.5, 0.03,
            f"Estado atual: {current_phase}  |  Atualizado: {TODAY.strftime('%B/%Y')}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8, color=phase_col, fontstyle="italic")


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

def run_analysis() -> None:

    # ── 1. ONI histórico e atual (ERSST) ────────────────────────────────────
    LOG.info("=== Carregando dados ERSST ===")
    sst_full  = _ersst_nino34(1950, LATEST_GODAS)
    oni_full  = _compute_oni(sst_full)

    oni_recent  = oni_full["2010":]
    oni_current = oni_full[str(LATEST_GODAS)]

    oni_last_val    = float(oni_full.dropna().iloc[-1])
    oni_3m_vals     = oni_full.dropna().iloc[-4:]
    oni_trend_3m    = float(np.polyfit(range(len(oni_3m_vals)),
                                       oni_3m_vals.values, 1)[0])
    current_phase   = _classify(oni_last_val)

    # ── 2. T200 Niño 3.4 (GODAS) ────────────────────────────────────────────
    LOG.info("=== Carregando T200 GODAS ===")
    t200_raw  = _godas_t200_nino34(2000, LATEST_GODAS)
    t200_anom = _t200_anom(t200_raw)
    t200_recent = t200_anom["2020":]
    t200_cur    = float(t200_anom.dropna().iloc[-1])

    # ── 3. SSH map (GODAS, mês mais recente) ────────────────────────────────
    LOG.info("=== Carregando SSH GODAS ===")
    ssh_month_idx = LATEST_GODAS_MONTHS - 1
    try:
        ssh_map = _godas_ssh_map(LATEST_GODAS, ssh_month_idx)
        # SSH médio no Pacífico Leste como indicador
        ssh_ep = float(
            ssh_map.sel(
                lat=slice(EP["lat"][0], EP["lat"][1]),
                lon=slice(EP["lon"][0], EP["lon"][1]),
            ).mean()
        )
    except Exception as e:
        LOG.warning("SSH falhou: %s", e)
        ssh_map = None
        ssh_ep  = 0.0

    # ── 4. T200 map (GODAS, mês mais recente) ───────────────────────────────
    try:
        t200_map = _godas_t200_map(LATEST_GODAS, ssh_month_idx)
    except Exception as e:
        LOG.warning("T200 map falhou: %s", e)
        t200_map = None

    # ── 5. Análogos históricos ───────────────────────────────────────────────
    LOG.info("=== Calculando análogos históricos ===")
    n_months_cur = len(oni_current.dropna())
    analog_years, traces_df, analog_df = _historical_analogs(
        oni_full, n_months=n_months_cur, top_k=6)
    analog_prob = _analog_el_nino_prob(analog_df)

    # ── 6. Avaliação de probabilidade ────────────────────────────────────────
    result = assess_probability(
        oni_current=oni_last_val,
        oni_trend_3m=oni_trend_3m,
        t200_anom_current=t200_cur,
        ssh_ep_current=ssh_ep,
        analog_prob=analog_prob,
    )

    # ── 7. Resumo textual ────────────────────────────────────────────────────
    _print_summary(oni_last_val, oni_trend_3m, t200_cur, ssh_ep,
                   analog_prob, analog_years, result, current_phase)

    # ── 8. Figura principal ──────────────────────────────────────────────────
    LOG.info("=== Gerando figuras ===")
    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(4, 2, figure=fig,
                            height_ratios=[1.2, 0.9, 1.0, 1.1],
                            hspace=0.42, wspace=0.30)

    # Painel 1 — ONI série temporal completa (linha toda)
    ax1 = fig.add_subplot(gs[0, :])
    _plot_oni_series(ax1, oni_recent)

    # Painel 2 — T200 anomalia temporal
    ax2 = fig.add_subplot(gs[1, 0])
    _plot_t200_series(ax2, t200_recent)

    # Painel 3 — SSH map ou fallback
    ax3 = fig.add_subplot(gs[1, 1])
    if ssh_map is not None:
        month_name = TODAY.strftime("%B/%Y") if ssh_month_idx == LATEST_GODAS_MONTHS - 1 \
            else f"Mês {ssh_month_idx+1}/{LATEST_GODAS}"
        _plot_field_map(ax3, ssh_map,
                        f"SSH — {month_name}",
                        cmap="RdBu_r", vmin=-0.2, vmax=0.2,
                        label="SSH (m)", symmetric=True)
    else:
        ax3.text(0.5, 0.5, "SSH não disponível", transform=ax3.transAxes,
                 ha="center", va="center", fontsize=11)
        ax3.set_title("Sea Surface Height", fontsize=10)

    # Painel 4 — T200 map
    ax4 = fig.add_subplot(gs[2, 0])
    if t200_map is not None:
        _plot_field_map(ax4, t200_map,
                        f"Temperatura 200 m — Abril/{LATEST_GODAS}",
                        cmap="RdYlBu_r", vmin=4, vmax=26,
                        label="Temperatura (°C)")
    else:
        ax4.text(0.5, 0.5, "T200 map não disponível",
                 transform=ax4.transAxes, ha="center", va="center")
        ax4.set_title("T200 map", fontsize=10)

    # Painel 5 — Análogos históricos
    ax5 = fig.add_subplot(gs[2, 1])
    _plot_analogs(ax5, traces_df, oni_current.dropna(), analog_df)

    # Painel 6 — Avaliação
    ax6 = fig.add_subplot(gs[3, 0])
    _plot_assessment(ax6, result, oni_last_val, t200_cur, ssh_ep,
                     analog_prob, current_phase)

    # Painel 7 — Barra de indicadores
    ax7 = fig.add_subplot(gs[3, 1])
    _plot_indicator_bars(ax7, result)

    # Título geral
    fig.suptitle(
        f"Análise de El Niño — noawclg  |  {TODAY.strftime('%d/%m/%Y')}",
        fontsize=13, fontweight="bold", y=1.002,
    )

    out = OUT_DIR / "enso_analysis.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Figura salva: %s", out)

    # ── 9. Figura de indicadores líderes ────────────────────────────────────
    _make_indicators_figure(oni_full, t200_anom, result)


def _plot_indicator_bars(ax: plt.Axes, result: dict) -> None:
    """Gráfico de barras horizontais com pontuação de cada indicador."""
    scores  = result["scores"]
    weights = result["weights"]

    labels = [
        "ONI atual",
        "Tendência ONI",
        "T200 Anomalia",
        "SSH Pacífico L.",
        "Análogos Hist.",
    ]
    keys   = ["oni", "trend", "t200", "ssh", "analog"]
    vals   = [scores[k] * 100 for k in keys]
    colors = ["#d73027" if v > 60 else ("#4575b4" if v < 40 else "#f0a500")
              for v in vals]

    y = np.arange(len(labels))
    bars = ax.barh(y, vals, color=colors, alpha=0.82, height=0.55)
    ax.set_xlim(0, 105)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Pontuação (0–100)", fontsize=8)
    ax.set_title("Pontuação por indicador", fontsize=10, fontweight="bold")

    # Percentual de peso
    for i, (bar, k) in enumerate(zip(bars, keys)):
        w   = weights[k] * 100
        val = vals[i]
        ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}pt  (peso {w:.0f}%)",
                va="center", fontsize=8, color="#333333")

    ax.axvline(50, color="#aaa", lw=0.8, ls="--")
    ax.grid(axis="x", ls="--", alpha=0.3)


def _make_indicators_figure(
    oni_full: pd.Series,
    t200_anom: pd.Series,
    result: dict,
) -> None:
    """Segunda figura: séries históricas longas + contexto climatológico."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    fig.suptitle("Contexto Histórico ENOS — noawclg", fontsize=12,
                 fontweight="bold")

    # Painel A — ONI histórico longo (desde 1950)
    ax = axes[0]
    oni50 = oni_full["1950":]
    t, v  = oni50.index, oni50.values
    ax.fill_between(t, v, 0, where=v >= 0.5,  color="#d73027", alpha=0.40)
    ax.fill_between(t, v, 0, where=v <= -0.5, color="#4575b4", alpha=0.40)
    ax.plot(t, v, color="#111111", lw=0.9)
    ax.axhline(0.5, color="#d73027", lw=0.7, ls="--", alpha=0.6)
    ax.axhline(-0.5, color="#4575b4", lw=0.7, ls="--", alpha=0.6)
    ax.axhline(0, color="#555", lw=0.4)
    ax.set_ylabel("ONI (°C)", fontsize=9)
    ax.set_title("ONI histórico 1950–2026 (ERSST v5)", fontsize=10, fontweight="bold")
    ax.grid(axis="y", ls="--", alpha=0.3)

    # Destaque eventos El Niño fortes
    for yr in [1972, 1982, 1997, 2015, 2023]:
        ax.axvline(pd.Timestamp(f"{yr}-06-01"), color="#d73027",
                   lw=0.7, ls=":", alpha=0.7)
        ax.text(pd.Timestamp(f"{yr}-06-01"), float(v.max()) * 0.87,
                str(yr), color="#d73027", fontsize=7, ha="center",
                rotation=90, va="top")

    # Painel B — Sazonalidade do ONI (ciclo anual)
    ax2 = axes[1]
    oni_clim = oni_full[
        (oni_full.index.year >= CLIM_START) &
        (oni_full.index.year <= CLIM_END)
    ]
    monthly_mean = oni_clim.groupby(oni_clim.index.month).mean()
    monthly_std  = oni_clim.groupby(oni_clim.index.month).std()

    months     = list(range(1, 13))
    month_labs = ["Jan","Fev","Mar","Abr","Mai","Jun",
                  "Jul","Ago","Set","Out","Nov","Dez"]

    ax2.fill_between(months,
                     monthly_mean - monthly_std,
                     monthly_mean + monthly_std,
                     alpha=0.25, color="#888", label="±1σ climatologia")
    ax2.plot(months, monthly_mean.values, "o-", color="#555", lw=1.4,
             ms=5, label="Climatologia 1991–2020")

    # Sobreposição do ano atual
    cur = oni_full[str(LATEST_GODAS)].dropna()
    if len(cur):
        ax2.plot(cur.index.month.tolist(), cur.values,
                 "s-", color="#d73027", lw=2.0, ms=6,
                 label=f"2026 (atual)", zorder=5)

    ax2.axhline(0.5, color="#d73027", lw=0.7, ls="--", alpha=0.6)
    ax2.axhline(-0.5, color="#4575b4", lw=0.7, ls="--", alpha=0.6)
    ax2.axhline(0, color="#555", lw=0.4)
    ax2.set_xticks(months); ax2.set_xticklabels(month_labs, fontsize=8)
    ax2.set_ylabel("ONI (°C)", fontsize=9)
    ax2.set_title("Ciclo sazonal do ONI e posição de 2026 na climatologia",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right", framealpha=0.85)
    ax2.grid(axis="y", ls="--", alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "enso_indicators.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Figura salva: %s", out)


# ══════════════════════════════════════════════════════════════════════════════
# Resumo textual
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(oni: float, trend: float, t200: float, ssh: float,
                   analog_prob: float, analogs: list,
                   result: dict, phase: str) -> None:
    prob     = result["prob"]
    category = result["category"]
    bar      = "█" * round(prob * 20) + "░" * (20 - round(prob * 20))

    print()
    print("=" * 62)
    print("  ANÁLISE DE EL NIÑO — noawclg")
    print(f"  {TODAY.strftime('%d de %B de %Y')}")
    print("=" * 62)
    print()
    print(f"  Estado atual do ENOS : {phase}")
    print(f"  ONI mais recente     : {oni:+.2f} °C")
    print(f"  Tendência ONI (3m)   : {trend:+.3f} °C/mês")
    print(f"  T200 anom. Niño 3.4  : {t200:+.2f} °C")
    print(f"  SSH Pacífico Leste   : {ssh*100:+.1f} cm")
    print()
    print(f"  Análogos históricos  : {analogs}")
    print(f"  El Niño nos análogos : {analog_prob*100:.0f}% das ocorrências")
    print()
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │  PROBABILIDADE EL NIÑO (6 meses)            │")
    print(f"  │  [{bar}] {prob*100:.0f}%   │")
    print(f"  │  Classificação: {category:<28} │")
    print(f"  └─────────────────────────────────────────────┘")
    print()
    print("  Interpretação dos indicadores:")
    descs = {
        "oni":    f"ONI atual ({oni:+.2f}°C)",
        "trend":  f"Tendência {trend:+.3f}°C/mês",
        "t200":   f"T200 anom. {t200:+.2f}°C",
        "ssh":    f"SSH L.P. {ssh*100:+.1f}cm",
        "analog": f"Análogos {analog_prob*100:.0f}%",
    }
    for k, desc in descs.items():
        s = result["scores"][k]
        bar_i = "█" * round(s * 10) + "░" * (10 - round(s * 10))
        arrow = "▲" if s > 0.6 else ("▼" if s < 0.4 else "●")
        print(f"    {arrow} {desc:<30} [{bar_i}] {s*100:.0f}pt")

    print()
    _print_enso_context(result["prob"])
    print()
    print(f"  Figuras salvas em: {OUT_DIR.resolve()}")
    print("=" * 62)


def _print_enso_context(prob: float) -> None:
    if prob >= 0.75:
        msg = ("  O conjunto de indicadores aponta fortemente para El Niño.\n"
               "  Subsuperfície quente e ONI positivo convergem: aquecimento\n"
               "  provavelmente se consolidará nos próximos trimestres.\n"
               "  Impactos esperados: seca no NE do Brasil e norte da Austrália,\n"
               "  chuvas acima da média na costa do Peru e sul dos EUA.")
    elif prob >= 0.55:
        msg = ("  Sinais favoráveis ao El Niño, mas ainda inconclusivos.\n"
               "  Monitorar a propagação de ondas de Kelvin no Pacífico\n"
               "  e a evolução do WWV nas próximas 4–8 semanas.\n"
               "  Período crítico: Jun–Ago 2026 (barreira de previsibilidade\n"
               "  de primavera boreal já foi transposta).")
    elif prob >= 0.40:
        msg = ("  Indicadores mistos — sem sinal claro de El Niño.\n"
               "  Condições neutras ou de transição são plausíveis.\n"
               "  Probabilidade de La Niña moderada se os padrões\n"
               "  subsuperficiais se inverterem.")
    else:
        msg = ("  Indicadores apontam para La Niña ou condições neutras.\n"
               "  Subsuperfície mais fria que o normal ou ONI negativo\n"
               "  reduzem a probabilidade de El Niño nos próximos meses.\n"
               "  Monitorar: se a anomalia negativa persistir ≥ 3 meses,\n"
               "  La Niña pode se desenvolver.")

    print(msg)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_analysis()
