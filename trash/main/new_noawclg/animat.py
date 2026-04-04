import requests
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
import os

# ─────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────
DATA_INICIO = datetime(2026, 4, 3)
CICLO       = "00"
# 16 dias = passos de 6h → 0,6,12,...,384
HORAS       = list(range(0, 385, 6))   # 65 frames
VARIAVEL    = "t2m"
BASE        = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl"
DIR_GRIB    = "grib_frames"
GIF_OUT     = "gfs_t2m_mercator_16d.gif"
FPS         = 8     # frames por segundo no GIF

os.makedirs(DIR_GRIB, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. DOWNLOAD DE TODOS OS FRAMES
# ─────────────────────────────────────────────────────────────
def baixar_frame(hora):
    nome = os.path.join(DIR_GRIB, f"gfs_{CICLO}z_f{hora:03d}.grib2")
    if os.path.exists(nome):
        return nome
    params = {
        "file":      f"gfs.t{CICLO}z.pgrb2.0p25.f{hora:03d}",
        "var_TMP":   "on",          # só temperatura → arquivo menor
        "lev_2_m_above_ground": "on",
        "leftlon":   0,  "rightlon":  360,
        "toplat":   90,  "bottomlat": -90,
        "dir": f"/gfs.{DATA_INICIO.strftime('%Y%m%d')}/{CICLO}/atmos",
    }
    r = requests.get(BASE, params=params, stream=True, timeout=180)
    if r.status_code != 200 or int(r.headers.get("Content-Length", 1)) < 1000:
        return None
    with open(nome, "wb") as f:
        for chunk in r.iter_content(1024 * 256):
            f.write(chunk)
    return nome

print(f"Baixando {len(HORAS)} frames (só TMP 2m — ~2-5 MB cada)...")
arquivos = []
for i, h in enumerate(HORAS):
    arq = baixar_frame(h)
    if arq:
        arquivos.append((h, arq))
        print(f"  [{i+1:02d}/{len(HORAS)}] f{h:03d} ✓", end="\r")
    else:
        print(f"  f{h:03d} indisponível — pulando")

print(f"\n{len(arquivos)} frames prontos.")

# ─────────────────────────────────────────────────────────────
# 2. LEITURA DOS FRAMES
# ─────────────────────────────────────────────────────────────
def ler_t2m(path):
    ds = xr.open_dataset(
        path, engine="cfgrib",
        filter_by_keys={"shortName": "2t", "typeOfLevel": "heightAboveGround"},
        backend_kwargs={"errors": "ignore"},
    )
    da = ds["t2m"] - 273.15   # K → °C
    lons = da.longitude.values
    lats = da.latitude.values
    vals = da.values
    # 0-360 → -180-180
    if lons.max() > 180:
        idx  = np.where(lons > 180, lons - 360, lons)
        order = np.argsort(idx)
        lons  = idx[order]
        vals  = vals[:, order]
    return lons, lats, vals

print("Lendo campos...")
frames = []
horas_ok = []
for h, arq in arquivos:
    try:
        lons, lats, vals = ler_t2m(arq)
        frames.append(vals)
        horas_ok.append(h)
    except Exception as e:
        print(f"  Erro f{h:03d}: {e}")

print(f"{len(frames)} frames carregados.")

if not frames:
    raise RuntimeError("Nenhum frame válido foi carregado; não é possível gerar o GIF.")

# ─────────────────────────────────────────────────────────────
# 3. ANIMAÇÃO GIF — MERCATOR
# ─────────────────────────────────────────────────────────────
PROJ   = ccrs.Mercator()
DATA_T = ccrs.PlateCarree()
VMIN, VMAX = -50, 45
CMAP   = "RdBu_r"
LEVELS = np.linspace(VMIN, VMAX, 25)

print("Gerando animação...")

fig = plt.figure(figsize=(14, 7), dpi=100)
ax  = plt.axes(projection=PROJ)
ax.set_global()
ax.add_feature(cfeature.COASTLINE,  linewidth=0.5, edgecolor="k")
ax.add_feature(cfeature.BORDERS,    linewidth=0.25, edgecolor="gray")

# frame inicial
cf = ax.contourf(lons, lats, frames[0],
                 levels=LEVELS, cmap=CMAP,
                 vmin=VMIN, vmax=VMAX,
                 transform=DATA_T, extend="both")

cb = plt.colorbar(cf, ax=ax, orientation="horizontal",
                  pad=0.03, shrink=0.75, aspect=40)
cb.set_label("Temperatura a 2m (°C)", fontsize=10)

gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                  color="gray", alpha=0.5, linestyle="--")
gl.top_labels   = False
gl.right_labels = False

titulo = ax.set_title("", fontsize=11, pad=8)

def atualizar(i):
    global cf
    # Matplotlib/Cartopy recentes podem não expor `collections` em GeoContourSet.
    collections = getattr(cf, "collections", None)
    if collections is not None:
        for coll in collections:
            coll.remove()
    elif hasattr(cf, "remove"):
        cf.remove()

    cf = ax.contourf(lons, lats, frames[i],
                     levels=LEVELS, cmap=CMAP,
                     vmin=VMIN, vmax=VMAX,
                     transform=DATA_T, extend="both")
    dt_frame = DATA_INICIO + timedelta(hours=horas_ok[i])
    titulo.set_text(
        f"GFS T2m (°C) — {DATA_INICIO.strftime('%Y-%m-%d')} {CICLO}z  "
        f"+{horas_ok[i]:03d}h  [{dt_frame.strftime('%d/%m %HZ')}]"
    )
    return [titulo]

ani = animation.FuncAnimation(
    fig, atualizar,
    frames=len(frames),
    interval=1000 // FPS,
    blit=False,
)

# salva GIF com Pillow
writer = animation.PillowWriter(fps=FPS)
ani.save(GIF_OUT, writer=writer, dpi=100)
plt.close(fig)

print(f"\nGIF salvo: {GIF_OUT}")
print("  Resolução : 1400×700 px")
print(f"  Frames    : {len(frames)}")
print(f"  Duração   : ~{len(frames)/FPS:.0f}s  ({FPS} fps)")