import cfgrib
from datetime import datetime
import requests
import os



DATA  = datetime(2026, 4, 3)
CICLO = "00"
F_HORA = 0
arquivo = f"gfs_global_{DATA.strftime('%Y%m%d')}_{CICLO}z_f{F_HORA:03d}.grib2"

BASE = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl"

params = {
    "file":      f"gfs.t{CICLO}z.pgrb2.0p25.f{F_HORA:03d}",
    "all_var":   "on",
    "all_lev":   "on",
    "leftlon":   0, "rightlon": 360,
    "toplat":    90, "bottomlat": -90,
    "dir": f"/gfs.{DATA.strftime('%Y%m%d')}/{CICLO}/atmos",
}

arquivo = f"gfs_global_{DATA.strftime('%Y%m%d')}_{CICLO}z_f{F_HORA:03d}.grib2"

if not os.path.exists(arquivo):
    print("Baixando...")
    r = requests.get(BASE, params=params, stream=True, timeout=300)
    r.raise_for_status()
    with open(arquivo, "wb") as f:
        for chunk in r.iter_content(1024*512):
            f.write(chunk)
    print(f"Salvo: {arquivo} ({os.path.getsize(arquivo)/1e6:.1f} MB)")

print("Abrindo todos os datasets do GRIB2...\n")
datasets = cfgrib.open_datasets(arquivo)

print(f'{len(datasets)} datasets encontrados')