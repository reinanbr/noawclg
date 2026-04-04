import requests, os

base = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl"

params_base = {
    "dir":    "/gfs.20260403/06/atmos",
    "var_TMP": "on",
    "lev_2_m_above_ground": "on",
    # Recorte para o Brasil:
    "subregion": "",
    "toplat":   "5",
    "leftlon":  "-75",
    "rightlon": "-34",
    "bottomlat": "-35",
}

# Baixa de 6 em 6h até 384h (16 dias)
horas = list(range(0, 121, 6)) + list(range(123, 385, 3))

for h in horas:
    params = params_base.copy()
    params["file"] = f"gfs.t06z.pgrb2.0p25.f{h:03d}"
    r = requests.get(base, params=params, timeout=120)
    with open(f"gfs_TMP2m_f{h:03d}.grib2", "wb") as f:
        f.write(r.content)
    print(f"f{h:03d} baixado ({len(r.content)/1024:.0f} KB)")
    