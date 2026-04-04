import cfgrib
import pandas as pd
from datetime import datetime
# ─────────────────────────────────────────────
# INVENTÁRIO COMPLETO DE VARIÁVEIS
# ─────────────────────────────────────────────
DATA  = datetime(2026, 4, 3)
CICLO = "00"
F_HORA = 0
arquivo = f"gfs_global_{DATA.strftime('%Y%m%d')}_{CICLO}z_f{F_HORA:03d}.grib2"

print("Abrindo todos os datasets do GRIB2...\n")
datasets = cfgrib.open_datasets(arquivo)

registros = []

print(f'all keys: {datasets[0].variables.keys()}')
#print(f'all attrs: {datasets[0].variables["TMP"].attrs}')
for ds in datasets:
    print(f"Dataset: {ds}")
    print(f'all keys: {ds.variables.keys()}')
    print(f'all attrs: {ds.variables[list(ds.variables.keys())[0]].attrs}')

    type_of_level = ds.attrs.get("GRIB_typeOfLevel", "?")

    niveis = None
    for coord in ["isobaricInhPa", "heightAboveGround", "level"]:
        if coord in ds.coords:
            val = ds.coords[coord].values
            # scalar (float/0-d array) → envolve em lista
            if val.ndim == 0:
                niveis = [float(val)]
            else:
                niveis = sorted(val.tolist())
            break

    for nome, da in ds.data_vars.items():
        registros.append({
            "shortName":   nome,
            "long_name":   da.attrs.get("long_name", ""),
            "units":       da.attrs.get("units", ""),
            "typeOfLevel": type_of_level,
            "niveis":      str(niveis) if niveis else "-",
            "shape":       str(da.shape),
    })

df = pd.DataFrame(registros).drop_duplicates(subset=["shortName", "typeOfLevel"])
df = df.sort_values(["typeOfLevel", "shortName"]).reset_index(drop=True)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", 50)
pd.set_option("display.width", 120)
print(df.to_string(index=True))
print(f"\nTotal: {len(df)} variáveis únicas\n")

csv_out = "gfs_variaveis.csv"
df.to_csv(csv_out, index=False)
print(f"Inventário salvo em: {csv_out}")

print("\n── Contagem por typeOfLevel ──")
print(df.groupby("typeOfLevel")["shortName"].count().sort_values(ascending=False).to_string())