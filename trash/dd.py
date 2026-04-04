import xarray as xr

FILE = "gfs.t00z.pgrb2.0p25.anl"

level_types = [
    'meanSea', 'hybrid', 'isobaricInPa', 'isobaricInhPa', 'surface',
    'atmosphereSingleLayer', 'tropopause', 'maxWind', 'heightAboveGround',
    'heightAboveSea', 'isothermZero', 'highestTroposphericFreezing',
    'pressureFromGroundLayer', 'sigmaLayer', 'sigma', 'potentialVorticity'
]

all_vars = {}

for level in level_types:
    try:
        ds = xr.open_dataset(
            FILE,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"typeOfLevel": level}}
        )
        for var in ds.data_vars:
            info = {
                "typeOfLevel": level,
                "shape":       ds[var].shape,
                "dims":        list(ds[var].dims),
                "units":       ds[var].attrs.get("units", "?"),
                "long_name":   ds[var].attrs.get("long_name", "?"),
            }
            all_vars[f"{var}@{level}"] = info

    except Exception:
        pass  # already handled conflict errors before

# Print clean table
print(f"\n{'KEY':<40} {'DIMS':<45} {'UNITS':<15} {'LONG NAME'}")
print("-" * 140)
for key, info in all_vars.items():
    dims  = str(info["dims"])
    units = info["units"]
    lname = info["long_name"]
    print(f"{key:<40} {dims:<45} {units:<15} {lname}")