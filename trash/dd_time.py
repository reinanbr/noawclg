import xarray as xr
import warnings
import logging

logging.getLogger("cfgrib").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

FILE = "gfs.t00z.pgrb2.0p25.anl"

level_types = [
    'meanSea', 'hybrid', 'isobaricInPa', 'isobaricInhPa', 'surface',
    'atmosphereSingleLayer', 'tropopause', 'maxWind', 'heightAboveGround',
    'heightAboveSea', 'isothermZero', 'highestTroposphericFreezing',
    'pressureFromGroundLayer', 'sigmaLayer', 'sigma', 'potentialVorticity'
]

for level in level_types:
    try:
        ds = xr.open_dataset(FILE, engine="cfgrib",
                             backend_kwargs={"filter_by_keys": {"typeOfLevel": level}})
        
        print(f"\n=== {level} ===")
        
        # Time coordinates
        for coord in ['time', 'step', 'valid_time']:
            if coord in ds.coords:
                print(f"  {coord}: {ds.coords[coord].values}")
        
        break  # just check first one, they should all be the same
    except Exception:
        pass