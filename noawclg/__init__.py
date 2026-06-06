from noawclg.catalog import (
    HOURS_10DAYS_3H as HOURS_10DAYS_3H,
    HOURS_16DAYS as HOURS_16DAYS,
    HOURS_16DAYS_3H as HOURS_16DAYS_3H,
    HOURS_5DAYS_1H as HOURS_5DAYS_1H,
    MULTILEVEL_VARS as MULTILEVEL_VARS,
    SURFACE_VARS as SURFACE_VARS,
    VARIABLES as VARIABLES,
)
from noawclg.coords import BoundingBox as BoundingBox, auto_date as auto_date
from noawclg.gfs_dataset import GFSDatasetManager as GFSDatasetManager
from noawclg.load import load as load
from noawclg.ocean import (
    GODAS_VARS as GODAS_VARS,
    NINO_BOXES as NINO_BOXES,
    classify_enso as classify_enso,
    enso_summary as enso_summary,
    get_currents as get_currents,
    get_godas as get_godas,
    get_nino_anomaly as get_nino_anomaly,
    get_ocean_temp as get_ocean_temp,
    get_oni as get_oni,
    get_salinity as get_salinity,
    get_ssh as get_ssh,
    get_sst_series as get_sst_series,
    get_thermocline_depth as get_thermocline_depth,
    get_warm_water_volume as get_warm_water_volume,
    open_ersst as open_ersst,
    open_godas as open_godas,
)
from noawclg.query import get_noaa_data as get_noaa_data

__all__ = [
    "VARIABLES",
    "SURFACE_VARS",
    "MULTILEVEL_VARS",
    "HOURS_16DAYS",
    "HOURS_5DAYS_1H",
    "HOURS_10DAYS_3H",
    "HOURS_16DAYS_3H",
    "GFSDatasetManager",
    "get_noaa_data",
    "load",
    "auto_date",
    "BoundingBox",
    # Ocean / ENSO
    "GODAS_VARS",
    "NINO_BOXES",
    "open_godas",
    "open_ersst",
    "get_godas",
    "get_ocean_temp",
    "get_salinity",
    "get_currents",
    "get_ssh",
    "get_sst_series",
    "get_nino_anomaly",
    "get_oni",
    "classify_enso",
    "get_thermocline_depth",
    "get_warm_water_volume",
    "enso_summary",
]

__version__ = "2.3.0"
__author__ = "Reinan Br"
