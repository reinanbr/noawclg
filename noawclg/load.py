from typing import Optional
from noawclg.main import get_noaa_data


def load(
    date: Optional[str] = None,
    cycle: str = "00",
    keys: list[str] = ["t2m"],
    hours: Optional[list[int]] = None,
    *,
    lat_dim: Optional[str] = None,
    lon_dim: Optional[str] = None,
    time_dim: Optional[str] = None,
):
    """
    Load NOAA GFS data for a specific date and cycle.

    Parameters:
        date (str): Date in 'YYYYMMDD' format. If None, uses the latest available date.
        cycle (str): Model run cycle ("00", "06", "12", "18"). Default is "00".
        keys (list[str]): List of variable names to load. Default is ["t2m"].
        hours (list[int]): List of forecast hours to load. If None, loads all available hours.
        lat_dim (str): Name of the latitude dimension in the dataset. If None, uses the default name.
        lon_dim (str): Name of the longitude dimension in the dataset. If None, uses the default name.
        time_dim (str): Name of the time dimension in the dataset. If None, uses the default name.
    Returns:
        xarray.Dataset: The loaded dataset containing the requested variables and dimensions.
    """
    return get_noaa_data(
        date=date,
        cycle=cycle,
        keys=keys,
        hours=hours,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        time_dim=time_dim,
    )._ds
