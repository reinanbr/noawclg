"""Save and load GFS datasets to/from disk (NetCDF4 and Zarr)."""

from __future__ import annotations

import logging
from pathlib import Path

import xarray as xr

LOG = logging.getLogger(__name__)


def save_netcdf(
    ds: xr.Dataset,
    filename: str | Path,
    output_dir: Path,
    complevel: int = 4,
) -> Path:
    """Save *ds* to a zlib-compressed NetCDF4 file.

    Relative *filename* is resolved against *output_dir*;
    absolute paths are used as-is.
    """
    path = Path(filename)
    if not path.is_absolute():
        path = output_dir / path
    encoding = {v: {"zlib": True, "complevel": complevel} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=encoding)
    mb = path.stat().st_size / 1024**2
    LOG.info("Saved NetCDF: %s  (%.1f MB)", path, mb)
    print(f"[save] NetCDF → {path}  ({mb:.1f} MB)")
    return path


def save_zarr(
    ds: xr.Dataset,
    store: str | Path,
    output_dir: Path,
) -> Path:
    """Save *ds* as a chunked Zarr store.

    Relative *store* is resolved against *output_dir*;
    absolute paths are used as-is.
    """
    path = Path(store)
    if not path.is_absolute():
        path = output_dir / path
    ds.chunk({"time": 1}).to_zarr(path, mode="w")
    LOG.info("Saved Zarr: %s", path)
    print(f"[save] Zarr  → {path}")
    return path


def load_netcdf(path: str | Path) -> xr.Dataset:
    """Lazily open a previously saved NetCDF file (Dask-backed)."""
    ds = xr.open_dataset(path, chunks="auto")
    print(f"[load] {path}  →  {dict(ds.sizes)}")
    return ds


def load_zarr(store: str | Path) -> xr.Dataset:
    """Lazily open a previously saved Zarr store."""
    ds = xr.open_zarr(store)
    print(f"[load] {store}  →  {dict(ds.sizes)}")
    return ds
