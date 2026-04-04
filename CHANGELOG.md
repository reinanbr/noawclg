# Changelog

All notable changes to **noawclg** are documented here.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.1.11]

### Planned
- Async download support via `asyncio` / `aiohttp`.
- `build_timeseries` helper to concatenate multi-run Datasets along a run dimension.
- Optional Dask-parallelised extraction across forecast hours.

---

## [2.1.0] — 2026-04-03

### Added
- **Google-style docstrings** on every public and private method, class, and
  module-level function — full coverage verified with `ast` introspection.
- `_non_empty` closure inside `_open_var` now has an explicit docstring.
- Module-level `Example:` block in the package docstring uses proper
  reStructuredText syntax compatible with Sphinx autodoc.

### Changed
- `_open_var` now logs the actual exception (`last_exc`) instead of a generic
  warning, making it easier to diagnose cfgrib filter failures.
- `__del__` now has a return type annotation (`-> None`).

---

## [2.0.0] — 2026-04-03

### Added
- **Single-download architecture** (`download_hours`): one HTTP request per
  forecast hour regardless of how many variables are requested.  
  Reduces requests from `N_vars × N_hours` to `N_hours`.
- Multi-variable grib-filter URL builder (`_var_params`): bundles all
  `&var_XXX=on&lev_XXX=on` pairs into a single URL.
- Disk cache with deterministic filenames encoding date, cycle, variable set,
  region tag, and forecast hour — global and regional downloads never collide.
- Retry strategy (`urllib3.Retry`): exponential back-off on HTTP 429/500/502/
  503/504 — up to 4 retries with delays of 2 s, 4 s, 8 s, 16 s.
- Browser-like `User-Agent` header (NOMADS blocks the default `python-requests`
  agent with HTTP 403).
- `pause` constructor parameter (default `1.5 s`) to throttle requests.
- Download progress logging: percentage, elapsed time, and estimated remaining
  time per file.
- `tqdm` progress bar when the package is installed (graceful degradation when
  not available).
- `_open_var` cascade filter strategy: tries up to 5 filter combinations
  (`shortName + typeOfLevel + level` → full `cfgrib.open_datasets` scan) to
  handle GRIB table inconsistencies across GFS versions.
- `_extract` robustness fixes:
  - Empty dataset guard before any array access.
  - Curvilinear 2-D lat/lon grids collapsed to 1-D.
  - Iterative leading-singleton-dimension removal (replaces blind `np.squeeze`
    that caused `IndexError` on certain GRIB messages).
  - Longitude normalisation 0–360 → −180–180 with column reordering.
- `output_dir` resolved to an absolute path in `__init__` — eliminates the
  path-duplication bug in `save_netcdf` when a relative filename was passed.
- `__del__` method closes the shared `requests.Session` on garbage collection.
- `HOURS_16DAYS_3H` pre-defined hour sequence.

### Changed
- `build_multi_dataset` now calls `download_hours` once (not once per
  variable), then iterates `_build_single_var_ds` over the cached files.
- `save_netcdf` and `save_zarr` resolve relative filenames against the
  (now-absolute) `output_dir` — no more duplicate path segments.
- All Portuguese-language comments, docstrings, log messages, and variable
  names translated to English.

### Removed
- Dependency on `get_all_data_16_days` from the external `get.py` module —
  the manager is now fully self-contained.
- `download_variable` (per-variable download method) — superseded by
  `download_hours`.

### Fixed
- `IndexError: list index out of range` in `_extract` caused by `np.squeeze`
  removing the level dimension on certain GRIB2 messages with shape
  `(1, 1, lat, lon)`.
- `PermissionError` in `save_netcdf` when a relative filename was passed to a
  manager whose `output_dir` was itself relative — the path was doubly
  prefixed.
- `KeyError` in `_open_var` when `cfg["levels"]` was `None` (e.g. `gust`,
  `prmsl`) — now uses `cfg.get("levels")`.

---

## [1.1.0] — 2026-03-15

### Added
- `save_zarr` / `load_zarr` persistence methods.
- `HOURS_5DAYS_1H` and `HOURS_10DAYS_3H` pre-defined hour sequences.
- `SURFACE_VARS` and `MULTILEVEL_VARS` module-level lists.

### Changed
- `build_multi_dataset` now uses `xr.merge(..., join="inner")` instead of
  `join="outer"` to avoid NaN-filled padding when grids differ slightly.

### Fixed
- cfgrib index files (`.idx`) were left behind in `output_dir`; now passed
  `indexpath=None` to suppress them.

---

## [1.0.0] — 2026-02-28

### Added
- Initial public release.
- `GFSDatasetManager` class with `build_dataset`, `build_multi_dataset`,
  `save_netcdf`, `load_netcdf`.
- Variable catalogue with 40+ GFS 0.25° pgrb2 variables.
- NOMADS grib-filter integration with optional spatial sub-region.
- `HOURS_16DAYS` pre-defined hour sequence.

---

[Unreleased]: https://github.com/reinanbr/noawclg/compare/v2.1.0...HEAD
[2.1.0]:      https://github.com/reinanbr/noawclg/compare/v2.0.0...v2.1.0
[2.0.0]:      https://github.com/reinanbr/noawclg/compare/v1.1.0...v2.0.0
[1.1.0]:      https://github.com/reinanbr/noawclg/compare/v1.0.0...v1.1.0
[1.0.0]:      https://github.com/reinanbr/noawclg/releases/tag/v1.0.0