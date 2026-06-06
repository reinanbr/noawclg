# Changelog

All notable changes to **noawclg** are documented here.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.3.0] — 2026-06-05

### Added

#### `noawclg/ocean.py` — ocean data module (GODAS + ERSST)
- `open_godas(year, variable, depth_m, region)` — lazy OPeNDAP access to all
  five GODAS variables: `pottmp` (temperature, K→°C), `salt` (salinity,
  kg/kg→PSU), `ucur`/`vcur` (ocean currents, m/s), `sshg` (SSH, m).
- `get_godas(year_start, year_end, variable, depth_m, region)` — multi-year
  concatenation for any GODAS variable.
- `get_ocean_temp` / `get_salinity` / `get_currents` / `get_ssh` — typed
  convenience wrappers with sensible defaults.
- `open_ersst(year_start, year_end, region)` — NOAA ERSST v5 via OPeNDAP,
  SST back to 1854; handles ERSST's decreasing-latitude grid automatically.
- `get_sst_series(box, source)` — monthly Niño-box SST from GODAS or ERSST.
- `get_nino_anomaly` / `get_oni` — SST anomalies and the Oceanic Niño Index.
- `classify_enso(oni)` — CPC ONI rule: El Niño / La Niña / Neutral.
- `get_thermocline_depth` — depth of the 20 °C isotherm (D20) from GODAS.
- `get_warm_water_volume` — equatorial Pacific WWV index (leading indicator).
- `enso_summary(year_start, year_end)` — DataFrame with SST, anomaly, ONI,
  and ENSO phase for any year range.
- `GODAS_VARS` catalogue dict and `NINO_BOXES` (Niño 1+2 / 3 / 3.4 / 4).
- All symbols exported from `noawclg.__init__`.

#### `plots.py` — six new ocean/ENSO plot functions
- `plot_enso_index` — ONI time series with El Niño / La Niña shading.
- `plot_ocean_temp_map` — global T map with optional Niño-box overlays.
- `plot_thermocline_section` — depth–longitude cross-section + 20 °C isotherm.
- `plot_ssh_map` — symmetric SSH anomaly map.
- `plot_ocean_currents` — speed-filled map with quiver arrows.
- `plot_globe` — any 2-D field on a cartopy Orthographic globe.

#### `enso_forecast.py` — real-data ENSO analysis script
- Downloads live GODAS (T200, SSH) + ERSST (ONI) via `noawclg.ocean`.
- Five-indicator probability model: ONI current, ONI trend, T200 anomaly,
  SSH eastern Pacific, historical analogs.
- Generates `gfs_plots/enso/enso_analysis.png` (7-panel figure) and
  `gfs_plots/enso/enso_indicators.png` (historical context).
- Full terminal summary with probability bar.

#### `docs/` — Sphinx/ReadTheDocs site
- `furo` theme, MyST-Parser (Markdown), `sphinx-autodoc-typehints`.
- Pages: Installation · Quick start · GFS examples · ENSO analysis ·
  Maps & globe · API reference (GFS, Ocean, Plots) · Gallery (20 plots).
- `.readthedocs.yaml` configuration for automatic RTD builds.
- `docs/_static/plots/` — all 20 gallery plots.

#### GitHub Actions
- New `docs` job: validates Sphinx build on every push; triggers ReadTheDocs
  API build on version tags when `RTD_TOKEN` secret is configured.

### Changed
- `setup.py`: added `extras_require` groups `[plots]`, `[docs]`, `[dev]`;
  updated classifiers, keywords, and `python_requires = ">=3.10"`.
- `README.md` condensed from 1 029 → 218 lines; links to ReadTheDocs docs.
- `requirements.txt`: removed `basemap==2.0.0`, `basemap_data==2.0.0`,
  `cffi==1.17.1` (incompatible with Python ≥ 3.14).

### Fixed
- `open_ersst`: region subsetting now correctly handles ERSST's
  **decreasing latitude** axis (88 → −88); `lat_min`/`lat_max` slice is
  automatically reversed when needed.
- `get_sst_series(source="ersst")`: uses 0–360 longitude convention
  (matching ERSST grid) instead of an incorrect sign-flip to −180/+180.
- `plots.py` / `make_readme_plots.py`: xarray ≥ 2025.x non-index coordinate
  selection — replaced `ds.sel(forecast_hour=h, method="nearest")` with
  `_hour_sel(ds, h)` using `isel` + `argmin`.
- `plots.py` `plot_precip_heatmap`: pandas CoW FutureWarning eliminated by
  replacing chained assignment with `.assign()`.

### Tests
- New `tests/test_ocean.py`: 44 offline tests covering GODAS catalogue,
  `open_godas`, `get_godas`, typed wrappers, `open_ersst`, ENSO indices,
  `classify_enso`, `get_thermocline_depth`, `enso_summary`, and
  `assess_probability` from `enso_forecast.py`.

---

## [2.2.7] - 2026-05-03
### Added
- ```noawclg.load``` function for get direct dataset noaa

## [2.2.6] — 2026-04-19

### Fixed
- CI release pipeline follow-up after `v2.2.5` to ensure publish runs from the corrected commit.

## [2.2.5] — 2026-04-19

### Changed
- Release metadata bumped to `2.2.5` to publish with a fresh tag.
- `tests/test_main.py` formatting aligned with Ruff.

## [2.2.4] — 2026-04-19

### Added
- README API reference now documents `load` with parameters, return type, and usage example.
- New tests for `load` in `tests/test_main.py`, including:
  - return value validation as `xr.Dataset`,
  - argument forwarding validation to `get_noaa_data`.

### Changed
- Version metadata updated to `2.2.4`.

## [2.2.3] — 2026-04-17

### Fixed
- Release tagging adjustment to trigger GitHub Actions publish flow to PyPI.
- Version metadata updated to `2.2.3`.

## [2.2] — 2026-04-17

### Added
- New test suite for `noawclg.main` in `tests/test_main.py`, covering:
  - helper functions (`_parse_date`, `_normalize_lon`, `_find_dim`),
  - `_DatasetView` and `BoundingBox`,
  - class `get_noaa_data` (single/multi variable init, point/place queries,
    time series and validation errors).
- Consolidated project report in `WORKLOG_2.2.md` with the previous and current
  delivery summaries.

### Changed
- README API docs now include complete reference for `get_noaa_data`.
- Local CI execution validated through Pipenv (`ruff`, `mypy`, offline tests,
  integration tests, build and `twine check`).

### Fixed
- Removed unused import in `noawclg/main.py` to satisfy lint checks.

## [2.1.13]
- Config PyPi Publish.

## [2.1.12]
- Added NumPy as required library.

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

[Unreleased]: https://github.com/reinanbr/noawclg/compare/v2.3.0...HEAD
[2.3.0]:      https://github.com/reinanbr/noawclg/compare/v2.2.7...v2.3.0
[2.2.7]:      https://github.com/reinanbr/noawclg/compare/v2.2.6...v2.2.7
[2.2.6]:      https://github.com/reinanbr/noawclg/compare/v2.2.5...v2.2.6
[2.2.5]:      https://github.com/reinanbr/noawclg/compare/v2.2.4...v2.2.5
[2.2.4]:      https://github.com/reinanbr/noawclg/compare/v2.2.3...v2.2.4
[2.2.3]:      https://github.com/reinanbr/noawclg/compare/v2.2...v2.2.3
[2.2]:        https://github.com/reinanbr/noawclg/compare/v2.1.13...v2.2
[2.1.0]:      https://github.com/reinanbr/noawclg/compare/v2.0.0...v2.1.0
[2.0.0]:      https://github.com/reinanbr/noawclg/compare/v1.1.0...v2.0.0
[1.1.0]:      https://github.com/reinanbr/noawclg/compare/v1.0.0...v1.1.0
[1.0.0]:      https://github.com/reinanbr/noawclg/releases/tag/v1.0.0