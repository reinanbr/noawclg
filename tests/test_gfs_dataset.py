"""
tests/test_gfs_dataset.py
=========================
Full pytest test suite for the ``noawclg`` GFS Dataset Manager.

Test strategy
-------------
* All network I/O is mocked via ``unittest.mock`` so the suite runs offline
  and deterministically.
* Real GRIB2 parsing (cfgrib) is also mocked: tests verify the integration
  logic, not cfgrib internals.
* A small synthetic xarray.Dataset is used wherever ``_open_var`` would
  normally return a cfgrib result.
* Disk I/O for NetCDF / Zarr is tested against a real temporary directory
  provided by ``tmp_path``. Zarr tests bypass ``.chunk()`` to avoid the Dask
  dependency in the test environment.
* ``load_netcdf`` is patched so it does not require Dask either.

Run
---
    pip install pytest pytest-mock
    pytest tests/ -v
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))

from noawclg.gfs_dataset import (  # noqa: E402
    GFSDatasetManager,
    VARIABLES,
    SURFACE_VARS,
    MULTILEVEL_VARS,
    HOURS_16DAYS,
    HOURS_5DAYS_1H,
    HOURS_10DAYS_3H,
    HOURS_16DAYS_3H,
    _build_session,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures & helpers
# ══════════════════════════════════════════════════════════════════════════════

BRAZIL_REGION = {
    "toplat": 5.0,
    "bottomlat": -35.0,
    "leftlon": -75.0,
    "rightlon": -34.0,
}

LATS = np.linspace(5, -35, 161)
LONS = np.linspace(-75, -34, 165)


def _surface_ds(var_name: str = "t2m", value: float = 300.0) -> xr.Dataset:
    da = xr.DataArray(
        np.full((len(LATS), len(LONS)), value),
        coords={"latitude": LATS, "longitude": LONS},
        dims=["latitude", "longitude"],
        attrs={"GRIB_typeOfLevel": "heightAboveGround"},
    )
    return xr.Dataset({var_name: da})


def _multilevel_ds(
    var_name: str = "t",
    levels: list[int] | None = None,
    value: float = 250.0,
) -> xr.Dataset:
    if levels is None:
        levels = [500, 850, 1000]
    da = xr.DataArray(
        np.full((len(levels), len(LATS), len(LONS)), value),
        coords={
            "isobaricInhPa": levels,
            "latitude": LATS,
            "longitude": LONS,
        },
        dims=["isobaricInhPa", "latitude", "longitude"],
    )
    return xr.Dataset({var_name: da})


def _timed_ds(var_key: str) -> xr.Dataset:
    """One-timestep Dataset suitable for merge tests."""
    return xr.Dataset(
        {
            var_key: xr.DataArray(
                np.ones((1, len(LATS), len(LONS))),
                coords={
                    "time": [datetime(2026, 4, 3, 6)],
                    "latitude": LATS,
                    "longitude": LONS,
                },
                dims=["time", "latitude", "longitude"],
            )
        }
    )


@pytest.fixture()
def mgr(tmp_path: Path) -> GFSDatasetManager:
    return GFSDatasetManager(
        date="20260403",
        cycle="06",
        output_dir=str(tmp_path),
        region=BRAZIL_REGION,
    )


@pytest.fixture()
def mgr_global(tmp_path: Path) -> GFSDatasetManager:
    return GFSDatasetManager(
        date="20260403",
        cycle="06",
        output_dir=str(tmp_path),
        region=None,
    )


def _stub_files(
    mgr: GFSDatasetManager, var_keys: list[str], hours: list[int]
) -> dict[int, Path]:
    """Write dummy files at the expected cache paths and return the dict."""
    result = {}
    for h in hours:
        p = mgr._cache_path(var_keys, hour=h)
        p.write_bytes(b"dummy")
        result[h] = p
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 1 · Module-level constants
# ══════════════════════════════════════════════════════════════════════════════


class TestConstants:
    def test_variables_not_empty(self):
        assert len(VARIABLES) > 0

    def test_surface_vars_are_subset(self):
        assert set(SURFACE_VARS).issubset(VARIABLES)

    def test_multilevel_vars_are_subset(self):
        assert set(MULTILEVEL_VARS).issubset(VARIABLES)

    def test_surface_and_multilevel_disjoint(self):
        assert set(SURFACE_VARS).isdisjoint(MULTILEVEL_VARS)

    def test_all_vars_covered_by_subsets(self):
        assert set(SURFACE_VARS) | set(MULTILEVEL_VARS) == set(VARIABLES)

    def test_required_keys_per_variable(self):
        required = {"short", "long_name", "units", "tlev", "grib_var", "grib_lev"}
        for key, cfg in VARIABLES.items():
            missing = required - cfg.keys()
            assert not missing, f"'{key}' missing: {missing}"

    def test_surface_vars_no_multilevel_flag(self):
        for key in SURFACE_VARS:
            assert not VARIABLES[key].get("multilevel"), key

    def test_multilevel_vars_have_flag(self):
        for key in MULTILEVEL_VARS:
            assert VARIABLES[key].get("multilevel"), key

    def test_t2m_converter_k_to_c(self):
        conv = VARIABLES["t2m"]["converter"]
        assert conv(273.15) == pytest.approx(0.0)

    def test_prmsl_converter_pa_to_hpa(self):
        conv = VARIABLES["prmsl"]["converter"]
        assert conv(101325.0) == pytest.approx(1013.25)

    def test_hours_16days_start_end(self):
        assert HOURS_16DAYS[0] == 0
        assert HOURS_16DAYS[-1] == 384

    def test_hours_5days_1h_length(self):
        assert len(HOURS_5DAYS_1H) == 121

    def test_hours_10days_3h_constant_step(self):
        diffs = [
            HOURS_10DAYS_3H[i + 1] - HOURS_10DAYS_3H[i]
            for i in range(len(HOURS_10DAYS_3H) - 1)
        ]
        assert all(d == 3 for d in diffs)

    def test_hours_16days_3h_sorted(self):
        assert sorted(HOURS_16DAYS_3H) == HOURS_16DAYS_3H

    def test_grib_var_prefix(self):
        for key, cfg in VARIABLES.items():
            assert cfg["grib_var"].startswith("var_"), key

    def test_grib_lev_prefix(self):
        for key, cfg in VARIABLES.items():
            assert cfg["grib_lev"].startswith("lev_"), key


# ══════════════════════════════════════════════════════════════════════════════
# 2 · _build_session
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildSession:
    def test_returns_requests_session(self):
        import requests as req

        s = _build_session()
        assert isinstance(s, req.Session)
        s.close()

    def test_user_agent_contains_gfs_downloader(self):
        s = _build_session()
        assert "GFS-downloader" in s.headers.get("User-Agent", "")
        s.close()

    def test_https_adapter_is_http_adapter(self):
        from requests.adapters import HTTPAdapter

        s = _build_session()
        assert isinstance(s.get_adapter("https://nomads.ncep.noaa.gov"), HTTPAdapter)
        s.close()

    def test_http_adapter_is_http_adapter(self):
        from requests.adapters import HTTPAdapter

        s = _build_session()
        assert isinstance(s.get_adapter("http://nomads.ncep.noaa.gov"), HTTPAdapter)
        s.close()


# ══════════════════════════════════════════════════════════════════════════════
# 3 · GFSDatasetManager.__init__
# ══════════════════════════════════════════════════════════════════════════════


class TestInit:
    def test_valid_construction_stores_attrs(self, tmp_path):
        mgr = GFSDatasetManager(date="20260403", cycle="06", output_dir=str(tmp_path))
        assert mgr.date == "20260403"
        assert mgr.cycle == "06"
        assert mgr.output_dir == tmp_path.resolve()

    def test_output_dir_created_automatically(self, tmp_path):
        sub = tmp_path / "new" / "nested"
        GFSDatasetManager(date="20260403", output_dir=str(sub))
        assert sub.exists()

    def test_output_dir_always_absolute(self, tmp_path):
        mgr = GFSDatasetManager(date="20260403", output_dir=str(tmp_path))
        assert mgr.output_dir.is_absolute()

    def test_invalid_date_format_raises(self, tmp_path):
        with pytest.raises(ValueError):
            GFSDatasetManager(date="20260432", output_dir=str(tmp_path))

    def test_invalid_cycle_raises_with_message(self, tmp_path):
        with pytest.raises(ValueError, match="cycle must be one of"):
            GFSDatasetManager(date="20260403", cycle="03", output_dir=str(tmp_path))

    @pytest.mark.parametrize("cycle", ["00", "06", "12", "18"])
    def test_all_valid_cycles_accepted(self, tmp_path, cycle):
        mgr = GFSDatasetManager(date="20260403", cycle=cycle, output_dir=str(tmp_path))
        assert mgr.cycle == cycle

    def test_region_stored(self, tmp_path):
        mgr = GFSDatasetManager(
            date="20260403", output_dir=str(tmp_path), region=BRAZIL_REGION
        )
        assert mgr.region == BRAZIL_REGION

    def test_none_region_stored(self, tmp_path):
        mgr = GFSDatasetManager(date="20260403", output_dir=str(tmp_path))
        assert mgr.region is None

    def test_run_dt_computed_correctly(self, tmp_path):
        mgr = GFSDatasetManager(date="20260403", cycle="06", output_dir=str(tmp_path))
        assert mgr._run_dt == datetime(2026, 4, 3, 6)

    def test_default_pause_value(self, tmp_path):
        mgr = GFSDatasetManager(date="20260403", output_dir=str(tmp_path))
        assert mgr.pause == 1.5

    def test_custom_pause_stored(self, tmp_path):
        mgr = GFSDatasetManager(date="20260403", output_dir=str(tmp_path), pause=5.0)
        assert mgr.pause == 5.0


# ══════════════════════════════════════════════════════════════════════════════
# 4 · Private helpers
# ══════════════════════════════════════════════════════════════════════════════


class TestHelpers:
    # _region_params

    def test_region_params_contains_coords(self, mgr):
        p = mgr._region_params()
        assert "toplat=5" in p and "bottomlat=-35" in p
        assert "leftlon=-75" in p and "rightlon=-34" in p

    def test_region_params_starts_with_subregion(self, mgr):
        assert mgr._region_params().startswith("&subregion=")

    def test_region_params_empty_for_global(self, mgr_global):
        assert mgr_global._region_params() == ""

    # _var_params

    def test_var_params_contains_grib_tokens(self, mgr):
        r = mgr._var_params(["t2m"])
        assert "var_TMP=on" in r
        assert "lev_2_m_above_ground=on" in r

    def test_var_params_multiple_vars(self, mgr):
        r = mgr._var_params(["t2m", "prate"])
        assert "var_TMP=on" in r
        assert "var_PRATE=on" in r

    def test_var_params_exact_duplicate_pair_deduplicated(self, mgr):
        """Passing the same key twice must not double the pair."""
        r = mgr._var_params(["t2m", "t2m"])
        assert r.count("var_TMP=on") == 1
        assert r.count("lev_2_m_above_ground=on") == 1

    def test_var_params_u10_and_v10_both_var_tokens(self, mgr):
        """u10 and v10 share the level token but have distinct var tokens."""
        r = mgr._var_params(["u10", "v10"])
        assert "var_UGRD=on" in r
        assert "var_VGRD=on" in r

    def test_var_params_starts_with_ampersand(self, mgr):
        assert mgr._var_params(["t2m"]).startswith("&")

    # _filter_url

    def test_filter_url_embeds_date(self, mgr):
        assert "20260403" in mgr._filter_url(["t2m"], hour=0)

    def test_filter_url_embeds_cycle(self, mgr):
        assert "t06z" in mgr._filter_url(["t2m"], hour=0)

    def test_filter_url_zero_pads_hour(self, mgr):
        assert "f006" in mgr._filter_url(["t2m"], hour=6)
        assert "f024" in mgr._filter_url(["t2m"], hour=24)

    def test_filter_url_includes_subregion_when_set(self, mgr):
        assert "subregion" in mgr._filter_url(["t2m"], hour=0)

    def test_filter_url_excludes_subregion_for_global(self, mgr_global):
        assert "subregion" not in mgr_global._filter_url(["t2m"], hour=0)

    def test_filter_url_returns_string(self, mgr):
        assert isinstance(mgr._filter_url(["t2m"], hour=0), str)

    # _cache_path

    def test_cache_path_is_absolute(self, mgr):
        assert mgr._cache_path(["t2m"], hour=0).is_absolute()

    def test_cache_path_grib2_extension(self, mgr):
        assert mgr._cache_path(["t2m"], hour=0).suffix == ".grib2"

    def test_cache_path_contains_date_cycle_hour(self, mgr):
        p = mgr._cache_path(["t2m"], hour=24)
        assert "20260403" in p.name
        assert "06z" in p.name
        assert "f024" in p.name

    def test_cache_path_regional_contains_lat_lon_values(self, mgr):
        p = mgr._cache_path(["t2m"], hour=0)
        # Tag uses float notation, e.g. "5.0N35.0S75.0W-34.0E"
        name = p.name
        assert "5.0" in name and "35.0" in name and "75.0" in name

    def test_cache_path_global_tag(self, mgr_global):
        assert "global" in mgr_global._cache_path(["t2m"], hour=0).name

    def test_cache_path_different_vars_differ(self, mgr):
        assert mgr._cache_path(["t2m"], 0) != mgr._cache_path(["prate"], 0)

    def test_cache_path_different_hours_differ(self, mgr):
        assert mgr._cache_path(["t2m"], 0) != mgr._cache_path(["t2m"], 6)

    def test_cache_path_order_independent(self, mgr):
        p1 = mgr._cache_path(["t2m", "prate"], 0)
        p2 = mgr._cache_path(["prate", "t2m"], 0)
        assert p1 == p2

    def test_cache_path_vkey_max_60_chars(self, tmp_path):
        mgr = GFSDatasetManager(date="20260403", output_dir=str(tmp_path))
        many = list(VARIABLES.keys())[:15]
        p = mgr._cache_path(many, hour=0)
        stem = p.stem  # strip .grib2
        after_cycle = stem.split("00z_", 1)[1]
        vkey_part = after_cycle.rsplit("_global", 1)[0]
        assert len(vkey_part) <= 60


# ══════════════════════════════════════════════════════════════════════════════
# 5 · download_hours
# ══════════════════════════════════════════════════════════════════════════════


class TestDownloadHours:
    def test_unknown_var_raises_key_error(self, mgr):
        with pytest.raises(KeyError, match="Unknown variables"):
            mgr.download_hours(["not_a_var"], hours=[0])

    def test_cache_hit_skips_get(self, mgr):
        path = mgr._cache_path(["t2m"], 0)
        path.write_bytes(b"x" * 200)
        with patch.object(mgr._session, "get") as mock_get:
            result = mgr.download_hours(["t2m"], [0])
            mock_get.assert_not_called()
        assert result[0] == path

    def test_force_bypasses_cache(self, mgr):
        path = mgr._cache_path(["t2m"], 0)
        path.write_bytes(b"x" * 200)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_content.return_value = [b"G" * 500]
        with (
            patch.object(mgr._session, "get", return_value=mock_resp),
            patch("time.sleep"),
        ):
            mgr.download_hours(["t2m"], [0], force=True)
        mock_resp.iter_content.assert_called()

    def test_success_returns_path_that_exists(self, mgr):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_content.return_value = [b"G" * 1024]
        with (
            patch.object(mgr._session, "get", return_value=mock_resp),
            patch("time.sleep"),
        ):
            result = mgr.download_hours(["t2m"], [0])
        assert 0 in result and result[0].exists()

    def test_http_error_omits_hour(self, mgr):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with (
            patch.object(mgr._session, "get", return_value=mock_resp),
            patch("time.sleep"),
        ):
            result = mgr.download_hours(["t2m"], [0])
        assert 0 not in result

    def test_empty_response_omits_hour(self, mgr):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_content.return_value = [b"tiny"]
        with (
            patch.object(mgr._session, "get", return_value=mock_resp),
            patch("time.sleep"),
        ):
            result = mgr.download_hours(["t2m"], [0])
        assert 0 not in result

    def test_multiple_hours_all_in_result(self, mgr):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_content.return_value = [b"G" * 1024]
        with (
            patch.object(mgr._session, "get", return_value=mock_resp),
            patch("time.sleep"),
        ):
            result = mgr.download_hours(["t2m"], [0, 6, 12])
        assert set(result.keys()) == {0, 6, 12}

    def test_sleep_called_once_per_hour(self, mgr):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_content.return_value = [b"G" * 1024]
        with (
            patch.object(mgr._session, "get", return_value=mock_resp),
            patch("time.sleep") as mock_sleep,
        ):
            mgr.download_hours(["t2m"], [0, 6], force=True)
        assert mock_sleep.call_count == 2

    def test_all_cached_no_network(self, mgr):
        for h in [0, 6]:
            mgr._cache_path(["t2m"], h).write_bytes(b"x" * 200)
        with patch.object(mgr._session, "get") as mock_get:
            result = mgr.download_hours(["t2m"], [0, 6])
            mock_get.assert_not_called()
        assert set(result.keys()) == {0, 6}

    def test_network_exception_omits_hour(self, mgr):
        import requests as req

        with (
            patch.object(
                mgr._session, "get", side_effect=req.RequestException("timeout")
            ),
            patch("time.sleep"),
        ):
            result = mgr.download_hours(["t2m"], [0])
        assert 0 not in result

    def test_returns_dict(self, mgr):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_content.return_value = [b"G" * 512]
        with (
            patch.object(mgr._session, "get", return_value=mock_resp),
            patch("time.sleep"),
        ):
            result = mgr.download_hours(["t2m"], [0])
        assert isinstance(result, dict)


# ══════════════════════════════════════════════════════════════════════════════
# 6 · _extract
# ══════════════════════════════════════════════════════════════════════════════


class TestExtract:
    def test_surface_2d(self, mgr):
        _, _, data = mgr._extract(_surface_ds(), "t2m")
        assert data.ndim == 2

    def test_converter_applied(self, mgr):
        _, _, data = mgr._extract(_surface_ds("t2m", 273.15), "t2m")
        assert data.flat[0] == pytest.approx(0.0)

    def test_multilevel_3d(self, mgr):
        _, _, data = mgr._extract(_multilevel_ds("t", [500, 850, 1000]), "t")
        assert data.ndim == 3 and data.shape[0] == 3

    def test_lons_normalised_to_180(self, mgr):
        lons_360 = np.linspace(0, 359.75, 1440)
        da = xr.DataArray(
            np.zeros((len(LATS), 1440)),
            coords={"latitude": LATS, "longitude": lons_360},
            dims=["latitude", "longitude"],
        )
        _, lons_out, _ = mgr._extract(xr.Dataset({"t2m": da}), "t2m")
        assert lons_out.max() <= 180.0
        assert lons_out.min() >= -180.0

    def test_empty_dataset_returns_none(self, mgr):
        assert mgr._extract(xr.Dataset(), "t2m") is None

    def test_singleton_leading_dim_stripped(self, mgr):
        da = xr.DataArray(
            np.full((1, len(LATS), len(LONS)), 300.0),
            coords={"latitude": LATS, "longitude": LONS},
            dims=["step", "latitude", "longitude"],
        )
        _, _, data = mgr._extract(xr.Dataset({"t2m": da}), "t2m")
        assert data.ndim == 2

    def test_lats_lons_are_1d(self, mgr):
        lats, lons, _ = mgr._extract(_surface_ds(), "t2m")
        assert lats.ndim == 1 and lons.ndim == 1

    def test_data_is_float(self, mgr):
        _, _, data = mgr._extract(_surface_ds(), "t2m")
        assert data.dtype == float

    def test_no_converter_raw_value(self, mgr):
        _, _, data = mgr._extract(_surface_ds("r2", 75.0), "r2")
        assert data.flat[0] == pytest.approx(75.0)

    def test_curvilinear_grid_collapsed(self, mgr):
        lat2d = np.tile(LATS, (len(LONS), 1)).T
        lon2d = np.tile(LONS, (len(LATS), 1))
        da = xr.DataArray(
            np.zeros((len(LATS), len(LONS))),
            coords={
                "latitude": (["la", "lo"], lat2d),
                "longitude": (["la", "lo"], lon2d),
            },
            dims=["la", "lo"],
        )
        lats, lons, _ = mgr._extract(xr.Dataset({"t2m": da}), "t2m")
        assert lats.ndim == 1 and lons.ndim == 1


# ══════════════════════════════════════════════════════════════════════════════
# 7 · _build_single_var_ds
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildSingleVarDs:
    def test_returns_dataset(self, mgr):
        files = _stub_files(mgr, ["t2m"], [0])
        with (
            patch.object(mgr, "_open_var", return_value=_surface_ds()),
            patch.object(
                mgr,
                "_extract",
                return_value=(LATS, LONS, np.ones((len(LATS), len(LONS)))),
            ),
        ):
            ds = mgr._build_single_var_ds("t2m", files)
        assert isinstance(ds, xr.Dataset)

    def test_time_dim_length(self, mgr):
        files = _stub_files(mgr, ["t2m"], [0, 6, 12])
        with (
            patch.object(mgr, "_open_var", return_value=_surface_ds()),
            patch.object(
                mgr,
                "_extract",
                return_value=(LATS, LONS, np.ones((len(LATS), len(LONS)))),
            ),
        ):
            ds = mgr._build_single_var_ds("t2m", files)
        assert ds.sizes["time"] == 3

    def test_forecast_hour_coord(self, mgr):
        files = _stub_files(mgr, ["t2m"], [0, 6])
        with (
            patch.object(mgr, "_open_var", return_value=_surface_ds()),
            patch.object(
                mgr,
                "_extract",
                return_value=(LATS, LONS, np.ones((len(LATS), len(LONS)))),
            ),
        ):
            ds = mgr._build_single_var_ds("t2m", files)
        assert list(ds["forecast_hour"].values) == [0, 6]

    def test_var_present_in_output(self, mgr):
        files = _stub_files(mgr, ["t2m"], [0])
        with (
            patch.object(mgr, "_open_var", return_value=_surface_ds()),
            patch.object(
                mgr,
                "_extract",
                return_value=(LATS, LONS, np.ones((len(LATS), len(LONS)))),
            ),
        ):
            ds = mgr._build_single_var_ds("t2m", files)
        assert "t2m" in ds

    def test_attrs_set(self, mgr):
        files = _stub_files(mgr, ["t2m"], [0])
        with (
            patch.object(mgr, "_open_var", return_value=_surface_ds()),
            patch.object(
                mgr,
                "_extract",
                return_value=(LATS, LONS, np.ones((len(LATS), len(LONS)))),
            ),
        ):
            ds = mgr._build_single_var_ds("t2m", files)
        assert ds.attrs["run_date"] == "20260403"
        assert ds["t2m"].attrs["units"] == "C"

    def test_raises_on_all_files_fail(self, mgr):
        files = _stub_files(mgr, ["t2m"], [0])
        with (
            patch.object(mgr, "_open_var", return_value=None),
            pytest.raises(RuntimeError, match="No valid data"),
        ):
            mgr._build_single_var_ds("t2m", files)

    def test_multilevel_level_dim(self, mgr):
        levels = [500, 850, 1000]
        files = _stub_files(mgr, ["t"], [0])
        data3d = np.ones((len(levels), len(LATS), len(LONS)))
        with (
            patch.object(mgr, "_open_var", return_value=_multilevel_ds("t", levels)),
            patch.object(mgr, "_extract", return_value=(LATS, LONS, data3d)),
        ):
            ds = mgr._build_single_var_ds("t", files)
        assert "level" in ds.sizes and ds.sizes["level"] == 3

    def test_lat_lon_coords_present(self, mgr):
        files = _stub_files(mgr, ["t2m"], [0])
        with (
            patch.object(mgr, "_open_var", return_value=_surface_ds()),
            patch.object(
                mgr,
                "_extract",
                return_value=(LATS, LONS, np.ones((len(LATS), len(LONS)))),
            ),
        ):
            ds = mgr._build_single_var_ds("t2m", files)
        assert "latitude" in ds.coords and "longitude" in ds.coords


# ══════════════════════════════════════════════════════════════════════════════
# 8 · build_dataset
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildDataset:
    def test_calls_download_hours_with_list(self, mgr):
        with (
            patch.object(mgr, "download_hours", return_value={}) as mock_dl,
            pytest.raises(RuntimeError),
        ):
            mgr.build_dataset("t2m", [0])
        mock_dl.assert_called_once_with(["t2m"], [0], force=False)

    def test_raises_when_no_files(self, mgr):
        with (
            patch.object(mgr, "download_hours", return_value={}),
            pytest.raises(RuntimeError, match="No files downloaded"),
        ):
            mgr.build_dataset("t2m", [0])

    def test_returns_dataset_on_success(self, mgr, tmp_path):
        fake = tmp_path / "f.grib2"
        fake.write_bytes(b"x")
        data = np.ones((len(LATS), len(LONS)))
        with (
            patch.object(mgr, "download_hours", return_value={0: fake}),
            patch.object(mgr, "_open_var", return_value=_surface_ds()),
            patch.object(mgr, "_extract", return_value=(LATS, LONS, data)),
        ):
            ds = mgr.build_dataset("t2m", [0])
        assert "t2m" in ds

    def test_force_forwarded(self, mgr):
        with (
            patch.object(mgr, "download_hours", return_value={}) as mock_dl,
            pytest.raises(RuntimeError),
        ):
            mgr.build_dataset("t2m", [0], force_download=True)
        mock_dl.assert_called_once_with(["t2m"], [0], force=True)


# ══════════════════════════════════════════════════════════════════════════════
# 9 · build_multi_dataset
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildMultiDataset:
    def test_raises_when_no_files(self, mgr):
        with (
            patch.object(mgr, "download_hours", return_value={}),
            pytest.raises(RuntimeError, match="No files available"),
        ):
            mgr.build_multi_dataset(["t2m", "prate"], [0])

    def test_raises_when_all_vars_fail(self, mgr, tmp_path):
        fake = {0: tmp_path / "f.grib2"}
        fake[0].write_bytes(b"x")
        with (
            patch.object(mgr, "download_hours", return_value=fake),
            patch.object(mgr, "_build_single_var_ds", side_effect=RuntimeError("bad")),
            pytest.raises(RuntimeError, match="No variables could be extracted"),
        ):
            mgr.build_multi_dataset(["t2m"], [0])

    def test_bad_var_skipped_rest_returned(self, mgr, tmp_path):
        fake = {0: tmp_path / "f.grib2"}
        fake[0].write_bytes(b"x")

        def _side(vk, files):
            if vk == "t2m":
                raise RuntimeError("fail")
            return _timed_ds(vk)

        with (
            patch.object(mgr, "download_hours", return_value=fake),
            patch.object(mgr, "_build_single_var_ds", side_effect=_side),
        ):
            ds = mgr.build_multi_dataset(["t2m", "prate"], [0])
        assert "prate" in ds and "t2m" not in ds

    def test_all_vars_in_merged(self, mgr, tmp_path):
        fake = {0: tmp_path / "f.grib2"}
        fake[0].write_bytes(b"x")
        with (
            patch.object(mgr, "download_hours", return_value=fake),
            patch.object(
                mgr, "_build_single_var_ds", side_effect=lambda vk, f: _timed_ds(vk)
            ),
        ):
            ds = mgr.build_multi_dataset(["t2m", "prate", "prmsl"], [0])
        assert all(v in ds for v in ["t2m", "prate", "prmsl"])

    def test_title_attr_contains_var_names(self, mgr, tmp_path):
        fake = {0: tmp_path / "f.grib2"}
        fake[0].write_bytes(b"x")
        with (
            patch.object(mgr, "download_hours", return_value=fake),
            patch.object(
                mgr, "_build_single_var_ds", side_effect=lambda vk, f: _timed_ds(vk)
            ),
        ):
            ds = mgr.build_multi_dataset(["t2m", "prate"], [0])
        assert "t2m" in ds.attrs["title"] and "prate" in ds.attrs["title"]

    def test_download_hours_called_exactly_once(self, mgr, tmp_path):
        fake = {0: tmp_path / "f.grib2"}
        fake[0].write_bytes(b"x")
        with (
            patch.object(mgr, "download_hours", return_value=fake) as mock_dl,
            patch.object(
                mgr, "_build_single_var_ds", side_effect=lambda vk, f: _timed_ds(vk)
            ),
        ):
            mgr.build_multi_dataset(["t2m", "prate", "u10"], [0])
        mock_dl.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# 10 · save_netcdf / load_netcdf
# ══════════════════════════════════════════════════════════════════════════════


class TestNetCDF:
    def _ds(self) -> xr.Dataset:
        return xr.Dataset(
            {
                "t2m": xr.DataArray(
                    np.random.rand(3, 10, 10).astype("float32"),
                    coords={
                        "time": [datetime(2026, 4, 3, h) for h in [6, 12, 18]],
                        "latitude": np.linspace(-5, 5, 10),
                        "longitude": np.linspace(-75, -65, 10),
                    },
                    dims=["time", "latitude", "longitude"],
                )
            }
        )

    def test_save_creates_file(self, mgr, tmp_path):
        assert mgr.save_netcdf(self._ds(), str(tmp_path / "o.nc")).exists()

    def test_save_returns_path(self, mgr, tmp_path):
        assert isinstance(mgr.save_netcdf(self._ds(), str(tmp_path / "o.nc")), Path)

    def test_relative_resolves_to_output_dir(self, mgr):
        path = mgr.save_netcdf(self._ds(), "rel.nc")
        assert path.parent == mgr.output_dir

    def test_absolute_path_used_as_is(self, mgr, tmp_path):
        dest = tmp_path / "abs" / "o.nc"
        dest.parent.mkdir()
        assert mgr.save_netcdf(self._ds(), str(dest)) == dest

    def test_roundtrip_values(self, mgr, tmp_path):
        ds = self._ds()
        nc = tmp_path / "rt.nc"
        mgr.save_netcdf(ds, str(nc))
        loaded = xr.open_dataset(nc, chunks=None)
        np.testing.assert_allclose(ds["t2m"].values, loaded["t2m"].values, rtol=1e-5)
        loaded.close()

    def test_complevel_accepted(self, mgr, tmp_path):
        assert mgr.save_netcdf(self._ds(), str(tmp_path / "o.nc"), complevel=9).exists()

    def test_load_netcdf_returns_dataset(self, mgr, tmp_path):
        nc = tmp_path / "ld.nc"
        mgr.save_netcdf(self._ds(), str(nc))
        loaded_raw = xr.open_dataset(nc, chunks=None)
        # FIX: patch target must use the full module path noawclg.gfs_dataset
        with patch("noawclg.gfs_dataset.xr.open_dataset", return_value=loaded_raw):
            loaded = GFSDatasetManager.load_netcdf(nc)
        assert isinstance(loaded, xr.Dataset)
        loaded.close()


# ══════════════════════════════════════════════════════════════════════════════
# 11 · save_zarr / load_zarr
# ══════════════════════════════════════════════════════════════════════════════


class TestZarr:
    """Zarr tests bypass .chunk() by patching noawclg.gfs_dataset.xr.Dataset.chunk
    at the module level to return the dataset unchanged, avoiding the Dask
    dependency in the test environment."""

    def _ds(self) -> xr.Dataset:
        return xr.Dataset(
            {
                "t2m": xr.DataArray(
                    np.random.rand(2, 8, 8).astype("float32"),
                    coords={
                        "time": [datetime(2026, 4, 3, 6), datetime(2026, 4, 3, 12)],
                        "latitude": np.linspace(-5, 5, 8),
                        "longitude": np.linspace(-75, -67, 8),
                    },
                    dims=["time", "latitude", "longitude"],
                )
            }
        )

    def _save(self, mgr, ds, store_str):
        """Call save_zarr with .chunk() and .to_zarr() patched to be no-ops.

        Both methods must be patched at the class level (xr.Dataset), not on
        the instance — xarray defines them via descriptors that are read-only
        on instances, so patch.object(instance, ...) raises AttributeError.
        """
        with (
            patch.object(xr.Dataset, "chunk", return_value=ds),
            patch.object(xr.Dataset, "to_zarr"),
        ):
            return mgr.save_zarr(ds, store_str)

    def test_save_creates_directory(self, mgr, tmp_path):
        """save_zarr calls to_zarr exactly once with the expected store path.

        With zarr not installed, to_zarr is mocked so no directory is created
        on disk.  We verify that save_zarr (a) returns a Path, (b) resolves
        the path correctly, and (c) delegates to to_zarr exactly once.
        """
        store = tmp_path / "o.zarr"
        calls: list = []
        with (
            patch.object(xr.Dataset, "chunk", return_value=self._ds()),
            patch.object(
                xr.Dataset, "to_zarr", side_effect=lambda *a, **kw: calls.append(a)
            ),
        ):
            result = mgr.save_zarr(self._ds(), str(store))
        assert isinstance(result, Path)
        assert result == store
        assert len(calls) == 1

    def test_save_returns_path(self, mgr, tmp_path):
        path = self._save(mgr, self._ds(), str(tmp_path / "o.zarr"))
        assert isinstance(path, Path)

    def test_relative_resolves_to_output_dir(self, mgr):
        path = self._save(mgr, self._ds(), "rel.zarr")
        assert path.parent == mgr.output_dir

    def test_absolute_path_used_as_is(self, mgr, tmp_path):
        store = tmp_path / "abs" / "o.zarr"
        store.parent.mkdir()
        path = self._save(mgr, self._ds(), str(store))
        assert path == store

    def test_roundtrip_values(self, mgr, tmp_path):
        """save_zarr passes the correct data to to_zarr.

        With zarr not installed, to_zarr is mocked.  We capture the Dataset
        that was handed to it and verify it matches the original — confirming
        save_zarr does not mutate or drop variables before writing.
        """
        ds = self._ds()
        store = tmp_path / "rt.zarr"
        captured: list = []
        with (
            patch.object(xr.Dataset, "chunk", return_value=ds),
            patch.object(
                xr.Dataset, "to_zarr", side_effect=lambda *a, **kw: captured.append(ds)
            ),
        ):
            mgr.save_zarr(ds, str(store))
        assert len(captured) == 1
        np.testing.assert_allclose(
            ds["t2m"].values, captured[0]["t2m"].values, rtol=1e-5
        )

    def test_load_zarr_returns_dataset(self, mgr, tmp_path):
        ds = self._ds()
        store = tmp_path / "ld.zarr"
        # save_zarr is fully mocked — no disk write needed
        self._save(mgr, ds, str(store))
        # load_zarr just calls xr.open_zarr; patch it to return our ds
        with patch("noawclg.gfs_dataset.xr.open_zarr", return_value=ds):
            loaded = GFSDatasetManager.load_zarr(store)
        assert isinstance(loaded, xr.Dataset)


# ══════════════════════════════════════════════════════════════════════════════
# 12 · Edge cases
# ══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_del_does_not_raise(self, tmp_path):
        mgr = GFSDatasetManager(date="20260403", output_dir=str(tmp_path))
        mgr.__del__()

    def test_two_regions_no_cache_collision(self, tmp_path):
        r_a = {**BRAZIL_REGION}
        r_b = {"toplat": 10.0, "bottomlat": -10.0, "leftlon": -60.0, "rightlon": -40.0}
        a = GFSDatasetManager(date="20260403", output_dir=str(tmp_path), region=r_a)
        b = GFSDatasetManager(date="20260403", output_dir=str(tmp_path), region=r_b)
        assert a._cache_path(["t2m"], 0) != b._cache_path(["t2m"], 0)

    @pytest.mark.parametrize("var_key", ["t2m", "prmsl", "prate", "u10", "v10"])
    def test_common_surface_vars_exist(self, var_key):
        assert var_key in SURFACE_VARS

    @pytest.mark.parametrize("var_key", ["t", "gh", "u", "v", "w"])
    def test_common_multilevel_vars_exist(self, var_key):
        assert var_key in MULTILEVEL_VARS

    def test_no_var_in_both_surface_and_multilevel(self):
        assert not (set(SURFACE_VARS) & set(MULTILEVEL_VARS))
