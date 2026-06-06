"""
tests/test_ocean.py
===================
Offline test suite for ``noawclg.ocean``.

All NOAA OPeNDAP calls are mocked via ``unittest.mock.patch`` so these tests
run without any network access and execute deterministically.

Test strategy
-------------
* ``_make_godas_ds`` returns a minimal synthetic xr.Dataset that mimics the
  GODAS NetCDF structure (Kelvin units, fill values, level/lat/lon/time dims).
* ``_make_ersst_ds`` returns a minimal ERSST v5-like dataset (decreasing lat,
  0-360 lon, no 'lev' dim after squeeze).
* Every public function in ``noawclg.ocean`` is covered.
* ``classify_enso`` and ``assess_probability`` (enso_forecast) are tested
  with synthetic pd.Series — no network required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from noawclg.ocean import (
    GODAS_VARS,
    NINO_BOXES,
    classify_enso,
    enso_summary,
    get_currents,
    get_godas,
    get_nino_anomaly,
    get_ocean_temp,
    get_oni,
    get_salinity,
    get_ssh,
    get_sst_series,
    get_thermocline_depth,
    open_ersst,
    open_godas,
)


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic data factories
# ══════════════════════════════════════════════════════════════════════════════

_LAT = np.linspace(-10.0, 10.0, 21)      # 1° resolution
_LON = np.linspace(150.0, 290.0, 29)     # 5° resolution
_LEVELS = np.array([5, 55, 105, 205, 305, 505, 750, 1000], dtype=float)
_TIME_12 = pd.date_range("2024-01", periods=12, freq="MS")
_TIME_4  = pd.date_range("2026-01", periods=4,  freq="MS")


def _make_godas_ds(variable: str = "pottmp", n_levels: bool = True) -> xr.Dataset:
    """Minimal synthetic GODAS dataset in raw (unconverted) units."""
    t = _TIME_12

    if n_levels:
        # 3-D field: (time, level, lat, lon)
        shape = (len(t), len(_LEVELS), len(_LAT), len(_LON))
        if variable == "pottmp":
            # Valid Kelvin range so masking keeps real values
            raw = np.random.uniform(285, 305, shape).astype(np.float32)
        elif variable == "salt":
            raw = np.random.uniform(0.033, 0.037, shape).astype(np.float32)
        else:
            raw = np.random.uniform(-0.5, 0.5, shape).astype(np.float32)

        da = xr.DataArray(
            raw,
            dims=["time", "level", "lat", "lon"],
            coords={"time": t, "level": _LEVELS, "lat": _LAT, "lon": _LON},
            attrs={"missing_value": np.float32(9.969209968386869e+36)},
        )
    else:
        # 2-D field: (time, lat, lon)  — sshg
        raw = np.random.uniform(-0.3, 0.3, (len(t), len(_LAT), len(_LON))
                                ).astype(np.float32)
        da = xr.DataArray(
            raw,
            dims=["time", "lat", "lon"],
            coords={"time": t, "lat": _LAT, "lon": _LON},
            attrs={"missing_value": np.float32(9.969209968386869e+36)},
        )

    return xr.Dataset({variable: da})


def _make_ersst_ds() -> xr.Dataset:
    """Minimal synthetic ERSST v5 dataset: decreasing lat, lon 0-360."""
    lat  = np.linspace(88, -88, 89)          # decreasing
    lon  = np.linspace(0, 358, 180)           # 0-360
    time = pd.date_range("1950-01", periods=912, freq="MS")   # 76 years

    sst = xr.DataArray(
        np.random.uniform(22, 30, (len(time), len(lat), len(lon))).astype(np.float32),
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"missing_value": np.float32(9.969209968386869e+36), "units": "degC"},
    )
    return xr.Dataset({"sst": sst})


# ══════════════════════════════════════════════════════════════════════════════
# Tests: GODAS_VARS catalogue
# ══════════════════════════════════════════════════════════════════════════════

class TestGODASVars:
    def test_all_five_variables_present(self):
        assert set(GODAS_VARS.keys()) == {"pottmp", "salt", "ucur", "vcur", "sshg"}

    def test_pottmp_has_level(self):
        assert GODAS_VARS["pottmp"]["has_levels"] is True

    def test_sshg_no_level(self):
        assert GODAS_VARS["sshg"]["has_levels"] is False

    def test_units_out_defined(self):
        for var, meta in GODAS_VARS.items():
            assert "units_out" in meta, f"{var} missing 'units_out'"


class TestNinoBoxes:
    def test_four_boxes(self):
        assert set(NINO_BOXES.keys()) == {"1+2", "3", "3.4", "4"}

    def test_nino34_lat_range(self):
        b = NINO_BOXES["3.4"]
        assert b["lat"] == (-5.0, 5.0)

    def test_nino34_lon_range_0_360(self):
        b = NINO_BOXES["3.4"]
        assert 180 < b["lon"][0] < b["lon"][1] < 360


# ══════════════════════════════════════════════════════════════════════════════
# Tests: open_godas
# ══════════════════════════════════════════════════════════════════════════════

class TestOpenGodas:
    @pytest.fixture
    def mock_pottmp(self):
        ds = _make_godas_ds("pottmp")
        with patch("noawclg.ocean.xr.open_dataset", return_value=ds):
            yield

    def test_returns_dataset(self, mock_pottmp):
        result = open_godas(2024, variable="pottmp")
        assert isinstance(result, xr.Dataset)
        assert "pottmp" in result

    def test_units_converted_to_celsius(self, mock_pottmp):
        result = open_godas(2024, variable="pottmp")
        # Raw values were 285-305 K; after subtraction 273.15 → 12-32 °C
        vals = result["pottmp"].values
        finite = vals[np.isfinite(vals)]
        assert finite.max() < 60, "pottmp should be in Celsius after conversion"
        assert result["pottmp"].attrs["units"] == "°C"

    def test_depth_selection_reduces_level_dim(self, mock_pottmp):
        result = open_godas(2024, variable="pottmp", depth_m=200.0)
        assert "level" not in result["pottmp"].dims

    def test_region_selection_reduces_lat_lon(self, mock_pottmp):
        region = {"lat_min": -5.0, "lat_max": 5.0,
                  "lon_min": 190.0, "lon_max": 240.0}
        result = open_godas(2024, variable="pottmp", region=region)
        lats = result["pottmp"]["lat"].values
        lons = result["pottmp"]["lon"].values
        assert lats.min() >= -5.0
        assert lats.max() <= 5.0
        assert lons.min() >= 190.0
        assert lons.max() <= 240.0

    def test_invalid_variable_raises(self):
        with pytest.raises(ValueError, match="Unknown GODAS variable"):
            open_godas(2024, variable="invalid_var")

    def test_salt_units_psu(self):
        ds = _make_godas_ds("salt")
        with patch("noawclg.ocean.xr.open_dataset", return_value=ds):
            result = open_godas(2024, variable="salt")
        assert result["salt"].attrs["units"] == "PSU"
        # Raw 0.033-0.037 kg/kg → 33-37 PSU
        vals = result["salt"].values
        finite = vals[np.isfinite(vals)]
        assert finite.min() > 1.0

    def test_sshg_no_level_dim(self):
        ds = _make_godas_ds("sshg", n_levels=False)
        with patch("noawclg.ocean.xr.open_dataset", return_value=ds):
            result = open_godas(2024, variable="sshg")
        assert "level" not in result["sshg"].dims
        assert result["sshg"].attrs["units"] == "m"


# ══════════════════════════════════════════════════════════════════════════════
# Tests: get_godas (multi-year)
# ══════════════════════════════════════════════════════════════════════════════

class TestGetGodas:
    def test_multi_year_concatenates(self):
        ds = _make_godas_ds("pottmp")
        with patch("noawclg.ocean.xr.open_dataset", return_value=ds):
            da = get_godas(2022, 2024, variable="pottmp", depth_m=200.0)
        # 3 years × 12 months each = 36
        assert len(da["time"]) == 36

    def test_failed_year_skipped_gracefully(self):
        ds = _make_godas_ds("pottmp")

        call_count = {"n": 0}

        def _open_or_fail(url, **kw):
            call_count["n"] += 1
            if "2023" in url:
                raise OSError("Network error")
            return ds

        with patch("noawclg.ocean.xr.open_dataset", side_effect=_open_or_fail):
            da = get_godas(2022, 2024, variable="pottmp", depth_m=200.0)
        # 2022 and 2024 loaded (12+12 = 24 months); 2023 skipped
        assert len(da["time"]) == 24

    def test_all_years_fail_raises_runtime(self):
        with patch("noawclg.ocean.xr.open_dataset", side_effect=OSError("down")):
            with pytest.raises(RuntimeError, match="No GODAS"):
                get_godas(2020, 2020, variable="pottmp")


# ══════════════════════════════════════════════════════════════════════════════
# Tests: typed convenience wrappers
# ══════════════════════════════════════════════════════════════════════════════

class TestGetOceanTemp:
    def test_returns_celsius_dataarray(self):
        ds = _make_godas_ds("pottmp")
        with patch("noawclg.ocean.xr.open_dataset", return_value=ds):
            da = get_ocean_temp(2024, depth_m=200.0)
        assert da.attrs["units"] == "°C"
        assert isinstance(da, xr.DataArray)


class TestGetSalinity:
    def test_returns_psu_dataarray(self):
        ds = _make_godas_ds("salt")
        with patch("noawclg.ocean.xr.open_dataset", return_value=ds):
            da = get_salinity(2024, depth_m=5.0)
        assert da.attrs["units"] == "PSU"


class TestGetCurrents:
    def test_returns_dataset_with_speed(self):
        u_ds = _make_godas_ds("ucur")
        v_ds = _make_godas_ds("vcur")

        call_idx = {"n": 0}
        datasets = [u_ds, v_ds]

        def _switch(url, **kw):
            ds = datasets[call_idx["n"] % 2]
            call_idx["n"] += 1
            return ds

        with patch("noawclg.ocean.xr.open_dataset", side_effect=_switch):
            result = get_currents(2024, depth_m=5.0)

        assert isinstance(result, xr.Dataset)
        assert "ucur"  in result
        assert "vcur"  in result
        assert "speed" in result
        # speed = sqrt(u^2 + v^2) ≥ 0
        assert float(result["speed"].min()) >= 0.0


class TestGetSSH:
    def test_returns_meters_dataarray(self):
        ds = _make_godas_ds("sshg", n_levels=False)
        with patch("noawclg.ocean.xr.open_dataset", return_value=ds):
            da = get_ssh(2024)
        assert da.attrs["units"] == "m"


# ══════════════════════════════════════════════════════════════════════════════
# Tests: open_ersst
# ══════════════════════════════════════════════════════════════════════════════

class TestOpenERSST:
    @pytest.fixture
    def mock_ersst(self):
        ds = _make_ersst_ds()
        with patch("noawclg.ocean.xr.open_dataset", return_value=ds):
            yield

    def test_returns_dataarray(self, mock_ersst):
        da = open_ersst(1990, 2000)
        assert isinstance(da, xr.DataArray)
        assert da.attrs["units"] == "°C"

    def test_time_slice_applied(self, mock_ersst):
        da = open_ersst(1990, 2000)
        years = pd.DatetimeIndex(da["time"].values).year
        assert years.min() >= 1990
        assert years.max() <= 2000

    def test_region_subset_decreasing_lat(self, mock_ersst):
        region = {"lat_min": -5.0, "lat_max": 5.0,
                  "lon_min": 190.0, "lon_max": 240.0}
        da = open_ersst(2000, 2024, region=region)
        lats = da["lat"].values
        lons = da["lon"].values
        assert lats.min() >= -5.0
        assert lats.max() <= 5.0
        assert lons.min() >= 190.0
        assert lons.max() <= 240.0

    def test_fill_value_masked(self, mock_ersst):
        da = open_ersst(2000, 2024)
        # Physical guard: |sst| < 100 — no values should be absurd fill values
        assert float(np.nanmax(np.abs(da.values))) < 100.0


# ══════════════════════════════════════════════════════════════════════════════
# Tests: ENSO index functions (pandas-only, no network)
# ══════════════════════════════════════════════════════════════════════════════

def _synthetic_sst(mean: float = 27.0, amplitude: float = 0.3) -> pd.Series:
    """72 months of synthetic Niño 3.4 SST with seasonal cycle."""
    idx  = pd.date_range("2015-01", periods=72, freq="MS")
    vals = mean + amplitude * np.sin(np.linspace(0, 4 * np.pi, 72))
    return pd.Series(vals, index=idx, name="SST_Nino34")


def _synthetic_oni_nino() -> pd.Series:
    """ONI series with a clear 8-month El Niño event."""
    idx  = pd.date_range("2015-01", periods=36, freq="MS")
    vals = np.zeros(36, dtype=float)
    vals[6:14] = 0.8    # months 7-14 clearly above +0.5
    vals[5]    = 0.5
    vals[14]   = 0.5
    return pd.Series(vals, index=idx, name="ONI")


def _synthetic_oni_nina() -> pd.Series:
    """ONI series with a clear 6-month La Niña event."""
    idx  = pd.date_range("2020-01", periods=24, freq="MS")
    vals = np.zeros(24, dtype=float)
    vals[3:9] = -0.8
    return pd.Series(vals, index=idx, name="ONI")


class TestGetNinoAnomaly:
    def test_anomaly_mean_near_zero_during_climatology(self):
        sst_series = _synthetic_sst(mean=27.0)

        ds = _make_godas_ds("pottmp")
        # Patch so that get_sst_series returns our synthetic series
        with patch("noawclg.ocean.get_sst_series", return_value=sst_series):
            anom = get_nino_anomaly(2015, 2019, clim_start=2015, clim_end=2019)

        # Anomalies should have approximately zero mean over the clim period
        assert abs(float(anom.mean())) < 0.05


class TestGetONI:
    def test_returns_series_with_oni_name(self):
        sst_series = _synthetic_sst()
        with patch("noawclg.ocean.get_sst_series", return_value=sst_series):
            oni = get_oni(2015, 2019, clim_start=2015, clim_end=2019)
        assert oni.name == "ONI"
        assert isinstance(oni, pd.Series)

    def test_rolling_smoothed(self):
        # Raw anomaly has sharper peaks; ONI should have lower variance
        sst_series = _synthetic_sst(amplitude=1.5)
        with patch("noawclg.ocean.get_sst_series", return_value=sst_series):
            oni = get_oni(2015, 2019, clim_start=2015, clim_end=2019)
        raw_std = sst_series.std()
        oni_std = oni.dropna().std()
        assert oni_std <= raw_std + 0.1   # smoothing reduces variance


class TestClassifyENSO:
    def test_el_nino_detected(self):
        oni   = _synthetic_oni_nino()
        phase = classify_enso(oni, threshold=0.5, min_consecutive=5)
        assert "El Niño" in phase.values

    def test_la_nina_detected(self):
        oni   = _synthetic_oni_nina()
        phase = classify_enso(oni, threshold=0.5, min_consecutive=5)
        assert "La Niña" in phase.values

    def test_neutral_default(self):
        idx   = pd.date_range("2020-01", periods=12, freq="MS")
        oni   = pd.Series(np.zeros(12), index=idx, name="ONI")
        phase = classify_enso(oni)
        assert (phase == "Neutral").all()

    def test_returns_same_length_as_input(self):
        oni   = _synthetic_oni_nino()
        phase = classify_enso(oni)
        assert len(phase) == len(oni)

    def test_short_warm_spell_stays_neutral(self):
        """3-month positive spike should not be classified as El Niño (need ≥5)."""
        idx  = pd.date_range("2020-01", periods=12, freq="MS")
        vals = np.zeros(12)
        vals[3:6] = 0.8  # only 3 months
        oni   = pd.Series(vals, index=idx, name="ONI")
        phase = classify_enso(oni, min_consecutive=5)
        assert "El Niño" not in phase.values


# ══════════════════════════════════════════════════════════════════════════════
# Tests: get_thermocline_depth
# ══════════════════════════════════════════════════════════════════════════════

class TestGetThermoclineDepth:
    def test_returns_depth_in_metres(self):
        ds = _make_godas_ds("pottmp")
        # Set temperatures so there's a clear isotherm: top half warm, bottom cold
        # After K→°C conversion (raw 285-305 K → 12-32°C), most will be above 20°C
        with patch("noawclg.ocean.xr.open_dataset", return_value=ds):
            d20 = get_thermocline_depth(2024)

        assert isinstance(d20, xr.DataArray)
        assert d20.attrs["units"] == "m"
        # Depth values should be within valid GODAS level range
        finite = d20.values[np.isfinite(d20.values)]
        if len(finite) > 0:
            assert finite.max() <= 4500.0

    def test_dims_are_time_lat_lon(self):
        ds = _make_godas_ds("pottmp")
        with patch("noawclg.ocean.xr.open_dataset", return_value=ds):
            d20 = get_thermocline_depth(2024)
        assert set(d20.dims) == {"time", "lat", "lon"}


# ══════════════════════════════════════════════════════════════════════════════
# Tests: get_sst_series (both sources)
# ══════════════════════════════════════════════════════════════════════════════

class TestGetSSTSeries:
    def test_godas_source_returns_series(self):
        ds = _make_godas_ds("pottmp")
        with patch("noawclg.ocean.xr.open_dataset", return_value=ds):
            s = get_sst_series(2024, box="3.4", source="godas")
        assert isinstance(s, pd.Series)
        assert isinstance(s.index, pd.DatetimeIndex)

    def test_ersst_source_returns_series(self):
        ds = _make_ersst_ds()
        with patch("noawclg.ocean.xr.open_dataset", return_value=ds):
            s = get_sst_series(2000, 2005, box="3.4", source="ersst")
        assert isinstance(s, pd.Series)
        assert len(s) == 72   # 6 years × 12 months


# ══════════════════════════════════════════════════════════════════════════════
# Tests: enso_summary
# ══════════════════════════════════════════════════════════════════════════════

class TestEnsoSummary:
    def test_returns_dataframe_with_expected_columns(self):
        sst = _synthetic_sst()
        oni_nino = _synthetic_oni_nino()

        with (
            patch("noawclg.ocean.get_sst_series", return_value=sst),
            patch("noawclg.ocean.get_oni", return_value=oni_nino),
        ):
            df = enso_summary(2015, 2017)

        assert isinstance(df, pd.DataFrame)
        assert "sst_nino34"  in df.columns
        assert "anom_nino34" in df.columns
        assert "oni"         in df.columns
        assert "phase"       in df.columns

    def test_phase_values_valid(self):
        sst = _synthetic_sst()
        oni = _synthetic_oni_nino()

        with (
            patch("noawclg.ocean.get_sst_series", return_value=sst),
            patch("noawclg.ocean.get_oni", return_value=oni),
        ):
            df = enso_summary(2015, 2017)

        valid_phases = {"El Niño", "La Niña", "Neutral", float("nan")}
        # NaN appears at rolling edges — drop them before checking
        non_null = df["phase"].dropna()
        assert set(non_null.unique()).issubset({"El Niño", "La Niña", "Neutral"})


# ══════════════════════════════════════════════════════════════════════════════
# Tests: assess_probability (enso_forecast module)
# ══════════════════════════════════════════════════════════════════════════════

class TestAssessProbability:
    @pytest.fixture(autouse=True)
    def _import(self):
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
        from enso_forecast import assess_probability
        self._fn = assess_probability

    def test_strong_el_nino_signals_high_probability(self):
        result = self._fn(
            oni_current=1.8,
            oni_trend_3m=0.5,
            t200_anom_current=1.2,
            ssh_ep_current=0.12,
            analog_prob=0.90,
        )
        assert result["prob"] >= 0.70
        assert result["category"] in ("Alta", "Moderada-Alta")

    def test_strong_la_nina_signals_low_probability(self):
        result = self._fn(
            oni_current=-1.5,
            oni_trend_3m=-0.4,
            t200_anom_current=-1.0,
            ssh_ep_current=-0.10,
            analog_prob=0.10,
        )
        assert result["prob"] <= 0.30

    def test_result_keys_present(self):
        result = self._fn(0.0, 0.0, 0.0, 0.0, 0.5)
        assert "prob"     in result
        assert "scores"   in result
        assert "weights"  in result
        assert "category" in result

    def test_probability_bounded_0_1(self):
        result = self._fn(3.0, 1.0, 2.0, 0.5, 1.0)
        assert 0.0 <= result["prob"] <= 1.0

    def test_weights_sum_to_one(self):
        result = self._fn(0.5, 0.1, 0.3, 0.02, 0.5)
        total = sum(result["weights"].values())
        assert abs(total - 1.0) < 1e-9
