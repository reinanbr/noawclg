from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from noawclg.main import (
    BoundingBox,
    _DatasetView,
    _find_dim,
    _normalize_lon,
    _parse_date,
    get_noaa_data,
)


def _sample_dataset(var_name: str = "t2m") -> xr.Dataset:
    lat = np.array([-4.0, -3.0, -2.0])
    lon = np.array([320.0, 321.0, 322.0])
    time = np.array(["2026-04-03T00:00", "2026-04-03T03:00"], dtype="datetime64[ns]")

    data = xr.DataArray(
        np.arange(18).reshape(2, 3, 3),
        dims=["time", "latitude", "longitude"],
        coords={"time": time, "latitude": lat, "longitude": lon},
        attrs={"long_name": "2 metre temperature"},
    )

    return xr.Dataset({var_name: data})


class TestHelpers:
    def test_parse_date_br_format(self):
        assert _parse_date("17/04/2026") == "20260417"

    def test_normalize_lon_for_minus180_to_180(self):
        assert _normalize_lon(-38.5, lon_min=-179.5) == pytest.approx(-38.5)

    def test_normalize_lon_for_0_to_360(self):
        assert _normalize_lon(-38.5, lon_min=0.0) == pytest.approx(321.5)

    def test_find_dim_detects_existing(self):
        found = _find_dim(["step", "latitude", "longitude"], ("lat", "latitude"), "lat")
        assert found == "latitude"

    def test_find_dim_raises_with_helpful_message(self):
        with pytest.raises(KeyError, match="Cannot find a lat coordinate"):
            _find_dim(["x", "y"], ("lat", "latitude"), "lat")


class TestDatasetView:
    def test_getitem_returns_variable(self):
        view = _DatasetView(_sample_dataset())
        assert view["t2m"].attrs["long_name"] == "2 metre temperature"

    def test_getitem_missing_key_raises(self):
        view = _DatasetView(_sample_dataset())
        with pytest.raises(KeyError, match="Variable 'prate' not found"):
            view["prate"]


class TestBoundingBox:
    def test_contains(self):
        b = BoundingBox(lat_min=-10, lat_max=10, lon_min=-80, lon_max=-30)
        assert b.contains(-3.7, -38.5)
        assert not b.contains(20.0, -38.5)


class TestGetNoaaData:
    @pytest.fixture
    def ds(self) -> xr.Dataset:
        return _sample_dataset()

    def test_init_uses_single_dataset_for_one_key(self, ds):
        fake_manager = MagicMock()
        fake_manager.build_dataset.return_value = ds

        with patch("noawclg.main.GFSDatasetManager", return_value=fake_manager):
            noaa = get_noaa_data(date="03/04/2026", keys=["t2m"], hours=[0, 3])

        fake_manager.build_dataset.assert_called_once_with("t2m", [0, 3])
        assert noaa.date == "20260403"

    def test_init_uses_multi_dataset_for_many_keys(self, ds):
        fake_manager = MagicMock()
        fake_manager.build_multi_dataset.return_value = ds

        with patch("noawclg.main.GFSDatasetManager", return_value=fake_manager):
            get_noaa_data(date="03/04/2026", keys=["t2m", "prate"], hours=[0, 3])

        fake_manager.build_multi_dataset.assert_called_once_with(
            ["t2m", "prate"], [0, 3]
        )

    def test_init_invalid_key_raises(self):
        with pytest.raises(ValueError, match="Invalid variable keys"):
            get_noaa_data(date="03/04/2026", keys=["not-a-var"], hours=[0])

    def test_get_keys_returns_long_names(self, ds):
        fake_manager = MagicMock()
        fake_manager.build_dataset.return_value = ds

        with patch("noawclg.main.GFSDatasetManager", return_value=fake_manager):
            noaa = get_noaa_data(date="03/04/2026", keys=["t2m"], hours=[0, 3])

        keys = noaa.get_keys()
        assert keys["t2m"] == "2 metre temperature"

    def test_get_data_from_point_selects_nearest(self, ds):
        fake_manager = MagicMock()
        fake_manager.build_dataset.return_value = ds

        with patch("noawclg.main.GFSDatasetManager", return_value=fake_manager):
            noaa = get_noaa_data(date="03/04/2026", keys=["t2m"], hours=[0, 3])

        view = noaa.get_data_from_point((-3.1, -38.5))
        assert isinstance(view, _DatasetView)
        assert float(view.dataset["longitude"]) == pytest.approx(322.0)

    def test_get_data_from_place_uses_geocoder(self, ds):
        fake_manager = MagicMock()
        fake_manager.build_dataset.return_value = ds

        with (
            patch("noawclg.main.GFSDatasetManager", return_value=fake_manager),
            patch(
                "noawclg.main._GEOLOCATOR.geocode",
                return_value=SimpleNamespace(latitude=-3.73, longitude=-38.52),
            ),
        ):
            noaa = get_noaa_data(date="03/04/2026", keys=["t2m"], hours=[0, 3])
            view = noaa.get_data_from_place("Fortaleza, Brazil")

        assert isinstance(view, _DatasetView)

    def test_get_data_from_place_not_found_raises(self, ds):
        fake_manager = MagicMock()
        fake_manager.build_dataset.return_value = ds

        with (
            patch("noawclg.main.GFSDatasetManager", return_value=fake_manager),
            patch("noawclg.main._GEOLOCATOR.geocode", return_value=None),
        ):
            noaa = get_noaa_data(date="03/04/2026", keys=["t2m"], hours=[0, 3])
            with pytest.raises(ValueError, match="Could not geocode"):
                noaa.get_data_from_place("cidade-invalida")

    def test_get_time_series_variable_and_missing(self, ds):
        fake_manager = MagicMock()
        fake_manager.build_dataset.return_value = ds

        with patch("noawclg.main.GFSDatasetManager", return_value=fake_manager):
            noaa = get_noaa_data(date="03/04/2026", keys=["t2m"], hours=[0, 3])

        series = noaa.get_time_series((-3.1, -38.5), variable="t2m")
        assert isinstance(series, xr.DataArray)

        with pytest.raises(KeyError, match="Variable 'prate' not found"):
            noaa.get_time_series((-3.1, -38.5), variable="prate")
