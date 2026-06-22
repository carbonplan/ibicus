# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Tests for dask/xarray integration via Debiaser._apply_xarray().

These tests verify that:
- apply() accepts xarray DataArrays and returns a DataArray
- dask-backed inputs produce lazy (un-computed) outputs
- computed dask results are numerically identical to the numpy path
- output dimension order matches input
- running-window mode works correctly (time coords extracted automatically)
"""

import numpy as np
import pytest
import xarray as xr

from ibicus.debias import LinearScaling, QuantileMapping

dask = pytest.importorskip("dask")  # skip entire module if dask not installed


def make_time(periods, time_start="2000-01-01"):
    """Daily cftime coordinate, shared by the xarray wrapper and numpy reference runs."""
    return xr.date_range(time_start, periods=periods, freq="D", use_cftime=True)


def make_xarray(data, time_start="2000-01-01"):
    """Wrap a (time, x, y) numpy array in an xr.DataArray with a daily time coord."""
    time = make_time(data.shape[0], time_start)
    return xr.DataArray(data, dims=["time", "x", "y"], coords={"time": time})


class TestDaskIntegration:
    @classmethod
    def setup_class(cls):
        np.random.seed(12345)
        n = 1000
        cls.obs_np = np.random.normal(size=(n, 2, 2)) + 270
        cls.cm_hist_np = np.random.normal(size=(n, 2, 2)) + 272
        cls.cm_future_np = np.random.normal(size=(n, 2, 2)) + 274

    def test_xarray_eager_returns_dataarray(self):
        """apply() on a non-chunked xarray DataArray returns an xr.DataArray."""
        debiaser = QuantileMapping.from_variable("tas", running_window_mode=False)
        obs_xr = make_xarray(self.obs_np)
        cm_hist_xr = make_xarray(self.cm_hist_np)
        cm_future_xr = make_xarray(self.cm_future_np)

        result = debiaser.apply(obs_xr, cm_hist_xr, cm_future_xr)

        assert isinstance(result, xr.DataArray)
        assert result.dims == obs_xr.dims
        assert result.shape == cm_future_xr.shape

    def test_xarray_eager_matches_numpy(self):
        """apply() on non-chunked xarray gives the same result as the numpy path."""
        debiaser = QuantileMapping.from_variable("tas", running_window_mode=False)

        result_np = debiaser.apply(self.obs_np, self.cm_hist_np, self.cm_future_np)

        obs_xr = make_xarray(self.obs_np)
        cm_hist_xr = make_xarray(self.cm_hist_np)
        cm_future_xr = make_xarray(self.cm_future_np)
        result_xr = debiaser.apply(obs_xr, cm_hist_xr, cm_future_xr)

        np.testing.assert_allclose(result_xr.values, result_np, rtol=1e-5)

    def test_dask_output_is_lazy(self):
        """apply() on dask-backed xarray returns a lazy (un-computed) DataArray."""
        debiaser = QuantileMapping.from_variable("tas", running_window_mode=False)

        obs_xr = make_xarray(self.obs_np).chunk({"x": 1, "y": 1})
        cm_hist_xr = make_xarray(self.cm_hist_np).chunk({"x": 1, "y": 1})
        cm_future_xr = make_xarray(self.cm_future_np).chunk({"x": 1, "y": 1})

        result = debiaser.apply(obs_xr, cm_hist_xr, cm_future_xr)

        assert isinstance(result, xr.DataArray)
        assert result.chunks is not None, "Expected a dask-backed (lazy) DataArray"

    def test_dask_matches_numpy(self):
        """Computed dask result is numerically identical to the numpy path."""
        debiaser = QuantileMapping.from_variable("tas", running_window_mode=False)

        result_np = debiaser.apply(self.obs_np, self.cm_hist_np, self.cm_future_np)

        obs_xr = make_xarray(self.obs_np).chunk({"x": 1, "y": 1})
        cm_hist_xr = make_xarray(self.cm_hist_np).chunk({"x": 1, "y": 1})
        cm_future_xr = make_xarray(self.cm_future_np).chunk({"x": 1, "y": 1})

        result_dask = debiaser.apply(obs_xr, cm_hist_xr, cm_future_xr).compute()
        np.testing.assert_allclose(result_dask.values, result_np, rtol=1e-5)

    def test_dim_order_preserved(self):
        """Output DataArray has the same dimension order as the input."""
        debiaser = LinearScaling.from_variable("tas", running_window_mode=False)
        obs_xr = make_xarray(self.obs_np).chunk({"x": 1})
        cm_hist_xr = make_xarray(self.cm_hist_np).chunk({"x": 1})
        cm_future_xr = make_xarray(self.cm_future_np).chunk({"x": 1})

        result = debiaser.apply(obs_xr, cm_hist_xr, cm_future_xr)

        assert result.dims == obs_xr.dims

    def test_running_window_with_dask(self):
        """Running-window mode works with dask (time coords extracted automatically)."""
        np.random.seed(12345)
        n = 3650  # ~10 years of daily data
        obs_np = np.random.normal(size=(n, 2, 2)) + 270
        cm_hist_np = np.random.normal(size=(n, 2, 2)) + 272
        cm_future_np = np.random.normal(size=(n, 2, 2)) + 274

        debiaser = QuantileMapping.from_variable("tas", running_window_mode=True)

        obs_xr = make_xarray(obs_np).chunk({"x": 1, "y": 1})
        cm_hist_xr = make_xarray(cm_hist_np).chunk({"x": 1, "y": 1})
        cm_future_xr = make_xarray(cm_future_np).chunk({"x": 1, "y": 1})

        result = debiaser.apply(obs_xr, cm_hist_xr, cm_future_xr).compute()

        assert result.shape == cm_future_xr.shape
        assert np.all(np.isfinite(result.values))

    def test_running_window_dask_matches_numpy(self):
        """Running-window dask result matches the numpy path."""
        np.random.seed(12345)
        n = 3650
        obs_np = np.random.normal(size=(n, 2, 2)) + 270
        cm_hist_np = np.random.normal(size=(n, 2, 2)) + 272
        cm_future_np = np.random.normal(size=(n, 2, 2)) + 274

        debiaser = QuantileMapping.from_variable("tas", running_window_mode=True)

        # The dask path derives time coords from the DataArray; give the numpy
        # reference the same time information so both window the year identically.
        time = make_time(n).values
        result_np = debiaser.apply(
            obs_np,
            cm_hist_np,
            cm_future_np,
            time_obs=time,
            time_cm_hist=time,
            time_cm_future=time,
        )

        obs_xr = make_xarray(obs_np).chunk({"x": 1, "y": 1})
        cm_hist_xr = make_xarray(cm_hist_np).chunk({"x": 1, "y": 1})
        cm_future_xr = make_xarray(cm_future_np).chunk({"x": 1, "y": 1})

        result_dask = debiaser.apply(obs_xr, cm_hist_xr, cm_future_xr).compute()

        np.testing.assert_allclose(result_dask.values, result_np, rtol=1e-5)

    def test_precipitation_hurdle_gamma_dask(self):
        """QuantileMapping('pr') (PrecipitationHurdleModelGamma) runs lazily over dask
        and matches the numpy path when randomization is disabled."""
        rng = np.random.default_rng(12345)
        n = 1000
        shape = (n, 2, 2)

        def make_precip():
            # gamma-distributed wet days with ~40% dry (zero) days to exercise the
            # hurdle p0 branch
            data = rng.gamma(shape=2.0, scale=2.0, size=shape)
            data[rng.random(shape) < 0.4] = 0.0
            return data

        obs_np, cm_hist_np, cm_future_np = make_precip(), make_precip(), make_precip()

        debiaser = QuantileMapping.for_precipitation(
            model_type="hurdle",
            running_window_mode=False,
            hurdle_model_randomization=False,
        )

        result_np = debiaser.apply(obs_np, cm_hist_np, cm_future_np)

        obs_xr = make_xarray(obs_np).chunk({"x": 1, "y": 1})
        cm_hist_xr = make_xarray(cm_hist_np).chunk({"x": 1, "y": 1})
        cm_future_xr = make_xarray(cm_future_np).chunk({"x": 1, "y": 1})

        result = debiaser.apply(obs_xr, cm_hist_xr, cm_future_xr)
        assert result.chunks is not None, "Expected a dask-backed (lazy) DataArray"

        result_dask = result.compute()
        assert result_dask.shape == cm_future_xr.shape
        assert np.all(np.isfinite(result_dask.values))
        np.testing.assert_allclose(result_dask.values, result_np, rtol=1e-5)
