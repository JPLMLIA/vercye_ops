"""Tests for vercye_ops.matching_sim_real.generate_apsim_yield_map"""

import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine

from vercye_ops.matching_sim_real.generate_apsim_yield_map import process_geotiff


def _create_reference_tif(path, data, nodata=None):
    transform = Affine(0.01, 0, 30.0, 0, -0.01, 50.0)
    h, w = data.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": data.dtype,
        "crs": "EPSG:4326",
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(data, 1)


def _create_csv(path, apsim_value):
    pd.DataFrame({"apsim_mean_yield_estimate_kg_ha": [apsim_value]}).to_csv(str(path), index=False)


class TestProcessGeotiff:
    def test_broadcasts_scalar_to_valid_pixels(self, tmp_path):
        # Mix of finite and NaN cells; finite cells are cropland mask
        data = np.array([[100.0, np.nan], [200.0, 300.0]], dtype=np.float32)
        ref = tmp_path / "ref.tif"
        csv = tmp_path / "cf.csv"
        out = tmp_path / "out.tif"
        _create_reference_tif(ref, data, nodata=np.nan)
        _create_csv(csv, 4200.0)

        process_geotiff(str(ref), str(csv), str(out))

        with rasterio.open(str(out)) as src:
            result = src.read(1)
            assert src.nodata is not None and np.isnan(src.nodata)

        # Valid pixels = APSIM scalar; NaN preserved
        assert np.isnan(result[0, 1])
        np.testing.assert_allclose(result[0, 0], 4200.0)
        np.testing.assert_allclose(result[1, 0], 4200.0)
        np.testing.assert_allclose(result[1, 1], 4200.0)

    def test_missing_apsim_column_raises(self, tmp_path):
        data = np.array([[1.0]], dtype=np.float32)
        ref = tmp_path / "ref.tif"
        csv = tmp_path / "cf.csv"
        out = tmp_path / "out.tif"
        _create_reference_tif(ref, data)
        pd.DataFrame({"conversion_factor": [500]}).to_csv(str(csv), index=False)

        with pytest.raises(KeyError, match="apsim_mean_yield_estimate_kg_ha"):
            process_geotiff(str(ref), str(csv), str(out))

    def test_preserves_spatial_metadata(self, tmp_path):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        ref = tmp_path / "ref.tif"
        csv = tmp_path / "cf.csv"
        out = tmp_path / "out.tif"
        _create_reference_tif(ref, data, nodata=np.nan)
        _create_csv(csv, 3000.0)

        process_geotiff(str(ref), str(csv), str(out))

        with rasterio.open(str(ref)) as src_in, rasterio.open(str(out)) as src_out:
            assert src_in.crs == src_out.crs
            assert src_in.transform == src_out.transform
            assert src_in.width == src_out.width
            assert src_in.height == src_out.height

    def test_all_nan_reference_produces_all_nan_output(self, tmp_path):
        data = np.full((2, 2), np.nan, dtype=np.float32)
        ref = tmp_path / "ref.tif"
        csv = tmp_path / "cf.csv"
        out = tmp_path / "out.tif"
        _create_reference_tif(ref, data, nodata=np.nan)
        _create_csv(csv, 5000.0)

        process_geotiff(str(ref), str(csv), str(out))

        with rasterio.open(str(out)) as src:
            result = src.read(1)
        assert np.all(np.isnan(result))
