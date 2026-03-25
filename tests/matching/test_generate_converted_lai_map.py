"""Tests for vercye_ops.matching_sim_real.generate_converted_lai_map"""

import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine

from vercye_ops.matching_sim_real.generate_converted_lai_map import process_geotiff

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_tif(path, data, nodata=None):
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


def _create_csv(path, conversion_factor):
    pd.DataFrame({"conversion_factor": [conversion_factor]}).to_csv(str(path), index=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProcessGeotiff:
    def test_basic_conversion(self, tmp_path):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tif_in = tmp_path / "in.tif"
        csv_path = tmp_path / "factor.csv"
        tif_out = tmp_path / "out.tif"
        _create_tif(tif_in, data)
        _create_csv(csv_path, 500.0)

        process_geotiff(str(tif_in), str(csv_path), str(tif_out))

        with rasterio.open(str(tif_out)) as src:
            result = src.read(1)
        np.testing.assert_allclose(result, data * 500.0)

    def test_negative_values_clipped_to_zero(self, tmp_path):
        """Negative LAI values should be clipped to 0 before applying factor."""
        data = np.array([[-1.0, 2.0], [3.0, -0.5]], dtype=np.float32)
        tif_in = tmp_path / "in.tif"
        csv_path = tmp_path / "factor.csv"
        tif_out = tmp_path / "out.tif"
        _create_tif(tif_in, data)
        _create_csv(csv_path, 100.0)

        process_geotiff(str(tif_in), str(csv_path), str(tif_out))

        with rasterio.open(str(tif_out)) as src:
            result = src.read(1)
        expected = np.array([[0.0, 200.0], [300.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(result, expected)

    def test_missing_conversion_factor_column_raises(self, tmp_path):
        data = np.array([[1.0]], dtype=np.float32)
        tif_in = tmp_path / "in.tif"
        csv_path = tmp_path / "factor.csv"
        tif_out = tmp_path / "out.tif"
        _create_tif(tif_in, data)
        pd.DataFrame({"wrong_col": [100]}).to_csv(str(csv_path), index=False)

        with pytest.raises(KeyError, match="conversion_factor"):
            process_geotiff(str(tif_in), str(csv_path), str(tif_out))

    def test_output_is_lzw_compressed(self, tmp_path):
        data = np.array([[1.0, 2.0]], dtype=np.float32)
        tif_in = tmp_path / "in.tif"
        csv_path = tmp_path / "factor.csv"
        tif_out = tmp_path / "out.tif"
        _create_tif(tif_in, data)
        _create_csv(csv_path, 1.0)

        process_geotiff(str(tif_in), str(csv_path), str(tif_out))

        with rasterio.open(str(tif_out)) as src:
            assert src.compression == rasterio.enums.Compression.lzw

    def test_conversion_factor_of_zero(self, tmp_path):
        """Edge case: conversion factor 0 should produce all-zero output."""
        data = np.array([[5.0, 10.0]], dtype=np.float32)
        tif_in = tmp_path / "in.tif"
        csv_path = tmp_path / "factor.csv"
        tif_out = tmp_path / "out.tif"
        _create_tif(tif_in, data)
        _create_csv(csv_path, 0.0)

        process_geotiff(str(tif_in), str(csv_path), str(tif_out))

        with rasterio.open(str(tif_out)) as src:
            result = src.read(1)
        np.testing.assert_allclose(result, 0.0)

    def test_preserves_spatial_metadata(self, tmp_path):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tif_in = tmp_path / "in.tif"
        csv_path = tmp_path / "factor.csv"
        tif_out = tmp_path / "out.tif"
        _create_tif(tif_in, data)
        _create_csv(csv_path, 1.0)

        process_geotiff(str(tif_in), str(csv_path), str(tif_out))

        with rasterio.open(str(tif_in)) as src_in, rasterio.open(str(tif_out)) as src_out:
            assert src_in.crs == src_out.crs
            assert src_in.transform == src_out.transform
            assert src_in.width == src_out.width
            assert src_in.height == src_out.height
