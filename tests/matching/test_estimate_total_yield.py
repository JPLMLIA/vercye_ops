"""Tests for vercye_ops.matching_sim_real.estimate_total_yield"""

import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine

from vercye_ops.matching_sim_real.estimate_total_yield import estimate_yield

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_yield_tif(path, data, transform=None, crs="EPSG:4326"):
    """Write a single-band GeoTIFF with the given 2D array."""
    if transform is None:
        transform = Affine(0.01, 0, 30.0, 0, -0.01, 50.0)
    h, w = data.shape
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEstimateYield:
    def test_basic_yield_estimation(self, tmp_path):
        """Simple 2x2 raster, all valid pixels."""
        data = np.array([[1000.0, 2000.0], [3000.0, 4000.0]], dtype=np.float32)
        tif_path = tmp_path / "yield.tif"
        csv_path = tmp_path / "yield.csv"
        _create_yield_tif(tif_path, data)

        estimate_yield(str(tif_path), str(csv_path), target_crs="EPSG:32636")

        df = pd.read_csv(csv_path)
        assert df["mean_yield_kg_ha"].iloc[0] == int(np.mean(data))
        assert df["median_yield_kg_ha"].iloc[0] == int(np.median(data))
        assert df["total_area_ha"].iloc[0] > 0

    def test_with_nan_pixels(self, tmp_path):
        """NaN pixels should be excluded from mean/count but not crash."""
        data = np.array([[1000.0, np.nan], [np.nan, 4000.0]], dtype=np.float32)
        tif_path = tmp_path / "yield.tif"
        csv_path = tmp_path / "yield.csv"
        _create_yield_tif(tif_path, data)

        estimate_yield(str(tif_path), str(csv_path), target_crs="EPSG:32636")

        df = pd.read_csv(csv_path)
        # Only 2 valid pixels
        assert df["mean_yield_kg_ha"].iloc[0] == int(np.nanmean(data))
        # Total area should reflect 2 pixels, not 4
        # (exact value depends on pixel size, but should be roughly half)

    def test_all_nan_returns_defaults(self, tmp_path):
        """All-NaN data should produce zero yields and -1 area."""
        data = np.full((2, 2), np.nan, dtype=np.float32)
        tif_path = tmp_path / "yield.tif"
        csv_path = tmp_path / "yield.csv"
        _create_yield_tif(tif_path, data)

        estimate_yield(str(tif_path), str(csv_path), target_crs="EPSG:32636")

        df = pd.read_csv(csv_path)
        assert df["mean_yield_kg_ha"].iloc[0] == 0
        assert df["median_yield_kg_ha"].iloc[0] == 0
        assert df["total_area_ha"].iloc[0] == -1
        assert df["total_yield_production_kg"].iloc[0] == 0

    def test_total_yield_tons_conversion(self, tmp_path):
        """total_yield_production_ton should be total_yield_production_kg / 1000."""
        data = np.array([[5000.0, 5000.0]], dtype=np.float32)
        tif_path = tmp_path / "yield.tif"
        csv_path = tmp_path / "yield.csv"
        _create_yield_tif(tif_path, data)

        estimate_yield(str(tif_path), str(csv_path), target_crs="EPSG:32636")

        df = pd.read_csv(csv_path)
        kg = df["total_yield_production_kg"].iloc[0]
        ton = df["total_yield_production_ton"].iloc[0]
        assert ton == pytest.approx(kg / 1000, abs=0.01)

    def test_non_degree_crs_raises(self, tmp_path):
        """Input in projected CRS (meters) should raise ValueError."""
        data = np.array([[1000.0]], dtype=np.float32)
        transform = Affine(10, 0, 500000, 0, -10, 5000000)
        tif_path = tmp_path / "yield.tif"
        csv_path = tmp_path / "yield.csv"
        _create_yield_tif(tif_path, data, transform=transform, crs="EPSG:32636")

        with pytest.raises(ValueError, match="not in degrees"):
            estimate_yield(str(tif_path), str(csv_path), target_crs="EPSG:32636")

    def test_csv_has_expected_columns(self, tmp_path):
        data = np.array([[1000.0]], dtype=np.float32)
        tif_path = tmp_path / "yield.tif"
        csv_path = tmp_path / "yield.csv"
        _create_yield_tif(tif_path, data)

        estimate_yield(str(tif_path), str(csv_path), target_crs="EPSG:32636")

        df = pd.read_csv(csv_path)
        expected_cols = {
            "mean_yield_kg_ha",
            "median_yield_kg_ha",
            "total_area_ha",
            "total_yield_production_kg",
            "total_yield_production_ton",
        }
        assert set(df.columns) == expected_cols
