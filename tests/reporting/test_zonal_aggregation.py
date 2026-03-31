"""Tests for zonal_aggregation module."""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box


def _create_test_raster(path, bounds, shape=(100, 100), crs="EPSG:32637", data=None, nodata=float("nan")):
    """Create a test raster in an equal-area CRS."""
    left, bottom, right, top = bounds
    transform = from_bounds(left, bottom, right, top, shape[1], shape[0])

    if data is None:
        data = np.random.uniform(1000, 5000, shape).astype(np.float32)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": shape[1],
        "height": shape[0],
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _create_test_coverage(path, bounds, shape=(100, 100), crs="EPSG:32637", coverage_pct=1.0):
    """Create a test coverage mask raster."""
    left, bottom, right, top = bounds
    transform = from_bounds(left, bottom, right, top, shape[1], shape[0])

    data = np.ones(shape, dtype=np.uint8)
    # Set a fraction of pixels to 0 (not covered)
    n_uncovered = int(shape[0] * shape[1] * (1 - coverage_pct))
    if n_uncovered > 0:
        flat = data.flatten()
        flat[:n_uncovered] = 0
        data = flat.reshape(shape)

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": shape[1],
        "height": shape[0],
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": 255,
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _create_test_shapefile(path, polygons, crs="EPSG:32637", extra_cols=None):
    """Create a test shapefile with given polygons."""
    gdf = gpd.GeoDataFrame(polygons, crs=crs)
    if extra_cols:
        for col, values in extra_cols.items():
            gdf[col] = values
    gdf.to_file(path)


@pytest.fixture
def test_data(tmp_path):
    """Create test raster and shapefile data."""
    # Use UTM Zone 37N (meters)
    crs = "EPSG:32637"
    bounds = (300000, 5400000, 400000, 5500000)  # 100km x 100km area

    # Create yield raster with known values
    shape = (100, 100)
    yield_data = np.full(shape, 3000.0, dtype=np.float32)  # 3000 kg/ha everywhere
    yield_data[0:10, :] = np.nan  # 10% nodata

    yield_tif = str(tmp_path / "yield_mosaic.tif")
    _create_test_raster(yield_tif, bounds, shape, crs, yield_data)

    # Create coverage mask (full coverage)
    cov_tif = str(tmp_path / "coverage.tif")
    _create_test_coverage(cov_tif, bounds, shape, crs, coverage_pct=1.0)

    # Create shapefile with two polygons splitting the area
    mid_x = (bounds[0] + bounds[2]) / 2
    poly1 = box(bounds[0], bounds[1], mid_x, bounds[3])
    poly2 = box(mid_x, bounds[1], bounds[2], bounds[3])

    shp_path = str(tmp_path / "regions.shp")
    _create_test_shapefile(
        shp_path,
        {"geometry": [poly1, poly2], "NAME": ["Region_A", "Region_B"]},
        crs=crs,
        extra_cols={"ref_yield": [2800.0, 3200.0]},
    )

    return {
        "yield_tif": yield_tif,
        "coverage_tif": cov_tif,
        "shapefile": shp_path,
        "bounds": bounds,
        "shape": shape,
    }


class TestComputeZonalYieldStats:
    def test_basic_stats(self, test_data):
        from vercye_ops.reporting.zonal_aggregation import compute_zonal_yield_stats

        result = compute_zonal_yield_stats(
            yield_mosaic_tif=test_data["yield_tif"],
            coverage_mask_tif=test_data["coverage_tif"],
            shapefile_path=test_data["shapefile"],
            name_column="NAME",
        )

        assert len(result) == 2
        assert "region" in result.columns
        assert "mean_yield_kg_ha" in result.columns
        assert "total_production_kg" in result.columns
        assert "total_cropland_area_ha" in result.columns
        assert "coverage_pct" in result.columns

    def test_yield_values_reasonable(self, test_data):
        from vercye_ops.reporting.zonal_aggregation import compute_zonal_yield_stats

        result = compute_zonal_yield_stats(
            test_data["yield_tif"],
            test_data["coverage_tif"],
            test_data["shapefile"],
            "NAME",
        )

        # All valid pixels have yield 3000, so mean should be 3000
        for _, row in result.iterrows():
            assert abs(row["mean_yield_kg_ha"] - 3000) < 100  # some tolerance for edge effects

    def test_coverage_pct(self, test_data):
        from vercye_ops.reporting.zonal_aggregation import compute_zonal_yield_stats

        result = compute_zonal_yield_stats(
            test_data["yield_tif"],
            test_data["coverage_tif"],
            test_data["shapefile"],
            "NAME",
        )

        # Full coverage mask, so coverage should be ~100%
        for _, row in result.iterrows():
            assert row["coverage_pct"] >= 90  # allow some tolerance

    def test_reference_yield_column(self, test_data):
        from vercye_ops.reporting.zonal_aggregation import compute_zonal_yield_stats

        result = compute_zonal_yield_stats(
            test_data["yield_tif"],
            test_data["coverage_tif"],
            test_data["shapefile"],
            "NAME",
            reference_yield_column="ref_yield",
        )

        assert "reported_mean_yield_kg_ha" in result.columns
        ref_values = result["reported_mean_yield_kg_ha"].tolist()
        assert 2800.0 in ref_values
        assert 3200.0 in ref_values

    def test_missing_name_column_raises(self, test_data):
        from vercye_ops.reporting.zonal_aggregation import compute_zonal_yield_stats

        with pytest.raises(ValueError, match="not found in shapefile"):
            compute_zonal_yield_stats(
                test_data["yield_tif"],
                test_data["coverage_tif"],
                test_data["shapefile"],
                "NONEXISTENT_COLUMN",
            )

    def test_missing_reference_column_raises(self, test_data):
        from vercye_ops.reporting.zonal_aggregation import compute_zonal_yield_stats

        with pytest.raises(ValueError, match="Reference yield column"):
            compute_zonal_yield_stats(
                test_data["yield_tif"],
                test_data["coverage_tif"],
                test_data["shapefile"],
                "NAME",
                reference_yield_column="nonexistent",
            )

    def test_production_is_yield_times_area(self, test_data):
        from vercye_ops.reporting.zonal_aggregation import compute_zonal_yield_stats

        result = compute_zonal_yield_stats(
            test_data["yield_tif"],
            test_data["coverage_tif"],
            test_data["shapefile"],
            "NAME",
        )

        for _, row in result.iterrows():
            expected_production = row["mean_yield_kg_ha"] * row["total_cropland_area_ha"]
            # Allow 10% tolerance due to pixel-level aggregation
            assert abs(row["total_production_kg"] - expected_production) / expected_production < 0.1


class TestExtractReferenceFromShapefile:
    def test_basic_extraction(self, test_data):
        from vercye_ops.reporting.zonal_aggregation import extract_reference_from_shapefile

        result = extract_reference_from_shapefile(test_data["shapefile"], "NAME", "ref_yield")

        assert len(result) == 2
        assert "region" in result.columns
        assert "reported_mean_yield_kg_ha" in result.columns

    def test_missing_column_raises(self, test_data):
        from vercye_ops.reporting.zonal_aggregation import extract_reference_from_shapefile

        with pytest.raises(ValueError):
            extract_reference_from_shapefile(test_data["shapefile"], "NAME", "nonexistent")
