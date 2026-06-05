"""Tests for zonal_aggregation module."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box, mapping


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
        assert "total_area_ha" in result.columns
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

    def test_missing_name_column_raises(self, test_data):
        from vercye_ops.reporting.zonal_aggregation import compute_zonal_yield_stats

        with pytest.raises(ValueError, match="not found in shapefile"):
            compute_zonal_yield_stats(
                test_data["yield_tif"],
                test_data["coverage_tif"],
                test_data["shapefile"],
                "NONEXISTENT_COLUMN",
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
            expected_production = row["mean_yield_kg_ha"] * row["total_area_ha"]
            # Allow 10% tolerance due to pixel-level aggregation
            assert abs(row["total_production_kg"] - expected_production) / expected_production < 0.1

    def test_id_named_column_not_mislabeled(self, tmp_path):
        """Regression: a name_column literally called 'id' must not be overridden by
        exactextract with the 0-based feature index, which previously assigned each
        region the geometry-at-position-N's stats (region '91' got a neighbor's yield).
        """
        from vercye_ops.reporting.zonal_aggregation import compute_zonal_yield_stats

        crs = "EPSG:32637"
        bounds = (300000, 5400000, 400000, 5500000)
        shape = (100, 100)

        # Left half = 1000 kg/ha, right half = 8000 kg/ha. Distinct so a mislabel is visible.
        yield_data = np.empty(shape, dtype=np.float32)
        yield_data[:, : shape[1] // 2] = 1000.0
        yield_data[:, shape[1] // 2 :] = 8000.0
        yield_tif = str(tmp_path / "yield.tif")
        _create_test_raster(yield_tif, bounds, shape, crs, yield_data)

        cov_tif = str(tmp_path / "cov.tif")
        _create_test_coverage(cov_tif, bounds, shape, crs, coverage_pct=1.0)

        mid_x = (bounds[0] + bounds[2]) / 2
        left = box(bounds[0], bounds[1], mid_x, bounds[3])
        right = box(mid_x, bounds[1], bounds[2], bounds[3])
        shp_path = str(tmp_path / "regions_id.shp")
        # Non-sequential ids; "10" sorts before "2" so any positional fallback diverges.
        _create_test_shapefile(shp_path, {"geometry": [left, right], "id": ["2", "10"]}, crs=crs)

        result = compute_zonal_yield_stats(yield_tif, cov_tif, shp_path, name_column="id")

        assert set(result["region"]) == {"2", "10"}
        left_mean = result.loc[result["region"] == "2", "mean_yield_kg_ha"].iloc[0]
        right_mean = result.loc[result["region"] == "10", "mean_yield_kg_ha"].iloc[0]
        assert abs(left_mean - 1000) < 100, f"region '2' (left) should be ~1000, got {left_mean}"
        assert abs(right_mean - 8000) < 100, f"region '10' (right) should be ~8000, got {right_mean}"


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

    def test_name_column_with_null_value_raises(self, tmp_path):
        """A null region identifier is a data error and must fail loudly rather than
        being silently coerced or dropped.
        """
        from vercye_ops.reporting.zonal_aggregation import compute_zonal_yield_stats

        crs = "EPSG:32637"
        bounds = (300000, 5400000, 400000, 5500000)
        shape = (100, 100)
        yield_tif = str(tmp_path / "y.tif")
        _create_test_raster(yield_tif, bounds, shape, crs, np.full(shape, 3000.0, dtype=np.float32))
        cov_tif = str(tmp_path / "c.tif")
        _create_test_coverage(cov_tif, bounds, shape, crs, coverage_pct=1.0)

        mid_x = (bounds[0] + bounds[2]) / 2
        poly1 = box(bounds[0], bounds[1], mid_x, bounds[3])
        poly2 = box(mid_x, bounds[1], bounds[2], bounds[3])
        shp_path = str(tmp_path / "regions_null.shp")
        _create_test_shapefile(shp_path, {"geometry": [poly1, poly2], "FID": [10.0, None]}, crs=crs)

        with pytest.raises(ValueError, match="non-null"):
            compute_zonal_yield_stats(yield_tif, cov_tif, shp_path, name_column="FID")

    def test_string_numeric_reference_with_null_first(self, tmp_path):
        """Regression: a reference column stored as strings with a null first value is
        read as object dtype. It must still be coerced to numeric (not carried through
        as strings) and the null row dropped.
        """
        import json

        from vercye_ops.reporting.zonal_aggregation import extract_reference_from_shapefile

        geoms = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)]
        # First value null, the rest numeric-but-stored-as-strings -> object dtype on read.
        vals = [None, "3200", "2800"]
        features = [
            {"type": "Feature", "properties": {"name": n, "ref": v}, "geometry": mapping(g)}
            for n, v, g in zip(["A", "B", "C"], vals, geoms)
        ]
        gj = str(tmp_path / "level.geojson")
        with open(gj, "w") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f)

        result = extract_reference_from_shapefile(gj, "name", "ref")

        # Null row dropped, remaining values coerced to numeric.
        assert len(result) == 2
        assert pd.api.types.is_numeric_dtype(result["reported_mean_yield_kg_ha"])
        assert sorted(result["reported_mean_yield_kg_ha"].tolist()) == [2800.0, 3200.0]
