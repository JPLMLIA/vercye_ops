"""Tests for mosaic_and_reproject module."""

import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds


def _create_test_tif(path, bounds, shape=(10, 10), crs="EPSG:4326", data=None):
    """Create a test GeoTIFF with given bounds and optional data."""
    left, bottom, right, top = bounds
    transform = from_bounds(left, bottom, right, top, shape[1], shape[0])

    if data is None:
        data = np.random.uniform(1000, 5000, shape).astype(np.float32)
        # Add some NaN pixels (simulating non-cropland)
        data[0:2, 0:2] = np.nan

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": shape[1],
        "height": shape[0],
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": float("nan"),
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


@pytest.fixture
def two_region_tifs(tmp_path):
    """Create two adjacent region yield TIFs."""
    region1_dir = tmp_path / "region1"
    region1_dir.mkdir()
    region2_dir = tmp_path / "region2"
    region2_dir.mkdir()

    tif1 = str(region1_dir / "region1_yield_map.tif")
    tif2 = str(region2_dir / "region2_yield_map.tif")

    # Two adjacent regions in Ukraine-ish area
    _create_test_tif(tif1, bounds=(34.0, 49.0, 35.0, 50.0))
    _create_test_tif(tif2, bounds=(35.0, 49.0, 36.0, 50.0))

    return [tif1, tif2], tmp_path


class TestCreateYieldMosaic:
    def test_produces_three_outputs(self, two_region_tifs):
        from vercye_ops.reporting.mosaic_and_reproject import create_yield_mosaic

        tif_paths, tmp_path = two_region_tifs

        mosaic_4326 = str(tmp_path / "mosaic_4326.tif")
        mosaic_proj = str(tmp_path / "mosaic_proj.tif")
        coverage = str(tmp_path / "coverage.tif")

        create_yield_mosaic(
            region_yield_tif_paths=tif_paths,
            output_mosaic_4326_path=mosaic_4326,
            output_mosaic_projected_path=mosaic_proj,
            output_coverage_mask_projected_path=coverage,
            target_crs="EPSG:9854",
        )

        assert os.path.exists(mosaic_4326)
        assert os.path.exists(mosaic_proj)
        assert os.path.exists(coverage)

    def test_mosaic_4326_has_correct_crs(self, two_region_tifs):
        from vercye_ops.reporting.mosaic_and_reproject import create_yield_mosaic

        tif_paths, tmp_path = two_region_tifs

        mosaic_4326 = str(tmp_path / "mosaic_4326.tif")
        mosaic_proj = str(tmp_path / "mosaic_proj.tif")
        coverage = str(tmp_path / "coverage.tif")

        create_yield_mosaic(tif_paths, mosaic_4326, mosaic_proj, coverage, "EPSG:9854")

        with rasterio.open(mosaic_4326) as src:
            assert src.crs.to_epsg() == 4326

    def test_projected_mosaic_has_target_crs(self, two_region_tifs):
        from vercye_ops.reporting.mosaic_and_reproject import create_yield_mosaic

        tif_paths, tmp_path = two_region_tifs

        mosaic_4326 = str(tmp_path / "mosaic_4326.tif")
        mosaic_proj = str(tmp_path / "mosaic_proj.tif")
        coverage = str(tmp_path / "coverage.tif")

        create_yield_mosaic(tif_paths, mosaic_4326, mosaic_proj, coverage, "EPSG:9854")

        with rasterio.open(mosaic_proj) as src:
            assert src.crs.to_epsg() == 9854

    def test_mosaic_spans_both_regions(self, two_region_tifs):
        from vercye_ops.reporting.mosaic_and_reproject import create_yield_mosaic

        tif_paths, tmp_path = two_region_tifs

        mosaic_4326 = str(tmp_path / "mosaic_4326.tif")
        mosaic_proj = str(tmp_path / "mosaic_proj.tif")
        coverage = str(tmp_path / "coverage.tif")

        create_yield_mosaic(tif_paths, mosaic_4326, mosaic_proj, coverage, "EPSG:9854")

        with rasterio.open(mosaic_4326) as src:
            # Should span from 34 to 36 degrees longitude
            assert src.bounds.left < 34.5
            assert src.bounds.right > 35.5

    def test_coverage_mask_is_binary(self, two_region_tifs):
        from vercye_ops.reporting.mosaic_and_reproject import create_yield_mosaic

        tif_paths, tmp_path = two_region_tifs

        mosaic_4326 = str(tmp_path / "mosaic_4326.tif")
        mosaic_proj = str(tmp_path / "mosaic_proj.tif")
        coverage = str(tmp_path / "coverage.tif")

        create_yield_mosaic(tif_paths, mosaic_4326, mosaic_proj, coverage, "EPSG:9854")

        with rasterio.open(coverage) as src:
            data = src.read(1)
            unique_vals = set(np.unique(data))
            # Should only contain 0, 1, or 255 (nodata)
            assert unique_vals.issubset({0, 1, 255})

    def test_nodata_preserved(self, two_region_tifs):
        from vercye_ops.reporting.mosaic_and_reproject import create_yield_mosaic

        tif_paths, tmp_path = two_region_tifs

        mosaic_4326 = str(tmp_path / "mosaic_4326.tif")
        mosaic_proj = str(tmp_path / "mosaic_proj.tif")
        coverage = str(tmp_path / "coverage.tif")

        create_yield_mosaic(tif_paths, mosaic_4326, mosaic_proj, coverage, "EPSG:9854")

        with rasterio.open(mosaic_4326) as src:
            data = src.read(1)
            # Should have some NaN values (from the test data)
            assert np.isnan(data).any()

    def test_empty_input_raises(self):
        from vercye_ops.reporting.mosaic_and_reproject import create_yield_mosaic

        with pytest.raises(ValueError, match="No input TIF files"):
            create_yield_mosaic([], "a.tif", "b.tif", "c.tif", "EPSG:9854")
