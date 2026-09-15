"""Tests for vercye_ops.matching_sim_real.generate_converted_lai_map.

The conversion is MEAN-ANCHORED: the APSIM matched regional yield is distributed
across cropland pixels in proportion to each pixel's peak LAI, so that

    mean_cropland(yield_px) == apsim_mean_yield_estimate_kg_ha

by construction, and LAI only sets the within-region spatial pattern. That
identity is the defining property and is what most of these tests assert.

(The previous ratio method divided by the max of the smoothed regional-median LAI
while multiplying the per-pixel max of the RAW LAI - inconsistent statistics that
over-predicted in proportion to LAI noise. It has been removed.)
"""

import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine

from vercye_ops.matching_sim_real.generate_converted_lai_map import process_geotiff

APSIM_COL = "apsim_mean_yield_estimate_kg_ha"


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


def _create_csv(path, apsim_mean_yield, extra=None):
    row = {APSIM_COL: [apsim_mean_yield]}
    if extra:
        row.update({k: [v] for k, v in extra.items()})
    pd.DataFrame(row).to_csv(str(path), index=False)


def _run(tmp_path, data, apsim_yield, nodata=None, extra=None):
    tif_in, csv_path, tif_out = tmp_path / "in.tif", tmp_path / "f.csv", tmp_path / "out.tif"
    _create_tif(tif_in, data, nodata=nodata)
    _create_csv(csv_path, apsim_yield, extra=extra)
    process_geotiff(str(tif_in), str(csv_path), str(tif_out))
    # read everything while the dataset is open; returning `src` itself would hand
    # back a closed handle and any later attribute access fails inside GDAL
    with rasterio.open(str(tif_out)) as src:
        return src.read(1), {"compression": src.compression, "crs": src.crs,
                             "transform": src.transform, "width": src.width, "height": src.height}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMeanAnchoring:
    def test_cropland_mean_equals_apsim_yield(self, tmp_path):
        """The defining property: cropland-mean output == APSIM matched yield."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result, _ = _run(tmp_path, data, 3000.0)
        assert result[result > 0].mean() == pytest.approx(3000.0, rel=1e-6)

    def test_holds_for_skewed_lai(self, tmp_path):
        """Anchoring must not depend on the LAI distribution's shape."""
        data = np.array([[0.1, 0.2, 9.0], [0.15, 0.3, 8.0]], dtype=np.float32)
        result, _ = _run(tmp_path, data, 1234.5)
        assert result[np.isfinite(result) & (result > 0)].mean() == pytest.approx(1234.5, rel=1e-6)

    def test_spatial_pattern_is_preserved(self, tmp_path):
        """Output must stay proportional to input: ratios between pixels unchanged."""
        data = np.array([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32)
        result, _ = _run(tmp_path, data, 2000.0)
        np.testing.assert_allclose(result / result[0, 0], data / data[0, 0], rtol=1e-5)

    def test_nan_nodata_preserved_and_excluded(self, tmp_path):
        """NaN pixels stay NaN and must not enter the anchor denominator."""
        data = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
        result, _ = _run(tmp_path, data, 800.0, nodata=np.nan)
        assert np.isnan(result[0, 1])
        valid = result[np.isfinite(result) & (result > 0)]
        assert valid.mean() == pytest.approx(800.0, rel=1e-6)

    def test_negative_values_clipped_before_anchoring(self, tmp_path):
        data = np.array([[-1.0, 2.0], [3.0, -0.5]], dtype=np.float32)
        result, _ = _run(tmp_path, data, 500.0)
        assert result[0, 0] == 0.0 and result[1, 1] == 0.0
        assert result[result > 0].mean() == pytest.approx(500.0, rel=1e-6)


class TestContract:
    def test_missing_apsim_column_raises(self, tmp_path):
        tif_in, csv_path, tif_out = tmp_path / "in.tif", tmp_path / "f.csv", tmp_path / "out.tif"
        _create_tif(tif_in, np.array([[1.0]], dtype=np.float32))
        pd.DataFrame({"conversion_factor": [100]}).to_csv(str(csv_path), index=False)
        with pytest.raises(KeyError, match=APSIM_COL):
            process_geotiff(str(tif_in), str(csv_path), str(tif_out))

    def test_legacy_conversion_factor_column_is_ignored(self, tmp_path):
        """The CSV still carries the old ratio factor; it must not be used."""
        data = np.array([[1.0, 3.0]], dtype=np.float32)
        result, _ = _run(tmp_path, data, 1000.0, extra={"conversion_factor": 999999.0})
        assert result[result > 0].mean() == pytest.approx(1000.0, rel=1e-6)

    def test_no_valid_pixels_yields_zeros_not_crash(self, tmp_path):
        """An all-nodata region must not raise ZeroDivisionError."""
        data = np.array([[np.nan, np.nan]], dtype=np.float32)
        result, _ = _run(tmp_path, data, 1500.0, nodata=np.nan)
        assert not np.any(np.isfinite(result) & (result > 0))

    def test_all_zero_lai_yields_zeros(self, tmp_path):
        data = np.zeros((2, 2), dtype=np.float32)
        result, _ = _run(tmp_path, data, 1500.0)
        np.testing.assert_allclose(result, 0.0)

    def test_zero_apsim_yield_gives_zero_map(self, tmp_path):
        data = np.array([[5.0, 10.0]], dtype=np.float32)
        result, _ = _run(tmp_path, data, 0.0)
        np.testing.assert_allclose(result, 0.0)


class TestRasterOutput:
    def test_output_is_lzw_compressed(self, tmp_path):
        _, meta = _run(tmp_path, np.array([[1.0, 2.0]], dtype=np.float32), 1000.0)
        assert meta["compression"] == rasterio.enums.Compression.lzw

    def test_preserves_spatial_metadata(self, tmp_path):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tif_in, csv_path, tif_out = tmp_path / "in.tif", tmp_path / "f.csv", tmp_path / "out.tif"
        _create_tif(tif_in, data)
        _create_csv(csv_path, 1000.0)
        process_geotiff(str(tif_in), str(csv_path), str(tif_out))
        with rasterio.open(str(tif_in)) as a, rasterio.open(str(tif_out)) as b:
            assert a.crs == b.crs
            assert a.transform == b.transform
            assert (a.width, a.height) == (b.width, b.height)

    def test_rejects_multiband_input(self, tmp_path):
        transform = Affine(0.01, 0, 30.0, 0, -0.01, 50.0)
        tif_in, csv_path, tif_out = tmp_path / "in.tif", tmp_path / "f.csv", tmp_path / "out.tif"
        with rasterio.open(str(tif_in), "w", driver="GTiff", height=2, width=2, count=2,
                           dtype="float32", crs="EPSG:4326", transform=transform) as dst:
            dst.write(np.ones((2, 2), dtype="float32"), 1)
            dst.write(np.ones((2, 2), dtype="float32"), 2)
        _create_csv(csv_path, 1000.0)
        with pytest.raises(ValueError, match="single band"):
            process_geotiff(str(tif_in), str(csv_path), str(tif_out))
