from datetime import datetime

import numpy as np
import pytest
import rasterio as rio
from affine import Affine
from rasterio.transform import from_origin

from vercye_ops.lai.lai_analysis import (
    apply_akima_interpolation,
    apply_savgol_smoothing,
    assert_grids_aligned,
    build_date_range,
    build_lai_window_for_cropmask,
    clip_negative_lai,
    compute_basic_stats,
    compute_cloud_snow_percentage,
    compute_lai_adjusted,
    ensure_output_dirs,
    ensure_raster_alignment,
    make_empty_stat_row,
    mask_lai_with_binary_cropmask,
    process_geometry,
    process_single_date,
    raster_and_mask_intersect,
    update_max_lai,
    validate_maxlai_keep_bands,
    write_max_lai_raster,
    write_statistics_csv,
)

# WKT for WGS84 - avoids EPSG lookups that fail when the PROJ database is mismatched
WGS84_WKT = (
    'GEOGCS["WGS 84",'
    'DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],'
    'UNIT["degree",0.0174532925199433]]'
)
WGS84_CRS = rio.CRS.from_wkt(WGS84_WKT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_lai_raster(path, data, transform, crs=None):
    """Write a single-band float32 GeoTIFF. Returns the CRS as roundtripped by rasterio/GDAL."""
    if crs is None:
        crs = WGS84_CRS
    h, w = data.shape
    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(data, 1)
    # Read back the CRS that GDAL actually wrote (may differ in axis order / authority)
    with rio.open(path) as src:
        return src.crs


def _make_geometry_dict(array, transform, bounds=None, crs=None):
    """Build the raster-mode geometry dict expected by the analysis functions."""
    h, w = array.shape
    if bounds is None:
        bounds = rio.transform.array_bounds(h, w, transform)
    return {
        "array": array,
        "res": (abs(transform.a), abs(transform.e)),
        "bounds": bounds,
        "transform": transform,
        "crs": crs if crs is not None else WGS84_CRS,
    }


# ===================================================================
# Pure-array / unit tests (no raster I/O)
# ===================================================================


def test_clip_negative_lai():
    arr = np.array([[1.0, -0.5], [0.0, 2.0]], dtype=float)
    out = clip_negative_lai(arr)
    assert np.isnan(out[0, 1])
    assert out[0, 0] == 1.0
    assert out[1, 0] == 0.0
    assert out[1, 1] == 2.0


def test_compute_lai_adjusted_wheat_maize_none():
    x = np.array([0.0, 1.0, 2.0], dtype=float)

    wheat = compute_lai_adjusted(x, "wheat")
    maize = compute_lai_adjusted(x, "maize")
    none = compute_lai_adjusted(x, "none")
    none_none = compute_lai_adjusted(x, None)

    assert np.allclose(none, x)
    assert np.allclose(none_none, x)
    assert wheat.shape == x.shape
    assert maize.shape == x.shape


def test_compute_lai_adjusted_invalid_raises():
    x = np.array([1.0], dtype=float)
    with pytest.raises(ValueError):
        compute_lai_adjusted(x, "soybean")


def test_compute_basic_stats_ignores_nans():
    arr = np.array([[1.0, np.nan], [3.0, 5.0]], dtype=float)
    mean, median, stddev, n_pixels = compute_basic_stats(arr)

    assert n_pixels == 3
    assert mean == pytest.approx((1.0 + 3.0 + 5.0) / 3.0)
    assert median == pytest.approx(3.0)
    assert stddev > 0.0


def test_update_max_lai_basic():
    a = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=float)
    b = np.array([[0.5, 3.0], [5.0, np.nan]], dtype=float)

    out1 = update_max_lai(None, a)
    assert np.all(np.isnan(out1) == np.isnan(a))
    assert np.allclose(out1[~np.isnan(out1)], a[~np.isnan(a)])

    out2 = update_max_lai(a, b)
    assert out2.shape == a.shape
    assert out2[0, 0] == 1.0
    assert out2[0, 1] == 3.0
    assert out2[1, 0] == 5.0
    assert out2[1, 1] == 4.0 or np.isnan(out2[1, 1])


def test_raster_and_mask_intersect_true_and_false():
    # overlapping boxes
    lai_bounds = (0.0, 0.0, 10.0, 10.0)
    cm_bounds = (5.0, 5.0, 15.0, 15.0)
    assert raster_and_mask_intersect(lai_bounds, cm_bounds)

    # disjoint boxes
    cm_bounds2 = (20.0, 20.0, 30.0, 30.0)
    assert not raster_and_mask_intersect(lai_bounds, cm_bounds2)


def test_compute_cloud_snow_percentage():
    lai = np.array(
        [
            [np.nan, 2.0, np.nan],
            [1.0, np.nan, 3.0],
        ],
        dtype=float,
    )
    cropmask = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=int,
    )
    # masked crop pixels: positions with cropmask==1
    # NaNs at (0,0) and (1,1) -> 2 NaNs out of 4 crop pixels => 50%
    pct = compute_cloud_snow_percentage(lai, cropmask)
    assert pct == pytest.approx(50.0)


def test_mask_lai_with_binary_cropmask():
    lai = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    cm = np.array([[1, 0], [1, 1]], dtype=int)

    masked = mask_lai_with_binary_cropmask(lai, cm)
    assert masked.shape == lai.shape
    assert np.isnan(masked[0, 1])
    assert masked[0, 0] == 1.0
    assert masked[1, 0] == 3.0
    assert masked[1, 1] == 4.0


def test_mask_lai_with_binary_cropmask_shape_mismatch_raises():
    lai = np.ones((2, 2), dtype=float)
    cm = np.ones((3, 3), dtype=int)
    with pytest.raises(ValueError):
        mask_lai_with_binary_cropmask(lai, cm)


def test_build_date_range_simple():
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 3)
    iso, slash = build_date_range(start, end)

    assert iso == ["2020-01-01", "2020-01-02", "2020-01-03"]
    assert slash == ["01/01/2020", "02/01/2020", "03/01/2020"]


def test_make_empty_stat_row_defaults():
    row = make_empty_stat_row("01/01/2020")
    assert row["Date"] == "01/01/2020"
    assert row["n_pixels"] == 0
    assert row["interpolated"] == 0
    assert row["LAI Mean"] is None
    assert row["Cloud or Snow Percentage"] is None


def test_validate_maxlai_keep_bands_ok_and_empty():
    bands = validate_maxlai_keep_bands(["estimateLAImax", "adjustedLAImax"])
    assert bands == ["estimateLAImax", "adjustedLAImax"]

    with pytest.raises(ValueError):
        validate_maxlai_keep_bands([])


def test_apply_savgol_smoothing_adds_unsmoothed_and_nonnegative():
    # at least 4 valid points so current implementation's window_length logic works
    stats = []
    for i in range(6):
        stats.append(
            {
                "Date": f"{i+1:02d}/01/2020",
                "n_pixels": 10,
                "interpolated": 0,
                "LAI Mean": float(i + 1),
                "LAI Median": float(i + 1),
                "LAI Stddev": 0.1 * (i + 1),
                "LAI Mean Adjusted": float(i + 2),
                "LAI Median Adjusted": float(i + 2),
                "LAI Stddev Adjusted": 0.2 * (i + 1),
                "Cloud or Snow Percentage": 0.0,
            }
        )

    out = apply_savgol_smoothing(stats)
    for rec in out:
        assert rec["LAI Mean Unsmoothed"] is not None
        assert rec["LAI Mean"] >= 0.0
        assert rec["LAI Mean Adjusted"] >= 0.0


def test_apply_akima_interpolation_fills_nans_and_marks_interpolated():
    stats = [
        {
            "Date": "01/01/2020",
            "n_pixels": 10,
            "interpolated": 0,
            "LAI Mean": 1.0,
            "LAI Median": 1.0,
            "LAI Stddev": 0.1,
            "LAI Mean Adjusted": 2.0,
            "LAI Median Adjusted": 2.0,
            "LAI Stddev Adjusted": 0.2,
            "Cloud or Snow Percentage": 0.0,
        },
        {
            "Date": "02/01/2020",
            "n_pixels": 0,
            "interpolated": 0,
            "LAI Mean": None,
            "LAI Median": None,
            "LAI Stddev": None,
            "LAI Mean Adjusted": None,
            "LAI Median Adjusted": None,
            "LAI Stddev Adjusted": None,
            "Cloud or Snow Percentage": None,
        },
        {
            "Date": "03/01/2020",
            "n_pixels": 10,
            "interpolated": 0,
            "LAI Mean": 3.0,
            "LAI Median": 3.0,
            "LAI Stddev": 0.3,
            "LAI Mean Adjusted": 4.0,
            "LAI Median Adjusted": 4.0,
            "LAI Stddev Adjusted": 0.4,
            "Cloud or Snow Percentage": 0.0,
        },
    ]

    out = apply_akima_interpolation(stats, smoothed=False)
    assert out[1]["LAI Mean"] is not None
    assert out[1]["interpolated"] == 1


def test_assert_grids_aligned_ok_and_failure():
    st = Affine.translation(0, 0) * Affine.scale(10, -10)
    gt_ok = Affine.translation(20, -20) * Affine.scale(10, -10)
    gt_bad_pix = Affine.translation(25, -25) * Affine.scale(10, -10)

    class DummySrc:
        def __init__(self, transform):
            self.transform = transform

    src = DummySrc(st)

    # aligned (integer pixel shift)
    assert_grids_aligned(src, gt_ok, tol_pix=1e-6)

    with pytest.raises(SystemExit):
        assert_grids_aligned(src, gt_bad_pix, tol_pix=1e-6)


def test_ensure_raster_alignment_ok_and_crs_mismatch():
    class DummySrc:
        def __init__(self, transform, crs, res):
            self.transform = transform
            self.crs = crs
            self.res = res

    st = Affine.translation(0, 0) * Affine.scale(10, -10)
    src_ok = DummySrc(st, WGS84_CRS, (10.0, 10.0))
    geom_ok = {
        "crs": WGS84_CRS,
        "res": (10.0, 10.0),
        "transform": st,
    }

    ensure_raster_alignment(src_ok, geom_ok)

    geom_crs_bad = geom_ok.copy()
    geom_crs_bad["crs"] = "EPSG:3857"
    with pytest.raises(SystemExit):
        ensure_raster_alignment(src_ok, geom_crs_bad)


def test_ensure_output_dirs(tmp_path):
    stats_path = tmp_path / "out" / "stats.csv"
    max_path = tmp_path / "out" / "max.tif"

    ensure_output_dirs(stats_path, max_path)
    assert stats_path.parent.exists()
    assert max_path.parent.exists()


def test_write_statistics_csv(tmp_path):
    stats_path = tmp_path / "stats.csv"
    statistics = [
        make_empty_stat_row("01/01/2020"),
        make_empty_stat_row("02/01/2020"),
    ]

    write_statistics_csv(stats_path, statistics)

    assert stats_path.exists()
    content = stats_path.read_text()
    assert "Date" in content
    assert "01/01/2020" in content
    assert "02/01/2020" in content


def test_assert_grids_aligned_int_pixel_shift_ok_non_int_fails_and_rotation():
    st = Affine.translation(0, 0) * Affine.scale(10, -10)

    # exact 2-pixel shift in both directions -> ok
    gt_ok = Affine.translation(20, -20) * Affine.scale(10, -10)

    # 2.5 pixel shift -> misalignment
    gt_bad_pix = Affine.translation(25, -25) * Affine.scale(10, -10)

    # rotated grid (non-zero shear) -> not allowed
    gt_rot = Affine(10, 1, 0, 0, -10, 0)

    class DummySrc:
        def __init__(self, transform):
            self.transform = transform

    src = DummySrc(st)

    assert_grids_aligned(src, gt_ok, tol_pix=1e-6)

    with pytest.raises(SystemExit):
        assert_grids_aligned(src, gt_bad_pix, tol_pix=1e-6)

    with pytest.raises(SystemExit):
        assert_grids_aligned(src, gt_rot, tol_pix=1e-6)


def test_ensure_raster_alignment_ok_crs_mismatch_res_mismatch_and_nonint_shift():
    st = Affine.translation(0, 0) * Affine.scale(10, -10)

    class DummySrc:
        def __init__(self, transform, crs, res):
            self.transform = transform
            self.crs = crs
            self.res = res

    src_ok = DummySrc(st, WGS84_CRS, (10.0, 10.0))

    geom_ok = {
        "crs": WGS84_CRS,
        "res": (10.0, 10.0),
        "transform": st,
    }

    ensure_raster_alignment(src_ok, geom_ok)

    geom_crs_bad = dict(geom_ok)
    geom_crs_bad["crs"] = "EPSG:3857"
    with pytest.raises(SystemExit):
        ensure_raster_alignment(src_ok, geom_crs_bad)

    geom_res_bad = dict(geom_ok)
    geom_res_bad["res"] = (20.0, 10.0)
    with pytest.raises(SystemExit):
        ensure_raster_alignment(src_ok, geom_res_bad)

    # non-integer pixel shift in transform
    geom_shift_bad = dict(geom_ok)
    geom_shift_bad["transform"] = Affine.translation(5, -5) * Affine.scale(10, -10)
    with pytest.raises(SystemExit):
        ensure_raster_alignment(src_ok, geom_shift_bad)


# ===================================================================
# Raster I/O tests - build_lai_window_for_cropmask
# ===================================================================


def test_build_lai_window_for_cropmask(tmp_path):
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    transform = from_origin(0, 2, 1, 1)  # left, top, xres, yres
    raster_path = tmp_path / "lai.tif"
    _create_test_lai_raster(raster_path, data, transform)

    with rio.open(raster_path) as src:
        # cropmask bounds identical to raster bounds
        bounds = src.bounds
        cm_shape = (2, 2)

        window_arr = build_lai_window_for_cropmask(src, bounds, cm_shape)
        assert window_arr.shape == (2, 2)
        assert np.allclose(window_arr, data)


def test_build_lai_window_for_cropmask_cropmask_equal_to_raster(tmp_path):
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    transform = from_origin(0, 2, 1, 1)
    raster_path = tmp_path / "lai.tif"
    _create_test_lai_raster(raster_path, data, transform)

    with rio.open(raster_path) as src:
        bounds = src.bounds
        cm_shape = (2, 2)
        window_arr = build_lai_window_for_cropmask(src, bounds, cm_shape)
        assert window_arr.shape == (2, 2)
        assert np.allclose(window_arr, data)


def test_build_lai_window_for_cropmask_cropmask_larger_than_raster(tmp_path):
    """Cropmask extends 1 pixel beyond LAI on all sides → outer ring is NaN."""
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    transform = from_origin(0, 2, 1, 1)  # bounds: left=0, top=2, right=2, bottom=0
    raster_path = tmp_path / "lai_large_test.tif"
    _create_test_lai_raster(raster_path, data, transform)

    with rio.open(raster_path) as src:
        # cropmask extends one pixel beyond raster on all sides
        cropmask_bounds = (-1.0, -1.0, 3.0, 3.0)
        cm_shape = (4, 4)

        window_arr = build_lai_window_for_cropmask(src, cropmask_bounds, cm_shape)
        assert window_arr.shape == (4, 4)

        # central 2x2 should match the raster, outer ring should be NaN
        inner = window_arr[1:3, 1:3]
        assert np.allclose(inner, data)

        outer_mask = np.ones_like(window_arr, dtype=bool)
        outer_mask[1:3, 1:3] = False
        assert np.all(np.isnan(window_arr[outer_mask]))


def test_build_lai_window_for_cropmask_cropmask_smaller_than_raster(tmp_path):
    """Cropmask is a subset of the LAI raster → only the subset is returned."""
    data = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
        dtype=np.float32,
    )
    transform = from_origin(0, 4, 1, 1)  # 4x4 raster: left=0, top=4
    raster_path = tmp_path / "lai_small_cm.tif"
    _create_test_lai_raster(raster_path, data, transform)

    with rio.open(raster_path) as src:
        # cropmask covers only the central 2x2
        cropmask_bounds = (1.0, 1.0, 3.0, 3.0)
        cm_shape = (2, 2)

        window_arr = build_lai_window_for_cropmask(src, cropmask_bounds, cm_shape)
        assert window_arr.shape == (2, 2)
        # rows 1-2, cols 1-2 of the original data
        expected = data[1:3, 1:3]
        assert np.allclose(window_arr, expected)


def test_build_lai_window_for_cropmask_partial_overlap(tmp_path):
    """Cropmask partially overlaps LAI - overlapping part has data, rest is NaN."""
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    transform = from_origin(0, 2, 1, 1)  # bounds: left=0, top=2, right=2, bottom=0
    raster_path = tmp_path / "lai_partial.tif"
    _create_test_lai_raster(raster_path, data, transform)

    with rio.open(raster_path) as src:
        # cropmask shifted 1 pixel right and 1 pixel down relative to LAI
        # LAI: [0,2] x [0,2], cropmask: [1,3] x [-1,1]
        cropmask_bounds = (1.0, -1.0, 3.0, 1.0)
        cm_shape = (2, 2)

        window_arr = build_lai_window_for_cropmask(src, cropmask_bounds, cm_shape)
        assert window_arr.shape == (2, 2)

        # Only top-left pixel of this window overlaps the LAI raster:
        # cropmask pixel (0,0) maps to LAI row=1, col=1 → data[1,1] = 4.0
        assert window_arr[0, 0] == pytest.approx(4.0)
        # The other pixels are outside LAI bounds → NaN
        assert np.isnan(window_arr[0, 1])
        assert np.isnan(window_arr[1, 0])
        assert np.isnan(window_arr[1, 1])


def test_build_lai_window_for_cropmask_no_overlap(tmp_path):
    """Cropmask completely outside LAI → all NaN."""
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    transform = from_origin(0, 2, 1, 1)
    raster_path = tmp_path / "lai_nooverlap.tif"
    _create_test_lai_raster(raster_path, data, transform)

    with rio.open(raster_path) as src:
        # cropmask is far away from the LAI raster
        cropmask_bounds = (100.0, 100.0, 102.0, 102.0)
        cm_shape = (2, 2)

        window_arr = build_lai_window_for_cropmask(src, cropmask_bounds, cm_shape)
        assert window_arr.shape == (2, 2)
        assert np.all(np.isnan(window_arr))


# ===================================================================
# Masking polarity - verify 0=excluded, 1=kept (not inverted)
# ===================================================================


def test_mask_polarity_zero_excludes_one_keeps():
    """Cropmask 0 → NaN (excluded), 1 → LAI value (kept). Not inverted."""
    lai = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float)
    cm = np.array([[1, 0], [0, 1]], dtype=int)

    masked = mask_lai_with_binary_cropmask(lai, cm)

    # Kept pixels
    assert masked[0, 0] == pytest.approx(10.0)
    assert masked[1, 1] == pytest.approx(40.0)
    # Excluded pixels
    assert np.isnan(masked[0, 1])
    assert np.isnan(masked[1, 0])


def test_mask_all_zeros_gives_all_nan():
    """Cropmask of all zeros → entire LAI is NaN."""
    lai = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    cm = np.zeros((2, 2), dtype=int)

    masked = mask_lai_with_binary_cropmask(lai, cm)
    assert np.all(np.isnan(masked))


def test_mask_all_ones_preserves_lai():
    """Cropmask of all ones → LAI values unchanged."""
    lai = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    cm = np.ones((2, 2), dtype=int)

    masked = mask_lai_with_binary_cropmask(lai, cm)
    assert np.allclose(masked, lai)


def test_mask_preserves_nan_in_lai():
    """NaN in LAI stays NaN even where cropmask is 1."""
    lai = np.array([[np.nan, 2.0], [3.0, np.nan]], dtype=float)
    cm = np.ones((2, 2), dtype=int)

    masked = mask_lai_with_binary_cropmask(lai, cm)
    assert np.isnan(masked[0, 0])
    assert np.isnan(masked[1, 1])
    assert masked[0, 1] == pytest.approx(2.0)
    assert masked[1, 0] == pytest.approx(3.0)


# ===================================================================
# process_single_date - raster mode integration tests
# ===================================================================


def test_process_single_date_raster_mode_basic(tmp_path):
    """Basic raster mode: LAI and cropmask exactly aligned and same extent."""
    data = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
    transform = from_origin(0, 2, 1, 1)
    raster_path = tmp_path / "region_10m_2020-01-01_LAI.tif"
    rt_crs = _create_test_lai_raster(raster_path, data, transform)

    geometry = _make_geometry_dict(
        np.array([[1, 1], [1, 1]], dtype=np.uint8),
        transform,
        crs=rt_crs,
    )

    stat, lai_estimate, lai_adjusted, src_meta = process_single_date(
        lai_dir=str(tmp_path),
        region="region",
        resolution=10,
        lai_file_ext="tif",
        geometry=geometry,
        mode="raster",
        adjustment="none",
        date_iso="2020-01-01",
        date_slash="01/01/2020",
        cloudcov_threshold=None,
        src_meta=None,
    )

    assert stat["Date"] == "01/01/2020"
    assert stat["n_pixels"] == 3
    assert lai_estimate is not None
    assert lai_adjusted is not None
    assert src_meta is not None
    assert src_meta["height"] == 2
    assert src_meta["width"] == 2


def test_process_single_date_missing_file_returns_empty(tmp_path):
    geometry = _make_geometry_dict(
        np.ones((2, 2), dtype=np.uint8),
        from_origin(0, 2, 1, 1),
    )
    # No raster written - CRS comparison not reached for missing files

    stat, lai_estimate, lai_adjusted, src_meta = process_single_date(
        lai_dir=str(tmp_path),
        region="region",
        resolution=10,
        lai_file_ext="tif",
        geometry=geometry,
        mode="raster",
        adjustment="none",
        date_iso="2020-01-02",
        date_slash="02/01/2020",
        cloudcov_threshold=None,
        src_meta=None,
    )

    assert stat["Date"] == "02/01/2020"
    assert stat["n_pixels"] == 0
    assert lai_estimate is None
    assert lai_adjusted is None
    assert src_meta is None


def test_process_single_date_cropmask_partially_masks(tmp_path):
    """Cropmask zeros out some pixels - only cropmask=1 pixels contribute to stats."""
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    transform = from_origin(0, 2, 1, 1)
    raster_path = tmp_path / "region_10m_2020-01-01_LAI.tif"
    rt_crs = _create_test_lai_raster(raster_path, data, transform)

    # Only top-left and bottom-right are cropland
    cm = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    geometry = _make_geometry_dict(cm, transform, crs=rt_crs)

    stat, lai_estimate, lai_adjusted, src_meta = process_single_date(
        lai_dir=str(tmp_path),
        region="region",
        resolution=10,
        lai_file_ext="tif",
        geometry=geometry,
        mode="raster",
        adjustment="none",
        date_iso="2020-01-01",
        date_slash="01/01/2020",
        cloudcov_threshold=None,
        src_meta=None,
    )

    assert stat["n_pixels"] == 2
    assert stat["LAI Mean"] == pytest.approx((1.0 + 4.0) / 2.0)


def test_process_single_date_cropmask_larger_than_lai(tmp_path):
    """Cropmask extends beyond LAI - only the overlapping crop pixels contribute."""
    # LAI covers [0, 2] x [0, 2]
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    transform = from_origin(0, 2, 1, 1)
    raster_path = tmp_path / "region_10m_2020-01-01_LAI.tif"
    rt_crs = _create_test_lai_raster(raster_path, data, transform)

    # Cropmask covers [0, 3] x [-1, 2] → 3x3 grid, extends beyond LAI
    cm_transform = from_origin(0, 3, 1, 1)
    cm_array = np.ones((4, 3), dtype=np.uint8)
    geometry = _make_geometry_dict(cm_array, cm_transform, crs=rt_crs)

    stat, lai_estimate, lai_adjusted, src_meta = process_single_date(
        lai_dir=str(tmp_path),
        region="region",
        resolution=10,
        lai_file_ext="tif",
        geometry=geometry,
        mode="raster",
        adjustment="none",
        date_iso="2020-01-01",
        date_slash="01/01/2020",
        cloudcov_threshold=None,
        src_meta=None,
    )

    # Only 4 pixels have actual LAI data, rest are NaN from boundless read
    assert stat["n_pixels"] == 4
    assert stat["LAI Mean"] == pytest.approx((1.0 + 2.0 + 3.0 + 4.0) / 4.0)


def test_process_single_date_lai_larger_than_cropmask(tmp_path):
    """LAI covers a bigger area than cropmask - only cropmask extent is analyzed."""
    # LAI covers [0, 4] x [0, 4]
    data = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    transform = from_origin(0, 4, 1, 1)
    raster_path = tmp_path / "region_10m_2020-01-01_LAI.tif"
    rt_crs = _create_test_lai_raster(raster_path, data, transform)

    # Cropmask covers only [1, 3] x [1, 3] - center 2x2
    cm_transform = from_origin(1, 3, 1, 1)
    cm_array = np.ones((2, 2), dtype=np.uint8)
    geometry = _make_geometry_dict(cm_array, cm_transform, crs=rt_crs)

    stat, lai_estimate, lai_adjusted, src_meta = process_single_date(
        lai_dir=str(tmp_path),
        region="region",
        resolution=10,
        lai_file_ext="tif",
        geometry=geometry,
        mode="raster",
        adjustment="none",
        date_iso="2020-01-01",
        date_slash="01/01/2020",
        cloudcov_threshold=None,
        src_meta=None,
    )

    # Center 2x2 of the 4x4 grid: rows 1-2, cols 1-2 → values 6,7,10,11
    expected_mean = (6.0 + 7.0 + 10.0 + 11.0) / 4.0
    assert stat["n_pixels"] == 4
    assert stat["LAI Mean"] == pytest.approx(expected_mean)


def test_process_single_date_partial_overlap(tmp_path):
    """LAI and cropmask partially overlap - only intersection contributes."""
    # LAI covers [0, 2] x [0, 2]
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    transform = from_origin(0, 2, 1, 1)
    raster_path = tmp_path / "region_10m_2020-01-01_LAI.tif"
    rt_crs = _create_test_lai_raster(raster_path, data, transform)

    # Cropmask covers [1, 3] x [-1, 1] - only bottom-right pixel of LAI overlaps
    cm_transform = from_origin(1, 1, 1, 1)
    cm_array = np.ones((2, 2), dtype=np.uint8)
    geometry = _make_geometry_dict(cm_array, cm_transform, crs=rt_crs)

    stat, lai_estimate, lai_adjusted, src_meta = process_single_date(
        lai_dir=str(tmp_path),
        region="region",
        resolution=10,
        lai_file_ext="tif",
        geometry=geometry,
        mode="raster",
        adjustment="none",
        date_iso="2020-01-01",
        date_slash="01/01/2020",
        cloudcov_threshold=None,
        src_meta=None,
    )

    # Only 1 pixel of the 2x2 cropmask overlaps LAI: position [0,0] of the window
    # which corresponds to LAI data[1, 1] = 4.0
    assert stat["n_pixels"] == 1
    assert stat["LAI Mean"] == pytest.approx(4.0)


def test_process_single_date_no_overlap_returns_empty(tmp_path):
    """LAI and cropmask don't overlap at all - returns empty stats."""
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    transform = from_origin(0, 2, 1, 1)
    raster_path = tmp_path / "region_10m_2020-01-01_LAI.tif"
    rt_crs = _create_test_lai_raster(raster_path, data, transform)

    # Cropmask far from LAI
    cm_transform = from_origin(100, 102, 1, 1)
    cm_array = np.ones((2, 2), dtype=np.uint8)
    geometry = _make_geometry_dict(cm_array, cm_transform, crs=rt_crs)

    stat, lai_estimate, lai_adjusted, src_meta = process_single_date(
        lai_dir=str(tmp_path),
        region="region",
        resolution=10,
        lai_file_ext="tif",
        geometry=geometry,
        mode="raster",
        adjustment="none",
        date_iso="2020-01-01",
        date_slash="01/01/2020",
        cloudcov_threshold=None,
        src_meta=None,
    )

    assert stat["n_pixels"] == 0
    assert lai_estimate is None


def test_process_single_date_all_lai_nan_in_cropmask(tmp_path):
    """LAI is all NaN within cropmask area - should return empty-like stats."""
    data = np.full((2, 2), np.nan, dtype=np.float32)
    transform = from_origin(0, 2, 1, 1)
    raster_path = tmp_path / "region_10m_2020-01-01_LAI.tif"
    rt_crs = _create_test_lai_raster(raster_path, data, transform)

    cm_array = np.ones((2, 2), dtype=np.uint8)
    geometry = _make_geometry_dict(cm_array, transform, crs=rt_crs)

    stat, lai_estimate, lai_adjusted, src_meta = process_single_date(
        lai_dir=str(tmp_path),
        region="region",
        resolution=10,
        lai_file_ext="tif",
        geometry=geometry,
        mode="raster",
        adjustment="none",
        date_iso="2020-01-01",
        date_slash="01/01/2020",
        cloudcov_threshold=None,
        src_meta=None,
    )

    # All NaN → should be flagged as no data
    assert stat["n_pixels"] == 0
    assert lai_estimate is None


def test_process_single_date_cloud_threshold_rejects_cloudy(tmp_path):
    """Cloud coverage above threshold → date is skipped."""
    # 2 out of 4 pixels are NaN (50% cloudy)
    data = np.array([[np.nan, 2.0], [np.nan, 4.0]], dtype=np.float32)
    transform = from_origin(0, 2, 1, 1)
    raster_path = tmp_path / "region_10m_2020-01-01_LAI.tif"
    rt_crs = _create_test_lai_raster(raster_path, data, transform)

    cm_array = np.ones((2, 2), dtype=np.uint8)
    geometry = _make_geometry_dict(cm_array, transform, crs=rt_crs)

    # Threshold of 40% - 50% cloud exceeds it
    stat, lai_estimate, lai_adjusted, src_meta = process_single_date(
        lai_dir=str(tmp_path),
        region="region",
        resolution=10,
        lai_file_ext="tif",
        geometry=geometry,
        mode="raster",
        adjustment="none",
        date_iso="2020-01-01",
        date_slash="01/01/2020",
        cloudcov_threshold=0.4,
        src_meta=None,
    )

    assert stat["n_pixels"] == 0
    assert lai_estimate is None
    assert stat["Cloud or Snow Percentage"] == pytest.approx(50.0)


def test_process_single_date_cloud_threshold_accepts_clear(tmp_path):
    """Cloud coverage below threshold → date is processed normally."""
    # 1 out of 4 pixels is NaN (25% cloudy)
    data = np.array([[np.nan, 2.0], [3.0, 4.0]], dtype=np.float32)
    transform = from_origin(0, 2, 1, 1)
    raster_path = tmp_path / "region_10m_2020-01-01_LAI.tif"
    rt_crs = _create_test_lai_raster(raster_path, data, transform)

    cm_array = np.ones((2, 2), dtype=np.uint8)
    geometry = _make_geometry_dict(cm_array, transform, crs=rt_crs)

    # Threshold of 40% - 25% cloud is under
    stat, lai_estimate, lai_adjusted, src_meta = process_single_date(
        lai_dir=str(tmp_path),
        region="region",
        resolution=10,
        lai_file_ext="tif",
        geometry=geometry,
        mode="raster",
        adjustment="none",
        date_iso="2020-01-01",
        date_slash="01/01/2020",
        cloudcov_threshold=0.4,
        src_meta=None,
    )

    assert stat["n_pixels"] == 3
    assert lai_estimate is not None


# ===================================================================
# process_geometry - multi-date integration
# ===================================================================


def test_process_geometry_accumulates_max(tmp_path):
    # one date raster
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    transform = from_origin(0, 2, 1, 1)
    raster_path = tmp_path / "region_10m_2020-01-01_LAI.tif"
    rt_crs = _create_test_lai_raster(raster_path, data, transform)

    geometry = _make_geometry_dict(
        np.ones((2, 2), dtype=np.uint8),
        transform,
        crs=rt_crs,
    )

    stats, lai_max, lai_adj_max, src_meta = process_geometry(
        lai_dir=str(tmp_path),
        region="region",
        resolution=10,
        lai_file_ext="tif",
        geometry=geometry,
        mode="raster",
        adjustment="none",
        dates_iso=["2020-01-01"],
        dates_slash=["01/01/2020"],
        cloudcov_threshold=None,
    )

    assert len(stats) == 1
    assert lai_max is not None
    assert lai_adj_max is not None
    assert src_meta is not None
    assert np.allclose(lai_max[~np.isnan(lai_max)], data[~np.isnan(data)])


def test_process_geometry_two_dates_max_is_elementwise(tmp_path):
    """Max LAI raster across two dates is the elementwise maximum."""
    transform = from_origin(0, 2, 1, 1)

    data1 = np.array([[5.0, 1.0], [3.0, 2.0]], dtype=np.float32)
    data2 = np.array([[1.0, 6.0], [4.0, 1.0]], dtype=np.float32)

    rt_crs = _create_test_lai_raster(tmp_path / "region_10m_2020-01-01_LAI.tif", data1, transform)
    _create_test_lai_raster(tmp_path / "region_10m_2020-01-02_LAI.tif", data2, transform)

    geometry = _make_geometry_dict(
        np.ones((2, 2), dtype=np.uint8),
        transform,
        crs=rt_crs,
    )

    stats, lai_max, lai_adj_max, src_meta = process_geometry(
        lai_dir=str(tmp_path),
        region="region",
        resolution=10,
        lai_file_ext="tif",
        geometry=geometry,
        mode="raster",
        adjustment="none",
        dates_iso=["2020-01-01", "2020-01-02"],
        dates_slash=["01/01/2020", "02/01/2020"],
        cloudcov_threshold=None,
    )

    expected_max = np.array([[5.0, 6.0], [4.0, 2.0]], dtype=np.float32)
    assert len(stats) == 2
    assert np.allclose(lai_max, expected_max)


# ===================================================================
# write_max_lai_raster
# ===================================================================


def test_write_max_lai_raster(tmp_path):
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    transform = from_origin(0, 2, 1, 1)
    out_path = tmp_path / "max_lai.tif"

    src_meta = {
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "float32",
        "crs": WGS84_CRS,
        "transform": transform,
    }

    write_max_lai_raster(
        output_max_tif_fpath=out_path,
        src_meta=src_meta,
        maxlai_keep_bands=["estimateLAImax", "adjustedLAImax"],
        lai_max=data,
        lai_adjusted_max=data + 1.0,
    )

    assert out_path.exists()
    with rio.open(out_path) as src:
        assert src.count == 2
        band1 = src.read(1)
        band2 = src.read(2)
        assert np.allclose(band1, data)
        assert np.allclose(band2, data + 1.0)


# ===================================================================
# Pixel-precise alignment verification
# ===================================================================


def test_pixel_values_not_shifted_after_masking(tmp_path):
    """
    End-to-end check: create a LAI raster with a known gradient pattern and a
    cropmask that selects specific pixels. Verify the aggregated mean corresponds
    exactly to the expected pixel values - catching any pixel shift.
    """
    # 5x5 LAI raster with unique values per pixel
    data = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
    transform = from_origin(10, 15, 1, 1)  # left=10, top=15
    raster_path = tmp_path / "region_10m_2020-06-15_LAI.tif"
    rt_crs = _create_test_lai_raster(raster_path, data, transform)

    # Cropmask: only the center 3x3 is cropland, offset by 1
    cm_data = np.zeros((5, 5), dtype=np.uint8)
    cm_data[1:4, 1:4] = 1
    geometry = _make_geometry_dict(cm_data, transform, crs=rt_crs)

    stat, lai_estimate, lai_adjusted, src_meta = process_single_date(
        lai_dir=str(tmp_path),
        region="region",
        resolution=10,
        lai_file_ext="tif",
        geometry=geometry,
        mode="raster",
        adjustment="none",
        date_iso="2020-06-15",
        date_slash="15/06/2020",
        cloudcov_threshold=None,
        src_meta=None,
    )

    # The center 3x3 values: rows 1-3, cols 1-3
    center_values = data[1:4, 1:4]
    expected_mean = float(np.mean(center_values))
    expected_median = float(np.median(center_values))

    assert stat["n_pixels"] == 9
    assert stat["LAI Mean"] == pytest.approx(expected_mean)
    assert stat["LAI Median"] == pytest.approx(expected_median)


def test_pixel_values_not_shifted_with_offset_cropmask(tmp_path):
    """
    Cropmask is offset by integer pixels from LAI - verify the correct
    LAI pixels are read at the cropmask location (no shift from misaligned origin).
    """
    # 4x4 LAI raster, each pixel has a unique value
    data = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    lai_transform = from_origin(0, 4, 1, 1)  # left=0, top=4
    raster_path = tmp_path / "region_10m_2020-01-01_LAI.tif"
    rt_crs = _create_test_lai_raster(raster_path, data, lai_transform)

    # Cropmask at offset origin (2, 2), covering 2x2 area
    cm_transform = from_origin(2, 2, 1, 1)  # left=2, top=2
    cm_array = np.ones((2, 2), dtype=np.uint8)
    geometry = _make_geometry_dict(cm_array, cm_transform, crs=rt_crs)

    stat, lai_estimate, lai_adjusted, src_meta = process_single_date(
        lai_dir=str(tmp_path),
        region="region",
        resolution=10,
        lai_file_ext="tif",
        geometry=geometry,
        mode="raster",
        adjustment="none",
        date_iso="2020-01-01",
        date_slash="01/01/2020",
        cloudcov_threshold=None,
        src_meta=None,
    )

    # Cropmask covers LAI rows 2-3, cols 2-3 (0-indexed)
    # data[2:4, 2:4] = [[11, 12], [15, 16]]
    expected_values = data[2:4, 2:4]
    expected_mean = float(np.mean(expected_values))

    assert stat["n_pixels"] == 4
    assert stat["LAI Mean"] == pytest.approx(expected_mean)
