import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

# TODO rename files to newer python naming convention with no leading digits
module_path = Path("vercye_ops/lai/3_analysis_LAI.py").resolve()
spec = importlib.util.spec_from_file_location("analysis_module", module_path)
module = importlib.util.module_from_spec(spec)
sys.modules["analysis_module"] = module
spec.loader.exec_module(module)
pad_to_raster = module.pad_to_raster


# Helper to create a mock in-memory raster
def create_raster(data, transform, crs="EPSG:4326"):
    memfile = MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        transform=transform,
        crs=crs,
    ) as dataset:
        dataset.write(data, 1)
    return memfile.open()


# Common transform: 1x1 pixel, top-left at (0, 10)
default_transform = from_origin(0, 10, 1, 1)

# Slightly shifted transform (simulate misalignment)
shifted_transform = from_origin(0.5, 10.5, 1, 1)


def test_padding_lai_fully_within_mask():
    """LAI raster is fully enclosed by the mask"""
    lai_data = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    mask_data = np.array([[1, 1], [1, 1]], dtype=np.uint8)

    # Bounds for padding logic
    lai_bounds = (0, 8, 2, 10)
    mask_bounds = (0, 8, 2, 10)

    padded, _ = pad_to_raster(lai_bounds, (1, 1), lai_data, mask_data, mask_bounds)
    masked = padded * mask_data

    assert np.isclose(np.nanmedian(masked), 0.25)


def test_padding_mask_fully_within_lai():
    """Mask is fully enclosed by the LAI raster"""
    lai_data = np.ones((5, 5), dtype=np.float32)
    mask_data = np.zeros((5, 5), dtype=np.uint8)
    mask_data[1:4, 1:4] = 1  # mask 3x3 in the middle

    lai_bounds = (0, 0, 5, 5)
    mask_bounds = (0, 0, 5, 5)

    padded, _ = pad_to_raster(lai_bounds, (1, 1), lai_data, mask_data, mask_bounds)
    masked = padded * mask_data
    masked[mask_data == 0] = np.nan

    assert np.sum(~np.isnan(masked)) == 9
    assert np.isclose(np.nanmedian(masked), 1.0)


def test_padding_partial_overlap():
    """True partial overlap: Mask partially intersects LAI (1x1 overlap)"""
    lai_data = np.full((3, 3), 2.0, dtype=np.float32)

    # Mask spans 3x3 starting from (2,2) to (5,5), only the top-left pixel overlaps LAI
    mask_data = np.zeros((3, 3), dtype=np.uint8)
    mask_data[0, 0] = 1  # top-left of mask corresponds to (2,2) in global coords

    lai_bounds = (0, 0, 3, 3)  # LAI spans 0 to 3
    mask_bounds = (2, 2, 5, 5)  # Mask spans 2 to 5

    padded, _ = pad_to_raster(lai_bounds, (1, 1), lai_data, mask_data, mask_bounds)
    masked = padded * mask_data

    assert padded.shape == (3, 3)  # due to union of bounds (2–3 in both X and Y)
    assert np.sum(~np.isnan(masked)) == 1  # only 1 overlapping pixel
    assert np.isclose(np.nanmedian(masked), 2.0)


def test_padding_partial_extent_overlap():
    """Mask and LAI have partial extent overlap (right half)"""
    lai_data = np.full((4, 4), 5.0, dtype=np.float32)
    lai_bounds = (0, 0, 4, 4)  # LAI from (0,0) to (4,4)

    mask_data = np.ones((4, 4), dtype=np.uint8)  # mask shape = 4x4
    mask_bounds = (2, 0, 6, 4)  # Mask from (2,0) to (6,4), so only overlaps right half of LAI

    padded_lai, was_padded = pad_to_raster(lai_bounds, (1, 1), lai_data, mask_data, mask_bounds)

    assert was_padded is True
    assert padded_lai.shape == (4, 6)  # padded LAI must be wider to match mask extent

    # Apply the mask
    masked = padded_lai * mask_data

    # Only 2 columns of LAI are overlapped (cols 2,3), so 2x4 = 8 pixels
    assert np.sum(~np.isnan(masked)) == 8
    assert np.isclose(np.nanmedian(masked), 5.0)


def test_padding_no_overlap_fails():
    """pad_to_raster should raise sys.exit (handled here with pytest.raises) if rasters don't overlap"""
    lai_bounds = (0, 0, 2, 2)
    mask_bounds = (10, 10, 12, 12)

    with pytest.raises(SystemExit):
        pad_to_raster(lai_bounds, (1, 1), np.ones((2, 2)), np.ones((2, 2)), mask_bounds)


def test_padding_needed():
    """Ensure padding occurs when mask bounds are larger than LAI bounds"""
    lai_data = np.ones((2, 2), dtype=np.float32)
    mask_data = np.ones((4, 4), dtype=np.uint8)

    lai_bounds = (0, 0, 2, 2)
    mask_bounds = (0, 0, 4, 4)

    padded, was_padded = pad_to_raster(lai_bounds, (1, 1), lai_data, mask_data, mask_bounds)
    assert was_padded is True
    assert padded.shape == (4, 4)
    assert np.count_nonzero(~np.isnan(padded)) == 4  # only original 2x2 data is valid


def test_negative_lai_clipped():
    """Check negative LAI values are clipped to zero before stats"""
    lai_data = np.array([[-1.0, 0.5], [0.8, -0.2]], dtype=np.float32)
    mask_data = np.ones((2, 2), dtype=np.uint8)

    lai_bounds = (0, 0, 2, 2)
    mask_bounds = (0, 0, 2, 2)

    padded, _ = pad_to_raster(lai_bounds, (1, 1), lai_data, mask_data, mask_bounds)
    masked = padded * mask_data

    # Clip negative
    masked[masked < 0] = 0
    assert np.isclose(np.nanmedian(masked), 0.25)


def test_nan_handling_median():
    """Check that NaN values are ignored in median computation"""
    lai_data = np.array([[np.nan, 0.5], [1.0, np.nan]], dtype=np.float32)
    mask_data = np.ones((2, 2), dtype=np.uint8)

    lai_bounds = (0, 0, 2, 2)
    mask_bounds = (0, 0, 2, 2)

    padded, _ = pad_to_raster(lai_bounds, (1, 1), lai_data, mask_data, mask_bounds)
    masked = padded * mask_data

    assert np.isclose(np.nanmedian(masked), 0.75)
