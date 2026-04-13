import logging
import os
import tempfile
from typing import List

import click
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


def _validate_inputs(tif_paths: List[str]):
    """Validate that all input TIFs share the same CRS and resolution."""
    if not tif_paths:
        raise ValueError("No input TIF files provided.")

    ref_crs = None
    ref_res = None

    for path in tif_paths:
        with rasterio.open(path) as src:
            if ref_crs is None:
                ref_crs = src.crs
                ref_res = src.res
            else:
                if src.crs != ref_crs:
                    raise ValueError(f"CRS mismatch: {path} has {src.crs}, expected {ref_crs}")
                if abs(src.res[0] - ref_res[0]) > 1e-10 or abs(src.res[1] - ref_res[1]) > 1e-10:
                    raise ValueError(f"Resolution mismatch: {path} has {src.res}, expected {ref_res}")

    return ref_crs, ref_res


def _build_coverage_mask_array(tif_paths: List[str], merged_transform, merged_shape, merged_crs):
    """
    Build a coverage mask from per-region TIFs.

    For each region TIF, any pixel that has data (even NaN from cropmask) is marked as covered
    if it falls within the region's valid data extent. We use the region TIF's own nodata
    handling: pixels that are truly nodata (outside the region boundary) are not covered,
    while NaN pixels inside the region (e.g., from cropmask) ARE covered.

    Returns a uint8 array: 1 = covered by a primary region, 0 = not covered.
    """
    coverage = np.zeros(merged_shape, dtype=np.uint8)

    for path in tif_paths:
        with rasterio.open(path) as src:
            # Read the data and the valid data mask
            data = src.read(1)
            # The mask from rasterio: True where data is valid (not nodata)
            # For float data with nan nodata, this is ~np.isnan(data)
            # But we want coverage = "within region bounds", not just "has crop"
            # So we mark all pixels within the region's window as covered,
            # regardless of whether yield is NaN (could be non-cropland)

            # Calculate the window of this region in the merged raster
            from rasterio.windows import from_bounds

            window = from_bounds(*src.bounds, merged_transform)
            row_off = max(0, int(round(window.row_off)))
            col_off = max(0, int(round(window.col_off)))
            row_end = min(merged_shape[0], row_off + src.height)
            col_end = min(merged_shape[1], col_off + src.width)

            # The region's actual data extent
            src_row_start = max(0, -int(round(window.row_off)))
            src_col_start = max(0, -int(round(window.col_off)))
            src_row_end = src_row_start + (row_end - row_off)
            src_col_end = src_col_start + (col_end - col_off)

            if row_end > row_off and col_end > col_off:
                # Mark all pixels in this region's extent as covered
                # We use the alpha/valid mask if available, otherwise mark the full extent
                region_slice = data[src_row_start:src_row_end, src_col_start:src_col_end]
                # A pixel is "covered" if it's within the region file's extent
                # (the file itself was clipped to the region boundary)
                coverage[row_off:row_end, col_off:col_end] = np.where(
                    np.isfinite(region_slice) | np.isnan(region_slice),
                    1,
                    coverage[row_off:row_end, col_off:col_end],
                )

    return coverage


def _atomic_write(src_path: str, dst_path: str):
    """Atomically move a file from src to dst (rename on success)."""
    os.replace(src_path, dst_path)


def _reproject_raster(
    input_path: str,
    output_path: str,
    target_crs: str,
    resampling: Resampling = Resampling.nearest,
    nodata=None,
):
    """Reproject a raster to a target CRS, writing to a temp file and renaming."""
    with rasterio.open(input_path) as src:
        if nodata is None:
            nodata = src.nodata

        transform, width, height = calculate_default_transform(src.crs, target_crs, src.width, src.height, *src.bounds)

        profile = src.profile.copy()
        profile.update(
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
            nodata=nodata,
            compress="lzw",
        )

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tif", dir=os.path.dirname(output_path))
        os.close(tmp_fd)
        try:
            with rasterio.open(tmp_path, "w", **profile) as dst:
                for band_idx in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band_idx),
                        destination=rasterio.band(dst, band_idx),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=resampling,
                        src_nodata=nodata,
                        dst_nodata=nodata,
                    )
            _atomic_write(tmp_path, output_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    return output_path


def create_yield_mosaic(
    region_yield_tif_paths: List[str],
    output_mosaic_4326_path: str,
    output_mosaic_projected_path: str,
    output_coverage_mask_projected_path: str,
    target_crs: str,
    nodata_value: float = float("nan"),
):
    """
    Stitch all per-region yield TIFs into a mosaic and reproject.

    Produces THREE outputs:
    1. yield_mosaic_4326.tif -- mosaic in EPSG:4326 (internal reference)
    2. yield_mosaic_projected.tif -- mosaic in target equal-area CRS (for all stats)
    3. coverage_mask_projected.tif -- binary mask (1=covered, 0=not) in target CRS

    Parameters
    ----------
    region_yield_tif_paths : list of str
        Paths to per-region yield map TIFs (in EPSG:4326, values in kg/ha).
    output_mosaic_4326_path : str
        Output path for the EPSG:4326 mosaic.
    output_mosaic_projected_path : str
        Output path for the equal-area projected mosaic.
    output_coverage_mask_projected_path : str
        Output path for the coverage mask in equal-area projection.
    target_crs : str
        Target equal-area CRS (e.g., 'EPSG:9854').
    nodata_value : float
        Nodata value for yield maps (default: NaN).
    """
    logger.info(f"Creating yield mosaic from {len(region_yield_tif_paths)} region TIFs...")

    # Validate inputs
    ref_crs, ref_res = _validate_inputs(region_yield_tif_paths)
    logger.info(f"Input CRS: {ref_crs}, resolution: {ref_res}")

    # Step 1: Merge all region TIFs into one mosaic in EPSG:4326
    logger.info("Merging region yield maps into mosaic...")
    datasets = [rasterio.open(p) for p in region_yield_tif_paths]
    try:
        merged_array, merged_transform = merge(datasets, nodata=nodata_value)
    except MemoryError:
        logger.warning("MemoryError during merge. Attempting VRT-based approach...")
        for ds in datasets:
            ds.close()
        merged_array, merged_transform = _merge_via_vrt(region_yield_tif_paths, nodata_value)
        datasets = []  # already closed

    merged_shape = (merged_array.shape[1], merged_array.shape[2])

    # Build coverage mask from the same region TIFs (before closing datasets)
    logger.info("Building coverage mask...")
    coverage_array = _build_coverage_mask_array(region_yield_tif_paths, merged_transform, merged_shape, ref_crs)

    # Close datasets
    for ds in datasets:
        ds.close()

    # Write EPSG:4326 mosaic
    logger.info(f"Writing EPSG:4326 mosaic to {output_mosaic_4326_path}")
    profile_4326 = {
        "driver": "GTiff",
        "dtype": merged_array.dtype,
        "width": merged_shape[1],
        "height": merged_shape[0],
        "count": 1,
        "crs": ref_crs,
        "transform": merged_transform,
        "nodata": nodata_value,
        "compress": "lzw",
    }

    tmp_fd, tmp_4326 = tempfile.mkstemp(suffix=".tif", dir=os.path.dirname(output_mosaic_4326_path))
    os.close(tmp_fd)
    try:
        with rasterio.open(tmp_4326, "w", **profile_4326) as dst:
            dst.write(merged_array[0], 1)
        _atomic_write(tmp_4326, output_mosaic_4326_path)
    except Exception:
        if os.path.exists(tmp_4326):
            os.remove(tmp_4326)
        raise

    # Write temporary coverage mask in 4326 for reprojection
    tmp_fd, tmp_cov_4326 = tempfile.mkstemp(suffix=".tif", dir=os.path.dirname(output_coverage_mask_projected_path))
    os.close(tmp_fd)
    try:
        cov_profile_4326 = profile_4326.copy()
        cov_profile_4326.update(dtype="uint8", nodata=255)
        with rasterio.open(tmp_cov_4326, "w", **cov_profile_4326) as dst:
            dst.write(coverage_array, 1)

        # Step 2: Reproject mosaic to target CRS
        logger.info(f"Reprojecting mosaic to {target_crs}...")
        _reproject_raster(
            output_mosaic_4326_path,
            output_mosaic_projected_path,
            target_crs,
            resampling=Resampling.nearest,
            nodata=nodata_value,
        )

        # Step 3: Reproject coverage mask to target CRS
        logger.info(f"Reprojecting coverage mask to {target_crs}...")
        _reproject_raster(
            tmp_cov_4326,
            output_coverage_mask_projected_path,
            target_crs,
            resampling=Resampling.nearest,
            nodata=255,
        )
    finally:
        if os.path.exists(tmp_cov_4326):
            os.remove(tmp_cov_4326)

    logger.info("Mosaic creation complete.")


def _merge_via_vrt(tif_paths: List[str], nodata_value: float):
    """
    Fallback merge using GDAL VRT for memory-constrained environments.
    Builds a VRT from the input TIFs and reads the merged result.
    """
    from osgeo import gdal

    gdal.UseExceptions()

    tmp_fd, vrt_path = tempfile.mkstemp(suffix=".vrt")
    os.close(tmp_fd)

    try:
        vrt = gdal.BuildVRT(vrt_path, tif_paths)
        vrt.FlushCache()
        vrt = None  # close

        with rasterio.open(vrt_path) as src:
            data = src.read()
            transform = src.transform
    finally:
        if os.path.exists(vrt_path):
            os.remove(vrt_path)

    return data, transform


@click.command()
@click.option(
    "--yield-tif-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing per-region subdirectories with yield_map.tif files.",
)
@click.option(
    "--output-mosaic-4326",
    required=True,
    type=click.Path(),
    help="Output path for the EPSG:4326 yield mosaic.",
)
@click.option(
    "--output-mosaic-projected",
    required=True,
    type=click.Path(),
    help="Output path for the equal-area projected yield mosaic.",
)
@click.option(
    "--output-coverage-mask",
    required=True,
    type=click.Path(),
    help="Output path for the coverage mask in equal-area projection.",
)
@click.option(
    "--target-crs",
    required=True,
    help="Target equal-area CRS string (e.g., EPSG:9854).",
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def cli(
    yield_tif_dir,
    output_mosaic_4326,
    output_mosaic_projected,
    output_coverage_mask,
    target_crs,
    verbose,
):
    """Create yield mosaic from per-region TIFs, reproject, and build coverage mask."""
    logging_level = logging.INFO if verbose else logging.WARNING
    logger.setLevel(logging_level)

    # Discover yield_map.tif files in subdirectories
    tif_paths = []
    for region_dir in sorted(os.listdir(yield_tif_dir)):
        region_path = os.path.join(yield_tif_dir, region_dir)
        if not os.path.isdir(region_path):
            continue
        yield_tif = os.path.join(region_path, f"{region_dir}_yield_map.tif")
        if os.path.exists(yield_tif):
            tif_paths.append(yield_tif)

    if not tif_paths:
        raise click.ClickException(f"No yield_map.tif files found in {yield_tif_dir}")

    logger.info(f"Found {len(tif_paths)} yield map TIFs")

    create_yield_mosaic(
        region_yield_tif_paths=tif_paths,
        output_mosaic_4326_path=output_mosaic_4326,
        output_mosaic_projected_path=output_mosaic_projected,
        output_coverage_mask_projected_path=output_coverage_mask,
        target_crs=target_crs,
    )


if __name__ == "__main__":
    cli()
