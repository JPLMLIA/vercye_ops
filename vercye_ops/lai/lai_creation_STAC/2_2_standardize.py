import math
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from glob import glob
from pathlib import Path

import click
import rasterio as rio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()
logger.setLevel("INFO")


def is_within_date_range(vf, start_date, end_date):
    # files have pattern f"{s2_dir}/*_{resolution}m_{date}_LAI_tile.tif"
    date = Path(vf).stem.split("_")[-3]
    date = datetime.strptime(date, "%Y-%m-%d")
    return start_date <= date <= end_date


def get_most_common_crs(files):
    """Identify the most common CRS from all files"""
    crs_counts = {}
    for f in files:
        with rio.open(f) as src:
            if src.crs is None:
                raise ValueError(f"CRS is None for {f}")

            crs = src.crs
            if crs in crs_counts:
                crs_counts[crs][0] += 1
            else:
                crs_counts[crs] = [1, f]

    # Get the most common CRS
    most_common_el = max(crs_counts.items(), key=lambda x: x[1][0])
    return most_common_el[0], most_common_el[1][1]


def determine_target_resolution(lai_file, target_crs="EPSG:4326"):
    """
    Sample one file to determine appropriate resolution for the target CRS
    that maintains similar level of detail
    """
    # TODO Use first non-empty file for sampling
    with rio.open(lai_file) as src:
        # if src.count > 0 and not np.all(src.read(1) == 0):
        # Calculate transform to target CRS
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )

        # Calculate resolution in target CRS units
        x_res = (src.bounds.right - src.bounds.left) / src.width
        y_res = (src.bounds.top - src.bounds.bottom) / src.height

        # Get sample points for resolution calculation
        left, bottom, right, top = rio.warp.transform_bounds(src.crs, target_crs, *src.bounds)

        # Calculate equivalent resolution in target CRS
        x_res_target = (right - left) / dst_width
        y_res_target = (top - bottom) / dst_height

        logger.info(f"Sample file: {Path(lai_file).name}")
        logger.info(f"Original resolution: {x_res:.8f}, {y_res:.8f} in {src.crs}")
        logger.info(f"Target resolution: {x_res_target:.8f}, {y_res_target:.8f} in {target_crs}")
        src = None  # Explicitely Close the file
    src = None  # Explicitely Close the file
    return (x_res_target, y_res_target)


def identify_target_resolution(file_paths, target_crs, output_dir):
    """
    Identify the target CRS and resolution based on the input files.
    """

    # If there are already files in the output dir, match their crs & resolution
    if not len(os.listdir(output_dir)) == 0:
        with rio.open(os.path.join(output_dir, os.listdir(output_dir)[0])) as src:
            crs = src.crs
            if crs is None:
                raise ValueError("CRS of reference file is None.")
            xres = abs(src.transform.a)
            yres = abs(src.transform.e)
            target_res = (xres, yres)
        return target_res

    # Identify the most common CRS to use this as the reference for target resolution
    most_common_crs, sample_file = get_most_common_crs(file_paths)
    logger.info(f"Most common CRS: {target_crs}")

    # Determine target resolution
    target_resolution = determine_target_resolution(sample_file, target_crs)
    logger.info(f"Target resolution: {target_resolution}")

    return target_resolution


def snap_bounds_to_grid(bounds, xres, yres):
    left, bottom, right, top = bounds
    left = math.floor(left / xres) * xres
    right = math.ceil(right / xres) * xres
    bottom = math.floor(bottom / yres) * yres
    top = math.ceil(top / yres) * yres
    return left, bottom, right, top


def standardize_lai(args):
    lai_file, output_dir, target_crs, target_resolution, remove_original = args
    xres, yres = target_resolution
    try:
        with rio.open(lai_file) as src:
            # Create output filename
            output_file = Path(lai_file).stem.replace("_LAI_tile", "_LAI_tile_standardized") + ".tif"
            output_file = Path(output_dir) / output_file

            raw_bounds = rio.warp.transform_bounds(src.crs, target_crs, *src.bounds)
            left, bottom, right, top = snap_bounds_to_grid(raw_bounds, xres, yres)

            # Compute dimensions (use round to avoid float slop)
            dst_width = int(round((right - left) / xres))
            dst_height = int(round((top - bottom) / yres))

            dst_transform = from_origin(left, top, xres, yres)

            # Create metadata for the new file
            meta = src.meta.copy()
            meta.update(
                {
                    "crs": target_crs,
                    "transform": dst_transform,
                    "width": dst_width,
                    "height": dst_height,
                    "dtype": "float32",
                    "compress": "deflate",
                    "predictor": 3,
                    "tiled": True,
                    "blockxsize": 512,
                    "blockysize": 512,
                }
            )

            # Reproject and write to new file
            with rio.open(output_file, "w", **meta) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rio.band(src, i),
                        destination=rio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear,
                    )
        logger.info(f"Standardized LAI file saved: {output_file}")

        if remove_original:
            try:
                os.remove(lai_file)
                logger.info(f"Removed original file: {lai_file}")
            except OSError as e:
                logger.warning(f"Failed to remove {lai_file}: {e}")
    except Exception as e:
        logger.warning(f"Failed to process {lai_file}: {e}")
        raise e


@click.command()
@click.argument("input-dir", type=click.Path(exists=True))
@click.argument("output-dir", type=click.Path(file_okay=False))
@click.argument("resolution", type=int)
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date. Optional to constrain the range of tiles in the folder that should be standardized.",
    required=False,
    default=None,
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date. Optional to constrain the range of tiles in the folder that should be standardized.",
    required=False,
    default=None,
)
@click.option(
    "--remove-original",
    is_flag=True,
    help="Remove original files after standardization",
    default=False,
)
@click.option(
    "--target-crs",
    type=str,
    default="EPSG:4326",
    help="Target CRS for standardized files. Default is EPSG:4326.",
    required=False,
)
@click.option(
    "--num-cores",
    type=int,
    default=64,
    help="Number of workers (cores) to use.",
)
def main(input_dir, output_dir, resolution, start_date, end_date, remove_original, target_crs, num_cores):

    lai_files = sorted(glob(f"{input_dir}/*_{resolution}m_*_LAI_tile.tif"))
    if start_date is not None and end_date is not None:
        lai_files = [vf for vf in lai_files if is_within_date_range(vf, start_date, end_date)]

    logger.info(f"Found {len(lai_files)} tif files at {resolution}m in {input_dir}")

    target_resolution = identify_target_resolution(lai_files, target_crs, output_dir)

    args = [(lai_file, output_dir, target_crs, target_resolution, remove_original) for lai_file in lai_files]
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        list(executor.map(standardize_lai, args))


if __name__ == "__main__":
    main()
