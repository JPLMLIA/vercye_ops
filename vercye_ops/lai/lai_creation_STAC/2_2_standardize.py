import logging
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from glob import glob
from pathlib import Path

import click
import rasterio as rio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
        left, bottom, right, top = rio.warp.transform_bounds(
            src.crs, target_crs, *src.bounds
        )

        # Calculate equivalent resolution in target CRS
        x_res_target = (right - left) / dst_width
        y_res_target = (top - bottom) / dst_height

        logger.info(f"Sample file: {Path(lai_file).name}")
        logger.info(f"Original resolution: {x_res:.8f}, {y_res:.8f} in {src.crs}")
        logger.info(
            f"Target resolution: {x_res_target:.8f}, {y_res_target:.8f} in {target_crs}"
        )
        src = None  # Explicitely Close the file
    src = None  # Explicitely Close the file
    return (round(x_res_target, 6), round(y_res_target, 6))


def identify_target_resolution(file_paths, target_crs):
    """
    Identify the target CRS and resolution based on the input files.
    """
    # Identify the most common CRS
    most_common_crs, sample_file = get_most_common_crs(file_paths)
    logger.info(f"Most common CRS: {target_crs}")

    # Determine target resolution
    target_resolution = determine_target_resolution(sample_file, target_crs)
    logger.info(f"Target resolution: {target_resolution}")

    return target_resolution


def standardize_lai(args):
    lai_file, output_dir, target_crs, target_resolution, remove_original = args
    try:
        with rio.open(lai_file) as src:
            # Create output filename
            output_file = (
                Path(lai_file).stem.replace("_LAI_tile", "_LAI_tile_standardized")
                + ".tif"
            )
            output_file = Path(output_dir) / output_file

            # Calculate transform to target CRS
            left, bottom, right, top = rio.warp.transform_bounds(
                src.crs, target_crs, *src.bounds
            )
            xres, yres = target_resolution

            dst_width = int((right - left) / xres)
            dst_height = int((top - bottom) / yres)
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
                    "compress": "lzw",
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
                        resampling=Resampling.nearest,
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
    help="Start date",
    required=False,
    default=None,
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date",
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
def main(
    input_dir,
    output_dir,
    resolution,
    start_date,
    end_date,
    remove_original,
    target_crs,
    num_cores,
):

    lai_files = sorted(glob(f"{input_dir}/*_{resolution}m_*_LAI_tile.tif"))
    if start_date is not None and end_date is not None:
        lai_files = [
            vf for vf in lai_files if is_within_date_range(vf, start_date, end_date)
        ]

    logger.info(f"Found {len(lai_files)} VRT files at {resolution}m in {input_dir}")

    target_resolution = identify_target_resolution(lai_files, target_crs)

    args = [
        (lai_file, output_dir, target_crs, target_resolution, remove_original)
        for lai_file in lai_files
    ]
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        list(executor.map(standardize_lai, args))


if __name__ == "__main__":
    main()
