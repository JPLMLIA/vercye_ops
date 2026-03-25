import math
import multiprocessing
import os
import subprocess
from collections import defaultdict
from datetime import datetime
from glob import glob
from pathlib import Path

import click
import geopandas as gpd
import rasterio as rio

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()
logger.setLevel("INFO")


def is_within_date_range(vf, start_date, end_date):
    # files are expected to have the pattern f"{s2_dir}/{region}_{resolution}m_{date}_LAI_tile_standardized.tif"
    date = Path(vf).stem.split("_")[-4]
    date = datetime.strptime(date, "%Y-%m-%d")
    return start_date <= date <= end_date


def resolutions_are_close(res_set, tolerance=1e-3):
    res_list = list(res_set)
    base_res = res_list[0]
    for r in res_list[1:]:
        if not (
            math.isclose(base_res[0], r[0], abs_tol=tolerance) and math.isclose(base_res[1], r[1], abs_tol=tolerance)
        ):
            return False
    return True


def get_geojson_bounds(geojson_path, target_crs):
    """Extract bounds from GeoJSON file, reprojecting to target CRS if needed."""
    gdf = gpd.read_file(geojson_path)

    # Reproject to target CRS if needed
    if gdf.crs != target_crs:
        logger.info(f"Reprojecting GeoJSON from {gdf.crs} to {target_crs}")
        gdf = gdf.to_crs(target_crs)

    # Get total bounds (minx, miny, maxx, maxy)
    return gdf.total_bounds


def check_crs_and_resolution(lai_file):
    with rio.open(lai_file) as lai_ds:
        lai_crs = lai_ds.crs
        res_x, res_y = lai_ds.res
    return lai_crs, (res_x, res_y)


def build_vrt(args):
    date, paths, out_dir, region_out_prefix, res_x, res_y, crs_str, minx, miny, maxx, maxy, resolution = args
    out_file = os.path.join(out_dir, f"{region_out_prefix}_{resolution}m_{date}_LAI.vrt")
    logger.info(f"Processing for {out_file}")
    result = subprocess.run(
        [
            "gdalbuildvrt",
            "-tap",
            "-te",
            str(minx),
            str(miny),
            str(maxx),
            str(maxy),
            "-tr",
            str(res_x),
            str(res_y),
            "-a_srs",
            crs_str,
            out_file,
        ]
        + paths
    )
    if result.returncode != 0:
        logger.error(f"Error creating VRT for {date}: {result.stderr}")
    else:
        logger.info(f"VRT created successfully for {date} at {out_file}")


@click.command()
@click.argument("lai-dir", type=click.Path(exists=True, file_okay=False))
@click.argument("out-dir", type=click.Path(file_okay=False))
@click.argument("resolution", type=int)
@click.option(
    "--region-out-prefix",
    type=str,
    default="merged_regions",
    help="Prefix for the output VRT files",
)
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
    "--geojson-path",
    type=click.Path(file_okay=True),
    help="ROI GeoJson path for aligned extent.",
    required=True,
    default=None,
)
@click.option(
    "--num-workers",
    type=int,
    default=1,
    help="Number of cores to use for parallel processing.",
)
def main(lai_dir, out_dir, resolution, region_out_prefix, start_date, end_date, geojson_path, num_workers):
    """Generate daily VRTs for LAI data, combining all regions in the provided folder into a single one.

    Parameters
    ----------
    resolution : int
        The resolution of the data in meters.
    lai_dir : str
        The directory where the LAI data is stored.
    out_dir : str
        The directory where the output VRTs will be stored.
    start_date : str
        The start date for the VRTs in YYYY-MM-DD.
    end_date : str
        The end date for the VRTs in YYYY-MM-DD.
    geojson_path: str
        Path to ROI GeoJSON for common extent.
    """

    # Validate that all files have the same CRS and resolution
    crs = []
    resolutions = []

    lai_files = sorted(glob(f"{lai_dir}/*_{resolution}m_*_LAI_tile_standardized.tif"))
    if start_date is not None and end_date is not None:
        lai_files = [f for f in lai_files if is_within_date_range(f, start_date, end_date)]

    if not lai_files:
        logger.error(f"No LAI files found in {lai_dir} with resolution {resolution}m")
        return

    # Validate that all input files have the same CRS and resolution
    # This allows for downstream tasks to assume consistent spatial properties
    # and to be faster as they don't need to reproject or resample on the fly on each regional read
    logger.info(f"Found {len(lai_files)} LAI files in {lai_dir} with resolution {resolution}m")
    logger.info("Validating CRS and Resolution...")
    with multiprocessing.Pool(num_workers) as pool:
        crs_resolutions = pool.map(check_crs_and_resolution, lai_files)

    crs = [cr for cr, _ in crs_resolutions]
    resolutions = [res for _, res in crs_resolutions]

    resolutions_set = set(resolutions)
    crs_set = set(crs)
    if not resolutions_are_close(resolutions_set):
        logger.info(f"Resolutions found: {resolutions_set}")
        raise Exception("LAI files have different resolutions. Please use the same resolution for all LAI files.")

    if len(crs_set) > 1:
        logger.info(f"CRS found: {crs_set}")
        raise Exception("LAI files have different CRS. Please use the same CRS for all LAI files.")

    # Group files by date
    date_groups = defaultdict(list)
    for region_file in lai_files:
        date = region_file.split("_")[-4]
        date_groups[date].append(region_file)

    # Use the most frequent resolution
    res_x, res_y = max(resolutions, key=resolutions.count)

    with rio.open(lai_files[0]) as src:
        crs = src.crs
        crs_str = crs.to_string()

    # Ensure geojson is in same crs
    bounds = get_geojson_bounds(geojson_path, crs)
    minx, miny, maxx, maxy = bounds

    # Create vrt per group
    # Ensure all have same extent and resolution and crs to make them stackable

    os.makedirs(out_dir, exist_ok=True)

    with multiprocessing.Pool(num_workers) as pool:
        pool.map(
            build_vrt,
            [
                (date, paths, out_dir, region_out_prefix, res_x, res_y, crs_str, minx, miny, maxx, maxy, resolution)
                for date, paths in date_groups.items()
            ],
        )

    logger.info(f"VRTS created successfully in {out_dir} with prefix {region_out_prefix}")


if __name__ == "__main__":
    main()
