import math
import os
import subprocess
from collections import defaultdict
from datetime import datetime
from glob import glob
from pathlib import Path

import click
import rasterio as rio


def is_within_date_range(vf, start_date, end_date):
    # files are expected to have the pattern f"{s2_dir}/{region}_{resolution}m_{date}_LAI.tif"
    date = Path(vf).stem.split("_")[-2]
    date = datetime.strptime(date, "%Y-%m-%d")
    return start_date <= date <= end_date


def resolutions_are_close(res_set, tolerance=1e-3):
    res_list = list(res_set)
    base_res = res_list[0]
    for r in res_list[1:]:
        if not (
            math.isclose(base_res[0], r[0], abs_tol=tolerance)
            and math.isclose(base_res[1], r[1], abs_tol=tolerance)
        ):
            return False
    return True


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
def main(lai_dir, out_dir, resolution, region_out_prefix, start_date, end_date):
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
    """

    # Validate that all files have the same CRS and resolution
    crs = []
    resolutions = []

    lai_files = sorted(glob(f"{lai_dir}/*_{resolution}m_*_LAI.tif"))
    if start_date is not None and end_date is not None:
        lai_files = [f for f in lai_files if is_within_date_range(f, start_date, end_date)]

    if not lai_files:
        print(f"No LAI files found in {lai_dir} with resolution {resolution}m")
        return

    print(f"Found {len(lai_files)} LAI files in {lai_dir} with resolution {resolution}m")
    print("Validating CRS and Resolution...")
    for lai_file in lai_files:
        with rio.open(lai_file) as lai_ds:
            lai_crs = lai_ds.crs
            res_x, res_y = lai_ds.res
            crs.append(lai_crs)
            resolutions.append((res_x, res_y))

    resolutions_set = set(resolutions)
    crs_set = set(crs)
    if not resolutions_are_close(resolutions_set):
        print(f"Resolutions found: {resolutions_set}")
        raise Exception(
            f"LAI files have different resolutions. Please use the same resolution for all LAI files."
        )

    if len(crs_set) > 1:
        print(f"CRS found: {crs_set}")
        raise Exception(f"LAI files have different CRS. Please use the same CRS for all LAI files.")

    # Group files by date
    date_groups = defaultdict(list)
    for region_file in lai_files:
        date = region_file.split("_")[-2]
        date_groups[date].append(region_file)

    # Use the most frequent resolution
    res_x, res_y = max(resolutions, key=resolutions.count)

    # Create vrt per group
    for date, paths in date_groups.items():
        out_file = os.path.join(out_dir, f"{region_out_prefix}_{resolution}m_{date}_LAI.vrt")
        subprocess.run(["gdalbuildvrt", "-tr", str(res_x), str(res_y), out_file] + paths)

    print(f"VRTS created successfully in {out_dir} with prefix {region_out_prefix}")


if __name__ == "__main__":
    main()
