from pathlib import Path
from glob import glob
import subprocess

import click

@click.command()
@click.argument('region', type=str)
@click.argument('in_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('vrt_dir', type=click.Path(file_okay=False))
@click.argument('resolution', type=int)
def main(region, in_dir, vrt_dir, resolution):
    """Generate VRTs of all region tifs in a directory
    
    Parameters
    ----------
    region: str
        String pattern to match when looking for rasters region_*
    in_dir: str
        Directory of region geotiffs
    vrt_dir: str
        Directory to which vrts should be written
    resolution: int
        The resolution of the sentinel-2 images
    """
    # Create the output directory, ok if it already exists
    vrt_dir = Path(vrt_dir)
    vrt_dir.mkdir(exist_ok=True)
    
    # Get all individual files
    downloaded_files = sorted(glob(f"{in_dir}/{region}_{resolution}m*.tif"))
    print(f"Found {len(downloaded_files)}  files in {in_dir} for {region}")

    # Get unique dates
    # Files can be formatted as 
    # poltava_2022-04-01-0000000000-0000014080.tif
    # poltava_2022-04-16.tif

    # date parsing magic
    dates = sorted(list(set(['-'.join(Path(f).stem.split('_')[-1].split('-')[:3]) for f in downloaded_files])))
    for d in dates: print(d)
    print(f"Found {len(dates)} unique dates")

    for date in dates:
        print("Processing", date)
        # Get all the files for this date
        this_date_files = list(glob(f"{in_dir}/*{date}*.tif"))
        out_file = vrt_dir / Path(f"{region}_{resolution}m_{date}.vrt")

        # Build VRT
        subprocess.run(["gdalbuildvrt", out_file] + this_date_files)

if __name__ == "__main__":
    main()