import os
import sys
import subprocess
import logging
from glob import glob
from datetime import datetime
import click

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def is_within_date_range(file_path: str, start_date: datetime.date, end_date: datetime.date) -> bool:
    """
    Check if the file date is within the specified range.
    """
    try:
        basename = os.path.basename(file_path)
        # Expecting format: prefix_YYYYMMDD_suffix
        date_str = basename.split("_")[-3]
        file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        return start_date <= file_date <= end_date
    except Exception as e:
        logger.warning(f"Skipping file '{file_path}' due to date parsing error: {e}")
        return False


def run_subprocess(cmd: list, step_desc: str):
    """
    Execute a subprocess and stream its output to the console.
    If it fails, log error and raise.
    """
    logger.info(f"Starting: {step_desc}\n  Command: {' '.join(cmd)}")
    try:
        # Inherit stdout/stderr so user sees real-time output
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during {step_desc}: return code {e.returncode}")
        raise RuntimeError(f"{step_desc} failed (exit code {e.returncode})")
    logger.info(f"Completed: {step_desc}")


def run_pipeline(
    start_date: str,
    end_date: str,
    resolution: int,
    geojson_path: str,
    out_dir: str,
    region_out_prefix: str,
    from_step: int = 0,
):
    """
    Run the LAI creation pipeline in sequential steps.
    If any step fails, no further steps are run.
    """
    # Validate dates
    try:
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError as e:
        raise click.BadParameter(f"Invalid date format: {e}")

    if start_date_dt > end_date_dt:
        raise click.BadParameter("start-date must be on or before end-date")

    # Validate steps
    max_step = 3
    if not (0 <= from_step <= max_step):
        raise click.BadParameter(f"--from-step must be between 0 and {max_step}")

    # Ensure GDAL is available
    try:
        run_subprocess(["gdalinfo", "--version"], "Check GDAL installation")
    except RuntimeError:
        raise click.ClickException("GDAL is not installed or not in the PATH.")

    # Paths
    tiles_out_dir = os.path.join(out_dir, "tiles")
    lai_dir = os.path.join(out_dir, "tile-lai")
    merged_lai_dir = os.path.join(out_dir, "merged-lai")

    # Step 0: Download tiles
    if from_step <= 0:
        os.makedirs(tiles_out_dir, exist_ok=True)
        cmd = [
            sys.executable, "1_download_S2_tiles.py",
            "--start-date", start_date,
            "--end-date", end_date,
            "--resolution", str(resolution),
            "--geojson-path", geojson_path,
            "--output-dir", tiles_out_dir
        ]
        run_subprocess(cmd, "Download Sentinel-2 tiles")

    # Step 1: Create LAI per tile
    if from_step <= 1:
        os.makedirs(lai_dir, exist_ok=True)
        cmd = [
            sys.executable, "2_1_primary_LAI_tiled.py",
            tiles_out_dir,
            lai_dir,
            str(resolution),
            "--start-date", start_date,
            "--end-date", end_date
        ]
        run_subprocess(cmd, "Compute LAI for each tile")

    # Step 2: Clean up intermediate tiles
    if from_step <= 2:
        logger.info("Deleting intermediate LAI tiles outside date range or mismatched resolution...")
        pattern = f"{lai_dir}/*_{resolution}m_*_LAI_tile.tif"
        for fpath in glob(pattern):
            if is_within_date_range(fpath, start_date_dt, end_date_dt):
                try:
                    os.remove(fpath)
                    logger.debug(f"Removed: {fpath}")
                except OSError as e:
                    logger.warning(f"Failed to remove {fpath}: {e}")
        logger.info("Cleanup complete.")

    # Step 3: Build daily VRTs
    if from_step <= 3:
        os.makedirs(merged_lai_dir, exist_ok=True)
        cmd = [
            sys.executable, "2_2_build_daily_LAI_vrts.py",
            lai_dir,
            merged_lai_dir,
            str(resolution),
            "--region-out-prefix", region_out_prefix,
            "--start-date", start_date,
            "--end-date", end_date
        ]
        run_subprocess(cmd, "Build daily VRTs for LAI")

    logger.info("Pipeline completed successfully.")


@click.command()
@click.option(
    "--start-date", type=click.DateTime(formats=["%Y-%m-%d"]), required=True,
    help="Start date in YYYY-MM-DD format."
)
@click.option(
    "--end-date", type=click.DateTime(formats=["%Y-%m-%d"]), required=True,
    help="End date in YYYY-MM-DD format."
)
@click.option(
    "--resolution", type=int, required=True,
    help="Spatial resolution in meters."
)
@click.option(
    "--geojson-path", type=click.Path(exists=True), required=True,
    help="Path to the GeoJSON file defining the region."
)
@click.option(
    "--out-dir", type=click.Path(file_okay=False), required=True,
    help="Output directory for all generated data."
)
@click.option(
    "--region-out-prefix", type=str, required=True,
    help="Prefix for the output VRT filenames."
)
@click.option(
    "--from-step", type=click.IntRange(0, 3), default=0,
    help="Pipeline step to start from (0: download, 1: tile LAI, 2: cleanup, 3: VRT build). Use on failure to resume."
)
def main(start_date, end_date, resolution, geojson_path, out_dir, region_out_prefix, from_step):
    """
    Entry point for the LAI creation pipeline.
    """
    try:
        run_pipeline(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            resolution,
            geojson_path,
            out_dir,
            region_out_prefix,
            from_step
        )
    except Exception as e:
        logger.error(f"Pipeline terminated with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
