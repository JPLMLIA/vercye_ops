import os
import sys
import subprocess
import logging
from glob import glob
from datetime import datetime
import click
import pandas as pd
from dateutil.relativedelta import relativedelta
import time

import yaml


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
        date_str = basename.split("_")[-1]
        date_str = date_str.split(".")[0]  # Remove file extension
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
    t0 = time.time()
    try:
        # Inherit stdout/stderr so user sees real-time output
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during {step_desc}: return code {e.returncode}")
        raise RuntimeError(f"{step_desc} failed (exit code {e.returncode})")
    logger.info(f"Completed: {step_desc} in {time.time() - t0:.2f} seconds")


def batch_date_range(start_date, end_date, chunk_days=30):
        start = start_date
        end = end_date + relativedelta(days=1)
        while start < end:
            next_start = min(start + relativedelta(days=chunk_days), end)
            yield start.strftime('%Y-%m-%d'), next_start.strftime('%Y-%m-%d')
            start = next_start


def run_pipeline(config):
    date_ranges = config["date_ranges"]
    resolution = config["resolution"]
    geojson_path = config["geojson_path"]
    out_dir = config["out_dir"]
    region_out_prefix = config["region_out_prefix"]
    from_step = config.get("from_step", 0)
    num_workers = config.get("num_cores", 1)
    chunk_days = config.get("chunk_days", 30)

    if from_step not in [0, 2, 3]:
        raise ValueError("Invalid from_step value. Must be 0, 2, or 3.")

    tiles_out_dir = os.path.join(out_dir, "tiles")
    lai_dir = os.path.join(out_dir, "tile-lai")
    standardize_lai_dir = os.path.join(out_dir, "standardized-lai")
    merged_lai_dir = os.path.join(out_dir, "merged-lai")

    try:
        run_subprocess(["gdalinfo", "--version"], "Check GDAL installation")
    except RuntimeError:
        raise RuntimeError("GDAL is not installed or not in the PATH.")

    all_starts = []
    all_ends = []

    # Process all date ranges for step 0 and 1
    for i, dr in enumerate(date_ranges):
        try:
            start_date = datetime.strptime(dr["start_date"], "%Y-%m-%d").date()
            end_date = datetime.strptime(dr["end_date"], "%Y-%m-%d").date()

            all_starts.append(start_date)
            all_ends.append(end_date)

            logger.info(f"Processing date range {i+1}: {start_date} to {end_date}")

            for start, end in batch_date_range(start_date, end_date, chunk_days=chunk_days):
                if from_step <= 0:
                    os.makedirs(tiles_out_dir, exist_ok=True)
                    cmd = [
                        sys.executable, "1_download_S2_tiles.py",
                        "--start-date", start,
                        "--end-date", end,
                        "--resolution", str(resolution),
                        "--geojson-path", geojson_path,
                        "--output-dir", tiles_out_dir
                    ]
                    run_subprocess(cmd, f"Download tiles {start} to {end}")

                if from_step <= 1:
                    os.makedirs(lai_dir, exist_ok=True)
                    cmd = [
                        sys.executable, "2_1_primary_LAI_tiled.py",
                        tiles_out_dir,
                        lai_dir,
                        str(resolution),
                        "--start-date", start,
                        "--end-date", end,
                        "--num-cores", str(num_workers),
                        "--remove-original",
                    ]
                    run_subprocess(cmd, f"Compute LAI for {start} to {end}")

        except Exception as e:
            logger.error(f"Aborting further processing due to error in date range {start_date} to {end_date}: {e}")
            raise  # Reraise to halt the pipeline

    # Steps 2 and 3 are run once after all ranges
    overall_start = min(all_starts)
    overall_end = max(all_ends)

    if from_step <= 2:
        os.makedirs(standardize_lai_dir, exist_ok=True)
        cmd = [
            sys.executable, "2_2_standardize.py",
            lai_dir,
            standardize_lai_dir,
            str(resolution),
            "--start-date", overall_start.strftime("%Y-%m-%d"),
            "--end-date", overall_end.strftime("%Y-%m-%d"),
            "--num-cores", str(num_workers),
            "--remove-original",
        ]
        run_subprocess(cmd, "Standardize LAI files")

    if from_step <= 3:
        os.makedirs(merged_lai_dir, exist_ok=True)
        cmd = [
            sys.executable, "2_3_build_daily_LAI_vrts.py",
            standardize_lai_dir,
            merged_lai_dir,
            str(resolution),
            "--region-out-prefix", region_out_prefix,
            "--start-date", overall_start.strftime("%Y-%m-%d"),
            "--end-date", overall_end.strftime("%Y-%m-%d")
        ]
        run_subprocess(cmd, "Build daily VRTs for LAI")

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_pipeline.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    try:
        run_pipeline(config)
    except Exception as e:
        logger.error(f"Pipeline terminated with error: {e}")
        sys.exit(1)
