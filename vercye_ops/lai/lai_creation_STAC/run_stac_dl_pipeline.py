import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from itertools import product

import click
import geopandas as gpd
import yaml
from dateutil.relativedelta import relativedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def rel_path(*path_parts):
    return os.path.join(BASE_DIR, *path_parts)


def run_subprocess(cmd: list, step_desc: str, logger):
    """
    Execute a subprocess, capturing and logging all output.
    Raises RuntimeError on failure with full error output.
    """
    logger.info(f"Starting: {step_desc}\n  Command: {' '.join(cmd)}")
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during {step_desc}: return code {e.returncode}")
        logger.error(f"Stdout:\n{e.stdout}")
        logger.error(f"Stderr:\n{e.stderr}")
        raise RuntimeError(f"{step_desc} failed (exit code {e.returncode})")
    logger.info(f"Completed: {step_desc} in {time.time() - t0:.2f} seconds")

    logger.info(f"Completed: {step_desc} in {time.time() - t0:.2f} seconds")


def batch_date_range(start_date, end_date, chunk_days=30):
    start = start_date
    end = end_date
    if start == end:
        end = end + +relativedelta(days=1)

    while start < end:
        next_start = min(start + relativedelta(days=chunk_days), end)
        yield start.strftime("%Y-%m-%d"), next_start.strftime("%Y-%m-%d")
        start = next_start


def init_meta(meta_file, resolution, geojson_path):
    meta = {}
    if os.path.exists(meta_file):
        with open(meta_file, "r", encoding="utf-8") as file:
            meta = json.load(file)

    if "centroid" not in meta:
        gdf = gpd.read_file(geojson_path)
        gdf = gdf.to_crs("EPSG:4326")
        merged_geometry = gdf.union_all()
        centroid = (merged_geometry.centroid.y, merged_geometry.centroid.x)
        # meta['merged_geometry'] = mapping(merged_geometry)
        # Temporarily disabling to keep files smaller
        meta["merged_geometry"] = {}

        meta["centroid"] = centroid

    if "resolutions" in meta and resolution not in meta["resolutions"]:
        meta["resolutions"].append(resolution)

    if "resolutions" not in meta:
        meta["resolutions"] = [resolution]

    if "status" not in meta:
        meta["status"] = {}

    meta["status"][str(resolution)] = "generating"

    if "dates" not in meta:
        meta["dates"] = {}

    if resolution not in meta["dates"]:
        meta["dates"][str(resolution)] = []

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f)


def update_status(meta_file, resolution, status):
    with open(meta_file, "r", encoding="utf-8") as file:
        meta = json.load(file)

    meta["status"][str(resolution)] = status

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f)


def run_pipeline(config, logger):
    date_ranges = config["date_ranges"]
    satellite = config["satellite"]
    resolution = config["resolution"]
    geojson_path = config["geojson_path"]
    out_dir = config["out_dir"]
    region_out_prefix = config["region_out_prefix"]
    from_step = config.get("from_step", 0)
    num_workers_lai = config.get("num_cores_lai", 1)
    num_workers_download = config.get("num_cores_download", 1)
    chunk_days = config.get("chunk_days", 30)
    source = config["imagery_src"]
    keep_imagery = config["keep_imagery"]

    if satellite.lower() == "s2":
        if source.lower() == "es_s2c1":
            downloader_script_path = rel_path("1_download_S2_earthsearch.py")
        elif source.lower() == "mpc":
            downloader_script_path = rel_path("1_download_S2_MPC.py")
        else:
            raise ValueError("Invalid Source Provided")
    elif satellite.lower() == "hls":
        if source.lower() != "mpc":
            raise ValueError("Invalid Source Provided")
        downloader_script_path = rel_path("1_download_HLS_MPC.py")

    os.makedirs(out_dir, exist_ok=True)

    if from_step not in [0, 2, 3, 4]:
        raise ValueError("Invalid from_step value. Must be 0, 2, or 3, 4.")

    tiles_out_dir = os.path.join(out_dir, "tiles")
    lai_dir = os.path.join(out_dir, "tile-lai")
    standardize_lai_dir = os.path.join(out_dir, "standardized-lai")
    merged_lai_dir = os.path.join(out_dir, "merged-lai")
    metadata_index_file = os.path.join(out_dir, "meta.json")

    try:
        run_subprocess(["gdalinfo", "--version"], "Check GDAL installation", logger=logger)
    except RuntimeError:
        raise RuntimeError("GDAL is not installed or not in the PATH.")

    all_starts = []
    all_ends = []

    # Validate that if there is already LAI data present it was produced by the same shapefile
    shapefile_copy_path = os.path.join(out_dir, "region.geojson")
    if os.path.exists(shapefile_copy_path):
        with open(geojson_path) as f1, open(shapefile_copy_path) as f2:
            geo1 = json.load(f1)
            geo2 = json.load(f2)

            if geo1 != geo2:
                raise ValueError(
                    "Cant create LAI data in an output directory with existing LAI data that was produced with a different shapefile."
                )
    else:
        # Copy geojson to outdir for reproducability
        shutil.copyfile(geojson_path, shapefile_copy_path)

    init_meta(metadata_index_file, resolution, geojson_path)
    # Process all date ranges for step 0 and 1
    for i, dr in enumerate(date_ranges):
        try:
            start_date = datetime.strptime(dr["start_date"], "%Y-%m-%d").date()
            end_date = datetime.strptime(dr["end_date"], "%Y-%m-%d").date()

            all_starts.append(start_date)
            all_ends.append(end_date)

            logger.info(f"Processing date range {i+1}: {start_date} to {end_date}")

            satellites = ["HLS_L30"] if satellite == "HLS" else ["S2"]
            for (start, end), cur_satellite in product(
                batch_date_range(start_date, end_date, chunk_days=chunk_days), satellites
            ):
                # Use different folders for s30 and l30
                if from_step <= 0:
                    os.makedirs(tiles_out_dir, exist_ok=True)
                    cmd = [
                        sys.executable,
                        downloader_script_path,
                        "--start-date",
                        start,
                        "--end-date",
                        end,
                        "--resolution",
                        str(resolution),
                        "--geojson-path",
                        geojson_path,
                        "--output-dir",
                        tiles_out_dir,
                        "--num-workers",
                        str(num_workers_download),
                        "--satellite",
                        cur_satellite,
                    ]
                    run_subprocess(cmd, f"Download tiles {start} to {end} for {cur_satellite}", logger=logger)

                if from_step <= 1:
                    os.makedirs(lai_dir, exist_ok=True)
                    cmd = [
                        sys.executable,
                        rel_path("2_1_primary_LAI_tiled.py"),
                        tiles_out_dir,
                        lai_dir,
                        str(resolution),
                        "--start-date",
                        start,
                        "--end-date",
                        end,
                        "--num-cores",
                        str(num_workers_lai),
                        "--satellite",
                        cur_satellite,
                    ]

                    if not keep_imagery:
                        cmd.append("--remove-original")

                    run_subprocess(cmd, f"Compute LAI for {start} to {end}", logger=logger)

        except Exception as e:
            logger.error(f"Aborting further processing due to error in date range {start_date} to {end_date}: {e}")
            raise e  # Reraise to halt the pipeline

    # Steps 2 and 3 are run once after all ranges
    overall_start = min(all_starts)
    overall_end = max(all_ends)

    if from_step <= 2:
        os.makedirs(standardize_lai_dir, exist_ok=True)
        cmd = [
            sys.executable,
            rel_path("2_2_standardize.py"),
            lai_dir,
            standardize_lai_dir,
            str(resolution),
            "--start-date",
            overall_start.strftime("%Y-%m-%d"),
            "--end-date",
            overall_end.strftime("%Y-%m-%d"),
            "--num-cores",
            str(num_workers_lai),
            "--remove-original",
        ]
        update_status(metadata_index_file, resolution, "standardizing")
        run_subprocess(cmd, "Standardize LAI files", logger=logger)

    if from_step <= 3:
        os.makedirs(merged_lai_dir, exist_ok=True)
        cmd = [
            sys.executable,
            rel_path("2_3_build_daily_LAI_vrts.py"),
            standardize_lai_dir,
            merged_lai_dir,
            str(resolution),
            "--region-out-prefix",
            region_out_prefix,
            "--start-date",
            overall_start.strftime("%Y-%m-%d"),
            "--end-date",
            overall_end.strftime("%Y-%m-%d"),
        ]
        update_status(metadata_index_file, resolution, "merging")
        run_subprocess(cmd, "Build daily VRTs for LAI", logger=logger)

    # Create/Update metadata index file
    if from_step <= 4:
        update_status(metadata_index_file, resolution, "finalizing")

        with open(metadata_index_file, "r", encoding="utf-8") as file:
            meta = json.load(file)

        # Update all found dates per resolution
        for resolution in meta["resolutions"]:
            # Dates will have format YYYY-MM-DD
            dates = [f.split("_")[2] for f in os.listdir(merged_lai_dir)]
            meta["dates"][str(resolution)] = dates

        with open(metadata_index_file, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        update_status(metadata_index_file, resolution, "completed")

    logger.info("Pipeline completed successfully.")


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def main(config_path):
    """Run the pipeline with the specified CONFIG_PATH and SOURCE."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    try:
        run_pipeline(config, logger=logger)
    except Exception as e:
        logger.error(f"Pipeline terminated with error: {e}")
        raise e


if __name__ == "__main__":
    main()
