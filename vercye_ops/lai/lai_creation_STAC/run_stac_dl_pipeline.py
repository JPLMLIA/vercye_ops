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
        end = end + relativedelta(days=1)

    while start < end:
        next_start = min(start + relativedelta(days=chunk_days), end)
        yield start.strftime("%Y-%m-%d"), next_start.strftime("%Y-%m-%d")
        start = next_start


def init_meta(meta_file, resolution, geojson_path, imagery_source):
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
    meta["imagery_source"] = imagery_source

    if "dates" not in meta:
        meta["dates"] = {}

    if resolution not in meta["dates"]:
        meta["dates"][str(resolution)] = []

    if "downloaded_dateranges" not in meta:
        meta["downloaded_dateranges"] = []

    if "lai_created_dateranges" not in meta:
        meta["lai_created_dateranges"] = []

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    return meta


def update_status(meta_file, resolution, status):
    with open(meta_file, "r", encoding="utf-8") as file:
        meta = json.load(file)

    meta["status"][str(resolution)] = status

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    return meta


def update_processed(meta_file, date_range, process_type):
    if process_type == "dl":
        with open(meta_file, "r", encoding="utf-8") as file:
            meta = json.load(file)
        meta["downloaded_dateranges"].append(date_range)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f)
    elif process_type == "lai":
        with open(meta_file, "r", encoding="utf-8") as file:
            meta = json.load(file)
        meta["lai_created_dateranges"].append(date_range)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f)
    else:
        raise ValueError("Invalid process type provided.")

    return meta


def _compose_azcopy_url(base_url, sas_token):
    """Append ``sas_token`` to ``base_url`` as a query string if provided.

    If ``sas_token`` is empty, returns ``base_url`` unchanged- azcopy will
    then fall back to the auth mode selected via AZCOPY_AUTO_LOGIN_TYPE
    (MSI / SPN / AZCLI), which is the recommended path when running on an
    Azure VM with a managed identity.
    """
    if not sas_token:
        return base_url
    token = sas_token.lstrip("?")
    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}{token}"


def _redact_url_query(arg):
    """Redact the query string from a URL-like argv element.

    azcopy commands embed the SAS token in the destination URL as a query
    string. We must NOT leak that into logs. Anything with a scheme that
    contains a ``?`` gets its query replaced with ``?<redacted>``. Plain
    filesystem paths pass through untouched.
    """
    if "://" in arg and "?" in arg:
        prefix, _, _ = arg.partition("?")
        return f"{prefix}?<redacted>"
    return arg


def _run_azcopy(cmd, step_desc, logger):
    """Run an azcopy command, redacting SAS tokens from the logged cmdline.

    Mirrors ``run_subprocess`` (same check=True + timing semantics) but the
    log line shows ``?<redacted>`` instead of the real SAS. The full cmd
    only ever lives in the child process argv, never in our log stream.
    """
    safe = [_redact_url_query(a) for a in cmd]
    logger.info(f"Starting: {step_desc}\n  Command: {' '.join(safe)}")
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during {step_desc}: return code {e.returncode}")
        raise RuntimeError(f"{step_desc} failed (exit code {e.returncode})")
    logger.info(f"Completed: {step_desc} in {time.time() - t0:.2f} seconds")


def upload_outputs_to_blob(config, out_dir, logger):
    """Push selected LAI outputs to Azure Blob via azcopy, driven by config.

    Uses ``azcopy copy --overwrite=ifSourceNewer --recursive`` so Blob
    accumulates outputs across runs: new files are uploaded, changed files
    are replaced, and files that exist only on Blob (e.g. last month's
    data the LAI VM has since cleaned up locally) are left alone. Sync
    semantics are deliberately NOT used — they would delete destination-only
    tiles and break historical VRTs.

    Mirrors the local layout under ``destination_url`` because the VRTs
    produced by ``2_3_build_daily_LAI_vrts.py`` reference ``standardized-lai/``
    tiles via ``relativeToVRT="1"`` paths, and flattening the tree would
    break those references.
    """
    cfg = config.get("blob_upload") or {}
    if not cfg.get("enabled"):
        logger.info("blob_upload not enabled in config; skipping upload.")
        return

    destination_url = cfg.get("destination_url", "").rstrip("/")
    if not destination_url:
        raise ValueError(
            "blob_upload.enabled is true but blob_upload.destination_url is missing."
        )

    sas_token = cfg.get("sas_token", "")
    include = cfg.get("include") or ["standardized-lai", "merged-lai", "meta.json", "region.geojson"]
    # include entries address things inside out_dir via relative paths;
    # reject absolute paths and ``..`` segments so a malformed config
    # cannot accidentally upload anything outside the LAI run directory.
    for rel in include:
        if os.path.isabs(rel) or ".." in rel.replace("\\", "/").split("/"):
            raise ValueError(
                f"blob_upload.include entry {rel!r} must be a relative path "
                f"inside out_dir (no absolute paths, no '..' segments)."
            )
    overwrite = cfg.get("overwrite", "ifSourceNewer")
    if overwrite not in ("ifSourceNewer", "true", "false", "prompt"):
        raise ValueError(
            f"blob_upload.overwrite must be one of "
            f"['ifSourceNewer', 'true', 'false', 'prompt']; got {overwrite!r}."
        )

    if shutil.which("azcopy") is None:
        raise RuntimeError(
            "azcopy not found on PATH. Install azcopy v10+ and ensure it is "
            "reachable, or disable blob_upload in the LAI config."
        )

    for rel in include:
        src = os.path.join(out_dir, rel)
        if not os.path.exists(src):
            logger.warning(f"Skipping upload of {rel!r}: {src} does not exist.")
            continue

        is_dir = os.path.isdir(src)
        dest_path = f"{destination_url}/{rel}"
        if is_dir:
            src_arg = src.rstrip("/") + "/"
            dest_arg = _compose_azcopy_url(dest_path + "/", sas_token)
        else:
            src_arg = src
            dest_arg = _compose_azcopy_url(dest_path, sas_token)

        cmd = ["azcopy", "copy", src_arg, dest_arg, f"--overwrite={overwrite}"]
        if is_dir:
            cmd.append("--recursive")

        # Use _run_azcopy (not run_subprocess) so any SAS token embedded in
        # dest_arg's query string is redacted from the log stream.
        _run_azcopy(cmd, f"Upload {rel!r} to blob", logger=logger)


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

    max_download_retries = config.get("max_download_retries", 2)

    all_starts = []
    all_ends = []

    # Validate that if there is already LAI data present it was produced by the same shapefile
    shapefile_copy_path = os.path.join(out_dir, "region.geojson")
    if os.path.exists(shapefile_copy_path):
        uploaded_gdf = gpd.read_file(geojson_path)
        existing_gdf = gpd.read_file(shapefile_copy_path)
        if not uploaded_gdf.equals(existing_gdf):
            raise ValueError(
                "Can't create LAI data in an output directory with existing LAI data that was produced with a different shapefile. Ensure to use the same!"
            )
    else:
        # Copy geojson to outdir for reproducability
        gdf = gpd.read_file(geojson_path)
        gdf.to_file(shapefile_copy_path)

    meta = init_meta(metadata_index_file, resolution, geojson_path, source)
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
                if from_step <= 0 and not [start, end] in meta["downloaded_dateranges"]:
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
                    # Retry the download subprocess to handle transient failures
                    # (e.g. expired SAS tokens). Already-downloaded items are
                    # skipped automatically by stac_downloader's status tracking.
                    for attempt in range(1, max_download_retries + 1):
                        try:
                            run_subprocess(cmd, f"Download tiles {start} to {end} for {cur_satellite}", logger=logger)
                            break
                        except RuntimeError:
                            if attempt < max_download_retries:
                                logger.warning(
                                    f"Download attempt {attempt}/{max_download_retries} failed for "
                                    f"{start} to {end}. Retrying (already-downloaded items will be skipped)..."
                                )
                            else:
                                raise
                    meta = update_processed(metadata_index_file, (start, end), "dl")

                if from_step <= 1 and not [start, end] in meta["lai_created_dateranges"]:
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
                    meta = update_processed(metadata_index_file, (start, end), "lai")

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
        meta = update_status(metadata_index_file, resolution, "standardizing")
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
            "--geojson-path",
            geojson_path,
            "--num-workers",
            str(num_workers_lai),
        ]
        meta = update_status(metadata_index_file, resolution, "merging")
        run_subprocess(cmd, "Build daily VRTs for LAI", logger=logger)

    # Create/Update metadata index file
    if from_step <= 4:
        meta = update_status(metadata_index_file, resolution, "finalizing")

        with open(metadata_index_file, "r", encoding="utf-8") as file:
            meta = json.load(file)

        # Update all found dates per resolution
        for resolution in meta["resolutions"]:
            # Dates will have format YYYY-MM-DD
            dates = [f.split("_")[-2] for f in os.listdir(merged_lai_dir)]
            meta["dates"][str(resolution)] = dates

        with open(metadata_index_file, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        meta = update_status(metadata_index_file, resolution, "completed")

    # Optional post-processing: push the relevant output subtree to Azure Blob
    if (config.get("blob_upload") or {}).get("enabled"):
        meta = update_status(metadata_index_file, resolution, "uploading")
        upload_outputs_to_blob(config, out_dir, logger=logger)
        meta = update_status(metadata_index_file, resolution, "uploaded")

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
