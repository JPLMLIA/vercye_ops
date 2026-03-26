import json
import os
import shutil
import signal
from collections import deque
from datetime import datetime
from typing import List

import geopandas as gpd
import yaml
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from models import AddLaiDatesConfig, LAIEntry, LAIGenerationConfig
from worker import run_lai_generation

from vercye_ops.utils.env_utils import read_lai_dir_from_env

router = APIRouter(
    prefix="/lai",
    tags=["lai"],
)

lai_dir = read_lai_dir_from_env()


def get_num_processing_cores(resolution):
    if resolution <= 10:
        num_cores_download = 120
        num_cores_lai = 35
    else:
        num_cores_download = 120
        num_cores_lai = 85

    return num_cores_download, num_cores_lai


@router.get("")
def get_all_lai_entries() -> List[LAIEntry]:
    if not os.path.exists(lai_dir):
        return []

    entries = []
    for f in os.listdir(lai_dir):
        if not os.path.isdir(os.path.join(lai_dir, f)):
            continue

        entry_id = f
        metadata_path = os.path.join(lai_dir, entry_id, "meta.json")

        if not os.path.exists(metadata_path):
            continue

        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        merged_geometry = metadata["merged_geometry"]
        lat = metadata["centroid"][0]
        lon = metadata["centroid"][1]
        for resolution in metadata["resolutions"]:
            resolution = str(resolution)
            status = metadata["status"][resolution]
            dates = metadata["dates"][resolution]
            entries.append(
                LAIEntry.from_shapely(
                    id=entry_id,
                    lat=lat,
                    lng=lon,
                    dates=dates,
                    status=status,
                    resolution=resolution,
                    geometry=merged_geometry,
                )
            )

    return entries


@router.post("/actions/add")
def add_dates(config: str = Form(...)):
    try:
        lai_config = AddLaiDatesConfig(**json.loads(config))
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON for lai_config")

    region_out_prefix = lai_config.name
    out_dir = os.path.join(lai_dir, region_out_prefix)

    if not os.path.exists(out_dir):
        raise HTTPException(status_code=400, detail=f"No existing lai for region {region_out_prefix}.")

    # Validate date ranges
    # TODO might want to check with existing dates that there is no doubles but maybe also ok for now
    for dr in lai_config.date_ranges:
        start = datetime.strptime(dr.start_date, "%Y-%m-%d")
        end = datetime.strptime(dr.end_date, "%Y-%m-%d")
        if end <= start:
            raise HTTPException(
                status_code=400,
                detail=f"End date {dr.end_date} must be after start date {dr.start_date}",
            )

    num_cores_download, num_cores_lai  = get_num_processing_cores(lai_config.resolution)

    geojson_path = os.path.join(out_dir, "region.geojson")

    meta_file = os.path.join(out_dir, "meta.json")
    with open(meta_file, "r", encoding="utf-8") as file:
        meta = json.load(file)

    for res, status in meta["status"].items():
        if status in ["generating", "standardizing", "merging", "finalizing"]:
            raise HTTPException(
                status_code=400,
                detail="There already is an execution of this LAI product running. Wait for it to complete before adding dates.",
            )

    imagery_src = meta["imagery_source"]

    # Prepare YAML configuration
    config_dict = {
        "date_ranges": [{"start_date": dr.start_date, "end_date": dr.end_date} for dr in lai_config.date_ranges],
        "resolution": lai_config.resolution,
        "geojson_path": geojson_path,
        "out_dir": out_dir,
        "region_out_prefix": region_out_prefix,
        "from_step": 0,
        "num_cores_download": num_cores_download,
        "num_cores_lai": num_cores_lai,
        "chunk_days": lai_config.chunk_days,
        "imagery_src": imagery_src,
        "keep_imagery": lai_config.keep_imagery,
        "satellite": "S2",
    }

    executions_dir = os.path.join(out_dir, "executions")
    suffix = (
        max(
            [int(f) for f in os.listdir(executions_dir) if os.path.isdir(os.path.join(executions_dir, f))],
            default=0,
        )
        + 1
    )
    execuction_dir = os.path.join(out_dir, "executions", str(suffix))
    os.makedirs(execuction_dir, exist_ok=True)

    config_path = os.path.join(execuction_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, sort_keys=False)

    run_lai_generation.delay(config_path)


@router.post("/actions/generate")
def generate_lai(lai_config: str = Form(...), region_shapefile: UploadFile = File(...)):
    try:
        lai_config = LAIGenerationConfig(**json.loads(lai_config))
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON for lai_config")

    region_out_prefix = lai_config.name
    out_dir = os.path.join(lai_dir, region_out_prefix)

    # Validate date ranges
    for dr in lai_config.date_ranges:
        start = datetime.strptime(dr.start_date, "%Y-%m-%d")
        end = datetime.strptime(dr.end_date, "%Y-%m-%d")
        if end <= start:
            raise HTTPException(
                status_code=400,
                detail=f"End date {dr.end_date} must be after start date {dr.start_date}",
            )

    # Compare with existing geojson if exists
    geojson_path = os.path.join(out_dir, "region.geojson")
    if os.path.exists(out_dir) and os.path.exists(geojson_path):
        uploaded_gdf = gpd.read_file(region_shapefile.file)
        existing_gdf = gpd.read_file(geojson_path)
        if not uploaded_gdf.equals(existing_gdf):
            raise HTTPException(status_code=400, detail="Uploaded region.geojson does not match existing one.")
    else:
        os.makedirs(out_dir, exist_ok=True)

    # TODO might want to do some validation here, but not sure yet if needed
    # we should update this if users encounter errors based on bad input

    # Save uploaded shapefile (its a geojson) to geojson_path
    gdf = gpd.read_file(region_shapefile.file)
    gdf.to_file(geojson_path)

    num_cores_download, num_cores_lai = get_num_processing_cores(lai_config.resolution)

    # Prepare YAML configuration
    config_dict = {
        "date_ranges": [{"start_date": dr.start_date, "end_date": dr.end_date} for dr in lai_config.date_ranges],
        "resolution": lai_config.resolution,
        "geojson_path": geojson_path,
        "out_dir": out_dir,
        "region_out_prefix": region_out_prefix,
        "from_step": 0,
        "num_cores_download": num_cores_download,
        "num_cores_lai": num_cores_lai,
        "chunk_days": lai_config.chunk_days,
        "imagery_src": lai_config.imagery_src,
        "keep_imagery": lai_config.keep_imagery,
        "satellite": "S2",
    }

    executions_dir = os.path.join(out_dir, "executions")
    os.makedirs(executions_dir, exist_ok=True)
    suffix = (
        max(
            [int(f) for f in os.listdir(executions_dir) if os.path.isdir(os.path.join(executions_dir, f))],
            default=0,
        )
        + 1
    )
    execuction_dir = os.path.join(out_dir, "executions", str(suffix))
    os.makedirs(execuction_dir, exist_ok=True)

    config_path = os.path.join(execuction_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, sort_keys=False)

    run_lai_generation.delay(config_path)


@router.post("/{lai_id}/{resolution}/actions/regenerate")
def regenerate_lai(lai_id: str, resolution: int):
    lai_entry_dir = os.path.join(lai_dir, lai_id)

    executions_dir = os.path.join(lai_entry_dir, "executions")
    last_suffix = max(
        [int(f) for f in os.listdir(executions_dir) if os.path.isdir(os.path.join(executions_dir, f))],
        default=0,
    )
    last_config_path = os.path.join(executions_dir, str(last_suffix), "config.yaml")

    # validate that it has the exected resolution and was failed otherwise we have an edge case caused by multiple entries to this.
    # this needs refactoring so that this cant occur

    with open(last_config_path, "r") as f:
        config = yaml.safe_load(f)

    if not config["resolution"] == resolution:
        raise HTTPException(
            status_code=500,
            detail="Unexpected last execution for LAI detected - resolution not matching. Create new LAI entry.",
        )

    suffix = last_suffix + 1
    execuction_dir = os.path.join(lai_entry_dir, "executions", str(suffix))
    os.makedirs(execuction_dir, exist_ok=True)
    new_config_path = os.path.join(execuction_dir, "config.yaml")
    shutil.copy(last_config_path, new_config_path)

    run_lai_generation.delay(new_config_path)


@router.get("/{lai_id}/logs")
def get_logs(lai_id: str):
    lai_entry_dir = os.path.join(lai_dir, lai_id)
    executions_dir = os.path.join(lai_entry_dir, "executions")
    suffix = max(
        [int(f) for f in os.listdir(executions_dir) if os.path.isdir(os.path.join(executions_dir, f))],
        default=0,
    )
    execuction_dir = os.path.join(executions_dir, str(suffix))
    logs_path = os.path.join(execuction_dir, "log.txt")

    if not os.path.exists(logs_path):
        raise HTTPException(status_code=404, detail="Logs file not found")

    with open(logs_path, "r", encoding="utf-8", errors="replace") as f:
        last_lines = deque(f, maxlen=1000)

    text = "SHOWING LAST 1000 lines only: \n\n...\n"
    text += "".join(last_lines)

    return PlainTextResponse(text, media_type="text/plain; charset=utf-8")


@router.post("/{lai_id}/{resolution}/actions/cancel")
def cancel_generation(lai_id: str, resolution: int, force: bool = False):
    """Cancel a running LAI generation process.

    Sends SIGINT for graceful shutdown, then escalates to SIGKILL after 10 seconds
    if the process is still running. Use force=True for immediate SIGKILL.
    """
    lai_entry_dir = os.path.join(lai_dir, lai_id)
    executions_dir = os.path.join(lai_entry_dir, "executions")
    suffix = max(
        [int(f) for f in os.listdir(executions_dir) if os.path.isdir(os.path.join(executions_dir, f))],
        default=0,
    )
    execuction_dir = os.path.join(executions_dir, str(suffix))
    task_id_file = os.path.join(execuction_dir, "task_id.txt")

    metadata_path = os.path.join(lai_dir, lai_id, "meta.json")

    if not os.path.exists(task_id_file):
        raise HTTPException(status_code=404, detail="LAI generation does not have an associated task.")

    with open(task_id_file, "r") as f:
        pgid = int(f.read().strip())

    # Check if process group is still running
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        # Already terminated — just update status
        pass
    else:
        try:
            if force:
                os.killpg(pgid, signal.SIGKILL)
            else:
                # Graceful: SIGINT then wait up to 10s, escalate to SIGKILL
                os.killpg(pgid, signal.SIGINT)
                import time

                for _ in range(100):
                    time.sleep(0.1)
                    try:
                        os.killpg(pgid, 0)
                    except ProcessLookupError:
                        break
                else:
                    os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # Exited during shutdown
        except Exception as e:
            print(f"Error killing LAI process group {pgid}: {e}")

    # Update status
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
        metadata["status"][str(resolution)] = "cancelled"

    with open(metadata_path, "w") as file:
        json.dump(metadata, file)

    return {"status": "cancelled"}
