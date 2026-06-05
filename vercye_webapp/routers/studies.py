import asyncio
import io
import json
import os
import shutil
import signal
import tempfile
import threading
import time
import zipfile
from collections import deque
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiofiles
import geopandas as gpd
import pandas as pd
import yaml
from celery.result import AsyncResult
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from models import (
    AggregationShapefileConfig,
    AggregationShapefileConfigWithColumns,
    DuplicateStudyRequest,
    LAIConfigRunParams,
    RegionExtractionResponse,
    RunConfigFormParams,
    RunID,
    RunSummary,
    SetupConfigTemplate,
    SetupSubmissionsRequest,
    ShapefileColumnInfo,
    ShapefileData,
    StudyCreateRequest,
    StudyID,
    WindowConfig,
    WindowNoId,
)
from pydantic import TypeAdapter, ValidationError
from pyproj import CRS
from utils import QuotedString
from worker import duplicate_vercye_study_task, run_vercye_task, setup_vercye_task

from vercye_ops.cli import init_study, validate_run_config
from vercye_ops.utils.env_utils import (
    get_run_config,
    get_run_config_file_path,
    get_setup_config,
    get_setup_config_file_path,
    get_study_path,
    load_yaml_ruamel,
    read_cropmasks_dir_from_env,
    read_lai_dir_from_env,
    read_studies_dir_from_env,
)
from vercye_ops.utils.init_logger import get_logger

logger = get_logger()
studies_dir = read_studies_dir_from_env()
lai_dir = read_lai_dir_from_env()
cropmasks_dir = read_cropmasks_dir_from_env()

# Cache for storing study status to deal with fast responses in status updates
status_cache: Dict[str, Tuple[float, str]] = {}


def _column_is_numeric(series) -> bool:
    """Numeric iff every non-null value parses as a number (robust to object dtype from
    null/string-typed columns, unlike a plain dtype.kind check)."""
    if series.dtype.kind in ("i", "f"):
        return True
    coerced = pd.to_numeric(series, errors="coerce")
    non_null = series.notna()
    has_bad = bool((coerced.isna() & non_null).any())
    return (not has_bad) and bool(coerced.notna().any())


router = APIRouter(
    prefix="/studies",
    tags=["studies"],
)


# Routes
@router.post("")
def register_new_study(request: StudyCreateRequest):
    """Register a new study and create a directory containing templates for the configs"""
    study_id = request.study_id
    try:
        init_study(study_name=study_id, studies_dir=studies_dir)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Error initializing a study: {str(e)}.")


@router.post("/{existing_study_id}/duplicate")
async def register_from_existing(existing_study_id: str, body: DuplicateStudyRequest):

    if os.path.exists(get_study_path(studies_dir, body.new_study_id)):
        raise HTTPException(
            status_code=409, detail="A study with this id already exists. Please choose a different id."
        )

    task = duplicate_vercye_study_task.delay(existing_study_id, body.new_study_id)

    # Poll celery task status with timeout
    max_wait_seconds = 300
    elapsed = 0
    while elapsed < max_wait_seconds:
        result = AsyncResult(task.id)
        if result.ready():
            if result.successful():
                return {"status": "success", "new_study_id": body.new_study_id}
            else:
                raise HTTPException(status_code=500, detail=str(result.result))
        await asyncio.sleep(2)
        elapsed += 2

    raise HTTPException(
        status_code=504, detail="Study duplication timed out. The task may still be running in the background."
    )


# TODO: This endpoint has gotten a bit messey especially it is hard to follow
# What related to duplication handling, to existing config modification and general setup.
# This needs to be cleaned up to remove ambiguity
# Also not super happy with the current empty file placeholder approach - should look into this again
@router.post("/{study_id}/setup")
async def setup_study(
    study_id: StudyID,
    setup_data: str = Form(...),
    shapefile: UploadFile = File(...),
    apsim_files: List[UploadFile] = File(...),
    aggregation_shapefiles: Optional[List[UploadFile]] = File(None),
):
    try:
        setup_dict = json.loads(setup_data)
        setup_submission = SetupSubmissionsRequest(**setup_dict)
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid setup data: {e}")

    setup_cfg = {}

    try:
        target_crs = CRS.from_user_input(setup_submission.region_extraction.target_projection)
        target_crs_string = target_crs.to_string()
        if "proj" in target_crs_string:
            setup_cfg["target_crs"] = QuotedString(f"'{target_crs_string}'")
        else:
            setup_cfg["target_crs"] = QuotedString(target_crs_string)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid projection specified: {str(e)}")

    # Upload files
    study_root = Path(studies_dir) / str(study_id)
    shapefile_dir = study_root / "shapefile"
    apsim_dir = study_root / "apsim"
    for d in (shapefile_dir, apsim_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not shapefile:
        raise HTTPException(status_code=400, detail="No shapefile provided.")

    data = await shapefile.read()

    # If data is empty, it was most likely a placeholder used for an already existing file on the server
    current_setup_config = get_setup_config(studies_dir, study_id)
    if data:
        # Clear shapefile dir, as it might contain data from a template
        shutil.rmtree(shapefile_dir)
        os.makedirs(shapefile_dir)

        filename_lower = (shapefile.filename or "").lower()

        if filename_lower.endswith(".geojson") or filename_lower.endswith(".json"):
            # GeoJSON file - save directly
            geojson_path = shapefile_dir / shapefile.filename
            async with aiofiles.open(geojson_path, "wb") as out:
                await out.write(data)
            shapefile_path = str(geojson_path)

        elif filename_lower.endswith(".zip"):
            # Zipped shapefile - extract
            is_zip = await run_in_threadpool(zipfile.is_zipfile, io.BytesIO(data))
            if not is_zip:
                raise HTTPException(status_code=400, detail="Not a valid ZIP")

            await run_in_threadpool(lambda: zipfile.ZipFile(io.BytesIO(data)).extractall(shapefile_dir))

            shp_files = list(shapefile_dir.glob("*.shp"))
            if len(shp_files) != 1:
                raise HTTPException(status_code=400, detail=f"Expected exactly one .shp file, found {len(shp_files)}")
            shapefile_path = str(shp_files[0])

        else:
            raise HTTPException(
                status_code=400,
                detail="Primary regions file must be a .zip (shapefile) or .geojson",
            )
    else:
        # Check if empty file is a placeholder by looking up if true filename exists
        if not Path(current_setup_config["regions_shp_name"]).name == shapefile.filename:
            raise HTTPException(status_code=400, detail="Empty shapefile provided!")
        shapefile_path = current_setup_config["regions_shp_name"]

    # Save apsim files and reference data files
    apsim_paths_to_modify = {}

    async def save_apsim_files(files, dest_dir):
        all_available_files = {
            Path(apsim_path).name: apsim_path for apsim_path in current_setup_config["APSIM_TEMPLATE_PATHS"].values()
        }
        if not files:
            return
        for uf in files:
            if not uf.filename:
                continue
            content = await uf.read()

            # Check if placeholder files were used - empty file = file from template
            if not content:
                if uf.filename not in all_available_files:
                    raise HTTPException(status_code=400, detail="Empty APSIM File provided!")

                apsim_paths_to_modify[uf.filename] = all_available_files[uf.filename]
            else:
                async with aiofiles.open(dest_dir / uf.filename, "wb") as out:
                    await out.write(content)

    await save_apsim_files(apsim_files, apsim_dir)

    setup_cfg["regions_shp_name"] = shapefile_path
    setup_cfg["regions_shp_col"] = setup_submission.region_extraction.admin_name_column

    if setup_submission.region_extraction.filter:
        setup_cfg["regions_shp_filter_col"] = setup_submission.region_extraction.filter.column
        setup_cfg["regions_shp_filter_values"] = setup_submission.region_extraction.filter.allow
    else:
        setup_cfg["regions_shp_filter_col"] = None
        setup_cfg["regions_shp_filter_values"] = None

    def make_tp_configs(simulation_windows: List[WindowConfig]) -> Dict:
        """Convert simulation windows into the nested timepoints_config structure"""
        timepoints_config = {}

        for window in simulation_windows:
            year = int(window.year)
            timepoint = window.timepoint

            if year not in timepoints_config:
                timepoints_config[year] = {}

            timepoints_config[year][timepoint] = {
                "sim_start_date": window.sim_start_date,
                "sim_end_date": window.sim_end_date,
                "met_start_date": window.met_start_date,
                "met_end_date": window.met_end_date,
                "lai_start_date": window.lai_start_date,
                "lai_end_date": window.lai_end_date,
            }

        return timepoints_config

    setup_cfg["years"] = [int(year) for year in setup_submission.years]
    setup_cfg["timepoints"] = setup_submission.timepoints
    setup_cfg["timepoints_config"] = make_tp_configs(setup_submission.simulation_windows)

    if setup_submission.apsim_column and setup_submission.apsim_column != "all":
        setup_cfg["APSIM_TEMPLATE_PATHS_FILTER_COL_NAME"] = setup_submission.apsim_column
    else:
        setup_cfg["APSIM_TEMPLATE_PATHS_FILTER_COL_NAME"] = None

    apsim_paths = {}
    for apsim_file_name, regions in setup_submission.apsim_mapping.items():
        for region in regions:
            apsim_path = apsim_paths_to_modify.get(apsim_file_name, str(apsim_dir / apsim_file_name))
            apsim_paths[region] = apsim_path
    setup_cfg["APSIM_TEMPLATE_PATHS"] = apsim_paths

    # Handle aggregation shapefiles
    agg_shp_dir = study_root / "aggregation_shapefiles"
    agg_shp_dir.mkdir(parents=True, exist_ok=True)

    agg_shp_config = {}

    # Check for real uploaded files (placeholder files from duplication have size 0)
    real_uploads = []
    if aggregation_shapefiles:
        for uf in aggregation_shapefiles:
            content = await uf.read()
            if content:
                real_uploads.append((uf, content))

    if real_uploads:
        agg_shp_configs_by_filename = {cfg.level_name: cfg for cfg in setup_submission.aggregation_shapefiles}

        for uf, content in real_uploads:
            if not uf.filename:
                continue

            filename_lower = uf.filename.lower()
            file_stem = Path(uf.filename).stem

            if filename_lower.endswith(".geojson") or filename_lower.endswith(".json"):
                # GeoJSON - save directly
                geojson_dest = agg_shp_dir / uf.filename
                async with aiofiles.open(geojson_dest, "wb") as out:
                    await out.write(content)
                resolved_path = str(geojson_dest)

            elif filename_lower.endswith(".zip"):
                # Zipped shapefile - extract
                is_zip = await run_in_threadpool(zipfile.is_zipfile, io.BytesIO(content))
                if not is_zip:
                    raise HTTPException(status_code=400, detail=f"Aggregation file {uf.filename} is not a valid ZIP")

                extract_dir = agg_shp_dir / file_stem
                extract_dir.mkdir(parents=True, exist_ok=True)
                await run_in_threadpool(lambda: zipfile.ZipFile(io.BytesIO(content)).extractall(extract_dir))

                shp_files = list(extract_dir.glob("*.shp"))
                if len(shp_files) != 1:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Expected exactly one .shp file in {uf.filename}, found {len(shp_files)}",
                    )
                resolved_path = str(shp_files[0])

            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Aggregation file {uf.filename} must be a .zip or .geojson",
                )

            # Find matching config by matching filename to level configs
            for level_name, cfg in agg_shp_configs_by_filename.items():
                if (
                    file_stem == level_name
                    or uf.filename.replace(".zip", "").replace(".geojson", "").replace(".json", "") == level_name
                ):
                    agg_shp_config[level_name] = {
                        "shapefile_path": resolved_path,
                        "name_column": cfg.name_column,
                        "reference_yield_column": cfg.reference_yield_column,
                        "year_column": cfg.year_column,
                    }
                    break
            else:
                # If no matching config found, try to match by order
                for level_name, cfg in agg_shp_configs_by_filename.items():
                    if level_name not in agg_shp_config:
                        agg_shp_config[level_name] = {
                            "shapefile_path": resolved_path,
                            "name_column": cfg.name_column,
                            "reference_yield_column": cfg.reference_yield_column,
                            "year_column": cfg.year_column,
                        }
                        break

    elif setup_submission.aggregation_shapefiles:
        # No new files uploaded but configs exist (e.g. duplicated study) -
        # preserve existing shapefile paths from the current study config
        existing_config_path = get_setup_config_file_path(studies_dir, study_id)
        if os.path.exists(existing_config_path):
            existing_config = get_setup_config(studies_dir, study_id)
            existing_agg = existing_config.get("AGGREGATION_SHAPEFILES", {})
        else:
            existing_agg = {}

        for cfg in setup_submission.aggregation_shapefiles:
            existing_level = existing_agg.get(cfg.level_name, {})
            existing_path = existing_level.get("shapefile_path", "")
            if existing_path and os.path.exists(existing_path):
                agg_shp_config[cfg.level_name] = {
                    "shapefile_path": existing_path,
                    "name_column": cfg.name_column,
                    "reference_yield_column": cfg.reference_yield_column,
                    "year_column": cfg.year_column,
                }

    setup_cfg["AGGREGATION_SHAPEFILES"] = agg_shp_config
    setup_cfg["creation_request_params"] = setup_submission.dict()

    setup_cfg_path = study_root / "setup_config.yaml"
    with setup_cfg_path.open("w") as f:
        yaml.dump(setup_cfg, f, default_flow_style=False, sort_keys=False)

    # Extract and prepare APSIM files and geojsons and create run config for user to fill in
    try:
        setup_vercye_task(study_id, studies_dir)
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{study_id}/setup-config")
def fetch_setup_config(study_id: StudyID):
    """Fetch setup config template for study"""
    setup_cfg_path = get_setup_config_file_path(studies_dir, study_id)

    if not os.path.exists(setup_cfg_path):
        raise HTTPException(status_code=404, detail="No study config template found!")

    config = get_setup_config(studies_dir, study_id)

    if "creation_request_params" not in config:
        return None

    raw = config["creation_request_params"]

    # region extraction
    region_cfg = raw.get("region_extraction", {})
    region_extraction = RegionExtractionResponse(
        adminNameColumn=str(region_cfg.get("admin_name_column", "")),
        targetProjection=str(region_cfg.get("target_projection", "")),
        filter=region_cfg.get("filter"),
    )

    # years & timepoints
    years = [str(y) for y in raw.get("years", [])]
    timepoints = [str(tp) for tp in raw.get("timepoints", [])]

    # windows
    windows = [WindowNoId(**w) for w in raw.get("simulation_windows", [])]

    # apsim
    apsim_mapping = raw.get("apsim_mapping", {})
    apsim_files = list(apsim_mapping.keys())
    apsim_column = str(raw.get("apsim_column", ""))

    # Region attribute table only. The frontend uses feature.properties to populate the
    # column/value dropdowns and never renders the geometry, so we skip reading geometry
    # entirely - this keeps the payload to a few hundred KB instead of several MB.
    try:
        regions_df = gpd.read_file(config["regions_shp_name"], ignore_geometry=True)
        for col in regions_df.select_dtypes(include=["datetime64[ns]"]).columns:
            regions_df[col] = regions_df[col].astype(str)

        records = regions_df.astype(object).where(pd.notna(regions_df), None).to_dict(orient="records")
        shapefile_geojson = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "geometry": None, "properties": rec} for rec in records],
        }
        shapefile_name = Path(config["regions_shp_name"]).name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read shapefile: {e}")

    # aggregation shapefiles - enrich with columns read from stored files
    agg_shp_raw = raw.get("aggregation_shapefiles", [])
    agg_shp_stored = config.get("AGGREGATION_SHAPEFILES", {})
    agg_shp_configs: list[AggregationShapefileConfigWithColumns] = []
    agg_shp_names: list[str] = []

    for s in agg_shp_raw or []:
        base = AggregationShapefileConfig(**s)
        columns: list[ShapefileColumnInfo] = []
        stored = agg_shp_stored.get(base.level_name, {})
        shp_path = stored.get("shapefile_path", "")
        if shp_path:
            agg_shp_names.append(Path(shp_path).name)
            try:
                # Only the attribute schema (column names/dtypes) is needed for the UI
                # dropdowns, so skip reading geometry.
                gdf_agg = gpd.read_file(shp_path, ignore_geometry=True)
                for col in gdf_agg.columns:
                    if col == "geometry":
                        continue
                    columns.append(
                        ShapefileColumnInfo(
                            name=col,
                            dtype=str(gdf_agg[col].dtype),
                            is_numeric=_column_is_numeric(gdf_agg[col]),
                        )
                    )
            except Exception:
                pass  # columns stay empty; user can re-upload
        agg_shp_configs.append(
            AggregationShapefileConfigWithColumns(
                **base.model_dump(),
                columns=columns,
            )
        )

    return SetupConfigTemplate(
        regionExtraction=region_extraction,
        apsimColumn=apsim_column,
        apsimMapping=apsim_mapping,
        apsimFiles=apsim_files,
        aggregationShapefiles=agg_shp_configs,
        aggregationShapefileNames=agg_shp_names,
        years=years,
        timepoints=timepoints,
        simulationWindows=windows,
        shapefileData=ShapefileData(**shapefile_geojson),
        shapefileName=shapefile_name,
    )


@router.post("/{study_id}/shapefile-columns")
async def get_shapefile_columns(
    study_id: StudyID,
    shapefile: UploadFile = File(...),
):
    """Extract column names and dtypes from an uploaded shapefile (.zip) or GeoJSON for UI dropdowns."""
    data = await shapefile.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file provided")

    filename_lower = (shapefile.filename or "").lower()

    if filename_lower.endswith(".geojson") or filename_lower.endswith(".json"):
        # GeoJSON - read directly from bytes
        with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            gdf = gpd.read_file(tmp_path)
        finally:
            os.unlink(tmp_path)

    elif filename_lower.endswith(".zip"):
        is_zip = await run_in_threadpool(zipfile.is_zipfile, io.BytesIO(data))
        if not is_zip:
            raise HTTPException(status_code=400, detail="Not a valid ZIP")

        with tempfile.TemporaryDirectory() as tmp_dir:
            await run_in_threadpool(lambda: zipfile.ZipFile(io.BytesIO(data)).extractall(tmp_dir))
            shp_files = list(Path(tmp_dir).glob("*.shp"))
            if len(shp_files) != 1:
                raise HTTPException(status_code=400, detail=f"Expected exactly one .shp file, found {len(shp_files)}")
            gdf = gpd.read_file(str(shp_files[0]))
    else:
        raise HTTPException(status_code=400, detail="File must be a .zip (shapefile) or .geojson")

    columns = []
    for col in gdf.columns:
        if col == "geometry":
            continue
        dtype = str(gdf[col].dtype)
        is_numeric = _column_is_numeric(gdf[col])
        columns.append({"name": col, "dtype": dtype, "is_numeric": is_numeric})

    return {"columns": columns}


@router.post("/{study_id}/run-config")
async def upload_run_config(
    study_id: str,
    run_cfg_file: UploadFile = File(...),
    cropmask_mapping: str = Form(None),  # JSON string
    lai_config: Optional[str] = Form(None),  # JSON string (optional)
):
    if not run_cfg_file.filename.endswith((".yaml", ".yml")):
        raise HTTPException(status_code=400, detail="Only YAML files are accepted")

    run_cfg_status_path = os.path.join(studies_dir, study_id, study_id, "config_status.txt")
    run_cfg_status_details_path = os.path.join(studies_dir, study_id, study_id, "config_status_details.txt")

    try:
        lai_config = json.loads(lai_config)
        lai_config = LAIConfigRunParams(**lai_config)
    except (json.JSONDecodeError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid laiconfig data: {e}")

    tmp_file_path = None
    try:
        file_bytes = run_cfg_file.file.read()
        config_data, yaml_ruamel = load_yaml_ruamel(io.BytesIO(file_bytes))

        # Set simulation head dir
        config_data["sim_study_head_dir"] = os.path.join(studies_dir, study_id, study_id)

        # Update lai related info only if lai_config provided
        if lai_config:
            config_data["lai_params"]["lai_dir"] = os.path.join(lai_dir, lai_config.lai_source_id, "merged-lai")
            lai_region = "_".join(os.listdir(config_data["lai_params"]["lai_dir"])[0].split("_")[:-3])

            config_data["lai_params"]["lai_region"] = lai_region
            config_data["lai_params"]["lai_resolution"] = int(lai_config.lai_source_resolution)
            config_data["lai_source"] = str(lai_config.lai_source_id)
        try:
            raw_mapping = json.loads(cropmask_mapping)
            # Coerce & validate keys as ints, values as str
            ta = TypeAdapter(Dict[int, str])
            parsed_cropmask = ta.validate_python(raw_mapping)
        except (json.JSONDecodeError, ValidationError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid cropmask_mapping: {e}")

        # TODO validate cropmasks exist
        parsed_cropmask = {year: os.path.join(cropmasks_dir, cropmask) for year, cropmask in parsed_cropmask.items()}
        config_data["lai_params"]["crop_mask"] = parsed_cropmask

        # Save updated YAML to tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w") as tmp_file:
            print(config_data)
            yaml_ruamel.dump(config_data, tmp_file)
            tmp_file_path = tmp_file.name

        validate_run_config(tmp_file_path)
    except Exception as e:

        with open(run_cfg_status_path, "w") as f:
            f.write("invalid")

        with open(run_cfg_status_details_path, "w") as f:
            f.write(str(e))

        raise HTTPException(status_code=400, detail=f"Invalid config: {str(e)}")
    finally:
        run_cfg_file.file.close()
        if tmp_file_path:
            run_cfg_path = os.path.join(studies_dir, study_id, study_id, "config.yaml")
            shutil.copyfile(tmp_file_path, run_cfg_path)
            os.remove(tmp_file_path)

    with open(run_cfg_status_path, "w") as f:
        f.write("valid")

    with open(run_cfg_status_details_path, "w") as f:
        f.write("OK!")

    return {"message": "Run config set successfully"}


@router.get("/{study_id}/run-config")
def get_run_config_file(study_id: StudyID):
    run_cfg_path = os.path.join(studies_dir, study_id, study_id, "config.yaml")
    if not os.path.isfile(run_cfg_path):
        raise HTTPException(status_code=404, detail="Run config not found")
    return FileResponse(run_cfg_path, filename="run_config.yaml")


@router.get("/{study_id}/run-config-formdata")
def get_run_config_formdata(study_id: StudyID):
    """Returns previous lai selection - cropmasks to display when reloading the form in the frontend."""
    if not os.path.exists(get_run_config_file_path(studies_dir, study_id)):
        return None

    # Check if its still the template or an actual already prefilled value to be returned
    run_config = get_run_config(
        studies_dir,
        study_id,
    )
    if run_config["lai_params"]["lai_region"] == "XXXX":
        return None

    lai_id = run_config["lai_params"]["lai_region"]
    lai_resolution = run_config["lai_params"]["lai_resolution"]

    cropmasks = {str(year): Path(mask_path).name for year, mask_path in run_config["lai_params"]["crop_mask"].items()}

    return RunConfigFormParams(laiId=lai_id, laiResolution=lai_resolution, cropmasks=cropmasks)


@router.get("/{study_id}/run-config-status")
def get_run_config_status(study_id: StudyID):
    run_cfg_status_path = os.path.join(studies_dir, study_id, study_id, "config_status.txt")
    run_cfg_status_details_path = os.path.join(studies_dir, study_id, study_id, "config_status_details.txt")

    if not os.path.isfile(run_cfg_status_path):
        raise HTTPException(status_code=404, detail="Run config not found")

    with open(run_cfg_status_path) as f:
        status = f.read()

    if not os.path.isfile(run_cfg_status_details_path):
        status_details = "No details available"
    else:
        with open(run_cfg_status_details_path) as f:
            status_details = f.read()

    return {"status": status, "details": status_details}


@router.get("/{study_id}/actions/validate-runconfig")
def do_runconfig_validation(study_id: StudyID):
    run_cfg_path = os.path.join(studies_dir, study_id, study_id, "config.yaml")
    if not os.path.isfile(run_cfg_path):
        raise HTTPException(status_code=404, detail="Run config not found")

    run_cfg_status_path = os.path.join(studies_dir, study_id, study_id, "config_status.txt")
    run_cfg_status_details_path = os.path.join(studies_dir, study_id, study_id, "config_status_details.txt")

    try:
        validate_run_config(run_cfg_path)
    except Exception as e:

        with open(run_cfg_status_path, "w") as f:
            f.write("invalid")

        with open(run_cfg_status_details_path, "w") as f:
            f.write(str(e))

        return {"status": "invalid", "detail": f"{e}"}

    with open(run_cfg_status_path, "w") as f:
        f.write("valid")

    with open(run_cfg_status_details_path, "w") as f:
        f.write("OK!")

    return {"status": "valid", "detail": ""}


@router.get("/{study_id}/required-years")
def get_required_years(study_id: StudyID):
    study_config_path = os.path.join(studies_dir, study_id, "setup_config.yaml")
    if not os.path.exists(study_config_path):
        raise HTTPException(status_code=404, detail="No setup config found for study.")

    with open(study_config_path, "r") as f:
        config = yaml.safe_load(f)

    years = config.get("years", [])
    return years


@router.post("/{study_id}/actions/run")
def run_study(study_id: StudyID, force_rerun: bool = Query(False, alias="forceRerun")):
    try:
        task = run_vercye_task.delay(study_id, force_rerun)
        task_id = task.id

        os.makedirs(os.path.join(studies_dir, study_id, "snakemake"), exist_ok=True)
        status_file_path = os.path.join(studies_dir, study_id, "snakemake", "status.txt")
        with open(status_file_path, "w") as f:
            f.write("queued")

        # Save worker celery task_id for later reference e.g for cancelling a job.
        task_id_file = os.path.join(studies_dir, study_id, "snakemake", "celery_task_id.txt")
        with open(task_id_file, "w") as f:
            f.write(task_id)

        return {"message": "Study run started in background."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


CANCEL_ESCALATION_TIMEOUT_SECONDS = 120


def _escalate_cancel(pgid: int, status_file_path: str, timeout: int = CANCEL_ESCALATION_TIMEOUT_SECONDS):
    """Background thread: wait for graceful shutdown, escalate to SIGKILL if needed.

    After sending SIGINT, snakemake should finish running jobs and exit. If it hasn't
    terminated after `timeout` seconds, we escalate to SIGKILL to ensure the process
    group is fully stopped.
    """
    time.sleep(timeout)
    try:
        os.killpg(pgid, 0)  # Check if still alive
    except ProcessLookupError:
        return  # Already exited cleanly

    logger.warning(f"Process group {pgid} still running after {timeout}s, escalating to SIGKILL")
    try:
        os.killpg(pgid, signal.SIGKILL)
        with open(status_file_path, "w") as f:
            f.write("cancelled")
    except ProcessLookupError:
        pass  # Exited between check and kill
    except Exception as e:
        logger.error(f"Failed to escalate kill for process group {pgid}: {e}")


@router.post("/{study_id}/actions/cancel")
def cancel_study(
    study_id: StudyID, force: bool = Query(False, description="Force kill with SIGKILL instead of graceful SIGINT")
):
    """Cancel a running snakemake study.

    By default sends SIGINT to the process group which triggers a graceful shutdown:
    - Snakemake stops accepting new jobs
    - Running jobs are allowed to complete
    - Snakemake cleans up and exits

    This mimics pressing Ctrl+C in the terminal. If snakemake does not exit within
    2 minutes, the signal is automatically escalated to SIGKILL.

    If force=True, sends SIGKILL immediately which terminates all processes
    without cleanup. Use this only if you need instant termination.
    """
    task_id_file = os.path.join(studies_dir, study_id, "snakemake", "snakemake_task_id.txt")
    status_file_path = os.path.join(studies_dir, study_id, "snakemake", "status.txt")

    if not os.path.exists(task_id_file):
        raise HTTPException(status_code=404, detail="Study does not have an associated task.")

    with open(task_id_file, "r") as f:
        pgid = int(f.read().strip())

    # Check if process group is still running
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        logger.info(f"Process group {pgid} for study {study_id} is not running")
        return {"status": "not_running", "detail": "The study process is no longer running."}
    except PermissionError:
        pass

    try:
        if force:
            os.killpg(pgid, signal.SIGKILL)
            logger.info(f"Sent SIGKILL to process group {pgid} for study {study_id}")
            status = "cancelled"
        else:
            os.killpg(pgid, signal.SIGINT)
            logger.info(f"Sent SIGINT to process group {pgid} for study {study_id}")
            status = "cancelling"

            # Spawn background thread to escalate to SIGKILL if snakemake doesn't exit
            escalation_thread = threading.Thread(
                target=_escalate_cancel,
                args=(pgid, status_file_path),
                daemon=True,
            )
            escalation_thread.start()
    except ProcessLookupError:
        logger.info(f"Process group {pgid} already terminated")
        return {"status": "not_running", "detail": "The study process terminated before signal could be sent."}
    except Exception as e:
        logger.error(f"Error sending signal to process group {pgid}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel study: {e}")

    with open(status_file_path, "w") as f:
        f.write(status)

    return {"status": status, "signal": "SIGKILL" if force else "SIGINT"}


@router.get("/{study_id}/result-timepoints")
def get_result_timepoints(study_id: StudyID):
    study_dir = os.path.join(studies_dir, study_id, study_id)
    years = {f: [] for f in os.listdir(study_dir) if os.path.isdir(os.path.join(study_dir, f)) and f.isdigit()}
    for year in years:
        year_timepoints = [
            f for f in os.listdir(os.path.join(study_dir, year)) if os.path.isdir(os.path.join(study_dir, year, f))
        ]
        years[year] = year_timepoints

    return {"timepoints": years}


@router.get("/{study_id}/map-result/{year}/{timepoint}/{resource}")
def get_map_resource(study_id: StudyID, year: int, timepoint: str, resource: str):
    base_path = (
        Path(studies_dir) / study_id / "snakemake" / "result_maps" / str(year) / str(timepoint) / "interactive_map"
    )
    file_path = base_path / resource

    try:
        resolved_path = file_path.resolve(strict=False)
        # Prevent traversal attacks
        if not resolved_path.is_file() or not str(resolved_path).startswith(str(base_path.resolve())):
            raise HTTPException(status_code=403, detail="Invalid resource path.")
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid resource path.")

    return FileResponse(resolved_path)


@router.get("/{study_id}/report/{year}/{timepoint}")
def get_report(study_id: StudyID, year: int, timepoint: str):
    report_dir = os.path.join(studies_dir, study_id, study_id, str(year), timepoint)

    report_canidates = glob(os.path.join(report_dir, "final_report_*.pdf"))

    if len(report_canidates) != 1:
        raise HTTPException(
            status_code=400,
            detail=f"Expected to find one final report, found {len(report_canidates)}",
        )

    report_path = report_canidates[0]

    if not os.path.isfile(report_path):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(report_path, filename=Path(report_path).name)


@router.get("/{study_id}/map-result/{year}/{timepoint}")
def get_map_results(study_id: StudyID, year: int, timepoint: str):
    map_zip_candidates = glob(
        os.path.join(studies_dir, study_id, study_id, str(year), str(timepoint), "interactive_map_*.zip")
    )
    if len(map_zip_candidates) != 1:
        raise HTTPException(
            status_code=500,
            detail=f"Excepted to find a single matching entry containing interactive_map_*.zip. Found {len(map_zip_candidates)}",
        )

    target_dir = (
        Path(studies_dir) / study_id / "snakemake" / "result_maps" / str(year) / str(timepoint) / "interactive_map"
    )
    with zipfile.ZipFile(map_zip_candidates[0]) as z:
        z.extractall(target_dir)

    map_path = os.path.join(target_dir, "vercye_results_map.html")
    if not os.path.exists(map_path):
        raise HTTPException(status_code=404, detail="No result map avilable.")

    with open(map_path, "r") as f:
        content = f.read()
        new_path_base = f"/api/studies/{study_id}/map-result/{year}/{timepoint}"
        # Replace imagery relative path
        content = content.replace(
            "img.src = props.simulationsImgPath;",
            f"img.src = `{new_path_base}/${{props.simulationsImgPath}}`",
        )
        return HTMLResponse(content=content)


@router.get("/{study_id}/multiyear-report/assets/{asset_path:path}")
def get_multiyear_report_asset(study_id: StudyID, asset_path: str):
    base_path = Path(studies_dir) / study_id / "snakemake" / "multiyear_report" / "assets"
    file_path = (base_path / asset_path).resolve(strict=False)

    try:
        if not file_path.is_file() or not str(file_path).startswith(str(base_path.resolve())):
            raise HTTPException(status_code=403, detail="Invalid asset path.")
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid asset path.")

    return FileResponse(file_path)


@router.get("/{study_id}/multiyear-report")
def get_multiyear_report(study_id: StudyID):
    report_path_candidates = glob(
        os.path.join(get_study_path(studies_dir, study_id), study_id, "multiyear_summary_*.zip")
    )
    if len(report_path_candidates) != 1:
        raise HTTPException(
            status_code=400,
            detail=f"Found {len(report_path_candidates)} entries with multiyear_summary_ in the name in study folder.",
        )

    target_dir = Path(studies_dir) / study_id / "snakemake" / "multiyear_report"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    with zipfile.ZipFile(report_path_candidates[0]) as z:
        z.extractall(target_dir)

    report_path = target_dir / "report.html"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="No multiyear report available.")

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    assets_base = f"/api/studies/{study_id}/multiyear-report/assets"
    content = content.replace('src="assets/', f'src="{assets_base}/')
    content = content.replace("src='assets/", f"src='{assets_base}/")

    return HTMLResponse(content=content)


@router.get("/{study_id}/log")
def get_study_log(study_id: StudyID):
    log_path = os.path.join(studies_dir, study_id, "snakemake", "log.txt")
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="No log available. Check individual rules logs if necessary.")

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        last_lines = deque(f, maxlen=1000)

    text = "SHOWING LAST 1000 lines only: \n\n...\n"
    text += "".join(last_lines)
    return PlainTextResponse(text, media_type="text/plain; charset=utf-8")


@router.get("/{study_id}/full-log")
def get_complete_study_log(study_id: StudyID):
    log_path = os.path.join(studies_dir, study_id, "snakemake", "log.txt")
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="No log available. Check individual rules logs if necessary.")

    return FileResponse(path=log_path, media_type="text/plain", filename=f"{study_id}_log.txt")


@router.get("/status")
async def get_many_status(ids: List[str] = Query(..., description="Repeated ?ids=studyA&ids=studyB")):
    """Batch status fetch for multiple studies."""

    async def read_status(study_id: str):
        status_file_path = os.path.join(studies_dir, study_id, "snakemake", "status.txt")
        exists = await asyncio.to_thread(os.path.exists, status_file_path)
        if not exists:
            return study_id, "pending"
        mtime = await asyncio.to_thread(lambda: os.path.getmtime(status_file_path))
        if study_id in status_cache and status_cache[study_id][0] == mtime:
            return study_id, status_cache[study_id][1]
        async with aiofiles.open(status_file_path, mode="r") as f:
            contents = await f.read()
        status_cache[study_id] = (mtime, contents)
        return study_id, contents

    pairs = await asyncio.gather(*(read_status(sid) for sid in ids))
    return {k: v for k, v in pairs}


@router.get("")
def get_all_studies(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=200)):
    """Paginated list of studies, newest first."""
    if not os.path.exists(studies_dir):
        return {"items": [], "total": 0, "page": page, "page_size": page_size}

    studies = [name for name in os.listdir(studies_dir) if os.path.isdir(os.path.join(studies_dir, name))]
    studies.sort(key=lambda name: os.path.getctime(os.path.join(studies_dir, name)), reverse=True)

    total = len(studies)
    start = (page - 1) * page_size
    end = start + page_size
    items = studies[start:end]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/combined-stats")
def get_combined_stats():
    """Compute the stats from multiple studies combined and with all years present in each study"""
    pass


@router.delete("/{study_id}")
def delete_study(study_id: StudyID):
    """Delete a study directory."""
    path = os.path.join(studies_dir, study_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Study not found.")

    # Safety: avoid deleting while a run might be active
    status_file_path = os.path.join(path, "snakemake", "status.txt")
    running_like = {"running", "queued"}
    if os.path.exists(status_file_path):
        try:
            with open(status_file_path) as f:
                current = f.read().strip()
            if current in running_like:
                raise HTTPException(status_code=409, detail="Study is currently running; cancel it before deleting.")
        except Exception:
            pass

    shutil.rmtree(path, ignore_errors=False)
    # also invalidate cache
    status_cache.pop(study_id, None)
    return {"status": "deleted", "study_id": study_id}


# Run snapshots related

def _run_results_root(study_id: str) -> Path:
    return Path(studies_dir) / study_id / study_id / "run_results"


def _run_dir(study_id: str, run_id: str) -> Path:
    return _run_results_root(study_id) / run_id


def _ensure_run_dir(study_id: str, run_id: str) -> Path:
    run_dir = _run_dir(study_id, run_id)
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found.")
    # Defensive path containment check.
    try:
        run_dir.resolve(strict=True).relative_to(_run_results_root(study_id).resolve())
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid run path.")
    return run_dir


def _scan_run_timepoints(run_dir: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for year_entry in sorted(run_dir.iterdir()):
        if not year_entry.is_dir() or not year_entry.name.isdigit():
            continue
        timepoints = sorted(tp.name for tp in year_entry.iterdir() if tp.is_dir())
        out[year_entry.name] = timepoints
    return out


def _build_run_summary(study_id: str, run_dir: Path) -> RunSummary:
    meta_path = run_dir / "_run_meta.json"
    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}

    timepoints = _scan_run_timepoints(run_dir)
    multiyear_zip = next(run_dir.glob("multiyear_summary_*.zip"), None)

    size_bytes = 0
    for dirpath, _dirnames, filenames in os.walk(run_dir):
        for name in filenames:
            try:
                size_bytes += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass

    created_at = meta.get("created_at")
    if not created_at:
        try:
            from datetime import datetime as _dt

            created_at = _dt.fromtimestamp(run_dir.stat().st_mtime).astimezone().isoformat(timespec="seconds")
        except Exception:
            created_at = None

    return RunSummary(
        run_id=run_dir.name,
        created_at=created_at,
        file_count=meta.get("file_count"),
        uploaded_to=meta.get("uploaded_to"),
        has_multiyear_report=multiyear_zip is not None,
        timepoints=timepoints,
        size_bytes=size_bytes or None,
    )


@router.get("/{study_id}/runs")
def list_runs(study_id: StudyID):
    """List snapshotted runs for a study, newest first."""
    root = _run_results_root(study_id)
    if not root.is_dir():
        return {"items": []}

    entries = [p for p in root.iterdir() if p.is_dir()]
    entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return {"items": [_build_run_summary(study_id, p) for p in entries]}


@router.get("/{study_id}/runs/{run_id}/result-timepoints")
def get_run_result_timepoints(study_id: StudyID, run_id: RunID):
    run_dir = _ensure_run_dir(study_id, run_id)
    return {"timepoints": _scan_run_timepoints(run_dir)}


@router.get("/{study_id}/runs/{run_id}/report/{year}/{timepoint}")
def get_run_report(study_id: StudyID, run_id: RunID, year: int, timepoint: str):
    run_dir = _ensure_run_dir(study_id, run_id)
    report_candidates = list((run_dir / str(year) / timepoint).glob("final_report_*.pdf"))
    if len(report_candidates) != 1:
        raise HTTPException(
            status_code=404,
            detail=f"Expected to find one final report, found {len(report_candidates)}",
        )
    return FileResponse(report_candidates[0], filename=report_candidates[0].name)


@router.get("/{study_id}/runs/{run_id}/map-result/{year}/{timepoint}/{resource}")
def get_run_map_resource(study_id: StudyID, run_id: RunID, year: int, timepoint: str, resource: str):
    base_path = (
        Path(studies_dir)
        / study_id
        / "snakemake"
        / "run_maps"
        / run_id
        / str(year)
        / str(timepoint)
        / "interactive_map"
    )
    file_path = base_path / resource
    try:
        resolved_path = file_path.resolve(strict=False)
        if not resolved_path.is_file() or not str(resolved_path).startswith(str(base_path.resolve())):
            raise HTTPException(status_code=403, detail="Invalid resource path.")
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid resource path.")
    return FileResponse(resolved_path)


@router.get("/{study_id}/runs/{run_id}/map-result/{year}/{timepoint}")
def get_run_map_result(study_id: StudyID, run_id: RunID, year: int, timepoint: str):
    run_dir = _ensure_run_dir(study_id, run_id)
    map_zip_candidates = list((run_dir / str(year) / str(timepoint)).glob("interactive_map_*.zip"))
    if len(map_zip_candidates) != 1:
        raise HTTPException(
            status_code=404,
            detail=f"Expected to find a single interactive_map_*.zip, found {len(map_zip_candidates)}",
        )

    target_dir = (
        Path(studies_dir)
        / study_id
        / "snakemake"
        / "run_maps"
        / run_id
        / str(year)
        / str(timepoint)
        / "interactive_map"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(map_zip_candidates[0]) as z:
        z.extractall(target_dir)

    map_path = target_dir / "vercye_results_map.html"
    if not map_path.exists():
        raise HTTPException(status_code=404, detail="No result map available.")

    with open(map_path, "r") as f:
        content = f.read()
        new_path_base = f"/api/studies/{study_id}/runs/{run_id}/map-result/{year}/{timepoint}"
        content = content.replace(
            "img.src = props.simulationsImgPath;",
            f"img.src = `{new_path_base}/${{props.simulationsImgPath}}`",
        )
        return HTMLResponse(content=content)


@router.get("/{study_id}/runs/{run_id}/multiyear-report/assets/{asset_path:path}")
def get_run_multiyear_report_asset(study_id: StudyID, run_id: RunID, asset_path: str):
    base_path = (
        Path(studies_dir) / study_id / "snakemake" / "run_multiyear_reports" / run_id / "assets"
    )
    file_path = (base_path / asset_path).resolve(strict=False)
    try:
        if not file_path.is_file() or not str(file_path).startswith(str(base_path.resolve())):
            raise HTTPException(status_code=403, detail="Invalid asset path.")
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid asset path.")
    return FileResponse(file_path)


@router.get("/{study_id}/runs/{run_id}/multiyear-report")
def get_run_multiyear_report(study_id: StudyID, run_id: RunID):
    run_dir = _ensure_run_dir(study_id, run_id)
    report_candidates = list(run_dir.glob("multiyear_summary_*.zip"))
    if len(report_candidates) != 1:
        raise HTTPException(
            status_code=404,
            detail=f"Found {len(report_candidates)} entries with multiyear_summary_ in the run folder.",
        )

    target_dir = Path(studies_dir) / study_id / "snakemake" / "run_multiyear_reports" / run_id
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(report_candidates[0]) as z:
        z.extractall(target_dir)

    report_path = target_dir / "report.html"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="No multiyear report available.")

    content = report_path.read_text(encoding="utf-8")
    assets_base = f"/api/studies/{study_id}/runs/{run_id}/multiyear-report/assets"
    content = content.replace('src="assets/', f'src="{assets_base}/')
    content = content.replace("src='assets/", f"src='{assets_base}/")
    return HTMLResponse(content=content)


@router.get("/{study_id}/runs/{run_id}/config")
def get_run_config_snapshot(study_id: StudyID, run_id: RunID):
    run_dir = _ensure_run_dir(study_id, run_id)
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.is_file():
        raise HTTPException(status_code=404, detail="Run config snapshot not available.")
    return FileResponse(cfg_path, filename=f"{study_id}_{run_id}_config.yaml")


@router.get("/{study_id}/runs/{run_id}/download")
def download_run_archive(study_id: StudyID, run_id: RunID):
    """Stream a zip of the entire run snapshot."""
    run_dir = _ensure_run_dir(study_id, run_id)
    tmp = tempfile.NamedTemporaryFile(prefix=f"{study_id}_{run_id}_", suffix=".zip", delete=False)
    tmp.close()
    try:
        with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
            for dirpath, _dirnames, filenames in os.walk(run_dir):
                for name in filenames:
                    if name == "_run_meta.json":
                        continue
                    full = Path(dirpath) / name
                    zf.write(full, full.relative_to(run_dir))
    except Exception:
        os.unlink(tmp.name)
        raise
    return FileResponse(
        tmp.name,
        filename=f"{study_id}_{run_id}.zip",
        media_type="application/zip",
        background=None,
    )


@router.get("/{study_id}/runs/{run_id}/apsim")
def get_run_apsim(study_id: StudyID, run_id: RunID):
    """Return the APSIM mapping + list of source-template files snapshotted with the run."""
    run_dir = _ensure_run_dir(study_id, run_id)
    manifest_path = run_dir / "_apsim_mapping.json"
    apsim_dir = run_dir / "apsim"

    manifest = {"filter_column": None, "region_to_template": {}, "files": []}
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text())
            if isinstance(data, dict):
                manifest.update(data)
        except Exception:
            pass

    files: List[dict] = []
    if apsim_dir.is_dir():
        for p in sorted(apsim_dir.iterdir()):
            if not p.is_file():
                continue
            files.append({"name": p.name, "size": p.stat().st_size})

    return {"manifest": manifest, "files": files}


@router.get("/{study_id}/runs/{run_id}/apsim/{filename}")
def download_run_apsim_file(study_id: StudyID, run_id: RunID, filename: str):
    """Download one APSIM source template from the snapshot."""
    run_dir = _ensure_run_dir(study_id, run_id)
    apsim_dir = (run_dir / "apsim").resolve()
    if not apsim_dir.is_dir():
        raise HTTPException(status_code=404, detail="No APSIM files snapshotted for this run.")

    file_path = (apsim_dir / filename).resolve()
    if not str(file_path).startswith(str(apsim_dir) + os.sep) or not file_path.is_file():
        raise HTTPException(status_code=404, detail="APSIM file not found.")
    return FileResponse(file_path, filename=filename, media_type="application/octet-stream")


@router.delete("/{study_id}/runs/{run_id}")
def delete_run(study_id: StudyID, run_id: RunID):
    run_dir = _ensure_run_dir(study_id, run_id)
    shutil.rmtree(run_dir, ignore_errors=False)

    # Clean up any extracted caches for this run.
    for extracted in [
        Path(studies_dir) / study_id / "snakemake" / "run_maps" / run_id,
        Path(studies_dir) / study_id / "snakemake" / "run_multiyear_reports" / run_id,
    ]:
        if extracted.exists():
            shutil.rmtree(extracted, ignore_errors=True)

    return {"status": "deleted", "run_id": run_id}
