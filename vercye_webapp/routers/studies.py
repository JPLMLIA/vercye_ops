import asyncio
import io
import json
import os
import shutil
import signal
import tempfile
import time
import zipfile
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiofiles
import yaml
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from models import (
    DuplicateStudyRequest,
    LAIConfigRunParams,
    SetupSubmissionsRequest,
    StudyCreateRequest,
    StudyID,
    WindowConfig,
)
from pydantic import TypeAdapter, ValidationError
from pyproj import CRS
from utils import QuotedString
from worker import duplicate_vercye_study_task, run_vercye_task, setup_vercye_task

from vercye_ops.cli import init_study, validate_run_config
from vercye_ops.utils.env_utils import (
    get_run_config_template_file_path,
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
def register_from_existing(existing_study_id: StudyID, body: DuplicateStudyRequest):
    """Registers a new study based on the config of an existing one.
    Uses the same setup config as in the template for creation."""

    try:
        duplicate_vercye_study_task.delay(existing_study_id, body.new_study_id)
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{study_id}/setup")
async def setup_study(
    study_id: StudyID,
    setup_data: str = Form(...),
    shapefile: UploadFile = File(...),
    apsim_files: List[UploadFile] = File(...),
    reference_files: Optional[List[UploadFile]] = File(None),
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
    refdata_dir = study_root / "reference_data"
    for d in (shapefile_dir, apsim_dir, refdata_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not shapefile:
        raise HTTPException(status_code=400, detail="No shapefile provided.")

    if not shapefile.filename or not shapefile.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Shapefile must be a .zip")

    data = await shapefile.read()

    # Check if the file is a valid ZIP
    is_zip = await run_in_threadpool(zipfile.is_zipfile, io.BytesIO(data))
    if not is_zip:
        raise HTTPException(status_code=400, detail="Not a valid ZIP")

    # Extract the ZIP into the shapefile_dir
    await run_in_threadpool(lambda: zipfile.ZipFile(io.BytesIO(data)).extractall(shapefile_dir))

    # find the .shp file. There must only be a single one
    shp_files = list(shapefile_dir.glob("*.shp"))
    if len(shp_files) != 1:
        raise HTTPException(status_code=400, detail=f"Expected exactly one .shp file, found {len(shp_files)}")

    # Save apsim files and reference data files
    async def save_many(files, dest_dir):
        if not files:
            return
        for uf in files:
            if not uf.filename:
                continue
            async with aiofiles.open(dest_dir / uf.filename, "wb") as out:
                content = await uf.read()
                await out.write(content)

    await save_many(apsim_files, apsim_dir)

    if reference_files:
        await save_many(reference_files, refdata_dir)

    setup_cfg["regions_shp_name"] = str(shp_files[0])
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
            apsim_paths[region] = str(apsim_dir / apsim_file_name)
    setup_cfg["APSIM_TEMPLATE_PATHS"] = apsim_paths

    ref_data_paths = defaultdict(list)
    for filename, admin_lvl in setup_submission.reference_mapping.items():
        if filename not in setup_submission.reference_years_mapping:
            raise HTTPException(
                status_code=400,
                detail="Reference data files must be mapped to year AND admin level.",
            )

        # Copy file to correct name
        refdata_year = setup_submission.reference_years_mapping[filename]
        refdata_file_dst_name = os.path.join(refdata_dir, f"referencedata_{admin_lvl}-{str(refdata_year)}.csv")
        refdata_src_path = os.path.join(refdata_dir, filename)
        if refdata_src_path != refdata_file_dst_name:
            shutil.copyfile(refdata_src_path, refdata_file_dst_name)
            os.remove(refdata_src_path)

        ref_data_paths[int(refdata_year)].append({admin_lvl: refdata_file_dst_name})

    setup_cfg["REFERENCE_DATA_PATHS"] = dict(ref_data_paths)

    setup_cfg_path = study_root / "setup_config.yaml"
    with setup_cfg_path.open("w") as f:
        yaml.dump(setup_cfg, f, default_flow_style=False, sort_keys=False)

    # If a run already exists this will be deleted for a clean restart
    # Avoids mixing outputs from different versions due to actual input file changes
    study_dir = os.path.join(get_study_path(studies_dir, study_id), study_id)
    if os.path.exists(study_dir):
        shutil.rmtree(study_dir)

    # Extract and prepare APSIM files and geojsons and create run config for user to fill in
    try:
        run_cfg_template_path = get_run_config_template_file_path(studies_dir, study_id)
        setup_vercye_task(study_id, run_cfg_template_path=run_cfg_template_path)
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{study_id}/setup-config")
def get_setup_config(study_id: StudyID):
    setup_cfg_path = os.path.join(studies_dir, study_id, "setup_config.yaml")

    if not os.path.exists(setup_cfg_path):
        raise HTTPException(status_code=404, detail="No study config template found!")

    return FileResponse(setup_cfg_path, filename="setup_config.yaml")


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
            config_data["lai_params"]["lai_region"] = os.listdir(config_data["lai_params"]["lai_dir"])[0].split("_")[0]
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
def get_run_config(study_id: StudyID):
    run_cfg_path = os.path.join(studies_dir, study_id, study_id, "config.yaml")
    if not os.path.isfile(run_cfg_path):
        raise HTTPException(status_code=404, detail="Run config not found")
    return FileResponse(run_cfg_path, filename="run_config.yaml")


@router.get("/{study_id}/run-config-status")
def get_run_config_status(study_id: StudyID):
    run_cfg_status_path = os.path.join(studies_dir, study_id, study_id, "config_status.txt")
    run_cfg_status_details_path = os.path.join(studies_dir, study_id, study_id, "config_status_details.txt")

    if not os.path.isfile(run_cfg_status_path):
        raise HTTPException(status_code=404, detail="Run config not found")

    with open(run_cfg_status_path) as f:
        status = f.read()

    if not os.path.isfile(run_cfg_status_path):
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
            f.write(e)

        return {"status": "invalid", "detail": "{e}"}

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
def run_study(study_id: StudyID):
    try:
        task = run_vercye_task.delay(study_id)
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


@router.post("/{study_id}/actions/cancel")
def cancel_study(study_id: StudyID, background_tasks: BackgroundTasks):
    task_id_file = os.path.join(studies_dir, study_id, "snakemake", "snakemake_task_id.txt")
    status_file_path = os.path.join(studies_dir, study_id, "snakemake", "status.txt")

    if not os.path.exists(task_id_file):
        raise HTTPException(status_code=404, detail="Study does not have an associated task.")

    with open(task_id_file, "r") as f:
        task_id = int(f.read().strip())

    # kill the whole group since snakemake has subtasks
    try:
        os.killpg(task_id, signal.SIGTERM)
    except Exception as e:
        print(f"Error killing Snakemake group: {e}")

    # Schedule escalation with SIGKILL if process doesn't terminate
    def escalate_kill(pgid: int, delay: int = 30):
        time.sleep(delay)
        try:
            os.killpg(pgid, signal.SIGKILL)
            print(f"Escalated: Sent SIGKILL to process group {pgid}")
        except ProcessLookupError:
            print(f"Process group {pgid} already terminated.")
        except Exception as e:
            print(f"Error during SIGKILL escalation: {e}")

    # Update status
    with open(status_file_path, "w") as f:
        f.write("cancelled")

    background_tasks.add_task(escalate_kill, task_id)

    # Update status
    with open(status_file_path, "w") as f:
        f.write("cancelled")

    return {"status": "cancelled"}


@router.get("/{study_id}/result-timepoints")
def get_result_timepoints(study_id: StudyID):
    study_dir = os.path.join(studies_dir, study_id, study_id)
    years = {f: [] for f in os.listdir(study_dir) if os.path.isdir(os.path.join(study_dir, f))}
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


@router.get("/{study_id}/multiyear-report")
def get_multiyear_report(study_id: StudyID):
    report_path_candidates = glob(
        os.path.join(get_study_path(studies_dir, study_id), study_id, "multiyear_summary_*.html")
    )
    if len(report_path_candidates) != 1:
        raise HTTPException(
            f"Found {len(report_path_candidates)} entries with multiyear_summary_ in the name in study folder."
        )

    with open(report_path_candidates[0]) as f:
        content = f.read()

    return HTMLResponse(content=content)


@router.get("/{study_id}/logs")
def get_study_logs(study_id: StudyID):
    log_path = os.path.join(studies_dir, study_id, "snakemake", "log.txt")
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="No log available. Check individual rules logs if necessary.")

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return PlainTextResponse(text, media_type="text/plain; charset=utf-8")


@router.get("/{study_id}/status")
async def get_study_status(study_id: str):
    status_file_path = os.path.join(studies_dir, study_id, "snakemake", "status.txt")

    # Async check if file exists
    if not await asyncio.to_thread(os.path.exists, status_file_path):
        return {"status": "pending"}

    # Get file modification time
    mtime = await asyncio.to_thread(lambda: os.path.getmtime(status_file_path))

    # Use cache if mtime matches
    if study_id in status_cache and status_cache[study_id][0] == mtime:
        return {"status": status_cache[study_id][1]}

    # Otherwise, read file and update cache
    async with aiofiles.open(status_file_path, mode="r") as f:
        contents = await f.read()

    status_cache[study_id] = (mtime, contents)
    return {"status": contents}


@router.get("")
def get_all_studies():
    if not os.path.exists(studies_dir):
        return []
    return [name for name in os.listdir(studies_dir) if os.path.isdir(os.path.join(studies_dir, name))]
