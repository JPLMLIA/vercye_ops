import io
import os
import shutil
import time
from typing import List, Optional
import zipfile
import yaml
from pathlib import Path
import signal
import tempfile

from fastapi import APIRouter, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse


from models import StudyID, StudyCreateRequest
from worker import run_vercye_task, prepare_vercye_task

from vercye_ops.cli import init_study, load_yaml_ruamel, read_studies_dir_from_env, validate_run_config
studies_dir = read_studies_dir_from_env()

router = APIRouter(
    prefix="/studies",
    tags=["studies"]
)

@router.post("/")
def register_new_study(request: StudyCreateRequest):
    study_id = request.study_id
    try:
        init_study(name=study_id, dir=studies_dir)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail='Error initializing a study.')


@router.post("/{study_id}/prepare")
def prepare_study(
    study_id: str,
    setup_file: UploadFile = File(...),
    shapefile_zip: Optional[UploadFile] = File(None),
    apsim_files: Optional[List[UploadFile]] = File(None, alias="apsim_files[]"),
    reference_data_files: Optional[List[UploadFile]] = File(None, alias="reference_data_files[]"),
):
    # Read & validate YAML (bytes are fine for PyYAML)
    contents = setup_file.file.read()
    try:
        yaml.safe_load(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML format: {e}")

    # Upload files
    study_root = Path(studies_dir) / str(study_id)
    shapefile_dir = study_root / "shapefile"
    apsim_dir = study_root / "apsim"
    refdata_dir = study_root / "reference_data"
    for d in (shapefile_dir, apsim_dir, refdata_dir):
        d.mkdir(parents=True, exist_ok=True)

    if shapefile_zip:
        if not shapefile_zip.filename or not shapefile_zip.filename.lower().endswith(".zip"):
            raise HTTPException(status_code=400, detail="Shapefile must be a .zip")

        data = shapefile_zip.file.read()

        # Check if the file is a valid ZIP
        if not zipfile.is_zipfile(io.BytesIO(data)):
            raise HTTPException(status_code=400, detail="Not a valid ZIP")

        # Extract the ZIP into the shapefile_dir
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(shapefile_dir)

    def save_many(files: Optional[List[UploadFile]], dest_dir: Path) -> None:
        if not files:
            return
        for uf in files:
            name = Path(uf.filename or "").name
            if not name:
                continue
            with (dest_dir / name).open("wb") as out:
                shutil.copyfileobj(uf.file, out)

    save_many(apsim_files, apsim_dir)
    save_many(reference_data_files, refdata_dir)
    (study_root / "setup_config.yaml").write_bytes(contents)

    setup_cfg_path = os.path.join(studies_dir, study_id, "setup_config.yaml")
    config, _ = load_yaml_ruamel(setup_cfg_path)

    real_output_dir = str(Path(setup_cfg_path).parent / Path(setup_cfg_path).parent.name)
    lai_config_path = str(Path(setup_cfg_path).parent / 'lai_config.yaml')

    # Create real run config for user to fill in
    try:
        prepare_vercye_task(study_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{study_id}/setup-config")
def get_setup_config(study_id: StudyID):
    setup_cfg_path = os.path.join(studies_dir, study_id, "setup_config.yaml")
    print(setup_cfg_path)

    if not os.path.exists(setup_cfg_path):
        raise HTTPException(status_code=404, detail="No study config template found!")
    
    return FileResponse(setup_cfg_path, filename="setup_config.yaml")


@router.post("/{study_id}/run-config")
def set_run_config(study_id: StudyID, run_cfg_file: UploadFile = File(...)):
    if not run_cfg_file.filename.endswith((".yaml", ".yml")):
        raise HTTPException(status_code=400, detail="Only YAML files are accepted")

    run_cfg_status_path = os.path.join(studies_dir, study_id, study_id, "config_status.txt")
    run_cfg_status_details_path = os.path.join(studies_dir, study_id, study_id, "config_status_details.txt")

    try:
        contents = run_cfg_file.file.read()

        # Check that Yaml was uploaded
        config_data = yaml.safe_load(contents)

        # Save to tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="wb") as tmp_file:
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name

        validate_run_config(tmp_file_path)
    except Exception as e:

        with open(run_cfg_status_path, 'w') as f:
            f.write('invalid')

        with open(run_cfg_status_details_path, 'w') as f:
            f.write(str(e))

        raise HTTPException(status_code=400, detail=f"Invalid config: {str(e)}")
    finally:
        run_cfg_file.file.close()

    run_cfg_path = os.path.join(studies_dir, study_id, study_id, "config.yaml")
    shutil.copyfile(tmp_file_path, run_cfg_path)

    with open(run_cfg_status_path, 'w') as f:
        f.write('valid')

    with open(run_cfg_status_details_path, 'w') as f:
        f.write('OK!')

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
        status_details = 'No details available'
    else:
        with open(run_cfg_status_details_path) as f:
            status_details = f.read()
    
    return {
        "status": status,
        "details": status_details
    }

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

        with open(run_cfg_status_path, 'w') as f:
            f.write('invalid')

        with open(run_cfg_status_details_path, 'w') as f:
            f.write(e)

        return {
            "status": "invalid",
            "detail": "{e}"
        }

    with open(run_cfg_status_path, 'w') as f:
        f.write('valid')

    with open(run_cfg_status_details_path, 'w') as f:
        f.write('OK!')

    return {
        "status": "valid",
        "detail": ""
    }



@router.post("/{study_id}/actions/run")
def run_study(study_id: StudyID):
    try:
        task = run_vercye_task.delay(study_id)
        task_id = task.id

        status_file_path = os.path.join(studies_dir, study_id, 'snakemake', 'status.txt')
        with open(status_file_path, 'w') as f:
            f.write('queued')

        # Save worker celery task_id for later reference e.g for cancelling a job.
        task_id_file = os.path.join(studies_dir, study_id, 'snakemake', 'celery_task_id.txt')
        with open(task_id_file, 'w') as f:
            f.write(task_id)

        return {"message": "Study run started in background."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{study_id}/actions/cancel")
def cancel_study(study_id: StudyID, background_tasks: BackgroundTasks):
    task_id_file = os.path.join(studies_dir, study_id, 'snakemake', 'snakemake_task_id.txt')
    status_file_path = os.path.join(studies_dir, study_id, 'snakemake', 'status.txt')

    if not os.path.exists(task_id_file):
        return HTTPException(status_code=404, detail='Study does not have an associated task.')

    with open(task_id_file, 'r') as f:
        task_id = f.read()

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
    with open(status_file_path, 'w') as f:
        f.write('cancelled')

    background_tasks.add_task(escalate_kill, task_id)
    
    # Update status
    with open(status_file_path, 'w') as f:
        f.write('cancelled')
    
    return {"status": "cancelled"}
    

@router.get("/{study_id}/result-timepoints")
def get_result_timepoints(study_id: StudyID):
    study_dir = os.path.join(studies_dir, study_id, study_id)
    years = {f: [] for f in os.listdir(study_dir) if os.path.isdir(os.path.join(study_dir, f))}
    for year in years:
        year_timepoints = [f for f in os.listdir(os.path.join(study_dir, year)) if os.path.isdir(os.path.join(study_dir, year, f))]
        years[year] = year_timepoints
    
    return {
        'timepoints': years
    }

@router.get("/{study_id}/map-result/{year}/{timepoint}/{resource}")
def get_map_resource(study_id: StudyID, year: int, timepoint: str, resource: str):
    base_path = Path(studies_dir) / study_id / study_id / str(year) / str(timepoint) / "interactive_map"
    file_path = base_path / resource

    try:
        resolved_path = file_path.resolve(strict=False)
        # Prevent traversal attacks
        if not resolved_path.is_file() or not str(resolved_path).startswith(str(base_path.resolve())):
            raise HTTPException(status_code=403, detail="Invalid resource path.")
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid resource path.")
    
    return FileResponse(resolved_path)

@router.get("/{study_id}/map-result/{year}/{timepoint}")
def get_map_results(study_id: StudyID, year:int, timepoint: str):
    map_dir = os.path.join(studies_dir, study_id, study_id, str(year), str(timepoint), 'interactive_map')
    map_path = os.path.join(map_dir, 'vercye_results_map.html')
    if not os.path.exists(map_path):
        raise HTTPException(status_code=404, detail="No result map avilable.")
    
    with open(map_path, 'r') as f:
        content = f.read()
        new_path_base = f"/api/studies/{study_id}/map-result/{year}/{timepoint}"
        content = content.replace("img.src = props.simulationsImgPath;", f"img.src = `{new_path_base}/${{props.simulationsImgPath}}`")
        return HTMLResponse(content=content)

@router.get("/{study_id}/logs")
def get_study_logs(study_id: StudyID):
    log_path = os.path.join(studies_dir, study_id, 'snakemake', 'log.txt')
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="No log available. Check individual rules logs if necessary.")

    return FileResponse(log_path)

@router.get("/{study_id}/status")
def get_study_status(study_id: StudyID):
    status_file_path = os.path.join(studies_dir, study_id, 'snakemake', 'status.txt')
    if not os.path.exists(status_file_path):
        return {
            'status': 'pending'
        }

    with open(status_file_path, 'r') as f:
        return {
            'status': f.read()
        }

@router.get("/")
def get_all_studies():
    if not os.path.exists(studies_dir):
        return []
    return [name for name in os.listdir(studies_dir) if os.path.isdir(os.path.join(studies_dir, name))]