import os
import time
import yaml
from pathlib import Path
import signal

from fastapi import BackgroundTasks, UploadFile, File, HTTPException, FastAPI
from fastapi.responses import FileResponse, HTMLResponse

from models import StudyID, StudyCreateRequest
from worker import run_vercye_task, prepate_vercye_task

from vercye_ops.cli import init_study, read_studies_dir_from_env, validate_run_config
studies_dir = read_studies_dir_from_env()

app = FastAPI()

# TODO
# On startup set all tasks to failed that have status running, because it means sever crashed/was stopped.
# If job is still running, kill?
# Add possibility to copy a study to adapt params
# seperate routers for UI, and studies
# show data availability as a plot for lai data to see actual dates where avail in a region.

# Frontend related routes

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    with open("study_dashboard.html", "r") as f:
        return HTMLResponse(content=f.read())
    
@app.get("/lai", response_class=HTMLResponse)
def serve_dashboard():
    with open("lai_dashboard.html", "r") as f:
        return HTMLResponse(content=f.read())

# API

@app.post("/studies")
def register_new_study(request: StudyCreateRequest):
    study_id = request.study_id
    new_study_path = init_study(name=study_id, dir=studies_dir)

    new_study_setup_cfg_path = os.path.join(new_study_path, "setup_config.yaml")
    return FileResponse(new_study_setup_cfg_path, filename="setup_config.yaml")


@app.post("/studies/{study_id}/prepare")
def prepare_study(study_id: StudyID, setup_file: UploadFile = File(...)):
    if not setup_file.filename.endswith((".yaml", ".yml")):
        raise HTTPException(status_code=400, detail="Only YAML files are accepted")

    try:
        contents = setup_file.file.read()
        yaml.safe_load(contents)  # Validate YAML
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid YAML format")
    finally:
        setup_file.file.close()

    # Overwrite study setup_config with uploaded file
    setup_cfg_path = os.path.join(studies_dir, study_id, "setup_config.yaml")
    with open(setup_cfg_path, "wb") as f:
        f.write(contents)

    # Create real run config for user to fill in
    try:
        prepate_vercye_task(study_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/studies/{study_id}/run-config")
def set_run_config(study_id: StudyID, run_cfg_file: UploadFile = File(...)):
    if not run_cfg_file.filename.endswith((".yaml", ".yml")):
        raise HTTPException(status_code=400, detail="Only YAML files are accepted")

    try:
        contents = run_cfg_file.file.read()
        config_data = yaml.safe_load(contents)
        validate_run_config(config_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid config: {str(e)}")
    finally:
        run_cfg_file.file.close()

    run_cfg_path = os.path.join(studies_dir, study_id, study_id, "config.yaml")
    with open(run_cfg_path, "wb") as f:
        f.write(contents)

    return {"message": "Run config set successfully"}


@app.get("/studies/{study_id}/run-config")
def get_run_config(study_id: StudyID):
    run_cfg_path = os.path.join(studies_dir, study_id, study_id, "config.yaml")
    if not os.path.isfile(run_cfg_path):
        raise HTTPException(status_code=404, detail="Run config not found")
    return FileResponse(run_cfg_path, filename="config.yaml")


@app.post("/studies/{study_id}/actions/run")
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

@app.post("/studies/{study_id}/actions/cancel")
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
    

@app.get("/studies/{study_id}/result-timepoints")
def get_result_timepoints(study_id: StudyID):
    study_dir = os.path.join(studies_dir, study_id, study_id)
    years = {f: [] for f in os.listdir(study_dir) if os.path.isdir(os.path.join(study_dir, f))}
    for year in years:
        year_timepoints = [f for f in os.listdir(os.path.join(study_dir, year)) if os.path.isdir(os.path.join(study_dir, year, f))]
        years[year] = year_timepoints
    
    return {
        'timepoints': years
    }

@app.get("/studies/{study_id}/map-result/{year}/{timepoint}/{resource}")
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

@app.get("/studies/{study_id}/map-result/{year}/{timepoint}")
def get_map_results(study_id: StudyID, year:int, timepoint: str):
    map_dir = os.path.join(studies_dir, study_id, study_id, str(year), str(timepoint), 'interactive_map')
    map_path = os.path.join(map_dir, 'vercye_results_map.html')
    if not os.path.exists(map_path):
        raise HTTPException(status_code=404, detail="No result map avilable.")
    
    with open(map_path, 'r') as f:
        content = f.read()
        new_path_base = f"/studies/{study_id}/map-result/{year}/{timepoint}"
        content = content.replace("img.src = props.simulationsImgPath;", f"img.src = `{new_path_base}/${{props.simulationsImgPath}}`")
        return HTMLResponse(content=content)

@app.get("/studies/{study_id}/logs")
def get_study_logs(study_id: StudyID):
    log_path = os.path.join(studies_dir, study_id, 'snakemake', 'log.txt')
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="No log available. Check individual rules logs if necessary.")

    return FileResponse(log_path)

@app.get("/studies/{study_id}/status")
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

@app.get("/studies")
def get_all_studies():
    if not os.path.exists(studies_dir):
        return []
    return [name for name in os.listdir(studies_dir) if os.path.isdir(os.path.join(studies_dir, name))]