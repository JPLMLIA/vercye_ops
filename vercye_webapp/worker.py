import datetime
import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

import yaml
from celery import Celery, Task

from vercye_ops.cli import init_study
from vercye_ops.cli import prepare_study as prepare_vercye_study
from vercye_ops.cli import run_study as run_vercye
from vercye_ops.lai.lai_creation_STAC.run_stac_dl_pipeline import update_status
from vercye_ops.met_data.download_chirps_data import run_chirps_download
from vercye_ops.utils.env_utils import (
    get_env_vars,
    get_run_config_file_path,
    get_setup_config,
    get_setup_config_file_path,
    get_snakemake_runlog_path,
    get_study_path,
    load_yaml_ruamel,
    read_studies_dir_from_env,
    save_setup_config,
)

celery_app = Celery(
    "vercye_tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)

celery_app.conf.task_routes = {
    "tasks.run_vercye_task": {"queue": "vercye_processing"},
    "tasks.prepare_vercye_task": {"queue": "vercye_prep"},
    "tasks.generate_lai_task": {"queue": "vercye_processing"},
    "tasks.duplicate_study_task": {"queue": "vercye_prep"},
}

studies_dir = read_studies_dir_from_env()


@celery_app.task(name="tasks.prepare_vercye_task")
def prepare_vercye_task(study_id: str):
    setup_cfg_path = get_setup_config_file_path(studies_dir, study_id)
    config, _ = load_yaml_ruamel(setup_cfg_path)

    real_output_dir = str(Path(setup_cfg_path).parent / Path(setup_cfg_path).parent.name)
    lai_config_path = str(Path(setup_cfg_path).parent / "lai_config.yaml")
    # Use a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output_dir = os.path.join(temp_dir, "output")

        try:
            config_path = prepare_vercye_study(config, temp_output_dir, lai_config_path)

            # Update the tmppath to real study dir
            new_config, ruamel_yaml = load_yaml_ruamel(config_path)
            new_config["sim_study_head_dir"] = real_output_dir
            with open(config_path, "w") as f:
                ruamel_yaml.dump(new_config, f)
            # Only move to final destination if everything succeeds
            if os.path.exists(real_output_dir):
                shutil.rmtree(real_output_dir)
            shutil.move(temp_output_dir, real_output_dir)

            # Set config status metadata after success
            run_cfg_status_path = os.path.join(studies_dir, study_id, study_id, "config_status.txt")
            run_cfg_status_details_path = os.path.join(studies_dir, study_id, study_id, "config_status_details.txt")
            with open(run_cfg_status_path, "w") as f:
                f.write("invalid")

            with open(run_cfg_status_details_path, "w") as f:
                f.write("Template not yet filled in")

        except Exception as e:
            raise RuntimeError(f"Failed to prepare study {study_id}: {e}")


# Wrapper to handle cleanup and status logging on failure
class VercyeTask(Task):
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        study_id = kwargs.get("study_id") or args[0]
        try:
            status_file_path = os.path.join(studies_dir, study_id, "snakemake", "status.txt")
            with open(status_file_path, "w") as f:
                f.write("failed")
        except Exception as e:
            print(f"[ERROR] Could not write failure status for {study_id}: {e}")


def load_config(study_id):
    config_path = get_run_config_file_path(studies_dir, study_id)
    return load_yaml_ruamel(config_path)[0]


def uses_chirps(study_id):
    config = load_config(study_id)
    return config["apsim_params"]["precipitation_source"] == "CHIRPS"


def ensure_chirps_daterange_complete(study_id):
    config = load_config(study_id)
    env_vars = get_env_vars()
    chirps_cache_dir = env_vars.get("CHIRPS_DIR", None)
    log_file_path = get_snakemake_runlog_path(studies_dir, study_id)

    def extract_dates(obj):
        dates = []
        if isinstance(obj, dict):
            for value in obj.values():
                dates.extend(extract_dates(value))
        elif isinstance(obj, str):
            try:
                # Try parsing to check if it's a date
                dt = datetime.strptime(obj, "%Y-%m-%d")
                dates.append(dt)
            except ValueError:
                pass
        return dates

    for year in config["apsim_params"]["time_bounds"]:
        dates = extract_dates(config["apsim_params"]["time_bounds"][year])
        min_start_date = min(dates)
        max_end_date = max(dates)
        with open(log_file_path, "a") as f:
            f.write(f"Fetching CHIRPS data from {min_start_date} to {max_end_date}...")
        run_chirps_download(min_start_date.date(), max_end_date.date(), chirps_cache_dir, 5)


@celery_app.task(name="tasks.run_vercye_task", base=VercyeTask)
def run_vercye_task(study_id: str):
    os.makedirs(os.path.join(studies_dir, study_id, "snakemake"), exist_ok=True)
    log_file_path = os.path.join(studies_dir, study_id, "snakemake", "log.txt")
    with open(log_file_path, "w") as f:
        f.write("")

    # Fetch missing chirps data if required
    if uses_chirps(study_id):
        with open(log_file_path, "a") as f:
            f.write("Fetching missing CHIRPS data....")
        ensure_chirps_daterange_complete(study_id)

    # Validate the Configfile and trigger a snakemake run
    run_vercye(studies_dir=studies_dir, study_name=study_id, validate_only=False)


class LAITask(Task):
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        config_path = kwargs.get("config_path") or args[0]
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            resolution = config["resolution"]
            metadata_index_file = os.path.join(config["out_dir"], "meta.json")
            update_status(metadata_index_file, resolution, "failed")
        except Exception as e:
            print(f"[ERROR] Could not write failure status from LAI generation: {e}")


@celery_app.task(name="tasks.generate_lai_task", base=LAITask)
def run_lai_generation(config_path):
    log_file_path = Path(config_path).parent / "log.txt"
    taskid_save_file = str(Path(config_path).parent / "task_id.txt")
    script_path = Path(__file__).parent.parent / "vercye_ops" / "lai" / "lai_creation_STAC" / "run_stac_dl_pipeline.py"

    pgid = None
    try:
        with open(log_file_path, "w", buffering=1) as log_file:
            proc = subprocess.Popen(
                ["python3", script_path, config_path],
                preexec_fn=os.setsid,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Save process group to allow sending kill signal
            pgid = os.getpgid(proc.pid)
            with open(taskid_save_file, "w") as f:
                f.write(str(pgid))

            for line in proc.stdout:
                # print(line, end='') # Print to console
                log_file.write(line)  # Write to log file (includes ANSI codes)

            proc.wait()

        if proc.returncode == 0:
            print("LAI Generation completed successfully!")
        else:
            print(f"Generation failed with exit code: {proc.returncode}")
            raise Exception(f"Failed to generate lai: {proc.returncode}")
    except Exception as e:
        if pgid:
            os.killpg(pgid, signal.SIGKILL)
        raise e


@celery_app.task(name="tasks.duplicate_study_task", base=LAITask)
def duplicate_vercye_study_task(existing_study_id, new_study_id):

    init_study(study_name=new_study_id, studies_dir=studies_dir)

    existing_config, config_ruamel = get_setup_config(studies_dir, existing_study_id, ruamel=True)
    new_study_dir = get_study_path(studies_dir, new_study_id)

    # RegionsShp copying
    new_regions_shp_fpath = os.path.join(new_study_dir, "shapefile", Path(existing_config["regions_shp_name"]).name)
    shutil.copy(existing_config["regions_shp_name"], new_regions_shp_fpath)
    existing_config["regions_shp_name"] = new_regions_shp_fpath

    # Refdata paths copying
    for year, year_fpaths in existing_config["REFERENCE_DATA_PATHS"].items():
        for idx, item in enumerate(year_fpaths):
            year_fpath = item[1]
            new_year_fpath = os.path.join(new_study_dir, "reference_data", Path(year_fpath).name)
            shutil.copy(year_fpath, new_year_fpath)
            existing_config["REFERENCE_DATA_PATHS"][year][idx][1] = new_year_fpath

    # APSIM files copying
    for key, apsim_fpath in existing_config["APSIM_TEMPLATE_PATHS"].items():
        new_apsim_fpath = os.path.join(new_study_dir, "apsim", Path(apsim_fpath).name)
        shutil.copy(apsim_fpath, new_apsim_fpath)
        existing_config["APSIM_TEMPLATE_PATHS"][key] = new_apsim_fpath

    save_setup_config(existing_config, studies_dir, new_study_id, config_ruamel)

    # Extract individual geometries, prepare APSIM files and reference data
    prepare_vercye_task(new_study_id)

    # Copy the runcofig
    existing_run_cfg_path = get_run_config_file_path(studies_dir, existing_study_id)
    shutil.copyfile(existing_run_cfg_path, get_run_config_file_path(studies_dir, new_study_id))

    # TODO modify sim_study_head_dir in new study runconfig
    # Skipping for now as this will be also done on upload
    # TODO add in description from which study it was copied (copied_from param )
