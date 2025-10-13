import os
import shutil
import signal
import subprocess
import tempfile
from datetime import datetime
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
    get_run_config,
    get_run_config_file_path,
    get_run_config_template_file_path,
    get_setup_config,
    get_setup_config_file_path,
    get_snakemake_runlog_path,
    get_study_path,
    load_yaml_ruamel,
    read_studies_dir_from_env,
    save_setup_config,
    update_study_status,
    write_yaml_ruamel,
)

celery_app = Celery(
    "vercye_tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)

celery_app.conf.task_routes = {
    "tasks.run_vercye_task": {"queue": "vercye_processing"},
    "tasks.setup_vercye_task": {"queue": "vercye_prep"},
    "tasks.generate_lai_task": {"queue": "vercye_processing"},
    "tasks.duplicate_study_task": {"queue": "vercye_prep"},
}

studies_dir = read_studies_dir_from_env()


@celery_app.task(name="tasks.setup_vercye_task")
def setup_vercye_task(study_id: str, run_cfg_template_path: str = None):
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

            if os.path.exists(run_cfg_template_path):
                # If a template file is existing, use this to prefill most run parameters
                # Since changes might have occured during the setup these need to be transferred from the new config
                template_config, ruamel_yaml = load_yaml_ruamel(run_cfg_template_path)
                template_config["regions_shp_name"] = new_config["regions_shp_name"]
                template_config["regions"] = new_config["regions"]
                template_config["years"] = new_config["years"]
                template_config["timepoints"] = new_config["timepoints"]
                template_config["apsim_params"]["time_bounds"] = new_config["apsim_params"]["time_bounds"]
                template_config["lai_params"]["time_bounds"] = new_config["lai_params"]["time_bounds"]

                for key in list(
                    set(new_config["lai_params"]["crop_mask"].keys())
                    - set(template_config["lai_params"]["crop_mask"].keys())
                ):
                    template_config["lai_params"]["crop_mask"][key] = new_config["lai_params"]["crop_mask"][key]

                for key in list(
                    set(new_config["eval_params"]["aggregation_levels"].keys())
                    - set(template_config["eval_params"]["aggregation_levels"].keys())
                ):
                    template_config["eval_params"]["aggregation_levels"][key] = new_config["eval_params"][
                        "aggregation_levels"
                    ][key]

                new_config = template_config

            new_config["sim_study_head_dir"] = real_output_dir
            new_config["study_id"] = study_id[:24]

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
            f.write(f"Ensuring CHIRPS data from {min_start_date} to {max_end_date} exists...\n")
        run_chirps_download(min_start_date.date(), max_end_date.date(), chirps_cache_dir, 2)


@celery_app.task(name="tasks.run_vercye_task", base=VercyeTask)
def run_vercye_task(study_id: str, force_rerun: bool):
    os.makedirs(os.path.join(studies_dir, study_id, "snakemake"), exist_ok=True)
    update_study_status(studies_dir, study_id, "running")
    log_file_path = os.path.join(studies_dir, study_id, "snakemake", "log.txt")
    with open(log_file_path, "w") as f:
        f.write("")

    # Fetch missing chirps data if required
    if uses_chirps(study_id):
        with open(log_file_path, "a") as f:
            f.write("Checking for missing CHIRPS data....\n")
        ensure_chirps_daterange_complete(study_id)

    # Add flag to force snakemake to ignore all intermediate files and rerun all
    extra_args = None
    if force_rerun:
        extra_args = ["-F"]

    # Validate the Configfile and trigger a snakemake run
    run_vercye(studies_dir=studies_dir, study_name=study_id, validate_only=False, extra_snakemake_args=extra_args)


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


@celery_app.task(name="tasks.duplicate_study_task")
def duplicate_vercye_study_task(existing_study_id, new_study_id):
    # cleanup on failure
    init_study(study_name=new_study_id, studies_dir=studies_dir)

    existing_config, config_ruamel = get_setup_config(studies_dir, existing_study_id, ruamel=True)
    new_study_dir = get_study_path(studies_dir, new_study_id)
    print(existing_config)

    # RegionsShp copying
    shapefile_dir = os.path.join(new_study_dir, "shapefile")
    os.makedirs(shapefile_dir, exist_ok=True)
    orig_shp = Path(existing_config["regions_shp_name"])
    shapefile_stem = orig_shp.stem

    # Copy all files with same stem (any extension)
    for src in orig_shp.parent.glob(f"{shapefile_stem}.*"):
        dst = os.path.join(shapefile_dir, src.name)
        shutil.copy(src, dst)

    existing_config["regions_shp_name"] = str(Path(shapefile_dir) / f"{shapefile_stem}.shp")

    # Refdata paths copying
    shutil.copytree(
        os.path.join(get_study_path(studies_dir, existing_study_id), "reference_data"),
        os.path.join(new_study_dir, "reference_data"),
    )

    for year, year_fpaths in existing_config["REFERENCE_DATA_PATHS"].items():
        for idx, item in enumerate(year_fpaths):
            item_agg_lvl = next(iter(item.keys()))  # first key
            year_fpath = item[item_agg_lvl]
            new_year_fpath = os.path.join(new_study_dir, "reference_data", Path(year_fpath).name)
            item[item_agg_lvl] = new_year_fpath
            existing_config["REFERENCE_DATA_PATHS"][year][idx] = item

    # APSIM files copying
    os.makedirs(os.path.join(new_study_dir, "apsim"), exist_ok=True)
    for key, apsim_fpath in existing_config["APSIM_TEMPLATE_PATHS"].items():
        new_apsim_fpath = os.path.join(new_study_dir, "apsim", Path(apsim_fpath).name)
        shutil.copy(apsim_fpath, new_apsim_fpath)
        existing_config["APSIM_TEMPLATE_PATHS"][key] = new_apsim_fpath

    save_setup_config(existing_config, studies_dir, new_study_id, config_ruamel)

    # Copy the runcofig
    if os.path.exists(get_run_config_file_path(studies_dir, existing_study_id)):
        run_config, run_config_ruamel = get_run_config(studies_dir, existing_study_id, ruamel=True)
        run_config["copied_from_study_id"] = existing_study_id

        # Save as template in new dir
        new_run_config_template_path = get_run_config_template_file_path(studies_dir, new_study_id)
        write_yaml_ruamel(run_config, run_config_ruamel, new_run_config_template_path)
