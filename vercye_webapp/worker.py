import datetime
from pathlib import Path
import shutil
import tempfile
from celery import Celery, Task

import os

from vercye_ops.cli import load_yaml_ruamel, run_study as run_vercye
from vercye_ops.cli import prepare_study as prepare_vercye_study
from vercye_ops.cli import read_studies_dir_from_env, validate_run_config, get_env_file_path

from vercye_ops.met_data.construct_chirps_precipitation_files import all_chirps_data_exists
from vercye_ops.met_data.download_chirps_data import run_chirps_download

from dotenv import dotenv_values

celery_app = Celery(
    "vercye_tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)

celery_app.conf.task_routes = {
    "tasks.run_vercye_task": {"queue": "vercye_runs"},
    "tasks.prepate_vercye_task": {"queue": "vercye_prep"}
}

studies_dir = read_studies_dir_from_env()

@celery_app.task(name="tasks.prepare_vercye_task")
def prepare_vercye_task(study_id: str):
    setup_cfg_path = os.path.join(studies_dir, study_id, "setup_config.yaml")
    config, _ = load_yaml_ruamel(setup_cfg_path)

    real_output_dir = str(Path(setup_cfg_path).parent / Path(setup_cfg_path).parent.name)
    lai_config_path = str(Path(setup_cfg_path).parent / 'lai_config.yaml')

    # Replace relative paths from uploaded files
    shp_basedir = os.path.join(studies_dir, study_id, 'shapefile')
    apsim_basedir = os.path.join(studies_dir, study_id, 'apsim')
    refdata_basedir = os.path.join(studies_dir, study_id, 'reference_data')

    if not os.path.isabs(config['regions_shp_name']):
        config['regions_shp_name'] =  os.path.join(shp_basedir, config['regions_shp_name'])

    for item_name in config['APSIM_TEMPLATE_PATHS']:
        if not os.path.isabs(config['APSIM_TEMPLATE_PATHS'][item_name]):
            config['APSIM_TEMPLATE_PATHS'][item_name] = os.path.join(apsim_basedir, config['APSIM_TEMPLATE_PATHS'][item_name])
    
    for year in config['REFERENCE_DATA_PATHS']:
        for entry in config['REFERENCE_DATA_PATHS'][year]:
            if not os.path.isabs(config['REFERENCE_DATA_PATHS'][year][entry][1]):
                config['REFERENCE_DATA_PATHS'][year][entry][0] = os.path.join(refdata_basedir, config['REFERENCE_DATA_PATHS'][entry][year][1])

    # Use a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output_dir = os.path.join(temp_dir, "output")

        try:
            config_path = prepare_vercye_study(config, temp_output_dir, lai_config_path)

            # Update the tmppath to real study dir
            new_config, ruamel_yaml = load_yaml_ruamel(config_path)
            new_config['sim_study_head_dir'] = real_output_dir
            with open(config_path, "w") as f:
                ruamel_yaml.dump(new_config, f)

            # Only move to final destination if everything succeeds
            if os.path.exists(real_output_dir):
                shutil.rmtree(real_output_dir)
            shutil.move(temp_output_dir, real_output_dir)

            # Set config status metadata after success
            run_cfg_status_path = os.path.join(studies_dir, study_id, study_id, "config_status.txt")
            run_cfg_status_details_path = os.path.join(studies_dir, study_id, study_id, "config_status_details.txt")

            with open(run_cfg_status_path, 'w') as f:
                f.write('invalid')

            with open(run_cfg_status_details_path, 'w') as f:
                f.write('Template not yet filled in')

        except Exception as e:
            raise RuntimeError(f"Failed to prepare study {study_id}: {e}")

class VercyeTask(Task):
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        study_id = kwargs.get('study_id') or args[0]
        try:
            status_file_path = os.path.join(studies_dir, study_id, 'snakemake', 'status.txt')
            with open(status_file_path, 'w') as f:
                f.write('failed')
        except Exception as e:
            print(f"[ERROR] Could not write failure status for {study_id}: {e}")

def load_config(study_id):
    config_path = os.path.join(studies_dir, study_id, study_id, 'config.yaml')
    return load_yaml_ruamel(config_path)[0]


def uses_chirps(study_id):
    config = load_config(study_id)
    return config['apsim_params']['precipitation_source'] == 'CHIRPS'

 
def ensure_chirps_daterange_complete(study_id):
    config = load_config(study_id)
    env_vars = dotenv_values(get_env_file_path())
    chirps_cache_dir =  env_vars.get('CHIRPS_DIR', None)
    log_file_path = os.path.join(studies_dir, study_id, 'snakemake', 'log.txt')

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

    for year in config['apsim_params']['time_bounds']:
        dates = extract_dates(config['apsim_params']['time_bounds'][year])
        min_start_date = min(dates)
        max_end_date = max(dates)
        with open(log_file_path, 'a') as f:
            f.write(f'Fetching CHIRPS data from {min_start_date} to {max_end_date}...')
        run_chirps_download(min_start_date.date(), max_end_date.date(), chirps_cache_dir, 5)


@celery_app.task(name="tasks.run_vercye_task", base=VercyeTask)
def run_vercye_task(study_id: str):
    log_file_path = os.path.join(studies_dir, study_id, 'snakemake', 'log.txt')
    with open(log_file_path, 'w') as f:
        f.write('')

    # Fetch missing chirps data if required
    if uses_chirps(study_id):
        with open(log_file_path, 'a') as f:
            f.write('Fetching missing CHIRPS data....')
        ensure_chirps_daterange_complete(study_id)

    # Validate the Configfile and trigger a snakemake run
    run_vercye(study_dir=studies_dir, study_name=study_id, validate_only=False)
