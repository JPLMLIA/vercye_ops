from pathlib import Path
import shutil
import tempfile
from celery import Celery, Task

import os

from vercye_ops.cli import load_yaml_ruamel, run_study as run_vercye
from vercye_ops.cli import prepare_study as prepare_vercye_study
from vercye_ops.cli import read_studies_dir_from_env, validate_run_config

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

    # Use a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output_dir = os.path.join(temp_dir, "output")

        try:
            prepare_vercye_study(config, temp_output_dir, lai_config_path)

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

@celery_app.task(name="tasks.run_vercye_task", base=VercyeTask)
def run_vercye_task(study_id: str):
    log_file_path = os.path.join(studies_dir, study_id, 'snakemake', 'log.txt')
    with open(log_file_path, 'w') as f:
        f.write('')
    run_vercye(study_dir=studies_dir, study_name=study_id, validate_only=False)
