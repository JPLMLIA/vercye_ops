from celery import Celery, Task

import os

from vercye_ops.cli import run_study as run_vercye
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
def prepate_vercye_task(study_id: str):
    setup_cfg_path = os.path.join(studies_dir, study_id, "setup_config.yaml")
    run_cfg_path = prepare_vercye_study(setup_cfg_path)


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
