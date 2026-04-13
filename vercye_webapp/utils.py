import json
import os
import signal
import time

import yaml

from vercye_ops.utils.env_utils import (
    get_snakemake_run_status_file_path,
    read_lai_dir_from_env,
    read_studies_dir_from_env,
    update_study_status,
)


# Helper to save proj string enclosed in quotes in yaml
class QuotedString(str):
    pass


def quoted_scalar_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


yaml.add_representer(QuotedString, quoted_scalar_representer)


def _graceful_kill_pg(pid: int, grace_seconds: int = 5):
    """Send SIGINT to a process group, wait, then escalate to SIGKILL if needed."""
    try:
        os.killpg(pid, signal.SIGINT)
        for _ in range(grace_seconds * 10):
            time.sleep(0.1)
            try:
                os.killpg(pid, 0)  # check if still alive
            except ProcessLookupError:
                return  # exited cleanly
        # Still alive after grace period - force kill
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # already gone


def clean_running_tasks():
    print("cleaning running tasks")
    studies_dir = read_studies_dir_from_env()
    lai_dir = read_lai_dir_from_env()

    # Kill running studies
    for study in os.listdir(studies_dir):
        study_dir = os.path.join(studies_dir, study)
        if not os.path.isdir(study_dir):
            continue

        status_file_pth = get_snakemake_run_status_file_path(studies_dir, study)
        if not os.path.exists(status_file_pth):
            continue
        try:
            with open(status_file_pth, "r") as f:
                status = f.read()

            if status.lower() in ["running", "queued", "cancelling"]:
                task_id_file = os.path.join(studies_dir, study, "snakemake", "snakemake_task_id.txt")
                if os.path.exists(task_id_file):
                    with open(task_id_file, "r") as f:
                        pid = int(f.read().strip())

                    _graceful_kill_pg(pid)
                update_study_status(studies_dir, study, "cancelled" if status.lower() == "cancelling" else "failed")
        except Exception as e:
            print(f"Error during killing of study task: {str(e)}")

    # Kill running lai generation tasks
    for entry in os.listdir(lai_dir):
        entry_path = os.path.join(lai_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        metadata_path = os.path.join(lai_dir, entry, "meta.json")
        print(entry)
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            any_killed = False

            for res, status in metadata["status"].items():
                print(status)
                if status in ["running", "generating", "standardizing", "finalizing"]:
                    any_killed = True

                    lai_entry_dir = os.path.join(lai_dir, entry)
                    executions_dir = os.path.join(lai_entry_dir, "executions")
                    suffix = max(
                        [int(f) for f in os.listdir(executions_dir) if os.path.isdir(os.path.join(executions_dir, f))],
                        default=0,
                    )
                    execuction_dir = os.path.join(executions_dir, str(suffix))
                    task_id_file = os.path.join(execuction_dir, "task_id.txt")

                    with open(task_id_file, "r") as f:
                        pid = int(f.read().strip())

                    _graceful_kill_pg(pid)
                    metadata["status"][res] = "failed"

            # Update metadata
            if any_killed:
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f)
        except Exception as e:
            print(f"Error during killing of lai task: {str(e)}")
