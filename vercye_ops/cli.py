import logging
import os
import shutil
import subprocess
import sys
import tempfile
from typing import List

import click
import yaml

from vercye_ops.lai.lai_creation_STAC.run_stac_dl_pipeline import run_pipeline as run_imagery_dl_pipeline
from vercye_ops.met_data.download_chirps_data import run_chirps_download
from vercye_ops.prepare_yieldstudy import load_yaml_ruamel, prepare_study
from vercye_ops.snakemake.config_validation import validate_run_config
from vercye_ops.utils.env_utils import (
    get_lai_config_path,
    get_run_config,
    get_run_config_file_path,
    get_run_profile_path,
    get_setup_config_file_path,
    get_snakemake_run_status_file_path,
    get_snakemake_rundir_path,
    get_snakemake_runlog_path,
    get_study_path,
    is_env_set,
    read_lai_dir_from_env,
    read_studies_dir_from_env,
    replace_in_file,
    update_study_status,
)
from vercye_ops.utils.file_sync import sync_tree_content_aware

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def rel_path(*path_parts):
    "Get relative path to a file"
    return os.path.join(BASE_DIR, *path_parts)


def init_study(study_name, studies_dir):
    """Initializes a directory for a new study with a number of templates of options"""
    new_study_path = get_study_path(studies_dir, study_name)

    if os.path.exists(new_study_path):
        logger.info(f"Error: A yieldstudy with this name already exists under {new_study_path}")
        return

    os.makedirs(new_study_path)

    # Template paths
    imagery_template_path = rel_path("lai/lai_creation_STAC/config_example.yaml")
    preparation_config_path = rel_path("examples/setup_config_template.yaml")
    profile_template_path = rel_path("snakemake/profiles/hpc/config.yaml")

    # Copy and adjust templates to new study dir
    run_profile_path = get_run_profile_path(studies_dir, study_name)
    new_imagery_config_path = get_lai_config_path(studies_dir, study_name)
    new_setup_config_path = get_setup_config_file_path(studies_dir, study_name)
    os.makedirs(run_profile_path)
    shutil.copy(profile_template_path, run_profile_path)
    shutil.copy(preparation_config_path, new_setup_config_path)
    shutil.copy(imagery_template_path, new_imagery_config_path)

    # Set an initial value for the LAI output dir if env variable set.
    lai_dir = read_lai_dir_from_env() if is_env_set() else None
    if lai_dir:
        lai_dir_placeholder = os.path.join(lai_dir, "XXXX")
        replace_in_file(new_imagery_config_path, "out_dir: XXXX", f"out_dir: {lai_dir_placeholder}")

    logger.info(f"Template successfully created! Navigate to {new_study_path} and start adjusting your options.")

    return new_study_path


def create_lai_data(study_name, studies_dir):
    """Download imagery through the STAC pipeline and create daily LAI"""
    lai_config_path = get_lai_config_path(studies_dir, study_name)

    with open(lai_config_path, "r") as f:
        lai_config = yaml.safe_load(f)

    try:
        run_imagery_dl_pipeline(lai_config)
    except Exception as e:
        logger.info(f"Error: LAI Pipeline terminated with error: {e}")


def download_chirps(
    studies_dir: str,
    study_name: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    num_workers: int,
):
    """Download chirps precipitation data in a specified timerange.

    Can be used with explicit start and end date to fill that range. Alternatively if a study name and dir are
    provided, all required dates can be derived from the ['apsim_params']['time_bounds'] met dates.

    Args:
        studies_dir (str): path to the studies base directory
        study_name (str): name of the study of interest. Must match the name it has in study_dir
        start_date (ste): Date from when to fetch data in YYYY-MM-DD format.
        end_date (str): Date until when to fetch data in YYYY-MM-DD format.
        output_dir (str): Directory where to save chirps data. Typically a global directory.
        num_workers (int): Number of cores for parallel processing to use.
            Should be limited at 5 to avoid hitting the API to often.

    Returns:
        None
    """
    if not start_date and end_date and output_dir:
        if start_date or end_date or output_dir:
            raise ValueError(
                "Must provide --chirps-start, --chirps-end and --chirps-dir, if providing any of these values."
            )
    elif studies_dir and study_name:
        config = get_run_config(studies_dir, study_name)
        output_dir = config["apsim_params"]["chirps_dir"]

        # Derive min and max start and end date
        all_start_dates = []
        all_end_dates = []

        for year in config["apsim_params"]["time_bounds"]:
            for timepoint in config["apsim_params"]["time_bounds"][year]:
                timepoint_data = config["apsim_params"]["time_bounds"][year][timepoint]
                all_start_dates.append(timepoint_data["met_start_date"])
                all_end_dates.append(timepoint_data["met_end_date"])

        start_date = min(all_start_dates)
        end_date = max(all_end_dates)
    else:
        raise ValueError("Must either provide --chirps-start, --chirps-end and --chirps-dir or --name and --dir.")

    run_chirps_download(start_date=start_date, end_date=end_date, output_dir=output_dir, num_workers=num_workers)


def _is_snakemake_running_for_dir(snakemake_run_dir: str) -> bool:
    """Check if any snakemake process is currently running with the given working directory.

    Uses pgrep to search for snakemake processes whose command line references this directory.
    Returns True if a conflicting process is found, False otherwise.
    """
    abs_dir = os.path.abspath(snakemake_run_dir)
    try:
        result = subprocess.run(
            ["pgrep", "-af", "snakemake"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False

        for line in result.stdout.strip().splitlines():
            if abs_dir in line and "--unlock" not in line:
                return True
        return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # If pgrep is unavailable or times out, err on the side of caution
        logger.warning("Could not check for running snakemake processes (pgrep unavailable or timed out).")
        return True


def _safe_unlock_if_needed(cmd: list, snakemake_run_dir: str):
    """Unlock snakemake working directory only if locks are stale.

    Checks for the presence of .snakemake/locks and whether any snakemake process
    is still actively using this directory. Only unlocks if locks exist but no
    snakemake process is running (i.e. locks are leftover from a crash/kill).

    Raises RuntimeError if locks exist and a snakemake process is still running.
    """
    lock_dir = os.path.join(snakemake_run_dir, ".snakemake", "locks")

    if not os.path.exists(lock_dir) or not os.listdir(lock_dir):
        return  # No locks present, nothing to do

    if _is_snakemake_running_for_dir(snakemake_run_dir):
        raise RuntimeError(
            f"Snakemake working directory '{snakemake_run_dir}' is locked and a snakemake "
            f"process is still running for this directory. Cancel the running process first."
        )

    logger.info(f"Stale snakemake lock detected in {snakemake_run_dir}. Auto-unlocking...")
    cmd_unlock = cmd.copy()
    cmd_unlock.append("--unlock")
    subprocess.run(cmd_unlock)
    logger.info("Unlock complete.")


def run_study(studies_dir: str, study_name: str, validate_only: bool, extra_snakemake_args: List[str] = None):
    """Runs the study pipeline with snakemake and the specified profile.

    The pipeline is launched in a subprocess and the output is saved to a logfile. The run status in written to a log file.
    Before starting the config file is validated to catch most user errors.

    Args:
        studies_dir (str): Base path of all studies
        study_name (str): Name of the study
        validate_only (bool): If set to true, won't run the study but just validate the config.
        extra_snakemake_args (list(str)): Additional arguments to pass to snakemake. See snakemake cli docs for arguments.

    Returns:
        None
    """

    config_file_path = get_run_config_file_path(studies_dir, study_name)
    profile_dir = get_run_profile_path(studies_dir, study_name)
    snakemake_run_dir = get_snakemake_rundir_path(studies_dir, study_name)
    log_file_path = get_snakemake_runlog_path(studies_dir, study_name)
    snakefile_path = rel_path("snakemake/Snakefile")
    os.makedirs(snakemake_run_dir, exist_ok=True)
    update_study_status(studies_dir, study_name, "running")

    # Validate config - snaity checking for a bunch of user errors
    validate_run_config(config_file_path)

    if validate_only:
        return

    # TODO load taskset max num cores from profile file
    # Limiting with taskset since APSIM does not respect the maximum provided number of cores

    profile_file = os.path.join(profile_dir, "config.yaml")
    if not os.path.exists(profile_file):
        raise ValueError("Profile file does not exist. Can't specify number of cores.")

    with open(profile_file, "r") as f:
        profile = yaml.safe_load(f)
        num_cores = str(min(int(profile["cores"]), os.cpu_count()))

    cmd = [
        "taskset",
        "-c",
        f"0-{num_cores}",
        "snakemake",
        "--snakefile",
        snakefile_path,
        "--configfile",
        config_file_path,
        "--profile",
        profile_dir,
        "--directory",
        snakemake_run_dir,
        "--printshellcmds",
        "--rerun-incomplete",
    ]

    logger.info("Running snakemake cmd: %s", " ".join(cmd))

    # Auto-unlock only if locks are stale (no running snakemake for this directory)
    _safe_unlock_if_needed(cmd, snakemake_run_dir)

    # Add extra snakemake args if provided, allows to run custom snaekmake options
    if extra_snakemake_args:
        logger.info(f"Running snakemake with additional args: {extra_snakemake_args}.")
        cmd.extend(extra_snakemake_args)

    try:
        update_study_status(studies_dir, study_name, "running")

        with open(log_file_path, "w", buffering=1) as log_file:  # line-buffered
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=os.setsid,
            )

            # Write process group ID for external termination (e.g., from webapp)
            # Since we use preexec_fn=os.setsid, the process becomes the leader of a new
            # process group, so process.pid == PGID. This allows us to use os.killpg()
            # to send signals to snakemake and all its child processes.
            pgid = process.pid
            with open(os.path.join(snakemake_run_dir, "snakemake_task_id.txt"), "w") as f:
                f.write(str(pgid))

            for line in process.stdout:
                # logger.info(line, end='') # Print to console
                log_file.write(line)  # Write to log file (includes ANSI codes)

            process.wait()

            if process.returncode == 0:
                logger.info("Snakemake completed successfully!")
                update_study_status(studies_dir, study_name, "completed")
            else:
                # Check if this was a cancellation (status set to "cancelling" by the webapp)
                # rather than a genuine failure. Don't overwrite "cancelling" with "failed".
                status_file = get_snakemake_run_status_file_path(studies_dir, study_name)
                current_status = ""
                try:
                    with open(status_file, "r") as f:
                        current_status = f.read().strip()
                except OSError:
                    pass

                if current_status in ("cancelling", "cancelled"):
                    logger.info("\nSnakemake was cancelled.")
                    update_study_status(studies_dir, study_name, "cancelled")
                else:
                    logger.info(f"\nSnakemake failed with exit code: {process.returncode}")
                    update_study_status(studies_dir, study_name, "failed")
                sys.exit(process.returncode)

    except Exception as e:
        logger.info(f"\nError running snakemake: {e}")
        update_study_status(studies_dir, study_name, "failed")
        if process:
            process.terminate()
        raise


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.argument("mode", type=click.Choice(["init", "prep", "lai", "chirps", "run"]))
@click.option("--name", required=False, help="Unique name for your yield study.")
@click.option(
    "--dir",
    required=False,
    help="Directory of your yieldstudy. Optional, if a env file is provided.",
)
@click.option("--chirps-dir", required=False, help="Directory to store CHIRPS data.")
@click.option("--chirps-start", required=False, help="Daterange start for CHIRPS data. Format: YYYY-MM-DD.")
@click.option("--chirps-end", required=False, help="Daterange end for CHIRPS data. Format: YYYY-MM-DD")
@click.option(
    "--chirps-cores",
    required=False,
    help="Optional: Number of cores to use for parallel chirps downloading. Should be max 5. Default 5.",
    default=5,
)
@click.option(
    "--validate",
    required=False,
    is_flag=True,
    default=False,
    help="Optional: Can be used with run. Add this flag, if you only want to validate your configuration, instead of running.",
)
@click.pass_context
def main(ctx, mode, name, dir, chirps_dir, chirps_start, chirps_end, chirps_cores, validate):
    """
    VeRCYe CLI usage instructions:

    MODE: - "init" to initialize a new study and create templates in dir/name. Must be used with --name and --dir. --dir is optional if a .env file is set.

          - "prep" to create the study base structure from the options specified in your dir/name/setup_config.yaml. Must be used with --name and --dir. --dir is optional if a .env file is set.

          - "lai" to download RS imagery from a STAC source specified in your dir/name/lai_config.yaml. Must be used with --name and --dir. --dir is optional if a .env file is set.

          - "chirps" to download CHIRPS precipitation data into a local shared registry.
          If --chirps-dir, --chirps-start and --chirps-end is not provided, --name and --dir must be provided and the values will be derived from the dir/name/run_config.yaml.
          In this case the minimum and maximum met dates are used to fetch the chirps data, even if the range is discontinous. --dir is optional if a .env file is set.

          - "run" to run the yield study with the options specified in your dir/name/run_config.yaml with the profile specified in dir/name/profile. Must be used with --name and --dir. --dir is optional if a .env file is set.

    Example usage:

    vercye init --name ukraine_study_2025-03-20 --dir /home/yieldstudies
    """

    extra_args = ctx.args

    if extra_args and mode != "run":
        raise ValueError(f'Received unknown arguments "{extra_args}".')

    # Load study dir from env if available and not yet set via cli
    if is_env_set() and not dir:
        env_study_dir = read_studies_dir_from_env()
        if env_study_dir:
            dir = env_study_dir

    try:
        if mode in ["init", "prep", "lai", "run"]:
            if not name or not dir:
                raise ValueError(f'Must provide --name and --dir for "{mode}" mode')

        if mode == "init":
            init_study(study_name=name, studies_dir=dir)
        elif mode == "prep":
            # Run prepare_study into a temp dir, then content-aware sync
            # into the real study dir. This way unchanged files keep their
            # mtimes and snakemake will not invalidate downstream rules
            # whose inputs did not actually change (e.g. an APSIM-only
            # template edit should not rerun LAI/met/cropmask rules).
            real_output_dir = get_study_path(dir, name)
            os.makedirs(real_output_dir, exist_ok=True)
            with tempfile.TemporaryDirectory() as tmp_output_dir:
                setup_cfg, _ = load_yaml_ruamel(get_setup_config_file_path(dir, name))
                tmp_config_path = prepare_study(
                    config=setup_cfg,
                    output_dir=tmp_output_dir,
                    lai_config_path=get_lai_config_path(dir, name),
                )
                # prepare_study bakes the (temp) output_dir into the generated
                # config as sim_study_head_dir / study_id. Rewrite both to the
                # real study path before syncing into place.
                tmp_run_cfg, ruamel_yaml = load_yaml_ruamel(tmp_config_path)
                tmp_run_cfg["sim_study_head_dir"] = real_output_dir
                tmp_run_cfg["study_id"] = os.path.basename(real_output_dir)
                with open(tmp_config_path, "w") as f:
                    ruamel_yaml.dump(tmp_run_cfg, f)
                sync_tree_content_aware(tmp_output_dir, real_output_dir)
        elif mode == "lai":
            create_lai_data(study_name=name, studies_dir=dir)
        elif mode == "chirps":
            download_chirps(
                studies_dir=dir,
                study_name=name,
                start_date=chirps_start,
                end_date=chirps_end,
                output_dir=chirps_dir,
                num_workers=chirps_cores,
            )
        elif mode == "run":
            logger.info(f"Running with extra args: {extra_args}")
            run_study(
                studies_dir=dir,
                study_name=name,
                validate_only=validate,
                extra_snakemake_args=extra_args,
            )
        else:
            raise ValueError('Invalid mode. Mode must be "init", "prep", "lai", "chirps" or "run".')
    except Exception as e:
        click.secho(f"ERROR: {e}. Aborted.", fg="red", err=True)
        sys.exit(1)

    click.secho(f"SUCCESS: {mode} completed!", fg="green")
