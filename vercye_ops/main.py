import os
from pathlib import Path
import shutil
import sys

import click
import yaml
from snakemake import snakemake

from vercye_ops.lai.lai_creation_STAC.run_stac_dl_pipeline import (
    run_pipeline as run_imagery_dl_pipeline,
)
from vercye_ops.met_data.download_chirps_data import run_chirps_download
from vercye_ops.prepare_yieldstudy import prepare_study
from vercye_ops.snakemake.config_validation import validate_run_config

from dotenv import dotenv_values

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def rel_path(*path_parts):
    return os.path.join(BASE_DIR, *path_parts)


def init_study(name, dir):
    """Initializes a directory for a new study with a number of templates of options"""
    new_study_path = os.path.join(dir, name)

    if os.path.exists(new_study_path):
        print(f"Error: A yieldstudy with this name already exists under {new_study_path}")
        return

    os.makedirs(new_study_path)

    imagery_template_path = rel_path("lai/lai_creation_STAC/config_example.yaml")
    shutil.copy(imagery_template_path, os.path.join(new_study_path, "lai_config.yaml"))

    preparation_config_path = rel_path("examples/setup_config_template.yaml")
    shutil.copy(preparation_config_path, os.path.join(new_study_path, "setup_config.yaml"))

    profile_template_path = rel_path("snakemake/profiles/hpc/config.yaml")
    os.makedirs(os.path.join(new_study_path, "profile"))
    shutil.copy(profile_template_path, os.path.join(new_study_path, "profile"))

    print(
        f"Template successfully created! Navigate to {new_study_path} and start adjusting your options."
    )

def create_lai_data(name, dir):
    """Download imagery through the STAC pipeline and create LAI"""
    lai_config_path = os.path.join(dir, name, "lai_config.yaml")

    with open(lai_config_path, "r") as f:
        lai_config = yaml.safe_load(f)

    try:
        run_imagery_dl_pipeline(lai_config)
    except Exception as e:
        print(f"Error: LAI Pipeline terminated with error: {e}")


def download_chirps(study_dir, study_name, start_date, end_date, output_dir, num_workers):
    if not start_date and end_date and output_dir:
        if start_date or end_date or output_dir:
            raise ValueError('Must provide --chirps-start, --chirps-end and --chirps-dir, if providing any of these values.')
    elif study_dir and study_name: 
        config_file_path = os.path.join(study_dir, study_name, study_name, 'config.yaml')

        with open(config_file_path, "r") as f:
            config = yaml.safe_load(f)

        output_dir = config['apsim_params']['chirps_dir']

        # Derive min and max start and end date
        all_start_dates = []
        all_end_dates = []

        for year in config['apsim_params']['time_bounds']:
            for timepoint in config['apsim_params']['time_bounds'][year]:
                timepoint_data = config['apsim_params']['time_bounds'][year][timepoint]
                all_start_dates.append(timepoint_data['met_start_date'])
                all_end_dates.append(timepoint_data['met_end_date'])

        start_date = min(all_start_dates)
        end_date = max(all_end_dates)
    else:
        raise ValueError('Must either provide --chirps-start, --chirps-end and --chirps-dir or --name and --dir.')
    
    run_chirps_download(start_date=start_date, end_date=end_date, output_dir=output_dir, num_workers=num_workers)


def run_study(study_dir, study_name, validate_only):
    """Runs the study with snakemake and the specified profile"""

    config_file_path = os.path.join(study_dir, study_name, study_name, 'config.yaml')
    profile_dir = os.path.join(study_dir, study_name, "profile")

    validate_run_config(config_file_path)

    if validate_only:
        return
    
    success = snakemake(
        snakefile="vercye_ops/snakemake/Snakefile",
        configfiles=[config_file_path],
        config_args=["--profile", profile_dir],
        use_conda=True,
        printshellcmds=True,
        workdir="vercye_ops/snakemake",
    )

    if success:
        print("Workflow completed successfully.")
    else:
        print(
            "Workflow failed. Examine the logs to identify the reasons, fix and continue running by rerunning."
        )

def get_env_file_path():
    return Path(BASE_DIR).parent / '.env'

def is_env_set():
    return get_env_file_path().exists()

def read_dir_from_env():
    env_vars = dotenv_values(get_env_file_path())
    return env_vars.get('STUDY_DIR', None)


@click.command()
@click.argument("mode", type=click.Choice(["init", "prep", "lai", "chirps", "run"]))
@click.option("--name", required=False, help="Unique name for your yield study.")
@click.option("--dir", required=False, help="Directory of your yieldstudy. Optional, if a env file is provided.")
@click.option("--chirps-dir", required=False, help="Directory to store CHIRPS data.")
@click.option("--chirps-start", required=False, help="Daterange start for CHIRPS data. Format: YYYY-MM-DD.")
@click.option("--chirps-end", required=False, help="Daterange end for CHIRPS data. Format: YYYY-MM-DD")
@click.option("--chirps-cores", required=False, help="Optional: Number of cores to use for parallel chirps downloading. Should be max 5. Default 5.", default=5)
@click.option("--validate", required=False, is_flag=True, default=False, help="Optional: Can be used with run. Add this flag, if you only want to validate your configuration, instead of running.")
def main(mode, name, dir, chirps_dir, chirps_start, chirps_end, chirps_cores, validate):
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

    # Load study dir from env if available and not set via cli
    if is_env_set() and not dir:
        env_study_dir = read_dir_from_env()
        if env_study_dir:
            dir = env_study_dir

    if mode in ['init', 'prep', 'lai', 'run']:
        if not name or not dir:
            raise ValueError(f'Must provide --name and --dir for "{mode}" mode')
    
    try:
        if mode == "init":
            init_study(name=name, dir=dir)
        elif mode == "prep":
            prepare_config = os.path.join(dir, name, "setup_config.yaml")
            prepare_study(prepare_config)
        elif mode == "lai":
            create_lai_data(name=name, dir=dir)
        elif mode == "chirps":
            download_chirps(study_dir= dir, study_name=name, start_date=chirps_start, end_date=chirps_end, output_dir=chirps_dir, num_workers=chirps_cores)
        elif mode == "run":
            run_study(study_dir= dir, study_name=name, validate_only=validate)
        else:
            raise ValueError('Invalid mode. Mode must be "init", "prep", "lai", "chirps" or "run".')
    except Exception as e:
        click.secho(f"ERROR: {e}. Aborted.", fg="red", err=True)
        sys.exit(1)

    click.secho(f"SUCCESS: {mode} completed!", fg="green")
