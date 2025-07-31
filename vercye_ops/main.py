import os
from pathlib import Path
import shutil
import subprocess
import sys

import click
import yaml

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
    new_imagery_config_path =  os.path.join(new_study_path, "lai_config.yaml")
    shutil.copy(imagery_template_path, new_imagery_config_path)

    # Set an initial value from env variable for the LAI storage dir if set.
    if is_env_set():
        lai_dir = read_lai_dir_from_env()

        if lai_dir:
            # Replace the lai dir, if env variable is set
            with open(new_imagery_config_path, "r") as file:
                content = file.read()

            lai_dir_placeholder = os.path.join(lai_dir, 'XXXX')
            content = content.replace("out_dir: XXXX", f"out_dir: {lai_dir_placeholder}")

            # Save the modified content back to the file
            with open(new_imagery_config_path, "w") as file:
                file.write(content)

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


def run_study(study_dir, study_name, validate_only, extra_snakemake_args=None):
    """Runs the study with snakemake and the specified profile"""

    config_file_path = os.path.join(study_dir, study_name, study_name, 'config.yaml')
    profile_dir = os.path.join(study_dir, study_name, "profile")

    snakemake_run_dir = os.path.join(study_dir, study_name, 'snakemake')
    os.makedirs(snakemake_run_dir, exist_ok=True)

    validate_run_config(config_file_path)

    if validate_only:
        return
    
    snakefile_path = rel_path('snakemake/Snakefile')
    workdir =  rel_path('snakemake')
    
    cmd = [
        "snakemake",
        "--snakefile", snakefile_path,
        "--configfile", config_file_path,
        "--profile", profile_dir,
        "--directory", snakemake_run_dir,
        "--printshellcmds",
        "--rerun-incomplete"
    ]

    # Add extra snakemake args if provided, allows to run custom options
    if extra_snakemake_args:
        print(f'Running snakemake with additional args: {extra_snakemake_args}.')
        cmd.extend(extra_snakemake_args)

    try:
        # Start process with real-time output and error capture
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout for real-time display
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )

        # Read and display output in real-time
        for line in process.stdout:
            print(line, end='')  # Print without extra newline since line already has one

        # Wait for process to complete
        process.wait()

        if process.returncode == 0:
            print("Snakemake completed successfully!")
        else:
            print(f"\nSnakemake failed with exit code: {process.returncode}")
            sys.exit(process.returncode)

    except Exception as e:
        print(f"\nError running snakemake: {e}")
        if process:
            process.terminate()
        raise
       

def get_env_file_path():
    return Path(BASE_DIR).parent / '.env'

def is_env_set():
    return get_env_file_path().exists()

def read_study_dir_from_env():
    env_vars = dotenv_values(get_env_file_path())
    return env_vars.get('STUDY_DIR', None)

def read_lai_dir_from_env():
    env_vars = dotenv_values(get_env_file_path())
    return env_vars.get('LAI_BASE_DIR', None)


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.argument("mode", type=click.Choice(["init", "prep", "lai", "chirps", "run"]))
@click.option("--name", required=False, help="Unique name for your yield study.")
@click.option("--dir", required=False, help="Directory of your yieldstudy. Optional, if a env file is provided.")
@click.option("--chirps-dir", required=False, help="Directory to store CHIRPS data.")
@click.option("--chirps-start", required=False, help="Daterange start for CHIRPS data. Format: YYYY-MM-DD.")
@click.option("--chirps-end", required=False, help="Daterange end for CHIRPS data. Format: YYYY-MM-DD")
@click.option("--chirps-cores", required=False, help="Optional: Number of cores to use for parallel chirps downloading. Should be max 5. Default 5.", default=5)
@click.option("--validate", required=False, is_flag=True, default=False, help="Optional: Can be used with run. Add this flag, if you only want to validate your configuration, instead of running.")
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

    if extra_args and mode != 'run':
        raise ValueError(f'Received unknown arguments "{extra_args}".')

    # Load study dir from env if available and not set via cli
    if is_env_set() and not dir:
        env_study_dir = read_study_dir_from_env()
        if env_study_dir:
            dir = env_study_dir

    try:
        if mode in ['init', 'prep', 'lai', 'run']:
            if not name or not dir:
                raise ValueError(f'Must provide --name and --dir for "{mode}" mode')

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
            print(extra_args)
            run_study(study_dir= dir, study_name=name, validate_only=validate, extra_snakemake_args=extra_args)
        else:
            raise ValueError('Invalid mode. Mode must be "init", "prep", "lai", "chirps" or "run".')
    except Exception as e:
        click.secho(f"ERROR: {e}. Aborted.", fg="red", err=True)
        sys.exit(1)

    click.secho(f"SUCCESS: {mode} completed!", fg="green")
