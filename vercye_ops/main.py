import os
import shutil

import click
import yaml
from snakemake import snakemake

from vercye_ops.lai.lai_creation_STAC.run_stac_dl_pipeline import (
    run_pipeline as run_imagery_dl_pipeline,
)
from vercye_ops.met_data.download_chirps_data import run_chirps_download
from vercye_ops.prepare_yieldstudy import prepare_study
from vercye_ops.snakemake.config_validation import validate_run_config

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
    shutil.copy(imagery_template_path, os.path.join(new_study_path, "imagery_config.yaml"))

    preparation_config_path = rel_path("examples/setup_config_template.yaml")
    shutil.copy(preparation_config_path, os.path.join(new_study_path, "setup_config.yaml"))

    profile_template_path = rel_path("snakemake/profiles/hpc/config.yaml")
    os.makedirs(os.path.join(new_study_path, "profile"))
    shutil.copy(profile_template_path, os.path.join(new_study_path, "profile"))

    print(
        f"Template successfully created! Navigate to {new_study_path} and start adjusting your options."
    )


def download_imagery(name, dir):
    """Download imagery through the STAC pipeline"""
    imagery_config_path = os.path.join(dir, name, "imagery_config.yaml")

    with open(imagery_config_path, "r") as f:
        imagery_config = yaml.safe_load(f)

    try:
        run_imagery_dl_pipeline(imagery_config)
    except Exception as e:
        print(f"Error: Download Pipeline terminated with error: {e}")


def run_study(study_dir, study_name):
    """Runs the study with snakemake and the specified profile"""

    config_file = os.path.join(study_dir, study_name, study_name, 'config.yaml')
    profile_dir = os.path.join(study_dir, study_name, "profile")

    validate_run_config(config_file)

    success = snakemake(
        snakefile="vercye_ops/snakemake/Snakefile",
        configfiles=[config_file],
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


@click.command()
@click.argument("mode")
@click.option("--name", required=False, help="Unique name for your yield study.")
@click.option("--dir", required=False, help="Directory of your yieldstudy.")
@click.option("--chirps-dir", required=False, help="Directory to store CHIRPS data.")
@click.option("--chirps-start", required=False, help="Daterange start for CHIRPS data. Format: YYYY-MM-DD.")
@click.option("--chirps-end", required=False, help="Daterange end for CHIRPS data. Format: YYYY-MM-DD")
@click.option("--chirps-cores", required=False, help="Number of cores to use for parallel chirps downloading. Should be max 5.", default=5)
def main(mode, name, dir, chirps_dir, chirps_start, chirps_end, chirps_cores):
    """
    VeRCYe CLI usage instructions:

    MODE: - "init" to initialize a new study and create templates in dir/name. Must be used with --name and --dir.

          - "prep" to create the study base structure from the options specified in your dir/name/setup_config.yaml. Must be used with --name and --dir.

          - "dl-rs" to download RS imagery from a STAC source specified in your dir/name/imagery_config.yaml. Must be used with --name and --dir.

          - "dl-chirps" to download CHIRPS precipitation data into a local shared registry. Must be used with --chirps-dir, --chirps-start and --chirps-end.

          - "run" to run the yield study with the options specified in your dir/name/run_config.yaml with the profile specified in dir/name/profile. Must be used with --name and --dir.

    Example usage:

    vercye init --name ukraine_study_2025-03-20 --dir /home/yieldstudies
    """

    if mode in ['init', 'prep', 'dl-rs', 'run']:
        if not name or not dir:
            raise ValueError(f'Must provide --name and --dir for {mode}')
        
    if mode == 'dl-chirps' and not (chirps_dir and chirps_start and chirps_end):
        raise ValueError('Must provide --chirps-dir, --chirps-start, --chirps-end for dl-chirps.')

    if mode == "init":
        init_study(name=name, dir=dir)
    elif mode == "prep":
        prepare_config = os.path.join(dir, name, "setup_config.yaml")
        prepare_study(prepare_config)
    elif mode == "dl-rs":
        download_imagery(name=name, dir=dir)
    elif mode == "dl-chirps":
        run_chirps_download(start_date=chirps_start, end_date=chirps_end, output_dir=chirps_dir, num_workers=chirps_cores)
    elif mode == "run":
        run_study(study_dir= dir, study_name=name)
    else:
        print('Invalid mode. Mode must be "init", "prepare" or "run".')
