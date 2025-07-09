import os
import shutil

import click
import yaml
from snakemake import snakemake

from vercye_ops.lai.lai_creation_STAC.run_stac_dl_pipeline import (
    run_pipeline as run_imagery_dl_pipeline,
)
from vercye_ops.prepare_yieldstudy import prepare_study

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

    config_template_path = rel_path("examples/run_config_template.yaml")
    shutil.copy(config_template_path, new_study_path)

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


def run_study(study_dir):
    """Runs the study with snakemake and the specified profile"""

    config_file = os.path.join(study_dir, "run_config.yaml")
    profile_dir = os.path.join(study_dir, "profile")

    success = snakemake(
        snakefile="vercye_ops/snakemake/Snakefile",
        configfiles=[config_file],
        use_conda=True,
        printshellcmds=True,
        profile=profile_dir,
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
@click.option("--name", required=True, help="Unique name for your yield study.")
@click.option("--dir", required=True, help="Directory of your yieldstudy.")
def main(mode, name, dir):
    """
    VeRCYe CLI usage instructions:

    MODE: - "init" to initialize a new study and create templates in dir/name.

          - "prep" to create the study base structure from the options specified in your dir/name/setup_config.yaml.

          - "dl" to download RS imagery from a STAC source specified in your dir/name/imagery_config.yaml.

          - "run" to run the yield study with the options specified in your dir/name/run_config.yaml with the profile specified in dir/name/profile.

    Example usage:

    vercye init --name ukraine_study_2025-03-20 --dir /home/yieldstudies
    """
    if mode == "init":
        init_study(name, dir)
    elif mode == "prep":
        prepare_config = os.path.join(dir, name, "setup_config.yaml")
        prepare_study(prepare_config)
    elif mode == "dl":
        download_imagery(name, dir)
    elif mode == "run":
        run_study(name, dir)
    else:
        print('Invalid mode. Mode must be "init", "prepare" or "run".')
