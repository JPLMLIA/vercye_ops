import os
from pathlib import Path

import yaml
from dotenv import dotenv_values
from ruamel.yaml import YAML

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_env_file_path() -> str:
    # Env file is in vercye_ops root directory. So 2 dirs up.
    return str(Path(BASE_DIR).parent.parent / ".env")


def is_env_set() -> bool:
    return os.path.exists(get_env_file_path())


def get_env_vars():
    return dotenv_values(get_env_file_path())


def read_studies_dir_from_env():
    env_vars = dotenv_values(get_env_file_path())
    return env_vars.get("STUDY_DIR", None)


def read_lai_dir_from_env():
    env_vars = dotenv_values(get_env_file_path())
    return env_vars.get("LAI_BASE_DIR", None)


def read_cropmasks_dir_from_env():
    env_vars = dotenv_values(get_env_file_path())
    return env_vars.get("CROPMASKS_BASE_DIR", None)


def get_study_path(studies_dir: str, study_name: str):
    return os.path.join(studies_dir, study_name)


def get_run_config_file_path(studies_dir: str, study_name: str):
    return os.path.join(get_study_path(studies_dir, study_name), study_name, "config.yaml")


def get_setup_config_file_path(studies_dir: str, study_name: str):
    return os.path.join(get_study_path(studies_dir, study_name), "setup_config.yaml")


def get_run_profile_path(studies_dir: str, study_name: str):
    return os.path.join(get_study_path(studies_dir, study_name), "profile")


def get_snakemake_rundir_path(studies_dir: str, study_name: str):
    return os.path.join(get_study_path(studies_dir, study_name), "snakemake")


def get_snakemake_runlog_path(studies_dir: str, study_name: str):
    return os.path.join(get_snakemake_rundir_path(studies_dir, study_name), "log.txt")


def get_snakemake_run_status_file_path(studies_dir: str, study_name: str):
    return os.path.join(get_snakemake_rundir_path(studies_dir, study_name), "status.txt")


def update_study_status(studies_dir: str, study_name: str, status: str):
    status_file_path = get_snakemake_run_status_file_path(studies_dir, study_name)
    with open(status_file_path, "w") as status_file:
        status_file.write(status)


def get_lai_config_path(studies_dir: str, study_name: str):
    return os.path.join(get_study_path(studies_dir, study_name), "lai_config.yaml")


def replace_in_file(file_path: str, old_val, new_val):
    with open(file_path, "r") as file:
        content = file.read()

    content = content.replace(old_val, new_val)

    with open(file_path, "w") as file:
        file.write(content)


def get_run_config(studies_dir: str, study_name: str):
    config_file_path = get_run_config(studies_dir, study_name)
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_yaml_ruamel(filepath: str):
    """ "Helper to load a yaml with its actual layout containing comments etc"""
    yaml_loader = YAML()
    yaml_loader.preserve_quotes = True
    with open(filepath, "r") as f:
        return yaml_loader.load(f), yaml_loader
