# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

VeRCYe Ops is the operational implementation of the Versatile Crop Yield Estimate (VeRCYe) pipeline. It estimates crop yield per region by: generating per-pixel Leaf Area Index (LAI) from satellite imagery, simulating crop development with APSIM (a process-based biophysical model), matching simulated LAI curves against observed LAI, and converting the best matches into pixel-level yield maps that are aggregated per region and evaluated against ground truth.

The repo has three layers:
- **Core library** (`vercye_ops/`) — standalone Python scripts, each exposed and self-documented via `--help`. Stable, used operationally.
- **Snakemake pipeline** (`vercye_ops/snakemake/Snakefile`) — orchestrates the scripts into a reproducible, resumable, parallel workflow. This is the heart of the system.
- **Webapp** (`vercye_webapp/`) — FastAPI + Celery + Redis backend with a React/Vite frontend that wraps the library for UI-driven runs. Functional but not security-audited; intended for local/secure deployments only.

## Environment & install

Conda is the only fully supported install path (GDAL, Snakemake, Earth Engine, torch and other native deps). pip-only installs are not supported.

```bash
conda env create -f environment/environment.yaml        # prod env, name: vercye
# or environment/environment.dev.yaml                   # dev env (lint/test tooling), name: vercye-dev
conda activate vercye
pip install -e .                                          # installs the `vercye` console script
```

APSIM (Docker-based or local binary) and Google Earth Engine auth (`earthengine authenticate`, or a service account via `EE_SERVICE_ACCOUNT_KEY`) are required for full runs.

## Common commands

```bash
# Tests (pytest, configured in pyproject.toml). Run from repo root.
pytest
pytest -m "not slow"              # skip tests that load heavy resources (PyTorch models)
pytest tests/lai/test_x.py::test_name   # single test

# Formatting / linting (pre-commit; line length 120, black + isort + flake8)
pre-commit run --all-files

# Pipeline DAG visualization
snakemake --configfile <config.yaml> --rulegraph | dot -Tpdf > dag.pdf
snakemake --configfile <config.yaml> --report report.html   # post-run report with runtimes
```

### Running a yield study (the `vercye` CLI)

The CLI (`vercye_ops/cli.py`, entrypoint `vercye_ops.cli:main`) wraps the study lifecycle. `--dir` (studies base dir) is optional if set via `.env`.

```bash
vercye init  --name <study> --dir <studies_dir>   # scaffold study dir with config templates
vercye prep  --name <study> --dir <studies_dir>   # build the per-region/year/timepoint dir structure from setup_config.yaml
vercye lai   --name <study> --dir <studies_dir>   # download imagery + generate daily LAI (STAC pipeline)
vercye chirps --name <study> --dir <studies_dir>  # download CHIRPS precipitation into a shared registry
vercye run   --name <study> --dir <studies_dir>   # run the full Snakemake pipeline
vercye run   --name <study> --dir <studies_dir> --validate   # validate run_config only, don't run
# Any extra args after `run` are passed through to snakemake (e.g. --touch, --forceall).
```

### Webapp

```bash
cd vercye_webapp && ./run.sh   # builds frontend, starts Redis, Celery workers, Uvicorn on a unix socket
cd vercye_webapp/frontend && npm install && npm run dev   # frontend dev server (Vite)
```

## Architecture notes

### The Snakemake pipeline is the core abstraction
`vercye_ops/snakemake/Snakefile` (~1300 lines) defines the entire flow as wildcard-based rules over **region × year × timepoint**. Snakemake resolves which jobs to create from output path patterns rather than explicit enumeration — understanding wildcard expansion is essential to reading the Snakefile. Re-runs only recompute missing/outdated outputs; failed runs resume from the point of failure.

Rough rule order: `reproject_cropmask` → `constrain_cropmask` → `validate_region_has_cropland` → (met data: ERA5/NASAPower/CHIRPS) → `lai_analysis` → `run_apsim` → `match_sim_real` → `generate_converted_lai_map` → `estimate_total_yield` → aggregation → `evaluate_yield_estimates` → reporting (`generate_final_report`, `generate_multiyear_comparison`).

Two things make the Snakefile hard to read, and both are intentional:
- **Checkpointing**: `validate_region_has_cropland` + the `all_regions_validated` checkpoint split the run in two so downstream rules only process regions with sufficient cropland. Downstream rules depend on valid regions via `get_valid_regions` helpers (in `snakefile_helpers.py`), which require the checkpoint to have completed.
- **Conditional dependencies**: e.g. evaluation only runs when reference/ground-truth data is present, wired through `get_evaluation_results_path_func` feeding the final rule. See the `sim_match_report_workaround` input on rule `all` for how extra outputs are forced.

### Configuration drives everything
A study is a directory containing YAML configs. Key files (paths resolved via `vercye_ops/utils/env_utils.py`):
- `setup_config.yaml` — input to `prep`; describes regions, templates, structure to generate.
- `run_config.yaml` — the pipeline config (data sources, CRS/resolution, year/timepoint ranges, parallelization caps, script paths). Validated by `snakemake/config_validation.py` before each run.
- `lai_config.yaml` — STAC imagery download + LAI generation config.
- `profile/config.yaml` — Snakemake profile (core counts etc.); the only built-in profile template is `snakemake/profiles/hpc/`.

`prep` runs into a temp dir then content-aware syncs into the real study dir (`utils/file_sync.py`) so unchanged files keep their mtimes and don't needlessly invalidate downstream Snakemake rules. Preserve this behavior when touching `prep`.

### LAI generation has two backends
- `vercye_ops/lai/lai_creation_STAC/` — the current/default path (Planetary Computer / EarthSearch STAC, HLS or S2). Driven by `run_stac_dl_pipeline.py`. Numbered scripts (`1_download_*`, `2_1_primary_LAI_tiled.py`, `2_2_standardize.py`, `2_3_build_daily_LAI_vrts.py`) run in sequence.
- `vercye_ops/lai/lai_creation_GEE/` — older Google Earth Engine + Google Drive export path, with its own `Snakefile`.

LAI prediction uses trained models in `vercye_ops/lai/trained_models/` (loaded via `lai/model/`). Tests touching these are marked `slow`.

### Pipeline step modules
- `apsim/` — builds `.apsimx` templates and injects met data / sowing dates (`update_apsimx_template.py`, `construct_met_files.py`).
- `met_data/` — ERA5 (`fetch_era5.py`, via GEE), NASAPower (`fetch_nasapower.py`, global cache to dodge rate limits), CHIRPS (`download_chirps_data.py`).
- `matching_sim_real/` — matches simulated vs observed LAI (`match_sim_rs_lai.py`), generates yield maps, estimates/aggregates total yield. **Note:** the true `match_sim_real` matching script is not in this public repo; its path must be set in `run_config.yaml` under `scripts.match_sim_real`.
- `evaluation/` — MAE/RMSE/R²/relative-RMSE against ground truth.
- `reporting/` — per-timepoint PDF reports, aggregated CSVs/maps, multi-year comparisons, interactive visualizations.

### Webapp wiring
`vercye_webapp/main.py` (FastAPI) serves the React build and mounts routers under `/api` (`studies`, `lai`, `cropmasks`, `analysis`). `worker.py` is a Celery worker that calls the same library functions (`init_study`, `prepare_study`, `run_study`) the CLI uses — the webapp is a thin orchestration layer, not a reimplementation. Long-running snakemake processes are launched in their own process group (`os.setsid`) so the webapp can terminate them via `os.killpg`; the PGID is written to `snakemake_task_id.txt`. Study status is tracked in a status file (`update_study_status`) with states like running/completed/failed/cancelling/cancelled.

## Conventions
- Python ≥3.10, formatted with black (line length 120) and isort (black profile); flake8 with `--max-line-length=120 --extend-ignore=E203,E501`. pre-commit also runs gitleaks and detect-private-key.
- Library scripts are click-based CLIs — keep each script independently runnable with documented `--help`.
- `version.py` is generated by setuptools_scm; don't hand-edit.
