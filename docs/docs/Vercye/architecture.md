# 📘 Vercye Architecture Documentation

## 🔍 Overview

The **Vercye system** orchestrates a modular, region-based crop yield simulation pipeline using **Snakemake** for dependency resolution and scalability. Each processing step is encapsulated in a standalone script, executed conditionally and in the correct order.

---

## 📂 Scripts

Each script is self-documented via CLI help using the `--help` flag. For additional specifics, you can run the script directly in the terminal.

---

## ⚙️ Pipeline Execution with Snakemake

Snakemake defines workflows as rules. Each rule specifies expected inputs and outputs, enabling:

- **Automatic dependency resolution** – Only missing or outdated outputs are recomputed.
- **Resumability** – Failed executions can resume from the point of failure.
- **Parallelization** – Regions and timepoints are processed independently, increasing efficiency.

---

## 🧾 Config File

Every pipeline run uses a configuration file (YAML), which defines key parameters like:

- Data sources (e.g., LAI, meteorology)
- Resolution and CRS specs
- Simulation ranges (years, timepoints)
- Runtime limits and job parallelization caps
- Regions

This design ensures reproducibility and clean separation of configuration from logic.

---

## 🔄 Pipeline Logic

The following lists a high level overview of some of the most important features of the snakemake based pipeline steps (non-exhaustive).
The complete logic is defined in `vecrye_ops/snakemake/Snakefile`.

### 1. Cropmask Reprojection
- **Rule**: `reproject_cropmask`
- Aligns the cropmask to match CRS and resolution of LAI data.
- **Requirement**: All LAI files in a year must have the same resolution and CRS.

---

### 2. Region Masking
- **Rule**: `constrain_cropmask`
- Crops the reprojected cropmask to each region for spatial filtering.

---

### 3. Region Validation
- **Rule**: `validate_region_has_cropland`
- Skips regions with too few cropland pixels to avoid wasting compute.
- Followed by a **checkpoint**: `all_regions_validated`, ensuring only valid regions are processed further. This enforces snakemake to first execute all validation jobs before continuing with processing the valid regions only.

---

### 4. Meteorological Data Acquisition

**Supported Sources**:
- **ERA5** (via Google Earth Engine): Max 10 concurrent jobs.
- **NASAPower**: Uses a global cache to avoid API rate limits.
    - First job: One-time cache fill per region for its full date range (single job per region to avoid race conditions in cache write).
    - Followed by region-timepoint-specific fetches.
- **CHIRPS**: Requires global files to be locally available.


---

### 5. LAI Analysis
- **Rule**: `lai_analysis`
- Computes daily LAI stats for the cropland pixels per region (e.g., mean LAI).
- Optimized with windowed reading for performance.

---

### 6. APSIM Simulation
- **Rule**: `run_apsim`
- Simulates many parameter combinations to model yield/LAI curves through APSIM.
- Supports both **Dockerized** and **local APSIM** installations depending on config parameter.

---

### 7. Simulation Matching
- **Rule**: `match_sim_real`
- Compares simulated outputs with observed LAI stats.
- Selects best-matching simulations based on curve similarity.

---

### 8. Yield Map Generation
- **Rule**: `generate_converted_lai_map`
- Uses conversion factors to transform predicted yield into pixel-level raster maps.

---

### 9. Yield Estimation & Aggregation
- **Rule**: `estimate_total_yield`
- Aggregates pixel-level predictions into total yield per region.
- Additional **aggregate rules** summarize results across all regions and timepoints.

---

### 10. Evaluation
- **Rule**: `evaluate_yield_estimates`
- Compares predictions to ground truth using metrics like:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² (coefficient of determination)
  - Relative RMSE

---

### 11. Final Outputs

#### Per-Timepoint:
- PDF report (via `generate_final_report`)
- Aggregated CSVs and maps

#### Across Years:
- **Rule**: `generate_multiyear_comparison`
- Creates multi-year overviews and trend visualizations

### Understanding the Snakefile
In some parts you might find the `Snakefile` hard to interpret directly, which is mainly due to two factors:

- Conditional Logic: Such as only running the evaluation rule if reference data files are found in the directories.
This is implemented a bit hacky through a the `get_evaluation_results_path_func` that checks which timepoints have reference data,
and creating a dependancy for the final rule (`final_report`).

- Checkpointing: This feature from `Snakemake` allows us, to not run all downstream jobs on every region.
By first validating which region has sufficient cropland to be considered for analysis, the pipeline is split into a two-step approach.
However, this comes at the cost of making the Snakefile slightly harder to follow:
For ensuring dependancies only on valid regions instead of all regions in the snakefile,
downstream jobs use `get_valid_regions` helper functions, which requires the checkpoint to have passed.
Additionally, if other outputs should be produced, this becomes slightly hacky due to the way `Snakemake` handles things
(see the `sim_match_report_workaround` input in the rule `all` as an example for how to deal with such cases).

---

## 📎 Pipeline Diagram

See the pipeline diagram `vercye_pipeline_highlevel.png` for a visual reference of the individual rules and their dependancies / outputs.

