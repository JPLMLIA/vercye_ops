# Quickstart - Yield Study

This guide walks you through the process of setting up and running a yield study using our framework, which helps you simulate crop yields across different regions.

## Prerequisites

Before starting, ensure you have:

- Completed the [setup](../index.md#Setup) and installed all requirements
- [Generated LAI data](../LAI/running.md) for your region of interest
- (Optional) [Downloaded](metdata.md#chirps) historical and current CHIRPS precipitation data. Only if using CHIRPS precipitation.

## Setup Process

Setting up a yield study involves three main steps:

1. Defining your regions of interest
2. Configuring your simulation parameters
3. Preparing your base directory structure
4. Adding reference (vlidation) data

While the individual steps are detailed in other pages of this living document, on this page we outline the quickstart using our setup helper.

## 1. Defining Regions of Interest

Your yield study will run simulations for each defined region, typically specified in a shapefile.

**Important requirements for your shapefile:**

- Contains geometries at the same administrative level only (e.g., all geometries are districts OR all counties)
- Includes an attribute column with region names (e.g., `NAME_2` containing "Cook County", "Orleans Parish")

If your shapefile contains mixed administrative levels, use the interactive helper script to normalize it to a single level:
```
python apsim/prepare_shapefile.py --shp_fpath /path/to/your.shp --output_dir /path/to/save/dir
```

## 2. Defining Your Configuration

Create a configuration file that controls simulation parameters:

0. Create an empty directory `new_basedir_path` that will be your basediretory of your study.
1. Navigate to `snakemake/example_setup/` and copy one of the example configurations to `new_basedir_path/config_template.yaml`.
2. Modify parameters according to your study needs (years, date ranges, meteorological data sources). For now, leave the regions fields empty, as it will be filled in by the setup helper.

The full configuration options are documented in the [Inputs documentation (Section 6)](inputs.md#6-Snakemake-Configuration).

## 3. Setting Up Your Base Directory

The **base directory** (your study head dir) organizes region-specific geometries and APSIM simulations by year, timepoint, and region [Details](inputs.md)

Use the provided Jupyter notebook (`vercye_setup_helper.ipynb`) to create this structure – just set the parameters below and run it.

1. **Input shapefile & region names**

    - **`SHAPEFILE_PATH`**  Path to your `.shp` containing all simulation-level regions (all geometries must share the same admin level).

    - **`ADMIN_COLUMN_NAME`**  Attribute column holding each region’s identifier (e.g. `NAME_2`).

2. **(Optional) Subset regions**

    If you only want a subset of regions (e.g. counties in Texas & Colorado), set:

    - **`FILTER_COL_NAME`** Column for the higher-level admin unit (e.g. `NAME_1`).

    - **`FILTER_COL_VALUES`** List of values to keep, e.g. `['Texas', 'Colorado']`.  
      To include *all* regions, set `FILTER_COL_NAME = None` and leave `FILTER_COL_VALUES = []`.

3. **Intermediate & output folders**

    - **`GEOJSONS_FOLDER`** Temporary folder where the notebook extracts each region as a GeoJSON polygon.

    - **`OUTPUT_DIR`** Your new base directory. **Place** `config_template.yaml` (your Snakemake config) here.

    - **`SNAKEFILE_CONFIG`** Path to that prefilled `config_template.yaml` (it lives in `OUTPUT_DIR/config_template.yaml`; you can leave its `regions:` field empty).

4. **APSIM configuration templates**

    Rather than manually copying and editing an APSIM file for each year/region, the helper will:

    1. Copy a template for each higher-level region (e.g. state) into every year’s folder.  
    2. Auto-adjust the simulation dates.

    Configure this by setting:

    - **`APSIM_TEMPLATE_PATHS_FILTER_COL_NAME`** Admin column that groups regions sharing a template (e.g. `NAME_1`).

    - **`APSIM_TEMPLATE_PATHS`** Dictionary mapping column values to template paths; e.g.  
      ```yaml
      APSIM_TEMPLATE_PATHS:
        Texas:    /path/to/texas_template.yaml
        Colorado: /path/to/colorado_template.yaml
      ```

    - **Single-template setup** If you only require one APSIM file for *all* regions, set:  
      ```yaml
      APSIM_TEMPLATE_PATHS_FILTER_COL_NAME: None
      APSIM_TEMPLATE_PATHS:
        all: /your/path/to/generalApsimTemplate.yaml
      ```



Once all parameters are defined, run the notebook. It will:

- Create your year/timepoint/region directory tree under `OUTPUT_DIR`.  
- Generate a final `config.yaml` that merges your Snakemake settings with the selected regions.

## 4. Adding Reported Validation Data

The VeRCYE pipeline can automatically generate validation metrics (e.g., R², RMSE) if reported data is available. To enable this, you must manually add validation data for each year.

Validation data can be provided at different geographic scales. It may be available at the smallest unit (e.g., ROI level used in simulations) or at a coarser level (e.g., government statistics). You must specify the scale so VeRCYE can aggregate predictions accordingly.

Define aggregation levels in your config file under eval_params.aggregation_levels. For each level, provide a key-value pair where the key is a descriptive name and the value is the column in your original shapefile used for aggregation. For example, if state-level ground truth uses the ADMIN_1 column, specify State: ADMIN_1. If the validation data is at ROI level, no specification is needed—it will be automatically recognized.

For each year and aggregation level, create a CSV file named: `{year}/groundtruth_{aggregation_name}-{year}.csv`, where aggregation_name matches the key in your config (case-sensitive).

Example: For 2024 state-level data, the file should be: `basedirectory/2024/groundtruth_State-2024.csv`
For simulation ROI-level data, use `primary` as the aggregation name: `basedirectory/2024/groundtruth_primary-2024.csv`

Each CSV must include:

- `region`: Region name, matching the cleaned names in your yield study
- `reported_mean_yield_kg_ha`: Mean yield in kg/ha
If unavailable, provide `reported_production_kg` instead. The mean yield will then be calculated using cropmask area (note: subject to cropmask accuracy).If you do not have validation data for certain regions, simply do not include these in your CSV.


## 5. Running the Yield Study

Once your setup is complete:

1. Transfer your base directory to your HPC cluster (if using one)
2. Verify the `sim_study_head_dir` path in `config.yaml` matches your the location you copied the directory to
3. Navigate to the snakemake directory: `cd vercye_ops/vercye_ops/snakemake`
4. Run the simulation (This will expect to use 120 CPU cores.):
   ```bash
   snakemake --profile profiles/hpc --configfile /path/to/your/config.yaml
   ```

5. For custom CPU core allocation, add the `-c` flag (e.g with 20 cores):
   ```bash
   snakemake --profile profiles/hpc --configfile /path/to/your/config.yaml -c 20
   ```

## Output

When the simulation completes, results will be available in your base directory. See the [Outputs Documentation](outputs.md) for details on interpreting the results.

To run the pipeline over the same region(s), either use Snakemake's `-F` flag or delete the log files at `vercye_ops/snakemake/logs_*`. Runtimes are in `vercye_ops/snakemake/benchmarks`.