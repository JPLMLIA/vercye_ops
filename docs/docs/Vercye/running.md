# Quickstart - Yield Study

This guide walks you through the process of setting up and running a yield study using our framework, which helps you simulate crop yields across different regions.

## Prerequisites

Before starting, ensure you have:

- Completed the [setup](../index.md#Setup) and installed all requirements
- [Generated LAI data](../LAI/running.md) for your region of interest
- (Optional) If using CHIRPS precipitation data: [Downloaded](metdata.md#chirps) historical and current CHIRPS precipitation data.
- (Optional) If using ERA5 Metdata: Run `earthengine authenticate` and copy the link to your browser to sign into earth engine.

## Setup Process

Setting up a yield study involves three main steps:

1. Defining your regions of interest
2. Configuring your simulation parameters
3. Preparing your base directory structure
4. Optional: Adding reference (validation) data

While the individual steps are detailed in other pages of this living document, on this page we outline the quickstart using our setup helper.


If you already have prepared a base directory and a configuration file, you can skip to step 4.
Otherwise follow the steps below. If you don't have a IDE with remote access, you might find it easier to run the setup (which esentially created a specific directory structure of APSIM templates & GeoJSON files) on **your local machine** and not a remote cluster. You can transfer the final setup to the cluster after creation. In this case, make sure you have installed the python requirements + the vercye package locally, as descriped in the [setup](../index.md#Setup). Otherwise, it is reccomended to use an IDE such as `Visual Studio Code` with the `Remote SSH` extension.

## 1. Defining Regions of Interest

Your yield study will run simulations for each defined region, typically specified in a shapefile.

**Important requirements for your shapefile:**

- Contains geometries at the same administrative level only (e.g., all geometries are districts OR all counties)
- Includes an attribute column with region names (could be for example `NAME_2` containing "Cook County", "Orleans"...)

[Optional] **Only required if your shapefile contains mixed administrative levels**: Use the interactive `remove_mixed_admin_levels.py` script to normalize it to a single level. The script will prompt you for a few inputs to remove mixed administrative levels:
```
python utils/remove_mixed_admin_levels.py --shp_fpath /path/to/your.shp --output_dir /path/to/save/dir
```

## 2. Defining Your Configuration

Create a configuration file that controls simulation parameters:

0. Create an empty directory `new_basedir_path` that will be your basediretory of your study. Recommended to give it a meaningfull name.
1. Navigate to `vercye_ops/examples` and copy the `run_config_template.yaml` configuration to `new_basedir_path/run_config_template.yaml`. Ensure you keep the file named `run_config_template.yaml`.
2. Modify parameters in the new configuration copy (`run_config_template.yaml`) according to your study needs (years, date ranges, meteorological data sources). For now, leave the `regions` fields empty, as it will be filled in by the setup helper. Ensure you read the description of each parameter in detail and refer to the [Inputs documentation (Section 6)](inputs.md#6-Snakemake-Configuration) for more details on a specific parameter.

**Note** The script for matching remotely sensed LAI and APSIM predicted LAI is not publicly available in this repository. You will have to set the path to the true matching script in your `run_config_template.yaml` file under `scripts.match_sim_real`.

The full configuration options are documented in the [Inputs documentation (Section 6)](inputs.md#6-Snakemake-Configuration).

## 3. Setting Up Your Base Directory

The **base directory** (your `study head dir`) organizes region-specific geometries and APSIM simulations by year, timepoint, and region [(See Details)](inputs.md).

Use the provided helper script (`prepare_yieldstudy.py`) to create this structure. For this, simply create an additional `setup_config.yaml` file in your base directory and fill it as described below. You can then run the setup helper with `python prepare_yieldstudy.py /path/to/basedirectory/setup_config.yaml`. For ease of use, start out with the example provided in `examples/setup_config.yaml`.

1. **Input shapefile & region names**

    - **`SHAPEFILE_PATH`**  Path to your `.shp` containing all simulation-level regions (all geometries must share the same admin level).

    - **`ADMIN_COLUMN_NAME`**  Attribute column holding each region’s identifier (e.g. `NAME_2`).

2. **(Optional) Subset regions**

    If you only want a subset of regions (e.g. counties in Texas & Colorado), set:

    - **`FILTER_COL_NAME`** Column for the higher-level admin unit (e.g. `NAME_1`).

    - **`FILTER_COL_VALUES`** List of values to keep, e.g. `['Texas', 'Colorado']`.  
      To include *all* regions, set `FILTER_COL_NAME = None` and leave `FILTER_COL_VALUES = []`.

3. **APSIM configuration templates**

    Rather than manually copying and editing an APSIM file for each year/region, the helper will:

    1. Copy a template for each higher-level region (e.g. state) into every year’s folder.  
    2. Auto-adjust the simulation dates. NOTE: This will replace the `Models.Clock` parameter in the APSIM simulation to with the value specified in the `run_config_template.yaml` under `apsim_params.time_bounds`. If you require different simulation start/end-dates for various regions during a season, you will have to configure this manually in the APSIM files in the extracted directories.

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

- Create your `year/timepoint/region` directory tree under `OUTPUT_DIR`.  
- Generate a final `run_config.yaml` that merges your Snakemake settings with the selected regions.

**Note**: Sometimes, you might want to add some custom conditionals or processing, that is why we have provided this code in a jupyter notebook. In that case make sure to read the [input documentation](inputs.md), to understand the required structure.

## 4. Adding Reported Validation Data

The VeRCYE pipeline can automatically generate validation metrics (e.g., R², RMSE) if reported data is available. To enable this, you must manually add validation data for each year.

Validation data can be provided at different geographic scales. It may be available at the smallest unit (e.g., ROI level used in simulations) or at a coarser level (e.g., government statistics). You must specify the scale so VeRCYE can aggregate predictions accordingly.

Define aggregation levels in your `config file` under `eval_params.aggregation_levels`. For each level, provide a key-value pair where the key is a descriptive name and the value is the column in your original shapefile used for aggregation. For example, if state-level ground truth uses the ADMIN_1 column, specify `State: ADMIN_1`. If the validation data is at ROI level, no specification is needed—it will be automatically recognized.

For each year and aggregation level, create a CSV file named: `{year}/groundtruth_{aggregation_name}-{year}.csv`, where aggregation_name matches the key in your config (case-sensitive!).

Example: For 2024 state-level data, the file should be: `basedirectory/2024/groundtruth_State-2024.csv`
For simulation ROI-level data, use `primary` as the aggregation name: `basedirectory/2024/groundtruth_primary-2024.csv`

**CSV Structure**

- `region`: Name matching GeoJSON folder (for `primary aggregation level`) or matching attribute table column values for custom aggregation level (Column as specified under `eval_params.aggregation_levels` in tour `config.yaml`)
- `reported_mean_yield_kg_ha`: Mean yield in kg/ha
If unavailable, provide `reported_production_kg` instead. The mean yield will then be calculated using cropmask area (note: subject to cropmask accuracy).If you do not have validation data for certain regions, simply do not include these in your CSV.
- If your reference data contains area, it is recommended to also include this under `reported_area` even though this is not yet used in the evaluation pipeline.


## 5. Running the Yield Study

Once your setup is complete:

1. Transfer your base directory to your HPC cluster (if having prepared the study locally).
2. Adjust the `sim_study_head_dir` path in `run_config.yaml` to match the location you copied the directory to.
3. Navigate to the snakemake directory: `cd vercye_ops/vercye_ops/snakemake`.
4. Open a `tmux` session or similar to start the long running job: `tmux new -s vercye`
6. Ensure you have activated your virtual environment if applicable.
7. Run the simulation in the tmux shell (This example will expect to use 110 CPU cores as defined in the `profile` file.). Ensure to not accidentally pass the `run_config_template.yaml` instead of the new `run_config.yaml`:
   ```bash
   snakemake --profile profiles/hpc --configfile /path/to/your/run_config.yaml
   ```

   Note: In the cli arguments simply specify the path to the folder of the `profile.yaml`m not the `.yaml` file itself. See the example command above.

  For custom CPU core allocation, add the `-c` flag (e.g with 20 cores) or adapt the `profiles/hpc/config.yaml` file:
   ```bash
   snakemake --profile profiles/hpc --configfile /path/to/your/config.yaml -c 20
   ```

## Output

When the simulation completes, results will be available in your base directory. See the [Outputs Documentation](outputs.md) for details on interpreting the results.

To run the pipeline over the same region(s), either use Snakemake's `-F` flag or delete the log files at `vercye_ops/snakemake/logs_*`. Runtimes are in `vercye_ops/snakemake/benchmarks`.

## Troubleshooting
This section contains a few tips on what to do if you are encountering errors during pipeline execution.

**Errors** 
A typical error occurs during the execution of the `LAI_analysis` rule if the LAI parameters were not correctly set. This error indicates that in all of your LAI data there are not two single dates that have sufficient pixels without clouds for the specific region.

However, this rarely should be the case when running with LAI data of multiple months (a typical season).