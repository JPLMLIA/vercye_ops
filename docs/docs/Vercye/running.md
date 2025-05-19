# Running a Yield Study

This guide walks you through the process of setting up and running a yield study using our framework, which helps you simulate crop yields across different regions.

## Prerequisites

Before starting, ensure you have:

- Completed the [setup](../index.md#Setup) with all requirements
- [Generated LAI data](../LAI/running.md) for your region of interest
- (Optional) Downloaded historical and current CHIRPS precipitation data. Only if using CHIRPS precipitation. [Instructions](metdata.md#chirps)

## Setup Process

Setting up a yield study involves three main steps:

1. Defining your regions of interest
2. Configuring your simulation parameters
3. Preparing your base directory structure

## 1. Defining Regions of Interest

Your yield study will run simulations for each defined region, typically specified in a shapefile.

**Important requirements for your shapefile:**
- Contains geometries at the same administrative level only (e.g., all districts, all counties)
- Includes an attribute column with region names (e.g., `NAME_2` containing "Cook County", "Orleans Parish")

If your shapefile contains mixed administrative levels, use the helper script:
```
python apsim/prepare_shapefile.py
```

## 2. Defining Your Configuration

Create a configuration file that controls simulation parameters:

0. Create an empty directory that will be your basediretory of your study.
1. Navigate to `snakemake/example_setup/` and copy one of the example configurations `new_basdir_path/config_template.yaml`.
2. Modify parameters according to your study needs (years, date ranges, meteorological data sources)

The full configuration options are documented in the [Inputs documentation (Section 6)](inputs.md#6-Snakemake-Configuration).

## 3. Setting Up Your Base Directory

The base directory is the heart of your individual study and contains region-specific geometries and APSIM simulation files. It should contain a directory tree with a folder for each year, timepoint and region and holds the actual polygons and APSIM configurations per region. The required structure of this directory is detailed in [Inputs documentation (Section 1)](inputs.md#1-Yield-Study-Setup)


Our setup helper simplifies and guides this setup process. To get startet you will have to:
1. Open the Jupyter notebook `vercye_setup_helper.ipynb` and adapt the following parameters:

- **`SHAPEFILE_PATH`**: The path to your .shp file containing all (simulation level) regions. (Ensure they all geometreies have the same admin level).
- **`ADMIN_COLUMN_NAME`**: The column in the attribute table of the shapefile that contains the name of each region (e.g often `NAME_2`...)

Sometimes, you might only want to run the yieldstudy on a subset of the regions in you shapefile. Let's say your shapefile contains all counties in the US, however you only wish to run it on all counties in Texas and Colorado. You can filter these by specifying:
-** `FILTER_COL_NAME`**: The column in the attribute table holding the higher level administrative names (e.g `NAME_1`) with values such as `Texas` and `Oregon`.
-** `FILTER_COL_VALUES`**: An array of all values that should be kept. So for example here `['Texas', 'Colorado']`.
However, this step is optional, if you do not wish to only run on a subset, leave the array empty and set `FILTER_COL_NAME` to None.

-**`GEOJSONS_FOLDER`**: An intermediate folder into which all geoemtries from the shapefile are initally extracted as polygons. This is **NOT** your base directory (study head dir)!

- **`OUTPUT_DIR`**: Your base directory (study head dir). In this directory your whole setup will be saved. Ensure you have placed the snakemake configuration file in here, with the name `config_template.yaml`.
- **`SNAKEFILE_CONFIG`**: The prefilled configuration file you have setup in the previous step. The regions value may be left empty. It should be saved under `OUTPUT_DIR/config_template.yaml`.

Now an essential part is to have individual APSIM configuration files for every season and region of your yield study. To reduce the manual load of copying each file into into an own directoty and adapting the simulation date range in the APSIM file, the helper script facilitates this process:
Typically you will have APSIM files for different regions if you are running for example a national scale study. However, the general APSIM configuration stays the same throughout the years for each region, and only the simulation dates change in the file.
By specyfing a template for each region the APSIM file template for a region will be copied to each years directory and the dates will be automatically be adjusted in by the helper script.
Now since, we do not want to specify an individual APSIM file for each simulation level region (e.g counties) as these can be houndreds, we typically share a template over a higher admin level (e.g states).
To match these, the following parameters in the config are used:


- **`APSIM_TEMPLATE_PATHS_FILTER_COL_NAME`**: defines the admin column name which holds the values of the higher level adminstrative region (e.g states such as `Texas` or `Colorado`)
- **`APSIM_TEMPLATE_PATHS`**: Is a dictionary mapping each of the values in this admin column to the APSIM templates for the regions sharing this configuration.

Sometimes, you may not want to specify individual APSIM templates for different regions and have only a single APSIM file for all regions in your study. IN this case, you can simple set `APSIM_TEMPLATE_PATHS_FILTER_COL_NAME` to `None` and set the dictionary to `{'all': '/your/path/to/generalApsimTemplate.yaml'}`

Once you have set all these parameters, you are ready to run the notebook. This will create your basedirectory setup and a new file called `config.yaml` that will contain all the regions of interest in addition to the values of your previous configured snakemake configuration file.


## Running the Yield Study

Once your setup is complete:

1. Transfer your base directory to your HPC cluster (if using one)
2. Verify the `sim_study_head_dir` path in `config.yaml` matches your the location you copied the directory to
3. Navigate to the snakemake directory: `cd vercye_ops/vercye_ops/snakemake`
4. Run the simulation (This will expect to use 120 CPU cores.):
   ```bash
   snakemake --profile profiles/hpc --configfile /path/to/your/config.yaml
   ```

5. For custom CPU core allocation, add the `-c` flag:
   ```bash
   snakemake --profile profiles/hpc --configfile /path/to/your/config.yaml -c 20
   ```

## Output

When the simulation completes, results will be available in your base directory. See the [Outputs Documentation](outputs.md) for details on interpreting the results.