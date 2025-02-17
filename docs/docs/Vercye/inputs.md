To run the vercye pipeline, a number of inputs are required. In the following, we define the formats of input files and the input parameters that can be set via the snakemake config file.

### Yield Study Setup Overview
Your yield study is organized within a single directory referred to as the `simulation head directory`. The directory structure is outlined below:

```
head dir
|   snakemake_config.yaml
|---Year1
|   |---TimePoint-1
|   |   |---region1
|   |   |   |   region1.geojson
|   |   |   |   region1_template.apsimx
|   |   |---region2
|   |   |   |   region2.geojson
|   |   |   |   region2_template.apsimx
|   |   |---regionN
|   |       |   regionN.geojson
|   |       |   regionN_template.apsimx
|   |---TimePoint-N
|       |---region1
|       |   |   region1.geojson
|       |   |   region1_template.apsimx
|       |---region2
|       |   |   region2.geojson
|       |   |   region2_template.apsimx
|       |---regionN
|           |   regionN.geojson
|           |   regionN_template.apsimx
|---Year2
|   |---TimePoint-1
|   |   |---region1
|   |   |   |   region1.geojson
|   |   |   |   region1_template.apsimx
|   |   |---region2
|   |   |   |   region2.geojson
|   |   |   |   region2_template.apsimx
|   |   |---regionN
|   |       |   regionN.geojson
|   |       |   regionN_template.apsimx
|   |---TimePoint-N
|       |---region1
|       |   |   region1.geojson
|       |   |   region1_template.apsimx
|       |---region2
|       |   |   region2.geojson
|       |   |   region2_template.apsimx
|       |---regionN
|           |   regionN.geojson
|           |   regionN_template.apsimx
```

The following sections describe the components of this structure, including the role of `snakemake_config.yaml`, region naming conventions, `GeoJSON` creation, timepoints, and `apsimx` templates.

### Region GeoJSONS
The pipeline allows execution across multiple Regions of Interest (ROIs) in a batch process. Each ROI is processed individually, and aggregated metrics across all regions are provided.

Each ROI must be represented by an individual GeoJSON file named in the format `regionname.geojson`. For every year-timepoint combination, the directory must contain a subfolder for each region, containing its corresponding GeoJSON file. For example, if studying regions in the US, a timepoint folder may contain a folder named `california` with a file `california.geojson`.

**I only have a .shp shapefile and instead of GeoJSON - What to do?**
If you only have .shp shapefiles instead of GeoJSON files, use the provided helper scripts to convert them:

__Option A: Standardized, Single Administrative Level Shapefile__

- Your shapefile contains geometries at a single administrative level (e.g., districts).
- It has a column named admin_name with the names of the regions.
- Use the apsim/convert_shapefile_to_geojson script to extract individual GeoJSON files.

__Option B: Mixed Administrative Levels__

- Your shapefile lacks the admin_name column or contains mixed levels of geometries.
- Run the apsim/prepare_shapefile.py script to standardize the shapefile, then proceed with Option A to extract GeoJSONs.

The documentation will soon be updated on how to process the shapefiles in more depth.


### Years and Timepoints
**Years**
Most likely you will want to run your yield study over multiple seasons, for example to observe the change over the years. Therefore, you can definie different years in the `simulation head directory`. These should be names like the year (e.g `2024`).

**Timepoints**
A timepoint specifies simulation configurations within that year. For example you might want to run one simulation that considers all metereological data available one simulation that only considers metereological data up until 30 days before the latest data, to observe forecasting abilities. 

Each timepoint within a year is defined by 6 parameters:
- The start date of the of the APSIM simulation
- The end date of the APSIM simulation
- The start date of the metereological data for that region to use
- The end date of the metereological data for that region to use
- The start date from when to use remotely sensed LAI data
- The end date from when to use remotely sensed LAI data

These parameters are set in in the `snakemake_config.yaml` later on and should be referred to with the name given in the `year folder` in the `simulation head directory`. So for example in the experiment described above you might want to create two timepoints called `T-0` and `T-30` as folders in each `year folder`.

### APSIMX Templates
Each region in each timepoint must contain its own `apsimx` template file called `regionname.apsimx`, so for the example above `california_template.apsimx`. The `apsimx` template defines the parameters for the yield simulation of this region with [`APSIMX`](https://github.com/APSIMInitiative/ApsimX).

This is the only step that requires expert knowledge to set region specific parameters for example for the soil characteristics.
For every template at the different timepoints, you will have to ensure that the dates of the apsimx template file match up with the desired simulation dates specified in your snakemake config (both in `apsim_params` and `lai_params` sections). We will provide details on this in the future.

### Snakemake Configuration File Parameters
Now that you know about all the files and their names required to create the `simulation head directory`, we will define the `snakemake_config.yaml`. This file is the heart of the pipeline and links the `simulation head directory` with the actual definition of yield study parameters.

Take a look at the [example configuration](https://github.com/JPLMLIA/vercye_ops/blob/main/vercye_ops/snakemake/config_local.yaml) for the format of the parameters. The following solely gives some background on the parameters.

- platform: Specify 'local' or 'umd' for dependency handling.
- sim_study_head_dir: The path to the simulation head directory.
- regions: List of regions to include (must match region folder names).
- years: List of years to include (must match year folder names).
- timepoints: List of timepoints to include (must match timepoint folder names).
- apsim_params:
    - precipitation_source: You can either use the precipitation data for the APSIM simulation from `NASA_POWER` or from `CHIRPS`. If you are using `CHIRPS`, you will have to manually download the precipitation data before starting the pipeline (`apsim/download_chirps_data.py`).
    - precipitation_agg_method: When using `CHIRPS` you can aggregate the precipitation data over all CHIRPS pixels within the region as their mean, or you can use the centroid of the region. `NASA_POWER` currently only allows using the centroid.
    - fallback_on_nasa_power_centroid: CHIRPS only provides coverage from -50 to 50 degrees. Therefore if the region partly falls outside of these bounds, `NASA_POWER` precipitation data can be used instead if setting to True.
    - chirps_dir: The directory where the downloaded CHIRPS precipitation data is stored. Should be downloaded with `apsim/download_chirps_data.py`.
    - time_bounds: Has to contain the definition of each timepoint for every year specified above. Hereby the timepoint names have to match the names specified above. For each timepoint the following parameters have to be set:
        - sim_start_date: Start date of the simulation in APSIM
        - sim_end_date: End date for the simulation in APSIM
        - nasa_power_start_date: Start date from when to include metereological data into APSIM.
        - nasa_power_end_date: End date up to when to include meteorological data in to APSIM.
- lai_params:
    - lai_dir: Root directory where remotely sensed LAI data (from the LAI pipeline) is stored 
    - lai_region: Region name, such that the files in `lai_dir` match `{region}_{date}_LAI.tif`. Use the LAI pipeline to create such files.
    - crop_name: The name of the crop for which to estimate the yield. Is related to the crop defined in APSIM. Currently "wheat" or "maize" is supprted.
    - use_crop_adjusted_lai: Whether to apply the adjustment to the estimated remotely sensed estimated LAI for the crop specified above. `True` or `False`.
    - lai_analysis_mode: 'raster'
    - time_bounds: For every year and timepoint the start and enddate from which to use LAI data. This should usually match the simulation dates.
    - crop_mask: path to the cropmasks for every year defined above.
- matching_params
    - target_epsg: (To be documented).

The following will rarely have to be changed:

- apsim_execution:
    - use_docker: True or False depending on whether you want to run `APSIM` in docker or with the local executable.
    - docker:
        - image: 'apsiminitiative/apsimng'
        - platform: Your devices platform e.g 'linux/amd64'
    - local: 
        executable_fpath: Path to the APSIM executable on your machine
        n_jobs: Number of threads to spawn. -1 is default in APSIM.
- scripts: The paths to the individual scripts used in the snakemake pipeline.