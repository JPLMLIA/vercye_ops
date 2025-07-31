# VeRCYe Pipeline Inputs Documentation

The VeRCYe pipeline enables large-scale yield studies by organizing simulation data within a structured directory and processing it with `snakemake`. This document details the required input files, directory structure, and configurable parameters. 

---

## 1. Yield Study Setup
To run a yieldstudy, you will need two things:

1. **Base Directory**: A directory that includes all your regions of interest, APSIM configurations and Reference Data.
2. **Configuration**: A configuration file for VeRCYe, specifying different parameters for the study.

We provide an example setup under [Example Setup](vercye_ops/examples/example_setup).


In this document we detail the steps for setting up a yield study and the structure that VeRCYe expects as its input. However, we recommend not doing the setup manually, but instead automating this process by using the `prepare_yieldstudy.py` helper script as outlined in [Running VeRCYe](running.md).

### Simulation Head Directory
Your yield study is structured within a single directory, referred to as the **simulation head directory**. This directory must follow a predefined structure to ensure compatibility with the pipeline.

### Directory Structure
```plaintext
head_dir/
|   config.yaml
|---Year1/
|   |   referencedata_-primary-Year1.csv (optional)
|   |---TimePoint-1/
|   |   |---region1/
|   |   |   |   region1.geojson
|   |   |   |   region1_template.apsimx
|   |---TimePoint-N/
|       |---regionN/
|           |   regionN.geojson
|           |   regionN_template.apsimx
|---Year2/
    ...
```
Each **year** contains **timepoints**, and each timepoint contains **regions** with their respective `geojson` and `apsimx` template files.
The names in this structure are just descriptive placeholders and should be adjusted as described below.
---

## 2. Region GeoJSON Files

### Purpose
Each **Region of Interest (ROI)** is represented as a GeoJSON file within its respective **timepoint** directory.

### Converting Shapefiles to GeoJSON
We expect your data to initially be a **shapefile (.shp)**. To extract individual regions into GeoJSONs, we provide a conversion script.
If your shapefile has

- A: **Geometries from different ddministrative levels:** Use `utils/prepare_shapefile.py` to standardize the shapefile before conversion. Then proceed with B.
- B: **Geoemtries at a single administrative level:** Use `utils/convert_shapefile_to_geojson.py` if the shapefile has a uniform administrative level.

> [!Warning]
> Ensure your shapefiles contains only geometries at the same administrative level if skipping `prepare_shapefile.py`!

### File Naming Convention
Each GeoJSON file must follow the format:
```plaintext
regionname.geojson
```

For example, if studying California, the file structure would be:
```plaintext
2024/T-0/california/california.geojson
```
If you do not use our conversion from shapefile to GeoJSON, you will need to manually ensure that each GeoJSON contains a centroid column that has the same format as created in `utils/convert_shapefile_to_geojson.py` and contains a column `cleaned_region_name_vercye`, containing the file's name.

---

## 3. Years and Timepoints

### Defining Years
Years should be named numerically (e.g., `2024`, `2025`) to represent different simulation periods.

### Defining Timepoints
Each **timepoint** represents a simulation scenario, e.g., using all available meteorological data vs. limiting it to 30 days before the latest observation.

Each timepoint must define:

- **APSIM Simulation Start & End Dates**
- **Meteorological Data Start & End Dates**
- **LAI (Leaf Area Index) Data Start & End Dates**

Years and Timepoints are referenced in `config.yaml` using their respective names. Therefore the folder names must match these.

---

## 4. APSIMX Templates

### Purpose
Each region and timepoint requires an **APSIMX template** (`regionname_template.apsimx`). This file defines crop and soil and other parameters for APSIM. The dates (`Models.Clock`) in the APSIM file must align with the simulation dates set in `config.yaml`! This is often the only change that needs to be applied for using the same APSIM files for different years.

If you are not using the `prepare_yieldstudy.py` helper script, you will likely want to copy the same file to a number of directories (regions):
```
cd path/to/apsim_simulation_20240607/2021/T-0  # Path where regions exist (e.g., from above shapefile conversion)
source_file="/path/to/my/template.apsimx"; for dir in *; do cp "$source_file" "${dir}/${dir}_template.apsimx"; done
```

**Adjustments for e.g soil properties and simulation constraints must be manually configured with domain knowledge.**

---

## 5. Validation Data (Optional)
If ground-truth yield data is available, it should be included as `referencedata_**_**{AggregationLevel}**-**{Year}.csv` in the corresponding year directory.
Hereby, `aggregation level`, specifies how the simulation level regions should be aggregated and must align with the `eval_params.aggregation_levels` in your `config.yaml`. The `year` must match the corresponding year directory.

### Reference CSV Specification
| Column Name               | Description |
|---------------------------|-------------|
| `region`                  | Name matching GeoJSON folder (for `primary aggregation level`) or matching attribute table column values for custom aggregation level |
| `reported_mean_yield_kg_ha` | Mean yield (kg/ha), if available |
| `reported_production_kg`        | Total production (kg) (optional, used to derive mean yield) |

If `reported_production_kg` is provided, the mean yield is computed as:
```plaintext
mean_yield_kg_ha = reported_production_kg / cropland_area_ha
```
with cropland_area_ha being the **computed cropland_area_ha** based on the provided cropland map!.

Validation data is optional and can also be provided for a subset of years where it is available. 
In this case a total reference value is ommited to avoid misinterpretation.

---

## 6. Snakemake Configuration
This file defines the study parameters and links the **simulation head directory** with the pipeline. You will have to adapt this for each yield study. An example configuration can be found here [example configuration file](/vercye_ops/examples/run_config_example.yaml). We reccomend, to use this as a template for adjustment. The following section describes the meaning of the paramters, but does not represent the exact syntax for how to organize the config. For this, please refer to the example.

### Key Parameters

#### General Settings
- `platform`: Choose between `'local'` or `'umd'` for dependency management.
- `sim_study_head_dir`: Path to the **simulation head directory**.
- `regions`: List of included regions. Must match folder names of regions.
- `years`: List of included years (`int`). Must match year folder names.
- `timepoints`: List of included timepoints. Must match timepoint folder names.
- `study_id`: Id/Name of that identifies the study. Freely choosable up to a length of 25 characters.
- `keep_apsim_db_files`: Delete actual APSIM DB files after processing and reporting to free space.(`True`/`False`).
- `create_per_region_html_report`: Create HTML reports of predicted yield curves per region. Requires a lot of space. (`True`/`False`).
- `title`: Free choice of a title, used as the heading in the report.
- `description`: Custom description (freetext) of the study.
- `lai_source`: Name of the shapefile used for LAI generation. Not used for processing, simply for reference.
- `regions_shp_name`: Name of the shapefile from which the GeoJSONs where created. Not used for processing, simply for reference.


#### APSIM Parameters (`apsim_params`)
- `met_source`: `'NASA_POWER'` or `'ERA5'`. If using `ERA5` ensure you call `earthengine authenticate` from your terminal before starting the pipeline.
- `met_agg_method`" `'centroid'` or `'mean'`. Aggregation method for the pixels within a region. `NASA_Power` only supports `centroid`.
- `precipitation_source`: Choose between `'NASA_POWER'` or `'CHIRPS'`. If you are using CHIRPS, you will have to manually download the precipitation data before starting the pipeline (apsim/download_chirps_data.py).
- `precipitation_agg_method`: Aggregation method for precipitation data (`mean` or `centroid`). `'NASA_POWER'` only supporting `centroid` currently.
- `fallback_precipitation`: Set `True` to use the original precipitation data (`NASAPower or ERA5`) if `CHIRPS` is unavailable. CHIRPS only provides coverage from -50 to 50 degrees.
- `chirps_dir`: Directory containing global CHIRPS precipitation data (tiffs).
- `nasapower_cache_dir`: Directory (typically outside of your yieldstudy, globally) containing already fetched dates of nasapower met data per region.
- `era5_cache_dir`: Directory (typically outside of your yieldstudy, globally) containing already fetched dates of nasapower met data per region.
- `ee_project`: Project ID in google earth engine. Required if using ERA5 meteorological data.
- `align_np_grid`: Weather to query only the closest NASA Power gridcell centroid instead of the exact lat/lon. Improves caching, however, solar radiation is currently incorrect (See met data section in the docs).
- `sowing_date_col`: Optional. If the shapefile/geojsons have a column containing the sowing date as YYYY-MM-DD, use this in the simulation. Set to empty string if not using.
- `time_bounds`: Defines timepoint parameters:
  - `sim_start_date`, `sim_end_date`: Start/End date of the simulation in APSIM.
  - `met_start_date`, `met_end_date`: Start/End date from when to include metereological data into APSIM.

#### LAI Parameters (`lai_params`)
- `lai_dir`: Directory for LAI data.
- `lai_region`: Region name prefix matching the LAI-file naming convention (`{region}_{date}_{resolution}m_LAI.tif`).
- `lai_resolution`: Spatial resolutin, used for matching as above. In meters/pixel.
- `crop_name`: Specify crop (`wheat` or `maize`). Is related to the cropname defined in APSIM.
- `use_crop_adjusted_lai`: Adjust LAI data for the crop specified (`True`/`False`).
- `lai_analysis_mode`: Set to `'raster'`.
- `time_bounds`: LAI start and end dates for each year.
- `lai_aggregation`: Method how to choose a representative LAI value from a region of pixels. `'mean'` or `'median'`.
- `smoothed`: Whether to smoth the original remotely sensed LAI curve using Sav-Gol Method. (`True`/`False`).
- `file_ext`: LAI files extension. If produced with GEE pipeline and regions are not merged, use `tif`. Else use `vrt`. (`tif`/`vrt`).
- `min_cropland_pixel_threshold`: Minimum number of cropland pixels each simulation ROI must contain. If region has less it is skipped. Cropmask pixels are counted at the resolution of the LAI data.
- `cloudcov_threshold`: Maximum cloud coverage percentage of a region for LAI of a date to be considered valid. E.g if set to 0.9, 90% of the pixels in the LAI data of a specific date must not be clouds/snow, otherwise this date will be ignored in the LAI curve.
- `crop_mask`: `Dict[year, cropmaskpath]`. Path to crop mask files for each year. The cropmask **must** be binary, with 1 indicating the crop and 0 otherwise.
- `do_cropmask_reprojection`: The cropmask must match the resolution and CRS of the LAI data. If set to `True` it will be automatically reprojected to match the LAI data. (`True`/`False`).


#### Matching Parameters (`matching_params`)
- `target_crs`: CRS string for coordinate reference system used for area calculation. Either a proj string (e.g `'"+proj=aea +lat_1=29 +lat_2=36 +lat_0=32.5 +lon_0=-5 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"'`) or authority string (e.g `'epsg:1234'`). Should be choosen with care to ensure **equal area**. If using a proj string ensure to enclose it with additional quotes as in the example.

#### APSIM Execution (`apsim_execution`)
- `use_docker`: Set `True` to run APSIM in Docker.
- `docker.image`: Docker image (`apsiminitiative/apsimng`). Should already have been downloaded.
- `docker.platform`: Device platform (e.g., `'linux/amd64'`).
- `local.executable_fpath`: Path to the local APSIM executable. Follow the setup instruction to install it and replace this path if not using docker.
- `local.n_jobs`: Number of threads (`-1` uses APSIM default).


#### Evaluation Parameters
- `aggregation_levels`: Allows evaluation at multiple administrative levels: Dictionary where the keys are names for levels at which results should be aggregated for evaluation and values are columns in the shapefile allowing to aggregate simulation level results. 

**IMPORTANT**: Neither the key nor the value may contain colons! If this is the case, please ensure you rename your shapefile.

**Example:** The shapefile might have a column `Admin2` and the simulations are run at `Admin3`. So all predictions that share the same `Admin2` region would be aggregated and metrics would be additionally computed for this aggregation.

#### Scripts Configuration
- Paths to scripts used within the Snakemake pipeline. See the example for details.
Relative paths are relative to the `Snakemake` file (! not the config file!).