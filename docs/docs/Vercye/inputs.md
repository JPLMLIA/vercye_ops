# VeRCYe Pipeline Documentation

## Overview
The VeRCYe pipeline enables large-scale yield studies by organizing simulation data within a structured directory and processing it with `snakemake`. This document details the required input files, directory structure, and configurable parameters.

---

## 1. Yield Study Setup

### Simulation Head Directory
Your yield study is structured within a single directory, referred to as the **simulation head directory**. This directory must follow a predefined structure to ensure compatibility with the pipeline.

### Directory Structure
```plaintext
head_dir/
|   snakemake_config.yaml
|---Year1/
|   |   groundtruth.csv (optional)
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
We expect you data to initially be in **.shp (shapefile)** format. Use the provided scripts to convert it:
- **Single Administrative Level:** Use `apsim/convert_shapefile_to_geojson.py` if the shapefile contains a uniform administrative level.
- **Mixed Administrative Levels:** Use `apsim/prepare_shapefile.py` to standardize the shapefile before conversion.

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
If you do not use our conversion from shapefile to GeoJSON, you will need to manually ensure that each GeoJSON contains a centroid column that has the same format as extracted in `convert_shapefile_to_geojson.py`.

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

Years and Timepoints are referenced in `snakemake_config.yaml` using their respective names. Therefore the folder names must match these.

---

## 4. APSIMX Templates

### Purpose
Each region and timepoint requires an **APSIMX template** (`regionname_template.apsimx`). This file defines crop growth parameters and the dates (`Models.Clock`) in the APSIM file must align with the simulation dates set in `snakemake_config.yaml`.

Adjustments for soil properties and simulation constraints must be manually configured with domain knowledge.

---

## 5. Validation Data (Optional)
If ground-truth yield data is available, it should be included as `groundtruth.csv` in the corresponding year directory.

### Refernece CSV Specification
| Column Name               | Description |
|---------------------------|-------------|
| `region`                  | Name matching GeoJSON folder |
| `reported_mean_yield_kg_ha` | Mean yield (kg/ha), if available |
| `reported_yield_kg`        | Total yield (kg) (optional, used to derive mean yield) |

If `reported_yield_kg` is provided, the mean yield is computed as:
```plaintext
mean_yield_kg_ha = reported_yield_kg / cropland_area_ha
```
with cropland_area_ha being the computed cropland_area_ha from VeRCYe.

Validation data is optional and cen also be provided for a subset of years where it is available.

---

## 6. Snakemake Configuration (`snakemake_config.yaml`)
This file defines the study parameters and links the **simulation head directory** with the pipeline. You will have to adapt this fill for each yield study. An example configuration can be found here [example configuration file](https://github.com/JPLMLIA/vercye_ops/blob/main/vercye_ops/snakemake/config_local.yaml). We reccomend, to use this as a template for adjustment. The following section describes the meaning of the paramters, but does not represent the syntac for how to organize the config. For this please refer to the example.

### Key Parameters

#### General Settings
- `platform`: Choose between `'local'` or `'umd'` for dependency management.
- `sim_study_head_dir`: Path to the **simulation head directory**.
- `regions`: List of included regions. Must match folder names of regions.
- `years`: List of included years (`int`). Must match year folder names.
- `timepoints`: List of included timepoints. Must match timepoint folder names.
- `study_id`: Id/Name of that identifies the study. Freely choosable up to a length of 25 characters.
- `keep_apsim_db_files`: Delete actual APSIM DB files after processing and reporting to free space.(`True`/`False`).

#### APSIM Parameters (`apsim_params`)
- `precipitation_source`: Choose between `'NASA_POWER'` or `'CHIRPS'`. If you are using CHIRPS, you will have to manually download the precipitation data before starting the pipeline (apsim/download_chirps_data.py).
- `precipitation_agg_method`: Aggregation method for precipitation data (`mean` or `centroid`). `'NASA_POWER'` only supporting `centroid` currently.
- `fallback_on_nasa_power_centroid`: Set `True` to use `NASA_POWER` data if `CHIRPS` is unavailable. CHIRPS only provides coverage from -50 to 50 degrees.
- `chirps_dir`: Directory containing CHIRPS data.
- `time_bounds`: Defines timepoint parameters:
  - `sim_start_date`, `sim_end_date`: Start/End date of the simulation in APSIM.
  - `nasa_power_start_date`, `nasa_power_end_date`: Start/End date from when to include metereological data into APSIM.

#### LAI Parameters (`lai_params`)
- `lai_dir`: Directory for LAI data.
- `lai_region`: Region name with naming convention (`{region}_{date}_{resolution}m_LAI.tif`).
- `lai_resolution`: Spatial resolutin, used for matching as above. In meters/pixel.
- `crop_name`: Specify crop (`wheat` or `maize`). Is related to the cropname defined in APSIM.
- `use_crop_adjusted_lai`: Adjust LAI data for the crop specified (`True`/`False`).
- `lai_analysis_mode`: Set to `'raster'`.
- `time_bounds`: LAI start and end dates for each year.
- `crop_mask`: Path to crop mask files for each year.

#### Matching Parameters (`matching_params`)
- `target_epsg`: EPSG code for coordinate reference system used for area calculation. Should be choosen with care to minimize distortions.

#### APSIM Execution (`apsim_execution`)
- `use_docker`: Set `True` to run APSIM in Docker.
- `docker.image`: Docker image (`apsiminitiative/apsimng`).
- `docker.platform`: Device platform (e.g., `'linux/amd64'`).
- `local.executable_fpath`: Path to APSIM executable.
- `local.n_jobs`: Number of threads (`-1` uses APSIM default).

#### Scripts Configuration
- Paths to scripts used within the Snakemake pipeline.

