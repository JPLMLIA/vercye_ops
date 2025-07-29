# 🌿 LAI Generation 
The pipeline produces LAI products for VERCYe and is intended for scaled deployment on servers or HPC with minimal human intervention. The Sentinel-2 LAI model is by Fernandes et al. from https://github.com/rfernand387/LEAF-Toolbox. We provide two methods for exporting remotely sensed imagery and deriving LAI products:

- **A:** Exporting RS imagery from **Google Earth Engine** (slow, more setup required, better cloudmasks)
- **B: **Downloading RS imagery through an open source **STAC catalog** and data hosted on AWS or MPC (fast, inferior cloudmasking).

The individual advantages are detailed in the [introduction](intro.md#lai-generation). This document details the instruction on how to download remotely sensed imagery and derive LAI data. For both approaches we provide pipelines that simply require specifying a configuration and then handle the complete process from exporting and downloading remotely sensed imagery to cloudmasking and deriving LAI estimates. Details of the invididual components of the pipelines can be found in the readme of the corresponding folders.


### Prequisites

Install the requirements as detailed in [Introduction - Setup](../index.md#setup)

### A - Google Earth Engine Pipeline

**Step 1: GeoJSON Extraction**

Extract `GeoJSON's` from a shapefile for each region of interest. Typically, you will want to break down large areas into individual geometries, such as from the `Admin 1 or 2` level depending on their size in the country of interest.

While it is possible, to also provide a national scale geometry directly, in the past we noticed the export through GEE to be significantly slower or failing then when processing multiple e.g `admin2` geometries in parallel. We therefore do not reccomend providing a single national geometry.

Use the `vercye_ops/lai/0_build_library.py` helper script to extract individual GeoJSONS from a shapefile. Ensure that the shapefile only contains geometries at the same administrative level (e.g do NOT mix polygons for states and districts in the same shapefile).

```bash
python 0_build_library.py --help
Usage: 0_build_library.py [OPTIONS] SHP_FPATH

  Wrapper around geojson generation func

Options:
  --admin_name_col [str]  Column name in the shapefile\'s attribute table
                          containg the geometries administrative name (e.g NAME_1).
  --output_head_dir DIRECTORY   Head directory where the region output dirs
                                will be created.
  --verbose                     Print verbose output.
  --help                        Show this message and exit.
```

**Step 2: Create your Google Earth Engine Credentials**

Follow the Google Drive Python Quickstart to download a `client_secret.json`: https://developers.google.com/drive/api/quickstart/python

> [!NOTE]  
> Google OAuth requires accessing a server-side browser via X11 forwarding to produce a `token.json` from your `client_secret `. This can get complicated, involving Xming or Xquartz along with the appropriate `$DISPLAY` and `.ssh/config` parameters. It may be easier to just run this locally to produce the `token.json`, then transfer the token to the server. For this, you will have to run `vercye_ops/vercye_ops/lai/lai_creation_GEE/1_1_gee_export_S2.py` with `--export-mode drive` and `--gdrive-credentials /path.to/your/credentials.json` and the `--token-only` flag. You can cancel the run, once you see that the earth engine login is completed. This will then produce the token that you have to transfer to the server. Otherwise, please discuss with your system administrator.

**Step 3: Setup the GEE-LAI Pipeline Configuration**

Create a LAI config file that defines the parameters of your study.

**3.1 Copy** `vercye_ops/lai/lai_creation_GEE/example_config.yaml` to `vercye_ops/lai/lai_creation_GEE/custom_configs/your_config_name.yaml`

**3.2 Set the parameters** in `vercye_ops/lai/lai_creation_GEE/custom_configs/your_config_name.yaml`:

```yaml
# Folder where GeoJSONS from Step1 are stored (output_head_dir from 0_build_library.py)
geojsons_dir: 'lai/regions/'

# Your Earth Engine project ID
ee_project: 'ee-project-id'

# Output directory for the LAI data and intermediate files
output_base_dir: 'outputdir'

# Path to the Google Earth Engine service account credentials
gdrive_credentials_path: '/path/to/client_secret.json'

# Timepoints for which to create LAI
timepoints:
  2023:
      start_date: '2023-10-10'
      end_date: '2024-04-03'
  2024:
    start_date: '2024-10-10'
    end_date: '2025-04-03'

# Spatial resolution in meters. Should typically be 10 or 20.
resolution: 20

# Set to True to merge the LAI produced for each region into single regions daily VRT files
merge_regions_lai: True
combined_region_name: 'merged_regions'
```

> [!NOTE]  
> If you only have very few regions and timepoints it might make sense to split a timepoint into multiple timepoints. E.g instead of having a timepoint with `start_date: '2023-10-10', end_date: '2024-04-03'` you would create multiple timepoints such as `20231: start_date: '2023-10-10', end_date: '2023-12-01'`, `20232: start_date: '2023-12-01', end_date: '2024-02-01'`, and `20241: start_date: '2024-02-01', end_date: '2023-04-03'`. This allows to leverage more parallel processing capabilities, since you are able to submit about 10 jobs in parallel in GEE.


**Step 4: Navigate to the Pipeline**
```bash
cd vercye_ops/lai/lai_creation_GEE
```

**Step 5: Run the Pipeline**
```bash
snakemake --configfile /your/configfile.yaml --cores 10
```

Replace `/your/configfile.yaml` with the actual path to your configuration file from Step 2.

This will orchestrates the following workflow:

- Export Management: Submits up to 10 parallel jobs to GEE for Sentinel-2 mosaic exports from your regions and date ranges
- Smart Downloads: Automatically downloads exported data from Google Drive to your local machine
- Storage Cleanup: Frees up Google Drive storage immediately after download
- Data Standardization: Processes and standardizes the imagery data if it was split into multiple files from GEE
- LAI Generation: Creates LAI products for each processed file
- Optional Merging: Combines regional data into single daily files if specified in your config

**Performance Tuning**
The `--cores` parameter controls parallel processing. While you can adjust this based on your system resources, there's usually no benefit to going beyond 10 cores - that's the maximum number of simultaneous export jobs allowed under GEE's educational and non-profit licenses.

**Output Structure**
Your LAI products will land in different locations depending on your configuration:

- Merged regional data: output_base_dir/merged_regions_lai (single daily files covering all regions)
- Individual regional data: output_base_dir/lai (separate files per region)


### B - STAC Catalog & AWS/Azure Pipeline

The STAC pipeline fetches Sentinel-2 Imagery either from an AWS bucket hosted by Element84 or from Azure in the Microsoft Planetary Computer. For Element84, it uses data from `Sentinel-2 L2A Collection 1`. All this data has been processed with `Baseline 5.0`. However, during downloading we ensure that both the shift in data from `Element84 Earthsearch` and from `Micrososft Planetary Computer` is harmonized to match a baseline < 4.0 to conform the LAI models training data.

To generate daily LAI data for your region of interest follow the steps below:


**Step 1: Define Your Configuration**

Here's an example of how you'd process a multiple years of Morocco data at 20m resolution (Save as `config.yaml`):

```yaml
date_ranges:
  - start_date: "2019-04-01"
    end_date: "2019-06-30"
  - start_date: "2020-03-15"
    end_date: "2020-07-15"
  - start_date: "2021-05-01"
    end_date: "2021-09-30"

resolution: 20
geojson_path: /data/morocco.geojson
out_dir: /data/morocco/lai
region_out_prefix: morocco
from_step: 0
num_cores: 64
chunk_days: 30
imagery_src: "MPC"
keep_imagery: false
```

- `date_ranges`: Define multiple seasonal or arbitrary time windows to process (in YYY-MM-DD format).

- `resolution`: Spatial resolution in meters. (Typically 10 or 20)
- `geojson-path`: Path to your regions of interest geojson. Will create a bounding box for each geometry and query the intersecting tiles.
- `out_dir`: Output directory for all generated data.
- `region_out_prefix`: Prefix for the output VRT filenames - typically the name of the GeoJSON region.
- `from_step`: Controls which part of the pipeline to resume from (0–3). Should be at 0 if not trying to recover a crashed run.
- `chunk_days`: Number of days to process in each batch. Default is 30 days. Can be used to control storage usage by avoiding to keep more than chunk-days of original tile data on disk at once.
- `num_cores`: Number of cores to use. Default is 1 (sequential). Increase for faster processing on multi-core systems.
- `imagery_src`: Imagery Source - Either "MPC" for Microsoft Planetary Computer or "ES_S2C1" for element84 earthsearch sentinel-2 l2a collection1.
- `keep_imagery`: Whether to store the original remotely sensed imagery after LAI was generated from it or not. Typically keep false.

**Step 3: Navigate to the Pipeline**
```bash
cd vercye_ops/lai/lai_creation_STAC
```

**Step 3: Run the LAI Generation Pipeline**
```bash
python run_stac_dl_pipeline.py /path/to/your/config.yaml
```

This will run the following steps:

- Step 0: Download imagery.
- Step 1: Generate LAI for individual tiles & Clean up original imagery after creation.
- Step 2: Reproject all LAI tiles to `EPSG:4326` and ensure exactly the same resolution, by coosing the most frequent expected resolutin after reprojection.
- Step 3: Build final daily VRT mosaics covering all tiles from this day.

**Step 4: Locate your imagery**
After the pipeline finishes , you'll find a `merged-lai` directory in your `out_dir` filled with daily `.vrt` files. Each file contains LAI data for your entire region, covering all tiles that had usable imagery on that date.

Happy LAI generating!
