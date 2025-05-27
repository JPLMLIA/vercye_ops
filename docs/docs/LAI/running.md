# LAI Generation
The pipeline produces LAI products for VERCYe and is intended for scaled deployment on servers or HPC with minimal human intervention. The Sentinel-2 LAI model is by Fernandes et al. from https://github.com/rfernand387/LEAF-Toolbox. We provide two methods for exporting remotely sensed imagery and deriving LAI products:


A: Exporting RS imagery from **Google Earth Engine** (slow, better cloudmasks)
B: Downloading RS imagery through an open source **STAC catalog** and data hosted on AWS (fast, inferior cloudmasking).

The individual advantages are detailed in the [introduction](intro.md#lai-generation). This document details the instruction on how to download remotely sensed imagery and derive LAI data. For both approaches we provide pipelines that simply require specifying a configuration and then handle the complete process from exporting and downloading remotely sensed imagery to cloudmasking and deriving LAI estimates. Details of the invididual components of the pipelines can be found in the readme of the corresponding folders.


### Prequisites

Install the requirements as detailed in [Introduction - Setup](../index.md#setup)

### A - Google Earth Engine Pipeline

**Step 1: GeoJSON Extraction**
Extract geosjsons from a shapefile for each region of interest. Typically, you will want to break down large areas into individual geometries, such as from the `Admin 1 or 2` level depending on their size in the country of interest.

While it is possible, to also provide a national scale geometry directly, in the past we noticed the export through GEE to be significantly slower then when processing multiple e.g `admin2` geometries in parallel. We therefore do not reccomend providing a single national geometry.

**Step2: Create your Google Earth Engine Credentials**
Follow the Google Drive Python Quickstart to download a `client_secret.json`: https://developers.google.com/drive/api/quickstart/python

> [!NOTE]  
> Google OAuth requires accessing a server-side browser via X11 forwarding to produce a `token.json`. This can get complicated, involving Xming or Xquartz along with the appropriate `$DISPLAY` and `.ssh/config` parameters. It may be easier to just run this locally to produce the `token.json`, then transfer the token to the server. For this, you will have to run `vercye_ops/vercye_ops/lai/lai_creation_GEE/1_1_gee_export_S2.py` with `--export_mode drive` and `--gdrive_credentials /path.to/your/credentials.json` and some other dummy parameters. You can cancel the run, once you see that the earth engine login is completed. This will then produce the token that you have to transfer to the server. Otherwise, please discuss with your system administrator.

**Step 3: Setup the GEE-LAI Pipeline Configuration**
Create a LAI config file that defines the parameters of your study.

**3.1 Copy** `vercye_ops/lai/lai_creation_GEE/example_config.yaml` to `vercye_ops/lai/lai_creation_GEE/custom_configs/your_config_name.yaml`

**3.2 Set the parameters** in `vercye_ops/lai/lai_creation_GEE/custom_configs/your_config_name.yaml`:

```yaml
# Folder where GeoJSONS from Step1 are stored
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
> If you only have very few regions and timepoints it might make sense to split a timepoint into multiple timepoints. E.g instead of having a timepoint with `start_date: '2023-10-10', end_date: '2024-04-03'` you would create multiple timepoints such as `start_date: '2023-10-10', end_date: '2023-12-01'`, `start_date: '2023-12-01', end_date: '2024-02-01'`, and `start_date: '2024-02-01', end_date: '2023-04-03'`. This allows to leverage more parallel processing capabilities, since you are able to submit about 10 jobs in parallel.


**Step 4: Navigate to the Pipeline**
```bash
cd vercye_ops/lai/lai_creation_GEE
```

**Step 5: Run the Pipeline**
```bash
snakemake --configfile /your/configfile.yaml --cores 10
```

Replace `/your/configfile.yaml` with the actual path to your configuration file from Step 2.

**What Happens In The Pipeline?**
The pipeline orchestrates a sophisticated workflow:

- Export Management: Submits up to 10 parallel jobs to GEE for Sentinel-2 mosaic exports from your regions and date ranges
- Smart Downloads: Automatically downloads exported data from Google Drive to your local machine
- Storage Cleanup: Frees up Google Drive storage immediately after download
- Data Standardization: Processes and standardizes the imagery data if it was split into multiple files from GEE
- LAI Generation: Creates LAI products for each processed file
- Optional Merging: Combines regional data into single daily files if specified in your config

**Performance Tuning**
The --cores parameter controls parallel processing. While you can adjust this based on your system resources, there's usually no benefit to going beyond 10 cores - that's the maximum number of simultaneous export jobs allowed under GEE's educational and non-profit licenses.

**Output Structure**
Your LAI products will land in different locations depending on your configuration:

- Merged regional data: output_base_dir/merged_regions_lai (single daily files covering all regions)
- Individual regional data: output_base_dir/lai (separate files per region)


### B - STAC Catalog & AWS Pipeline

The STAC pipeline fetches Sentinel-2 Imagery from an AWS bucket hosted by Element84. It uses data from `Sentinel-2 L2A Collection 1`.
All this data has been processed with `Baseline 5.0`.

To generate daily LAI data for your region of interest follow the steps blow:

**Step 1: Prep Your Area of Interest**
Create a GeoJSON of the convex hull of your study area for simplicity. You can do this for example in QGIS by first running
`Vector -> Geoprocessing Tools -> Dissolve`, followed by `Vector -> Geoprocessing Tools -> Convex Hull`. Then export the layer as `GeoJSON`.

**Step 2: Navigate to the Pipeline**
```bash
cd vercye_ops/lai/lai_creation_STAC
```

**Step 3: Create a Config** 
```bash
python run_stac_dl_pipeline.py [OPTIONS]
```

```bash
Options:
  --start-date [%Y-%m-%d]    Start date from when to download imagery in YYYY-MM-DD format.  [required]
  --end-date [%Y-%m-%d]      End date from when to download imagery in YYYY-MM-DD format.  [required]
  --resolution INTEGER       Spatial resolution in meters.  [required]
  --geojson-path PATH        Path to the GeoJSON file defining the region.
                             [required]
  --out-dir DIRECTORY        Output directory for all generated data.
                             [required]
  --region-out-prefix TEXT   Prefix for the output VRT filenames - typically the name of the GeoJSON region.  [required]
  --from-step INTEGER RANGE  Pipeline step to start from (0: download, 1: tile
                             LAI, 2: cleanup, 3: VRT build). Use on failure toresume.  [0<=x<=3]
  --num-cores INTEGER        Number of cores to use. Default is 1
                             (sequential). Increase for faster processing on
                             multi-core systems. Only used for LAI parallelism.
  --chunk-days INTEGER       Number of days to process in each batch. Default
                             is 30 days. Can be used to control storage usage
                             by avoiding to keep more than chunk-days of
                             original tile data on disk at once.
```

**Example**
Here's how you'd process a year of Morocco data at 20m resolution:
```bash
python run_stac_dl_pipeline.py --start-date 2020-01-17 --end-date 2021-01-20 --resolution 20 --geojson-path /data/morocco.geojson --out-dir /data/morrocco/lai --region-out-prefix morocco --num-cores 64
```

**Pipeline Steps Breakdown**
- Step 0: Download imagery from AWS
- Step 1: Generate LAI for individual tiles
- Step 2: Clean up temporary files
- Step 3: Build final VRT mosaics

The --from-step option is your friend when things don't go as planned - no need to start from scratch!

After the pipeline finishes , you'll find a merged-lai directory in your output folder packed with daily .vrt files. Each file contains LAI data for your entire region, covering all tiles that had usable imagery on that date.

**Tips**

- **Storage Management**: Use --chunk-days to control how much disk space you're using at once
- **Performance**: Increase--num-cores on multi-core systems for faster LAI processing, but be mindful of memory usage. Typically on HPC the memory should not be too much of a concern.
- **Recovery**: If something breaks, use --from-step to pick up where you left off
- **Quality Control**: The pipeline only processes tiles with available imagery, so some dates might be missing

Happy LAI generating! 🛰️🌱
