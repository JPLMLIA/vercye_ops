# 🌿 LAI Pipeline Documentation

This pipeline estimates **Leaf Area Index (LAI)** from **Sentinel-2** satellite imagery using Google Earth Engine (GEE) and machine learning. It supports automated and sequential processing for multiple regions.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Running the Complete Pipeline](#running-the-complete-pipeline)
- [Individual Scripts](#individual-scripts)
  - [1. Export Sentinel-2 Data](#1-export-sentinel-2-data)
  - [2. Manual Google Drive Download](#2-manual-google-drive-download)
  - [3. Standardize Data Format](#3-standardize-data-format)
  - [4. Generate LAI Estimates](#4-generate-lai-estimates)
  - [5. Merge LAI Estimates](#5-merge-lai-estimates)
- [Export Options](#export-options)
- [Troubleshooting](#troubleshooting)

---

## 🌍 Overview

The LAI pipeline processes **Sentinel-2 imagery** to generate LAI estimates via the following steps:

1. **Export** data from Google Earth Engine to Google Drive.
2. **Download** exported imagery.
3. **Standardize** data format (VRT).
4. **Generate** LAI using a trained neural network.
5. *(Optional)* Merge multiple regions into daily VRTs.
6. **Additional Scripts** Used for analysis and postprocessing

> Run steps individually or use the **Snakemake workflow** for automation.

> [!WARNING]  
> This requires a Google Earth Engine project with a linked Google Cloud Project.

---

# ⚙️ Running the Complete Pipeline
Use **Snakemake** for full automation as described in the docs.


## 🛠️ Individual Scripts

### 1.1 Export Sentinel-2 Data

**Script:** `1_1_gee_export_S2.py`

This step uses Google Earth Engine to export the Sentinel 2 rasters of a given region to Google Drive. Only the following bands required to predict LAI are exported: `['cosVZA', 'cosSZA', 'cosRAA', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']`

```bash
python 1_1_gee_export_S2.py \
  --project [GEE_PROJECT_ID] \
  --library [PATH_TO_GEOJSON_LIBRARY] \
  --region [REGION_NAME] \
  --start-date [YYYY-MM-DD] \
  --end-date [YYYY-MM-DD] \
  --resolution [10|20] \
  --export-mode [gdrive|gcs] \
  --snow-threshold [PERPIXEL_MAX_SNOWPROB] \
  --cloudy-threshold [PERTILE_MAX_CLOUDCOV] \
  --cs-threshold [PERPIXEL_MAX_CLOUDPROB] \
  [--export-bucket BUCKET_NAME --gcs-folder-path FOLDER_PATH] \
  [--gdrive-credentials PATH_TO_CREDENTIALS --download-folder LOCAL_PATH]
```

**Key Parameters:**
- `--project`: Earth Engine project ID
- `--library`: Path to the folder where the region geojsons are stored
- `--region`: Region name (must match GeoJSON name in the library)
- `--resolution`: Typically 10 or 20 (meters). Depending on which is selected different bands will be downloaded:
For 10m `["cosVZA", "cosSZA", "cosRAA", "B2", "B3", "B4", "B8"]` will be downloaded and for any other resolution 
`["cosVZA", "cosSZA", "cosRAA", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"] will be downloaded. These are the bands that the respective LAI model expects.
- `--export-mode`: Export to `gdrive` or `gcs`

If `--gdrive-credentials` and  `--download-folder` are specified, the script will automatically download the files directly after creation in GDrive to your local machine. This avoids storage limitations in GDrive. If you use this, ensure you follow the same setup instructions as for (2. Manual Google Drive Download)

**Export Options**

| Feature               | Google Drive (`--export-mode gdrive`)               | Google Cloud Storage (`--export-mode gcs`)          |
|-----------------------|-----------------------------------------------------|-----------------------------------------------------|
| ✅ **Pros**           | - Free storage<br>- Easy setup<br>- Auto-download support | - No file size limits<br>- Faster for large exports |
| ⚠️ **Cons**            | - Limited to Storage<br>- Slower for large files<br>- Splits files | - Egress costs apply<br>- Requires GCS bucket setup |
| **Tipps**             | - Use with automatical download+deletion of files after creation by specyfing `gdrive-credentials` and `download-folder` | Check egress costs beforehand |


**Example:**

```bash
python 1_1_gee_export_S2.py \
  --project vercye-1 \
  --library library/ \
  --region Poltava \
  --start-date 2022-04-01 \
  --end-date 2022-05-01 \
  --resolution 20
```

This looks for the file `library/Poltava.geojson` and uses its geometry to export Sentinel 2 rasters at a resolution of 20m between 2022-04-01 and 2022-05-01 to the Google Drive top-level directory `Poltava/`, which will be create if it does not already exist. The files will be named `Poltava_2022-04-01.tif` or, if the files are too big, they will be split into parts such as `Poltava_2022-04-01-0000000000-0000000000.tif`, and so on.

### Notes

- It is strongly recommended to run this in a `screen` or `tmux` instance, as shell interrupts including logging out, timing out, or losing internet connection will also stop the job otherwise.
  - Basic tutorial: https://askubuntu.com/a/542671
- The `--shpfile` flag can be used to specify a `.shp` file instead of looking for an existing region in the library. It will probably use the first Polygon within.
- Running multiple instances of this script will allow rasters to process in parallel instead of in sequence. For example, instead of calling this script with a three-year-long date range, call it three separate times in parallel with a one-year-long date range.

---

### 1.2 Manual Google Drive Download

**Script:** `1_2_gdrive_dl_S2.py`

This step uses the Google Drive Python API to download all files in a folder to a local directory. This avoids trying to zip and download hundreds of files through the browser, which can take a long time and often fail due to connection interruptions. You will have to manually identify the folder IDs in your Google Drive (last component of the URL).

> [!TIP]
> This step is optional and is intended for large-scale deployment. For a fewer number of files, downloading directly from the Google Drive website from the browser will be simpler.

> [!WARNING]
> This step requires you to follow the Google Drive Python Quickstart to download a `client_secret.json`: https://developers.google.com/drive/api/quickstart/python
> Google OAuth requires accessing a server-side browser via X11 forwarding to produce a `token.json`. This can get complicated, involving Xming or Xquartz along with the appropriate `$DISPLAY` and `.ssh/config` parameters. It may be easier to just run this locally to produce the `token.json`, then transfer the token to the server. Otherwise, please discuss with your system administrator.

```bash
python 1_2_gdrive_dl_S2.py \
  --secret-json [PATH_TO_CREDENTIALS] \
  --folder-id [GOOGLE_DRIVE_FOLDER_ID] \
  --outdir [LOCAL_OUTPUT_DIRECTORY]
```

---

### 1.3 Standardize Data Format

**Script:** `1_3_standardize_S2.py`

Organizes and converts Sentinel-2 files into standardized VRT format. Used to create a single file as exports might be split into mutliple files from GEE.

```bash
python 1_3_standardize_S2.py [REGION_NAME] [INPUT_DIR] [VRT_OUTPUT_DIR] [RESOLUTION]
```

> [!IMPORTANT]  
> If you skipped **1.2** and downloaded the rasters via the website, all rasters must be unzipped and placed into a single directory.

> [!WARNING]
> VRTs link to the original geotiff files - do not move or delete any `.tif` files until the primary LAI files have been produced.

**Example:**

```bash
python 1_3_standardize_S2.py \
  Poltava \
  /path/to/download/S2/Poltava/ \
  /path/to/save/VRT/Poltava/ \
  20
```

This will look for all rasters that match `Poltava_20m_*.tif` in `/path/to/download/S2/Poltava/` and mosaic tiles to create `Poltava_20m_*.vrt` in `/path/to/save/VRT/Poltava/`

### Notes

- If a raster was downloaded without any tiling, a VRT is basically a symlink that points to the original file.

---

### 2. Generate LAI Estimates

**Script:** `2_1_primary_LAI_GEE.py`

This script produces primary region-level LAI rasters using a deep learning model. All future analyses will be based on these rasters. After these rasters have been produced, the Sentinel-2 rasters can be archived or deleted.

The model used to predict LAI is an implementation of Weiss and Baret, 2016. The model weights were retrieved from [GitHub](https://github.com/rfernand387/LEAF-Toolbox/blob/master/LEAF-Toolbox-CCRS-Files/Inputs/Sentinel2/Sentinel_csv/s2_sl2p_weiss_or_prosail_NNT3_Single_0_1.csv), and converted to Pytorch weights for convenience and efficiency.

Note that wheat and maize adjustments are performed during the analysis step, not here.

```bash
python 2_1_primary_LAI_GEE.py [S2_DIR] [LAI_DIR] [REGION] [RESOLUTION] \
  [--start_date YYYY-MM-DD] [--end_date YYYY-MM-DD] \
  [--model_weights PATH_TO_WEIGHTS] [--channels CHANNEL_LIST]
```

> [!NOTE]  
> This script assumes that it is being run on a CPU HPC node with lots of RAM. Running this on GPU will be faster, but may face VRAM limitations.
> It is unlikely that this script will work at region-level on a laptop due to the RAM requirements.
> A compute-memory tradeoff is possible, and a version of this script that takes longer to run while using less memory is planned.
> Depending on the resolution provided a different model might be used: Currently there is a specific model for 10m resolution (which was trained on 10m data) and one for all other resolutions which was trained on 20m data.

---

### 5. Optional: Merge LAI Estimates

**Script:** `2_2_build_daily_vrts.py`

Combines LAI files from multiple regions into daily VRTs containing all regions in a single file.
This is often helpful to provide all this data to a VeRCYe in a single yield study.

```bash
python 2_2_build_daily_vrts.py [LAI_DIR] [OUTPUT_DIR] [RESOLUTION] \
  [--region-out-prefix PREFIX] \
  [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]
```

---

### 5. Additional Scripts
TODO Need to fill this back in from Jakes documentation

## 🧩 Troubleshooting

### 🔐 Authentication Issues
- Run `earthengine authenticate`
- Verify GDrive credentials are valid

### 🧠 Memory Errors during LAI prediction
- If using snakemake, reduce the number of cores.
- Split the regions into smaler tiles

---
