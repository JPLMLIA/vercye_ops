# VERCYE LAI Pipeline

## Description

This LAI Pipeline produces LAI products for VERCYE. It is intended for scaled deployment on servers or HPC with minimal human intervention.

The Sentinel-2 LAI model is by Fernandes et al. from https://github.com/rfernand387/LEAF-Toolbox

The pipeline steps are as follows, and the scripts are named accordingly:
1. Retrieving regional S2 rasters from GEE
  1. Export S2 rasters from GEE to Google Drive
  2. Download rasters from Google Drive to server
  3. Build VRT mosaics of raster parts
2. Produce regional LAI rasters from regional S2 rasters
3. Produce analysis products for a specified subregion from regional LAI rasters
4. Produce a quicklook to validate products

There are some auxiliary scripts marked with `0` that are provided for reference:
- Reprojecting a cropmask to the LAI CRS, resolution, and extent
- Exporting the S2 LAI model by Fernandes et al. from CSV to pytorch 
- Building the regional geojson library

## Python Package Dependencies
- Most included in [`requirements.txt`](../../requirements.txt)
- For `1_2_gdrive_dl_S2.py`, or downloading files from Google Drive via the API:
  - `pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib`

# Pipeline Steps

## 1.1 GEE Export of S2 Rasters

> [!TIP]
> Step 1 only needs to be run once per-region per-date.

> [!WARNING]  
> This step requires a Google Earth Engine project with a linked Google Cloud Project.

This step uses Google Earth Engine to export the Sentinel 2 rasters of a given region to Google Drive. Only the following bands required to predict LAI are exported: `['cosVZA', 'cosSZA', 'cosRAA', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']`

Rasters are currently exported at `20m`, and lowering the resolution would improve runtime and reduce filesize accordingly. However, since this is step is a one-time cost, it is recommended to run this at the highest resolution as the primary product.

### Usage

```
$ python 1_1_gee_export_S2.py --help
Usage: 1_1_gee_export_S2.py [OPTIONS]

Options:
  --project TEXT           GEE Project Name
  --library DIRECTORY      Local Path to the library folder
  --region TEXT            Region name (without apostrophes)
  --shpfile PATH           Local Path to the shapefile to override region
  --start-date [%Y-%m-%d]  Start date for the image collection
  --end-date [%Y-%m-%d]    End date for the image collection
  --resolution NUMBER      Spatial resolution
  --help                   Show this message and exit.
```

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

## 1.2 Download S2 Rasters from Google Drive

> [!TIP]
> This step is optional and is intended for large-scale deployment. For a fewer number of files, downloading directly from the Google Drive website from the browser will be simpler.

> [!WARNING]
> This step requires you to follow the Google Drive Python Quickstart to download a `client_secret.json`: https://developers.google.com/drive/api/quickstart/python
> Google OAuth requires accessing a server-side browser via X11 forwarding to produce a `token.json`. This can get complicated, involving Xming or Xquartz along with the appropriate `$DISPLAY` and `.ssh/config` parameters. It may be easier to just run this locally to produce the `token.json`, then transfer the token to the server. Otherwise, please discuss with your system administrator.

This step uses the Google Drive Python API to download all files in a folder to a local directory. This avoids trying to zip and download hundreds of files through the browser, which can take a long time and often fail due to connection interruptions.

### Usage

```
$ python 1_2_gdrive_dl_S2.py --help
Usage: 1_2_gdrive_dl_S2.py [OPTIONS]

  Download files from a Google Drive Folder

  Parameters 
  ---------- 
  secret_json: str
    Filepath to secrets JSON
    downloaded from setting up GDrive Python API
    https://developers.google.com/drive/api/quickstart/python
  folder_id: str
    Folder ID from Google Drive URL
  outdir: str
    Directory path to where files should be downloaded

Options:
  --secret-json PATH
  --folder-id TEXT
  --outdir DIRECTORY
  --help              Show this message and exit.
```

**Example:**
```bash
python 1_2_gdrive_dl_S2.py \
  --secret-json client_secret_12345.json \
  --folder-id 1A2B3C4D \
  --outdir /path/to/download/S2/Poltava/
```

This will use the `client_secret.json` downloaded from following the steps in the [Google Drive Python Quickstart](https://developers.google.com/drive/api/quickstart/python) and the Google Drive folder ID (its URL would have been https://drive.google.com/drive/folders/1A2B3C4D) to download all of its files to `/path/to/download/S2/Poltava/`. A `token.json` will also be produced inside the directory for `client_secret.json`, which will bypass the login screen until it expires and has to be regenerated.

### Notes

- It is strongly recommended to run this inside a `screen` session for the same reasons as `1_1`.
- The authentication is likely the most difficult aspect of this step.

## 1.3 Standardization of S2 Rasters

> [!IMPORTANT]  
> If you skipped **1.2** and downloaded the rasters via the website, all rasters must be unzipped and placed into a single directory.

> [!WARNING]
> VRTs link to the original geotiff files - do not move or delete any `.tif` files until the primary LAI files have been produced.

This step builds VRTs of all rasters of a given region. VRTs are a fast way to build mosaics of tiles, given that these files only need to be accessed once - to generate primary LAI files.

### Usage

```
$ python 1_3_standardize_S2.py --help
Usage: 1_3_standardize_S2.py [OPTIONS] REGION IN_DIR VRT_DIR

  Generate VRTs of all region tifs in a directory

  Parameters
  ----------
  region: str
    String pattern to match when looking for rasters region_*
  in_dir: str
    Directory of region geotiffs
  vrt_dir: str
    Directory to which vrts should be written
  resolution: str
    Spatial resolution. Used for matching rasters region_resolution*

Options:
  --help  Show this message and exit.
```

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

## 2. Primary LAI Raster Generation

> [!TIP]
> Step 2 only needs to be run once per-region per-date.

> [!NOTE]  
> This script assumes that it is being run on a CPU HPC node with lots of RAM. Running this on GPU will be faster, but may face VRAM limitations.
> It is unlikely that this script will work at region-level on a laptop due to the RAM requirements.
> A compute-memory tradeoff is possible, and a version of this script that takes longer to run while using less memory is planned.

This script produces primary region-level LAI rasters using a deep learning model. All future analyses will be based on these rasters. After these rasters have been produced, the Sentinel-2 rasters can be archived or deleted.

The model used to predict LAI is an implementation of Weiss and Baret, 2016. The model weights were retrieved from [GitHub](https://github.com/rfernand387/LEAF-Toolbox/blob/master/LEAF-Toolbox-CCRS-Files/Inputs/Sentinel2/Sentinel_csv/s2_sl2p_weiss_or_prosail_NNT3_Single_0_1.csv), and converted to Pytorch weights for convenience and efficiency.

Note that wheat and maize adjustments are performed during the analysis step, not here.

### Usage

```
$ python 2_primary_LAI.py --help
Usage: 2_primary_LAI.py [OPTIONS] S2_DIR LAI_DIR

  Main LAI batch prediction function

  S2_dir: Local Path to the .vrt Sentinel-2 images

  LAI_dir: Local Path to the LAI estimates

  region: Name of the region. Used to match file names beginning with region_

  resolution: Spatial resolution. Used to match file names beginning with region_resolution

  This pipeline does the following: 1. Looks for Sentinel-2 images in the
  specified directory in the format {geometry_name}_{resolution}m_{date}.vrt 2. Uses the
  pytorch model to predict LAI 3. Exports the LAI estimate to the specified
  directory in the format {geometry_name}_{resolution}m_{date}_LAI.tif

Options:
  --model_weights PATH  Local Path to the model weights
  --help                Show this message and exit.
```

**Example:**

```bash
python 2_primary_LAI.py \
  /path/to/save/VRT/Poltava/ \
  /path/to/save/LAI/Poltava/ \
```

This will use the default model in `models/s2_sl2p_weiss_or_prosail_NNT3_Single_0_1_LAI.pth` to produce LAI rasters from all VRT files in `/path/to/save/VRT/Poltava/` and write them to `/path/to/save/LAI/Poltava/`.

### Notes

- It's recommended to store the LAI products of a single region in a single directory for the next step.
- Once the primary region LAI raster is generated, ***it is safe to delete or archive the Sentinel-2 rasters.***

## 3. LAI Analysis

> [!WARNING]
> Cropmasks require preprocessing to match the grid and resolution of the LAI raster. A script, `0_reproj_mask.py`, is provided to do this. See *Notes* for more details.

This is the main analysis script of the pipeline. While the previous steps may be expensive in time and compute, this step is incredibly fast as it simply crops down from existing regional primary LAI rasters. This means that analysis of different regions and time ranges can be repeated quickly for reprocessing or research.

This step has three modes of operation:
1. `raster` - for analysis with crop masks
2. `poly_agg` - for a `.shp` file that contains multiple polygons, this treats all polygons together.
3. `poly_iter` -for a `.shp` file that contains multiple polygons, this treats each polygon separately.

The steps of the pipeline are as follows:
1. Read the field geometry or cropmask
2. For each field (if `--mode poly_iter`)...
3. For each date in the range...
4. Find the Region Primary LAI raster
  - If it doesn't exist, skip this date
5. Apply the field or crop mask
  - If the extents do not overlap, throw an error
  - If the raster is all NODATA after masking, skip this date
6. Apply wheat or maize adjustments
7. Calculate mean and stddev statistics
8. Update the running maximum raster
9. After all dates are processed, run interpolation
10. Write the stats CSV and the maximum raster

The output products are as follows:
- `region_geometry_start-date_end-date_STATS.csv`
  - This is the input product for APSIM. Its columns are as follows:
    - `Date` - the date in dd/mm/YYYY format
      - When opening the CSV in Excel, this column **will** be misparsed due to differences in regional datetimeformatting. Switching to the ISO standard of YYYY-mm-dd is recommended in the future.
    - `n_pixels` - the number of valid pixels aggregated for this day's statistics.
      - In the future, it may be useful to filter for a minimum value of this column, *prior to interpolation*.
    - `interpolated` - `1` if this date was interpolated with Akima, `0` if this date was observed.
    - `LAI Mean` - Mean LAI aggregate statistic.
    - `LAI Stddev` - Standard Deviation LAI aggregate statistic.
    - `LAI Mean Adjusted` - Adjusted Mean LAI aggregate statistic for wheat or maize.
    - `LAI Mean Stddev` - Adjusted Standard Deviation LAI aggregate statistic for wheat or maize.
  - In the case of multiple polygons, a separate CSV is produced for each polygon only in the `poly_iter` mode.
- `region_geometry_start-date_end-date_MAX.tif`
  - This is the maximum LAI raster across the entire time range. There are 2 bands:
    - `estimateLAImax` - Maximum LAI
    - `adjustedLAImax` - Adjusted maximum LAI for wheat or maize.

### Usage

```
$ python 3_analysis_LAI.py --help
Usage: 3_analysis_LAI.py [OPTIONS] LAI_DIR OUTDIR REGION GEOMETRY_PATH                                                                                                    LAI Analysis function                                                              
  LAI_dir: Local path to the directory containing regional primary LAI rasters

  outdir: Local path to the output directory where analysis products will be
  saved

  region: Name of the primary region from which regions should be cropped

  resolution: Resolution in meters of the LAI. Shoud match with the value used in export.

  This pipeline does the following: 1. For each date in the range start_date
  to end_date 2. Find the primary LAI raster for the given region, resolution and date 3.
  Apply the provided mask as specified by the mode 4. Calculate the
  appropriate LAI statistics for the CSV 5. Calculate a maximum LAI raster for
  the geometry and date range

Options:
  --mode [raster|poly_agg|poly_iter]
                                  What kind of geometry to expect and how to
                                  apply it.                 'raster' expects a
                                  pixelwise mask of zeros and ones.
                                  'poly_agg' expects a .shp or .geojson and
                                  combines all polygons.
                                  'poly_iter' iterates through each polygon.
  --adjustment [none|wheat|maize]
                                  Adjustment to apply to the LAI estimate
  --start_date [%Y-%m-%d]         Start date for the image collection
  --end_date [%Y-%m-%d]           End date for the image collection
  --help                          Show this message and exit.
```

**Field-Individual Example**

```bash
python 3_analysis_LAI.py \
  /path/to/save/LAI/Poltava \
  /path/to/save/output/20240904/single-field \
  Poltava \
  04VA-v_one_field_wheat_2022.shp \
  --mode poly_agg \
  --adjustment wheat \
  --start_date 2022-02-15 \
  --end_date 2022-08-15
```

This looks for LAI Primary rasters that match `Poltava_*.tif` inside `/LAI/Poltava`, then uses the polygon(s) defined in `04VA-v_one_field_wheat_2022.shp` to run its analysis. The `--mode poly_agg` does not affect the result if there is only one polygon in the `.shp` file. The `wheat` adjustment is included in the analysis. The script will look for rasters from `2022-02-15` to `2022-08-15` as specified. The following output products are saved to `20240904/single-field`:
- `Poltava_04VA-v_one_field_wheat_2022_2022-02-15_2022-08-15_STATS.csv`
- `Poltava_04VA-v_one_field_wheat_2022_2022-02-15_2022-08-15_MAX.tif`

**Field-Aggregate Example**

```bash
python 3_analysis_LAI.py \
  /path/to/save/LAI/Poltava \
  /path/to/save/output/20240904/agg-field \
  Poltava \
  Poltava_wheat_fields.shp \
  --mode poly_agg \
  --adjustment wheat \
  --start_date 2022-02-15 \
  --end_date 2022-08-15
```

This looks for LAI Primary rasters that match `Poltava_*.tif` inside `/LAI/Poltava`, then uses the polygon(s) defined in `Poltava_wheat_fields.shp` to run its analysis. The `--mode poly_agg` indicates that all polygons will be considered together. The `wheat` adjustment is included in the analysis. The script will look for rasters from `2022-02-15` to `2022-08-15` as specified. The following output products are saved to `20240904/agg-field`:
- `Poltava_Poltava_wheat_fields_2022-02-15_2022-08-15_STATS.csv`
- `Poltava_Poltava_wheat_fields_2022-02-15_2022-08-15_MAX.tif`

**Field-Iteration Example**

```bash
python 3_analysis_LAI.py \
  /path/to/save/LAI/Poltava \
  /path/to/save/output/20240904/agg-field \
  Poltava \
  Poltava_wheat_fields.shp \
  --mode poly_iter \
  --adjustment wheat \
  --start_date 2022-02-15 \
  --end_date 2022-08-15
```

This looks for LAI Primary rasters that match `Poltava_*.tif` inside `/LAI/Poltava`, then uses the polygon(s) defined in `Poltava_wheat_fields.shp` to run its analysis. The `--mode poly_iter` indicates that each polygon will be analyzed separately, producing its own analysis products. The `wheat` adjustment is included in the analysis. The script will look for rasters from `2022-02-15` to `2022-08-15` as specified. The following output products are saved to `20240904/iter-field`:
- `Poltava_Poltava_wheat_fields_6_2022-02-15_2022-08-15_MAX.tif`
- `Poltava_Poltava_wheat_fields_6_2022-02-15_2022-08-15_STATS.csv`
- `Poltava_Poltava_wheat_fields_7_2022-02-15_2022-08-15_MAX.tif`
- `Poltava_Poltava_wheat_fields_7_2022-02-15_2022-08-15_STATS.csv`
- ...
- (244) Files

**Cropmask Example**

```bash
python 3_analysis_LAI.py \
  /path/to/save/LAI/Poltava \
  /path/to/save/output/20240904/cropmask \
  Poltava \
  NASA_Harvest_WinterCereals_2022_Poltava_reproj.tif \
  --mode raster \
  --adjustment wheat \
  --start_date 2022-02-15 \
  --end_date 2022-08-15
```

This looks for LAI Primary rasters that match `Poltava_*.tif` inside `/LAI/Poltava`, then uses the cropmask raster `NASA_Harvest_WinterCereals_2022_Poltava_reproj.tif` to run its analysis. *Note that this cropmask was produced with `0_reproj_mask.py`.* The `--mode raster` indicates that this is a cropmask. The `wheat` adjustment is included in the analysis. The script will look for rasters from `2022-02-15` to `2022-08-15` as specified. The following output products are saved to `20240904/cropmask`:
- `Poltava_NASA_Harvest_WinterCereals_2022_Poltava_reproj_2022-02-15_2022-08-15_STATS.csv`
- `Poltava_NASA_Harvest_WinterCereals_2022_Poltava_reproj_2022-02-15_2022-08-15_MAX.tif`

### Notes

- This step requires a lot of RAM for large regions, which most laptops do not have. Field-scale analysis should be fine.
  - There is a way to reduce the memory footprint - instead of reading in the entire raster and using a 1x1-kernel CNN, read one pixel at a time and use an ANN only if said pixel is not NODATA. This will likely take an order of magnitude longer due to the IO overhead  but use far less memory. An update to the model architecture in `0_csv_to_pytorch.py` and `2_primary_LAI.py` (replace `nn.Conv2d()` with `nn.Linear()`, that's it) as well as a new dataloader (instead of feeding the raster in as an array) would be necessary.
- This is a relatively fragile script, where a lot of things could go wrong. Here is a list of anticipated issues, with suggested fixes:
  - `.shp` formatting for `poly_iter` - the script will attempt to fallback to an incrementing index of the attribute `FID` does not exist.
    - If the name of the attribute is different, replace any occurence of `'FID'` with the new key.
  - If the `.shp` geometry does not overlap the LAI raster extent at all, it will skip.
    - If this is occurring for every single date, it probably means you specified the wrong region for the primary LAI raster.
  - In some cases, GEE decides to export a raster with a smaller extent than expected to save filesize. When a field polygon or cropmask extent is larger than that of the LAI raster, the LAI raster must be padded. `pad_to_polygon()` and `pad_to_raster()` attempt to do this, but cases where the padding is off by 1 are possible.
    - Process all region LAI rasters to have the same extent, or correct the padding functions as necessary. Specifically, `np.ceil()` may be incorrect when calculating padding in some cases.
- It is suggested to change the date formatting in the output CSV from `dd/mm/YYYY` to `YYYY-mm-dd` to follow the ISO standard and avoid Excel import and other confusion. Look for the `datetime.strptime()` call around `L156`.
- There is currently no filter on when to accept or reject a measurement based on how many valid pixels were observed. This should likely be based on percentage - For example, if a polygon containing 20000 pixels only has 100 pixels within it due to cloud cover, it should probably be rejected.
  - This needs to occur in the script *prior to interpolation*. The best place to do this would be at the step of appending to statistics around `L280` - calculate `n_pixels` first and proceed accordingly.
- Akima Spline interpolation is currently implemented, but it can be modified (to use `makima`, [for example](https://blogs.mathworks.com/cleve/2019/04/29/makima-piecewise-cubic-interpolation/)) or replaced entirely by changing `L319` or thereabouts.
- Crop masks ***MUST*** be preprocessed to match the resolution, grid, and extent of the LAI region raster. A script to do this is provided as `0_reproj_mask.py` and is described below.
  - This script currently only checks if the crop mask and LAI rasters have the same resolution - it will behave unexpectedly if the resolution is the same but the extents do not line up. It will pad LAI rasters that are smaller in extent than the cropmask because it is expected behavior - however, if the cropmask raster is smaller than the LAI, it will likely fail due to inconsistent array sizes. `0_reproj_mask.py` corrects this by matching the cropmask's extent to that of the LAI raster.
  - For cropmasks that are *significantly* smaller than the region LAI raster, this can be annoying - future development may allow the LAI raster to be cropped down to the extent of the cropmask. This may be as simple as removing the `max(pad, 0)` in `pad_to_raster()` and allowing padding values to go negative. If the values are negative, slice instead of padding. However, this can be complicated if some axes have to be padded, and others have to be cropped.
- This list will be updated as other cases occur.

## 4. Produce a Quicklook

This script produces a quicklook PNG to quickly validate that no major errors occurred during anlaysis. The PNG is saved to the same directory as the CSV and raster output products.

### Usage

```
$ python 4_quicklook.py --help
Usage: 4_quicklook.py [OPTIONS] CSV_PATH MAX_PATH

  Produces a quicklook of LAI analysis results

  This script produces a quicklook plot of the LAI curve and the maximum
  raster to verify the successful completion of the script.

Options:
  --help  Show this message and exit.
```

**Examples**

![Field Quicklook](figs/Poltava_04VA-v_one_field_wheat_2022_2022-02-15_2022-08-15_QUICKLOOK.png)

![Cropmask Quicklook](figs/Poltava_NASA_Harvest_WinterCereals_2022_Poltava_reproj_2022-02-15_2022-08-15_QUICKLOOK.png)

## Auxiliary Scripts

### 0_reproj_mask.py

This script reprojects a cropmask to match the CRS, resolution, and extent of a given LAI raster.

```
$ python 0_reproj_mask.py --help
Usage: 0_reproj_mask.py [OPTIONS] MASK_PATH LAI_PATH OUT_PATH

  Reprojects a crop mask to the LAI raster

  mask_path is reprojected to match the projection, extent, and resolution of
  LAI_path. It is then saved as out_path.

Options:
  --help  Show this message and exit.
```

**Example:**
```
$ python 0_reproj_mask.py \
  NASA_Harvest_WinterCereals_2022_Poltava.tif \
  LAI/Poltava_2022-02-15_LAI.tif \
  Winter_2022_Poltava_reprojected.tif
```

### 0_build_library.py

This script takes a single `shp` file of region-defining polygons and exports each one to a `geojson` for future reference.
The shapefile attribute data must contain a column for the administrative division name (region) called "admin_name". The file is only allowed to contain entries that specify geometries for this administrative level. You can also use the `prepare_shapefile.py` script in the `apsim` directory to preprocess your shapefile to follow these conventions. If using the script, ensure you use the newly created file from the script in the following steps. We recommend to additionally manually ensure that your shapefile does not mix entries from different administrative levels.

```
$ python 0_build_library.py --help
Usage: 0_build_library.py [OPTIONS] SHP_FPATH

  Wrapper around geojson generation func

Options:
  --output_head_dir DIRECTORY   Head directory where the region output dirs
                                will be created.
  --verbose                     Print verbose output.
  --help                        Show this message and exit.
```

### 0_csv_to_pytorch.py

This script converts a LAI model defined as a CSV for GEE to a pytorch model weights file.

```
$ python 0_csv_to_pytorch.py --help
Usage: 0_csv_to_pytorch.py [OPTIONS] CSV_PATH
                           {LAI|fAPAR|fCOVER|CCC|CWC|Albedo|DASF}

  Convert a CSV file of model weights to a PyTorch model file. csv_path: Path
  to the CSV file containing the model weights. model: The model type to
  convert. One of ['LAI', 'fAPAR', 'fCOVER', 'CCC', 'CWC', 'Albedo', 'DASF']

Options:
  --help  Show this message and exit.
```

**Example:**
```
$ python csv_to_pytorch.py \
  s2_sl2p_weiss_or_prosail_NNT3_Single_0_1.csv \
  LAI
```