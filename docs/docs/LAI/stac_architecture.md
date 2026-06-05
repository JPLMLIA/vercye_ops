# STAC LAI Pipeline Architecture

This page documents the internals of the STAC-based LAI pipeline
(`vercye_ops/lai/lai_creation_STAC`), in particular **how the pipeline is
structured into steps** and **which data formats each step produces**. For
instructions on how to *run* the pipeline, see
[Generating LAI Data](running.md#b-stac-catalog-awsazure-pipeline).

The entrypoint is `run_stac_dl_pipeline.py`, which reads a single YAML config
and orchestrates all steps as subprocesses. It can be resumed from a given
step via the `from_step` config option (valid values: `0`, `2`, `3`, `4`).

## Pipeline steps

| Step | Script | Purpose |
|------|--------|---------|
| 0 | `1_download_S2_*.py` | Query the STAC catalog for Sentinel-2 tiles intersecting the ROI and download the required bands. |
| 1 | `2_1_primary_LAI_tiled.py` | Run the LAI neural network per tile to produce per-tile LAI rasters. |
| 2 | `2_2_standardize.py` | Reproject every LAI tile to a common CRS/resolution and quantize to Int16. |
| 3 | `2_3_build_daily_LAI_vrts.py` | Build one mosaicked daily VRT per date, clipped to the ROI extent. |
| 4 | (in `run_stac_dl_pipeline.py`) | Finalize `meta.json` (record available dates, set status). |

Steps 0 and 1 run per date-range chunk (`chunk_days`); steps 2–4 run once over
the whole time span after all chunks are processed.

## Output directory structure

Everything lands under the configured `out_dir`:

```
out_dir/
├── region.geojson        # Copy of the ROI used, for reproducibility
├── meta.json             # Status, available dates, processed date-ranges, centroid
├── tiles/                # (transient) Raw downloaded Sentinel-2 band tiles
├── tile-lai/             # (transient) Per-tile LAI rasters
├── standardized-lai/     # Per-tile LAI, reprojected + quantized (persisted)
└── merged-lai/           # Daily VRT mosaics — the final LAI product
```

> The `tiles/` and `tile-lai/` directories are intermediate. With
> `keep_imagery: false` (the default), the raw imagery is removed after LAI is
> computed, and the per-tile LAI files are removed by step 2 once standardized
> versions exist. In a normal completed run only `standardized-lai/` and
> `merged-lai/` remain.

## Data formats per stage

The pipeline deliberately changes representation between stages. The table
below summarizes each persisted raster format:

| Stage | Dir | Driver | Dtype | CRS | NoData | Scale | Tiling | Overviews |
|-------|-----|--------|-------|-----|--------|-------|--------|-----------|
| Downloaded imagery | `tiles/` | GeoTIFF | Int16 | native (UTM) | source-dependent | — | yes | no |
| Per-tile LAI | `tile-lai/` | GeoTIFF (LZW) | Float32 | native (UTM) | `NaN` | none | 256×256 | no |
| Standardized LAI | `standardized-lai/` | GeoTIFF (ZSTD, predictor 2) | **Int16** | **EPSG:4326** | `-32768` | **0.001** | 512×512 | **no** |
| Daily mosaic | `merged-lai/` | **VRT** | **Float32** | EPSG:4326 | `NaN` | applied (0.001) | inherited | inherited |

### Standardized tiles (`standardized-lai/*.tif`)

These are the canonical persisted pixel data. Key conventions
(see `2_2_standardize.py`):

- **Reprojected to `EPSG:4326`** and resampled to a single common resolution
  (the most frequent resolution across tiles after reprojection), so that all
  tiles are stackable without on-the-fly warping downstream.
- **Stored as Int16 with `scale_factor = 0.001`.** A stored value of `1234`
  therefore represents an LAI of `1.234`. This roughly halves storage vs.
  Float32 while preserving 3-decimal precision over the `0 … 32.767` range.
- **NoData sentinel is `-32768`.** Negative LAI values produced by the model
  are clipped to NoData during quantization.
- Compressed with **ZSTD** (predictor 2, level 19) and internally **tiled at
  512×512**. The `scales`/`offsets` metadata is written so any
  scale-aware reader returns physical LAI values automatically.

### Daily VRT mosaics (`merged-lai/*.vrt`) — the final product

One VRT per date, named `{region_out_prefix}_{resolution}m_{YYYY-MM-DD}_LAI.vrt`.
Each VRT mosaics together every standardized tile that had usable imagery on
that date, clipped to the ROI's bounding box (`gdalbuildvrt -te ... -tap`).

Although the underlying tiles are Int16, the VRT is **patched** (see
`vrt_patcher.py`) so that downstream readers see:

- `dataType="Float32"`
- `NoDataValue=nan`
- the `0.001` scale already applied (via per-source `ScaleRatio`/`NODATA`)

In other words, reading a `merged-lai` VRT yields ready-to-use floating-point
LAI with `NaN` for missing pixels — no manual rescaling required.

> **VRTs reference their source tiles by absolute path.** Do not move or delete
> `standardized-lai/` while the VRTs are in use, and be aware that the VRTs are
> only directly usable on a machine that can see those exact paths.

## Bring-your-own-LAI compatibility

Any external LAI product can be dropped in as long as it matches the naming and
spatial conventions described in
[Bring your own LAI data](intro.md#c-bring-your-own-lai-data): a consistent
prefix, resolution, `YYYY-MM-DD` date, identical CRS / extent / resolution
across files, and the same `0.001` scale convention used here.
