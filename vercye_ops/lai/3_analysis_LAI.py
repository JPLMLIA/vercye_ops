import csv
import math
import os.path as op
import sys
import time
import warnings
from datetime import timedelta
from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import rasterio as rio
from pyproj import CRS as ProjCRS
from rasterio.coords import BoundingBox
from rasterio.mask import mask
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import savgol_filter
from shapely import LineString, box


def pad_to_polygon(src, geometry, masked_src):
    """Pads masked_src to the extent of geometry if it is smaller"""
    if not rio.coords.disjoint_bounds(src.bounds, geometry.total_bounds):
        gx = abs(src.res[0])
        gy = abs(src.res[1])
        left_pad = max(0, int(np.round((src.bounds.left - geometry.total_bounds[0]) / gx)))
        bottom_pad = max(0, int(np.round((src.bounds.bottom - geometry.total_bounds[1]) / gy)))
        right_pad = max(0, int(np.round((geometry.total_bounds[2] - src.bounds.right) / gx)))
        top_pad = max(0, int(np.round((geometry.total_bounds[3] - src.bounds.top) / gy)))

        if left_pad + bottom_pad + right_pad + top_pad == 0:
            return masked_src, False
        else:
            print(f"Padding {masked_src.shape} by {[top_pad, bottom_pad, left_pad, right_pad]}")
            padded_src = np.pad(
                masked_src,
                pad_width=((top_pad, bottom_pad), (left_pad, right_pad)),
                mode="constant",
                constant_values=np.nan,
            )
            return padded_src, True

    else:
        print("ERROR: The polygon and the source raster do not overlap.")
        sys.exit(1)


def assert_grids_aligned(src, geom_t, tol_pix=1e-6):
    """
    Ensure src.transform and geom_t describe the same north-up grid,
    up to an integer pixel translation.
    Raises SystemExit with a helpful message if not aligned.
    """
    st = src.transform
    gt = geom_t

    # No rotation/shear (north-up)
    if any(abs(v) > 1e-9 for v in (st.b, st.d, gt.b, gt.d)):
        raise SystemExit("Transforms are rotated/sheared; only north-up grids are supported.")

    # Pixel sizes equal (allow tiny epsilon)
    if not math.isclose(st.a, gt.a, abs_tol=1e-9) or not math.isclose(st.e, gt.e, abs_tol=1e-9):
        raise SystemExit(f"Pixel sizes differ: src {st.a, st.e} vs mask {gt.a, gt.e}")

    # Origin alignment in *pixel units*
    px = abs(st.a)
    py = abs(st.e)

    dx_pix = (gt.c - st.c) / px
    dy_pix = (gt.f - st.f) / py

    mis_x = abs(dx_pix - round(dx_pix))
    mis_y = abs(dy_pix - round(dy_pix))

    if mis_x > tol_pix or mis_y > tol_pix:
        raise SystemExit(
            "Pixel grids are not aligned (origin mismatch).\n"
            f"  LAI origin:      ({st.c}, {st.f})\n"
            f"  Mask origin:     ({gt.c}, {gt.f})\n"
            f"  Pixel size:      ({st.a}, {st.e})\n"
            f"  Misalignment px: dx={dx_pix:.6f} (|Δ|={mis_x:.3g}), "
            f"dy={dy_pix:.6f} (|Δ|={mis_y:.3g})\n"
            "Reproject/snap the mask to the LAI grid."
        )


def build_lai_window_for_cropmask(src, cropmask_bounds, cropmask_shape):
    """
    Read a LAI window aligned to the cropmask extent on the LAI grid.
    """
    if isinstance(cropmask_bounds, BoundingBox):
        bounds_tuple = (
            cropmask_bounds.left,
            cropmask_bounds.bottom,
            cropmask_bounds.right,
            cropmask_bounds.top,
        )
    else:
        bounds_tuple = tuple(cropmask_bounds)

    window = rio.windows.from_bounds(*bounds_tuple, transform=src.transform)
    height, width = cropmask_shape

    lai_window_array_padded = src.read(
        1,
        window=window,
        boundless=True,
        fill_value=np.nan,
        out_shape=(height, width),
        resampling=rio.enums.Resampling.nearest,  # grids should be aligned, so no real resampling occurs
    ).astype(np.float32)

    return lai_window_array_padded


def compute_cloud_snow_percentage(lai_window_array_padded, cropmask_array):
    cropmask_bool = cropmask_array.astype(bool)
    cloud_snow_pixels = np.sum(np.isnan(lai_window_array_padded) & cropmask_bool)
    total_pixels_in_region = np.sum(cropmask_bool)
    cloud_snow_percentage = cloud_snow_pixels / total_pixels_in_region * 100 if total_pixels_in_region > 0 else 0.0
    return cloud_snow_percentage


def mask_lai_with_binary_cropmask(lai_window_array_padded, cropmask_array):
    """
    Apply a 0/1 binary cropmask to a LAI window.
    """
    if lai_window_array_padded.shape != cropmask_array.shape:
        raise ValueError("LAI window and cropmask must have the same shape.")

    cropmask_float = cropmask_array.astype(float)
    cropmask_float[cropmask_float == 0] = np.nan

    masked_src = lai_window_array_padded * cropmask_float

    return masked_src


def clip_negative_lai(lai_array):
    """
    Clip negative LAI values to NaN.
    """
    result = lai_array.astype(float, copy=True)
    result[result < 0] = np.nan
    return result


def compute_lai_adjusted(lai_estimate, adjustment):
    """
    Apply crop-specific adjustment to LAI estimate.
    """
    if adjustment == "wheat":
        return lai_estimate**2 * 0.0482 + lai_estimate * 0.9161 + 0.0026
    elif adjustment == "maize":
        return lai_estimate**2 * -0.078 + lai_estimate * 1.4 - 0.18
    elif adjustment in ("none", None):
        return lai_estimate
    else:
        raise ValueError(f"Unknown adjustment type: {adjustment}")


def compute_basic_stats(arr):
    """
    Compute basic statistics for a LAI array, ignoring NaNs.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = float(np.nanmean(arr))
        median = float(np.nanmedian(arr))
        stddev = float(np.nanstd(arr))
        n_pixels = int(np.sum(~np.isnan(arr)))
    return mean, median, stddev, n_pixels


def update_max_lai(current_max, new_lai):
    """
    Update the running maximum LAI raster.
    """
    if current_max is None:
        return new_lai
    return np.nanmax([current_max, new_lai], axis=0)


def raster_and_mask_intersect(lai_bounds, cropmask_bounds):
    """
    Check if LAI raster and cropmask have a non-empty, non-degenerate intersection.
    """
    bbox_lai = box(*lai_bounds)
    bbox_cropmask = box(*cropmask_bounds)
    intersection = bbox_lai.intersection(bbox_cropmask)
    if intersection.is_empty or isinstance(intersection, LineString):
        return False
    return True


def load_geometries(geometry_path, mode, geometry_name):
    """
    Load geometries based on the specified mode. Can be either raster,
    vector polygons aggregated (if multiple in file), or vector polygons iterated.
    """
    geometries = []
    if mode == "raster":
        print("MODE: raster")
        print(f"{geometry_name} will be read as a raster and used to 0/1 mask the primary raster.")
        print("NOTE: Preprocessing to project the raster mask to the primary LAI raster is required.")
        print("      Read the README and use 0_reproj_mask.py.")
        # The cropmask is expected to be cropped to the region and have the same resolution as the LAI rasters
        with rio.open(geometry_path) as ds:
            geometries.append(
                {
                    "array": ds.read(1),
                    "res": ds.res,
                    "bounds": ds.bounds,
                    "transform": ds.transform,
                    "crs": ds.crs,
                }
            )
    elif mode == "poly_agg":
        print("MODE: poly_agg")
        print(f"{geometry_name} and all of its polygons will be analyzed together.")
        shp = gpd.read_file(geometry_path)
        geometries.append(shp.geometry)
    elif mode == "poly_iter":
        print("MODE: poly_iter")
        print(f"{geometry_name} and its polygons will be analyzed separately.")
        shp = gpd.read_file(geometry_path)
        for s in shp.itertuples():
            try:
                row_dict = {"geometry": s.geometry, "FID": s.FID}
            except Exception:
                print("[WARN] Attribute 'FID' not found, ignoring")
                row_dict = {"geometry": s.geometry}
            row_gdf = gpd.GeoDataFrame([row_dict], geometry="geometry", crs=shp.crs)
            geometries.append(row_gdf)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return geometries


def ensure_output_dirs(output_stats_fpath, output_max_tif_fpath):
    """
    Ensure that the directories for output statistics and maximum LAI TIFF files exist.
    """
    Path(output_stats_fpath).parent.mkdir(parents=True, exist_ok=True)
    if output_max_tif_fpath:
        Path(output_max_tif_fpath).parent.mkdir(parents=True, exist_ok=True)


def validate_maxlai_keep_bands(maxlai_keep_bands):
    """
    Validate the bands to keep for the maximum LAI output.
    """
    bands = list(maxlai_keep_bands)
    if not bands:
        raise ValueError(
            "No bands to keep for the LAI data provided. " "Specify either estimateLAImax or adjustedLAImax."
        )
    return bands


def build_date_range(start_date, end_date):
    """
    Build lists of dates in ISO and slash formats between start_date and end_date.
    """
    all_dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    dates_iso = [d.strftime("%Y-%m-%d") for d in all_dates]
    dates_slash = [d.strftime("%d/%m/%Y") for d in all_dates]
    return dates_iso, dates_slash


def make_empty_stat_row(date_slash, cloud_snow_percentage=None, interpolated=0):
    return {
        "Date": date_slash,
        "n_pixels": 0,
        "interpolated": interpolated,
        "LAI Mean": None,
        "LAI Median": None,
        "LAI Stddev": None,
        "LAI Mean Adjusted": None,
        "LAI Median Adjusted": None,
        "LAI Stddev Adjusted": None,
        "Cloud or Snow Percentage": cloud_snow_percentage,
    }


def apply_savgol_smoothing(statistics):
    """
    Apply Savitzky-Golay smoothing to LAI statistics.
    Stores unsmoothed values in new columns with " Unsmoothed" suffix.
    """
    if not statistics:
        return statistics

    lai_cols = [
        "LAI Mean",
        "LAI Median",
        "LAI Stddev",
        "LAI Mean Adjusted",
        "LAI Median Adjusted",
        "LAI Stddev Adjusted",
    ]

    for rec in statistics:
        for col in lai_cols:
            rec[col + " Unsmoothed"] = rec[col]

    window_length_default = 9
    polyorder = 2

    for col in lai_cols:
        valid_idxs = []
        valid_values = []

        for idx_stat, row in enumerate(statistics):
            if row[col] is not None:
                valid_idxs.append(idx_stat)
                valid_values.append(row[col])

        if not valid_values:
            continue

        window_length = min(window_length_default, len(valid_values))
        if window_length % 2 == 0:
            window_length -= 1
        if window_length < polyorder + 2:
            window_length = polyorder + 2

        smooth_vals = savgol_filter(valid_values, window_length, polyorder)
        smooth_vals = np.clip(smooth_vals, 0, None)

        for idx_val, stats_row_idx in enumerate(valid_idxs):
            statistics[stats_row_idx][col] = float(smooth_vals[idx_val])

    return statistics


def apply_akima_interpolation(statistics, smoothed):
    """
    Apply Akima interpolation to fill missing LAI statistics.
    Marks interpolated entries with 'interpolated' = 1.
    """
    if not statistics:
        return statistics

    interpolation_cols = ["LAI Mean", "LAI Mean Adjusted", "LAI Median", "LAI Median Adjusted"]

    if smoothed:
        interpolation_cols += [
            "LAI Mean Unsmoothed",
            "LAI Mean Adjusted Unsmoothed",
            "LAI Median Unsmoothed",
            "LAI Median Adjusted Unsmoothed",
        ]

    for col in interpolation_cols:
        real_X = []
        real_Y = []
        nan_X = []

        for i, stat in enumerate(statistics):
            if stat[col] is None:
                nan_X.append(i)
            else:
                real_X.append(i)
                real_Y.append(stat[col])

        if not real_X:
            continue

        cs = Akima1DInterpolator(
            np.array(real_X),
            np.array(real_Y),
            method="akima",
            extrapolate=False,
        )
        for i in nan_X:
            val = cs(i)
            if np.isnan(val):
                continue
            statistics[i][col] = max(float(val), 0.0)
            statistics[i]["interpolated"] = 1

    return statistics


def ensure_raster_alignment(src, geometry, tol_pix=1e-6):
    """
    Ensure that the LAI raster and the cropmask raster are aligned.
    Required so that masking works correctly and no pixel shifts occur.
    """
    if not ProjCRS(src.crs).equals(ProjCRS(geometry["crs"])):
        print(f"ERROR: CRS mismatch: LAI {src.crs} vs cropmask {geometry['crs']}. Reproject the mask first.")
        sys.exit(1)

    if not np.isclose(src.res[0], geometry["res"][0]) or not np.isclose(src.res[1], geometry["res"][1]):
        print(f"ERROR: Resolution mismatch: LAI {src.res} vs cropmask {geometry['res']}")
        sys.exit(1)

    assert_grids_aligned(src, geometry["transform"], tol_pix=1e-6)


def write_statistics_csv(output_stats_fpath, statistics):
    if not statistics:
        return
    with open(output_stats_fpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=statistics[0].keys())
        writer.writeheader()
        writer.writerows(statistics)
    print(f"Exported stats to {output_stats_fpath}")


def write_max_lai_raster(
    output_max_tif_fpath,
    src_meta,
    maxlai_keep_bands,
    lai_max,
    lai_adjusted_max,
):
    """Write the maximum LAI raster to a GeoTIFF file."""
    if not output_max_tif_fpath or src_meta is None:
        return

    band_count = len(maxlai_keep_bands)
    src_meta = src_meta.copy()
    src_meta["count"] = band_count
    src_meta.update({"driver": "GTiff", "nodata": np.nan})

    with rio.open(output_max_tif_fpath, "w", **src_meta) as dst:
        band_idx = 1
        if "estimateLAImax" in maxlai_keep_bands:
            dst.write(lai_max, band_idx)
            dst.set_band_description(band_idx, "estimateLAImax")
            band_idx += 1

        if "adjustedLAImax" in maxlai_keep_bands:
            dst.write(lai_adjusted_max, band_idx)
            dst.set_band_description(band_idx, "adjustedLAImax")

    print(f"Exported max LAI tif to {output_max_tif_fpath}")


def process_single_date(
    lai_dir,
    region,
    resolution,
    lai_file_ext,
    geometry,
    mode,
    adjustment,
    date_iso,
    date_slash,
    cloudcov_threshold,
    src_meta,
):
    """
    Process LAI data for a single date, applying masking, adjustments, and computing statistics and maximum LAI rasters.
    """
    LAI_path = op.join(lai_dir, f"{region}_{str(resolution)}m_{date_iso}_LAI.{lai_file_ext}")

    if not op.exists(LAI_path):
        print(f"{Path(LAI_path).name} [DOES NOT EXIST]")
        return make_empty_stat_row(date_slash), None, None, src_meta

    # Open LAI raster
    src = rio.open(LAI_path)

    if "poly" in mode:
        try:
            masked_src, masked_transform = mask(
                src,
                geometry.geometry,
                crop=True,
                nodata=np.nan,
                indexes=1,
            )
        except ValueError:
            # mask doesn't intersect source
            print(f"{Path(LAI_path).name} [NODATA IN GEOMETRY]")
            return make_empty_stat_row(date_slash), None, None, src_meta

        masked_src, is_padded = pad_to_polygon(src, geometry, masked_src)

        if not is_padded and src_meta is None:
            src_meta = src.meta
            src_meta.update(
                {
                    "height": masked_src.shape[0],
                    "width": masked_src.shape[1],
                    "transform": masked_transform,
                    "count": 2,
                }
            )

        cloud_snow_percentage = -1  # Not yet supported for polygon mode

    elif mode == "raster":
        # Sanity check: ensure LAI raster and cropmask are aligned
        ensure_raster_alignment(src, geometry)

        cropmask_array = geometry["array"]
        cropmask_bounds = geometry["bounds"]
        cm_h, cm_w = cropmask_array.shape

        # Check for intersection between LAI raster and cropmask
        if not raster_and_mask_intersect(src.bounds, cropmask_bounds):
            print(f"{Path(LAI_path).name} [NO INTERSECTION OF CROPMASK AND LAI]")
            return make_empty_stat_row(date_slash), None, None, src_meta

        # Build LAI window array padded for dimensions of the cropmask
        lai_window_array_padded = build_lai_window_for_cropmask(
            src,
            cropmask_bounds,
            (cm_h, cm_w),
        )

        # Compute percentage of pixels that are cloud/snow and actually would be cropland (i.e., inside cropmask)
        cloud_snow_percentage = compute_cloud_snow_percentage(lai_window_array_padded, cropmask_array)

        # Mask LAI window array with binary cropmask
        masked_src = mask_lai_with_binary_cropmask(
            lai_window_array_padded,
            cropmask_array,
        )

        if src_meta is None:
            src_meta = src.meta
            src_meta.update(
                {
                    "height": masked_src.shape[0],
                    "width": masked_src.shape[1],
                    "transform": geometry["transform"],
                    "count": 2,
                    "compress": "lzw",
                    "driver": "GTiff",
                }
            )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    masked_src = clip_negative_lai(masked_src)

    should_skip = False
    if cloudcov_threshold is not None and cloud_snow_percentage > cloudcov_threshold * 100:
        print(f"{Path(LAI_path).name} [INSUFFICIENT DATA IN GEOMETRY]")
        should_skip = True

    if np.all(np.isnan(masked_src)):
        print(f"{Path(LAI_path).name} [NO DATA IN GEOMETRY]")
        should_skip = True

    if should_skip:
        stat = make_empty_stat_row(date_slash, cloud_snow_percentage=cloud_snow_percentage)
        return stat, None, None, src_meta

    LAI_estimate = masked_src
    LAI_adjusted = compute_lai_adjusted(LAI_estimate, adjustment)  # Adjust for croptype

    lai_mean, lai_median, lai_stddev, n_pixels = compute_basic_stats(LAI_estimate)
    lai_adj_mean, lai_adj_median, lai_adj_stddev, _ = compute_basic_stats(LAI_adjusted)

    stat = {
        "Date": date_slash,
        "n_pixels": n_pixels,
        "interpolated": 0,
        "LAI Mean": lai_mean,
        "LAI Median": lai_median,
        "LAI Stddev": lai_stddev,
        "LAI Mean Adjusted": lai_adj_mean,
        "LAI Median Adjusted": lai_adj_median,
        "LAI Stddev Adjusted": lai_adj_stddev,
        "Cloud or Snow Percentage": cloud_snow_percentage,
    }

    print(f"{Path(LAI_path).name} [SUCCESS]")
    return stat, LAI_estimate, LAI_adjusted, src_meta


def process_geometry(
    lai_dir,
    region,
    resolution,
    lai_file_ext,
    geometry,
    mode,
    adjustment,
    dates_iso,
    dates_slash,
    cloudcov_threshold,
):
    """Process all dates for a given geometry, returning statistics and maximum LAI rasters."""
    statistics = []
    lai_max = None
    lai_adjusted_max = None
    src_meta = None

    # Process all dates for this geometry
    for date_iso, date_slash in zip(dates_iso, dates_slash):
        stat, lai_estimate, lai_adjusted, src_meta = process_single_date(
            lai_dir=lai_dir,
            region=region,
            resolution=resolution,
            lai_file_ext=lai_file_ext,
            geometry=geometry,
            mode=mode,
            adjustment=adjustment,
            date_iso=date_iso,
            date_slash=date_slash,
            cloudcov_threshold=cloudcov_threshold,
            src_meta=src_meta,
        )
        statistics.append(stat)

        if lai_estimate is not None:
            lai_max = update_max_lai(lai_max, lai_estimate)
        if lai_adjusted is not None:
            lai_adjusted_max = update_max_lai(lai_adjusted_max, lai_adjusted)

    return statistics, lai_max, lai_adjusted_max, src_meta


def main(
    lai_dir,
    output_stats_fpath,
    output_max_tif_fpath,
    region,
    resolution,
    geometry_path,
    mode,
    adjustment,
    start_date,
    end_date,
    lai_file_ext,
    smoothed,
    cloudcov_threshold,
    maxlai_keep_bands,
):
    start = time.time()

    ensure_output_dirs(output_stats_fpath, output_max_tif_fpath)
    maxlai_keep_bands = validate_maxlai_keep_bands(maxlai_keep_bands)

    geometry_name = Path(geometry_path).stem
    geometries = load_geometries(geometry_path, mode, geometry_name)

    dates_iso, dates_slash = build_date_range(start_date, end_date)

    for idx, geometry in enumerate(geometries):
        print(f"Processing {idx+1} of {len(geometries)} geometries")

        statistics, lai_max, lai_adjusted_max, src_meta = process_geometry(
            lai_dir=lai_dir,
            region=region,
            resolution=resolution,
            lai_file_ext=lai_file_ext,
            geometry=geometry,
            mode=mode,
            adjustment=adjustment,
            dates_iso=dates_iso,
            dates_slash=dates_slash,
            cloudcov_threshold=cloudcov_threshold,
        )

        if smoothed:
            statistics = apply_savgol_smoothing(statistics)

        statistics = apply_akima_interpolation(statistics, smoothed=smoothed)

        # geometry-specific naming hook preserved (if you later want to change output paths)
        new_geometry_name = geometry_name
        if mode == "poly_iter":
            try:
                new_geometry_name += f"_{geometry['FID'][0]}"
            except Exception:
                print("Tried to append polygon FID but failed.")
                print("Using incrementing index instead to avoid overwriting files.")
                new_geometry_name += f"_{str(idx)}"

        write_statistics_csv(output_stats_fpath, statistics)
        write_max_lai_raster(
            output_max_tif_fpath=output_max_tif_fpath,
            src_meta=src_meta,
            maxlai_keep_bands=maxlai_keep_bands,
            lai_max=lai_max,
            lai_adjusted_max=lai_adjusted_max,
        )

    print(f"Finished in {time.time()-start:.2f} seconds")


@click.command()
@click.argument("LAI_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_stats_fpath", type=click.Path(dir_okay=False))
@click.argument("output_max_tif_fpath", type=click.Path(dir_okay=False))
@click.argument("region", type=str)
@click.argument("resolution", type=int)
@click.argument("geometry_path", type=click.Path(exists=True))
@click.option(
    "--mode",
    type=click.Choice(["raster", "poly_agg", "poly_iter"]),
    default="raster",
    help="What kind of geometry to expect and how to apply it. "
    "'raster' expects a pixelwise mask of zeros and ones. "
    "'poly_agg' expects a .shp or .geojson and combines all polygons. "
    "'poly_iter' iterates through each polygon.",
)
@click.option(
    "--adjustment",
    type=click.Choice(["none", "wheat", "maize"]),
    default="none",
    help="Adjustment to apply to the LAI estimate",
)
@click.option(
    "--start_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date for the image collection",
)
@click.option(
    "--end_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date for the image collection",
)
@click.option(
    "--LAI_file_ext",
    type=click.Choice(["tif", "vrt"]),
    help="File extension of the LAI files",
    default="tif",
)
@click.option(
    "--smoothed",
    is_flag=True,
    help="Whether the LAI curve should be smoothed (savgol), before interpolation.",
)
@click.option(
    "--cloudcov_threshold",
    type=float,
    default=None,
    help="Precentage (Range [0,1]) of pixels in ROI that are allowed to be clouds or snow. "
    "If exceeded, the LAI from this date is ignored. Default: None (no threshold).",
)
@click.option(
    "--maxlai_keep_bands",
    multiple=True,
    type=str,
    default=["estimateLAImax", "adjustedLAImax"],
    help="Bands to keep in the max lai tif output. Space seperated. " "(estimateLAImax and/or adjustedLAImax)",
)
def cli(
    lai_dir,
    output_stats_fpath,
    output_max_tif_fpath,
    region,
    resolution,
    geometry_path,
    mode,
    adjustment,
    start_date,
    end_date,
    lai_file_ext,
    smoothed,
    cloudcov_threshold,
    maxlai_keep_bands,
):
    main(
        lai_dir,
        output_stats_fpath,
        output_max_tif_fpath,
        region,
        resolution,
        geometry_path,
        mode,
        adjustment,
        start_date,
        end_date,
        lai_file_ext,
        smoothed,
        cloudcov_threshold,
        maxlai_keep_bands,
    )


if __name__ == "__main__":
    cli()
