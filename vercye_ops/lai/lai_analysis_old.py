import csv
import math
import os.path as op
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.mask import mask
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import savgol_filter
from shapely import LineString, box

def pad_to_polygon(src, geometry, masked_src):
    """Pads masked_src to the extent of geometry if it is smaller"""
    if not rio.coords.disjoint_bounds(src.bounds, geometry.total_bounds):
        gx = abs(src.res[0])
        gy = abs(src.res[1])
        left_pad   = int(np.round((src.bounds.left   - geometry.total_bounds[0]) / gx))
        bottom_pad = int(np.round((src.bounds.bottom - geometry.total_bounds[1]) / gy))
        right_pad  = int(np.round((geometry.total_bounds[2] - src.bounds.right ) / gx))
        top_pad    = int(np.round((geometry.total_bounds[3] - src.bounds.top   ) / gy))

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
    """LAI Analysis function

    LAI_dir: Local path to the directory containing regional primary LAI rasters

    region: Name of the primary region from which regions should be cropped

    resolution: Resolution in meters of the LAI. Shoud match with the value used in export.

    This pipeline does the following:
    1. For each date in the range start_date to end_date
    2. Find the primary LAI raster for the given region, resolution and date
    3. Apply the provided mask as specified by the mode
    4. Calculate the appropriate LAI statistics for the CSV
    5. Calculate a maximum LAI raster for the geometry and date range
    """

    if mode != "raster":
        raise NotImplementedError("Only raster mode is currently correctly implemented and tested.")

    start = time.time()

    # Make sure output directory exists
    Path(output_stats_fpath).parent.mkdir(parents=True, exist_ok=True)
    if output_max_tif_fpath:
        Path(output_max_tif_fpath).parent.mkdir(parents=True, exist_ok=True)

    # Geometry
    geometry_name = Path(geometry_path).stem

    maxlai_keep_bands = list(maxlai_keep_bands)
    if len(maxlai_keep_bands) == 0:
        raise ValueError("No bands to keep for the LAI data provided. Specify either estimateLAImax or adjustedLAImax.")

    geometries = []
    if mode == "raster":
        print("MODE: raster")
        print(f"{geometry_name} will be read as a raster and used to 0/1 mask the primary raster.")
        print("NOTE: Preprocessing to project the raster mask to the primary LAI raster is required.")
        print("      Read the README and use 0_reproj_mask.py.")
        # The cropmask is expected to be cropped to the region and have the same resolution as the LAI rasters
        # If it is not cropped to the region, this will make the code very slow when running at national scale
        with rio.open(geometry_path) as ds:
            # Validate that the cropmask raster is binary
            # if not np.array_equal(np.unique(ds.read(1)), [0, 1]):
            #     raise Exception(f"Cropmask {geometry_name} is not binary.")

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
                row_dict = {"geometry": s.gemoetry}
            row_gdf = gpd.GeoDataFrame([row_dict], geometry="geometry", crs=shp.crs)
            geometries.append(row_gdf)

    # "geometries" list is now either:
    # [a single raster mask]
    # [a single shp geometry with multiple polygons]
    # [polygon, polygon, polygon, polygon]

    for idx, geometry in enumerate(geometries):
        print(f"Processing {idx+1} of {len(geometries)} geometries")

        # Iterate through each date
        dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
        dates = [date.strftime("%Y-%m-%d") for date in dates]

        # Keep track of statistics
        statistics = []
        # Keep track of maximum raster
        lai_max = None
        # Keep track of adjusted maximum raster
        lai_adjusted_max = None
        src_meta = None
        for d in dates:
            # The csv requires the date in day/month/year format
            d_slash = datetime.strptime(d, "%Y-%m-%d").strftime("%d/%m/%Y")

            # See if the LAI raster exists
            LAI_path = op.join(lai_dir, f"{region}_{str(resolution)}m_{d}_LAI.{lai_file_ext}")
            if not op.exists(LAI_path):
                print(f"{Path(LAI_path).name} [DOES NOT EXIST]")

                stat = {
                    "Date": d_slash,
                    "n_pixels": 0,
                    "interpolated": 0,
                    "LAI Mean": None,
                    "LAI Median": None,
                    "LAI Stddev": None,
                    "LAI Mean Adjusted": None,
                    "LAI Median Adjusted": None,
                    "LAI Stddev Adjusted": None,
                    "Cloud or Snow Percentage": None,
                }
                statistics.append(stat)
                continue

            # Load the LAI raster
            src = rio.open(LAI_path)

            # Sanity checks to ensure raster alignment
            if src.crs != geometry["crs"]:
                print(f"ERROR: CRS mismatch: LAI {src.crs} vs cropmask {geometry['crs']}. Reproject the mask first.")
                sys.exit(1)

            if (
                not np.isclose(src.res[0], geometry["res"][0])
                or not np.isclose(src.res[1], geometry["res"][1])
            ):
                print(f"ERROR: Resolution mismatch: LAI {src.res} vs cropmask {geometry['res']}")
                sys.exit(1)

            assert_grids_aligned(src, geometry["transform"], tol_pix=1e-6)

            # Cropping via rasters or polygons
            if "poly" in mode:
                # Mask the raster with the shp
                try:
                    masked_src, masked_transform = mask(src, geometry.geometry, crop=True, nodata=np.nan, indexes=1)
                except ValueError:
                    # mask doesn't intersect source
                    print(f"{Path(LAI_path).name} [NODATA IN GEOMETRY]")
                    stat = {
                        "Date": d_slash,
                        "n_pixels": 0,
                        "interpolated": 0,
                        "LAI Mean": None,
                        "LAI Median": None,
                        "LAI Stddev": None,
                        "LAI Mean Adjusted": None,
                        "LAI Median Adjusted": None,
                        "LAI Stddev Adjusted": None,
                        "Cloud or Snow Percentage": None,
                    }
                    statistics.append(stat)
                    continue

                # There is an edge case where if the extent of src is less than that of geometry,
                # then rio.mask.mask will not pad - it will only crop. This function will pad
                # so that masked_src always fills the extent of geometry.
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

                cloud_snow_percentage = -1  # Not yet supported

            elif mode == "raster":
                # Mask the raster with the raster
                cropmask_array = geometry["array"]
                cropmask_array_bool = cropmask_array.copy().astype(bool)
                cropmask_array = geometry["array"].astype(float)
                cropmask_bounds = geometry["bounds"]
                cm_h, cm_w = cropmask_array.shape

                # If is empty or has no height or width, ignore
                bbox_lai = box(*src.bounds)
                bbox_cropmask = box(*cropmask_bounds)
                intersection = bbox_lai.intersection(bbox_cropmask)
                if intersection.is_empty or isinstance(intersection, LineString):
                    print(f"{Path(LAI_path).name} [NO INTERSECTION OF CROPMASK AND LAI]")
                    stat = {
                        "Date": d_slash,
                        "n_pixels": 0,
                        "interpolated": 0,
                        "LAI Mean": None,
                        "LAI Median": None,
                        "LAI Stddev": None,
                        "LAI Mean Adjusted": None,
                        "LAI Median Adjusted": None,
                        "LAI Stddev Adjusted": None,
                        "Cloud or Snow Percentage": None,
                    }
                    statistics.append(stat)
                    continue

                # Build the LAI window aligned to the cropmask extent on the LAI grid
                window_lai_for_mask = rio.windows.from_bounds(*cropmask_bounds, transform=src.transform)

                # Read boundlessly and fill outside with NaN; Will get same shape as cropmask for stacking
                lai_window_array_padded = src.read(
                    1,
                    window=window_lai_for_mask,
                    boundless=True,
                    fill_value=np.nan,
                    out_shape=(cm_h, cm_w),
                    resampling=rio.enums.Resampling.nearest,  # grids should be aligned, so no real resampling occurs
                ).astype(np.float32)

                # Computing the percentage of crop-pixels that are clouds or snow (and thus are nan)
                cloud_snow_pixels = np.sum(np.isnan(lai_window_array_padded) & cropmask_array_bool)
                total_pixels_in_region = np.sum(cropmask_array_bool)
                cloud_snow_percentage = (
                    cloud_snow_pixels / total_pixels_in_region * 100 if total_pixels_in_region > 0 else 0
                )

                # Set the cropmask to NaN where it is 0
                cropmask_array[cropmask_array == 0] = np.nan

                # Sanity check - same shape

                # apply binary mask
                masked_src = lai_window_array_padded * cropmask_array

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

            # clip all negative values to nan, as they are invalid and should be ignored
            masked_src[masked_src < 0] = np.nan

            should_skip = False
            if cloudcov_threshold is not None and cloud_snow_percentage > cloudcov_threshold * 100:
                print(f"{Path(LAI_path).name} [INSUFFICIENT DATA IN GEOMETRY]")
                should_skip = True

            if np.all(np.isnan(masked_src)):
                print(f"{Path(LAI_path).name} [NO DATA IN GEOMETRY]")
                should_skip = True

            # If the raster after masking is all nan, skip
            if should_skip:
                stat = {
                    "Date": d_slash,
                    "n_pixels": 0,
                    "interpolated": 0,
                    "LAI Mean": None,
                    "LAI Median": None,
                    "LAI Stddev": None,
                    "LAI Mean Adjusted": None,
                    "LAI Median Adjusted": None,
                    "LAI Stddev Adjusted": None,
                    "Cloud or Snow Percentage": cloud_snow_percentage,
                }
                statistics.append(stat)
                continue

            LAI_estimate = masked_src

            # Wheat and Maize calibrations
            if adjustment == "wheat":
                LAI_adjusted = LAI_estimate**2 * 0.0482 + LAI_estimate * 0.9161 + 0.0026
            elif adjustment == "maize":
                LAI_adjusted = LAI_estimate**2 * -0.078 + LAI_estimate * 1.4 - 0.18
            else:
                LAI_adjusted = LAI_estimate

            # Calculate statistics for valid raster
            # Catching runtime warnings when numpy complains that a pixel only has NaNs
            # This is expected

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                statistics.append(
                    {
                        "Date": d_slash,
                        "n_pixels": np.sum(~np.isnan(masked_src)),
                        "interpolated": 0,
                        "LAI Mean": np.nanmean(LAI_estimate),
                        "LAI Median": np.nanmedian(LAI_estimate),
                        "LAI Stddev": np.nanstd(LAI_estimate),
                        "LAI Mean Adjusted": np.nanmean(LAI_adjusted),
                        "LAI Median Adjusted": np.nanmedian(LAI_adjusted),
                        "LAI Stddev Adjusted": np.nanstd(LAI_adjusted),
                        "Cloud or Snow Percentage": cloud_snow_percentage,
                    }
                )

                # Update running maximum rasters
                if lai_max is None:
                    lai_max = LAI_estimate
                else:
                    lai_max = np.nanmax([lai_max, LAI_estimate], axis=0)

                if lai_adjusted_max is None:
                    lai_adjusted_max = LAI_adjusted
                else:
                    lai_adjusted_max = np.nanmax([lai_adjusted_max, LAI_adjusted], axis=0)

            print(f"{Path(LAI_path).name} [SUCCESS]")

        if smoothed:
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

            # Apply Savitzky–Golay Smoothing
            window_length_default = 9
            polyorder = 2

            for col in lai_cols:
                valid_idxs = []
                valid_values = []

                for idx, row in enumerate(statistics):
                    if row[col] is not None:
                        valid_idxs.append(idx)
                        valid_values.append(row[col])

                window_length = min(window_length_default, len(valid_values))
                # window length should be uneven and bigger than polyorder to avoid jitter
                if window_length % 2 == 0:
                    window_length -= 1
                if window_length < polyorder + 2:
                    window_length = polyorder + 2

                smooth_vals = savgol_filter(valid_values, window_length, polyorder)
                smooth_vals = np.clip(smooth_vals, 0, None)

                for idx, stats_row_idx in enumerate(valid_idxs):
                    statistics[stats_row_idx][col] = smooth_vals[idx]

        # Akima spline interpolation
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

            # Akima Interpolation
            cs = Akima1DInterpolator(np.array(real_X), np.array(real_Y), method="akima", extrapolate=False)
            for i in nan_X:
                # Clip negative values to zero
                statistics[i][col] = max(cs(i), 0)
                # Mark row as interpolated
                statistics[i]["interpolated"] = 1

        new_geometry_name = geometry_name
        if mode == "poly_iter":
            try:
                new_geometry_name += f"_{geometry['FID'][0]}"
            except Exception:
                print("Tried to append polygon FID but failed.")
                print("Using incrementing index instead to avoid overwriting files.")
                new_geometry_name += f"_{str(idx)}"

        ########################
        # Generating LAI outputs

        # Write statistics to CSV
        with open(output_stats_fpath, "w") as f:
            writer = csv.DictWriter(f, fieldnames=statistics[0].keys())
            writer.writeheader()
            writer.writerows(statistics)
        print(f"Exported stats to {output_stats_fpath}")

        src_meta.update({"driver": "GTiff", "nodata": np.nan})

        # Export running maximum
        if output_max_tif_fpath:
            band_count = len(maxlai_keep_bands)
            src_meta["count"] = band_count
            with rio.open(output_max_tif_fpath, "w", **src_meta) as dst:
                if "estimateLAImax" in maxlai_keep_bands:
                    dst.write(lai_max, 1)
                    dst.set_band_description(1, "estimateLAImax")

                    if "adjustedLAImax" in maxlai_keep_bands:
                        dst.write(lai_adjusted_max, 2)
                        dst.set_band_description(2, "adjustedLAImax")

                elif "adjustedLAImax" in maxlai_keep_bands:
                    dst.write(lai_adjusted_max, 1)
                    dst.set_band_description(1, "adjustedLAImax")
            print(f"Exported max LAI tif to {output_max_tif_fpath}")

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
    help="What kind of geometry to expect and how to apply it. \
                'raster' expects a pixelwise mask of zeros and ones. \
                'poly_agg' expects a .shp or .geojson and combines all polygons. \
                'poly_iter' iterates through each polygon.",
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
    help="Precentage (Range [0,1]) of pixels in ROI that are allowed to be clouds or snow. If exceeded, the LAI from this date is ignored. Default: None (no threshold).",
)
@click.option(
    "--maxlai_keep_bands",
    multiple=True,
    type=str,
    default=["estimateLAImax", "adjustedLAImax"],
    help="Bands to keep in the max lai tif output. Space seperated. (estimateLAImax and/or adjustedLAImax)",
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
