import sys
import os.path as op
from pathlib import Path
from datetime import datetime, timedelta
import time
import click
import csv
import warnings

import numpy as np
from scipy.interpolate import Akima1DInterpolator
import rasterio as rio
from rasterio.mask import mask

import geopandas as gpd

def pad_to_polygon(src, geometry, masked_src):
    """Pads masked_src to the extent of geometry if it is smaller"""

    if not rio.coords.disjoint_bounds(src.bounds, geometry.total_bounds):
        left_pad = int(np.round((src.bounds.left - geometry.total_bounds[0]) / src.res[0]))
        left_pad = int(max(left_pad, 0))
        bottom_pad = int(np.round((src.bounds.bottom - geometry.total_bounds[1]) / src.res[1]))
        bottom_pad = int(max(bottom_pad, 0))
        right_pad = int(np.round((geometry.total_bounds[2] - src.bounds.right) / src.res[0]))
        right_pad = int(max(right_pad, 0))
        top_pad = int(np.round((geometry.total_bounds[3] - src.bounds.top) / src.res[1]))
        top_pad = int(max(top_pad, 0))

        if left_pad + bottom_pad + right_pad + top_pad == 0:
            return masked_src, False
        else:
            print(f"Padding {masked_src.shape} by {[top_pad, bottom_pad, left_pad, right_pad]}")
            padded_src  = np.pad(masked_src, pad_width=((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant', constant_values=np.nan)
            return padded_src, True

    else:
        print("ERROR: The polygon and the source raster do not overlap.")
        sys.exit(1)

def pad_to_raster(src, src_array, cropmask, cropmask_bounds):
    
    if not rio.coords.disjoint_bounds(src.bounds, cropmask_bounds):
        left_pad = int(np.round((src.bounds.left - cropmask_bounds[0]) / src.res[0]))
        left_pad = int(max(left_pad, 0))
        bottom_pad = int(np.round((src.bounds.bottom - cropmask_bounds[1]) / src.res[1]))
        bottom_pad = int(max(bottom_pad, 0))
        right_pad = int(np.round((cropmask_bounds[2] - src.bounds.right) / src.res[0]))
        right_pad = int(max(right_pad, 0))
        top_pad = int(np.round((cropmask_bounds[3] - src.bounds.top) / src.res[1]))
        top_pad = int(max(top_pad, 0))

        if left_pad + bottom_pad + right_pad + top_pad == 0:
            return src_array, False
        else:
            print(f"Padding {src_array.shape} by {[top_pad, bottom_pad, left_pad, right_pad]} to {cropmask.shape}")
            padded_src = np.pad(src_array, pad_width=((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant', constant_values=np.nan)
            return padded_src, True
    else:
        print("ERROR: The cropmask and the source raster do not overlap.")
        sys.exit(1)

@click.command()
@click.argument('LAI_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('output_stats_fpath', type=click.Path(dir_okay=False))
@click.argument('output_max_tif_fpath', type=click.Path(dir_okay=False))
@click.argument('region', type=str)
@click.argument('resolution', type=int)
@click.argument('geometry_path', type=click.Path(exists=True))
@click.option('--mode', type=click.Choice(['raster', 'poly_agg', 'poly_iter']), default='raster', 
              help="What kind of geometry to expect and how to apply it. \
                'raster' expects a pixelwise mask of zeros and ones. \
                'poly_agg' expects a .shp or .geojson and combines all polygons. \
                'poly_iter' iterates through each polygon.")
@click.option('--adjustment', type=click.Choice(["none", "wheat", "maize"]), default="none", help='Adjustment to apply to the LAI estimate')
@click.option('--start_date', type=click.DateTime(formats=["%Y-%m-%d"]), help='Start date for the image collection')
@click.option('--end_date', type=click.DateTime(formats=["%Y-%m-%d"]), help='End date for the image collection')
def main(lai_dir, output_stats_fpath, output_max_tif_fpath, region, resolution, geometry_path, mode, adjustment, start_date, end_date):
    """ LAI Analysis function

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

    start = time.time()

    # Make sure output directory exists
    Path(output_stats_fpath).parent.mkdir(parents=True, exist_ok=True)
    Path(output_max_tif_fpath).parent.mkdir(parents=True, exist_ok=True)

    # Geometry
    geometry_name = Path(geometry_path).stem

    geometries = []
    if mode == "raster":
        print("MODE: raster")
        print(f"{geometry_name} will be read as a raster and used to 0/1 mask the primary raster.")
        print("NOTE: Preprocessing to project the raster mask to the primary LAI raster is required.")
        print("      Read the README and use 0_reproj_mask.py.")
        with rio.open(geometry_path) as ds:
            # mask_array = ds.read(1)
            # mask_res = ds.res[0]
            # mask_bounds = [ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top]
            geometries.append({
                'array': ds.read(1), 
                'res': ds.res[0],
                'bounds': [ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top],
                'transform': ds.transform})
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
                row_dict = {'geometry': s.geometry, 'FID': s.FID}
            except:
                print("[WARN] Attribute 'FID' not found, ignoring")
                row_dict = {'geometry': s.gemoetry}
            row_gdf = gpd.GeoDataFrame([row_dict], geometry='geometry', crs=shp.crs)
            geometries.append(row_gdf)
            
    # "geometries" list is now either:
    # [a single raster mask]
    # [a single shp geometry with multiple polygons]
    # [polygon, polygon, polygon, polygon]

    for idx, geometry in enumerate(geometries):
        print(f"Processing {idx+1} of {len(geometries)} geometries")

        # Iterate through each date
        dates = [start_date + timedelta(days=i) for i in range((end_date-start_date).days+1)]
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
            LAI_path = op.join(lai_dir, f"{region}_{str(resolution)}m_{d}_LAI.tif")
            if not op.exists(LAI_path):
                print(f"{Path(LAI_path).name} [DOES NOT EXIST]")
                
                stat = {
                    "Date": d_slash,
                    "n_pixels": 0,
                    "interpolated": 0,
                    "LAI Mean": None,
                    "LAI Stddev": None,
                    "LAI Mean Adjusted": None,
                    "LAI Stddev Adjusted": None,
                    "Cloud or Snow Percentage": None
                }
                statistics.append(stat)
                continue
                
                
            # Load the LAI raster
            src = rio.open(LAI_path)

            # Cropping via rasters or polygons
            if 'poly' in mode:
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
                        "LAI Stddev": None,
                        "LAI Mean Adjusted": None,
                        "LAI Stddev Adjusted": None,
                        "Cloud or Snow Percentage": None
                    }
                    statistics.append(stat)
                    continue


                # There is an edge case where if the extent of src is less than that of geometry,
                # then rio.mask.mask will not pad - it will only crop. This function will pad
                # so that masked_src always fills the extent of geometry.
                masked_src, is_padded = pad_to_polygon(src, geometry, masked_src)

                if not is_padded and src_meta is None:
                    src_meta = src.meta
                    src_meta.update({
                        "height": masked_src.shape[0],
                        "width": masked_src.shape[1],
                        "transform": masked_transform,
                        "count": 2
                    })

                cloud_snow_percentage = -1 # Not yet supported

            elif mode == "raster":
                # Mask the raster with the raster
                masked_src = src.read(1)

                cropmask_array = geometry['array']
                cropmask_res = geometry['res']
                cropmask_bounds = geometry['bounds']
                if src.res[0] != cropmask_res:
                    print(f"ERROR: cropmask resolution {cropmask_res} != LAI resolution {src.res[0]}")
                    print(f"Preprocessing to project the cropmask to the primary region LAI is required.")
                    print(f"See the README and use 0_reproj_mask.py")
                    sys.exit(1)
                
                # 0_reproj_mask.py should've ensured that the cropmask is the same size as a complete LAI raster
                # Sometimes, however, an LAI raster is partial because of coverage.
                # Pad the LAI raster to match the extent of the cropmask 
                masked_src, is_padded = pad_to_raster(src, masked_src, cropmask_array, cropmask_bounds)

                # replace zeros with NaN's
                cropmask_array_bool = cropmask_array.copy().astype(bool)
                cropmask_array = cropmask_array.astype(float)

                # Computing the percentage of pixels that are clouds or snow (and thus are nan)
                cloud_snow_pixels = np.sum(np.isnan(masked_src) & cropmask_array_bool)
                total_pixels_in_region = np.sum(cropmask_array_bool)
                cloud_snow_percentage = cloud_snow_pixels / total_pixels_in_region * 100 if total_pixels_in_region > 0 else 0

                cropmask_array[cropmask_array==0] = np.nan
                
                # apply mask
                masked_src *= cropmask_array

                if not is_padded and src_meta is None:
                    src_meta = src.meta
                    src_meta.update({
                        "height": masked_src.shape[0],
                        "width": masked_src.shape[1],
                        "transform": geometry['transform'],
                        "count": 2,
                        "compress": "lzw"})

            # If the raster after masking is all nan, skip
            if np.all(np.isnan(masked_src)):
                print(f"{Path(LAI_path).name} [NODATA IN GEOMETRY]")
                stat = {
                    "Date": d_slash,
                    "n_pixels": 0,
                    "interpolated": 0,
                    "LAI Mean": None,
                    "LAI Stddev": None,
                    "LAI Mean Adjusted": None,
                    "LAI Stddev Adjusted": None,
                    "Cloud or Snow Percentage": cloud_snow_percentage
                }
                statistics.append(stat)
                continue

            # Rasters get read as (1, height, width)
            #LAI_estimate = masked_src[0]
            LAI_estimate = masked_src

            # clip all negative values to zero
            LAI_estimate[LAI_estimate < 0] = 0

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
                statistics.append({
                    "Date": d_slash,
                    "n_pixels": np.sum(~np.isnan(masked_src)),
                    "interpolated": 0,
                    "LAI Mean": np.nanmean(LAI_estimate),
                    "LAI Stddev": np.nanstd(LAI_estimate),
                    "LAI Mean Adjusted": np.nanmean(LAI_adjusted),
                    "LAI Stddev Adjusted": np.nanstd(LAI_adjusted),
                    "Cloud or Snow Percentage": cloud_snow_percentage,
                })

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

        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        # Akima spline interpolation
        for col in ["LAI Mean", "LAI Mean Adjusted"]:
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
            cs = Akima1DInterpolator(np.array(real_X), np.array(real_Y), method='akima', extrapolate=False)
            for i in nan_X:
                # Clip negative values to zero
                statistics[i][col] = max(cs(i),0)
                # Mark row as interpolated
                statistics[i]["interpolated"] = 1

        new_geometry_name = geometry_name
        if mode == "poly_iter":
            try:
                new_geometry_name += f"_{geometry['FID'][0]}"
            except:
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
        
        # Export running maximum
        # Set 0 to nan
        with rio.open(output_max_tif_fpath, 'w', **src_meta) as dst:
            dst.write(lai_max, 1)
            dst.write(lai_adjusted_max, 2)
            dst.set_band_description(1, "estimateLAImax")
            dst.set_band_description(2, "adjustedLAImax")
        print(f"Exported max LAI tif to {output_max_tif_fpath}")

    print(f"Finished in {time.time()-start:.2f} seconds")

if __name__ == "__main__":
    main()
