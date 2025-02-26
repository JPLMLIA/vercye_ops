import click
import glob
import defaultdict
import pandas as pd
import geopandas as gpd
import os.path as op
import rasterio as rio
from rasterio.mask import mask
from rasterio.transform import rowcol
import numpy as np

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()

def get_region_name(region_geometry_file):
    return region_geometry_file.split('/')[-1].split('.')[0]

def get_dates_range(start_date, end_date):
    """Generate a range of dates between the start and end dates."""
    return pd.date_range(start_date, end_date)

def all_chirps_data_exists(dates, chirps_dir):
    """
    Validate that the CHIRPS data files exist for the given date range.
    """
    for date in dates:
        chirps_file_path = op.join(chirps_dir, f'chirps-v2.0.{date.strftime("%Y.%m.%d")}.cog')
        if not op.exists(chirps_file_path):
            logger.error("CHIRPS data not found for date %s. This data should be present under: %s", date, chirps_file_path)
            return False
    
    return True

def is_bbox_within_bounds(inner_bbox, outer_bbox):
    """
    Check if an inner bounding box is completely within an outer bounding box.
    """
    minx, miny, maxx, maxy = inner_bbox
    outer_bbox_lon_min, outer_bbox_lat_min, outer_bbox_lon_max, outer_bbox_lat_max = outer_bbox

    return (
        outer_bbox_lon_min <= minx <= outer_bbox_lon_max and
        outer_bbox_lat_min <= miny <= outer_bbox_lat_max and
        outer_bbox_lon_min <= maxx <= outer_bbox_lon_max and
        outer_bbox_lat_min <= maxy <= outer_bbox_lat_max
    )

def is_coord_within_bounds(lon, lat, bounds):
    """
    Validate that the given coordinates are within the bounds.
    """
    return (lon >= bounds[0] and lon <= bounds[2] and 
            lat >= bounds[1] and lat <= bounds[3])

def load_geometry(geometry_path, epsg=4326):
    geometry = gpd.read_file(geometry_path)

    if geometry.crs != rio.crs.CRS.from_epsg(epsg):
        geometry = geometry.to_crs(rio.crs.CRS.from_epsg(epsg))

def read_chirps_file(chirps_dir, date):
    """Generate the file path for a given date and read the CHIRPS data file."""
    chirps_file_path = op.join(chirps_dir, f'chirps-v2.0.{date.strftime("%Y.%m.%d")}.cog')
    return rio.open(chirps_file_path)

def process_centroid_data(chirps_dir, date, centroids):
    """Process CHIRPS data using the centroid aggregation method."""
    centroid_values = []

    with read_chirps_file(chirps_dir, date) as src:
        for centroid in centroids:
            lon, lat = centroid
            row, col = rowcol(src.transform, lon, lat)
            try:
                centroid_value = list(src.sample([(lon, lat)]))[0][0]
                centroid_values.append(centroid_value)
            except IndexError:
                raise IndexError(f"Coordinates out of bounds for date {date}. Could not read value at row={row} col={col}.")

    return centroid_values

def process_mean_data(chirps_dir, date, geometries):
    """Process CHIRPS data using the mean aggregation method."""
    mean_values = []

    with read_chirps_file(chirps_dir, date) as src:
        for geometry in geometries:
            chirps_data, _ = mask(src, geometry.geometry, crop=True, nodata=np.nan)
            chirps_mean = np.nanmean(chirps_data)
            mean_values.append(chirps_mean)

    return mean_values

def get_centroid(gdf):
    return gdf['centroid']

def construct_chirps_precipitation_files(dates, aggregation_method, regions_base_dir, chirps_dir, date_start, date_end, output_dir):
    region_geometry_files = glob.glob(regions_base_dir + '/*.geojson')
    chirps_epsg = 4326 # Could be read dynamically from the CHIRPS data in the future
    region_names = [get_region_name(region_geometry_file) for region_geometry_file in region_geometry_files]
    prec_data = []
    chirps_bounds = [-180.0, -50.0, 180.0, 50.0]

    region_gdfs_unfiltered = [load_geometry(region_geometry_file, chirps_epsg) for region_geometry_file in region_geometry_files]

    if aggregation_method == 'centroid':
        centroids_unfiltered = [get_centroid(geometry) for geometry in region_gdfs_unfiltered]
        region_centroids = [c for c in centroids_unfiltered if is_coord_within_bounds(c[0], c[1], chirps_bounds)]
        region_geometries = None
    else:
        region_gdfs = [gdf for gdf in region_gdfs_unfiltered if is_bbox_within_bounds(gdf.total_bounds, chirps_bounds)]
        centroids = None
    
    for date in dates:
        if aggregation_method == 'centroid':
            regional_results = process_centroid_data(chirps_dir, date, region_centroids)
        elif aggregation_method == 'mean':
            regional_results = process_mean_data(chirps_dir, date, region_gdfs)
        else:
            raise ValueError(f"Invalid aggregation method: {aggregation_method}")
        prec_data.append(regional_results)

    return pd.DataFrame(prec_data, index=dates, columns=region_names)
        

@click.command()
@click.option('--start_date', type=click.DateTime(formats=["%Y-%m-%d"]), required=True, help="Start date for data collection in YYYY-MM-DD format.")
@click.option('--end_date', type=click.DateTime(formats=["%Y-%m-%d"]), required=True, help="End date for data collection in YYYY-MM-DD format.")
@click.option('--regions_base_dir', help='Regions directory', type=click.Path(exists=True))
@click.option('--chirps_dir', help='CHIRPS directory', type=click.Path(exists=True))
@click.option('--precipitation_aggregation_method', type=click.Choice(['centroid', 'mean'], case_sensitive=False), default='centroid', show_default=True, help="Method to spatially aggregate precipitation data.")
@click.option('--output_file', help='Output File', type=click.Path())
def cli_wrapper(start_date, end_date, regions_base_dir, chirps_dir, output_file):
    # TODO: Currently keeping all results in memory for every region and date. 
    # We might want to add an iterative approach, defining a max number of regions to process at once
    # and then saving the results to disk before moving on to the next batch.

    required_dates = get_dates_range(start_date, end_date)

    # Download CHIRPS data for the given data range
    if not all_chirps_data_exists(required_dates, chirps_dir):
        raise FileNotFoundError(f"CHIRPS data incomplete. Please download the data first. You may use the download_chirps_data.py script.")
    
    logger.info("Constructing Chirps precipitation data for all regions")
    prec_df = construct_chirps_precipitation_files(required_dates, regions_base_dir, chirps_dir)

    op.makedirs(op.dirname(output_file), exist_ok=True)

    prec_df.to_csv(output_file)
    logger.info("Chirps precipitation data saved to %s", output_file)

    # TODO think about how to deal with multiple date spans just take min and max and create for all inbetween?

    
if __name__ == '__main__':
    cli_wrapper()