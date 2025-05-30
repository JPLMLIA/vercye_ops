import os
import os.path as op

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.mask import raster_geometry_mask
from shapely.wkt import loads
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


def get_region_name(region_geometry_file):
    return region_geometry_file.split("/")[-1].split(".")[0]


def get_dates_range(start_date, end_date):
    """Generate a range of dates between the start and end dates."""
    return pd.date_range(start_date, end_date)


def all_chirps_data_exists(dates, chirps_dir):
    """
    Validate that the CHIRPS data files exist for the given date range.
    """
    for date in dates:
        chirps_file_path = op.join(chirps_dir, f'chirps-v2.0.{date.strftime("%Y.%m.%d")}.cog')
        chirps_prelim_file_path = op.join(
            chirps_dir, f'chirps-v2.0.{date.strftime("%Y.%m.%d")}_prelim.tif'
        )
        if not op.exists(chirps_file_path) and not op.exists(chirps_prelim_file_path):
            logger.error(
                "CHIRPS data not found for date %s. This data should be present under: %s",
                date,
                chirps_file_path,
            )
            return False

    return True


def is_bbox_within_bounds(inner_bbox, outer_bbox):
    """
    Check if an inner bounding box is completely within an outer bounding box.
    """
    minx, miny, maxx, maxy = inner_bbox
    outer_bbox_lon_min, outer_bbox_lat_min, outer_bbox_lon_max, outer_bbox_lat_max = outer_bbox

    return (
        outer_bbox_lon_min <= minx <= outer_bbox_lon_max
        and outer_bbox_lat_min <= miny <= outer_bbox_lat_max
        and outer_bbox_lon_min <= maxx <= outer_bbox_lon_max
        and outer_bbox_lat_min <= maxy <= outer_bbox_lat_max
    )


def is_coord_within_bounds(lon, lat, bounds):
    """
    Validate that the given coordinates are within the bounds.
    """
    return lon >= bounds[0] and lon <= bounds[2] and lat >= bounds[1] and lat <= bounds[3]


def load_geometry(geometry_path, epsg=4326):
    geometry = gpd.read_file(geometry_path)

    if geometry.crs != rio.crs.CRS.from_epsg(epsg):
        geometry = geometry.to_crs(rio.crs.CRS.from_epsg(epsg))

    return geometry


def read_chirps_file(chirps_dir, date):
    """Generate the file path for a given date and read the CHIRPS data file."""
    chirps_file_path = op.join(chirps_dir, f'chirps-v2.0.{date.strftime("%Y.%m.%d")}.cog')

    if not op.exists(chirps_file_path):
        chirps_file_path = op.join(
            chirps_dir, f'chirps-v2.0.{date.strftime("%Y.%m.%d")}_prelim.tif'
        )

    if not op.exists(chirps_file_path):
        raise FileNotFoundError(f"CHIRPS data not found for date {date}.")

    return rio.open(chirps_file_path)


def process_centroid_data(chirps_dir, date, centroids):
    """Process CHIRPS data using the centroid aggregation method."""
    centroid_values = []

    with read_chirps_file(chirps_dir, date) as src:
        try:
            centroid_values = [val[0] for val in src.sample(centroids)]
        except IndexError as e:
            raise IndexError(f"Coordinates out of bounds for date {date}: {e}")

    return centroid_values


def get_centroid(gdf):
    if "centroid" not in gdf:
        raise KeyError(
            '"centroid" not in the attributes of the shape. Please ensure you have used the convert_shapefile_to_geojson script.'
        )
    # Convert WKT string to a Shapely Point object
    centroid_geom = loads(gdf["centroid"].iloc[0])  # Extract the first (and only) centroid

    # Extract (longitude, latitude) from the Point
    return (centroid_geom.x, centroid_geom.y)


def rasterize_geometry(dataset, gdf):
    # using this function to to stay consistent with the rio.mask.mask function
    mask_array, transform, window = raster_geometry_mask(
        dataset, gdf.geometry, crop=True, invert=True, all_touched=False
    )

    # For small fields there might be not a single pixels whose centroid falls within geometry. In this case use all_touching pixels
    if not np.any(mask_array):
        logger.info("Using all touched for rasterization of small geometry.")
        mask_array, transform, window = raster_geometry_mask(
            dataset, gdf.geometry, crop=True, invert=True, all_touched=True
        )

    return mask_array, window


def rasterize_geometries(dataset, geometries):
    """Rasterizes multiple geometries once and returns their corresponding masks."""
    masks = []
    for geometry in geometries:
        mask_array, mask_window = rasterize_geometry(dataset, geometry)
        masks.append((mask_array, mask_window))

    return masks


def process_mean_data(chirps_dir, date, geometry_masks):
    """Process CHIRPS data using the mean aggregation method."""
    mean_values = []

    with read_chirps_file(chirps_dir, date) as src:
        for mask_array, mask_window in geometry_masks:
            chirps_data = src.read(1, window=mask_window)
            masked_data = np.where(mask_array, chirps_data, np.nan)
            mean_values.append(np.nanmean(masked_data))
    return mean_values


def construct_chirps_precipitation_files(dates, aggregation_method, regions_base_dir, chirps_dir):
    region_names = [
        region_name
        for region_name in os.listdir(regions_base_dir)
        if op.isdir(op.join(regions_base_dir, region_name))
        and os.path.exists(op.join(regions_base_dir, region_name, f"{region_name}.geojson"))
    ]
    region_geometry_files = [
        op.join(regions_base_dir, region_name, f"{region_name}.geojson")
        for region_name in region_names
    ]
    chirps_epsg = 4326  # Could be read dynamically from CHIRPS data in the future
    region_names = [region_name for region_name in region_names]

    # Only keep those regions that have a valid geometry file
    prec_data = []
    chirps_bounds = [-180.0, -50.0, 180.0, 50.0]

    region_gdfs_unfiltered = [
        load_geometry(region_geometry_file, chirps_epsg)
        for region_geometry_file in region_geometry_files
    ]

    if aggregation_method == "centroid":
        centroids_unfiltered = [get_centroid(geometry) for geometry in region_gdfs_unfiltered]
        valid_indices = [
            i
            for i, c in enumerate(centroids_unfiltered)
            if is_coord_within_bounds(c[0], c[1], chirps_bounds)
        ]
        region_centroids = [centroids_unfiltered[i] for i in valid_indices]
        region_names = [region_names[i] for i in valid_indices]

    elif aggregation_method == "mean":
        valid_indices = [
            i
            for i, gdf in enumerate(region_gdfs_unfiltered)
            if is_bbox_within_bounds(gdf.total_bounds, chirps_bounds)
        ]
        region_gdfs = [region_gdfs_unfiltered[i] for i in valid_indices]
        region_names = [region_names[i] for i in valid_indices]

        logger.info("Rasterizing regions of interest.")
        geometry_masks = None
        with read_chirps_file(chirps_dir, dates[0]) as ds:
            geometry_masks = rasterize_geometries(ds, region_gdfs)
    else:
        raise ValueError(f"Invalid aggregation method: {aggregation_method}")

    logger.info(
        f'Computing daily results for each region with aggregation method "{aggregation_method}".'
    )
    with logging_redirect_tqdm():
        for date in tqdm(dates, desc="Aggregating Chirps Data"):
            if aggregation_method == "centroid":
                regional_results = process_centroid_data(chirps_dir, date, region_centroids)
            elif aggregation_method == "mean":
                regional_results = process_mean_data(chirps_dir, date, geometry_masks)
            prec_data.append(regional_results)

    df = pd.DataFrame(prec_data, index=dates, columns=region_names)
    df.index.name = "Date"
    return df


@click.command()
@click.option(
    "--start_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=True,
    help="Start date for data collection in YYYY-MM-DD format.",
)
@click.option(
    "--end_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=True,
    help="End date for data collection in YYYY-MM-DD format.",
)
@click.option("--regions_base_dir", help="Regions directory", type=click.Path(exists=True))
@click.option("--chirps_dir", help="CHIRPS directory", type=click.Path(exists=True))
@click.option(
    "--aggregation_method",
    type=click.Choice(["centroid", "mean"], case_sensitive=False),
    default="centroid",
    show_default=True,
    help="Method to spatially aggregate precipitation data.",
)
@click.option("--output_file", help="Output File", type=click.Path())
def cli_wrapper(
    start_date, end_date, regions_base_dir, chirps_dir, aggregation_method, output_file
):
    # TODO: Currently keeping all results in memory for every region and date.
    # We might want to add an iterative approach, defining a max number of regions to process at once
    # and then saving the results to disk before moving on to the next batch.
    logger.setLevel("INFO")
    required_dates = get_dates_range(start_date, end_date)

    # Download CHIRPS data for the given data range
    logger.info("Validating that CHIRPS data is available for all dates.")
    if not all_chirps_data_exists(required_dates, chirps_dir):
        raise FileNotFoundError(
            f"CHIRPS data incomplete. Please download the data first. You may use the download_chirps_data.py script."
        )

    logger.info("Constructing Chirps precipitation data for all regions")
    prec_df = construct_chirps_precipitation_files(
        required_dates, aggregation_method, regions_base_dir, chirps_dir
    )

    logger.info("Saving Chirps precipitation data to %s", output_file)
    os.makedirs(op.dirname(output_file), exist_ok=True)
    prec_df.to_parquet(output_file, engine="pyarrow", index=True)

    logger.info("Constructing Chirps data completed.")


if __name__ == "__main__":
    cli_wrapper()
