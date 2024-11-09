from pathlib import Path

import click
import geopandas as gpd
from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


# TODO: Extend this in the future to allow for more points to be created
def generate_met_points(gdf_row):
    """Helper to create point(s) from a polygon"""
    centroid = gdf_row.geometry.centroid
    return centroid


def convert_shapefile_to_geojson(shp_fpath, admin_level, projection_epsg, output_head_dir):
    """
    Read a shapefile using Geopandas, add centroid information to each polygon, and export each as a geojson file.

    Parameters:
    -----------
    shp_fpath : str
        The path to the .shp file.
    admin_level : str
        `oblast` or `raion` specifying the administrative level to process in the shapefile.
    output_dir : str
        The directory where the GeoJSON files will be saved.

    Returns:
    --------
    None
    """
    # Convert shp_filepath and output_dir to Path objects
    output_head_dir = Path(output_head_dir)

    # Ensure output directory exists
    output_head_dir.mkdir(parents=True, exist_ok=True)
    
    # Read shapefile
    gdf = gpd.read_file(shp_fpath)

    
    if gdf.empty:
        raise ValueError("The shapefile does not contain any polygons.")
    if gdf.crs.to_epsg() != 4326:
        raise ValueError("The shapefile coordinate system is not WGS 84.")
    
    # Add a new column for the centroid of each polygon
    gdf_proj = gdf.to_crs(epsg=projection_epsg)  # Calculate this in flattened projection instead of geodesic space
    raw_centroids = gdf_proj.apply(generate_met_points, axis=1).set_crs(epsg=projection_epsg)
    
    # TODO: produce plot/report for transparency

    # Convert back to geodesic, and then to WKT. WKT is needed since we can't save multiple geometries to geojson
    gdf['centroid'] = raw_centroids.to_crs(epsg=4326).to_wkt()
    
    logger.info('Processing %i %s regions.', len(gdf), admin_level)

    # Iterate over the GeoDataFrame rows, saving each to geojson
    for _, row in gdf.iterrows():

        # Generate the output fpath
        if admin_level == 'oblast':
            region_name = row["NAME_1"]
        else:
            region_name = row["NAME_2"]

        # Take out any apostrophes as these cause headaches down the line with scripting the filename processing
        region_name = region_name.replace("'", "")

        output_dir = output_head_dir / Path(region_name)
        output_dir.mkdir(exist_ok=True)
        output_fpath = output_dir / Path(f'{region_name}.geojson')

        # Create a new GeoDataFrame for the current row
        single_row_gdf = gpd.GeoDataFrame([row], crs=gdf.crs)

        # Write the single row GeoDataFrame to a GeoJSON file
        single_row_gdf.to_file(output_fpath, driver='GeoJSON')

        logger.info('GeoJSON file written to %s', output_fpath)

    logger.info('Processing Complete')


@click.command()
@click.option('--shp_fpath', type=click.Path(exists=True), help='Path to the .shp file.')
@click.option('--admin_level', type=click.Choice(['oblast', 'raion']), default='oblast', help='Level of administration to process. `oblast` corresponds to Level 1, `raion` corresponds to Level 2')
@click.option('--projection_epsg', type=int, default=6381, help='EPSG code to define projection. Default is for Ukraine.')
@click.option('--output_head_dir', type=click.Path(file_okay=False), help='Head directory where the region output dirs will be created.')
@click.option('--verbose', is_flag=True, help="Print verbose output.")
def cli(shp_fpath, admin_level, projection_epsg, output_head_dir, verbose):
    """Wrapper around geojson generation func"""
    
    if verbose:
        logger.setLevel('INFO')
    convert_shapefile_to_geojson(shp_fpath, admin_level, projection_epsg, output_head_dir)
    
    
if __name__ == '__main__':
    cli()