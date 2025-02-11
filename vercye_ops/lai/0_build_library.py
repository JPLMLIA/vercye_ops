from pathlib import Path

import click
import geopandas as gpd

import logging
import re

logging.basicConfig(level=logging.INFO)

def convert_shapefile_to_geojson(shp_fpath, admin_level, output_head_dir, verbose):
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
    verbose : bool
        Enables verbose output.

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
        gdf = gdf.to_crs(epsg=4326)
        #raise ValueError("The shapefile coordinate system is not WGS 84.")
    
    if verbose:
        logging.info('Processing %i %s regions.', len(gdf), admin_level)

    # Iterate over the GeoDataFrame rows, saving each to geojson
    for _, row in gdf.iterrows():

        # Generate the output fpath
        if admin_level == 'oblast':
            region_name = row["NAME_1"]
        else:
            region_name = row["NAME_2"]

        # Take out any apostrophes and other special chars as these cause headaches down the line with scripting the filename processing
        region_name = region_name.replace("'", "").replace('"', "")
        region_name = re.sub(r"[^\w.-]", "_", region_name)

        output_fpath = output_head_dir / Path(f'{region_name}.geojson')

        # Create a new GeoDataFrame for the current row
        single_row_gdf = gpd.GeoDataFrame([row], crs=gdf.crs)

        # Write the single row GeoDataFrame to a GeoJSON file
        single_row_gdf.to_file(output_fpath, driver='GeoJSON')

        if verbose:
            logging.info('GeoJSON file written to %s', output_fpath)

    if verbose:
        logging.info('Processing Complete')


@click.command()
@click.argument('shp_fpath', required=True, type=click.Path(exists=True))
@click.option('--admin_level', type=click.Choice(['oblast', 'raion']), default='oblast', help='Level of administration to process. `oblast` corresponds to Level 1, `raion` corresponds to Level 2')
@click.option('--output_head_dir', type=click.Path(file_okay=False), default='library', help='Head directory where the region output dirs will be created.')
@click.option('--verbose', is_flag=True, help="Print verbose output.")
def cli(shp_fpath, admin_level, output_head_dir, verbose):
    """Wrapper around geojson generation func"""
    convert_shapefile_to_geojson(shp_fpath, admin_level, output_head_dir, verbose)
    
if __name__ == '__main__':
    cli()