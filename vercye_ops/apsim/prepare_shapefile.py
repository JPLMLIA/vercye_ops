from pathlib import Path

import click
import geopandas as gpd

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


def standardize_shapefile(shp_fpath, output_dir):
    """
    Read a shapefile using Geopandas, standardize the column names of the administrative divisions, and save the file to a new location.

    Parameters:
    -----------
    shp_fpath : str
        The path to the .shp file.
    output_dir : str
        The dir where the new shapefile will be saved.

    Returns:
    --------
    None
    """

    logger.info('This script allows you to standardize the column names of the administrative divisions in a shapefile. It will create a new shapefile containing only the entries for your selected administrative division level.')

    shp_fpath = Path(shp_fpath)

    if output_dir is None:
        output_dir = Path(shp_fpath.parent, shp_fpath.stem + '_standardized')

        if output_dir.exists():
            raise Exception(f'There already exists a standardized shapefile under the default directory ({output_dir}). Please use --output_dir to specify a different location.')

    output_fpath = Path(output_dir, shp_fpath.stem + '_standardized' + shp_fpath.suffix)
    
    gdf = gpd.read_file(shp_fpath)

    logger.info('The administrative division column specifies the administrative level of the region, e.g this could typically be a state or a district etc.')

    logger.info('Step 1: Selecting the Column for the Administrative Division Level Name of interest.')

    # Print examples for values in each column
    logger.info('Columns in the shapefile:')
    for col in gdf.columns:
        if col == 'geometry':
            continue
        entries_to_log = min(5, len(gdf[col].unique()))
        logger.info(f'Column name: "{col}". Examples: {gdf[col].unique()[:entries_to_log]}')

    logger.info('Please type the name of the column that contains the name of the administrative level for which you want to do the analysis.')
    
    admin_column_name = input()

    if admin_column_name not in gdf.columns:
        raise ValueError('The column name provided is not valid. Please provide a valid column name from the list above.')
    
    # Copy the column to a new column called 'admin_name'
    gdf['admin_name'] = gdf[admin_column_name]

    logger.info('Step 2: Ensuring that only entries at the same administrative division level are in your shapefile.')

    logger.info('If you are absolutely sure that the shapefile contains only entries at the same administrative division level, you can skip this step. For this, type "skip" and press enter. Otherwise type anything else and press enter.')

    user_input = input()

    if user_input.lower() != 'skip':

        # TODO we could also use presets for common admin levels such as name_1, name_2, name_3 etc.
        logger.info('Please specify all columns that contain the names of different admin levels. Separate the column names with a comma. It is important that you ensure that the ordering is from the highest (e.g country) to the lowest (e.g neighborhood) administrative level.')

        admin_column_names = input().split(',')
        admin_column_names = [col.strip() for col in admin_column_names]

        # Drop all columns that have null at the admin_column_name
        gdf = gdf.dropna(subset=[admin_column_name])

        # Drop all column that have a value other than null in a column after the admin_column_name
        deeper_admin_columns = admin_column_names[admin_column_names.index(admin_column_name) + 1:]
        for col in deeper_admin_columns:
            gdf = gdf[gdf[col].isnull()]


    # Save as modified shapefile
    output_dir.mkdir(exist_ok=True)
    gdf.to_file(output_fpath)
    
    logger.info('Processing Complete. Please manually ensure the resulting shapefile administrative division column names are as expected.')
    logger.info(f'Saving shapefile to {output_fpath}. Please use this file as input for the following steps.')


@click.command()
@click.option('--shp_fpath', type=click.Path(exists=True), help='Path to the .shp file.')
@click.option('--output_dir', type=click.Path(file_okay=False), default=None, help='Optional: Dir where the standardized output file for the Vercye pipeline is saved.')
def cli(shp_fpath, output_dir):
    
    logger.setLevel('INFO')
    standardize_shapefile(shp_fpath, output_dir)
    
    
if __name__ == '__main__':
    cli()