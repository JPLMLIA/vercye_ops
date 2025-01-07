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

    shp_fpath = Path(shp_fpath)

    if output_dir is None:
        output_dir = Path(shp_fpath.parent, shp_fpath.stem + '_standardized')

        if output_dir.exists():
            raise Exception(f'There already exists a standardized shapefile under the default directory ({output_dir}). Please use --output_dir to specify a different location.')

    output_fpath = Path(output_dir, shp_fpath.stem + '_standardized' + shp_fpath.suffix)
    
    # Read shapefile
    gdf = gpd.read_file(shp_fpath)

    logger.info('The administrative division column specifies the administrative level of the region, e.g this could typically be a state or a district etc.')

    # Validate that only a single administrative division name column is present
    if 'NAME_1' in gdf.columns and not 'NAME_2' in gdf.columns and not 'NAME_3' in gdf.columns:
        logger.info('The shapefile is already standardized as contains a single administrative division name column "Name_1".')
        
        logger.info('Would you like to set a different column as the administrative division level? (y/n)')
        user_input = input()

        if user_input.lower() == 'n':
            return
    
    if sum(col in gdf.columns for col in ['NAME_0', 'NAME_1', 'NAME_2', 'NAME_3']) > 1:
        logger.info('The shapefile contains multiple known administrative division name columns (e.g "Name_0", "Name_1", "Name_2", "Name_3"). Please select the administrative level for which you want to do the analysis')

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
        
    # Drop Name_2, Name_3 columns if they exist
    gdf.drop(columns=['NAME_2', 'NAME_3'], errors='ignore', inplace=True)

    # Rename the column to NAME_1
    gdf.rename(columns={admin_column_name: 'NAME_1'}, inplace=True)

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