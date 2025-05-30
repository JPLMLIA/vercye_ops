from pathlib import Path

import click
import geopandas as gpd

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


def log_shapefile_columns(gdf):
    """Log columns in the shapefile with example values."""
    logger.info("Columns in the shapefile:")
    for col in gdf.columns:
        if col == "geometry":
            continue
        entries_to_log = min(5, len(gdf[col].unique()))
        logger.info(
            f'Column name: "{col}". Examples: {gdf[col].unique()[:entries_to_log]}'
        )


def input_admin_column_name(gdf):
    """Prompt user to select the column for administrative division."""
    logger.info(
        "Please type the name of the column that contains the name of the administrative level for analysis."
    )
    admin_column_name = input().strip()

    if admin_column_name not in gdf.columns:
        raise ValueError(
            f'The column name "{admin_column_name}" is not valid. Please choose from the listed columns. Ensure the caseing is correct.'
        )

    return admin_column_name


def input_admin_hierarchy_columns(gdf, admin_column_name):
    """Identify or prompt for columns representing hierarchical admin levels."""

    # Often used admin level column names are used to suggest the hierarchy
    standard_col_names = ["NAME_0", "NAME_1", "NAME_2", "NAME_3", "NAME_4"]
    present_columns = [col for col in gdf.columns if col in standard_col_names]

    if present_columns:
        logger.info(
            f"Suggested hierarchy of admin level columns: {present_columns}. Are these correct? (y/n)"
        )
        if input().strip().lower() == "y":
            return present_columns

    logger.info(
        "Please specify all columns for admin levels, ordered from largest to smallest level. Separate names with commas. Ensure the order of columns is correct."
    )
    logger.info(
        'For example, if the columns are "Country", "State", "District", type: Country, State, District. As you can see, the order is from largest to smallest area.'
    )
    admin_column_names = input().split(",")
    return [col.strip() for col in admin_column_names]


def filter_by_admin_level(gdf, admin_column_name, admin_column_names):
    """Filter the GeoDataFrame to ensure only entries for the specified admin level are present."""

    # Drop all columns that have null at the admin_column_name, as these are from larger admin levels
    gdf = gdf.dropna(subset=[admin_column_name])

    # check if there are duplicate values in the admin_column_name

    # Drop all column that have a value other than null in a admin level column deeper
    # than the admin_column_name, as these are from smaller admin levels
    deeper_admin_columns = admin_column_names[
        admin_column_names.index(admin_column_name) + 1 :
    ]
    for col in deeper_admin_columns:
        gdf = gdf[gdf[col].isnull()]

    if gdf[admin_column_name].duplicated().any():
        raise Exception(
            f"Multiple entries with the same key found in the admin column {admin_column_name}. Cleaning of data unsuccesfull."
        )

    return gdf


def standardize_shapefile(shp_fpath, output_dir):
    """
    Standardize shapefile by filtering for admin level, ensuring only a single level is used.

    Parameters:
        shp_fpath (str): Path to the input shapefile.
        output_dir (str): Directory to save the standardized shapefile.

    Returns:
        None
    """
    logger.info(
        "This script allows you to create a new shapefile containing only the entries for your selected administrative division level."
    )
    logger.info(
        "The administrative division level specifies the administrative level of the region, e.g this could typically be a state or a district etc."
    )

    # Define paths
    shp_fpath = Path(shp_fpath)
    output_dir = Path(output_dir or shp_fpath.parent / f"{shp_fpath.stem}_standardized")
    output_fpath = output_dir / f"{shp_fpath.stem}_standardized{shp_fpath.suffix}"

    if output_fpath.exists():
        raise FileExistsError(
            f"Standardized shapefile already exists at {output_fpath}. Provide a different output directory."
        )

    gdf = gpd.read_file(shp_fpath)

    # Print the available columns in the shapefile and prompt user to select the admin column
    log_shapefile_columns(gdf)
    admin_column_name = input_admin_column_name(gdf)

    logger.info("Now, we need to identify all admin level columns in the shapefile.")
    admin_column_names = input_admin_hierarchy_columns(gdf, admin_column_name)
    gdf = filter_by_admin_level(gdf, admin_column_name, admin_column_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_fpath)

    logger.info(f"Processing complete. Shapefile saved to {output_fpath}.")


@click.command()
@click.option(
    "--shp_fpath",
    type=click.Path(exists=True),
    required=True,
    help="Path to the .shp file.",
)
@click.option(
    "--output_dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Directory for saving the standardized shapefile.",
)
def cli(shp_fpath, output_dir):
    logger.setLevel("INFO")
    try:
        standardize_shapefile(shp_fpath, output_dir)
    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    cli()
