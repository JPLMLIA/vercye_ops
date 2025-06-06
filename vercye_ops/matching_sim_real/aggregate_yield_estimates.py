import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


def aggregate_yields(yield_dir, columns_to_keep, chirps_path=None):
    """
    Aggregate yield estimates from multiple regions within a directory.

    Parameters
    ----------
    yield_dir : str
        Path to the yield directory (e.g., year, timepoint) containing region subdirectories.
    columns_to_keep : str
        Comma-separated list of features from the geojson to keep. If not provided, no additional columns from the geojsons will be kept.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing aggregated yield information for all regions.
    """
    all_yields = []

    regions_using_chirps = []
    if chirps_path is not None:
        parquet_file = pq.ParquetFile(chirps_path)
        regions_using_chirps = parquet_file.schema.names
    
    for region_dir in Path(yield_dir).iterdir():
        if region_dir.is_dir():
            region_name = region_dir.name
            yield_estimate_csv_path = region_dir / f"{region_name}_converted_map_yield_estimate.csv"
            conv_factor_csv_path = region_dir / f"{region_name}_conversion_factor.csv"
            lai_stats_csv_path = region_dir / f"{region_name}_LAI_STATS.csv"
            geojson_path = region_dir / f"{region_name}.geojson"
            
            if yield_estimate_csv_path.exists():
                yield_df = pd.read_csv(yield_estimate_csv_path)
                yield_df['total_yield_production_kg'] = yield_df['total_yield_production_kg'].astype(int)
                yield_df['total_area_ha'] = yield_df['total_area_ha'].round(2)
                yield_df['region'] = region_name
            else:
                logger.warning(f"Converted map yield estimate CSV file not found for region: {region_name}. Dropping from Output.")
                continue

            if conv_factor_csv_path.exists():
                conv_df = pd.read_csv(conv_factor_csv_path)
                conv_df = conv_df[['max_rs_lai', 
                                   'max_rs_lai_date', 
                                   'apsim_max_matched_lai', 
                                   'apsim_max_matched_lai_date', 
                                   'apsim_max_all_lai', 
                                   'apsim_max_all_lai_date',
                                   'apsim_mean_yield_estimate_kg_ha', 
                                   'apsim_matched_std_yield_estimate_kg_ha',
                                   'apsim_all_std_yield_estimate_kg_ha',
                                   'apsim_matched_maxlai_std',
                                   'apsim_all_maxlai_std',
                                   ]]

                conv_df = conv_df.fillna(-1)
                
                conv_df['max_rs_lai'] = conv_df['max_rs_lai'].round(2)
                conv_df['apsim_mean_yield_estimate_kg_ha'] = conv_df['apsim_mean_yield_estimate_kg_ha'].astype(int)
                conv_df['apsim_max_matched_lai'] = conv_df['apsim_max_matched_lai'].round(2)
                conv_df['apsim_max_all_lai'] = conv_df['apsim_max_all_lai'].round(2)
                conv_df['apsim_matched_maxlai_std'] = conv_df['apsim_matched_maxlai_std'].round(2)
                conv_df['apsim_all_maxlai_std'] = conv_df['apsim_all_maxlai_std'].round(2)
                conv_df['apsim_matched_std_yield_estimate_kg_ha'] = conv_df['apsim_matched_std_yield_estimate_kg_ha'].astype(int)
                conv_df['apsim_all_std_yield_estimate_kg_ha'] = conv_df['apsim_all_std_yield_estimate_kg_ha'].astype(int)
                conv_df['region'] = region_name
            else:
                logger.warning(f"Conversion factor CSV file not found for region: {region_name}. Dropping from Output.")
                continue

            # Merge the DataFrames
            combined_df = pd.merge(yield_df, conv_df, on='region', how='outer')

            # Add additional information for easier analysis, as sometimes we fallback from CHIRPS
            precipitation_src = 'CHIRPS' if region_name in regions_using_chirps else 'Met Source'
            
            n_days_with_rs_data_valid = None
            mean_cloud_snow_percentage = None

            # TODO use parameter from CLI for the threshold
            if lai_stats_csv_path.exists():
                rs_df = pd.read_csv(lai_stats_csv_path)
                n_days_with_rs_data_valid =  rs_df[(rs_df['interpolated'] == 0) & (rs_df['Cloud or Snow Percentage'] < 100)].shape[0]
                mean_cloud_snow_percentage = rs_df[(rs_df['interpolated'] == 0) & (rs_df['Cloud or Snow Percentage'] < 100)]['Cloud or Snow Percentage'].mean()
            
            extra_info_df = pd.DataFrame({
                'region': [region_name],
                'precipitation_src': [precipitation_src],
                'n_days_with_rs_data_valid': [n_days_with_rs_data_valid],
                'mean_cloud_snow_percentage': [mean_cloud_snow_percentage]
            })

            combined_df = pd.merge(combined_df, extra_info_df, on='region', how='outer')

            # Add columns from geojson if specified
            if columns_to_keep is not None and not combined_df.empty:
                columns_to_keep_list = [col.strip() for col in columns_to_keep.split(',')]
                if geojson_path.exists():
                    gdf = gpd.read_file(geojson_path)

                    # Check if the columns to keep exist in the geojson
                    missing_columns = [col for col in columns_to_keep_list if col not in gdf.columns]
                    if missing_columns:
                        raise Exception(f"Columns {missing_columns} not found in the geojson file for region: {region_name}. Please ensure the columns are named correctly.")

                    gdf = gdf[columns_to_keep_list]
                    gdf['region'] = region_name
                    combined_df = pd.merge(combined_df, gdf, on='region', how='outer')
                else:
                    logger.info(geojson_path)
                    raise Exception(f"Geojson file not found for region: {region_name}. Please ensure the file exists.")

            all_yields.append(combined_df)
    
    aggregated_yields = pd.concat(all_yields, ignore_index=True)
    
    # Move 'region' to be the first column and sort alphabetically
    columns = ['region'] + [col for col in aggregated_yields.columns if col != 'region']
    aggregated_yields = aggregated_yields[columns].sort_values('region').reset_index(drop=True)
    
    return aggregated_yields

@click.command()
@click.option('--yield_dir', required=True, type=click.Path(exists=True), help='Path to the yield directory (e.g., year, timepoint) containing region subdirectories.')
@click.option('--output_csv', required=True, type=click.Path(), help='Path to save the aggregated yield estimates CSV.')
@click.option('--columns_to_keep', required=False, default=None, type=str, help='Comma-separated list of geojsons-columns to keep in the output CSV. If not provided, no additional columns from the geojsons will be kept.')
@click.option('--chirps_file', required=False, type=click.Path(exists=True), default=None, help='Path to the CHIRPS Parquet file for precipitation data. Used to identify if fallback or chirps was used for precipitation.')
@click.option('--verbose', is_flag=True, help='Enable verbose logging.')
def cli(yield_dir, output_csv, columns_to_keep, chirps_file, verbose):
    """Aggregate yield estimates from multiple regions within a directory."""
    if verbose:
        logger.setLevel('INFO')
    else:
        logger.setLevel('WARNING')

    logger.info(f"Processing directory: {yield_dir}")
    aggregated_yields = aggregate_yields(yield_dir, columns_to_keep, chirps_file)
    
    if aggregated_yields is not None:
        aggregated_yields.to_csv(output_csv, index=False)
        logger.info(f"Aggregated yield estimates saved to: {output_csv}")
    else:
        logger.error("Failed to aggregate yield estimates.")

if __name__ == '__main__':
    cli()

