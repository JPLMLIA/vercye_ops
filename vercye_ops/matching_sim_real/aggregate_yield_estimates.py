import os
import click
import pandas as pd
from pathlib import Path

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


def aggregate_yields(yield_dir):
    """
    Aggregate yield estimates from multiple regions within a directory.

    Parameters
    ----------
    yield_dir : str
        Path to the yield directory (e.g., year, timepoint) containing region subdirectories.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing aggregated yield information for all regions.
    """
    all_yields = []
    
    for region_dir in Path(yield_dir).iterdir():
        if region_dir.is_dir():
            region_name = region_dir.name
            yield_estimate_csv_path = region_dir / f"{region_name}_converted_map_yield_estimate.csv"
            conv_factor_csv_path = region_dir / f"{region_name}_conversion_factor.csv"
            
            if yield_estimate_csv_path.exists():
                yield_df = pd.read_csv(yield_estimate_csv_path)
                yield_df['region'] = region_name
            else:
                yield_df = pd.DataFrame()  # Empty DataFrame as a fallback
                logger.warning(f"Converted map yield estimate CSV file not found for region: {region_name}")

            if conv_factor_csv_path.exists():
                conv_df = pd.read_csv(conv_factor_csv_path)
                conv_df = conv_df[['max_rs_lai', 
                                   'max_rs_lai_date', 
                                   'apsim_mean_yield_estimate', 
                                   'max_matched_sim_lai', 
                                   'max_matched_sim_lai_date', 
                                   'max_total_sim_lai', 
                                   'max_total_sim_lai_date',
                                   'apsim_matched_std_yield_estimate',
                                   'apsim_total_std_yield_estimate',
                                   'apsim_matched_maxlai_std',
                                   'apsim_total_maxlai_std',
                                   ]]
                conv_df['region'] = region_name
            else:
                conv_df = pd.DataFrame()  # Empty DataFrame as a fallback
                logger.warning(f"Conversion factor CSV file not found for region: {region_name}")

            # Merge the DataFrames
            if not yield_df.empty or not conv_df.empty:
                combined_df = pd.merge(yield_df, conv_df, on='region', how='outer')
                all_yields.append(combined_df)
                
                if not all_yields:
                    logger.error("No yield estimate CSV files found in any region.")
                    return None
    
    aggregated_yields = pd.concat(all_yields, ignore_index=True)
    
    # Move 'region' to be the first column and sort alphabetically
    columns = ['region'] + [col for col in aggregated_yields.columns if col != 'region']
    aggregated_yields = aggregated_yields[columns].sort_values('region').reset_index(drop=True)
    
    return aggregated_yields

@click.command()
@click.option('--yield_dir', required=True, type=click.Path(exists=True), help='Path to the yield directory (e.g., year, timepoint) containing region subdirectories.')
@click.option('--output_csv', required=True, type=click.Path(), help='Path to save the aggregated yield estimates CSV.')
@click.option('--verbose', is_flag=True, help='Enable verbose logging.')
def cli(yield_dir, output_csv, verbose):
    """Aggregate yield estimates from multiple regions within a directory."""
    if verbose:
        logger.setLevel('INFO')
    else:
        logger.setLevel('WARNING')

    logger.info(f"Processing directory: {yield_dir}")
    aggregated_yields = aggregate_yields(yield_dir)
    
    if aggregated_yields is not None:
        aggregated_yields.to_csv(output_csv, index=False)
        logger.info(f"Aggregated yield estimates saved to: {output_csv}")
    else:
        logger.error("Failed to aggregate yield estimates.")

if __name__ == '__main__':
    cli()

