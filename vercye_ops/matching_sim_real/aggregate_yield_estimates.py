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
            csv_path = region_dir / f"{region_name}_converted_map_yield_estimate.csv"
            
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df['region'] = region_name
                all_yields.append(df)
            else:
                logger.warning(f"CSV file not found for region: {region_name}")
    
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

