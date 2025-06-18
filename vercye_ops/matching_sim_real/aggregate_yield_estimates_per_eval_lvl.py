import click
import pandas as pd

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()

def aggregate(estimation_data, col_name):
    # Aggregate the estimation data by the specified column.
    # total_yield_production_kg/ton and total_area_ha are simply summed per group.
    # mean_yield_kg_ha is calculated as a weighted mean using total_area_ha as weights.
    # median_yield_kg_ha is calculated as a weighted median using total_area_ha as weights.

    def weighted_mean(x, weights):
        return (x * weights).sum() / weights.sum()
    
    def weighted_median(x, weights):
        # Sort values and weights by the values
        sorted_indices = x.argsort()
        sorted_values = x.iloc[sorted_indices]
        sorted_weights = weights.iloc[sorted_indices]
        
        # Calculate cumulative weights
        cumulative_weights = sorted_weights.cumsum()
        
        # Find the median index (where cumulative weight exceeds half total weight)
        # Selecting the region from the sorted values so that half the total area (weight in ha) is below this value
        median_idx = (cumulative_weights >= cumulative_weights.iloc[-1] / 2).idxmax()
        
        return sorted_values.loc[median_idx]
    
    result = []
    for name, group in estimation_data.groupby(col_name):
        result.append({
            col_name: name,
            'mean_yield_kg_ha': weighted_mean(group['mean_yield_kg_ha'], group['total_area_ha']),
            'median_yield_kg_ha': weighted_median(group['mean_yield_kg_ha'], group['total_area_ha']),
            'total_yield_production_kg': group['total_yield_production_kg'].astype(int).sum(),
            'total_yield_production_ton': round(group['total_yield_production_ton'].sum(), 3),
            'total_area_ha': group['total_area_ha'].sum()
        })
    
    df = pd.DataFrame(result)
    df.rename(columns={col_name: 'region'}, inplace=True)

    return df

@click.command()
@click.option('--estimation_fpath', required=True, type=click.Path(exists=True), help='Filepath to the csv containing the estimations per region.')
@click.option('--aggregation_col', required=True, type=str, help='Column name to aggregate by.')
@click.option('--out_fpath', required=True, type=click.Path(), help='Filepath where the resulting metrics should be saved. Must be a .csv file.')
def cli(estimation_fpath, aggregation_col, out_fpath):
    logger.setLevel('INFO')

    logger.info("Loading data...")
    estimation_data = pd.read_csv(estimation_fpath)

    if aggregation_col not in estimation_data.columns:
        raise ValueError(f"Column '{aggregation_col}' not found in the input estimation csv. Please ensure that the columns are named correctly.")

    logger.info(f"Aggregating data by '{aggregation_col}'...")
    # Group by the specified column and calculate the mean yield
    aggregated_data = aggregate(estimation_data, aggregation_col)
    logger.info(f"Aggregated data by '{aggregation_col}' and saved to {out_fpath}")
    aggregated_data.to_csv(out_fpath, index=False)

if __name__ == '__main__':
    cli()