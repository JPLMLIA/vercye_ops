import click
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def load_csv(fpath):
    return pd.read_csv(fpath)


def compute_metrics(gt, pred):
    # Merging to ensure that the regions are in the same order
    combined = pd.merge(gt, pred, on='region')

    if 'mean_yield_kg_ha' not in combined.columns:
        raise ValueError("Column 'mean_yield_kg_ha' not found in the input estimation csv. Please ensure that the columns are named correctly.")

    # It might occur that the reported mean yield is not available in the input csv.
    if 'reported_mean_yield_kg_ha' not in combined.columns:
        if not 'reported_yield_kg' in combined.columns:
            raise ValueError("Could not compute metrics as neither 'reported_mean_yield_kg_ha' or 'reported_yield_kg' are not available in the input csv.")

        combined['reported_mean_yield_kg_ha'] = combined['total_area_ha'] / combined['reported_yield_kg']

    errors_kg_ha = combined['mean_yield_kg_ha'] - combined['reported_mean_yield_kg_ha']

    mean_err_kg_ha = np.mean(errors_kg_ha)
    median_err_kg_ha = np.median(errors_kg_ha)

    rmse = np.sqrt(mean_squared_error(combined['reported_mean_yield_kg_ha'], combined['mean_yield_kg_ha']))
    rrmse = rmse / np.mean(combined['reported_mean_yield_kg_ha']) # TODO check that this is the correct relative rmse formula used in downstream eval
    r2 = r2_score(combined['reported_mean_yield_kg_ha'], combined['mean_yield_kg_ha'])
    r2_rsq_excel = (np.corrcoef(combined['reported_mean_yield_kg_ha'], combined['mean_yield_kg_ha'])[0, 1]) ** 2

    aggregated_metrics = {
        'mean_err_kg_ha': mean_err_kg_ha,
        'median_err_kg_ha': median_err_kg_ha,
        'rmse_kg_ha': rmse,
        'rrmse': rrmse,
        'r2_scikit': r2,
        'r2_rsq_excel': r2_rsq_excel
    }

    return aggregated_metrics


def write_metrics(metrics, out_fpath):
    pd.DataFrame(metrics, index=[0]).to_csv(out_fpath, index=False)
    return out_fpath


@click.command()
@click.option('--val_fpath', required=True, type=click.Path(exists=True), help='Filepath to the csv containing the validation data per region.')
@click.option('--estimation_fpath', required=True, type=click.Path(exists=True), help='Filepath to the estimations per region csv.')
@click.option('--out_fpath', required=True, type=click.Path(), help='Filepath where the resulting metrics csv should be saved.')
def cli(val_fpath, estimation_fpath, out_fpath):
    """Wrapper for validation cli"""

    regions_ground_truth = load_csv(val_fpath)
    regions_prediction = load_csv(estimation_fpath)

    metrics = compute_metrics(regions_ground_truth, regions_prediction)

    write_metrics(metrics, out_fpath)

if __name__ == '__main__':
    cli()