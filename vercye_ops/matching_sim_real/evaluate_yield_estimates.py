import click
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def load_csv(fpath):
    return pd.read_csv(fpath)


def compute_metrics(gt, pred):
    errors_kg_ha = pred['mean_yield_kg_ha'] - gt['reported_mean_yield_kg_ha']

    mean_err_kg_ha = np.mean(errors_kg_ha)
    median_err_kg_ha = np.median(errors_kg_ha)

    rmse = np.sqrt(mean_squared_error(gt['reported_mean_yield_kg_ha'], pred['mean_yield_kg_ha']))
    rrmse = rmse / np.mean(gt['reported_mean_yield_kg_ha']) # TODO check that this is the correct rrmse formulat used in downstream eval
    r2 = r2_score(gt['reported_mean_yield_kg_ha'], pred['mean_yield_kg_ha'])

    aggregated_metrics = {
        'mean_err_kg_ha': mean_err_kg_ha,
        'median_err_kg_ha': median_err_kg_ha,
        'rmse_kg_ha': rmse,
        'rrmse': rrmse,
        'r2': r2
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