import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import scipy

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()

def load_csv(fpath):
    return pd.read_csv(fpath)

def compute_metrics(preds, obs):
    if len(preds) != len(obs):
        raise ValueError("Length of the predictions and observations do not match.")
    errors_kg_ha = obs - preds
    mean_err_kg_ha = np.mean(errors_kg_ha)
    median_err_kg_ha = np.median(errors_kg_ha)
    mean_abs_err_kg_ha = np.mean(np.abs(errors_kg_ha))
    meadian_abs_err_kg_ha = np.median(np.abs(errors_kg_ha))

    rmse = np.sqrt(mean_squared_error(obs, preds))
    rrmse = rmse / np.mean(obs) * 100 # get percentage
    r2 = r2_score(obs, preds)
    r2_rsq_excel = (np.corrcoef(obs, preds)[0, 1]) ** 2

    theta = np.polyfit(preds, obs, 1)
    y_line = theta[1] + theta[0] * obs
    r2_scikit_bestfit = r2_score(preds, y_line)

    aggregated_metrics = {
        'n_regions': len(obs),
        'mean_err_kg_ha': mean_err_kg_ha,
        'median_err_kg_ha': median_err_kg_ha,
        'mean_abs_err_kg_ha': mean_abs_err_kg_ha,
        'median_abs_err_kg_ha': meadian_abs_err_kg_ha,
        'rmse_kg_ha': rmse,
        'rrmse': rrmse,
        'r2_scikit': r2,
        'r2_rsq_excel': r2_rsq_excel,
        'r2_scikit_bestfit': r2_scikit_bestfit
    }

    return aggregated_metrics


def write_metrics(metrics, out_fpath):
    pd.DataFrame(metrics, index=[0]).to_csv(out_fpath, index=False)
    return out_fpath


def create_scatter_plot(preds, obs, obs_years=None):
    preds = np.asarray(preds)
    obs   = np.asarray(obs)

    if obs_years is not None:
        obs_years = np.asarray(obs_years)
        if len(obs_years) != len(obs):
            raise ValueError("Length of obs_years must match length of obs")

    # determine plot range
    abs_max = max(preds.max(), obs.max())
    xlims = [-5, abs_max + 100]
    ylims = [-5, abs_max + 100]

    # compute regression line
    slope, intercept = np.polyfit(preds, obs, 1)
    x_line = np.linspace(0, abs_max, 100)
    y_line = intercept + slope * x_line

    # choose coloring
    if obs_years is None:
        # try KDE‐based density
        try:
            values = np.vstack([preds, obs])
            density = stats.gaussian_kde(values)(values)
            use_density = True
        except Exception:
            print("KDE failed — falling back to uniform color")
            density = None
            use_density = False

        if use_density:
            fig = px.scatter(
                x=preds,
                y=obs,
                color=density,
                color_continuous_scale=px.colors.sequential.Viridis,
                labels={"x": "Predicted (kg/ha)", "y": "Reference (kg/ha)", "color": "Point Density"},
                marginal_x="histogram",
                marginal_y="histogram",
            )
        else:
            fig = px.scatter(
                x=preds, y=obs,
                labels={"x": "Predicted (kg/ha)", "y": "Reference (kg/ha)"},
                marginal_x="histogram",
                marginal_y="histogram",
            )

    else:
        # color by year
        fig = px.scatter(
            x=preds, y=obs, color=obs_years,
            color_discrete_sequence=px.colors.qualitative.Plotly,
            labels={"x": "Predicted (kg/ha)", "y": "Reference (kg/ha)", "color": "Year"},
            marginal_x="histogram",
            marginal_y="histogram",
        )

    # add 1:1 line
    fig.add_trace(
        go.Scatter(
            x=[0, abs_max],
            y=[0, abs_max],
            mode="lines",
            name="1:1",
            line=dict(color="gray", dash="dash"),
        )
    )

    # add regression line
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"y = {slope:.2f}·x + {intercept:.2f}",
            line=dict(color="magenta"),
        )
    )

    fig.update_xaxes(range=xlims)
    fig.update_yaxes(range=ylims)

    fig.update_layout(template="simple_white", title="Predicted vs Reference Yield",)
    for ax in fig.layout:
        if ax.startswith('xaxis') or ax.startswith('yaxis'):
            fig.layout[ax].update(showgrid=False)
    return fig


def save_scatter_plot(fig, out_fpath, width=800, height=600):
    fig.write_image(out_fpath, width=width, height=height)
    return out_fpath

def get_preds_obs(estimation_fpath, val_fpath):
    gt = load_csv(val_fpath)
    pred = load_csv(estimation_fpath)

    gt["region"]   = gt["region"].astype(str)
    pred["region"] = pred["region"].astype(str)

    # Merging to ensure that the regions are in the same order
    combined = pd.merge(gt, pred, on='region')

    if 'mean_yield_kg_ha' not in combined.columns:
        raise ValueError("Column 'mean_yield_kg_ha' not found in the input estimation csv. Please ensure that the columns are named correctly.")

    # It might occur that the reported mean yield is not available in the input csv.
    # In this case, we can compute it from the reported yield and the total area.
    if 'reported_mean_yield_kg_ha' not in combined.columns:
        if not 'reported_production_kg' in combined.columns:
            raise ValueError("Could not compute metrics as neither 'reported_mean_yield_kg_ha' or 'reported_production_kg' are not available in the input csv.")

        combined['reported_mean_yield_kg_ha'] = combined['total_area_ha'] / combined['reported_production_kg']

    return {
        'obs': combined['reported_mean_yield_kg_ha'],
        'preds': combined['mean_yield_kg_ha']
    }


@click.command()
@click.option('--val_fpath', required=True, type=click.Path(exists=True), help='Filepath to the csv containing the refernece data per region.')
@click.option('--estimation_fpath', required=True, type=click.Path(exists=True), help='Filepath to the estimations per region csv.')
@click.option('--out_csv_fpath', required=True, type=click.Path(), help='Filepath where the resulting metrics should be saved (.csv).')
@click.option('--out_plot_fpath', required=True, type=click.Path(), help='Filepath where the resulting scatterplot should be saved (.png).')
@click.option('--verbose', is_flag=True, help='Set the logging level to DEBUG.')
def cli(val_fpath, estimation_fpath, out_csv_fpath, out_plot_fpath, verbose):
    """Wrapper for validation cli"""

    if verbose:
        logger.setLevel('INFO')

    logger.info("Loading data...")
    preds_obs = get_preds_obs(estimation_fpath=estimation_fpath, val_fpath=val_fpath)

    logger.info("Computing metrics...")
    metrics = compute_metrics(preds=preds_obs['preds'], obs=preds_obs['obs'])
    write_metrics(metrics, out_csv_fpath)

    logger.info("Creating scatter plot...")
    scatter_plot = create_scatter_plot(preds=preds_obs['preds'], obs=preds_obs['obs'])
    save_scatter_plot(scatter_plot, out_plot_fpath)

    logger.info("Done! Metrics and scatter plot saved successfully.")


if __name__ == '__main__':
    cli()
