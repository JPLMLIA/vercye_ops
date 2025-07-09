import click
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

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
    median_abs_err_kg_ha = np.median(np.abs(errors_kg_ha))

    rmse = np.sqrt(mean_squared_error(obs, preds))
    rrmse = rmse / np.mean(obs) * 100 # get percentage
    r2 = r2_score(obs, preds)
    r2_rsq_excel = (np.corrcoef(obs, preds)[0, 1]) ** 2

    theta = np.polyfit(preds, obs, 1)
    y_line = theta[1] + theta[0] * preds
    r2_scikit_bestfit = r2_score(obs, y_line)

    aggregated_metrics = {
        'n_regions': len(obs),
        'mean_err_kg_ha': mean_err_kg_ha,
        'median_err_kg_ha': median_err_kg_ha,
        'mean_abs_err_kg_ha': mean_abs_err_kg_ha,
        'median_abs_err_kg_ha': median_abs_err_kg_ha,
        'rmse_kg_ha': rmse,
        'rrmse': rrmse,
        'r2_scikit': r2,
        'r2_rsq_excel': r2_rsq_excel,
        'r2_scikit_bestfit': r2_scikit_bestfit
    }

    return aggregated_metrics


def compute_errors_per_region(preds, obs, region_names):
    errors_kg_ha = obs - preds
    rel_errors_percent = errors_kg_ha / obs

    return {
        'error_kg_ha': errors_kg_ha,
        'rel_error_percent': rel_errors_percent,
        'region': region_names
    }

def write_errors(errors, out_fpath):
    pd.DataFrame(errors).to_csv(out_fpath, index=False)
    return out_fpath


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
    min_plot  = min(0, min(preds.min(), obs.min()) - 100)
    max_val  = max(preds.max(), obs.max())
    max_plot = max_val + 100

    # Create explicit tick values with 500 or 1000 spacing
    range_size = max_plot - min_plot
    if range_size <= 5000:
        tick_step = 500
    else:
        tick_step = 1000
    
    # Start from the first multiple of tick_step at or below min_plot
    tick_start = (min_plot // tick_step) * tick_step
    tick_values = np.arange(tick_start, max_plot + tick_step, tick_step)

    # compute regression line (use full range for line)
    slope, intercept = np.polyfit(preds, obs, 1)
    x_line = np.linspace(min_plot, max_plot, 100)
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
            )
        else:
            fig = px.scatter(
                x=preds, y=obs,
                labels={"x": "Predicted (kg/ha)", "y": "Reference (kg/ha)"},
            )

    else:
        # color by year
        fig = px.scatter(
            x=preds, y=obs, color=obs_years,
            color_discrete_sequence=px.colors.qualitative.Plotly,
            labels={"x": "Predicted (kg/ha)", "y": "Reference (kg/ha)", "color": "Year"},
        )

    # add 1:1 line (corrected coordinates)
    fig.add_trace(
        go.Scatter(
            x=[min_plot, max_plot],
            y=[min_plot, max_plot],
            mode="lines",
            name="1:1",
            line=dict(color="gray", dash="dash"),
        )
    )

    if len(preds > 1):
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

    fig.update_layout(
        template="simple_white",
        title="Predicted vs Reference Yield",
        autosize=True,
        margin=dict(l=60, r=20, t=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center",  x=0.5
        )
    )
    fig.update_xaxes(tickangle=0, tickmode="array", tickvals=tick_values)
    fig.update_yaxes(tickangle=0, scaleanchor="x", scaleratio=1, tickmode="array", tickvals=tick_values)

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

    gt["region"] = gt["region"].astype(str)
    pred["region"] = pred["region"].astype(str)

    # Merging to ensure that the regions are in the same order
    combined = pd.merge(gt, pred, on='region')

    if 'mean_yield_kg_ha' not in combined.columns:
        raise ValueError("Column 'mean_yield_kg_ha' not found in the input estimation csv. Please ensure that the columns are named correctly.")

    # It might occur that the reported mean yield is not available in the input csv.
    # In this case, we can compute it from the reported yield and the total area.
    if 'reported_mean_yield_kg_ha' not in combined.columns:
        if not 'reported_production_kg' in combined.columns:
            raise ValueError("Could not compute metrics, since neither 'reported_mean_yield_kg_ha' or 'reported_production_kg' are not available in the input csv.")

        combined['reported_mean_yield_kg_ha'] = combined['reported_production_kg'] / combined['total_area_ha']

    # Drop all entries where the reported mean yield is NaN
    combined = combined.dropna(subset=['reported_mean_yield_kg_ha', 'mean_yield_kg_ha'])

    return {
        'obs': combined['reported_mean_yield_kg_ha'],
        'preds': combined['mean_yield_kg_ha'],
        'region': combined['region']
    }


@click.command()
@click.option('--val-fpath', required=True, type=click.Path(exists=True), help='Filepath to the csv containing the refernece data per region.')
@click.option('--estimation-fpath', required=True, type=click.Path(exists=True), help='Filepath to the estimations per region csv.')
@click.option('--out-eval-fpath', required=True, type=click.Path(), help='Filepath where the overall resulting metrics should be saved (.csv).')
@click.option('--out-errors-fpath', required=True, type=click.Path(), help='Filepath where the region-wise resulting metrics should be saved (.csv).')
@click.option('--out-plot-fpath', required=True, type=click.Path(), help='Filepath where the resulting scatterplot should be saved (.png).')
@click.option('--verbose', is_flag=True, help='Set the logging level to DEBUG.')
def cli(val_fpath, estimation_fpath, out_eval_fpath, out_errors_fpath, out_plot_fpath, verbose):
    """Wrapper for validation cli"""

    if verbose:
        logger.setLevel('INFO')

    logger.info("Loading data...")
    preds_obs = get_preds_obs(estimation_fpath=estimation_fpath, val_fpath=val_fpath)

    logger.info("Computing metrics...")
    metrics = compute_metrics(preds=preds_obs['preds'], obs=preds_obs['obs'])
    write_metrics(metrics, out_eval_fpath)

    logger.info("Computing errors...")
    errors = compute_errors_per_region(preds_obs['preds'], obs=preds_obs['obs'], region_names=preds_obs['region'])
    write_errors(errors, out_errors_fpath)

    logger.info("Creating scatter plot...")
    scatter_plot = create_scatter_plot(preds=preds_obs['preds'], obs=preds_obs['obs'])
    save_scatter_plot(scatter_plot, out_plot_fpath)

    logger.info("Done! Metrics and scatter plot saved successfully.")


if __name__ == '__main__':
    cli()
