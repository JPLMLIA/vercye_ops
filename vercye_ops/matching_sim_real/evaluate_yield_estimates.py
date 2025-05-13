import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
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


def colorize_plot_by_density(rawr, values, preds):
    fallback_kernel_used = False

    try:
        kernel = scipy.stats.gaussian_kde(values)(values)
    except:
        print("KDE failed due to low-dimensional data — using uniform color instead.")
        kernel = np.ones_like(preds)  # fallback uniform density
        fallback_kernel_used = True

    rawr.plot_joint(sns.scatterplot, c=kernel, cmap='viridis')
    rawr.plot_marginals(sns.kdeplot, fill=True)

    # Add colormap for scatter plot (placing it outside the plot)
    if not fallback_kernel_used:
        norm = plt.Normalize(kernel.min(), kernel.max())
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        fig = rawr.figure
        cbar_ax = fig.add_axes([1, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Point Density")
        cbar.ax.yaxis.get_offset_text().set_visible(False)


def colorize_plot_by_years(rawr, obs_years, preds, obs):
    # Create a color map based on the years
    unique_years = np.unique(obs_years)
    colors = sns.color_palette("hsv", len(unique_years))

    # Create a mapping from year to color
    year_to_color = {year: colors[i] for i, year in enumerate(unique_years)}

    # Map the colors to the observations
    obs_colors = [year_to_color[year] for year in obs_years]

    rawr.plot_joint(sns.scatterplot, c=obs_colors)
    rawr.plot_marginals(sns.kdeplot, fill=True)


def create_scatter_plot(preds, obs, obs_years=None):
    # Pass obs_years if you want to use it for coloring the points
    # obs_years should be the year of each observation

    if len(preds) != len(obs):
        raise ValueError("Length of the predictions and observations do not match.")
    
    if obs_years is not None and len(obs_years) != len(obs):
        raise ValueError("Length of the years and observations do not match.")

    abs_max = max(obs.max(), preds.max())
    values = np.vstack([preds, obs])

    rawr = sns.JointGrid(xlim=(-5, abs_max + 100), ylim=(-5, abs_max + 100), x=preds, y=obs)
    theta = np.polyfit(preds, obs, 1)
    y_line = theta[1] + theta[0] * preds

    if obs_years is None:
        colorize_plot_by_density(rawr, values, preds)
    else:
        colorize_plot_by_years(rawr, obs_years, preds, obs)

    rawr.ax_joint.plot(np.arange(0, abs_max), np.arange(0, abs_max), label='1:1', color='gray')
    rawr.ax_joint.plot(preds, y_line, color='magenta', label=f"y = {theta[0]:.2f} x + {theta[1]:.2f}")
    rawr.ax_joint.legend()
    rawr.set_axis_labels("Predicted (kg/ha)", "Observed (kg/ha)")

    return rawr

def save_scatter_plot(scatter_plot, out_fpath):
    scatter_plot.savefig(out_fpath, bbox_inches='tight', dpi=500)
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
        if not 'reported_yield_kg' in combined.columns:
            raise ValueError("Could not compute metrics as neither 'reported_mean_yield_kg_ha' or 'reported_yield_kg' are not available in the input csv.")

        combined['reported_mean_yield_kg_ha'] = combined['total_area_ha'] / combined['reported_yield_kg']

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
