import click
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


def load_csv(fpath):
    return pd.read_csv(fpath)


def compute_metrics(preds, obs):
    if len(preds) != len(obs):
        raise ValueError("Length of the predictions and observations do not match.")

    if len(preds) == 0 or len(obs) == 0:
        return {
            "n_regions": 0,
            "mape": None,
            "mean_err_kg_ha": None,
            "median_err_kg_ha": None,
            "mean_abs_err_kg_ha": None,
            "median_abs_err_kg_ha": None,
            "rmse_kg_ha": None,
            "rrmse": None,
            "r2_scikit": None,
            "r2_rsq_excel": None,
            "r2_scikit_bestfit": None,
        }

    errors_kg_ha = preds - obs
    mean_err_kg_ha = np.mean(errors_kg_ha)
    median_err_kg_ha = np.median(errors_kg_ha)
    mean_abs_err_kg_ha = np.mean(np.abs(errors_kg_ha))
    median_abs_err_kg_ha = np.median(np.abs(errors_kg_ha))

    rmse = np.sqrt(mean_squared_error(obs, preds))
    rrmse = rmse / np.mean(obs) * 100  # get percentage
    r2 = r2_score(obs, preds)
    r2_rsq_excel = (np.corrcoef(obs, preds)[0, 1]) ** 2

    theta = np.polyfit(preds, obs, 1)
    y_line = theta[1] + theta[0] * preds
    r2_scikit_bestfit = r2_score(obs, y_line)

    mape = mean_absolute_percentage_error(obs, preds)

    return {
        "n_regions": int(len(obs)),
        "mape": round(float(mape), 4),
        "mean_err_kg_ha": round(float(mean_err_kg_ha), 1),
        "median_err_kg_ha": round(float(median_err_kg_ha), 1),
        "mean_abs_err_kg_ha": round(float(mean_abs_err_kg_ha), 1),
        "median_abs_err_kg_ha": round(float(median_abs_err_kg_ha), 1),
        "rmse_kg_ha": round(float(rmse), 1),
        "rrmse": round(float(rrmse), 1),
        "r2_scikit": round(float(r2), 3),
        "r2_rsq_excel": round(float(r2_rsq_excel), 3),
        "r2_scikit_bestfit": round(float(r2_scikit_bestfit), 3),
    }


def compute_errors_per_region(preds, obs, region_names):
    if len(preds) == 0 or len(obs) == 0:
        return {
            "error_kg_ha": None,
            "rel_error_percent": None,
            "region": region_names,
        }

    errors_kg_ha = preds - obs
    rel_errors_percent = (errors_kg_ha / obs) * 100

    return {
        "error_kg_ha": np.round(errors_kg_ha, 1),
        "rel_error_percent": np.round(rel_errors_percent, 1),
        "region": region_names,
    }


def write_errors(errors, out_fpath):
    pd.DataFrame(errors).to_csv(out_fpath, index=False)
    return out_fpath


def write_metrics(metrics, out_fpath):
    pd.DataFrame(metrics, index=[0]).to_csv(out_fpath, index=False)
    return out_fpath


def create_scatter_plot(preds, obs, obs_years=None):
    preds = np.asarray(preds)
    obs = np.asarray(obs)

    if preds.size == 0 or obs.size == 0 or np.all(np.isnan(preds)) or np.all(np.isnan(obs)):
        # Create empty placeholder plot
        fig = go.Figure()
        fig.add_annotation(
            text="No data available to plot",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            template="simple_white",
            title="Predicted vs Reference Yield (No Data)",
            xaxis_title="Reference (kg/ha)",
            yaxis_title="Predicted (kg/ha)",
        )
        return fig

    if obs_years is not None:
        obs_years = np.asarray(obs_years)
        if len(obs_years) != len(obs):
            raise ValueError("Length of obs_years must match length of obs")

    # Plot limits
    min_plot = 0  # force origin
    max_val = max(preds.max(), obs.max())
    max_plot = max_val * 1.05  # small buffer above max

    # Tick spacing
    range_size = max_plot - min_plot
    tick_step = 500 if range_size <= 5000 else 1000
    tick_values = np.arange(min_plot, max_plot + tick_step, tick_step)

    # Regression line
    slope, intercept = np.polyfit(obs, preds, 1)
    x_line = np.linspace(min_plot, max_plot, 100)
    y_line = intercept + slope * x_line

    # Scatter color
    if obs_years is None:
        # Try KDE-based density coloring
        try:
            values = np.vstack([obs, preds])
            density = stats.gaussian_kde(values)(values)
            fig = px.scatter(
                x=obs,
                y=preds,
                color=density,
                color_continuous_scale=px.colors.sequential.Viridis,
                labels={
                    "x": "Reference (kg/ha)",
                    "y": "Predicted (kg/ha)",
                    "color": "Point Density",
                },
            )
        except Exception:
            fig = px.scatter(
                x=obs,
                y=preds,
                labels={"x": "Reference (kg/ha)", "y": "Predicted (kg/ha)"},
            )
    else:
        # Color points by observation year
        fig = px.scatter(
            x=obs,
            y=preds,
            color=obs_years,
            color_discrete_sequence=px.colors.qualitative.Plotly,
            labels={
                "x": "Reference (kg/ha)",
                "y": "Predicted (kg/ha)",
                "color": "Year",
            },
        )

    # Add 1:1 line
    fig.add_trace(
        go.Scatter(
            x=[min_plot, max_plot],
            y=[min_plot, max_plot],
            mode="lines",
            name="1:1",
            line=dict(color="gray", dash="dash"),
        )
    )

    # Add regression line
    if len(preds) > 1:
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
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    fig.update_xaxes(
        range=[min_plot, max_plot],
        tickangle=0,
        tickmode="array",
        tickvals=tick_values,
        showgrid=False,
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=1,
        constrain="domain",
    )
    fig.update_yaxes(
        range=[min_plot, max_plot],
        tickangle=0,
        tickmode="array",
        tickvals=tick_values,
        showgrid=False,
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=1,
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


def save_scatter_plot(fig, out_fpath, width=800, height=600):
    fig.write_image(out_fpath, width=width, height=height)
    return out_fpath


def get_preds_obs(estimation_fpath, val_fpath, pixel_converted=True):
    gt = load_csv(val_fpath)
    pred = load_csv(estimation_fpath)

    predictions_column = "mean_yield_kg_ha" if pixel_converted else "apsim_mean_yield_estimate_kg_ha"

    if "reported_mean_yield_kg_ha" in pred.columns:
        pred.drop(["reported_mean_yield_kg_ha"], axis="columns", inplace=True)

    gt["region"] = gt["region"].astype(str)
    pred["region"] = pred["region"].astype(str)

    # Merging to ensure that the regions are in the same order
    combined = pd.merge(gt, pred, on="region")

    if predictions_column not in combined.columns:
        raise ValueError(
            f"Column {predictions_column} not found in the input estimation csv. Please ensure that the columns are named correctly."
        )

    # It might occur that the reported mean yield is not available in the input csv.
    # In this case, we can compute it from the reported yield and the total area.
    if "reported_mean_yield_kg_ha" not in combined.columns:
        if "reported_production_kg" not in combined.columns:
            raise ValueError(
                "Could not compute metrics, since neither 'reported_mean_yield_kg_ha' or 'reported_production_kg' are not available in the input csv."
            )

        combined["reported_mean_yield_kg_ha"] = combined["reported_production_kg"] / combined["total_area_ha"]

    # Drop all entries where the reported mean yield is NaN
    combined = combined.dropna(subset=["reported_mean_yield_kg_ha", predictions_column])

    return {
        "obs": combined["reported_mean_yield_kg_ha"],
        "preds": combined[predictions_column],
        "region": combined["region"],
    }


@click.command()
@click.option(
    "--val-fpath",
    required=True,
    type=click.Path(exists=True),
    help="Filepath to the csv containing the reference data per region.",
)
@click.option(
    "--estimation-fpath",
    required=True,
    type=click.Path(exists=True),
    help="Filepath to the estimations per region csv.",
)
@click.option(
    "--out-eval-fpath",
    required=True,
    type=click.Path(),
    help="Filepath where the overall resulting metrics should be saved (.csv).",
)
@click.option(
    "--out-errors-fpath",
    required=True,
    type=click.Path(),
    help="Filepath where the region-wise resulting metrics should be saved (.csv).",
)
@click.option(
    "--out-plot-fpath",
    required=True,
    type=click.Path(),
    help="Filepath where the resulting scatterplot should be saved (.png).",
)
@click.option("--verbose", is_flag=True, help="Set the logging level to DEBUG.")
def cli(val_fpath, estimation_fpath, out_eval_fpath, out_errors_fpath, out_plot_fpath, verbose):
    """Wrapper for validation cli"""

    if verbose:
        logger.setLevel("INFO")

    logger.info("Loading data...")
    preds_obs = get_preds_obs(estimation_fpath=estimation_fpath, val_fpath=val_fpath)
    # Try loading the apsim-only (non-pixel-converted) predictions.
    # After the aggregation refactor, CSVs may not contain apsim_mean_yield_estimate_kg_ha;
    # in that case, skip the apsim-only evaluation gracefully.
    try:
        preds_obs_apsimonly = get_preds_obs(
            estimation_fpath=estimation_fpath, val_fpath=val_fpath, pixel_converted=False
        )
    except ValueError:
        preds_obs_apsimonly = None

    logger.info("Computing metrics...")
    metrics = compute_metrics(preds=preds_obs["preds"], obs=preds_obs["obs"])
    write_metrics(metrics, out_eval_fpath)

    if preds_obs_apsimonly is not None:
        metrics_apsimonly = compute_metrics(preds=preds_obs_apsimonly["preds"], obs=preds_obs_apsimonly["obs"])
        out_eval_fpath_apsim = str(out_eval_fpath).replace(".csv", "_no-pixel-conversion.csv")
        write_metrics(metrics_apsimonly, out_eval_fpath_apsim)

    logger.info("Computing errors...")
    errors = compute_errors_per_region(preds_obs["preds"], obs=preds_obs["obs"], region_names=preds_obs["region"])
    write_errors(errors, out_errors_fpath)

    logger.info("Creating scatter plot...")
    scatter_plot = create_scatter_plot(preds=preds_obs["preds"], obs=preds_obs["obs"])
    save_scatter_plot(scatter_plot, out_plot_fpath)

    if preds_obs_apsimonly is not None:
        scatter_plot_apsimonly = create_scatter_plot(preds=preds_obs_apsimonly["preds"], obs=preds_obs_apsimonly["obs"])
        out_plot_fpath_apsimonly = str(out_plot_fpath).replace(".png", "_no-pixel-conversion.png")
        save_scatter_plot(scatter_plot_apsimonly, out_plot_fpath_apsimonly)

    logger.info("Done! Metrics and scatter plot saved successfully.")


if __name__ == "__main__":
    cli()
