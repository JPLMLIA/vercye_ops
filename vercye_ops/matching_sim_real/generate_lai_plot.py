import os
from glob import glob

import click
import pandas as pd
import plotly.graph_objects as go


def load_lai_files(input_dir: str):
    return glob(os.path.join(input_dir, "*", "*_LAI_STATS.csv"))


def parse_lai_file(filepath: str, lai_agg_type: str):
    df = pd.read_csv(filepath)
    filename = os.path.basename(filepath)
    region_name = "_".join(filename.split("_")[:-2])
    lai_type_col = "Mean" if lai_agg_type.lower() == "mean" else "Median"
    return df, region_name, lai_type_col


def create_lai_traces(
    df, region_name, lai_type_col, adjusted_color="blue", non_adjusted_color="orange"
):
    adjusted_trace = go.Scatter(
        y=df[f"LAI {lai_type_col} Adjusted"],
        mode="lines",
        name=f"{region_name} (adjusted)",
        line=dict(color=adjusted_color),
        showlegend=True,
    )

    non_adjusted_trace = go.Scatter(
        y=df[f"LAI {lai_type_col}"],
        mode="lines",
        name=f"{region_name} (non-adjusted)",
        line=dict(color=non_adjusted_color),
        showlegend=True,
    )

    return adjusted_trace, non_adjusted_trace


def generate_lai_figure(filepaths, lai_agg_type):
    fig = go.Figure()
    adjusted_color = "blue"
    non_adjusted_color = "orange"
    adjusted_traces = []
    non_adjusted_traces = []

    for filepath in filepaths:
        df, region_name, lai_type_col = parse_lai_file(filepath, lai_agg_type)
        adj_trace, non_adj_trace = create_lai_traces(
            df, region_name, lai_type_col, adjusted_color, non_adjusted_color
        )
        fig.add_trace(adj_trace)
        adjusted_traces.append(len(fig.data) - 1)
        fig.add_trace(non_adj_trace)
        non_adjusted_traces.append(len(fig.data) - 1)

    fig.update_layout(
        title=f"LAI {lai_type_col} Adjusted vs Non-Adjusted",
        xaxis_title="Time Index",
        yaxis_title="LAI",
        legend_title="Click to Toggle Traces",
        template="plotly_white",
    )

    fig = fig.update_traces(opacity=0.6)

    # Toggle buttons
    all_visible = [True] * len(fig.data)
    hide_adjusted = [i not in adjusted_traces for i in range(len(fig.data))]
    hide_non_adjusted = [i not in non_adjusted_traces for i in range(len(fig.data))]

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=[
                    dict(
                        args=[{"visible": all_visible}],
                        label="Show All",
                        method="update",
                    ),
                    dict(
                        args=[{"visible": hide_adjusted}],
                        label="Hide Adjusted",
                        method="update",
                    ),
                    dict(
                        args=[{"visible": hide_non_adjusted}],
                        label="Hide Non-Adjusted",
                        method="update",
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1,
                xanchor="left",
                y=1.1,
                yanchor="top",
            )
        ]
    )

    return fig


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True),
    help="Directory containing the input files.",
)
@click.option("--output-fpath", type=click.Path(), help="Path to the output file (.html).")
@click.option(
    "--lai_agg_type",
    required=True,
    type=click.Choice(["mean", "median"]),
    help='Type of how the LAI was aggregated over a ROI. "mean" or "median" supported.',
)
def cli(input_dir, output_fpath, lai_agg_type):
    all_lai_files = load_lai_files(input_dir)
    if not all_lai_files:
        raise ValueError(f"No LAI files found in {input_dir}. Please check the directory.")

    fig = generate_lai_figure(all_lai_files, lai_agg_type)

    with open(output_fpath, "w") as f:
        f.write(fig.to_html(include_plotlyjs="cdn", full_html=True))


if __name__ == "__main__":
    cli()
