import os
from glob import glob
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import click


def load_lai_files(input_dir: str):
    return glob(os.path.join(input_dir, '*', '*_LAI_STATS.csv'))


def parse_lai_file(filepath: str, lai_agg_type: str, adjusted: bool):
    df = pd.read_csv(filepath)
    filename = os.path.basename(filepath)
    region_name = "_".join(filename.split('_')[:-2])
    lai_type_col = 'Mean' if lai_agg_type.lower() == 'mean' else 'Median'

    if adjusted:
        lai_type_col += ' Adjusted'

    return df, region_name, lai_type_col


def create_lai_traces(df, shared_dates, region_name, lai_type_col, color):
    trace = go.Scatter(
        x=shared_dates,
        y=df[f'LAI {lai_type_col}'],
        mode='lines',
        name=f'{region_name}',
        line=dict(color=color),
        showlegend=True,
        opacity=0.5
    )

    return trace


def get_shared_dates(filepaths):
    """
    Extract dates from the first file to use as shared x-axis.
    Assumes all files have the same date sequence.
    """
    if not filepaths:
        raise ValueError("No file paths provided.")
    
    # Read the first file to get the date column
    first_df = pd.read_csv(filepaths[0])
    if 'Date' in first_df.columns:
        return pd.to_datetime(first_df['Date'], format='%d/%m/%Y')
    else:
        raise ValueError(f"'Date' column not found in {filepaths[0]}. Please check the file format.")


def generate_lai_figure(filepaths, lai_agg_type, adjusted):
    fig = go.Figure()

    shared_dates = get_shared_dates(filepaths)

    # generate len(filepaths) colors that are distinct
    color_sequence = px.colors.qualitative.Plotly  # or D3, Set3, Pastel, Dark2, etc.

    combined_df = None

    for idx, filepath in enumerate(filepaths):
        df, region_name, lai_type_col = parse_lai_file(filepath, lai_agg_type, adjusted)
        trace = create_lai_traces(df, shared_dates, region_name, lai_type_col, color_sequence[idx % len(color_sequence)])
        fig.add_trace(trace)


    fig.update_layout(
        title=f'LAI {lai_type_col}',
        xaxis_title='Date',
        yaxis_title='LAI',
        legend_title='Click to Toggle Traces',
        template='plotly_white'
    )

    return fig


@click.command()
@click.option('--input-dir', type=click.Path(exists=True), help='Directory containing the input files.')
@click.option('--output-fpath', type=click.Path(), help='Path to the output file (.html).')
@click.option('--lai_agg_type', required=True, type=click.Choice(['mean', 'median']), help='Type of how the LAI was aggregated over a ROI. "mean" or "median" supported.')
@click.option('--adjusted', is_flag=True, default=False, help='Whether to plot adjusted LAI values.')
def cli(input_dir, output_fpath, lai_agg_type, adjusted):
    all_lai_files = load_lai_files(input_dir)
    if not all_lai_files:
        raise ValueError(f"No LAI files found in {input_dir}. Please check the directory.")
    
    fig = generate_lai_figure(all_lai_files, lai_agg_type, adjusted)
    
    with open(output_fpath, 'w') as f:
        f.write(fig.to_html(include_plotlyjs='cdn', full_html=True))

if __name__ == "__main__":
    cli()