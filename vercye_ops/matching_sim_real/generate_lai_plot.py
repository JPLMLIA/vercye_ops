import os
from glob import glob
import pandas as pd
import plotly.graph_objects as go
import click

@click.command()
@click.option('--input-dir', type=click.Path(exists=True), help='Directory containing the input files.')
@click.option('--output-fpath', type=click.Path(), help='Path to the output file (.html).')
@click.option('--lai_agg_type', required=True, type=click.Choice(['mean', 'median']), help='Type of how the LAI was aggregated over a ROI. "mean" or "median" supported.')
def cli(input_dir, output_fpath, lai_agg_type):
    all_lai_files = glob(os.path.join(input_dir, '*', '*_LAI_STATS.csv'))
    fig = go.Figure()
    adjusted_color = 'blue'
    non_adjusted_color = 'orange'

    lai_type_col = 'Mean' if lai_agg_type.lower() == 'mean' else 'Median'
    
    adjusted_traces = []
    non_adjusted_traces = []
    
    # Add LAI traces for each file
    for lai_file in all_lai_files:
        df = pd.read_csv(lai_file)
        filename = os.path.basename(lai_file)
        region_name = "_".join(filename.split('_')[:-2])
        
        adjusted_trace = go.Scatter(
            y=df[f'LAI {lai_type_col} Adjusted'],
            mode='lines',
            name=f'{region_name} (adjusted)',
            line=dict(color=adjusted_color),
            showlegend=True
        )
        fig.add_trace(adjusted_trace)
        adjusted_traces.append(len(fig.data) - 1)
        
        non_adjusted_trace = go.Scatter(
            y=df[f'LAI {lai_type_col}'],
            mode='lines',
            name=f'{region_name} (non-adjusted)',
            line=dict(color=non_adjusted_color),
            showlegend=True
        )

        fig.add_trace(non_adjusted_trace)
        non_adjusted_traces.append(len(fig.data) - 1)
    
    fig.update_layout(
        title=f'LAI {lai_type_col} Adjusted vs Non-Adjusted',
        xaxis_title='Time Index',
        yaxis_title='LAI',
        legend_title='Click to Toggle Traces',
        template='plotly_white'
    )
    
    # Create visibility lists for the toggles
    all_visible = [True] * len(fig.data)
    hide_adjusted = all_visible.copy()
    for idx in adjusted_traces:
        hide_adjusted[idx] = False
        
    hide_non_adjusted = all_visible.copy()
    for idx in non_adjusted_traces:
        hide_non_adjusted[idx] = False
    
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=[
                    dict(
                        args=[{"visible": all_visible}],
                        label="Show All",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": hide_adjusted}],
                        label="Hide Adjusted",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": hide_non_adjusted}],
                        label="Hide Non-Adjusted",
                        method="update"
                    )
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            )
        ]
    )
    
    with open(output_fpath, 'w') as f:
        f.write(fig.to_html(include_plotlyjs='cdn', full_html=True))

if __name__ == "__main__":
    cli()