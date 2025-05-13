import os
import click
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from vercye_ops.matching_sim_real.evaluate_yield_estimates import compute_metrics, create_scatter_plot, get_preds_obs
from vercye_ops.matching_sim_real.generate_lai_plot import load_lai_files, parse_lai_file


color_palette = px.colors.qualitative.Plotly


def get_available_years(input_dir):
    dirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    return dirs

def get_available_timepoints(reference_year_dir):
    timepoints = sorted([d for d in os.listdir(reference_year_dir) if os.path.isdir(os.path.join(reference_year_dir, d))])
    return timepoints

def plot_lai_figure(input_dir, timepoint, years, lai_agg_type, adjusted):
    filepaths = {}
    for year in years:
        year_dir = os.path.join(input_dir, year, timepoint)
        filepaths[year] = load_lai_files(year_dir)

    fig = go.Figure()

    combined_data = []

    for filepath in filepaths:
        df, region_name, _ = parse_lai_file(filepath, lai_agg_type)
        df['Date'] = pd.to_datetime(df['Date'])
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['Year'] = df['Date'].dt.year
        df['Region'] = region_name
        combined_data.append(df)

    full_df = pd.concat(combined_data)

    lai_col_name = 'LAI '
    if lai_agg_type.lower() == 'mean':
        lai_col_name += 'Mean'
    elif lai_agg_type.lower() == 'median':
        lai_col_name += 'Median'
    else:
        raise ValueError("Invalid LAI aggregation type. Choose 'mean' or 'median'.")
    
    if adjusted:
        lai_col_name += ' Adjusted'

    color_index = 0
    for (region, year), group in full_df.groupby(['Region', 'Year']):
        color = color_palette[color_index % len(color_palette)]
        label_name = f"{region} {year}"

        # Adjusted trace
        fig.add_trace(go.Scatter(
            x=group['DayOfYear'],
            y=group[lai_col_name],
            mode='lines',
            name=f'{label_name}',
            line=dict(width=2, color=color),
            showlegend=True
        ))

        color_index += 1

    adjusted_str = 'Adjusted' if adjusted else 'Non-Adjusted'   
    fig.update_layout(
        title=f'LAI {adjusted_str} by Day of Year',
        xaxis_title='Day of Year',
        yaxis_title=f'LAI {adjusted_str}',
        template='plotly_white',
        legend_title='Region-Year',
        height=700
    )

    return fig

def load_obs_preds(input_dir, timepoint, years):
    obs = []
    preds = []
    obs_years = []

    for year in years:
        year_dir = os.path.join(input_dir, year, timepoint)
        estimation_fpath = os.path.join(year_dir, 'estimation.csv')
        val_fpath = os.path.join(year_dir, 'validation.csv')

        data = get_preds_obs(estimation_fpath, val_fpath)
        obs.append(data['obs'])
        preds.append(data['preds'])
        obs_years.append([year] * len(data['obs']))

    return pd.concat(obs), pd.concat(preds), pd.concat(obs_years)
  

@click.command()
@click.option('--input_dir', type=click.Path(exists=True), help='Directory containing Years.')
@click.option('--lai_agg_type', type=click.Choice(['mean', 'median'], case_sensitive=False), default='mean', help='Type of LAI aggregation to plot.')
@click.option('--adjusted', is_flag=True, help='Plot adjusted LAI values.')
@click.option('--output_file', type=click.Path(), help='File in which to save the resulting plots (.html).')
def main(input_dir, lai_agg_type, adjusted, output_file):
    years = get_available_years(input_dir)

    reference_year_dir  = os.path.join(input_dir, years[0])
    timepoints = get_available_timepoints(reference_year_dir)

    # For each timepoint creating figures, combining all years into a single plot and compute metrics
    full_html = ""

    for timepoint in timepoints:
        # LAI plot
        lai_fig = plot_lai_figure(input_dir, timepoint, years, lai_agg_type, adjusted)

        # Scatter plot
        obs, preds, obs_years = load_obs_preds(input_dir, timepoint, years)
        scatter_fig = create_scatter_plot(obs, preds, obs_years)

        # Metrics
        metrics = compute_metrics(input_dir, timepoint, years)
        metrics_html = f"<pre><b>Metrics for {timepoint}:</b>\n{metrics}</pre>"

        # Combine plots using subplots
        subplot_fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.15,
            subplot_titles=(f"Scatter Plot - {timepoint}", f"LAI Plot - {timepoint}")
        )

        for trace in scatter_fig['data']:
            subplot_fig.add_trace(trace, row=1, col=1)

        for trace in lai_fig['data']:
            subplot_fig.add_trace(trace, row=2, col=1)

        subplot_fig.update_layout(height=1000, showlegend=True)

        # Convert to HTML and add metrics
        fig_html = pio.to_html(subplot_fig, include_plotlyjs='cdn', full_html=False)
        full_html += f"<h2>{timepoint}</h2>{fig_html}{metrics_html}<hr>"

    # Save all in one HTML file
    full_html_doc = f"""
    <html>
        <head>
            <title>Multiyear Interactive Summary</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            {full_html}
        </body>
    </html>
    """
    with open(output_file, 'w') as f:
        f.write(full_html_doc)

if __name__ == "__main__":
    main()