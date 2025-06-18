import os
from glob import glob
import click
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from vercye_ops.matching_sim_real.evaluate_yield_estimates import (
    compute_metrics, create_scatter_plot, get_preds_obs, load_csv
)
from vercye_ops.matching_sim_real.generate_lai_plot import load_lai_files, parse_lai_file

# Use Plotly's qualitative palette
color_palette = px.colors.qualitative.Plotly
mean_palette  = px.colors.qualitative.Set1


# HTML template with Bootstrap for a modern, responsive layout
# Note: double braces {{ }} in CSS to escape Python formatting
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>{title}</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ background-color: #f8f9fa; font-family: 'Arial', sans-serif; padding: 20px; }}
        h1, h2, h3 {{ color: #343a40; }}
        .card {{ box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .plot-container {{ margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="my-4 text-center">{title}</h1>
        {content}
    </div>
</body>
</html>
"""


def get_available_years(input_dir):
    # Might want to revisit this as this assumes that the input dir only contains directories named by year
    years = []

    for year in os.listdir(input_dir):
        year_path = os.path.join(input_dir, year)
        if os.path.isdir(year_path) and year.isdigit():
            years.append(year)

    return sorted(years)


def get_available_timepoints(reference_year_dir):
    return sorted([d for d in os.listdir(reference_year_dir) if os.path.isdir(os.path.join(reference_year_dir, d))])


def plot_lai_figure(input_dir, timepoint, years, lai_agg_type, adjusted):
    combined = []
    for year in years:
        for fp in load_lai_files(os.path.join(input_dir, year, timepoint)):
            df, region, _ = parse_lai_file(fp, lai_agg_type, adjusted)
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
            start = df['Date'].min()
            df['DaysAfterStart'] = (df['Date'] - start).dt.days
            df['Year']   = int(year)
            df['Region'] = region
            combined.append(df)
    
    if not combined:
        return None

    full_df = pd.concat(combined, ignore_index=True)

    col = 'LAI ' + ('Mean' if lai_agg_type=='mean' else 'Median')
    if adjusted:
        col += ' Adjusted'

    fig = go.Figure()
    year_traces = {y: [] for y in years}

    for (region, year), grp in full_df.groupby(['Region','Year']):
        idx = len(fig.data)
        fig.add_trace(go.Scatter(
            x=grp['DaysAfterStart'], y=grp[col], mode='lines',
            name=f"{region} {year}",
            legendgroup=str(year),
            line=dict(width=2, color=color_palette[int(year) % len(color_palette)]),
            opacity=0.5,
            showlegend=True,   # hide from legend to avoid clutter
            visible=False
        ))
        year_traces[str(year)].append(idx)

    for i, year in enumerate(years):
        df_y = full_df[full_df['Year']==int(year)]
        if df_y.empty:
            continue
        m = df_y.groupby('DaysAfterStart')[col].mean().reset_index()
        fig.add_trace(go.Scatter(
            x=m['DaysAfterStart'], y=m[col],
            mode='lines',
            name=f"{year} Mean",
            legendgroup=str(year),
            line=dict(width=6, color=mean_palette[i % len(mean_palette)]),
            opacity=1,
            showlegend=True,
            visible=True
        ))

    year_buttons = []
    for year, idxs in year_traces.items():
        if not idxs:
            continue
        year_buttons.append(dict(
            label=year,
            method='restyle',
            args=[{'visible': True}, idxs],
            args2=[{'visible': False}, idxs]
        ))

    fig.update_layout(
        title=dict(text=f"{col} by Days after Simulation Start", x=0.5),
        xaxis_title='Days after Simulation Start',
        yaxis_title=f'LAI {"Adjusted" if adjusted else "Non-Adjusted"}',
        template='plotly_white',
        margin=dict(l=40, r=40, t=80, b=40),
        height=600,
        font=dict(family='Arial, sans-serif', size=12),
        updatemenus=[
            dict(
                type='buttons',
                direction='left',
                buttons=year_buttons,
                x=1.1, y=1.1,
                showactive=False,
                bgcolor='white',
                bordercolor='LightSkyBlue',
                borderwidth=1,
                font=dict(size=12)
            )
        ]
    )

    global_start = full_df['Date'].min().strftime('%d/%m/%Y')
    fig.add_annotation(
        x=0, y=1.05, xref='x', yref='paper',
        text=f"Simulation start: {global_start}",
        showarrow=False, font=dict(size=12), align='left'
    )

    return fig

def load_obs_preds(input_dir, timepoint, years, agg_levels):
    results = {}
    for lvl in agg_levels:
        all_preds = []
        all_preds_years = []
        preds_for_obs = []
        all_obs = []
        all_obs_years = []

        for year in years:
            base = os.path.join(input_dir, year, timepoint)
            est = glob(os.path.join(base, f'agg_yield_estimates_{lvl}*.csv'))

            if not est:
                continue

            if len(est) > 1:
                raise ValueError(f"Multiple yield estimate files found for {year} at level {lvl}: {est}")

            preds_df = load_csv(est[0])
            all_preds.extend(preds_df['mean_yield_kg_ha'])
            all_preds_years.extend([year]*len(preds_df))
            
            val = glob(os.path.join(input_dir, year, f'groundtruth_{lvl}*.csv'))
            if val:
                data = get_preds_obs(est[0], val[0])
                all_obs.extend(data['obs'])
                preds_for_obs.extend(data['preds'])
                all_obs_years.extend([year]*len(data['obs']))
    
        results[lvl] = {
            'only_preds': (all_preds, all_preds_years),
            'obs_preds': (all_obs, preds_for_obs, all_obs_years)
        }
    return results


def create_predictions_plot(preds, years):
    df = pd.DataFrame({'Predictions': preds, 'Year': years})
    fig = go.Figure()
    for yr, grp in df.groupby('Year'):
        fig.add_trace(
            go.Violin(
                x=[yr]*len(grp), y=grp['Predictions'], name=yr,
                box_visible=True, meanline_visible=True
            )
        )
    fig.update_layout(
        title=dict(text='Yield Predictions Distribution by Year from all simulation regions.', x=0.5),
        template='plotly_white', xaxis_title='Year', yaxis_title='Yield (kg/ha)',
        margin=dict(l=40, r=40, t=60, b=40), height=500,
        font=dict(family='Arial, sans-serif', size=12)
    )
    return fig


def identify_agg_levels(input_dir, years):
    lvls = set(['primary'])
    for y in years:
        files = glob(os.path.join(input_dir, y, '*', 'agg_yield_estimates_*.csv'))
        for f in files:
            lvl = os.path.basename(f).split('_')[3]
            lvls.add(lvl)
    return sorted(lvls)


@click.command()
@click.option('--input-dir', type=click.Path(exists=True), required=True)
@click.option('--lai-agg-type', type=click.Choice(['mean', 'median']), default='mean')
@click.option('--adjusted', is_flag=True)
@click.option('--title', type=str, default='Multiyear Interactive Summary', help='Title for the HTML report. Enclose in quotes if it contains spaces.')
@click.option('--output-file', type=click.Path(), required=True)
def main(input_dir, lai_agg_type, adjusted, title, output_file):
    years = get_available_years(input_dir)
    reference = os.path.join(input_dir, years[0])
    timepoints = get_available_timepoints(reference)
    agg_levels = identify_agg_levels(input_dir, years)

    content = []
    for tp in timepoints:
        # LAI Plot
        lai_fig = plot_lai_figure(input_dir, tp, years, lai_agg_type, adjusted)
        if lai_fig is not None:
            lai_html = pio.to_html(lai_fig, include_plotlyjs='cdn', full_html=False)
            content.append(
                f"""
                <div class='card mb-4'>
                <div class='card-header'><h2>{tp} - LAI</h2></div>
                <div class='card-body plot-container'>
                    {lai_html}
                    <p><em>Click on the 'year' buttons (top right) to toggle visibility of all individual LAI traces from that year (this might take a few seconds to load).</br>
                    Notice: You might have to click twice.</em></p>
                </div>
                </div>
                """
            )

        # Predictions + Metrics per aggregation level
        obs_preds = load_obs_preds(input_dir, tp, years, agg_levels)

        for lvl, data in obs_preds.items():
            all_preds, preds_years = data['only_preds']

            if len(all_preds) == 0:
                continue

            pred_fig = create_predictions_plot(all_preds, preds_years)
            pred_html = pio.to_html(pred_fig, include_plotlyjs='cdn', full_html=False)
            metrics_html = "<p><em>No ground-truth available for metrics.</em></p>"

            obs, preds, yrs = data['obs_preds']
            if obs:
                scatter = create_scatter_plot(preds, obs, yrs)
                scatter_html = pio.to_html(scatter, include_plotlyjs='cdn', full_html=False)
                metrics = compute_metrics(np.array(preds), np.array(obs))
                metrics_rows = "".join(
                    f"<tr><th scope='row'>{k}</th><td>{v:.3f}</td></tr>"
                    for k, v in metrics.items()
                )

                metrics_html = f"""
                <div class='metrics-table mb-3'>
                    <strong>Metrics ({lvl}):</strong>
                    <table class='table table-sm table-bordered mt-2'>
                    <thead class='thead-light'>
                        <tr><th>Metric</th><th>Value</th></tr>
                    </thead>
                    <tbody>
                        {metrics_rows}
                    </tbody>
                    </table>
                </div>
                {scatter_html}
                """

            content.append(
                f"""
                <div class='card mb-4'>
                  <div class='card-header'><h3>{tp} - Predictions {lvl}</h3></div>
                  <div class='card-body plot-container'>{pred_html}{metrics_html}</div>
                </div>
                """
            )

    # Render full HTML and write
    full_html = HTML_TEMPLATE.format(content=''.join(content), title=title)
    with open(output_file, 'w') as f:
        f.write(full_html)

if __name__ == '__main__':
    main()